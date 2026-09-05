#include "llama-moecache.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cinttypes>
#include <algorithm>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <sstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

bool cache_env_enabled(const char * name) {
    const char * value = std::getenv(name);
    return value && std::strcmp(value, "1") == 0;
}

struct layer_state {
    llama_moe_cache_layer pub;

    // LRU bookkeeping (host side; the tables mirror expert_slot)
    std::vector<int32_t>  slot_expert;   // slot -> expert id, -1 when empty
    std::vector<int32_t>  expert_slot;   // expert id -> slot, -1 when uncached
    std::vector<uint64_t> slot_last_use; // slot -> lamport clock of last hit
    std::vector<int32_t>  pending;       // admitted uncached ids observed since last step
    std::vector<uint16_t> miss_score;    // miss observations; admission proxy for expected reuse
    std::vector<float> recent_frequency;

    std::vector<bool>     slot_in_flight;   // slot has an upload pending
    std::vector<bool>     expert_in_flight; // expert already has an upload pending

    uint64_t n_hit       = 0;
    uint64_t n_miss      = 0;
    uint64_t n_cpu_cold  = 0;
    uint64_t n_admitted  = 0;
    uint64_t adaptive_hits = 0;
    uint64_t adaptive_misses = 0;
    uint64_t adaptive_uploads = 0;
    uint64_t adaptive_weight_bytes = 0;
};

struct upload_job {
    size_t  layer_idx;
    int32_t expert;
    int32_t slot;
    bool    done = false;
};

struct moe_cache {
    int32_t n_slots       = 0;
    int32_t max_inserts   = 2;
    int32_t admit_after   = 1;
    bool    down_only          = false;
    bool    vulkan_host_mode   = false;
    bool    vulkan_import_mode = false;
    bool    vulkan_exact_mode  = false;

    uint64_t clock   = 0;
    uint64_t n_steps = 0;

    std::mutex mtx; // guards pending lists + clock (observe runs during graph exec)

    std::vector<layer_state> layers;
    std::map<const ggml_tensor *, size_t> by_up_src;

    std::vector<ggml_context *>         ctxs;
    std::vector<ggml_backend_buffer_t>  bufs;

    // async upload worker: slices are copied to the device off the decode
    // thread; the new table mapping is only published at a later step() once
    // the upload has completed, so a running graph never reads a torn slot
    std::thread              worker;
    std::mutex               wmtx;
    std::condition_variable  wcv;
    std::deque<upload_job>   todo;
    std::vector<upload_job>  done;
    bool                     stop = false;

    // Static-hotset family bypasses the legacy observer/worker. Adaptive Vulkan
    // uses this path too, but refreshes residency in update_vulkan().
    bool                     static_hotset = false;
    bool                     adaptive = false;
    bool                     frequency_admission = false;
    bool                     layer_stats = false;
    bool                     adaptive_stats = false;
    ggml_tensor *            route_ids = nullptr;
    ggml_backend_buffer_t    transfer_buffer = nullptr;
    bool                     enabled = true;
    std::string              control_file;
};

moe_cache * g_cache = nullptr;
std::mutex g_init_mtx;
bool g_init_done = false;
const llama_model * g_owner = nullptr;

int parse_layer_from_name(const char * name) {
    // "blk.<il>.ffn_gate_exps.weight"
    if (strncmp(name, "blk.", 4) != 0) {
        return -1;
    }
    return atoi(name + 4);
}

void moe_obs_cb(const char * name, const struct ggml_tensor * ids, void * ud) {
    moe_cache * mc = (moe_cache *) ud;

    const int64_t n_ids    = ids->ne[0];
    const int64_t n_tokens = ids->ne[1];
    if (n_tokens > 4) {
        return; // batch/prefill: the cache graph is not built there, don't pollute the LRU
    }

    const int il = parse_layer_from_name(name);
    if (il < 0) {
        return;
    }

    layer_state * ls = nullptr;
    for (auto & l : mc->layers) {
        if (l.pub.il == il) { ls = &l; break; }
    }
    if (!ls) {
        return;
    }

    std::lock_guard<std::mutex> lock(mc->mtx);
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t i = 0; i < n_ids; ++i) {
            const int32_t id = *(const int32_t *) ((const char *) ids->data + t*ids->nb[1] + i*ids->nb[0]);
            if (id < 0 || id >= (int32_t) ls->expert_slot.size()) {
                continue;
            }
            const int32_t slot = ls->expert_slot[id];
            if (slot >= 0) {
                ls->n_hit++;
                ls->slot_last_use[slot] = ++mc->clock;
            } else {
                ls->n_miss++;

                // FreeToken-lite admission: misses remain on CPU until reuse
                // justifies paying PCIe + GPU residency cost.
                if (ls->expert_in_flight[id]) {
                    ls->n_cpu_cold++;
                    continue;
                }

                uint16_t & score = ls->miss_score[id];
                if (score != UINT16_MAX) {
                    ++score;
                }
                if (score < mc->admit_after) {
                    ls->n_cpu_cold++;
                    continue;
                }

                bool dup = false;
                for (int32_t p : ls->pending) {
                    if (p == id) { dup = true; break; }
                }
                if (!dup) {
                    ls->pending.push_back(id);
                    ls->n_admitted++;
                }
            }
        }
    }
}

void upload_slice(ggml_tensor * dst_c, const ggml_tensor * src, int32_t expert, int32_t slot) {
    const size_t sz = src->nb[2];
    if ((size_t) slot*dst_c->nb[2] + sz > ggml_nbytes(dst_c) || (size_t) expert*sz + sz > ggml_nbytes(src)) {
        LLAMA_LOG_ERROR("moe-cache: bad upload %s <- %s expert=%d slot=%d sz=%zu dst_nb2=%zu dst_bytes=%zu src_bytes=%zu\n",
                dst_c->name, src->name, expert, slot, sz, dst_c->nb[2], ggml_nbytes(dst_c), ggml_nbytes(src));
        return;
    }
    if (src->data) {
        ggml_backend_tensor_set(dst_c, (const char *) src->data + (size_t) expert*sz, (size_t) slot*dst_c->nb[2], sz);
    } else {
        std::vector<uint8_t> tmp(sz);
        ggml_backend_tensor_get(src, tmp.data(), (size_t) expert*sz, sz);
        ggml_backend_tensor_set(dst_c, tmp.data(), (size_t) slot*dst_c->nb[2], sz);
    }
}

bool verify_slice(const ggml_tensor * dst_c, const ggml_tensor * src, int32_t expert, int32_t slot) {
    const size_t sz = src->nb[2];
    std::vector<uint8_t> a(sz), b(sz);
    if (src->data) {
        memcpy(a.data(), (const char *) src->data + (size_t) expert*sz, sz);
    } else {
        ggml_backend_tensor_get(src, a.data(), (size_t) expert*sz, sz);
    }
    ggml_backend_tensor_get(dst_c, b.data(), (size_t) slot*dst_c->nb[2], sz);
    return a == b;
}

void set_table_entry(llama_moe_cache_layer & pub, int32_t expert, int32_t slot_or_dummy) {
    const bool force_miss = pub.vulkan_host && cache_env_enabled("LLAMA_MOE_CACHE_FORCE_MISS");
    const bool is_hot = !force_miss && slot_or_dummy < pub.n_slots;
    const int32_t v = pub.vulkan_exact
        ? (is_hot ? slot_or_dummy : pub.n_slots)
        : (pub.vulkan_host
            ? (int32_t) ((((uint32_t) expert) << 16) | (is_hot ? (uint32_t) slot_or_dummy : 0xffffu))
            : slot_or_dummy);
    for (int64_t lane = 0; lane < pub.dev_table->ne[2]; ++lane) {
        const size_t off = (size_t) lane*pub.dev_table->nb[2] + (size_t) expert*pub.dev_table->nb[1];
        ggml_backend_tensor_set(pub.dev_table, &v, off, sizeof(int32_t));
    }
    if (pub.cold_table) {
        const int32_t cold = is_hot ? pub.cold_dummy : expert;
        for (int64_t lane = 0; lane < pub.cold_table->ne[2]; ++lane) {
            const size_t off = (size_t) lane*pub.cold_table->nb[2] + (size_t) expert*pub.cold_table->nb[1];
            ggml_backend_tensor_set(pub.cold_table, &cold, off, sizeof(int32_t));
        }
    }
    ggml_backend_tensor_set(pub.host_table, &v, (size_t) expert*sizeof(int32_t), sizeof(int32_t));
}

void zero_expert_slice(ggml_tensor * tensor, int32_t expert) {
    const size_t sz = tensor->nb[2];
    std::vector<uint8_t> zero(sz, 0);
    ggml_backend_tensor_set(tensor, zero.data(), (size_t) expert*sz, sz);
}

// Import an existing CPU tensor's storage into Vulkan without copying it. The returned
// tensor is metadata-only and points into an external-memory VkBuffer wrapper. The CPU
// tensor remains authoritative and keeps its original buffer, so prefill performance is
// unchanged. Alignment is discovered conservatively by trying power-of-two import regions
// fully contained in the original host buffer.
bool import_cpu_tensor_alias(
        ggml_context * ctx,
        ggml_backend_dev_t vk_dev,
        const ggml_tensor * src,
        ggml_tensor ** out_alias,
        ggml_backend_buffer_t * out_buf) {
    if (!ctx || !vk_dev || !src || !src->buffer || !src->data || !out_alias || !out_buf) {
        return false;
    }
    if (!ggml_backend_buffer_is_host(src->buffer)) {
        return false;
    }

    auto * host_base = (uint8_t *) ggml_backend_buffer_get_base(src->buffer);
    const size_t host_size = ggml_backend_buffer_get_size(src->buffer);
    auto * src_ptr = (uint8_t *) src->data;
    const size_t src_size = ggml_nbytes(src);
    if (!host_base || src_ptr < host_base || src_ptr + src_size > host_base + host_size) {
        return false;
    }

    const uintptr_t host_begin = (uintptr_t) host_base;
    const uintptr_t host_end   = host_begin + host_size;
    const uintptr_t data_begin = (uintptr_t) src_ptr;
    const uintptr_t data_end   = data_begin + src_size;
    static const size_t alignments[] = {
        4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576
    };

    const bool import_debug = cache_env_enabled("LLAMA_MOE_CACHE_IMPORT_DEBUG");
    if (import_debug) {
        fprintf(stderr, "FREETOKEN_IMPORT_BEGIN name=%s buf=%s host=%p host_size=%zu data=%p bytes=%zu offset=%zu\n",
                src->name, ggml_backend_buffer_name(src->buffer), host_base, host_size, src_ptr, src_size,
                (size_t)(data_begin - host_begin));
    }
    for (size_t align : alignments) {
        const uintptr_t import_begin = data_begin & ~(uintptr_t)(align - 1);
        const uintptr_t import_end   = (data_end + align - 1) & ~(uintptr_t)(align - 1);
        // The logical ggml CPU buffer may end part-way through the final mapped page.
        // VK_EXT_external_memory_host requires an aligned import size; allowing the import
        // to cover the remainder of that already-mapped tail page is safe because the
        // tensor's descriptor/shape never reads beyond data_end.
        const uintptr_t mapped_host_end = (host_end + align - 1) & ~(uintptr_t)(align - 1);
        if (import_begin < host_begin || import_end > mapped_host_end || import_end <= import_begin) {
            if (import_debug) fprintf(stderr, "FREETOKEN_IMPORT_SKIP align=%zu begin_ok=%d end_ok=%d\n",
                    align, import_begin >= host_begin ? 1 : 0, import_end <= mapped_host_end ? 1 : 0);
            continue;
        }
        const size_t import_size = (size_t)(import_end - import_begin);
        ggml_backend_buffer_t buf = ggml_backend_dev_buffer_from_host_ptr(
                vk_dev, (void *) import_begin, import_size, src_size);
        if (import_debug) fprintf(stderr, "FREETOKEN_IMPORT_TRY align=%zu ptr=%p size=%zu ok=%d\n",
                align, (void *) import_begin, import_size, buf ? 1 : 0);
        if (!buf) {
            continue;
        }

        ggml_tensor * alias = ggml_dup_tensor(ctx, src);
        alias->buffer = buf;
        alias->data = (uint8_t *) ggml_backend_buffer_get_base(buf) + (data_begin - import_begin);
        alias->view_src = nullptr;
        alias->view_offs = 0;
        ggml_format_name(alias, "%s.ftvk", src->name);
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        if (ggml_backend_buffer_init_tensor(buf, alias) != GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(buf);
            continue;
        }

        *out_alias = alias;
        *out_buf = buf;
        return true;
    }
    return false;
}

bool file_exists(const std::string & path) {
    if (path.empty()) {
        return true;
    }
    std::ifstream f(path);
    return f.good();
}

std::map<int, std::vector<int32_t>> load_hotset(const char * path) {
    std::map<int, std::vector<int32_t>> out;
    if (!path || !path[0]) {
        return out;
    }
    std::ifstream f(path);
    if (!f) {
        LLAMA_LOG_ERROR("moe-cache: cannot open hotset file '%s'\n", path);
        return out;
    }
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream iss(line);
        int il = -1;
        if (!(iss >> il)) {
            continue;
        }
        int32_t id = -1;
        auto & ids = out[il];
        while (iss >> id) {
            ids.push_back(id);
        }
    }
    return out;
}

} // namespace

void llama_moe_cache_init(const llama_model & model, int32_t n_slots, int32_t max_inserts) {
    std::lock_guard<std::mutex> init_lock(g_init_mtx);
    fprintf(stderr, "FREETOKEN_INIT slots=%d inserts=%d done=%d\n", n_slots, max_inserts, g_init_done ? 1 : 0);
    if (g_init_done) {
        return;
    }
    [&]() {
        if (n_slots <= 0) {
            // Do not poison global init: server dry-run/draft contexts may use 0
            // before the real target context requests an expert cache.
            return;
        }

        auto * mc = new moe_cache();
        auto fail_init = [&](const char * reason) {
            fprintf(stderr, "FREETOKEN_DISABLED reason=%s\n", reason);
            for (auto * buffer : mc->bufs) ggml_backend_buffer_free(buffer);
            for (auto * ctx : mc->ctxs) ggml_free(ctx);
            delete mc;
            g_init_done = true;
            g_owner = &model;
        };
        mc->n_slots = n_slots;
        if (max_inserts > 0) {
            mc->max_inserts = max_inserts;
        }
        const char * mode = std::getenv("LLAMA_MOE_CACHE_MODE");
        // Rejected split/import paths remain archived, not runtime-selectable.
        if (!mode || std::strcmp(mode, "vulkan_host") != 0) {
            fail_init("only_vulkan_host_mode_supported");
            return;
        }
        mc->vulkan_host_mode = true;
        const char * hotset_path = std::getenv("LLAMA_MOE_CACHE_HOTSET");
        const auto hotset = load_hotset(hotset_path);
        const bool static_requested = !hotset.empty();
        mc->adaptive = mc->vulkan_host_mode && !mc->vulkan_import_mode && !mc->vulkan_exact_mode &&
            cache_env_enabled("LLAMA_MOE_CACHE_ADAPTIVE");
        const char * admission = std::getenv("LLAMA_MOE_CACHE_ADMISSION");
        mc->frequency_admission = mc->adaptive && admission && std::strcmp(admission, "recent_frequency") == 0;
        mc->layer_stats = mc->adaptive && cache_env_enabled("LLAMA_MOE_CACHE_LAYER_STATS");
        mc->adaptive_stats = mc->adaptive && cache_env_enabled("LLAMA_MOE_CACHE_ADAPTIVE_STATS");
        if (mc->adaptive && cache_env_enabled("LLAMA_MOE_CACHE_FORCE_MISS")) {
            fail_init("force_miss_not_supported_with_adaptive");
            return;
        }
        if (mc->frequency_admission) {
            fprintf(stderr, "FREETOKEN_ADMISSION policy=recent_frequency decay=0.95\n");
        }
        if (mc->vulkan_host_mode && hotset.empty() && !mc->adaptive) {
            LLAMA_LOG_WARN("%s: vulkan_host mode requires LLAMA_MOE_CACHE_HOTSET; disabling cache\n", __func__);
            delete mc;
            g_init_done = true;
            g_owner = &model;
            return;
        }

        if (const char * env = std::getenv("LLAMA_MOE_CACHE_ADMIT_AFTER")) {
            const long parsed = std::strtol(env, nullptr, 10);
            if (parsed >= 1 && parsed <= 32) {
                mc->admit_after = (int32_t) parsed;
            } else {
                LLAMA_LOG_WARN("LLAMA_MOE_CACHE_ADMIT_AFTER=%ld outside safe range [1,32]; using 1\n", parsed);
            }
        }

        // collect the host-resident expert layers, grouped by the device buffer
        // type of that layer's router (the cache lives next to the router)
        struct cand { int il; const llama_layer * l; };
        std::map<ggml_backend_buffer_type_t, std::vector<cand>> groups;

        for (size_t il = 0; il < model.layers.size(); ++il) {
            const auto & l = model.layers[il];
            if (mc->vulkan_host_mode && hotset.find((int) il) != hotset.end()) {
                const char * ub = (l.ffn_up_exps && l.ffn_up_exps->buffer) ? ggml_backend_buft_name(ggml_backend_buffer_get_type(l.ffn_up_exps->buffer)) : "null";
                const char * rb = (l.ffn_gate_inp && l.ffn_gate_inp->buffer) ? ggml_backend_buft_name(ggml_backend_buffer_get_type(l.ffn_gate_inp->buffer)) : "null";
                fprintf(stderr, "FREETOKEN_LAYER il=%zu up=%s gate=%s down=%s ubuf=%s rbuf=%s rhost=%d data=%d/%d/%d\n",
                    il, l.ffn_up_exps ? ggml_type_name(l.ffn_up_exps->type) : "null",
                    l.ffn_gate_exps ? ggml_type_name(l.ffn_gate_exps->type) : "null",
                    l.ffn_down_exps ? ggml_type_name(l.ffn_down_exps->type) : "null",
                    ub, rb, (l.ffn_gate_inp && l.ffn_gate_inp->buffer && ggml_backend_buffer_is_host(l.ffn_gate_inp->buffer)) ? 1 : 0,
                    l.ffn_up_exps && l.ffn_up_exps->data ? 1 : 0, l.ffn_gate_exps && l.ffn_gate_exps->data ? 1 : 0, l.ffn_down_exps && l.ffn_down_exps->data ? 1 : 0);
            }
            if (!l.ffn_up_exps || !l.ffn_gate_exps || !l.ffn_down_exps || !l.ffn_gate_inp) {
                continue;
            }
            // common_fit_params constructs temporary model tensors before storage is allocated.
            // Do not preload from those; leave global init open so the real model can retry.
            if (!l.ffn_up_exps->data || !l.ffn_gate_exps->data || !l.ffn_down_exps->data) {
                continue;
            }
            if (!l.ffn_up_exps->buffer) {
                continue;
            }
            const char * expert_buft_name = ggml_backend_buft_name(ggml_backend_buffer_get_type(l.ffn_up_exps->buffer));
            if (mc->vulkan_host_mode) {
                if (mc->vulkan_import_mode) {
                    if (!ggml_backend_buffer_is_host(l.ffn_up_exps->buffer)) {
                        continue; // import mode starts from the normal CPU expert placement
                    }
                } else if (std::strcmp(expert_buft_name, "Vulkan_Host") != 0) {
                    continue; // legacy single-backend mode requires Vulkan_Host allocation
                }
                const auto hybrid_type_ok = [](ggml_type t) {
                    return t == GGML_TYPE_IQ2_S || t == GGML_TYPE_IQ3_XXS;
                };
                if (!hybrid_type_ok(l.ffn_up_exps->type) ||
                    !hybrid_type_ok(l.ffn_gate_exps->type) ||
                    !hybrid_type_ok(l.ffn_down_exps->type)) {
                    continue; // current hybrid kernel intentionally covers only Tiel's IQ2_S/IQ3_XXS experts
                }
            } else if (!ggml_backend_buffer_is_host(l.ffn_up_exps->buffer)) {
                continue; // legacy cache requires CPU-host experts
            }
            if (!l.ffn_gate_inp->buffer || ggml_backend_buffer_is_host(l.ffn_gate_inp->buffer)) {
                continue; // no device home for the cache
            }
            if (static_requested && hotset.find((int) il) == hotset.end()) {
                continue;
            }
            groups[ggml_backend_buffer_get_type(l.ffn_gate_inp->buffer)].push_back({(int) il, &l});
        }

        fprintf(stderr, "FREETOKEN_GROUPS groups=%zu hotset_layers=%zu mode_vulkan_host=%d\n", groups.size(), hotset.size(), mc->vulkan_host_mode ? 1 : 0);
        if (groups.empty()) {
            LLAMA_LOG_INFO("%s: LLAMA_MOE_CACHE_SLOTS=%d but no host-resident expert layers found - disabled\n", __func__, n_slots);
            delete mc;
            return;
        }
        if (mc->adaptive && (groups.size() != 1 || model.hparams.n_expert_used <= 0)) {
            fail_init("unsupported_adaptive_layout");
            return;
        }

        // host buffer for the CPU-side tables
        std::vector<cand> all;
        for (auto & g : groups) {
            all.insert(all.end(), g.second.begin(), g.second.end());
        }

        std::map<int, int32_t> layer_slots;
        if (const char * plan = std::getenv("LLAMA_MOE_CACHE_LAYER_SLOTS")) {
            bool valid = mc->adaptive && hotset.empty();
            std::istringstream entries(plan);
            std::string entry;
            while (std::getline(entries, entry, ',')) {
                std::istringstream item(entry);
                int layer, slots;
                char colon;
                if (!(item >> layer >> colon >> slots) || colon != ':' || slots < 1 || slots >= 65535 ||
                    !(item >> std::ws).eof() || !layer_slots.emplace(layer, slots).second) {
                    valid = false;
                    break;
                }
            }
            uint64_t requested = 0, baseline = 0;
            size_t matched = 0;
            for (const auto & candidate : all) {
                const auto it = layer_slots.find(candidate.il);
                const int32_t slots = it == layer_slots.end() ? n_slots : it->second;
                if (it != layer_slots.end()) ++matched;
                valid = valid && slots <= candidate.l->ffn_up_exps->ne[2];
                const uint64_t bytes = candidate.l->ffn_up_exps->nb[2] + candidate.l->ffn_gate_exps->nb[2] + candidate.l->ffn_down_exps->nb[2];
                baseline += uint64_t(n_slots)*bytes;
                requested += uint64_t(slots)*bytes;
            }
            valid = valid && !layer_slots.empty() && matched == layer_slots.size() && requested <= baseline;
            if (!valid) {
                fprintf(stderr, "FREETOKEN_LAYER_SLOTS rejected: require unique host layers, adaptive mode, no hotset, and original byte budget\n");
                layer_slots.clear(); // Preserve the uniform cache on invalid experimental input.
            } else {
                fprintf(stderr, "FREETOKEN_LAYER_SLOTS accepted bytes=%" PRIu64 " budget=%" PRIu64 "\n", requested, baseline);
            }
        }

        auto alloc_group = [&](ggml_backend_buffer_type_t buft, const std::vector<cand> & cands, bool tables_only) -> bool {
            ggml_init_params ip = {
                /*.mem_size  =*/ ggml_tensor_overhead()*(cands.size()*6 + 8),
                /*.mem_buffer=*/ nullptr,
                /*.no_alloc  =*/ true,
            };
            ggml_context * ctx = ggml_init(ip);
            if (!ctx) {
                return false;
            }
            mc->ctxs.push_back(ctx);

            for (const auto & c : cands) {
                layer_state * ls = nullptr;
                for (auto & l : mc->layers) {
                    if (l.pub.il == c.il) { ls = &l; break; }
                }
                if (!ls) {
                    mc->layers.push_back({});
                    ls = &mc->layers.back();
                    ls->pub.il       = c.il;
                    ls->pub.n_slots   = layer_slots.count(c.il) ? layer_slots.at(c.il) : n_slots;
                    ls->pub.down_only  = mc->down_only;
                    ls->pub.vulkan_host = mc->vulkan_host_mode;
                    ls->pub.vulkan_import = mc->vulkan_import_mode;
                    ls->pub.vulkan_exact = mc->vulkan_exact_mode;
                    ls->pub.up_src   = c.l->ffn_up_exps;
                    ls->pub.gate_src = c.l->ffn_gate_exps;
                    ls->pub.down_src = c.l->ffn_down_exps;
                }

                if (tables_only) {
                    ls->pub.host_table = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, ls->pub.up_src->ne[2]);
                    ggml_format_name(ls->pub.host_table, "moe_cache_htbl.%d", c.il);
                } else {
                    const ggml_tensor * u = c.l->ffn_up_exps;
                    const ggml_tensor * g = c.l->ffn_gate_exps;
                    const ggml_tensor * d = c.l->ffn_down_exps;
                    const int32_t cache_mats = ls->pub.n_slots + (mc->vulkan_host_mode ? 0 : 1);
                    if (!mc->down_only) {
                        ls->pub.up_c   = ggml_new_tensor_3d(ctx, u->type, u->ne[0], u->ne[1], cache_mats);
                        ls->pub.gate_c = ggml_new_tensor_3d(ctx, g->type, g->ne[0], g->ne[1], cache_mats);
                    }
                    ls->pub.down_c = ggml_new_tensor_3d(ctx, d->type, d->ne[0], d->ne[1], cache_mats);
                    ls->pub.dev_table = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, 1, u->ne[2], 4);
                    if (mc->vulkan_host_mode) {
                        ls->pub.cold_table = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, 1, u->ne[2], 4);
                    }
                    if (!mc->down_only) {
                        ggml_format_name(ls->pub.up_c,   "moe_cache_up.%d",   c.il);
                        ggml_format_name(ls->pub.gate_c, "moe_cache_gate.%d", c.il);
                    }
                    ggml_format_name(ls->pub.down_c,    "moe_cache_down.%d", c.il);
                    ggml_format_name(ls->pub.dev_table, "moe_cache_tbl.%d",  c.il);
                    if (ls->pub.cold_table) ggml_format_name(ls->pub.cold_table, "moe_cache_cold_tbl.%d", c.il);
                }
            }

            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
            if (!buf) {
                LLAMA_LOG_WARN("%s: failed to allocate MoE cache buffer on %s - cache disabled\n",
                        __func__, ggml_backend_buft_name(buft));
                return false;
            }
            ggml_backend_buffer_clear(buf, 0);
            mc->bufs.push_back(buf);
            return true;
        };

        bool ok = alloc_group(ggml_backend_cpu_buffer_type(), all, /*tables_only=*/true);
        for (auto & g : groups) {
            if (!ok) {
                break;
            }
            ok = alloc_group(g.first, g.second, /*tables_only=*/false);
        }

        if (!ok) {
            for (auto * b : mc->bufs) { ggml_backend_buffer_free(b); }
            for (auto * c : mc->ctxs) { ggml_free(c); }
            delete mc;
            g_init_done = true; // a real model was seen and allocation failed: stay disabled
            g_owner = &model;
            return;
        }

        if (mc->vulkan_import_mode) {
            ggml_init_params ip = {
                /*.mem_size  =*/ ggml_tensor_overhead()*(mc->layers.size()*3 + 8),
                /*.mem_buffer=*/ nullptr,
                /*.no_alloc  =*/ true,
            };
            ggml_context * import_ctx = ggml_init(ip);
            if (!import_ctx) {
                ok = false;
            } else {
                mc->ctxs.push_back(import_ctx);
                for (auto & ls : mc->layers) {
                    ggml_tensor * hot_ref = ls.pub.up_c ? ls.pub.up_c : ls.pub.down_c;
                    ggml_backend_dev_t vk_dev = hot_ref && hot_ref->buffer
                        ? ggml_backend_buft_get_device(ggml_backend_buffer_get_type(hot_ref->buffer))
                        : nullptr;
                    ggml_backend_buffer_t ub = nullptr, gb = nullptr, db = nullptr;
                    if (!vk_dev ||
                        !import_cpu_tensor_alias(import_ctx, vk_dev, ls.pub.up_src,   &ls.pub.up_vk,   &ub) ||
                        !import_cpu_tensor_alias(import_ctx, vk_dev, ls.pub.gate_src, &ls.pub.gate_vk, &gb) ||
                        !import_cpu_tensor_alias(import_ctx, vk_dev, ls.pub.down_src, &ls.pub.down_vk, &db)) {
                        if (ub) ggml_backend_buffer_free(ub);
                        if (gb) ggml_backend_buffer_free(gb);
                        if (db) ggml_backend_buffer_free(db);
                        ok = false;
                        break;
                    }
                    mc->bufs.push_back(ub);
                    mc->bufs.push_back(gb);
                    mc->bufs.push_back(db);
                }
            }
            if (!ok) {
                LLAMA_LOG_WARN("%s: failed to import CPU MoE weights into Vulkan external host buffers - cache disabled\n", __func__);
                for (auto * b : mc->bufs) { ggml_backend_buffer_free(b); }
                for (auto * c : mc->ctxs) { ggml_free(c); }
                delete mc;
                g_init_done = true;
                g_owner = &model;
                return;
            }
        }

        // init LRU state + tables (everything uncached -> dummy slot n_slots)
        size_t vram = 0;
        for (auto & ls : mc->layers) {
            const int64_t n_expert = ls.pub.up_src->ne[2];
            const int32_t local_slots = ls.pub.n_slots;
            ls.slot_expert.assign(local_slots, -1);
            ls.expert_slot.assign(n_expert, -1);
            ls.slot_last_use.assign(local_slots, 0);
            ls.miss_score.assign(n_expert, 0);
            if (mc->frequency_admission) ls.recent_frequency.assign(n_expert, 0.0f);
            ls.slot_in_flight.assign(local_slots, false);
            ls.expert_in_flight.assign(n_expert, false);
            ls.pub.cold_dummy = ls.pub.vulkan_exact ? (int32_t) n_expert : (int32_t) n_expert - 1;

            std::vector<int32_t> dummy(n_expert, local_slots);
            std::vector<int32_t> dev_dummy((size_t) n_expert * ls.pub.dev_table->ne[2], local_slots);
            if (ls.pub.vulkan_host && !ls.pub.vulkan_exact) {
                for (int64_t lane = 0; lane < ls.pub.dev_table->ne[2]; ++lane) {
                    for (int32_t ex = 0; ex < (int32_t) n_expert; ++ex) {
                        dev_dummy[(size_t) lane*n_expert + ex] =
                            (int32_t) ((((uint32_t) ex) << 16) | 0xffffu);
                    }
                }
            }
            ggml_backend_tensor_set(ls.pub.dev_table, dev_dummy.data(), 0, dev_dummy.size()*sizeof(int32_t));
            if (ls.pub.cold_table) {
                std::vector<int32_t> cold((size_t) n_expert * ls.pub.cold_table->ne[2]);
                for (int64_t lane = 0; lane < ls.pub.cold_table->ne[2]; ++lane) {
                    for (int32_t ex = 0; ex < (int32_t) n_expert; ++ex) {
                        cold[(size_t) lane*n_expert + ex] = ex;
                    }
                }
                ggml_backend_tensor_set(ls.pub.cold_table, cold.data(), 0, cold.size()*sizeof(int32_t));
            }
            ggml_backend_tensor_set(ls.pub.host_table, dummy.data(), 0, n_expert*sizeof(int32_t));

            mc->by_up_src[ls.pub.up_src] = &ls - mc->layers.data();
            if (ls.pub.up_c)   vram += ggml_nbytes(ls.pub.up_c);
            if (ls.pub.gate_c) vram += ggml_nbytes(ls.pub.gate_c);
            if (ls.pub.down_c) vram += ggml_nbytes(ls.pub.down_c);
            LLAMA_LOG_DEBUG("moe-cache: init layer %d '%s' %zu bytes/expert\n",
                    ls.pub.il, ls.pub.up_src->name, ls.pub.up_src->nb[2]);
        }

        if (hotset_path && hotset_path[0] && hotset.empty()) {
            LLAMA_LOG_WARN("%s: static hotset requested but no entries loaded; falling back to LRU\n", __func__);
        }

        size_t loaded = 0;
        if (!hotset.empty() || mc->adaptive) {
            mc->static_hotset = true;
            const char * ctl = std::getenv("LLAMA_MOE_CACHE_CONTROL_FILE");
            if (ctl && ctl[0]) {
                mc->control_file = ctl;
                mc->enabled = file_exists(mc->control_file);
            }

            for (auto & ls : mc->layers) {
                const auto it = hotset.find(ls.pub.il);
                if (it == hotset.end()) {
                    continue;
                }
                std::vector<int32_t> wanted;
                for (int32_t id : it->second) {
                    wanted.push_back(id);
                }
                int32_t slot = 0;
                for (int32_t id : wanted) {
                    if (slot >= ls.pub.n_slots) break;
                    if (id < 0 || id >= (int32_t) ls.expert_slot.size() || ls.expert_slot[id] >= 0) continue;
                    if (!mc->down_only) {
                        upload_slice(ls.pub.up_c,   ls.pub.up_src,   id, slot);
                        upload_slice(ls.pub.gate_c, ls.pub.gate_src, id, slot);
                    }
                    upload_slice(ls.pub.down_c, ls.pub.down_src, id, slot);
                    if (cache_env_enabled("LLAMA_MOE_CACHE_VERIFY")) {
                        bool same = verify_slice(ls.pub.down_c, ls.pub.down_src, id, slot);
                        if (!mc->down_only) {
                            same = same && verify_slice(ls.pub.up_c, ls.pub.up_src, id, slot);
                            same = same && verify_slice(ls.pub.gate_c, ls.pub.gate_src, id, slot);
                        }
                        if (!same) {
                            fprintf(stderr, "FREETOKEN_VERIFY_FAIL layer=%d expert=%d slot=%d\n", ls.pub.il, id, slot);
                            fail_init("hotset_verification");
                            return;
                        }
                    }
                    ls.slot_expert[slot] = id;
                    ls.expert_slot[id] = slot;
                    set_table_entry(ls.pub, id, slot);
                    ++slot;
                    ++loaded;
                }
            }
            LLAMA_LOG_INFO("%s: FreeToken hotset: %zu experts preloaded, adaptive=%d, enabled=%d, control='%s'\n",
                    __func__, loaded, mc->adaptive ? 1 : 0, mc->enabled ? 1 : 0, mc->control_file.c_str());
        } else {
            mc->worker = std::thread([mc]() {
                for (;;) {
                    upload_job j;
                    {
                        std::unique_lock<std::mutex> lk(mc->wmtx);
                        mc->wcv.wait(lk, [mc]() { return mc->stop || !mc->todo.empty(); });
                        if (mc->stop) {
                            return;
                        }
                        j = mc->todo.front();
                        mc->todo.pop_front();
                    }
                    auto & ls = mc->layers[j.layer_idx];
                    if (!mc->down_only) {
                        upload_slice(ls.pub.up_c,   ls.pub.up_src,   j.expert, j.slot);
                        upload_slice(ls.pub.gate_c, ls.pub.gate_src, j.expert, j.slot);
                    }
                    upload_slice(ls.pub.down_c, ls.pub.down_src, j.expert, j.slot);
                    {
                        std::lock_guard<std::mutex> lk(mc->wmtx);
                        j.done = true;
                        mc->done.push_back(j);
                    }
                }
            });
            ggml_set_moe_obs_callback(moe_obs_cb, mc);
        }

        if (mc->adaptive) {
            auto buft = groups.begin()->first;
            auto host_buft = ggml_backend_dev_host_buffer_type(ggml_backend_buft_get_device(buft));
            if (!host_buft) {
                fail_init("missing_host_buffer_type");
                return;
            }
            ggml_context * route_ctx = ggml_init({ggml_tensor_overhead() * (mc->layers.size() + 2), nullptr, true});
            if (!route_ctx) {
                fail_init("route_context_allocation");
                return;
            }
            mc->ctxs.push_back(route_ctx);
            mc->route_ids = ggml_new_tensor_3d(route_ctx, GGML_TYPE_I32, model.hparams.n_expert_used, 4, mc->layers.size());
            ggml_set_name(mc->route_ids, "moe_cache_routes");
            size_t transfer_size = ggml_nbytes(mc->route_ids);
            for (size_t li = 0; li < mc->layers.size(); ++li) {
                auto & pub = mc->layers[li].pub;
                pub.route_ids = ggml_view_2d(route_ctx, mc->route_ids, mc->route_ids->ne[0], 4,
                                            mc->route_ids->nb[1], li * mc->route_ids->nb[2]);
                ggml_format_name(pub.route_ids, "moe_cache_routes.%d", pub.il);
                transfer_size += ggml_nbytes(pub.dev_table);
            }
            auto route_buf = ggml_backend_alloc_ctx_tensors_from_buft(route_ctx, buft);
            if (!route_buf) {
                fail_init("route_buffer_allocation");
                return;
            }
            mc->bufs.push_back(route_buf);
            mc->transfer_buffer = ggml_backend_buft_alloc_buffer(host_buft, transfer_size);
            if (!mc->transfer_buffer) {
                fail_init("transfer_buffer_allocation");
                return;
            }
            mc->bufs.push_back(mc->transfer_buffer);
            if (ggml_backend_buffer_get_type(mc->transfer_buffer) != host_buft) {
                fail_init("transfer_buffer_type");
                return;
            }
            ggml_backend_buffer_clear(route_buf, 0);
            fprintf(stderr, "FREETOKEN_ADAPTIVE layers=%zu inserts_per_layer=%d route_bytes=%zu\n",
                           mc->layers.size(), mc->max_inserts, ggml_nbytes(mc->route_ids));
        }

        g_cache = mc;
        g_init_done = true;
        g_owner = &model;

        if (mc->static_hotset) {
            fprintf(stderr, "FREETOKEN_ACTIVE layers=%zu loaded=%zu slots=%d enabled=%d\n", mc->layers.size(), loaded, mc->n_slots, mc->enabled ? 1 : 0);
        }

        const char * mode_name = mc->vulkan_exact_mode ? "vulkan_exact" : (mc->vulkan_import_mode ? "vulkan_import" : (mc->vulkan_host_mode ? "vulkan_host" : (mc->down_only ? "down" : "full")));
        LLAMA_LOG_INFO("%s: MoE expert cache enabled: %zu layers x %d slots, mode=%s, %d inserts/step, admit-after=%d, %.1f MiB device memory\n",
                __func__, mc->layers.size(), n_slots, mode_name, mc->max_inserts, mc->admit_after, vram/1024.0/1024.0);
    }();
}

void llama_moe_cache_free(const llama_model & model) {
    std::lock_guard<std::mutex> init_lock(g_init_mtx);
    if (g_owner != &model) return;
    auto * mc = g_cache;
    g_cache = nullptr;
    g_owner = nullptr;
    g_init_done = false;
    if (!mc) return;
    if (!mc->static_hotset) ggml_set_moe_obs_callback(nullptr, nullptr);
    {
        std::lock_guard<std::mutex> worker_lock(mc->wmtx);
        mc->stop = true;
    }
    mc->wcv.notify_all();
    if (mc->worker.joinable()) mc->worker.join();
    for (auto * buffer : mc->bufs) ggml_backend_buffer_free(buffer);
    for (auto * ctx : mc->ctxs) ggml_free(ctx);
    delete mc;
}

const llama_moe_cache_layer * llama_moe_cache_lookup(const ggml_tensor * up_exps) {
    if (!g_cache || (g_cache->static_hotset && !g_cache->enabled)) {
        return nullptr;
    }
    auto it = g_cache->by_up_src.find(up_exps);
    if (it == g_cache->by_up_src.end()) {
        return nullptr;
    }
    return &g_cache->layers[it->second].pub;
}

void llama_moe_cache_update_vulkan(ggml_backend_sched * sched, int32_t n_tokens) {
    auto * mc = g_cache;
    if (!mc || !mc->adaptive || !mc->enabled || n_tokens < 1 || n_tokens > 4) {
        return;
    }
    auto backend = ggml_backend_sched_get_tensor_backend(sched, mc->layers.front().pub.up_c);
    if (!backend) {
        if (mc->adaptive_stats) {
            fprintf(stderr, "FREETOKEN_UPDATE_SKIP tokens=%d reason=cache_not_in_graph\n", n_tokens);
        }
        return; // this graph did not use the hybrid cache
    }
    const int64_t start_us = ggml_time_us();
    auto * staging = (uint8_t *) ggml_backend_buffer_get_base(mc->transfer_buffer);
    ggml_backend_tensor_get_async(backend, mc->route_ids, staging, 0, ggml_nbytes(mc->route_ids));
    ggml_backend_sched_synchronize(sched);
    const int64_t readback_done_us = ggml_time_us();

    size_t table_offset = ggml_nbytes(mc->route_ids);
    uint64_t hits = 0, misses = 0, uploads = 0;
    for (size_t li = 0; li < mc->layers.size(); ++li) {
        auto & ls = mc->layers[li];
        auto & pub = ls.pub;
        const int32_t n_expert = ls.expert_slot.size();
        std::vector<int32_t> pending;
        std::vector<int32_t> counts(n_expert, 0);
        if (mc->frequency_admission) {
            for (float & score : ls.recent_frequency) score *= 0.95f;
        }
        auto * routes = (const int32_t *) (staging + li * mc->route_ids->nb[2]);
        for (int32_t t = 0; t < n_tokens; ++t) {
            for (int64_t r = 0; r < mc->route_ids->ne[0]; ++r) {
                const int32_t id = (uint32_t) routes[t * mc->route_ids->ne[0] + r] >> 16;
                GGML_ASSERT(id >= 0 && id < n_expert);
                if (mc->frequency_admission) ls.recent_frequency[id] += 1.0f;
                const int32_t slot = ls.expert_slot[id];
                if (slot >= 0) {
                    ++hits;
                    if (mc->layer_stats) ++ls.adaptive_hits;
                    ls.slot_last_use[slot] = ++mc->clock;
                } else {
                    ++misses;
                    if (mc->layer_stats) ++ls.adaptive_misses;
                    if (counts[id]++ == 0) pending.push_back(id);
                }
            }
        }
        std::stable_sort(pending.begin(), pending.end(), [&](int32_t a, int32_t b) {
            return mc->frequency_admission ? ls.recent_frequency[a] > ls.recent_frequency[b] : counts[a] > counts[b];
        });
        const size_t n_insert = std::min(pending.size(), (size_t) mc->max_inserts);
        bool table_changed = false;
        for (size_t i = 0; i < n_insert; ++i) {
            const int32_t id = pending[i];
            int32_t slot = 0;
            for (int32_t s = 0; s < pub.n_slots; ++s) {
                if (ls.slot_expert[s] < 0) { slot = s; break; }
                if (mc->frequency_admission) {
                    const float score = ls.recent_frequency[ls.slot_expert[s]];
                    const float best = ls.recent_frequency[ls.slot_expert[slot]];
                    if (score < best || (score == best && ls.slot_last_use[s] < ls.slot_last_use[slot])) slot = s;
                } else if (ls.slot_last_use[s] < ls.slot_last_use[slot]) slot = s;
            }
            const int32_t victim = ls.slot_expert[slot];
            if (mc->frequency_admission && victim >= 0 && ls.recent_frequency[id] <= ls.recent_frequency[victim]) break;
            if (victim >= 0) ls.expert_slot[victim] = -1;
            for (auto tensors : {std::make_pair(pub.up_c, pub.up_src), std::make_pair(pub.gate_c, pub.gate_src),
                                 std::make_pair(pub.down_c, pub.down_src)}) {
                const auto * src = tensors.second;
                ggml_backend_tensor_set_async(backend, tensors.first, (const uint8_t *) src->data + id * src->nb[2],
                                              slot * tensors.first->nb[2], src->nb[2]);
            }
            ls.expert_slot[id] = slot;
            ls.slot_expert[slot] = id;
            ls.slot_last_use[slot] = ++mc->clock;
            ++uploads;
            if (mc->layer_stats) {
                ++ls.adaptive_uploads;
                ls.adaptive_weight_bytes += pub.up_src->nb[2] + pub.gate_src->nb[2] + pub.down_src->nb[2];
            }
            table_changed = true;
        }
        if (table_changed) {
            auto * table = (int32_t *) (staging + table_offset);
            for (int32_t e = 0; e < n_expert; ++e) {
                const uint32_t slot = ls.expert_slot[e] >= 0 ? ls.expert_slot[e] : 0xffffu;
                table[e] = (int32_t) (((uint32_t) e << 16) | slot);
            }
            memcpy(pub.host_table->data, table, n_expert * sizeof(int32_t));
            for (int lane = 1; lane < 4; ++lane) memcpy(table + lane * n_expert, table, n_expert * sizeof(int32_t));
            ggml_backend_tensor_set_async(backend, pub.dev_table, table, 0, ggml_nbytes(pub.dev_table));
        }
        table_offset += ggml_nbytes(pub.dev_table);
    }
    // Weights and mappings must both be visible before the next graph reads them.
    if (uploads > 0) ggml_backend_synchronize(backend);
    const int64_t update_done_us = ggml_time_us();
    ++mc->n_steps;
    if (mc->layer_stats) {
        for (auto & ls : mc->layers) {
            const auto & pub = ls.pub;
            const uint64_t expert_bytes = pub.up_src->nb[2] + pub.gate_src->nb[2] + pub.down_src->nb[2];
            fprintf(stderr, "FREETOKEN_LAYER step=%" PRIu64 " layer=%d slots=%d hits=%" PRIu64 " misses=%" PRIu64
                    " uploads=%" PRIu64 " upload_bytes=%" PRIu64 " cold_route_bytes=%" PRIu64 "\n",
                    mc->n_steps, pub.il, pub.n_slots, ls.adaptive_hits, ls.adaptive_misses,
                    ls.adaptive_uploads, ls.adaptive_weight_bytes, ls.adaptive_misses*expert_bytes);
            ls.adaptive_hits = ls.adaptive_misses = ls.adaptive_uploads = ls.adaptive_weight_bytes = 0;
        }
    }
    if (mc->adaptive_stats) {
        fprintf(stderr, "FREETOKEN_UPDATE step=%" PRIu64 " hits=%" PRIu64 " misses=%" PRIu64 " uploads=%" PRIu64 " wait_us=%" PRId64 " update_us=%" PRId64 "\n",
                       mc->n_steps, hits, misses, uploads, readback_done_us - start_us, update_done_us - readback_done_us);
    }
}

void llama_moe_cache_step() {
    moe_cache * mc = g_cache;
    if (!mc) {
        return;
    }

    if (mc->static_hotset) {
        if (!mc->control_file.empty()) {
            const bool now = file_exists(mc->control_file);
            if (now != mc->enabled) {
                mc->enabled = now;
                LLAMA_LOG_INFO("moe-cache: static hotset runtime %s\n", now ? "ENABLED" : "DISABLED");
            }
        }
        return;
    }

    // 1) publish completed uploads (sync point: no graph is executing)
    {
        std::lock_guard<std::mutex> wlk(mc->wmtx);
        std::lock_guard<std::mutex> lk(mc->mtx);
        for (const auto & j : mc->done) {
            auto & ls = mc->layers[j.layer_idx];
            ls.slot_expert[j.slot]     = j.expert;
            ls.expert_slot[j.expert]   = j.slot;
            ls.slot_last_use[j.slot]      = ++mc->clock;
            ls.slot_in_flight[j.slot]     = false;
            ls.expert_in_flight[j.expert] = false;
            set_table_entry(ls.pub, j.expert, j.slot);
        }
        mc->done.clear();
    }

    std::lock_guard<std::mutex> lock(mc->mtx);
    mc->n_steps++;

    // 2) schedule new uploads: evict at a sync point (clear the victim's table
    //    entry now), then hand the slice copies to the worker
    for (size_t li = 0; li < mc->layers.size(); ++li) {
        auto & ls = mc->layers[li];
        if (ls.pending.empty()) {
            continue;
        }

        int budget = mc->max_inserts;
        for (auto it = ls.pending.rbegin(); it != ls.pending.rend() && budget > 0; ++it, --budget) {
            const int32_t id = *it;
            if (ls.expert_slot[id] >= 0 || ls.expert_in_flight[id]) {
                continue;
            }

            // victim: an empty non-in-flight slot if any, else the LRU non-in-flight slot
            int32_t slot = -1;
            uint64_t best = UINT64_MAX;
            for (int32_t s = 0; s < ls.pub.n_slots; ++s) {
                if (ls.slot_in_flight[s]) {
                    continue;
                }
                if (ls.slot_expert[s] < 0) { slot = s; break; }
                if (ls.slot_last_use[s] < best) { best = ls.slot_last_use[s]; slot = s; }
            }
            if (slot < 0) {
                break; // every slot is in flight; try again next step
            }

            const int32_t victim = ls.slot_expert[slot];
            if (victim >= 0) {
                ls.expert_slot[victim] = -1;
                ls.slot_expert[slot]   = -1;
                set_table_entry(ls.pub, victim, ls.pub.n_slots);
            }
            ls.slot_in_flight[slot] = true;
            ls.expert_in_flight[id]  = true;

            std::lock_guard<std::mutex> wlk(mc->wmtx);
            mc->todo.push_back({li, id, slot});
        }
        ls.pending.clear();
    }
    mc->wcv.notify_one();

    if (mc->n_steps % 128 == 0) {
        uint64_t h = 0, m = 0, cpu = 0, admitted = 0;
        for (auto & ls : mc->layers) {
            h += ls.n_hit;
            m += ls.n_miss;
            cpu += ls.n_cpu_cold;
            admitted += ls.n_admitted;
        }
        LLAMA_LOG_INFO("moe-cache: steps=%" PRIu64 " hits=%" PRIu64 " misses=%" PRIu64
                " hit-rate=%.1f%% cpu-miss=%" PRIu64 " admitted=%" PRIu64 "\n",
                mc->n_steps, h, m, h + m ? 100.0*h/(h + m) : 0.0, cpu, admitted);
    }
}
