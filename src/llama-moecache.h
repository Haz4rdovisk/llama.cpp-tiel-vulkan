#pragma once

// GPU cache for host-resident MoE expert weights.
//
// Vulkan-host mode uses one MUL_MAT_ID with cold src[0] and hot src[3].
// Packed IDs contain the original expert in the high 16 bits and the hot slot
// in the low 16 bits; 0xffff selects cold weights. Prefill uses normal weights.
// Each layer has fixed cache capacity. Adaptive mode updates LRU residency at
// target decode boundaries, after capturing routes for batches of up to four.
// Uploads and table changes complete before the next graph reads the cache.
//
// Enabled via llama_context_params.n_moe_cache_slots (CLI: --moe-expert-cache).

#include <cstdint>
#include <cstddef>

struct llama_model;
struct ggml_tensor;
struct ggml_backend_sched;

struct llama_moe_cache_layer {
    int il = -1;

    int32_t n_slots = 0;
    int32_t n_extra_slots = 0;
    ggml_tensor * up_extra = nullptr;
    ggml_tensor * gate_extra = nullptr;
    ggml_tensor * down_extra = nullptr;
    bool vulkan_host = false;   // Vulkan-host family of cache modes

    // host-resident source weights (the authoritative experts)
    ggml_tensor * up_src   = nullptr;
    ggml_tensor * gate_src = nullptr;
    ggml_tensor * down_src = nullptr;

    // Quantized weights for n_slots cached experts.
    ggml_tensor * up_c   = nullptr;
    ggml_tensor * gate_c = nullptr;
    ggml_tensor * down_c = nullptr;

    // Device table has four lanes of packed IDs.
    ggml_tensor * dev_table  = nullptr;
    ggml_tensor * route_ids = nullptr; // persistent [n_expert_used, 4] packed routes for adaptive Vulkan
};

// build the cache for every host-resident expert layer of the model.
// Repeated calls reuse the owning model's cache. Other models do not replace it.
void llama_moe_cache_init(const llama_model & model, int32_t n_slots, int32_t max_inserts);

// Release only the owning model's cache, after its contexts have been destroyed.
void llama_moe_cache_free(const llama_model & model);

// nullptr when the cache is disabled or this tensor has no cached layer
const llama_moe_cache_layer * llama_moe_cache_lookup(const ggml_tensor * up_exps);

// Refresh the optional control-file switch between graph executions.
void llama_moe_cache_step();

// Caller must synchronize and discard graphs before changing extra-bank storage.
size_t llama_moe_cache_extra(const llama_model & model, int32_t slots, size_t budget);

// Consume captured routes and publish complete uploads after target graph execution.
void llama_moe_cache_update_vulkan(ggml_backend_sched * sched, int32_t n_tokens);
