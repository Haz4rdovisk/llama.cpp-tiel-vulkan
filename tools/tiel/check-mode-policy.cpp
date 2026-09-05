#include "llama-model.h"
#include "llama-graph.h"
#include "llama-moecache.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>

struct synthetic_model : llama_model {
    synthetic_model() : llama_model(llama_model_default_params()) {}
    void load_stats(llama_model_loader &) override {}
    void load_hparams(llama_model_loader &) override {}
    void load_vocab(llama_model_loader &) override {}
    bool load_tensors(llama_model_loader &) override { return true; }
    void load_arch_hparams(llama_model_loader &) override {}
    void load_arch_tensors(llama_model_loader &) override {}
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params &) const override { return {}; }
};

int main() {
    const char * modes[] = {nullptr, "", "full", "down", "vulkan_import", "vulkan_exact", "typo"};
    for (const char * mode : modes) {
        for (const char * opt : {"0", "1"}) {
            if (mode) setenv("LLAMA_MOE_CACHE_MODE", mode, 1);
            else unsetenv("LLAMA_MOE_CACHE_MODE");
            setenv("LLAMA_MOE_CACHE_ALLOW_EXPERIMENTAL", opt, 1);
            ggml_tensor tensor{};
            {
                synthetic_model model;
                llama_moe_cache_init(model, 24, 1);
                assert(llama_moe_cache_lookup(&tensor) == nullptr);
                llama_moe_cache_init(model, 24, 1);
            }
            std::printf("REJECT mode=%s opt=%s PASS\n", mode ? mode : "unset", opt);
        }
    }
    // No GPU allocation: accepted mode with no expert tensors remains disabled.
    setenv("LLAMA_MOE_CACHE_MODE", "vulkan_host", 1);
    setenv("LLAMA_MOE_CACHE_ADAPTIVE", "1", 1);
    unsetenv("LLAMA_MOE_CACHE_HOTSET");
    synthetic_model model;
    llama_moe_cache_init(model, 24, 1);
    std::puts("MODE_POLICY_DONE");
}
