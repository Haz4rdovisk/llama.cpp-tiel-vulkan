# Tiel Vulkan experimental snapshot

This private development snapshot preserves the RX590 work before further cleanup. It is not a production release or an upstream submission. Code was developed with AI assistance under user direction. Original authorship, history and licenses remain intact.

Base commit: `bccbacdb8945680f1cfc7e6bffd1e59014705750`, the expert-cache branch by csantiago78, on top of llama.cpp. The initial snapshot preserved 31 DEV implementation files byte-for-byte. The legacy-marker follow-up was also applied to DEV. Production was not changed.

## Candidate architecture

- Single Vulkan MUL_MAT_ID dispatch with cold host weights and hot VRAM weights; no second CPU/GPU chain in the candidate path.
- Packed expert/slot IDs; IQ2_S gate/up and IQ3_XXS down kernels.
- Per-layer adaptive cache, small target/MTP batches up to four, publication at graph boundaries.
- Normal non-hybrid scheduling for prefill; optional tool-message checkpoints for incremental prompts.
- Hardware scope: RX590 8GB, i7-7700, one model/slot. Not a complete implementation of every FreeToken/ATSInfer technique.

## Mechanisms and source map

### Decode: one dispatch, two weight locations

For each eligible MoE projection, src[0] remains the authoritative cold expert tensor in Vulkan_Host memory and src[3] references quantized hot weights in VRAM. The table lookup produces a packed I32 route: high 16 bits are the original expert ID; low 16 bits are the cache slot, or 0xffff on a miss. The Vulkan shader selects the weight source within the same dispatch. No separate expert-to-slot descriptor binding or CPU/GPU output merge is required.

Gate/up use IQ2_S, down uses IQ3_XXS for this model. The cache copies quantized slices, not re-quantized approximations. This preserves stored weights, but does not guarantee identical floating-point reduction order against every baseline kernel. MTP acceptance and output hashes therefore remain regression signals.

`src/llama-graph.cpp:build_moe_ffn` attaches the hot tensor and writes op_params[4]. The table has four residency lanes so batches 1 through 4 can use distinct routes, including MTP verification; the configured limit is four. The shader selection/bindings live in `ggml/src/ggml-vulkan/ggml-vulkan.cpp`, with the IQ2_S and IQ3_XXS implementations under `vulkan-shaders/`. This is Vulkan/Polaris-specific work, not CUDA graph code.

### Residency: per-layer LRU and safe publication

`src/llama-moecache.cpp:llama_moe_cache_update_vulkan` consumes routes captured by the graph into persistent buffers. It batches route readback, waits for graph completion, updates per-layer residency and admits at most the configured number of experts per layer per step. Weight uploads and changed mapping tables complete before the next graph uses them. There is no promised overlap of publication with the next decode step.

The candidate has 26 host layers x 24 slots = 624 complete gate/up/down expert entries, about 638.625 MiB of quantized weight cache, excluding tables and scratch buffers. Capacity is fixed at initialization; LRU residency changes at runtime. Optional recent-frequency admission and layer-slot controls exist, but the measured default remains fixed per-layer capacity and LRU. They are not an online bandwidth-aware placement solver.

`llama_moe_cache_init/free` track ownership and clean up failed initialization; model destruction releases the owning cache. Captured routes avoid the profiler's per-node host synchronization. The candidate remains scoped to one model/slot; ownership checks do not establish unrestricted multi-model concurrency.

### Prefill: preserve the working scheduler

Large batches bypass the hybrid cache branch and use normal model weights and scheduler handling. llama.cpp deliberately uses GPU host buffers for some CPU-side weights, so Vulkan_Host does not mean an operation must run on CPU. An early global CPU-forcing rule caused the PP collapse; removing that rule and narrowly recognizing hybrid access recovered PP.

`ggml/src/ggml-backend.cpp` anchors only recognized hybrid nodes to the hot-cache backend and permits their cold source without a staging copy. Generic Vulkan_Host support is not globally forced on. `ggml_vk_tensor_subbuffer` resolves pinned host buffers; graph overlap bookkeeping uses the actual subbuffer instead of casting incompatible buffer contexts. A later host-view fix removed a duplicated view offset.

### Agent state: reuse, not faster raw prefill

`common/chat.h` identifies optional TOOL boundaries alongside USER boundaries. `tools/server/server-context.cpp` uses them in the existing checkpoint selection policy when LLAMA_SERVER_TOOL_CHECKPOINTS=1. It retains the existing state serialization/reuse machinery and checkpoint limits rather than implementing a second state cache. When a tool result changes, an earlier valid checkpoint can spare recomputation of the unchanged prefix. This improves incremental wall time, not the kernel's PP rate. Existing MTP, quantized KV and checkpoint infrastructure remain upstream mechanisms.

## Articles, adaptation and deliberate omissions

Primary references, checked against the papers rather than Reddit commentary:

- [FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution, arXiv:2608.16157v1](https://arxiv.org/html/2608.16157v1), especially sections 3.1-3.3 and 4.1.
- [ATSInfer: Automated Tensor Scheduling for Hybrid CPU-GPU LLM Inference on Consumer Devices, arXiv:2607.10183v1](https://arxiv.org/html/2607.10183v1), especially sections 4.1-4.4.
- [User-supplied ATSInfer discussion](https://www.reddit.com/r/LocalLLaMA/comments/1v0vp9k/paper_automated_tensor_scheduling_for_hybrid/) is discovery context, not benchmark evidence for this fork.

| Paper mechanism | This fork |
| --- | --- |
| FreeToken shared LRU expert cache | Adapted to fixed per-layer slots and a Vulkan hybrid dispatch; no global pool. |
| FreeToken q-star split of misses between CPU execution and GPU cache fills | Not integrated; cold misses are read by the Vulkan kernel. |
| FreeToken prefill transfer/computation double buffering | Not integrated; normal prefill scheduling retained. |
| FreeToken semantic-boundary recurrent-state reuse | Narrow adaptation: TOOL boundaries in existing llama.cpp checkpoints, not the complete paper policy. |
| FreeToken elastic cache resizing and loading layout | Not implemented; no automatic reaction to changing free VRAM. |
| ATSInfer profiled tensor placement with memory and switching costs | Not implemented as a solver; host26 is an explicitly selected placement. |
| ATSInfer load-aware dynamic transfer and asynchronous CPU/GPU coordination | Not integrated as a general runtime scheduler. |

These comparisons describe inspiration and differences, not an official port or reproduction of either paper's speedups. The starting expert-cache implementation is the csantiago78 branch named above; the single-dispatch kernel and scheduler integration are the adaptation here.

Hardware-driven exclusions from our own experiments: the dual CPU/GPU FFN chains plus merge cost more than the single hybrid chain; a more realistic concurrent fixed split also lost after output assembly. Global-capacity trace replay showed only about 0.2 percentage points of additional hits for the compared frequency policy, insufficient to justify a large refactor then. Prefill overlap did not establish a deployable benefit. These are local observations, not claims that the papers' techniques cannot work. Revisit only with a specific cost bottleneck and a bounded experiment.

## Change history and accepted checkpoint

| Stage | Change and outcome |
| --- | --- |
| Base bccbacdb | Original expert-cache branch; original single-token consumption needed adaptation for MTP. |
| Development before publication | Packed routes, single-dispatch quant kernels, adaptive residency, narrow scheduler access, pinned-buffer/view fixes, lifecycle checks and optional TOOL checkpoints. Preserved together in initial private commit a40e826e8. |
| d7fb6792b | Four legacy marker writes fixed; full DEV build and server restart completed. Short output regression: PP187.47, TG32.00, 60/67 MTP, hash identical to prior same-profile output. |
| Current consolidation | Reject all cache modes except explicit vulkan_host, even with old experimental opt-in. Preserve the active algorithm and add runtime checkpoint tooling and this mechanism map. |

The completed coding task and prior quality checks are accepted evidence; this consolidation does not request repeating them. They do not imply every possible task or context length is correct. The short 128-token completion is a separate regression check, not a completed coding task.

## Build

Use a fresh build directory with a Vulkan SDK/toolchain and Ninja available. Never reuse another tree's CMakeCache.txt.

```sh
cmake -S . -B build-tiel -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DLLAMA_BUILD_SERVER=ON -DGGML_CCACHE=OFF
cmake --build build-tiel --target llama-server
```

Do not replace a production executable. No model, binary, user conversation, API key or private benchmark request is distributed here.

## Candidate runtime profile

Model used locally: Tiel-Coder-35B-A3B-MTP-UD-IQ3_XXS.gguf (obtain separately). Settings below describe the measured profile, not safe defaults for every machine.

Environment (start without inherited experimental GGML_VK_, LLAMA_MOE_CACHE_, LLAMA_SERVER_, LD_PRELOAD or RADV_PERFTEST settings):

```text
GGML_VK_ALLOW_GRAPHICS_QUEUE=1
GGML_VK_FA_GCN_OCCUPANCY_KB=18
LLAMA_MOE_CACHE_MODE=vulkan_host
LLAMA_MOE_CACHE_MAX_TOKENS=4
LLAMA_MOE_CACHE_ADAPTIVE=1
LLAMA_MOE_CACHE_ADAPTIVE_STATS=1
LLAMA_SERVER_TOOL_CHECKPOINTS=1
```

Server options:

```text
-m <model.gguf> --host 127.0.0.1 --port 8081
-c 65536 -np 1 -ngl 99 -t 8 -tb 8 -b 512 -ub 512
-ctk q4_0 -ctv q4_0 -fa on -lm none --poll 100 --jinja
--spec-type draft-mtp --spec-draft-n-max 1 --spec-draft-p-min 0
--spec-draft-type-k q4_0 --spec-draft-type-v q4_0 --spec-draft-ngl all
--moe-expert-cache 24 --moe-expert-cache-inserts 1
--ctx-checkpoints 8 --cache-ram 512 --checkpoint-min-step 8192
```

Also supply `-ot` with 26 comma-separated overrides, one for each layer 0 through 25:
`blk\.N\.ffn_(up|down|gate|gate_up)_(ch|)exps=Vulkan_Host`, replacing N with each layer number. Do not assume `-ncmoe` alone reproduces this explicit placement. Adaptive initialization should report `FREETOKEN_ACTIVE layers=26 loaded=0 slots=24 enabled=1`; zero loaded is expected before adaptive admission. A disabled cache is not a valid cache benchmark.

Keep only one GPU server active. Bind locally unless deliberately configuring remote access. CPU/GPU frequency policy affects measurements; no privileged power helper is bundled. Context 65536 specifies capacity, not validation at a full 64K prompt.

## Evidence and limitations

Historical short measurements (not repeated during publication):

| Case | PP tok/s | TG tok/s |
| --- | ---: | ---: |
| Earlier host24 baseline | 189.48 | 25.12 |
| Host26/K24/64K candidate | 186.60 | 31.99 |

These used different configurations and are not a statistical estimate of speedup. An earlier hybrid achieved TG28.99 but PP51.68 and worse wall time; it is not the candidate prefill architecture.

With adaptive cache active on both sides, one USER-only versus TOOL-checkpoint A/B reduced the incremental request wall from 11.39591s to 6.58736s. Reused tokens increased 603 to 1601. Extracted code matched and passed 36 cases; generated message lengths differed. This does not prove universal agent speedup. A later 9553-token prompt test is not full-context qualification. Do not add gains from different experiments.

Weight caching does not train or improve model precision. Historical output hashes and MTP acceptance sometimes differed. Broad quality, memory-pressure, multi-model and multi-device regression coverage remains incomplete.

## Known defects / excluded paths

**Do not use legacy full/down split modes as a supported release.** The initial snapshot retained four writes at `op_params + 4*sizeof(int32_t)` although op_params is int32_t[16]. A follow-up changes them to op_params[4]; the old offset overwrote tensor flags. The changed graph translation unit compiled, and a standalone test with the real ggml_tensor header plus ASan/UBSan checked marker/flags behavior at four capacities. This is not full legacy-mode inference validation. Candidate vulkan_host already used op_params[4] and its branch is unchanged. The full DEV server build was subsequently relinked successfully; corrected libllama SHA256 is `602639b82e4935baf48ac19a66776311f8927105f0cbfcefecdc97a87efdf4a7`.

Only explicit LLAMA_MOE_CACHE_MODE=vulkan_host is now accepted when requesting a positive expert-cache size. Missing/empty mode, full, down, vulkan_import, vulkan_exact and unknown names disable the cache with `only_vulkan_host_mode_supported`. LLAMA_MOE_CACHE_ALLOW_EXPERIMENTAL no longer bypasses this check. Ordinary cache-off inference (slots <= 0) is unchanged. Static hotsets within vulkan_host remain available; the adaptive candidate profile is unchanged. Rejected implementations remain archived in source, not selectable at runtime. They can be removed in a separately reviewed cleanup; do not mistake retained code for supported features.

The initial publication did not run a fresh benchmark. The marker follow-up was built, deployed and checked with one 128-token completion. Consolidation was built and tested with 14 rejected-mode/opt-in combinations, then the existing synthetic Vulkan adaptive tests: 12 updates and 3 reloads each for IQ2_S and IQ3_XXS with batch4. No new coding-task run or complete backend suite was needed for this initialization gate. Existing benchmark values remain historical, not newly claimed gains from consolidation.

## Provenance pins

DEV patch SHA256: `a5f220f8623c90193d49654f8427b98571718880a4c37343647bcb90063f2b54`.

Build files on disk at publication (not every historical benchmark):

| File | SHA256 |
| --- | --- |
| llama-server | 4ee6495d28acd13ebfac8118a4cd63c90b513a87c453b7310baa42beb0c25854 |
| libggml-vulkan.so.0.22.0 | 6b284f8c13c61c517d371c0213d82f4c3fc6c479e8b531ce9683b056efd56f88 |
| libggml-base.so.0.22.0 | 45cd442cde32d69f2201fb43db18547cb32ac0bf6e173520a5093d9cdefa694b |
| libllama.so.0.3.0 | 35a6bbd293ce52e10711f9c53c29d449574d6c3ed0b9eeda36a950e8a5b74e7a |

The executable is dynamically linked: its hash alone does not identify backend behavior. GitHub Actions is disabled for this private snapshot; no inherited CI jobs should run automatically.

After mode consolidation, DEV libllama SHA256 is `379eabe2b1b9b47f2c4214df2e118be2c83f63f5c3364c2232220c713b79098d`. The remote DEV worktree intentionally retains its base HEAD plus dirty changes; `--version` can report bccbacdb8 even when these changes are compiled. Identify a deployment by the publication commit, source comparison and library hashes together, not by --version alone.

Focused checks supplied in this repository (Linux):

```sh
python tools/tiel/test_checkpoint.py
c++ -std=c++17 -UNDEBUG -I include -I src -I ggml/include tools/tiel/check-mode-policy.cpp -L build-tiel/bin -Wl,-rpath,$PWD/build-tiel/bin -lllama -lggml -lggml-base -o /tmp/tiel-check-mode-policy
/tmp/tiel-check-mode-policy
```

The mode test expects 14 `REJECT ... PASS` lines and 14 `only_vulkan_host_mode_supported` diagnostics, followed by `MODE_POLICY_DONE`; it uses empty synthetic models and no GPU. The adaptive Vulkan update/reload tests mentioned above predate this commit and remain local experimental harnesses, not a claimed comprehensive portable suite.

## Rollback contract: prepare before changing anything

Source checkpoint: tag `checkpoint/rx590-pp187-tg32-20260905` pins d7fb6792b3756217ad9e3970a67cb7943f9e4bc8. Never move or force-push a checkpoint tag. To recover source without deleting current work:

```sh
git fetch origin --tags
git worktree add --detach ../llama-tiel-checkpoint checkpoint/rx590-pp187-tg32-20260905
```

Build that worktree into a new directory, not an existing DEV/production build. A Git tag alone cannot restore a specific dynamically linked runtime. Before each implementation/build/profile change, use the local-only helper:

```sh
python tools/tiel/checkpoint.py save /absolute/checkpoints/unique-name --build /absolute/dev/build --profile /absolute/profile.json --source-commit <exact-commit>
python tools/tiel/checkpoint.py verify /absolute/checkpoints/unique-name
```

The profile JSON must contain `cmd` (full argv) and `env` (explicit runtime overrides). A checkpoint copies the bin directory, profile, CMake cache, helper, dependency listing and SHA256 manifest. It refuses to overwrite a checkpoint or place it within the build. It checks that llama/ggml/mtmd libraries resolve from the saved copy with LD_LIBRARY_PATH. Keep it private: local profiles may contain paths or secrets. Do not commit runtime checkpoints, models or local configuration.

To roll back the running DEV, first confirm its PID and idle slot, stop only that process and restore the same host power policy. Then:

```sh
python /absolute/checkpoints/unique-name/checkpoint.py verify /absolute/checkpoints/unique-name
python /absolute/checkpoints/unique-name/checkpoint.py start /absolute/checkpoints/unique-name
```

`start` verifies every saved file, refuses any existing llama-server and port8080, clears inherited experimental environment and starts the saved executable in the foreground using its saved libraries. It never overwrites a build, kills a process or changes production. Use the normal headless service supervisor if background operation is needed. Confirm /health and FREETOKEN_ACTIVE before resuming Pi; a restart discards in-memory prompt state, not saved conversations.

Limits: GGUF model, driver, system libraries, OS, power settings and KV state are not included. A changed OS/driver may require additional recovery; the saved CMake cache is provenance, not portable build configuration. Hash verification proves saved-file identity, not inference quality. Keep the model unchanged and preserve benchmark evidence alongside the private checkpoint. Do not claim a restore succeeded until health and library paths are checked.

Checkpoint verification performed: 29 saved files matched their manifest, ldd selected saved inference libraries, the saved executable ran --version, and `start` refused to proceed while DEV was running. Three no-GPU integrity tests cover corruption, missing files and an escaping manifest path. A full rollback/model load was deliberately not performed; the current DEV restarted successfully from its newly built library. Source tag and saved runtime remain available if that rollback becomes necessary.

For every future change: create/tag a checkpoint first; state scope and rollback target; change one mechanism; perform the smallest relevant regression check; publish only after reporting results; never silently advance the accepted checkpoint or repeat completed coding tasks without a reason.
