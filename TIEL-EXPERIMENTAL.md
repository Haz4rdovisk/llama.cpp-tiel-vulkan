# Tiel Vulkan experimental snapshot

This private development snapshot preserves the promoted long-prefill-residency/K40 RX590 checkpoint. It is not a production release or an upstream submission. Code was developed with AI assistance under user direction. Original authorship, history and licenses remain intact.

Base commit: `bccbacdb8945680f1cfc7e6bffd1e59014705750`, the expert-cache branch by csantiago78, on top of llama.cpp. This branch contains the cleaned single-dispatch adaptation, phase-safe K24/K40 capacity, router-stability fix and checkpoint tooling. Production was not changed.

## Promoted DEV architecture

- Single Vulkan MUL_MAT_ID dispatch with cold host weights and hot VRAM weights; no second CPU/GPU chain in the candidate path.
- Packed expert/slot IDs; IQ2_S gate/up and IQ3_XXS down kernels.
- Per-layer adaptive cache, small target/MTP batches up to four, publication at graph boundaries.
- K24 base residency plus a prefill-aware 16-slot per-layer decode bank (K40 effective), allocated only when scheduler/device budgets preserve a 128 MiB VRAM reserve and revoked before prefill returns.
- Optional profiled redistribution keeps exactly 416 transient slots but varies them by layer through `LLAMA_MOE_CACHE_EXTRA_PLAN`; an invalid plan preserves K24 rather than allocating an ambiguous bank.
- Source-generated, tested banked Vulkan SPIR-V and a seven-binding dispatch that selects cold host, base VRAM or extra-bank VRAM weights without a CPU/GPU merge.
- Long prefills may reuse the idle K24 base allocation for two complete device-local expert layers after three full ubatches; all decode slices and tables are restored before K40 decode. Short prefills keep normal scheduling.
- Optional tool-message checkpoints reuse unchanged agent prefixes.
- Hardware scope: RX590 8GB, i7-7700, one model/slot. Not a complete implementation of every FreeToken/ATSInfer technique.

## Mechanisms and source map

### Decode: one dispatch, three weight locations

For each eligible MoE projection, src[0] remains the authoritative cold expert tensor in Vulkan_Host memory and src[3] references quantized hot weights in VRAM. The table lookup produces a packed I32 route: high 16 bits are the original expert ID; low 16 bits are the cache slot, or 0xffff on a miss. The Vulkan shader selects the weight source within the same dispatch. No separate expert-to-slot descriptor binding or CPU/GPU output merge is required.

Gate/up use IQ2_S, down uses IQ3_XXS for this model. The cache copies quantized slices, not re-quantized approximations. This preserves stored weights, but does not guarantee identical floating-point reduction order against every baseline kernel. MTP acceptance and output hashes therefore remain regression signals.

`src/llama-graph.cpp:build_moe_ffn` attaches the hot tensor and writes op_params[4]. The table has four residency lanes so batches 1 through 4 can use distinct routes, including MTP verification; the configured limit is four. The shader selection/bindings live in `ggml/src/ggml-vulkan/ggml-vulkan.cpp`, with the IQ2_S and IQ3_XXS implementations under `vulkan-shaders/`. This is Vulkan/Polaris-specific work, not CUDA graph code.

### Residency: per-layer LRU and safe publication

`src/llama-moecache.cpp:llama_moe_cache_update_vulkan` consumes routes captured by the graph into persistent buffers. It batches route readback, waits for graph completion, updates per-layer residency and admits at most the configured number of experts per layer per step. Weight uploads and changed mapping tables complete before the next graph uses them. There is no promised overlap of publication with the next decode step.

The candidate has 26 host layers x 24 base slots = 624 complete gate/up/down expert entries, about 638.625 MiB of quantized weight cache, excluding tables and scratch buffers. After a real prefill-to-decode transition it may add 16 slots per layer (416 entries, 446,431,232 bytes), yielding effective K40 and about 1,064.375 MiB of expert weights. Server warmup alone cannot trigger this bank. The allocation budget is the greater of the scheduler phase-arena release and reliable Vulkan free-memory reporting minus a fixed 128 MiB reserve. Reports with `free == total` are treated as unreliable. Failure to allocate leaves K24 active; returning to prefill revokes only the transient bank. Capacity changes only at the synchronized PP/TG boundary; LRU residency changes at runtime. Optional recent-frequency admission exists, but the measured default remains uniform per-layer LRU. This is not an online bandwidth-aware placement solver or the paper's fully elastic global cache.

The base/extra split is a shader ABI. `ggml_backend_tiel_banked_abi` is published through the Vulkan backend registry and checked before allocating the extra bank. A mismatched rebuilt Vulkan library therefore disables K40 rather than silently interpreting packed extra-slot IDs with the wrong shader. All six banked IQ2_S/IQ3_XXS variants are generated from tracked GLSL by the normal Vulkan shader generator; no opaque prebuilt SPIR-V header is required. A fresh build produced byte-identical SPIR-V to the approved K32 runtime artifacts and passed `spirv-val`.

The profiled successor uses total per-layer capacities `32,30,31,35,49,35,43,36,35,43,38,58,45,43,33,40,34,44,41,34,39,42,42,46,48,44`. K24 remains the base; only the same 416 transient slots are redistributed. Omitting the variable retains uniform K40.

`llama_moe_cache_init/free` track ownership and clean up failed initialization; model destruction releases the owning cache. Captured routes avoid the profiler's per-node host synchronization. The candidate remains scoped to one model/slot; ownership checks do not establish unrestricted multi-model concurrency.

### Prefill: phase-resident complete layers

Large batches bypass the hybrid decode branch. llama.cpp deliberately uses GPU host buffers for some CPU-side weights, so Vulkan_Host does not mean an operation must run on CPU. An early global CPU-forcing rule caused the PP collapse; removing that rule and narrowly recognizing hybrid access recovered PP.

With `LLAMA_TIEL_PREFILL_RESIDENCY=1`, the third consecutive full prefill ubatch may repurpose the otherwise idle monolithic K24 base-cache allocation. Complete gate/up/down tensors are copied once for as many whole host layers as fit; the current 669,753,344-byte allocation holds two layers (549,453,824 bytes). The graph substitutes only those device-local aliases and leaves all other weights and normal MoE IDs unchanged. Before decode, every occupied K24 expert slice and all four mapping lanes are rebuilt from authoritative Vulkan_Host weights, then the profiled K40 bank is allocated normally. Activation, restoration or graph-reservation failure falls back without changing the immutable checkpoint.

This is persistent residency across ubatches, not the paper's transfer/computation double buffer: uploads are synchronized once at phase entry, and no transfer queue overlaps an active graph. The three-ubatch threshold avoids paying the transition cost for short prompts.

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
| FreeToken prefill transfer/computation double buffering | Narrow alternative for this hardware: two complete expert layers persist in the borrowed K24 allocation across long-prefill ubatches. There is no transfer/compute overlap or streaming double buffer. |
| FreeToken semantic-boundary recurrent-state reuse | Narrow adaptation: TOOL boundaries in existing llama.cpp checkpoints, not the complete paper policy. |
| FreeToken elastic cache resizing and loading layout | Partial, bounded adaptation: a synchronized K24-to-K40 per-layer bank follows the prefill/decode phase and reliable Vulkan memory budget, preserves 128 MiB, and falls back to K24. No continuous resizing, global pool or FTW layout. |
| ATSInfer profiled tensor placement with memory and switching costs | Narrow offline adaptation: measured routes drive a fixed host26 extra-slot plan under the same memory budget. No online solver or tensor switching policy. |
| ATSInfer load-aware dynamic transfer and asynchronous CPU/GPU coordination | Not integrated as a general runtime scheduler. |

These comparisons describe inspiration and differences, not an official port or reproduction of either paper's speedups. The starting expert-cache implementation is the csantiago78 branch named above; the single-dispatch kernel and scheduler integration are the adaptation here.

Hardware-driven exclusions from our own experiments: the dual CPU/GPU FFN chains plus merge cost more than the single hybrid chain; a more realistic concurrent fixed split also lost after output assembly. Global-capacity trace replay showed only about 0.2 percentage points of additional hits for the compared frequency policy, insufficient to justify a large refactor then. Prefill overlap did not establish a deployable benefit. These are local observations, not claims that the papers' techniques cannot work. Revisit only with a specific cost bottleneck and a bounded experiment.

## Change history and accepted checkpoint

| Stage | Change and outcome |
| --- | --- |
| Base bccbacdb | Original expert-cache branch; original single-token consumption needed adaptation for MTP. |
| Development before publication | Packed routes, single-dispatch quant kernels, adaptive residency, narrow scheduler access, pinned-buffer/view fixes, lifecycle checks and optional TOOL checkpoints. Preserved together in initial private commit a40e826e8. |
| d7fb6792b | Four legacy marker writes fixed; full DEV build and server restart completed. Short output regression: PP187.47, TG32.00, 60/67 MTP, hash identical to prior same-profile output. |
| Router-stable K32 | Keeps router logits alive for cached batches so K24/K32 use the same safe graph lifetime; fixed-sequence logits and a 512-token MTP run matched exactly. |
| Current consolidation | Removed rejected cache modes, prefill-copy and per-layer slot-plan runtime code; retained explicit negative policy tests. Added phase-banked capacity, checkpoint isolation and the Vulkan banked-ABI capability check. |
| K40 promotion | Expanded only the transient bank to 16 slots/layer, gated it on real prefill, added reliable Vulkan-budget handling and retained a 128 MiB reserve. Two controlled 512-token runs preserved K32 output/MTP while improving decode and wall time. |
| Profiled K40 redistribution | Dynamic programming over current routes redistributed the same 416 transient slots. Replay removed 445 misses (0.83%); clean TG was 35.54 with exact K40 output/MTP and no extra VRAM. |
| Long-prefill residency candidate | Reuse the idle K24 allocation for two full layers after three ubatches, then reconstruct K24 and allocate profiled K40 before decode. A 27,473-token code prompt improved PP from 86.29 to 152.87/148.91 with identical output and MTP. |

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
LLAMA_SERVER_TOOL_CHECKPOINTS=1
LLAMA_TIEL_PHASE_ARENA=1
LLAMA_TIEL_EXTRA_BANK=1
GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1
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
`blk\.N\.ffn_(up|down|gate|gate_up)_(ch|)exps=Vulkan_Host`, replacing N with each layer number. Do not assume `-ncmoe` alone reproduces this explicit placement. Adaptive initialization should report `FREETOKEN_ACTIVE layers=26 loaded=0 slots=24 enabled=1`; zero loaded is expected before adaptive admission. On the decode transition, a matching K40 runtime reports exactly one `TIEL_PHASE_BUDGET ... reliable=1 reserve=134217728 ...` followed by `FREETOKEN_EXTRA active=1 layers=26 slots=16 bytes=446431232`. Absence of the second line means safe K24 fallback, not a K40 benchmark. A disabled cache or `backend_abi_unavailable` run is invalid cache evidence.

The profiled profile additionally sets `LLAMA_MOE_CACHE_EXTRA_PLAN=8,6,7,11,25,11,19,12,11,19,14,34,21,19,9,16,10,20,17,10,15,18,18,22,24,20`. `FREETOKEN_EXTRA_PLAN total=416` confirms acceptance. Without it, K40 remains uniform.

Keep only one GPU server active. Bind locally unless deliberately configuring remote access. CPU/GPU frequency policy affects measurements; no privileged power helper is bundled. Context 65536 specifies capacity, not validation at a full 64K prompt.

## Evidence and limitations

Historical short measurements (not repeated during publication):

| Case | PP tok/s | TG tok/s |
| --- | ---: | ---: |
| Earlier host24 baseline | 189.48 | 25.12 |
| Host26/K24/64K candidate | 186.60 | 31.99 |
| Router-stable K24 control, 512 generated | 189.25 | 33.57 |
| Router-stable K32, 512 generated | 188.83 | 34.50 |
| Host26/K40 proof, 512 generated | 188.28 | 35.57 |
| Host26/K40 promotion validation, 512 generated | 189.91 | 35.18 |
| Host26/profiled-K40 validation, 512 generated | 188.25 | 35.54 |
| Profiled K40, prefill residency OFF, 27,473 prompt | 86.29 | 22.82 |
| Profiled K40, prefill residency ON, first cycle, 27,473 prompt | 152.87 | 22.68 |
| Profiled K40, prefill residency ON, second cycle, 27,473 prompt | 148.91 | 25.36 |

The first two rows used different configurations and are not a statistical estimate of speedup. The paired K24/K32 rows used the same corrected runtime: K32 improved TG 2.78% and wall 1.92%, with identical output SHA256 and MTP 240/271. The K40 rows used the same K32 prompt, seed and greedy output contract; both retained SHA256 `360c18cfe36f0a75ab97d2eeeb2a45a13c1a59617d45a00d9b17bf14dd496266`, 1,922 characters and MTP 240/271. K40 wall was 19.971 s and 20.082 s versus 20.391 s for K32. These bounded runs are not confidence intervals. An earlier hybrid achieved TG28.99 but PP51.68 and worse wall time; it is not the candidate prefill architecture.

With adaptive cache active on both sides, one USER-only versus TOOL-checkpoint A/B reduced the incremental request wall from 11.39591s to 6.58736s. Reused tokens increased 603 to 1601. Extracted code matched and passed 36 cases; generated message lengths differed. This does not prove universal agent speedup. A later 9553-token prompt test is not full-context qualification. Do not add gains from different experiments.

The long-prefill residency gate used the same 100,128-character llama.cpp code prompt, 27,473 prompt tokens, greedy seed and 24-token output. OFF took 318.377s of prompt evaluation (86.29 PP) and 319.539s wall. ON took 179.719s/180.781s on its first cycle and 184.495s/185.451s on its second cycle with prompt caching disabled, a 42-43% wall reduction. Both ON cycles and OFF produced SHA256 `30b392a378883b255cd6171e009e1be77063ebc48cd282b8c5d0dfdd8e469ae1` and MTP 9/13. A smaller 5,417-token A/B improved PP by 0.71% and wall by 0.82%, showing why activation is restricted to long prefills. The 24-token decode samples are too short to claim a TG change; they only verify that profiled K40 was restored and output remained exact.

Before promotion, the saved runtime also completed real Pi auto-compaction traffic. A fresh 16,420-token prompt with zero prompt-cache reuse reached 162.55 PP, then restored profiled K40 and decoded 793 tokens at 29.50 TG with MTP acceptance 341/452 (75.44%). A subsequent fresh 30,734-token compacted prompt reached 142.51 PP before restoring the same 416-slot profiled extra bank; early decode was about 27.5 TG. The larger prompt is not an exact A/B, but its PP is 69% above the historical 31,882-token result at 84.22 PP. These live runs confirm phase transitions under the target workflow; they do not replace the controlled A/B above.

Weight caching does not train or improve model precision. A manual coding run from the immutable K32 checkpoint was judged unusually strong by the user, but that is quality evidence for the checkpoint, not proof that caching trained or universally improved the model. Historical output hashes and MTP acceptance sometimes differed. Broad quality, memory-pressure, multi-model and multi-device regression coverage remains incomplete.

Bounded post-checkpoint experiments were retained as external evidence but not source code. b1024/ub1024 increased aggregate real-workload PP 14.54% but reduced TG 3.97%, changed the trajectory and reached 98.10% sampled VRAM use. An earlier static two-full-layer placement gained 1.01% PP but lost 1.80% TG because it remained active during decode; a six-down-tensor ATSInfer-lite placement gained 1.48% PP but lost 10.17% TG through a changed MTP trajectory. Both were removed completely. The later phase-resident candidate differs by borrowing the decode cache only after three prefill ubatches and restoring exact profiled K40 before generation. A layer-publication/dedicated-transfer prototype preserved output but lost about 1% TG; traces showed required scheduler/output synchronization drained the intended overlap. Rejected percentages are not added to the accepted checkpoint.

## Excluded paths and cleanup status

The initial snapshot contained legacy full/down/import/exact implementations, including invalid marker writes at `op_params + 4*sizeof(int32_t)`. Those runtime implementations and their unused interfaces were removed after their experiments were rejected. The supported vulkan_host path uses `op_params[4]`; its marker behavior was checked with the real ggml_tensor header under ASan/UBSan. Historical patches remain outside the fork for audit and recovery only.

Only explicit LLAMA_MOE_CACHE_MODE=vulkan_host is accepted when requesting a positive expert-cache size. Missing/empty mode, full, down, vulkan_import, vulkan_exact and unknown names disable the cache with `only_vulkan_host_mode_supported`. LLAMA_MOE_CACHE_ALLOW_EXPERIMENTAL does not bypass this check. Ordinary cache-off inference (slots <= 0) is unchanged. Static hotsets within vulkan_host remain available; the adaptive candidate profile is unchanged. Rejected implementations have been removed from the runtime source; their names remain only in the negative mode-policy regression test and historical evidence outside the fork.

The initial publication did not run a fresh benchmark. The marker follow-up was built, deployed and checked with one 128-token completion. Consolidation was built and tested with 14 rejected-mode/opt-in combinations, then the existing synthetic Vulkan adaptive tests: 12 updates and 3 reloads each for IQ2_S and IQ3_XXS with batch4. No new coding-task run or complete backend suite was needed for this initialization gate. Existing benchmark values remain historical, not newly claimed gains from consolidation.

K40 promotion reused the accepted K32 banked shaders. Its external host-buffer contracts covered IQ2_S and IQ3_XXS with 96 adaptive updates, real extra-bank routes, exact copied weights/maps/outputs, revocation, three reloads and cleanup. The final source build passed all five checkpoint integrity tests and one controlled model run; no K sweep was performed. Host27/K40 was valid but inferior and was not promoted. A warmup-triggered trial decoded at K24 and is explicitly excluded from K40 results.

For per-layer redistribution, a first plan derived from older traces regressed the current measured hit rate from 52.800% to 52.583% and was rejected. Temporary tracing then captured the current route sequence; its optimized plan replayed at 53.048% versus 52.656% uniform, removing 445 misses. Route-trace code was removed before promotion. The clean run matched the best uniform K40 within noise, so the throughput delta is not claimed as statistically isolated; the demonstrated gain is fewer cold PCIe routes under an unchanged memory budget.

## Provenance pins

The immutable source tag `checkpoint/rx590-k32-source-complete-20260908` identifies the cleaned, rebuildable K32 source checkpoint. The separately saved local runtime checkpoint contains its own SHA256 manifest and complete dynamic-library set; binaries, the GGUF model, profiles and private benchmark payloads are not distributed in this repository.

The accepted successor tag `checkpoint/rx590-k40-prefill-aware-20260909` adds only the prefill-aware 16-slot transient bank and guarded Vulkan memory budget. Its private runtime checkpoint is `/home/lucas/.local/state/tiel-agentic-dev/checkpoints/k40-prefill-aware-20260909`; verification matched commit `68e876f01` and 27 saved files. K32 remains immutable as the immediate rollback.

The profiled successor tag `checkpoint/rx590-k40-profiled-slots-20260909` adds the validated per-layer plan interface. Its private runtime checkpoint verifies commit `e79184ecf` and 27 saved files. Uniform K40 remains its immediate rollback.

The promoted successor tag `checkpoint/rx590-long-prefill-residency-20260910` adds phase-resident complete layers for long prefill while restoring the exact profiled K40 decode path at the PP/TG boundary. Its private runtime checkpoint is `/home/lucas/.local/state/tiel-agentic-dev/checkpoints/prefill-residency-v1-20260910`; 27 manifest files and source commit `220e29ccb` were verified before promotion. Profiled K40 remains the immediate rollback and was not moved or overwritten.

The earlier immutable tag `checkpoint/rx590-k32-router-stable-20260907` is retained for audit but is not a complete source rollback: it omitted the banked Vulkan source/header while the saved runtime already contained that backend. Do not deploy or rebuild K32 from that tag. It was not moved or force-updated; this follow-up restores the exact approved backend source and has its own replacement tag.

The server executable is dynamically linked, so its hash alone does not identify backend behavior. Identify a deployment by the immutable source tag plus the complete saved-runtime manifest, not by `--version`: the inherited version string can still report the upstream base commit. GitHub Actions is disabled for this private snapshot; no inherited CI jobs should run automatically.

Focused checks supplied in this repository (Linux):

```sh
python tools/tiel/test_checkpoint.py
c++ -std=c++17 -UNDEBUG -I include -I src -I ggml/include tools/tiel/check-mode-policy.cpp -L build-tiel/bin -Wl,-rpath,$PWD/build-tiel/bin -lllama -lggml -lggml-base -o /tmp/tiel-check-mode-policy
/tmp/tiel-check-mode-policy
```

The mode test expects 14 `REJECT ... PASS` lines and 14 `only_vulkan_host_mode_supported` diagnostics, followed by `MODE_POLICY_DONE`; it uses empty synthetic models and no GPU. The adaptive Vulkan update/reload tests mentioned above predate this commit and remain local experimental harnesses, not a claimed comprehensive portable suite.

## Rollback contract: prepare before changing anything

Accepted source checkpoint: immutable tag `checkpoint/rx590-long-prefill-residency-20260910`. Profiled `checkpoint/rx590-k40-profiled-slots-20260909` remains the immediate rollback, followed by uniform K40 and K32. Never move or force-push a checkpoint tag. To recover the accepted source without deleting current work:

```sh
git fetch origin --tags
git worktree add --detach ../llama-tiel-checkpoint checkpoint/rx590-long-prefill-residency-20260910
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

K32 checkpoint verification performed: 28 saved files matched their manifest, `ldd` selected the saved inference libraries, the saved executable ran `--version`, and the restored runtime reached healthy state on DEV port 8081. K40 runtime verification matched 27 files, selected its saved inference libraries and pinned source commit `68e876f01`. Five no-GPU integrity tests cover corruption, missing files, an escaping manifest path and the checkpoint/start safety contract. A fresh isolated source build passed the banked host-buffer contract for IQ2_S and IQ3_XXS: 96 adaptive updates, extra-bank accesses, byte-exact resident weights/maps, output equality, revocation and three reloads. K32 and K40 source tags and saved runtimes are independent rollback anchors.

For every future change: create/tag a checkpoint first; state scope and rollback target; change one mechanism; perform the smallest relevant regression check; publish only after reporting results; never silently advance the accepted checkpoint or repeat completed coding tasks without a reason.
