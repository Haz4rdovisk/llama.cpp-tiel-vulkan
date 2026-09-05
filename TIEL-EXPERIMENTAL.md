# Tiel Vulkan experimental snapshot

This private development snapshot preserves the RX590 work before further cleanup. It is not a production release or an upstream submission. Code was developed with AI assistance under user direction. Original authorship, history and licenses remain intact.

Base commit: `bccbacdb8945680f1cfc7e6bffd1e59014705750`, the expert-cache branch by csantiago78, on top of llama.cpp. The initial snapshot preserved 31 DEV implementation files byte-for-byte. The legacy-marker follow-up was also applied to DEV. Production was not changed.

## Candidate architecture

- Single Vulkan MUL_MAT_ID dispatch with cold host weights and hot VRAM weights; no second CPU/GPU chain in the candidate path.
- Packed expert/slot IDs; IQ2_S gate/up and IQ3_XXS down kernels.
- Per-layer adaptive cache, small target/MTP batches up to four, publication at graph boundaries.
- Normal non-hybrid scheduling for prefill; optional tool-message checkpoints for incremental prompts.
- Hardware scope: RX590 8GB, i7-7700, one model/slot. Not a complete implementation of every FreeToken/ATSInfer technique.

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

Down/import/exact require LLAMA_MOE_CACHE_ALLOW_EXPERIMENTAL=1; do not enable it for the candidate. Full remains accessible and is not guarded like those modes. The external-host-import experiment failed on this hardware. Exact mode retains a single-GPU/backend-0 assumption. Legacy custom backend tests need consolidation. These are release blockers, not evidence that all paths are correct.

No new inference, fresh build or complete backend suite was run to publish this snapshot. Prior build and focused microtests covered candidate lifecycle, host views, mode guards and booleans; they are not proof of every mode. Publication checks covered patch applicability, source identity and staged-file hygiene.

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
