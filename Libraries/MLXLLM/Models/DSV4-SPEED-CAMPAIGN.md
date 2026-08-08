# DSV4-Flash Speed Campaign — vmlx-swift + osaurus

Target: **30+ tok/s decode, ~400 pp/s prefill** for DSV4-Flash on M5 Max, cache quant pooling automatic, all proof via live osaurus harness. Land as ONE osaurus PR + one vmlx-swift main push (4-file repin).

## Baselines
- Python vmlx-engine (`/Users/eric/mlx/vllm-mlx`) DSV4 progression: 17.6 → 19.7 tok/s (RAM-scaled cache ceiling) → 1.20–1.45× native decode (lm_head exact-cache) → fused qmm lm_head default.
- Swift JANGTQ_K bench (`~/vmlx/bench-out/dsv4k_run3.log`): 12.71 tok/s, DSV4BatchGenerator single-batch, JANGTQ_WIRED_LIMIT_GB=110.

## Python perf inventory (v1.6.19..v1.6.24, repo /Users/eric/mlx/vllm-mlx)
| Item | Commits | Measured effect |
|---|---|---|
| Affine MoE decode Metal kernels (fused dequant+matvec, b2_g64/b3_g64 routed experts; N_CHUNKS=2, TG_Y=128; exact-layout guarded, stock SwitchGLU otherwise) | cb6a7d043, 97eb6a580, 9d0007e40 (#248) | decode uplift on affine DSV4 |
| RAM-scaled MLX cache ceiling (8GB fixed → scale w/ physical RAM: 16→4.0, 32→5.8, 64→11.5, 96→17.3, 128→23.0, 137→24 capped; DSV4_MLX_CACHE_LIMIT_GB override) | 05b32f0c9 | +11% decode <8k ctx, +8% above; 17.6→19.7 t/s |
| Metal live-buffer ceiling projection (~42 live buffers/token cumulative; Metal cap 499000 → death ~11.9k tok as HTTP 500; project buffer-count cap beside byte cap; oversized→413, unspecified→clamp+log) | efc4e67ea (R21-13) | removes hard crash wall |
| lm_head fused 8-bit quantized_matmul default (0.55 GB/tok packed vs 1.97 exact-cache; −0.62 ms/tok; greedy-identical 160 steps; rel diff 4.8e-04; VMLX_DSV4_LM_HEAD_MODE=exact-cache\|off\|native) | 23d2ac294 (supersedes fae852756 #250) | decode 1.20–1.45× native |
| Bounded exact RoPE cos/sin table sharing across equal-frequency instances | 4472fe876 (#250) | part of above |
| Delta restore geometry from records not literals | fc554c0f2 | restore correctness/speed |
| Answer-reserve budget | 1.6.23 | trim protects answer window |
| L2 validation read caching | fa9f4c50a | fewer disk reads on restore |
| Short append checkpoints | 36458e550 | cheaper incremental stores |
| Hot/warm/cold defaults | e1e6f93ad | tiered cache policy |
| Pre-refault eviction | 2c5324a1f | avoids paging stalls |
| Pool quant default ON | 949aab70c + .tmp-r20 (DSV4_POOL_QUANT_DEFAULT=True) | automatic cache quant pooling |

## Swift gap audit (vmlx-swift, this repo)
- **lmHeadFp32 = biggest gap**: `DeepseekV4MathHelpers.swift:794` dequantizes the WHOLE quantized lm_head to FP32 on EVERY call (called from DeepseekV4JANGTQ.swift:230). Port: fused `quantizedMM` with fp32 h (Python 23d2ac294). → top priority.
- **RoPE cos/sin per call per instance**: `DeepseekV4.swift:69-81` — port bounded shared tables (4472fe876).
- **Affine MoE decode kernels**: plain DSV4 MoE = SwitchGLU (`DeepseekV4.swift:585-612`), JANGTQ = TurboQuantSwitchGLU — port fused decode kernels (#248) behind exact-layout guard.
- **Live-buffer ceiling projection**: not present — port efc4e67ea guard.
- **Cache ceiling: ALREADY COVERED** — `ModelFactory.swift:738-767` `applyPlainDeepseekV4ProcessMemoryLimitsIfNeeded` (nativeCeiling = max(1GB, min(95% physical, 1.5× recommendedWorkingSet))).
- **Pool quant: ALREADY DEFAULT-ON** — `DeepseekV4Compressor.swift:288-351` `resolvePoolQuantizationDefault` (opt-out DSV4_POOL_QUANT∈{0,false,off,no}).
- Delta-restore geometry / L2 read caching / append checkpoints / hot-warm-cold / pre-refault eviction: audit each against Swift disk-L2 path, port where missing.

## TODO (progress)
- [x] Python inventory (above)
- [x] Swift gap audit (above)
- [x] Port 1: lm_head fused qmm (`DeepseekV4MathHelpers.lmHeadFp32`, default qmm, `VMLX_DSV4_LM_HEAD_MODE=exact` opt-out). A/B PROVEN: exact vs qmm greedy-identical T1/T2/T3 on DSV4-Flash (BENCH_DSV4_COHERENCE chat row, temp0, cache off); turn1 wall 37.36s→8.87s (whole-lm_head dequant eliminated).
- [x] Port 2: RoPE cos/sin table sharing (`DeepseekV4RoPE.sharedCosSin`, keyed single-entry table; same A/B covers it — both runs used shared tables).
- [x] Port 3: **SKIPPED — correct per Python's own final state.** 9d0007e40 made the affine MoE fastpath opt-in (`VMLX_DSV4_AFFINE_MOE_FASTPATH`, default "0") because "Current MLX on M5 Max is already faster on native gather_qmm (18.7 vs 17.1 tok/s in the exact 400-token keeper A/B)". PR #248's win was vs the older stock runtime; porting the kernels would REGRESS the M5 Max default.
- [x] Port 4: live-buffer ceiling projection (`MetalLiveBufferGuard` in MLXLMCommon; clamp at BatchSlot init + TokenIterator init; `DeepseekV4Cache: MetalBufferRetainingCache` = 1.0/layer/token; env parity VMLX_METAL_BUFFER_COUNT_GUARD/FRACTION/BASELINE; 12 pinning tests green, cap=10063 == Python (499000×0.90−16384)/43; inert for conventional KV by construction).
- [x] Port 5 audit: **Q1 delta geometry OK** (TQDiskSerializer meta `[keep, maxSize, step, offset, idx, compressRatio, slidingWindow]` fully record-derived, canRestore validates vs runtime, no literals). **Q2 no action** — on acceptance `DiskCache.touchRecency` re-stats the file via `_fileFingerprint(url:)`, but that is metadata-only (content bytes are read exactly once at fetch) AND is the orphan/stale-row guard; Python fa9f4c50a's win was avoiding repeated content reads, which Swift never does. **Q3 append checkpoints OK** (no minimum-length/alignment thresholds in store path). e1e6f93ad (panel UI) + 2c5324a1f (test attestation) N/A engine-side.
- [x] Bench (RunBench BENCH_SIMPLE, 2000-tok prompt, 400 gen, 2 runs each): **eager 22.7–22.8 tok/s decode, TTFT 6.92–7.51s → ~267–289 pp/s**. Compile-ON REGRESSES DSV4: 13.9–14.5 tok/s, TTFT 10.2–12.4s — compiled decode must stay OFF for this family (consistent with osaurus default). vs 12.71 tok/s JANGTQ baseline = **1.79×**.
- [ ] Mirror needed osaurus-side wiring (repin 4 files + any loader/settings surface)
- [ ] LIVE harness proof (background toolkit): multiturn + TTFT/tok-s from chat stats chips
- [ ] One osaurus PR + vmlx-swift main push

## Proof rules
- Live osaurus GUI only (background postToPid toolkit; no curl/API/CLI for proof).
- Bench-vs-live: chat stat chips (TTFT s • tok/s • tokens) are the accepted live numbers.
