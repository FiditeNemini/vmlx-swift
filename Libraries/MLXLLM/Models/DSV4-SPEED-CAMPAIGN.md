# DSV4-Flash Speed Campaign — vmlx-swift + osaurus

Target: **35+ tok/s stable decode and ~400 pp/s prefill** for DSV4-Flash on M5 Max, with **33 tok/s as the mandatory minimum acceptance floor**, cache quant pooling automatic, and all final proof via the live osaurus harness. Decode speed must hold for short and long answers, including the final generation window, and at very long prompt context. Land as ONE osaurus PR + one vmlx-swift main push (4-file repin).

## Non-negotiable acceptance gates
- **Generated-length stability:** report decode in 128-token windows, excluding TTFT. Every full window, including the final full window, must remain at or above the mandatory **33 tok/s** floor for both short-context and long-context prompts; **35+ tok/s** remains the target. An aggregate average cannot hide a late-output collapse.
- **Speed-first sequencing:** do not spend time on 1K-token, near-500K, full GUI, or long correctness campaigns until a short 257-token/two-window row clears 33 tok/s. Use those long rows only after the decode graph is fast enough to make them decision-useful.
- **Completion latency:** separately measure the interval from the final decoded token until the stream reports completion. SSD cache snapshot/persistence must not leave the UI visibly generating after decoding has ended.
- **Prefill:** target **~400 prompt tokens/s** using real runtime prefill telemetry, reported separately from cache restore and media/tool preparation.
- **Near-500k context:** run a real near-500,000-token synthetic performance row with quantized pooled cache enabled. Report prefill rate, every decode window, final-window rate, cache topology/counters, cache payload bytes, and Activity Monitor `phys_footprint`.
- **Very-long-context memory target:** the effective quantized pooled-cache payload at the near-500k row should remain approximately **6–8 GB**. Report the measured payload and process physical footprint separately; neither an estimate nor virtual size is proof.
- **Correctness at every scale:** performance rows remain partial until coherent multi-turn output, native DSML/tool-call rendering, stable prompt boundaries, disk prefix/suffix reuse, and bundle-driven generation defaults are independently proven.

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

## Newer DSV4 reference inventory (JANG / antirez-vMLX lane, 2026-08-07)
These changes are reference designs, not Swift proof. Each port still requires a guarded A/B, greedy/coherence parity, physical-footprint measurement, and short/long/near-500k validation.

| Reference change | Commit | Why it matters to the Swift collapse |
|---|---|---|
| Attention-only 512-token subchunks inside 2048-token outer prefill chunks; fine-grained single-token compiled regions | `81bfec4` | Preserves MoE throughput while bounding attention memory; reference 8k prefill improved 319 to 452 pp/s. This is not equivalent to the regressing whole-model compile mode already rejected in Swift. |
| Keep pooled state BF16 through a 64 MiB hot tier | `81bfec4` | Avoids repeated q8 dequant/materialization in ordinary long-context decode; the current Swift tier is only 2 MiB. |
| Defer per-token pool advancement until a compression-window boundary | `f5e1bc7` | Removes redundant per-token pool projections/advancement and reuses the last materialized pool between boundaries. |
| Enable layerwise prefill barriers only beyond 24,576 tokens or when attention subchunking is unavailable | `71082f1` | Avoids the 25–30% penalty of blanket per-layer evaluation at moderate context. |
| Heads-16 indexed-attention Metal kernel with KV staged once and fp32 online softmax/sink | `87ef169` | Targets the growing indexed pool-attention cost; reference uplift grows with context (about 8% at 8k, 30% at 15k, 59% at 40k). |
| Bound pool attention with a tiled view beyond 16,384 pool rows | `07a4c15` | Prevents very-long-prefill OOM while retaining hot BF16 storage. |
| Compact 64-row q8 segments into 16K-row slabs | `c8cea7f` | Prevents per-layer operation count from scaling with hundreds of tiny quantized segments near one-million-token context. |
| Restore q8 slabs lazily and defer compaction during delta-anchor restore | `12712dd` | Keeps very-long disk restore from turning slab reconstruction into a large eager latency/memory spike. |

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
- [x] Mirror osaurus-side wiring: pure repin, no loader/settings surface needed (all ports default-on with env opt-outs). Pin lives in FOUR files (OsaurusCore Package.swift + OsaurusCore/xcworkspace/App Package.resolved) + TWO tripwire tests (RuntimePolicySourceTests, ImageGenerationBridgeContractTests) — all updated in /Users/eric/osaurus-main-live working tree.
- [x] Add 128-token RunBench window telemetry. Baseline at 2K prompt/1025 generated exposed a 22.6 -> 17.4 tok/s late-output collapse (20.6 aggregate).
- [x] Port 64 MiB BF16 pool hot tier and compact q8 storage into <=16K-row slabs. Release product built before the later HC work; focused test execution is blocked by the unrelated existing `MLXPressPolicyTests` `no such module 'Testing'` failure.
- [x] Defer single-token compressor/indexer advancement to compression-window boundaries. A/B: 20.6 -> 22.3 tok/s aggregate and 17.4 -> 20.0 tok/s final window; still below acceptance.
- [x] Fuse decode HC-post into a Metal kernel. Short 257-token row: 23.8 and 23.4 tok/s windows, 23.6 aggregate; still below acceptance.
- [x] **FALSE VERDICT CORRECTED (2026-08-08):** the earlier "shared compiled HC-pre region REJECTED" result was wrong twice over. (a) Without `VMLX_ENABLE_UNSAFE_COMPILE=1` the Swift `CompiledFunction.call` wrapper re-read `ProcessInfo.environment` (full dict rebuild) and rebuilt the closure trampoline on EVERY call — the A/B measured the *disabled* path plus that tax, not the compile. (b) Benching `&&`-chained after an all-core build reads ~50% slow from thermal throttle. Fixes: cached policy check + cached backend closures in `Transforms+Compile.swift`; thermal-gated bench protocol (each leg waits for `NSProcessInfo.thermalState == 0`). The compiled HC-pre region is back in (`DeepseekV4Math.hcPreCompiled`) and the planned `deepseek_v4_hc_pre_decode` Metal kernel is superseded by it.
- [x] Port the fine-grained Python decode regions (never the regressing whole-model compile): per-layer compiled MoE region (`_decode_moe_region` parity, warm-gated so SwitchGLU's one-time fused-gate-up eval stays out of the trace, indices returned as outputs so advisor observe runs untraced), decoder tail region (`_decode_tail_region`: hc_post(attn)+hc_pre(ffn)+post-LN+MoE+hc_post(ffn)), attention pre region (3 qmm+norms+RoPE+KV QAT, cos/sin ride as array inputs), attention out region (inverse RoPE + grouped o-proj + woB). Nested-compile guard via `CompiledDecodeTrace.withActive`; `[unowned self]` captures. Cool-machine RunBench diagnostics: eager 27.9 → +MoE 30.3 → +tail 30.6 → +attn pre/out 30.9 tok/s, parity signature [71, 439, 368, 393, 1230…] intact. *Diagnostic only — acceptance comes from the live harness.*
- [x] Trusted-compile entry: `vmlxTrustedCompile` (Source/MLX/Transforms+Compile.swift) compiles validated graphs even without `VMLX_ENABLE_UNSAFE_COMPILE`, so the DSV4 regions/microfns fire in production osaurus. `MLX_DISABLE_COMPILE` stays a hard kill switch (also flips the backend mode off once). The untrusted-disallowed path no longer calls `mlx_disable_compile()` per call — that global mutation would degrade trusted closures. All DSV4 compiled call sites (SwiGLU ×4, selector, hcPre, 4 regions) switched to it; generic `SwitchLayers` compiled fns stay policy-gated.
- [ ] LIVE harness proof — **the only accepted results method** (Eric, 2026-08-08): dev osaurus with vmlx-swift pinned, multiturn GUI, chat stat chips; must cover tools + coding, short/long context, short/long outputs, all ≥33 tok/s. RunBench and any CLI/API numbers are diagnosis only, never results.
- [ ] Prefill toward ~400 pp/s (Python 81bfec4: attention-only 512-token subchunks in 2048 outer chunks)
- [ ] One osaurus PR + vmlx-swift main push

Full continuation state, measurements, artifact paths, risks, and landing order: `DSV4-SPEED-HANDOFF-2026-08-08.md`.

## Proof rules
- Live osaurus GUI only (background postToPid toolkit; no curl/API/CLI for proof). **RunBench is never proof** — diagnosis only.
- Bench-vs-live: chat stat chips (TTFT s • tok/s • tokens) are the accepted live numbers.
- Thermal gate every measurement leg: wait `NSProcessInfo.thermalState == 0` (JXA probe) — builds AND benching itself heat the machine; back-to-back legs are invalid.
