# Gathered Metal matmul validation

The core pin backports upstream MLX #4567 and extends expert-boundary row
scheduling to sorted affine gathered matmul. It also corrects a pre-existing
NAX partial-reduction bounds error found by the expanded regression matrix.
The Swift integration includes regenerated JIT sources and the new offsets
shader in the explicit AOT resource list.

## Scope

- Floating-point sorted gather uses the upstream scheduler.
- Affine sorted gather uses the scheduler when the existing dispatch is eligible.
- Small decode, unsorted gather, explicit LHS gather, and custom quantization
  kernels retain their dispatch. This is primarily a prefill optimization.
- TF32-disabled float tests exercise the non-NAX implementation on an M5;
  they do not establish physical M3/M4 performance.

## Proof checklist

- [x] Numerical comparisons against a CPU reference for expert boundaries,
  empty experts, partial tiles, affine bit widths 1/2/3/4/5/6/8, group sizes
  32/64/128, transpose, sorted/unsorted routing, and explicit LHS indices.
- [x] Reproduce the partial-K error on the baseline and pass after correction.
- [x] Check ordinary affine matmul because it shares the corrected NAX loader.
- [ ] Rerun numerical and scheduler/cache suites at the final engine commit.
- [ ] Stable paired full-model prefill/decode measurements. Existing timing
  controls show host variability; no final full-model speed claim yet.
- [ ] Real concurrent requests with bundle defaults, natural stops, throughput,
  physical footprint, and independent outputs.
- [ ] Image/video, repeated-media cache reuse, changed-media invalidation, and
  architecture companion-state restoration.
- [ ] Pinned Release app: GUI multi-turn, applicable settings, cache telemetry.
- [ ] Exact-head CI and dependent app proof before merging.

`GatherRowTileTests` includes 396 numerical cases and an opt-in synthetic
projection benchmark (`VMLX_GATHER_ROW_BENCH=1`). Benchmark numbers describe
one projection with synthetic weights, not an application speedup.

The concurrent RunBench diagnostic supports
`BENCH_BATCH_USE_GENERATION_CONFIG=1` and `BENCH_BATCH_SEED`. It records final
completion telemetry and rejects length-stopped or looping rows. Its raw
stream output is separate from parser/UI proof.

Subagent settings and residency orchestration are a separate follow-up. They
must prove same-model batching when enabled, default-ON smart swapping,
explicit-OFF coexistence, parent restoration, and cancellation through the
actual app controls before any orchestration fix is promoted.
