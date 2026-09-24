# Spark 2.5 / Raptor activation fusion checkpoint

No release or tag. Merge only a measured, qualified improvement. This PR is not ready to merge.

## Contract

The candidate fuses erf GELU and its following multiply, preserving BF16 rounding at each intermediate. Packed weights, projection quantization, residual precision, sampling, templates and cache policy are unchanged. CPU/non-Metal, other dtypes, broadcast/empty inputs and traced transformations retain the reference expression. `VMLX_SPARK_GELU_REFERENCE=1` is a diagnostic reference-path switch read before the first forward.

Current revision restricts fusion to at least 128 rows in the sequence dimension; shorter/decode inputs use the reference expression. f818a4ea below is the earlier unrestricted experiment. The tracer query is metadata-only. Eight isolated helper tests pass, including all 65,536 BF16 encodings with four multipliers (NaN payloads excluded), noncontiguous inputs, CPU/scoped streams, compiled/VJP/JVP/vmap paths and empty arrays. Fresh Release RunBench built successfully from a clean tree; source hashes and binary are archived.

## Evidence and rejected results

Evidence root: `/Users/eric/vmlx-private-evidence/raptor06-speed-2026-09-23/spark-mlp-r30`.

- The original CustomFunction wrapper (2420da85) slowed native parsed generation to 43.9 tok/s versus baseline median 99.3. It is rejected.
- The direct tracer-aware replacement has exact serial synthetic MLP outputs: reference/candidate median 4.995/4.570 ms for a decode-shaped 36-layer chain, 2.806/2.638 ms for M128 and 7.794/7.254 ms for M512. A separate GPU stream reproduced the synthetic benefit. These are not whole-model speed measurements.
- The replacement still slowed a complete 7,032-token parsed hash-table response to 72.3 tok/s. The same binary with reference activation completed at 94.0 tok/s. Both stopped natively with closed reasoning and matching visible answer. This single pair was initially attributed to fusion, but the later reference-only shape also measured 75.1 tok/s. Causality is unresolved; do not promote the unrestricted default decode path.
- One 10,031-token prefill/cache pair measured 2.458 s reference and 2.341 s fused. Both two-turn raw outputs and all 685 tensors across five retained checkpoints matched exactly. Quota eviction and resume occurred. One pair does not establish a statistical speed improvement. Physical footprint reached 3.86/4.32 GB; this does not establish a low-RAM family claim.
- The unchanged baseline word-count workload looped to 8,192 tokens without a visible answer. Retain that failed coherence row; no sampling or prompt masking was added to hide it.

The repeated same-binary long-prefill ABBA ran reference 2.636/2.456 s and fused 2.436/2.348 s, with eight native-stop turns. The raw-submit pair ran 94.5 tok/s fused versus 90.3 reference; host-delivery median/p95/max were 10.584/10.964/14.340 ms fused and 11.076/11.456/14.054 ms reference. These narrow wins do not excuse the parsed-decode regression. The initial shipping candidate therefore keeps decode on the reference expression. The eight revised focused tests exercise eligible prefill shapes, 127/128/129-row boundaries, exhaustive BF16 values and transformations; all passed.

The prefill-only 8a254032 native parsed row completed correctly at 75.1 tok/s even though the 44-token prompt and decode shapes took the helper reference path. Spotlight workers were active after copying app build directories; those owned copies were renamed `.noindex`. This is a potential confound, not an established root cause. The next revision returns the original MLP expression directly for short/decode input before evaluating `up(x)`, preserving graph construction and temporary lifetimes. Eight focused tests pass; full runtime/app proof must use that final revision.

The final size guard uses `Int32.max`, matching `MLXFastKernel`'s signed grid conversion. The ninth focused test constructs a lazy broadcast with 2^31 elements and verifies reference fallback without evaluating or allocating the logical tensor. All nine focused tests pass. Earlier eight-test receipts remain historical.

## Remaining work, in order

- [ ] Complete same-binary raw-submit token-delivery comparison; report median/p95/max as host delivery, not GPU kernel latency.
- [ ] Repeat long-prefill/cache A/B with output and persisted-tensor equality.
- [ ] Resolve the parsed-decode regression, or narrow fusion to measured prefill shapes and re-prove the resulting dispatch.
- [ ] Re-run exact final-source tests and full runtime build.
- [ ] Prove native multi-turn output, cold/growing/restart SSD cache behavior and actual isolated Release Osaurus Chat/Settings interaction.
- [ ] Finish applicable app evals and required Osaurus CI; merge engine then app pin, without any release.

The private CURRENT.json and NEXT-ACTIONS.md hold active process handles and later receipts. Negative results remain part of qualification. CPU sampling is bounded; do not run broad Metal System Trace on this host.
