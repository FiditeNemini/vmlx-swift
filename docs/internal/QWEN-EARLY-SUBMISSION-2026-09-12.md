# Qwen4Exp early AR submission checkpoint — PARTIAL

## Current app evidence and renewed acceptance bar

The isolated optimized local dev app at app `61e8e6d0`, engine `fe8b231d`,
core `73312d3e` has now executed the guarded path. Binary SHA256
`3b5c9f2ab86494b65f13f61e9768187931d70d9b2971b54829676ea18fc51638`,
UUID `7BF7323C-373F-3203-9CEA-280A624F0F72`; no DEBUG test hooks.
This is not a published release or an installed-app replacement.

Two measured app API rows per arm, following excluded warmups, retained the
same 891-token complete count, natural stop, request bytes, bundle sampler,
seed 829 and MTP-off route:

| Actual input tokens | Early off mean tok/s | Early on mean tok/s | Change |
|---|---:|---:|---:|
| 34 | 36.3948 | 45.3106 | +24.497% |
| 8339 | 32.8596 | 39.0530 | +18.848% |

On rolling one-second windows were 42–47 short and 36–41 at 8K. Reverse off
controls were 34.5004/30.0555; they are retained, not used to inflate the gain.
These app rows supersede the older runner-only timing for app performance,
not its source attribution. **The user's above-45 tok/s bar at both short and
long context is NOT met.** Repeated long rows restore a prefix, so their small
prepare durations are not full-prefill throughput. Separate nonce-prefill
rows are retained but are not an exact-wire comparison.

The visible first answer completed coherently at 37.6832 tok/s, but a raw-free
supervisor abort occurred before the GUI follow-up. Subsequent runs use the
existing kernel-free metric with unchanged pressure/swap/footprint bounds.
All API timing rows preceded that abort; it is not silently counted as a
successful whole-app session. Connected image rows had identical off/on
outputs including wrong OCR/science; they establish neither vision quality
nor a fresh vision-encoder pass.

Private receipts: `early-app-short-comparison-0912.json`,
`early-app-8k-comparison-0912.json`, reverse-control comparisons, the app build
log `QwenEarlySubmitAppRelease0912__182608`, and per-token stream traces.

## Additional shared-AR candidates, not yet app-qualified

A live native sample of the same app during 8K AR decode captured 391/6002
observations rebuilding the entire Foundation environment dictionary in two
GDN policy gates. `RuntimeEnvironment.value` now copies only the requested C
environment value and preserves current/legacy precedence, explicit snapshot
lookups and dynamic changes. Inclusive stack observations are not additive
GPU timings or a predicted speedup.

`Qwen4ExpQSA` retains score math and argPartition, specializing only the B1/S1
fully-causal boolean membership/tail mask. Future-key, prefill and batch paths
remain generic; `VMLX_QSA_DECODE_MASK=0` provides a same-binary control.

`Qwen4ExpHCCombine` is opt-in via `VMLX_QWEN4_EXACT_HC_COMBINE=1`. It derives
stream count, hidden width and dtype from actual tensors, preserves the
low-precision multiply rounding before addition, and disables FP contraction
and reassociation. No weight quant, norm, sigmoid or recurrent-state math is
replaced. Non-single-row and compile-trace calls retain the original graph.

The first QSA shader test failed compilation on an invalid scalar/metadata
interface; the failure is retained and the interface corrected. Combined
generated regression `QwenHostQSAHC0912b__193404` exited 0: 27 Swift Testing
cases plus the native-governor XCTest, zero failures, peak 4.09 GiB, unchanged
0.49 GiB swap and cleanup 0/0/0. QSA tests cover F16/BF16/F32, ratios 1/4/32,
ties, threshold/tail crossings through 32769 keys, batch and future-key fallback.
HC tests compare exact bit patterns for 54 shape/dtype/stride configurations
and three rounding counterexamples that distinguish multiplication-then-add
from FMA. Connected mixed-format cache and 240-token governor tests also ran
with both candidates enabled. The test semaphore recovered the previous
crashed runner's abandoned lock using its existing 90-second timeout.

In synchronized generated-mask diagnostics, the 8339-key full-mask call
averaged 0.5060 ms generic versus 0.3789 ms specialized; at32769 keys,
0.6660 versus 0.4857 ms. The environment diagnostic measured approximately
18.87 microseconds per snapshot lookup versus 0.317 microseconds direct.
These are optimized unit-run diagnostics, not app throughput predictions or
isolated GPU timings. The new app-performance, default-adoption and merge
gates remain pending.

## Scope

The opt-in `VMLX_QWEN4_EXP_EARLY_SUBMIT=1` submits each completed trunk
layer with `asyncEval(hidden)` on the caller's stream. The CPU can assemble
later layers while previously submitted GPU work executes. No new stream,
worker, arithmetic, quantization, sampler or cache format is introduced.

The implementation is model-family based, not a JANG filename or bit-depth
special case. Existing per-projection bit/group/dtype dispatch remains intact.
This does not enable the rejected outer compiled-decode path.

Eligibility requires B=1/S=1 and an explicit AR call site. Prefill (including
a one-token prepare tail), native seed/re-entry, prefix capture, external-PLE
compiled forward, and compile tracing do not use this specialization.
The optional `NativeMTPAutoregressiveBackboneModel` capability lets only
`NativeMTPTokenIterator.generateAutoregressiveToken` use the same scheduling
as ordinary AR. Existing models without that capability retain their forward.
Hidden states remain the pre-mixer trunk states required by native MTP.

Absent, invalid and non-1 settings remain disabled. Four-layer grouping was
experimentally slower at longer context and is not an accepted setting.

## Retained diagnostic evidence, before the call-site guards

Local M5 Max / 128 GiB, Release `-O -whole-module-optimization`, engine base
`6be76cc24be917cee811da60a7250043aa289ea7`, core
`73312d3e9ad0bd2e1bdf9a08d91e25571e255964`.
Runner SHA256 `68d088db16da1c1455f1818e7c75052a534d1f43ac34024df9c4e1b3484cf73d`;
diagnostic Qwen4Exp.swift SHA256
`543deebdf6bcb5044d070224f9dbfd7b8a807debd768867b092ff2ea5e1eac23`.

Actual Qwen3.8-Flash-Next-JANG_2L; bundle temperature 1/top-p 0.95/top-k 20/min-p 0,
explicit seed 829 and thinking-off. Both contexts produce the same complete
891-token count from 1 through 250, ending naturally at stop token 248046.
Two measured runs per arm follow one separately excluded warmup. These are
iterator first-to-last delivery timings, not UI refresh rates or GPU occupancy.

| Actual input tokens | Off median tok/s | Every layer median tok/s | Reverse off median tok/s | Four-layer diagnostic |
|---|---:|---:|---:|---:|
| 34 | 35.8496 | 39.7670 | 35.4182 | 40.8086 |
| 8339 | 32.2516 | 32.0621 | 31.4960 | 29.4297 |

Every-layer observed short-context improvement is 10.93%; longer-context
throughput is approximately neutral against these controls. Four-layer
grouping is rejected for its longer-context loss, despite a better short row.
Grouped measurements are not randomized thermal controls or a throughput floor.

Short every-layer 1-second rolling windows range 35–44 tok/s, 5-second windows
37.4–42.4; longer-context windows 26–37 and 31.2–34.0 respectively. Longest
iterator gaps are 40.87 ms and 44.58 ms. Complete token timestamps, outputs,
requests, sampler receipts, binary hashes and comparisons are retained in
private `early-short-abba-0912.json` and `early-8k-scheduling-0912.json` evidence.
The former's actual arm order is off/every-layer/four-layer/off, not strict ABBA.

Physical footprint peaks were 48.09 GiB short and 50.09 GiB longer-context; swap
remained 0.49 GiB and each owned process group/watchdog exited. No model files
or prefix checkpoints were deleted. Initial missing-metallib launch and the
first wrapper's stop-string assertion failure are retained as failed harness
rows; neither is silently relabeled as a successful invocation.

The measured host trace motivating this change has 23 complete token intervals,
27.557 ms average wall time and 8.562 ms mean largest gap between completed GPU
buffer spans, predominantly before the next host commit. Buffer-span union is
not useful GPU occupancy and does not attribute all remaining cost to the CPU.
Python/oMLX per-layer submission is a source comparison, not Swift proof. The
separate Python 6S 37.2713→45.8489 result holds its HC/QSA/PLE composition fixed;
its different bundle, sampling and runtime cannot supply Swift's missing rows.

## Revised-source generated tests

`QwenEarlySubmissionDisk0912__182055` executed three Swift Testing cases and
the XCTest governor handoff case, with zero failures. The mixed-format case
covered dense F32 and routed 2/3/4-bit group 32, 4/4/4-bit group 64, and
6/4/6-bit group 64. Off/on logits, dtypes, full native cache state and offsets
matched exactly across connected prefill/decode, the fixture QSA boundary,
and a safetensors round-trip through `TQDiskSerializer`/`restoreFromDiskArrays`.
This tiny matrix does not exercise every installed fused-kernel geometry.

Executed counters distinguished ordinary/explicit AR from native seed and
one-token prepare. The real governor iterator retained all 240 expected token
IDs and cache contents, and its specialized call count equaled its AR-fallback
count. Fixture throughput is not an installed-model performance result.

The initial raw-state restore in the new test was invalid: it omitted Mamba
offsets and did not provide independent disk-loaded storage. Its 16 failures
occurred with submission off and on and are retained; the test now exercises
the actual production serialization/restore functions, with no tolerance change.

Tests used an optimized unit-only binary with DEBUG hooks for an unrelated
existing evaluation-lock test. Initial compile and missing-Metal-library
failures are retained. The final runner used the colocated Metal library from
the same core revision (SHA256 `24d4cfcd3ca8b15ead691e46219f35adabbea64c9f8de4eae9bf293fd8d5eb7b`).
Peak tracked footprint was 3.96 GiB, swap remained 0.49 GiB, cleanup 0/0/0.

## Remaining gates

- Qualify the additional candidates, then rebuild the optimized local dev app
  without DEBUG test hooks; repeat short and long app/API rows with exact pins,
  executable UUID/hash, actual dispatch markers and rolling stream rates.
- Connected app history, real image/video and explicit MTP fallback/re-entry
  must retain their original output/cache contracts.
- Longer contexts, other installed quants, cancellation and the default-on
  decision remain unqualified. This is not a 40–50 tok/s family-wide claim.
- No new PR/merge or user-facing default enablement is claimed by this document.

The first Release test invocation failed before these tests because an existing
MLXTests test references a DEBUG-only evaluation-lock hook. A subsequent
optimized unit-only invocation enables that hook; it is not a performance
binary and must not be used as production Release evidence.
