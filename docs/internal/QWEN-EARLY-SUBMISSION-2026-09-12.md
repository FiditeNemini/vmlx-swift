# Qwen4Exp early AR submission checkpoint — PARTIAL

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

- Rebuild Release without DEBUG test hooks; repeat affected live app/API rows
  with current engine pin, executable UUID/hash and actual submission marker.
- Connected app history, real image/video and explicit MTP fallback/re-entry
  must retain their original output/cache contracts.
- Longer contexts, other installed quants, cancellation and the default-on
  decision remain unqualified. This is not a 40–50 tok/s family-wide claim.
- No new PR/merge or user-facing default enablement is claimed by this document.

The first Release test invocation failed before these tests because an existing
MLXTests test references a DEBUG-only evaluation-lock hook. A subsequent
optimized unit-only invocation enables that hook; it is not a performance
binary and must not be used as production Release evidence.
