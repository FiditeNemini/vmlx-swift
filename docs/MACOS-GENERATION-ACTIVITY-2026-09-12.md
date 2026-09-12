# Scoped macOS generation activity — PARTIAL

Runtime change commit: `3bcafad49d880ea21958f9a90322b4102beda84d`.

## Observed failure

A hidden Release Osaurus app remained `PROC_FLAG_SUPPRESSED` during an active
API request. Same binary, same 8,339-token prompt, seed/sampler/cache boundary
and exact 891-token answer: unsuppressed 34.70–34.95 tok/s; suppressed 15.97.
Both runs used normal process nice0. The hidden row's one-second token windows
were 14/16/20 tok/s (minimum/median/maximum), longest token gap 96.59ms.
Read-only native process-flag samples remained suppressed throughout decode.
These are one local Flash-Next JANG_2L workload, not all-quant throughput claims.

Receipt handles: `q6-nice0-candidate-b-8k-hidden-0912`, matching
`q6-nice0-candidate-b-hidden-policy-0912.jsonl`, supervisor
`SWIFTTEST_Q6Nice0CandidateB0912__162644.log`. The earlier q6 kernel experiment
is NOT included in this branch. Its binary supplies the reproduction only;
the corrected app must be compared enabled/disabled using one new binary.

## Change

`GenerationActivity.swift` owns one scoped ProcessInfo activity. macOS uses
`userInitiatedAllowingIdleSystemSleep`: no display or idle-system sleep lock,
no model-name/quant gate, and no permanent activity for an idle loaded model.
Other platforms retain their existing behavior.

- `Evaluate.swift::generateLoopTask` covers deferred iterator construction,
  prefill, AR/MTP decode, cancellation/error exits and cache/GPU finalization.
- `BatchEngine.swift::ensureLoopRunning` covers the non-solo scheduling loop.
- Each scope ends explicitly with `defer`; destruction is an idempotent fallback.
- `VMLX_DISABLE_GENERATION_ACTIVITY=1` is a diagnostic-only process-start opt-out
  for exact-binary A/B. It is not a recommended user setting.
- Debug logs in `vmlx/GenerationActivity` record acquisition, release or opt-out.

No sampler, tensor dtype, quant dispatch, GPU fence, cache representation,
MTP depth policy or inference worker thread is changed.

## Verification boundary

`GenerationActivityTests` ran six tests: normal completion including the tail,
throwing preparation, cancellation, overlapping scopes, idempotent/destruction
release, diagnostic opt-out and exact sleep-permitting options. Receipt
`SWIFTTEST_GenerationActivityLifecycle0912__163812.log`, exit0, peak8.12GB,
cleanup group/tracked/watchdog0/0/0. The warm test checkout contains other
pre-existing changes; these tests exercise the identical new Foundation helper,
not full model precision or throughput. Package.swift explicitly registers them.

Pending: exact isolated-source Release app, same-binary hidden enabled/disabled
and idle-return tests, real UI/API cancellation/continuation, batch/MTP routes,
short/long contexts and remaining quant/media coverage. No merge-readiness or
family-wide speed guarantee is claimed from the unit tests.

Apple's [app-level activity guidance](https://developer.apple.com/library/archive/documentation/Performance/Conceptual/power_efficiency_guidelines_osx/PrioritizeWorkAtTheAppLevel.html)
describes scoped user-initiated activities; its
[App Nap guide](https://developer.apple.com/library/archive/documentation/Performance/Conceptual/power_efficiency_guidelines_osx/AppNap.html)
describes priority/I/O reduction and foreground recovery.
