# Qwen AR composed producer work — PARTIAL

NOW: Port the missing producer/consumer work, retain qualified candidates,
and measure their composition in the local development app.
DO NOT: Claim 50 tok/s, discard a candidate from one noisy result, change
samplers, replace BF16 arithmetic with FP32, or begin sustained MTP.
BATCH OWNER: HC residual/next-norm fusion plus GDN Q/K normalization.
NEXT: Current-source numerical/cache tests, then counterbalanced app comparisons.

## Source comparison

Reviewed engine baseline `93baa9f2bd5299f0d9a36eacea9da0b5c3028a52`;
Python reference `1e022d8e6570bd487ae761a6e016f36e217502a4` in the active
terminal worktree, not the older wrapper checkout;
oMLX reference `7cbb407168ae628bbe0d7fe385be70e0954af303`;
antirez main `bd66c402070042bf0a79ad6ece8242de4c93680c` and Qwen PR991
reference refreshed to `ccea7688276a9da8fc1453b13dad3e5c1c044ce2` (still OPEN
in the live GitHub readback). The reviewed `metal/qwen4.metal` has no changes
from previous `30305544f96cc35357e0dd376eeff2b1ab3e365e`. New root-C changes
include checkpoint logits and steering; steering is not part of this port.

| Mechanism | Python / oMLX / C | Swift baseline and action |
| --- | --- | --- |
| Layer submission | Python `language.py:2256–2308`, oMLX `language.py:2905` submit each eligible layer. | `Qwen4ExpTextModel.forward` already submits AR each layer. Preserve it; fewer kernels can change the best submission grouping, so one old stride experiment is not a universal rejection. |
| HC producers | C Qwen `kernel_qwen4_hc_combine_norm` produces next residual and normalized input together; ds4 HC producer clusters preserve their original reduction trees. | The separate `combine` then `mix` constructs/reloads intermediates. New `Qwen4ExpHCCombineNorm` produces both outputs once and passes the norm directly to the next mixer. |
| HC arithmetic | oMLX fused HC takes precedence over its exact-hybrid path, with explicit quant/shape guards; its epilogues use different FP32 rounding. | Actual retained 2L HC weights were dense BF16, so oMLX's quantized-HC gate is not the missing switch here. Keep native projections and explicit BF16/F16 product, sum, norm, and scale rounding. |
| GDN front | C Qwen combines convolution, Q/K normalization and gate work; Python has grouped inputs and qualified front helpers. | Existing grouped projections and recurrent kernel remain. Restore the separately qualified Q/K norm component, then test with HC fusion; no claim that this is the entire C GDN front. |
| Projection grouping | Python prepares compatible QSA and shared gate/up projections. C concatenates disjoint output grids without replacing their reductions. | Previous private adapters actually executed, but whole-model results drifted over time. Keep grouping open for composed, repeated qualification; do not infer universal benefit from launch count or universal failure from those medians. |
| PLE reads and ownership | Python groups selected rows and owns prefetch tickets; C separates model shards, scratch and live state. | Swift uses persistent row readers and bounded prefetch, not whole-table reads per token. Prepared HC normalization must not cross a PLE residual update. No model-shard deletion or new row-cache policy. |
| QSA/cache | Derived pooling, raw-capacity append and sparse selection are distinct from attention dispatch and SSD prefix durability. | Keep existing capacity/shared-mask candidates separate in attribution, fixed in all new HC/GDN arms. Preserve native KV/GDN/PLE boundaries and media-position limitations. |

Primary sources: [oMLX host and HC work](https://github.com/jundot/omlx/pull/3469),
[C exact HC producer grouping](https://github.com/antirez/ds4/commit/92d83e6ecd4d2bbd07bfb947db9bd4afe044edc1),
[Qwen C/Metal work](https://github.com/antirez/ds4/pull/991).
Their measured speeds do not transfer to this build or these bundles.

## Implementation contract

- New switches are opt-in: `VMLX_QWEN4_HC_COMBINE_NORM=1` and
  `VMLX_QWEN4_GDN_QK_NORM=1`. Do not infer execution from their names:
  tests and runtime call counters must show the admitted paths.
- Admission is actual single-sequence/single-row explicit AR, including the
  scheduler's explicit AR fallback; not speculative seed/draft/verification,
  re-entry, prefill, external PLE embeddings, or outer compiled tracing.
- The HC target mixer owns the normalization weights. Both outputs are new
  arrays; prepared normalization is a forward-local value, not a retained
  token cache. Static compiled functions capture only mode/geometry, not the
  first generation's tensors.
- Intra-layer attention-combine/MLP-norm and inter-layer MLP-combine/next-norm
  may share a dispatch. A next layer with PLE does not receive a prepared
  attention norm, because PLE changes the residual first.
- Preserve current RMS four-values-per-thread and two-stage SIMD reduction,
  precise rsqrt, and each low-precision rounding. Keep original projections,
  quant layouts, sampling and persistent-state serialization.

## Correcting earlier experimental conclusions

Retained GDN standalone ABBA results had short medians 47.6923→49.5216 tok/s
on 2L and 41.7923→42.4483 on 4S; long medians 43.3984→43.5153 and
38.6614→38.6152. These are earlier standalone results, not current app proof.
They support retaining the candidate, not removing it because long-context
improvement was small. The tiny negative long 4S difference is inconclusive.

QSA grouping medians 42.3728/42.2291 and shared grouping 41.7543/40.8493 had
temporal drift and private-adapter overhead. Status: inconclusive for product
composition, not permanently rejected. Actual numerical failures remain safety
failures for the tested geometry; changing a kernel or narrowing eligibility
requires a fresh proof, not erasure of the failed result.

## Required comparison

One exact dev binary, four configurations: baseline, GDN only, HC only, both.
Use counterbalanced repeats with identical tokenized input, bundle, explicit
fixture seed, sampler, cache boundary and output allowance. Separate cold
prefill, restored-prefix decode and visible multiturn output. No profiling
sample during throughput rows. Record first/min/peak fixed-window token rates,
maximum token gap, wall decode, full output/stop, physical footprint, swap,
free memory, other heavy processes and available power/thermal telemetry.

Require short and long context and more than one order before attributing a
small change. Keep failed, interrupted and interference-confounded rows, but
do not mix them into a clean causal median. Qualification covers 1L, 2L, 4S,
4M and 6S; 1L has no native MTP tensors in the retained bundle inspection.
MTP and multimodal runtime proof are not provided by these AR text tests.

## Current execution receipt — 13:20 PDT, PARTIAL

The source changes are uncommitted on HEAD93baa. The four-way connected test
now covers PLE in layer 1, layer 2 and both layers, each across three mixed-quant
layouts, plus disk save/reopen continuation and explicit AR routing. The unit
matrices contain 384 HC and 336 GDN cases; these counts are intended coverage,
NOT executed passes.

At 13:01 the supervisor refused because a separate build was active. The
13:12:41 retry started at 63.6 GiB free / normal pressure / 0.49 GiB swap. It
compiled the engine modules, but at 13:19:11 the memory guard aborted at
29.8 GiB free (<30 GiB). Exit124; peak owned footprint4.77 GiB; swap unchanged;
cleanup confirmed zero owned group/tracked/watchdog processes at13:19:15.
No tests executed. This is neither a numerical failure nor a numerical pass.

Receipt: `../mtp-swift-2026-09-04/logs/SWIFTTEST_HCAndGDNComposition0913__131241.log`
under `/Users/eric/vmlx-private-evidence`, with `.mem` and `.procs` sidecars.
Another build started after our preflight: PID93544, start13:13:16, cwd confirmed
by lsof as `/Users/eric/osaurus-resident-child-ram`; its Swift frontend3563 was
still active at13:20. It was not signalled. A clear-machine window is required
before retrying; do not lower the memory guard to force concurrent runs.

Current source SHA256:

- Qwen4Exp.swift: `c2c4a330400a17678a67fecd1641b4935e9d6c7820f328dd2464c4c210adce5b`
- Qwen35.swift: `fa565d0fbe2d67ee1508d410a1a1973f631aa80f042e791eef72345928c8cce0`
- Qwen4ExpHCCombineNorm.swift: `f576c9204eecca77de38722d4e30d1f7465761ee9c7b296db6d7582a93d78b75`
- Qwen4ExpGDNQKNorm.swift: `39b4e2f0d203be0303b945f6e542821f9ade49df0b47c17c0e14f737742f94a5`

No app pin/build, PR commit, merge, release, or speed qualification occurred.
The existing dev binary does not contain these new changes. Both candidates
remain opt-in. Older standalone results cannot satisfy this gate.

## Current-source test receipt — 14:52 PDT, FAILED

Retry `SWIFTTEST_HCAndGDNCompositionRetry0913__144554.log` completed the
build in 361.81 seconds and ran seven Swift Testing tests across three suites.
It reported 16 issues, exit1; peak owned footprint4.60GiB, swap0.49GiB
unchanged, cleanup0/0/0. This is not an app or throughput result.

- All336 GDN arithmetic cases matched; unsupported-input fallbacks passed.
- HC matched383/384 cases. One FP16 case (streams4/width96/stride2/magnitude8)
  had one normalized-value mismatch while the residual remained exact.
- Connected tests observed only two HC calls where four were required;
  the mixed-quant MoE's FP32 block route was outside the helper's admission.
  Logits and disk/cache equality did not fail in the completed PLE1 cases.
- PLE2/both fixtures threw missing shard errors; the fixture only wrote
  layer0 shards. Those rows did not reach numerical comparison.

The follow-up preserves the original promoted FP32 block-product/add then
BF16 residual rounding, mirrors MLX's runtime RMS full/tail reduction branch,
and expands arithmetic coverage to768 cases. The connected fixture now has
two linear layers plus QSA, per-PLE-layer shards and distinct HC norm weights.
No expectation is weakened. Execution of these corrections is still required.

## Loaded-library RMS arithmetic — 15:17 PDT

The promoted-block retry150119 passed all connected mixed-quant, PLE and
native disk-continuation tests, with actual BF16-residual/FP32-block admissions.
It still failed the same one FP16 unit case. Disabled contraction, explicit
FMA, volatile intermediates and precise division did not remove that mismatch.
These failed diagnostics remain retained; they are not speed measurements.

The app build receipt113630 line2372 compiles `rms_norm.metal` with
`-fmetal-math-mode=fast -fmetal-math-fp32-functions=fast`. The runtime custom
kernel builder in `device.cpp` calls `setFastMathEnabled(false)`. Changing only
the custom mean from division to reciprocal-multiply matched all768 cases,
including a FP32 intermediate comparison that had102 mismatching cases before.
This is a measured arithmetic discrepancy, not the cause of the AR slowdown.

The helper now compares four small generated FP32 rows with the loaded RMS
primitive once per process. It selects division or reciprocal-multiply only
when exactly one matches; otherwise it declines fusion. It does not infer
the library's math mode from the app name, a config flag or a quant name.
No per-token qualification, persistent-state mutation or global math-mode
change was added.

- `SWIFTTEST_HCRMSLoadedLibraryB0913__151601.log`: same production helper,
  app library5313, selected reciprocal;768 cases/0failures, exit0.
- `SWIFTTEST_HCRMSStrictLibrary0913__151708.log`: identical executable
  c0a70d66, isolated librarye46116c4 rebuilt with only RMS strict; selected
  division;768 cases/0failures, exit0. App/baseline libraries were untouched.
- Both runs had flat0.49GiB swap and cleanup0/0/0. The first compile attempt
  151517 failed Swift expression type checking before execution; retained.

Current helper SHA256 after formatting:
`1788e10cd1ca7c0e9c30d18b807454a0c2539797a1e57dab48cd2713705e91db`.
Full generated/connected regression retry151752 is running; no new app binary
or app throughput proof yet. Both production switches remain off by default.

## Generated and connected regression checkpoint — 15:22 PDT

`SWIFTTEST_HCAndGDNLoadedRMS0913__151752.log`: optimized current source built
and Swift Testing reported14 tests/3suites, exit0. HC768 and GDN336 arithmetic
cases matched, including mixed FP32 blocks. Connected tests matched exact
logits and native GDN/QSA/PLE state across three routed-quant layouts, all
three PLE placements, four independent/composed feature arms, distinct mixer
weights and disk reopen. Explicit native AR dispatch counts matched; prepare,
seed and verify did not enter the new helpers. Peak3.97GiB, swap0.49flat,
cleanup0/0/0. Strict-fixture-only tests were skipped in this first run.

`SWIFTTEST_HCAndGDNStrictFixtures0913__152216.log`: same built test binary with
the explicit fixture setting `MLX_ENABLE_TF32=0`;14 tests/3suites, exit0.
This also exercised both token-cap publication cases, all three prefill chunk
cases, actual default-prefill progress and cancellation. Peak0.34GiB,
swap0.49flat, cleanup0/0/0. Only the optional isolated timing test was skipped.
No production sampler or precision default was changed by this fixture run.

This checkpoint qualifies the bounded numerical/state tests, not installed
model performance or media paths. The following app build must consume this
exact source; all-quant short/long UI and repeated throughput rows remain open.
