# Qwen3.8 Flash-Next staged-MTP GDN-front parity

Status: **PARTIAL** until bundle-driven Auto, served API, and Osaurus UI proof
are complete. The engine-level fixed-depth defect is reproduced and reversed
on the local `JANGQ-AI/Qwen3.8-Flash-Next-JANG_2L` bundle.

## Defect

Autoregressive width-1 decode entered `Qwen4ExpCompiledGDNInputs.callFront`,
while the width-4 native-MTP verifier could not enter that function because it
accepts only `input.dim(1) == 1`. The verifier instead used the native BF16
affine plus ordinary causal-convolution path. Those two paths were not
row-equivalent: the first GDN recurrent-state difference appeared in layer 0,
then propagated through PLE and later layers. This reduced draft acceptance
and could change greedy output relative to AR.

The staged cache commit itself was exonerated. Replaying the staged `q/k/v/a/b`
rows as one accepted prefix or as chained width-1 updates produced exact state
equality in all 36 GDN layers. Starting from the same pre-verify checkpoint,
however, a true sequential model replay differed from the staged cache at
layer 0 recurrent slot 1 (`maxdiff=4.172e-07`), proving the mismatch happened
before cache commit.

Diagnostic evidence:

- `vmlx-private-evidence/qwen38-staged-parity-20260830/267337e6/staged-commit-audit/d3-measurement.log`
- `vmlx-private-evidence/qwen38-staged-parity-20260830/267337e6/staged-commit-audit/cache-audit.log`

## Fix

The non-equivalent generic compiled GDN front is no longer a production
default. It remains available only through the explicit diagnostic environment
variable `VMLINUX_QWEN4_EXP_COMPILE_GDN_FRONT=1`. Default AR and staged-MTP
verification now share the native BF16 affine/conv path. PLE remains SSD-backed
through `pread` plus `F_NOCACHE`; this change does not make PLE or n-gram state
resident.

## Matched engine proof

Release `RunBench`, greedy sampling, identical prompt and output budget, no
JangPress, no KV-cache substitution. Prompt: count from 1 through 120 and emit
only the comma-separated sequence. Every arm produced the exact expected
490-byte sequence and SHA-256
`37524b0ce6e14373f1bb0b5f0a9f3bc637153beec13eb80bba74bb50d1b86fda`.

| Arm | tok/s | Peak physical footprint | MTP result |
| --- | ---: | ---: | --- |
| AR | 37.4 | 48,700 MiB | control |
| D1 | 33.3 | 51,101 MiB | 490/490 verified inputs committed |
| D2 | 44.2 | 51,169 MiB | 489/489 verified inputs committed |
| D3 | 68.6 | 51,195 MiB | 492/492 verified inputs committed |

D3 is 1.83x AR and is the measured winner. All rows ended with `stop`, no
unclosed reasoning, no loop, and no protocol leakage.

A second code-generation row was also byte-identical between AR and D3:

| Arm | tok/s | Output SHA-256 |
| --- | ---: | --- |
| AR | 30.5 | `e5fe77ed8ed443d36f346e7234002ae89a8fce111f78004416a093bed8464ce5` |
| D3 | 62.6 | `e5fe77ed8ed443d36f346e7234002ae89a8fce111f78004416a093bed8464ce5` |

The D3 code row committed 120 output tokens with 8 rejected drafts handled by
normal verifier correction; it did not fall back to AR.

Production-patch evidence:

- `vmlx-private-evidence/qwen38-staged-parity-20260830/267337e6/production-patch/`

## Test state and remaining gates

- Release `RunBench` product build: PASS.
- Added a numeric regression asserting exact native BF16 affine output for a
  four-row verifier batch versus four width-1 calls.
- The `MLXLMTests` target, including that regression source, compiles in
  Release with `-enable-testing` (283.26 seconds).
- Local full SwiftPM test execution is currently BLOCKED by unrelated existing
  package failures: `MLXArray.withEvalLockForTesting` is absent in `MLXTests`,
  and the generated Xcode package build fails because
  `DistributedModelInventory/main.swift` combines `@main` with top-level code.
  These are recorded as infrastructure blockers, not test passes.
- Bundle-driven Auto must consume an updated workload-general measured sidecar
  and resolve D3 without `VMLX_MTP_TUNING_MEASUREMENT`.
- The exact repinned Osaurus Release app must prove native Chat and served API
  D3 telemetry, visible output completion, multi-turn tool continuation,
  cancellation recovery, and cache restore before the user-facing lane is
  complete.
