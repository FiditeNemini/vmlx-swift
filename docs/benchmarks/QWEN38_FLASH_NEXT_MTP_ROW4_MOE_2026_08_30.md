# Qwen3.8 Flash-Next native-MTP row-4 MoE proof (2026-08-30)

## Scope and verdict

Runtime source head: `c1162b43a6144349e630fb13894d495ce74faa7e`

This change is a **PASS** for its bounded claim: native-MTP verification rows
2 through 4 now use the same fused routed-MoE arithmetic as ordinary row-1
decode, and fixed depth 3 exceeds 60 tok/s on the exact count workload.

This is **not** proof that native MTP should be forced on globally. A separate,
pre-existing whole-model staged-verifier parity failure remains reproducible on
rejection-heavy prose. This change does not enable MTP, change Auto policy, or
change generation parameters.

## Source trace

Before this change, `Qwen4ExpFusedAffineMoE` accepted exactly one hidden row.
A depth-3 verifier therefore fell back to generic routed MoE for its four-row
target forward. The patch:

- supports only rows 1 through 4 (decode plus the current maximum native-MTP
  verification depth), leaving larger prefill batches on the generic path;
- indexes input, route, score, activation, and output tensors by row;
- materializes only non-contiguous input views required by the flat Metal
  indexing contract;
- keeps q2/q3/q4/q5/q6 and group-size 32/64 support bounded to the two audited
  production shapes.

## Focused tests

Command:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  swift test --filter Qwen4ExpFusedAffineMoETests
```

Result on `c1162b43`: 6/6 passed.

The suite covers every shipped bit/group layout at rows 2, 3, and 4 against
independent row-1 execution, production-shaped non-contiguous views, the
rows-above-4 fallback, Ornith's separate 2048x512 top-8 contract, and invalid
metadata rejection. Every multi-row-versus-row-1 relative error printed `0.0`.

## Live Release matrix

Bundle: `JANGQ-AI/Qwen3.8-Flash-Next-JANG_2L`

Prompt: `Count from 1 to 200.`

Settings: greedy sampling, 1,024-token cap, fixed native-MTP depth, adaptive
depth disabled for measurement, mmap weights, no JangPress. The 22-token
prompt is below the 2,048-token QSA budget, so sparse-QSA changes are not part
of this measurement.

| Runtime | Depth | tok/s | verifier time | result |
|---|---:|---:|---:|---|
| before control (`c1cef3bb`, row-4 generic MoE) | 3 | 33.8 | 22.189 s | exact 1-200, `stop` |
| `c1162b43`, cold single row | 3 | 68.2 | 9.434 s | exact 1-200, `stop` |
| `c1162b43`, repeat 1 | 3 | 69.4 | 9.438 s | exact 1-200, `stop` |
| `c1162b43`, repeat 2 | 3 | 65.8 | 10.009 s | exact 1-200, `stop` |
| `c1162b43`, repeat 3 | 3 | 62.2 | 10.706 s | exact 1-200, `stop` |
| `c1162b43` | 2 | 54.1 | 12.914 s | exact 1-200, `stop` |
| `c1162b43` | 1 | 43.7 | 16.941 s | exact 1-200, `stop` |

Depth-3 repeat median: **65.8 tok/s**. All three measured outputs were the
same 890-character sequence and byte-matched the expected comma-separated
integers 1 through 200. Peak physical footprint was 51,376 MiB.

Additional greedy rows on `c1162b43`:

| Workload | AR tok/s | D3 tok/s | visible result |
|---|---:|---:|---|
| 250-word tide explanation | 40.1 | 43.3 | coherent, `stop`, no marker leak |
| Swift binary search | 39.8 | 57.1 | coherent, `stop`, no marker leak |

## Remaining blocker: whole-model greedy parity

AR prose repeats byte-identically, and explicit sequential MTP verification
matches those AR bytes. Batched staged D3 is deterministic but differs from
AR. Token-level traces locate the first wrong target decision at generated
token 231: from the same prefix, staged row-0 selects token `67550` while the
sequential target selects `10539`. This occurs before staged cache commit, so
it is not a rollback leak. Other mixed quantized/dense projections still use
different multi-row and row-1 kernels and can perturb a near-tied argmax.

Therefore:

- do not use this PR to force D3 or change Auto defaults;
- preserve the existing fail-closed tuning requirement;
- require a separate whole-model verifier-parity fix and matched AR/D3 proof
  before making a global policy change.

## Local raw evidence

Raw logs, binary hashes, source heads, memory snapshots, and full decoded text
are preserved under:

`/Users/eric/vmlx-private-evidence/qwen38-count-20260830/c1162b43/`

