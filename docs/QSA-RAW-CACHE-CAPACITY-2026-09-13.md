# QSA raw-key capacity checkpoint — PARTIAL

NOW: Qualify stepped raw-key storage against the original per-update concatenation.
DO NOT: Claim a model speedup or merge readiness from isolated cache timings.
BATCH OWNER: `QSAKVCache` in `Libraries/MLXLMCommon/KVCache.swift`.
NEXT: Rebuild the isolated local development app; compare both paths on identical
2L/4S requests at short, 8K and 35K contexts and exercise visible continuation.

## Scope and cache contract

The base is `4c6a0c0ccef1362663ac4ba74fb5b473730ac48f`. Main K/V buffers already
have stepped storage, and QSA already retains completed pooled blocks. This
change only gives the RAW indexer-key lane its own backing capacity and logical
row count; it does not change pooling, normalization, attention or quantization.

New rows overwrite pending rows at the committed offset, bounded by the raw
prefix actually present. Missing rows are never synthesized. Trim retains spare
capacity but removes logical rows and drops derived pooled state. Public reads
and three-array serialized state never expose spare capacity. Copy preserves
owned state; restored state reserves new storage only when subsequently needed.
Dtype changes retain concatenate's promotion instead of silently casting writes.
No model-name, quant-name or context-length allowlist is used.

`VMLX_QSA_RAW_CAPACITY=0` selects the old concatenation control at cache creation.
`VMLX_QSA_RAW_STORAGE_TRACE=1` emits an opt-in first-update receipt, including
logical length, backing capacity and dtype. These are diagnostic switches, not
new application settings. No per-token environment lookup is added.

The storage principle was compared with oMLX's `_QSAIndexerCache` at revision
`7cbb407168ae628bbe0d7fe385be70e0954af303`. Its geometric reservation and media
position schema were not copied. Swift's separate QSA media-position companion
gap remains outside this patch and must not be declared qualified by text rates.

## Executed focused tests

Local M5 Max, existing bounded supervisor, optimized test host only:
`SWIFTTEST_QSARawStorageFinal0913__005647.log`.

- 23 tests in two suites, including `QSAKVCacheStorageTests`,
  `Qwen4ExpQSATests` and existing QSA persistence/quantization exclusions.
- 180 exact sequence checks across BF16/F16/F32, batch1/2, strided rows,
  contexts1/255/2040/8339/34939, single/multiple pending rows and trim-to-zero.
- Mixed-dtype transitions, retained views and independent copies, empty updates,
  valid serialized lengths, damaged/missing lanes, and safetensors save/reopen
  followed by restored raw/KV continuation across the2048 budget.
- Exit0; tracked peak3.93GiB; swap0.49GiB unchanged; ownership cleanup0/0/0.
  Only `swift-format` whitespace/layout changes followed this test run.

Median of three alternating-order synthetic12-lane raw-cache measurements,
256 appended tokens per arm (milliseconds per12-lane update):

| Starting rows | Concatenation | Capacity |
|---|---:|---:|
| 1024 | 0.285697 | 0.227453 |
| 8339 | 0.541576 | 0.240424 |
| 34939 | 0.610332 | 0.209980 |

These synchronized raw-cache measurements do not include model compute or app
streaming. The internal growth counter counts explicit backing-store growth,
not physical Metal allocations; retained views can still prevent donation.
No full-model tok/s gain, universal45tok/s floor, VLM quality, RAM soak or MTP
performance claim is made. Dev-app source/pin/binary and live rows are pending.

Private detailed receipts: `vmlx-private-evidence/post-1653-qwen38-audit/`;
supervisor logs in sibling `mtp-swift-2026-09-04/logs/`.
