# Bonsai2 Hadamard / packed ternary runtime checkpoint

Status: **PARTIAL — implementation and tests authored, no Swift execution yet.**
This is a private, local implementation checkpoint, not permission to publish
the model repositories, merge a runtime PR, or advertise working model support.

## Isolation and source bindings

- Worktree: `/Users/eric/vmlx-bonsai2-runtime`, branch `feat/bonsai2-runtime`.
- Base: `osaurus-ai/vmlx-swift` main, `bfb34ff142817f3a35cf6502ad5d8dd742c4e87f`.
- Authoritative handoff:
  `/Users/eric/jang/docs/runtime/bonsai2-27b-2026-09-17/01-RUNTIME-HANDOFF.md`,
  SHA-256 `21dc937d27be84153dc982ba9757e5d686234cb753e806019682c1fb589b602f`.
- Python source read before editing:
  `/Users/eric/mlx/vllm-mlx/vmlx_engine/utils/jang_hadamard.py`,
  SHA-256 `94255da326cde1511743fe561f53bbbcad4322d5d57158f2e642b5db68d08a7d`;
  `jang_ternary_packed.py`, SHA-256
  `d3806c1af5e9f5918504983a0b8b7f5def0899f4dc0f55b9134e84c3ad34da78`;
  `jang_loader.py`, SHA-256
  `350af1678dd3b0fb297b769c66f1876bd163032df21b4b0ea78c2b3ad2b0d189`.
- Converter reference:
  `/Users/eric/jang/jang-tools/jang_tools/ternary_packed.py` and
  `convert_bonsai2_jang_affine.py`; no converter edits here.

Prior fixes inspected and retained:

| Commit | Owning behavior | This patch |
| --- | --- | --- |
| `948f03aec89501104a7ab93360508c7b83a1c3a8` | Native affine-1 storage and exact schema-2 manifests | Does not expand or replace affine-1 |
| `c0e869c8a3f0af5fe5db55dadd9fd9206f1e15c8` | Qwen raw-array GDN input/tail fusions | Declines only activation-rotated wrappers |
| `bf8b31995` | Later Qwen fusion policy | Retained |
| `5332a2a2b`, `4f9e2d176`, `f2b184841` | Post-load bf16 materialization policy and telemetry | Only validated Hadamard bundles preserve their stored F16/F32 contract |
| `2422cfb8e`, `7d949c263` | Affine mmap scale preservation and embedding output dtype | Ordinary bundle policy unchanged |
| `0aa728af5deab52141506828ba36a91d5fe2fd51` | Owned uncached resident reads | Unchanged |

## Bounded implementation

`Libraries/MLXLMCommon/JangHadamard.swift` validates both config owners, Prism's
sidecar, strict Boolean runtime markers, complete schema-2 manifest coverage,
non-overlapping directions, supported transform/block, exact F32 +/-1 signs,
source architecture dimensions and F16 quantization metadata. Unknown routes,
tied heads, MoE and MTP configurations refuse this port rather than silently
using plain affine weights. Wrappers reuse the existing quantized arrays;
forward is `H(signs*x)`, inverse embedding lookup is `signs*H(rows)`, with
normalized F32 computation and cast-back to the activation dtype.

`JangTernaryPacked.swift` expands UInt8 26-byte/128-trit groups into native
UInt32 affine-2 words, preserving scales and creating biases as `-scales`.
Head bytes above 242, tail bytes above 26, bad shape/dtype, missing modules and
pre-existing biases throw. Expansion runs before manifest shape inference and
sanitize in `Load.swift`. Coverage must equal the declared Hadamard set.

The real `lm_head` is 248320 x 5120: materializing all 1.27 billion trits as
UInt32 intermediates would be multi-GiB. The implementation evaluates row
chunks targeting 1,048,576 codes, minimum one full row, before final output
concatenation. This bounds each code intermediate by rows, not total process
footprint. Final output buffers, concatenation and allocator retention remain
measurement items. **No new global allocator/cache-limit mutation is made.**

Two additional source integration defects would otherwise invalidate a
wrapper-only port:

1. Qwen VLM GDN's uniform/grouped input projections and compiled output tail
   read `QuantizedLinear.weight/scales/biases` directly. Because the new wrapper
   is a subclass, an unguarded cast succeeds while bypassing its transform.
   The text GDN grouped-input path has the same issue. Targeted eligibility
   guards retain plain affine eligibility; whole-model compilation is not
   disabled.
2. Ordinary post-load bf16 conversion would round the bundle's F16 scales and
   F32 norms/state projections. Only a successfully parsed and validated
   Hadamard contract bypasses that conversion. Existing norm sanitize `+1`,
   parser, sampler, EOS, cache topology and media code are not replaced.

## Current source / template route matrix

Both private bundles have byte-identical tokenizer config, generation config
and Hadamard sidecar (hashes below). This is artifact evidence, not a rendered
template or model-output pass.

| Surface | Current source binding | Required live proof |
| --- | --- | --- |
| Storage A / B | Common `loadWeights` from `LLMModelFactory.swift:2032` and `VLMModelFactory.swift:754`; wrappers installed before parameter update | Both complete loads and identical parity logits |
| Text route | Wrapped `Qwen35Model`; standalone `Qwen35TextModel` intentionally refuses the full bundle's paths | Text multi-turn and cache restart |
| VLM route | `qwen3_5` factory entry; `Qwen3VLProcessor` registry; `Qwen35.prepare` image/video feature merge | Real image, OCR, video and same-media cache reuse |
| Reasoning | Actual `tokenizer_config.json` template reads `enable_thinking`; accepts exactly `low`, `medium`, `xhigh`, default `xhigh`; `preserve_thinking` defaults true | Off/low/medium/xhigh render context, visible answer/reasoning and no marker leakage |
| Tools | Template renders the actual `tools` array as JSON schemas; native XML `<function>` calls. Bundle declares `qwen3_coder` / `xml_function` | Real schemas, parsed arguments, tool round trip and multiple calls |
| Processor preservation | `Qwen3VL.swift:131` forwards tools and additionalContext; text return at 150 and media return at 233 retain `toolSchemas` and reasoning cache salt | End-to-end tool/media payload, not a keyword assertion |
| Cache ordering | `Evaluate.swift:4797` stores after decode and before stream finish; `CacheCoordinator.swift:915` persists typed KV and recurrent state; offset/key mismatches refuse storage | Disk write completion, subsequent hit, exact state/topology and prefill timings |

App source inspected read-only at
`/Users/eric/osaurus-native-tool-batches`, HEAD
`6267660b2811b7e8bdc13573ef88039d3501e61e` (the parent owns this worktree):

- `Packages/OsaurusCore/Services/DeclaredReasoningEffort.swift:178` resolves
  the JANG declaration before template fallback; `:331` prepends `none` to
  declared effort levels. `ModelOptions.swift:310` uses this capability for
  picker options, and preserves omission as the native default.
- `Services/ModelRuntime/MLXBatchAdapter.swift:1119` uses the declaration for
  effort transport; `:1138` only forwards explicit `preserveThinking`;
  `:1233` maps Off to `enable_thinking=false`. The template itself closes the
  think block; this port adds no forced markers or sampler overrides.
- The actual UI labels are None / Light / Medium / Extra High
  (`Models/Configuration/ModelOptions.swift:95`), with wire values
  `none / low / medium / xhigh`. The user's “med” means the native `medium`,
  not a new template value.
- `MLXBatchAdapter.swift:1675` drains the producer/cache store before yielding
  terminal info and releasing the solo lease. A tool event alone does not
  terminate the stream.

Important cache limit: a tool turn deliberately does **not** persist a
post-generated tool-call boundary (`includeGeneratedBoundary` excludes tool
calls). It persists eligible canonical prompt boundaries. A disabled disk
cache, memory-store budget refusal, unsafe offset, or quota eviction can
legitimately produce no durable entry. Do not claim “every tool always writes
to disk,” or use a cache-write screenshot as proof of a valid subsequent hit.
No cache/parser rewrite is justified by the new storage format alone.

## Metadata/header checks actually run, 2026-09-17

Read-only Node check: parse config/JANG/Prism/generation/index JSON; for each
indexed safetensor shard read only its 8-byte header length and JSON header;
validate every declared rotated weight/scale/sign/bias dtype and shape. No
weight data, MLX evaluation, tokenizer execution or generation was performed.

| Local bundle under `/Users/eric/models/OsaurusAI/` | Rotated layouts checked | Shard-header bytes read | `lm_head.weight` |
| --- | --- | --- | --- |
| `Bonsai-2-27B-Ternary-JANG` | 402 (401 forward, 1 inverse) | 320128 | U32 `[248320,320]` |
| `Bonsai-2-27B-1.75bit-JANG` | 402 (401 forward, 1 inverse) | 263696 | U8 `[248320,1040]` |

Each manifest declares and contains 485 modules (402 language, 83 vision).
Sign widths are 5120/6144/17408; final text norm is F32 `[5120]`.
Config EOS is 248046; generation EOS is `[248046,248044]`.

Metadata SHA-256 receipts:

| File | Ternary | Packed |
| --- | --- | --- |
| `config.json` | `f57b9c8cfc0d9d4edf35f65b75d230c1ee9c85467f40dda61a3bf8d07b3ee082` | `500b966bb0a564d601690d7673388268923f6d66a1690d4d4b4d07ef73cde16e` |
| `jang_config.json` | `387035edc801ffb7dc59c9a7dc41a00a8bca7408187d41e93f03bbe0dff021a4` | `f8104703441fb2ebc94889e76cc306771d91f919c534a18b95f724543fb63442` |
| `hadamard.json` | `7132a3ec364f0bdac1f08f905f24f0ad2f14245060f592637a0396826d3b5fe6` | same |
| `generation_config.json` | `875ee16774666031c8cff7a0d19b02ee2264c71229f0664079200c469384f5c5` | same |
| `tokenizer_config.json` | `60e96a382893c9efeb7116fb83e2c532cb149d00bfd859713640c0a8ae282019` | same |
| `model.safetensors.index.json` | `26e0c700f49648c6199011443fa96069118285e7bf3e94d2c5c7d7ad10071b88` | `7068f83e14656b9837df72a7bab31095a0f299fbb571457307d246a1c3d5aa75` |
| Combined indexed header names + raw header bytes, filename-sorted | `b61003b40bf9776ba6298b5aa91dd03b0fe85f7f405f785dce366ea3dac51783` | `a8a3982f2aaff65a067f887ca6de9103ee9b1ccbd2229dc186c78fb6c694e79f` |

## Tests and remaining proof

Authored, **unexecuted**:

- `JangHadamardContractTests`: native/packed declarations, fallback owners,
  strict flags, malformed sidecar/manifest/coverage, checked width arithmetic,
  ordinary bundle nil behavior and native reasoning vocabulary/transport.
- `JangHadamardRuntimeTests`: independent scalar trit encoding, exact UInt32
  round trips at representative widths, malformed packed bytes/layouts,
  independent F32 butterfly reference, wrapper array reuse, actual generic
  loader dtype preservation and ordinary affine bf16 regression.
- `Qwen35HadamardRoutingTests`: supported/refused routes, actual text/VLM
  sanitize +1 vs already-shifted norms, F32/F16 passthrough, raw-fusion bypass
  prevention, cache offset/count/content equality.

Checks actually executed:

```sh
git diff --check
xcrun swift-format lint --strict Libraries/MLXLMCommon/JangHadamard.swift Libraries/MLXLMCommon/JangTernaryPacked.swift Tests/MLXLMTests/JangHadamardContractTests.swift Tests/MLXLMTests/JangHadamardRuntimeTests.swift Tests/MLXLMTests/Qwen35HadamardRoutingTests.swift
```

Both returned exit 0 with no diagnostics. These are not compilation or runtime
tests. Parent's Gemma build/live proof owns the resource slot; no competing
build, Metal test or Bonsai model was launched.

Planned command, only after an explicit serialized slot with full Xcode and
an approved isolated scratch/build path (not shared SourcePackages/DerivedData):

```sh
swift test --scratch-path /Users/eric/vmlx-bonsai2-runtime/.build-bonsai2 -j 2 --no-parallel --filter 'JangHadamardContractTests|JangHadamardRuntimeTests|Qwen35HadamardRoutingTests|JangAffine1RuntimeContractTests|Qwen35FusedInputProjectionTests|NormConventionResolverTests'
```

Missing acceptance evidence: Swift compilation and the focused tests; exact
four-prompt Python parity and native-vs-packed logits; both real bundles'
coherent natural-stop multi-turn output with tok/s, TTFT/prefill and physical
footprint; native reasoning controls; real tool schema/round-trip/batch calls;
per-tool canonical disk persistence plus next-turn restore and SSM/KV state;
image/OCR/video and cache-salt fidelity; isolated app visuals and normal
unload/cancellation. No throughput number is claimed for this Swift port.

After local proof and parent review, the authorized PR/CI/merge workflow must
carry these exact receipts in GitHub comments. No GitHub comment, push, merge,
model publication, release or app installation has occurred in this lane.
