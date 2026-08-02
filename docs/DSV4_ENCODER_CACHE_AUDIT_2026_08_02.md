# DSV4 encoder × cache audit (2026-08-02)

Audit of `DeepseekV4ChatEncoder` against the official
`encoding/encoding_dsv4.py` shipped in the bundle
(`dsv4-chat-compatibility.json`: `template_source: official:encoding/encoding_dsv4.py`,
`official_encoder_tests_passed: true`), focused on the encoder behaviours that
prefix caching depends on.

Companion to `DSV4_TOOL_TURN_PREFILL_2026_08_02.md`, which covers the
continuation-boundary defect and its fix.

## Why the encoder matters to caching

`canonicalChatCacheBoundaries` publishes a boundary only when re-rendering a
message prefix reproduces an exact **token** prefix of the real prompt, and
cache entries are stored only **at** published boundaries. So any encoder
behaviour that is non-reproducible, order-dependent, or shifts history bytes
retroactively destroys reuse for the whole conversation after it.

## Findings

### 1. Non-deterministic tool-call re-render — FIXED

`renderToolCallInvoke(name:params:)` iterated an unordered Swift `Dictionary`,
emitting a different parameter order per process. Reached from
`renderToolCallInvoke(name:arguments:)` whenever the arguments string does not
scan as an ordered JSON object.

The official encoder walks `json.loads(...).items()` and is stable; the ordered
overload preserves the original JSON spelling. Only this fallback diverged.
Consequence: history containing a tool call re-renders to different bytes after
an app restart, so every boundary after that tool call stops matching. Same
class as the per-process-random ordering removed from the load path in #108.

Fixed by iterating `params.keys.sorted()`. Sorted order cannot recover the
model's original emission order, but it makes the bytes reproducible, which is
what reuse requires. Pinned by "tool-call re-render is byte-stable across
processes" (12 renders, one distinct result, plus an explicit order assertion).

### 2. `drop_thinking` flips when tools appear — BY DESIGN, cache-invalidating

Per the official README: *"With tools (on system or developer message):
`drop_thinking` is automatically disabled. All turns retain their reasoning."*
`encode` mirrors this:

```swift
if full.contains(where: { !($0.tools?.isEmpty ?? true) }) { effectiveDrop = false }
```

So enabling tools mid-conversation retroactively **restores** `<think>` blocks
to every earlier assistant turn. The history bytes for turns already cached
change, and every previously published boundary is invalidated.

This is correct — the prompt genuinely differs — but it means toggling Sandbox
mid-chat legitimately costs a full cold prefill. It is not a bug and must not
be "fixed" by suppressing the flip; recorded here so a future cache
investigation does not misread it as one.

### 3. `latest_reminder` gains a generation rail — INTENTIONAL DIVERGENCE

Official Python appends the assistant rail only for `user` / `developer`:

```python
elif messages[index].get("role") in ["user", "developer"]:
    prompt += ASSISTANT_SP_TOKEN
```

Swift also appends it for `.latestReminder`. That is required by
`requiredToolChoiceReminder`, which inserts a `latest_reminder` **after** the
last user message when `tool_choice: required`; without the rail that prompt
would never open an assistant turn. The divergence is deliberate and
load-bearing, but it was undocumented. It also means a transcript ending in a
reminder still gets a strippable rail, so boundary derivation is unaffected for
that shape.

### 4. Unsupported content blocks are dropped, not rendered

Python renders `[Unsupported {block_type}]` into the prompt for unknown block
types; `contentBlocks` returns nil for them, so they vanish. Dropping is the
safer choice — it keeps synthetic placeholder text out of the prompt — but it
is a divergence, and a message whose blocks are ALL unsupported degrades to
`msg.content`.

### 5. DSML is XML-shaped and the format defines no escaping

The official encoder writes string parameter values raw:

```python
value=v if isinstance(v, str) else to_json(v)
```

No XML escaping, and no escaping on the parse side either —
`parseCanonicalInvokes` takes the value as the bytes up to the first
`</｜DSML｜parameter>`. So a `string="true"` value that itself contains that
literal closer is unrepresentable in the format and will truncate the argument.

This is a property of the 0731 format, not a vMLX bug, and it is reachable in
practice: DSV4 writes whole source files through `file_write`, and the live
minesweeper run put 7 KB of HTML through exactly this path. Nothing in the
current parser detects the case.

Enforcement already present and verified correct:

- unregistered tool name rejects the whole envelope
- argument schema validated (`required`, `additionalProperties`, types)
- duplicate parameter names rejected
- `string` flag must be exactly `true` or `false`
- canonical envelope is all-or-nothing, so no prefix of a malformed parallel
  call block can execute
- a canonical envelope with a non-DSML body is quarantined and surfaced as
  `.malformedEnvelope` rather than leaked as visible text

### 6. Reasoning enforcement is clean — and `reasoning_effort` invalidates the STABLE prefix

`DeepseekV4ReasoningPolicy` satisfies the MLXPress non-negotiables. There are no
forced thinking tags, no injected `<think>`, no decode-loop close-token bias and
no prompt coercion. Rails are selected only by the caller's explicit controls:

- an explicit `enable_thinking: false` wins and drops `reasoning_effort` — a
  valid effort can never turn a caller's explicit `false` back on;
- an unsupported effort throws `invalidReasoningEffort` at the processor
  boundary rather than silently selecting another rail;
- both process-env overrides are deprecated, with the reason stated in the
  deprecation message ("Public request/template controls must win").

The caching consequence is the part to remember. The effort preface is emitted
at **index 0** in thinking mode:

```swift
if index == 0 && thinkingMode == .thinking { out += ...Preface }
```

so changing effort rewrites the very first bytes of the prompt. That
invalidates not just the history boundary but the **stable** system boundary
too (`[72, 1128]` in the live traces) — i.e. the effort chip visible in the
chat input ("DeepSeek V4 Flash… · Low") is a full-cache-invalidation control,
not a cheap toggle. Same for `enable_thinking`, which changes the terminal rail
`<think>` vs `</think>` and, with tools present, whether earlier turns keep
their reasoning at all (finding 2).

This is correct behaviour — the prompt genuinely differs — but a cache
investigation that changes effort between runs will measure a cold prefill and
must not read it as a regression.

### 7. Tool matrix, live on `db9250c2`

Driven through the real chat UI, isolated root, Sandbox on, no folder attached.

| tool | result | cache |
|---|---|---|
| artifact / file write | card started AND finished ("Wrote a file · 2.7s"), diff card, coherent answer | `HIT 1231 remaining=435` |
| file read | correctly reported it has no file-reading tool in this session (no folder attached) — honest refusal, no hallucinated envelope, no hang | 4 consecutive HITs across a 4-step agent loop: 1276 → 1380 → 2417 → 3433, **zero misses** |
| task tracking / todo | "Todo 0/3" panel rendered, all three items named back | `HIT 4515 remaining=295` |

The file-read row is worth keeping: the model had no matching tool and said so
in prose instead of emitting a malformed DSML envelope. That is the behaviour
the strict canonical parser is supposed to encourage, and it did not produce the
"Preparing tool call" stall.

**Mutable system sections move the whole-system boundary, by design.** The todo
tool writes a section into the system prompt, and the stable set shifted between
consecutive computations in the same session:

```
prompt=2596 stable=[72, 1160] all=[72, 1160, 2594]
prompt=4810 stable=[72, 1128] all=[72, 1128, 4714]
```

`1160 → 1128` as the todo section changed, while the **static** boundary `72`
held. That is exactly what `hintedStaticSystemBoundary` exists for, and it is
why the static hint must keep working: without it, every todo mutation would
cost the entire system prefix rather than just its mutable tail.

### 8. The `stored+2` re-warm miss — ROOT-CAUSED, still OPEN

vmlx#201 made this much rarer (warm-fetch miss rate 43% → 20%) by publishing
boundaries for continuation shapes, but the underlying gap is separate and
unfixed. Signature, seen on both pins:

```
store 1276 -> boundaries all=[72, 1128, 1279] -> fetch tokens=1278 MISS
store 2084 -> fetch 2086 MISS
store 2286 -> fetch 2288 MISS
store 2756 -> fetch 2758 MISS
```

It is **not** a probe-range problem. `CacheCoordinator` pages the whole index
for every stored entry `<= tokens.count` and content-address checks each one,
and the very same 1276 entry HITs for a later, larger request
(`HIT 1276 remaining=105` at `tokens=1381`). So the entry is good; the re-warm's
first 1276 tokens simply are not the stored ones.

Cause, `BatchEngine.swift:3255`:

```swift
let generatedBoundaryTokens = promptTokens + slot.generatedTokenIds
```

The post-answer entry stores **prompt + RAW generated tokens** — verified
arithmetic: `1223 + 53 = 1276`. The next request instead carries the
**re-rendered** assistant turn, `{reasoning}</think>{content}{tool_calls}` plus
`<｜end▁of▁sentence｜>`, then the 2-token generation rail. The re-rendered form
is not token-identical to the raw generation, so the stored sequence is not a
prefix of the next prompt and the content check correctly rejects it.

This is the same class the hybrid path already solved: the gen-suffix-stripped
boundary added in vmlx#125 (`BatchEngine.swift:2891`, `promptTokens.lastIndex(of:
turnStartToken)`), which stores back to the last turn start rather than trusting
the generated suffix. DSV4 does not take that path — its post-answer store is
gated on `!usesCanonicalHybridBoundary`, which is false for DSV4, so it stores
the raw-generated boundary instead.

Fix direction: store the post-answer boundary at the re-rendered history length,
or strip the generated suffix to the last turn start as the hybrids do. Impact
is confined to the background re-warm — visible turns hit — so this is a wasted
prefill rather than a user-visible stall. Needs its own live visual proof.

### 9. No partial-content validator — DELIBERATE, do not "fix" naively

`DSMLToolCallParser` does not override `isValidPartialContent`, so a canonical
envelope buffers until its closer. Adding a strict grammar check there looks
attractive (it is what LFM2 / Hunyuan / MiniMax do) but it **breaks** finding 5's
quarantine contract: `expectCanonicalRejection` requires a canonical envelope
with a garbage body to keep buffering so it can be reported as
`.malformedEnvelope` instead of flushed to the user. An attempt at this during
this audit turned 4 existing tests red for exactly that reason and was reverted.

Any future bound here must terminate the *runaway* case without flushing the
*quarantined* case — the two are distinguished by whether an invoke is open,
not by buffer size.
