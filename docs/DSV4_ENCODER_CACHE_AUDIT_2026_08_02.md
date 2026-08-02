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

### 6. No partial-content validator — DELIBERATE, do not "fix" naively

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
