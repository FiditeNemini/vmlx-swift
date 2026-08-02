# DSV4 tool-turn prefill: live findings (2026-08-02)

Live Osaurus Release run, DeepSeek-V4-Flash-0731-JANG, vMLX `9d22ac14`,
isolated root `osaurus-dsv4-double-prefill-proof-root.CjFIm55GAY`,
`VMLX_CACHE_FETCH_TRACE=1`.

## 1. Duplicate prefill on the visible send — FIXED, live-proven

Before `9d22ac14` the reusable-prefix warm-up was excluded from the DSV4
prefill-captured N-1 disk store, so the warm-up prefilled the prefix and the
visible send then logged `MISS all tiers` and prefilled the same tokens again.
That is the "prefill runs twice" the UI showed.

Live trace after the fix:

```
[vmlx][cache/fetch] MISS all tiers tokens=1129            <- warm-up, cold, stores seed
[vmlx][cache/fetch] HIT disk boundary=1127 remaining=96 tokens=1223   <- visible send
```

The visible send restores 1127 of 1223 tokens and prefills only the 96-token
user suffix. Turn finalized: reasoning opened and closed ("Thought for 3.6s"),
coherent answer, `TTFT 1.73s · 31.3 tok/s · 144 tokens`, Stop cleared, input
unlocked.

Two `[vmlx][cache/boundaries]` lines per warm-up are **not** a duplicate
prefill. They are the two probe renders in
`MLXBatchAdapter.warmupSendInvariantPrefixTokens`, which tokenize twice to find
the send-invariant prefix. Only `cache/fetch` marks real prefill work. An
earlier diagnosis mis-read those probe lines as a second prefill.

## 2. Tool turns lose the history cache boundary — ROOT-CAUSED AND FIXED

This is the defect behind "tool calls hang forever". The tool call itself is
fine: DSML parses, the tool runs, the artifact is written
(`minesweeper.html`, 7 KB). What fails is prefix reuse afterwards.

`canonicalChatCacheBoundaries` (Tokenizer.swift:62) publishes three kinds of
boundary. The **history** boundary comes from `exactPrefixBoundary(messages:)`,
which re-renders the whole message list with `addGenerationPrompt: false` and
requires the result to be an exact token prefix of the real prompt. Cache
entries can only be stored at a published boundary.

Observed timeline:

| step | prompt | `all=` boundaries | fetch |
|---|---|---|---|
| 1 | 1223 | `[72, 1128, 1221]` | HIT 1127, remaining 96 |
| 2 | 1378 | `[72, 1128, 1376]` | HIT 1367, remaining 9 |
| 3 | 1458 | `[72, 1128, 1456]` | HIT 1367, remaining 91 |
| 4 | **7017** | **`[72, 1128]`** | HIT 1457, **remaining 5560** |

At step 4 — the first prompt containing the completed tool call and its 7 KB
artifact — the history boundary disappears. Only the static system boundaries
survive. Two consequences:

1. that turn cold-prefills 5560 tokens; and
2. because stores happen **at** published boundaries, no history entry can be
   stored for this prompt either, so the next agent iteration cannot reuse it.

The agent loop therefore re-prefills the whole growing transcript every
iteration. Osaurus's stream log shows the loop turning over with nothing to
show for it:

```
Stream completed: 1 content/reasoning deltas in 46.73s ... toolHints=0
Starting stream wrapper for model: DeepSeek-V4-Flash-0731-JANG
```

Visible symptom: "Thinking" stays animating, a "Preparing tool call" card sits
under it, Stop stays active, the process holds ~130% CPU, and each round is
slower than the last.

### Ruled out

- **Not a permissions problem.** `config/tools.json` has empty `policy` and
  `grants`, all 78 tools enabled, and the process had no dialog or sheet
  pending while stuck.
- **Not the DSML parser.** First and second tool calls both round-trip
  correctly over `/v1/chat/completions`, streaming and non-streaming, including
  a call issued after a tool result is fed back.
- **Not the generation-prompt rail.** `DeepseekV4ToolHistoryPrefixBoundaryTests`
  pins that rendering with `addGenerationPrompt: false` stays an exact textual
  prefix of the real prompt for a plain history, a tool-call history carrying a
  large string payload, and a completed tool round followed by a new user turn.
  All three pass, so the rail contract is intact for those shapes.

### Root cause

`exactPrefixBoundary` requires the no-rail render to be **strictly shorter**
than the real prompt:

```swift
tokens.count < promptTokens.count,
promptTokens.prefix(tokens.count).elementsEqual(tokens)
```

DSV4 appends its `<｜Assistant｜>` rail only after a user / developer /
latest_reminder message — `renderMessage`, mirroring `encoding_dsv4.py`:

```python
elif messages[index].get("role") in ["user", "developer"]:
    prompt += ASSISTANT_SP_TOKEN + <think>/</think>
```

So when the transcript ends in an **assistant continuation** — the shape every
agent-loop turn takes once the model is mid-turn — rendering with and without
the generation prompt produces *identical* tokens. Measured on the real
encoder: `withRail.count → 1283` == `withoutRail.count → 1283`. The strict `<`
then rejects the boundary and `try?`/`?? []` turns that into a silent nil, so
the history boundary disappears with no diagnostic.

The encoder is correct here; the boundary derivation was not.

### Fix

`canonicalChatCacheBoundaries` gains `trailingContinuationBoundary()`: when the
whole-list render yields no boundary and the last message is an assistant turn,
retry against the transcript **without** that trailing turn. That render is
strictly shorter and is still proven by the same exact-token-prefix check — no
relaxation of token identity, no suffix matching.

Regression cover, both sabotage-verified red:

- `CanonicalChatCacheBoundariesTests."a trailing assistant continuation still publishes a history boundary"`
  — without the fallback the published history set is `[]`.
- `DeepseekV4ToolHistoryPrefixBoundaryTests."a transcript ending in an assistant turn appends no generation rail"`
  — pins the encoder contract the fallback exists for.

Full focused target: 22 pre-existing failures in `DeepseekV4ChatTemplateFallbackFocusedTests`
and `VMLXMemorySafetySettingsTests` both with and without this change — this
change adds none.

## 3. End-of-turn finalization — MEASURED, still OPEN

Live rows on `6c936110`, fresh isolated root, comparing the app's own reported
numbers against wall-clock time from Send until the Stop control cleared:

| turn | TTFT | generate | expected | wall | unaccounted |
|---|---|---|---|---|---|
| tool call | 17.39s | 3.4s (18 tok @ 5.3) | 20.8s | 41s | **20.2s** |
| follow-up | 1.91s | 29.2s (681 tok @ 23.3) | 31.1s | 45s | **13.9s** |

The same 4-turn session wrote **11 cache blobs totalling 462 MB**, 26–63 MB
each, two stores per turn (prompt boundary + post-answer). That is the work
sitting between the last visible token and the turn finalizing, and it matches
the ordering in `BatchEngine.swift:3264-3296` — `storeCacheEntry`, then
`Stream().synchronize()`, then `slot.continuation.finish()`.

Caveats on the numbers: the wall figure carries up to 5s of poll granularity,
and the app's tok/s denominator is itself "honest" (it includes more than pure
decode), so `expected` is a slight underestimate and `unaccounted` is therefore
an upper bound. The magnitude is well outside that error, and the blob sizes
corroborate it, but treat 14–20s as indicative rather than exact.

Not fixed here. The GPU drain must stay — it is what closes the Metal
concurrent-encoder crash class. The 25–63 MB serialization is CPU/IO only and
is the candidate to move after `continuation.finish()`, provided the store's
GPU eval has already drained and the snapshot is retained.

## 3b. Original note on the ordering

`BatchEngine.swift:3264-3296` runs the end-of-turn cache store, then
`Stream().synchronize()`, and only then `slot.continuation.finish()`. Each DSV4
entry serializes 25-61 MB (measured: 20 entries / 524 MB in one short session),
so that work sits between the last token and the turn finalizing. The GPU drain
is deliberate and must stay — it is what closes the Metal
concurrent-encoder crash class. The disk serialization is CPU/IO only and is a
candidate to move off the critical path once the store's GPU eval has drained.

Not yet measured in isolation: the plain-chat turn above cleared Stop within
the 10s poll interval, so this has not been shown to be user-visible on short
turns.
