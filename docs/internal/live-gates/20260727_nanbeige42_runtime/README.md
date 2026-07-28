# Nanbeige 4.2 vMLX Swift runtime gate — 2026-07-27

Status: `PARTIAL_OSAURUS_LIVE_UI_MISSING`

This note records the current vMLX Swift runtime proof for the local Nanbeige
4.2 bundles. It is text-only; no screenshots or binary proof artifacts are
committed.

## Source contract implemented

- Model registry route: `model_type = "nanbeige"` dispatches to
  `NanbeigeModel`.
- Runtime shape: one physical stack of 22 Llama-shaped layers runs for
  `num_loops = 2`.
- KV topology: 44 independent cache slots using
  `layerIndex + loopIndex * layers.count`.
- Bundle contract: `jang_runtime.cache_slots` must equal `layers * loops` and
  `cache_slot_formula` must be
  `layer_idx + loop_idx * num_hidden_layers`.
- Head dimension: requires explicit `head_dim` or `kv_channels`; it does not
  derive 64 from `hidden_size / num_attention_heads`.
- Future architecture flags are rejected instead of silently using the 4.2
  runtime path.
- RoPE, RMSNorm, MLP, attention, and tied output projection reuse the existing
  Llama-compatible implementations.

## Local bundle metadata checked

Bundle: `/Users/eric/models/JANGQ-AI/Nanbeige4.2-3B-JANG_4M`

- `config.json` includes `num_loops = 2` and `jang_runtime.cache_slots = 44`.
- `config.json` includes `jang_runtime.cache_slot_formula =
  "layer_idx + loop_idx * num_hidden_layers"`.
- `jang_config.json` includes `runtime.cache_slots = 44`.
- `jang_config.json` stamps `capabilities.reasoning_parser = "qwen3"` and
  `capabilities.tool_parser = "xml_function"`.
- `generation_config.json` defaults are `temperature = 0.6`, `top_p = 0.95`,
  `top_k = 20`, `do_sample = true`.

## Source tests

Command:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcrun swift test --no-parallel --filter NanbeigeRuntimeTests
```

Result: `PASS` — 6 tests, 1 suite.

Covered:

- registry dispatch to `NanbeigeModel`;
- `newCache` creates one cache per effective loop-layer slot;
- all four synthetic cache slots advance together in a two-layer/two-loop
  forward pass;
- `loop_loss_weights` override loop count as `count + 1`;
- mismatched `jang_runtime.cache_slots` is rejected;
- future incompatible Nanbeige flags are rejected;
- source formula check prevents modulo/aliasing regression.

## RunBench runtime probes

These are runtime diagnostics, not Osaurus live UI proof.

### JANG_4M

- Config smoke: `PASS`, route `nanbeige`, 22 layers, 48 query heads, 8 KV
  heads, profile `JANG_4M`.
- Template smoke: `PASS`, xml-function tools render, qwen3 thinking template
  renders.
- Thinking-on prompt: `PASS`, 248 generated tokens, 44.7 tok/s, closed
  `</think>`, `finish = stop`, visible content coherent.
- Thinking-off prompt: `PASS`, no reasoning chars, `finish = stop`, visible
  content coherent.
- Tool-call probe: `PASS`, parsed structured
  `get_weather({"location":"Tokyo"})`, no tool markers leaked to content.
- Disk L2 growing-cache probe: `PASS`, cache topology `layers=44,kvLayers=44`,
  prompt-boundary disk hit restored 533 tokens, post-answer disk hit restored
  540 tokens, turn-2 prompt time ratio 0.18, TurboQuant compressions 0.

### JANG_6M

- Config smoke: `PASS`, profile `JANG_6M`.
- Template smoke: `PASS`.
- Thinking-off prompt: `PASS`, no reasoning chars, `finish = stop`, visible
  content coherent.
- Disk L2 growing-cache probe: `PASS`, cache topology `layers=44,kvLayers=44`,
  disk hit restored 348 tokens, turn-2 prompt time ratio 0.49, TurboQuant
  compressions 0.

### MXFP8

- Config smoke: `PASS`, profile `MXFP8`.
- Template smoke: `PASS`.
- Thinking-off prompt: `PASS`, no reasoning chars, `finish = stop`, visible
  content coherent.
- Disk L2 growing-cache probe: `PASS`, cache topology `layers=44,kvLayers=44`,
  disk hit restored 348 tokens, turn-2 prompt time ratio 0.53, TurboQuant
  compressions 0.

## Still missing before user-facing compatibility

- Isolated Release Osaurus app build pinned to this vMLX source.
- Live UI load proof for each available Nanbeige bundle.
- Visible reasoning UI proof with thinking enabled and disabled.
- Live interleaved reasoning -> tool -> reasoning -> final answer proof.
- Stop-button/input-unlock/follow-up-turn proof.
- Osaurus Server/Cache UI counters for disk L2 reuse from the live app.
- App-level settings proof that bundle generation defaults are used unless
  overridden by visible user settings.

Until those rows are complete, this runtime is not release-verified for
Osaurus.
