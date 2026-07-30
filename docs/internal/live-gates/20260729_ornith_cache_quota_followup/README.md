# Ornith recurrent-state and linked disk-quota follow-up

Date: 2026-07-29

Status: `PARTIAL`. Focused source, cache, tiny-forward, and local-model replay
rows pass as recorded below. The coordinated Osaurus pin and real-app UI
lifecycle are separate downstream gates.

## Source and model identity

- vMLX baseline: `439f53694f3d630663e97612c264ae73e499121a`
- tested vMLX head: `68935c5c46e89e79827ea504cea7ac56b7527f20`
- model bundle:
  `/Users/eric/models/JANGQ-AI/Ornith-1.0-9B-JANG_4M`
- `config.json` SHA-256:
  `2f4828bffc846f5f8e300284addfdebef8ef7a1f9583636a982c0c137449cc63`
- `generation_config.json` SHA-256:
  `12a900f4edc6e82f7b03c94b1abaf8763b7fdec971c6e1440809461e21e8eae3`
- `tokenizer_config.json` SHA-256:
  `792fa3f0cb88b111e54ef3134c873531008c4df471d108da17903426e308aa7b`
- generation configuration contains no temperature, top-p, top-k, min-p,
  repetition-penalty, or sampling override. The replay uses bundle defaults,
  a 512-token prefill step, a named random seed, and the stated diagnostic
  output cap. Paged RAM is disabled.

## Focused automated results

| Lane | Result |
| --- | ---: |
| GatedDelta full-precision source contract | 1/1 passed |
| Qwen 3.5 VLM tiny forward and recurrent cache dtype | 1/1 passed |
| linked oversized-boundary quota regression | 1/1 passed |
| complete `DiskCache` focused lane | 18/18 passed |

The initial tiny VLM test launch failed because the ad-hoc local test bundle
lacked its packaged MLX metallib. Copying the already-built package metallibs
into that test bundle fixed the packaging-only failure; the unchanged test then
passed. No model/runtime assertion failed.

## Exact reported-history replay

The `BENCH_ORNITH_REPORTED_REPLAY=1` RunBench lane constructs the reported
sequence: a 9,523-byte Swift file result, failed capability discoveries,
repeated database tool calls, plan/review turns, and a second file result. Two
one-token precursor requests populate the same persistent boundary path before
the final continuation.

| Seed | Disk cache | Final observation |
| ---: | :---: | --- |
| 1 | on | `stop`; 272 generated tokens; 98 visible characters; one tool call; disk hits 2; SSM hits 2; zero exclamation run |
| 1 | off | `stop`; 267 generated tokens; 72 visible characters; one tool call; zero exclamation run |
| 2 | on/off | raw tokens matched; coherent open full-file `file_write` call reached the 512-token diagnostic cap |
| 2 | on | the same large envelope reached the 4,096-token diagnostic cap |
| 3 | on | valid tool call and `stop`; zero exclamation run |
| 4 | on | valid tool call and `stop`; zero exclamation run |
| 5 | on | coherent large full-file write envelope reached the 512-token diagnostic cap; zero exclamation run |

Seed 1 is the reported cache-dependent corruption row and passes with and
without disk cache after the fix. Seeds 2 and 5 are retained as failed terminal
lifecycle rows under their diagnostic output caps; they are coherent oversized
tool envelopes, not punctuation degeneration, and are not hidden with sampler,
prompt, parser, or forced-stop behavior.

## Proof boundary

- The replay is a non-UI runtime harness. It does not prove Osaurus tool-card,
  Stop-control, input-unlock, or follow-up-turn lifecycle.
- The downstream Osaurus PR must pin this exact vMLX revision (or its merged
  descendant), rebuild, and rerun affected focused/CI gates.
- Computer Use was explicitly stopped, so no real-app lifecycle claim is made
  here.
