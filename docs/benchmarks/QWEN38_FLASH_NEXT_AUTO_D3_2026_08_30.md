# Qwen3.8 Flash-Next Auto-D3 policy proof (2026-08-30)

## Scope and status

This record covers the vMLX policy that resolves **Auto** native MTP to depth 3
for bundles whose configuration identifies the `qwen4_exp` / Qwen3.8
Flash-Next family. It does not claim that an Osaurus UI or HTTP server built on
this revision has been exercised; those are downstream repin gates.

Status at source head `915f8bca`:

- policy/source tests: **PASS**;
- exact local-bundle census (2L, 4S, 4M): **PASS**;
- explicit D3 Swift decode performance inherited from the byte-identical
  verifier-row proof at `f5ebc0cc`: **PASS**;
- served API Auto-D3 proof: **PENDING downstream Osaurus repin**;
- visible native Chat Auto-D3 proof: **PENDING downstream Osaurus repin**.

## Safety boundary

The default is selected from `config.json` model types, never from a model path
or display name. The measured family aliases are:

- `qwen4_exp`
- `qwen4_exp_text`
- normalized `qwen4exp` equivalents

The family policy does not apply to Qwen3.5, Qwen3.6, or unrelated Qwen
families. An arbitrary bundle block and every `manual_blocked` record remain
authoritative. The only superseded block is the exact released 4M record that
names runtime commit `447a6a2b`, q4 quantization, the reason "measured slower
than AR at all depths", and the removed inability to verify through the fused
MoE path. Telemetry preserves both the original block and the explicit
`legacy_block_superseded` decision.

No generation sampler, reasoning setting, cache policy, model tensor, or
kernel is changed by this policy.

## Exact-head source and test evidence

Release `RunBench` SHA-256:

```text
bb874eefcec04bdb11d9d202e9ab799337764cd1d4818da86722ac0f4371ef63
```

Focused policy tests, each run independently on `915f8bca`:

| Contract | Result |
| --- | --- |
| complete Flash-Next head without sidecar resolves Auto D3 | 1/1 pass |
| explicit bundle block vetoes family cold start | 1/1 pass |
| exact frozen 4M block is superseded | 1/1 pass |
| Qwen3.5/Qwen3.6 do not inherit the policy | 1/1 pass |
| default server settings resolve Flash-Next Auto D3 | 1/1 pass |
| server supersedes only the frozen 4M block | 1/1 pass |

The full `VMLXServerRuntimeSettingsTests` run passed 38/38. The broader
`MTPRuntimeFocusedTests` run passed 90/91; its one failure is an inherited
Qwen3.5-VL source-string assertion expecting
`sigmoid(b).asType(.float32)` after the VL implementation fused beta inside
its Metal kernel. It is outside this four-file policy diff and is not counted
as a policy pass.

## Exact installed-bundle census

All three rows used the exact Release binary above. The complete logs are:

```text
/Users/eric/vmlx-private-evidence/qwen38-auto-d3-20260830/Qwen3.8-Flash-Next-JANG_2L-policy-915f8bca-census.log
/Users/eric/vmlx-private-evidence/qwen38-auto-d3-20260830/Qwen3.8-Flash-Next-JANG_4S-policy-915f8bca-census.log
/Users/eric/vmlx-private-evidence/qwen38-auto-d3-20260830/Qwen3.8-Flash-Next-JANG_4M-policy-915f8bca-census.log
```

| Bundle | MTP tensors | Bundle tuning | Auto result |
| --- | ---: | --- | --- |
| JANG_2L | 57 | D3, usable | `canAutoLaunch=true`, `measured_family=d3` |
| JANG_4S | 57 | D3, usable | `canAutoLaunch=true`, `measured_family=d3` |
| JANG_4M | 57 | frozen pre-parity block | `canAutoLaunch=true`, `legacy_block=superseded_by_row_parity_fix`, `measured_family=d3` |

## Decode evidence supporting D3

The exact predecessor verifier-row head `f5ebc0cc` produced the following
greedy `Count from 1 to 200.` rows. Both generated the exact sequence, stopped
normally, and had full-output SHA-256
`20dbc5d7831d7a43c23bfa2cedb198965012b4d5b722d8c81a47271eb8c58b33`.

| Bundle | D3 tok/s | Tail tok/s | Accepted verifies | Fallback | Peak physical footprint |
| --- | ---: | ---: | ---: | ---: | ---: |
| JANG_4S | 68.4 | 73.2 | 222/223 | 0 | 59,886 MiB |
| JANG_4M | 57.9 | 61.7 | 222/223 | 0 | 76,776 MiB |

The 4S PLE reader remained SSD-backed through `pread` with `F_NOCACHE`. The
4M row enabled the model-wide q4/g64 verifier-row path; the mixed 4S bundle did
not, preventing the earlier 4S slowdown caused by a per-block selector.

Reference logs:

```text
/Users/eric/vmlx-private-evidence/qwen38-auto-d3-20260830/4S-pr-f5ebc0cc-d3-count200.log
/Users/eric/vmlx-private-evidence/qwen38-auto-d3-20260830/4M-pr-f5ebc0cc-d3-count200-fixture.log
```

Python v1.6.47 provides the cross-runtime control rather than a Swift claim:
4S D3 measured 88.79 tok/s versus 40.17 AR (2.21x), and 4M D3 measured
74.76 tok/s versus 40.57 AR. Those Python wins used native D3 MTP, not CoreML,
ANE, NAX, or an n-gram drafter.

## Downstream acceptance gate

After the vMLX revision is pinned in Osaurus:

1. Build a Release app from the exact repin head.
2. Prove the served default model resolves `native_mtp:d3`, with verify,
   accepted, rejected, bonus, and AR-fallback counters plus TTFT/tok/s.
3. Drive the native Chat UI with exactly one app process and visually confirm
   Auto, depth-3 telemetry, coherent output, and no tail hang.
4. Run tool-result continuation, Stop/continue, multi-turn, and disk-cache
   restoration before making a user-facing readiness claim.

