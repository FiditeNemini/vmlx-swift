# Preserve Gemma tool-selection history

## Source-bound defect and change boundary

Gemma4Processor's required/named-tool adapter deletes earlier user turns and completed tool protocol, but image processing still consumes every original image. A real isolated Osaurus E2B8bit run reproduced two images with only one image placeholder: maskedScatter rejected860160 vision values versus430080 positions before decoding. An earlier-image-only follow-up silently lost the image entirely. Identical auto requests retained history; required and named requests did not.

Origin:2be648a3c12817b1d57e06c86d0b9987b038bcb3 introduced tool compaction;447d2a07c6f0e0da3a4f59e35e8221417bc9e096 added earlier-user deletion and a separate text-only adapter. The old tests assert this lossy behavior rather than the complete-request contract. This correction replaces those expectations with production-processor tests. It does not undo the later scalar system-content repair, structured tool-call parsing, media conversion, model defaults or cache repairs.

Plan: preserve the complete message list at both Gemma processor consumers, retain tool_choice/tool_choice_name as template arguments, and test exact history/metadata plus image-slot/pixel cardinality. No prompt summaries, coercion, sampler changes, model-name allowlists or cache bypasses. The text processor becomes internal (not public) so regression tests can exercise the real prepare method.

## Baseline and acceptance

Baseline app530c2f12e8afd815c8d578fceeb73460b5443bec, binary63699d39b89c09e671928fb177746f47efccdcff2f3950638798ab4fe990fecd, engine8ba593aff16c13cf526211b8477c0a037f0122af; E2B8bit snapshot433003a1e3fbfd10819ad15179d5e3c4d02d7ea7. Full catalogs: AgentLoop40/56passed,10failed,2errored,4skipped; Frontier23/42passed,16failed,3errored. Every nonpass retained/reviewed; no universal pass claim. Private evidence: handoff-parity-2026-09-16/implementation/REQUIRED-MEDIA-HISTORY-FAILURE.md and EVAL-REVIEW-530.md.

Current implementation/test/live status: pending. Required acceptance: red/green production-processor regressions; native first image/tool result/follow-up/new image after history; named/required/auto API pairing; complete tool cards and unlocked input; exact cache/media identity and throughput; rebuilt app and full catalogs with all failures attributed. A separate prefill-error propagation issue currently converts processing failure to cancellation/empty output and then a misleading app retry; it must not be hidden by this history correction.
