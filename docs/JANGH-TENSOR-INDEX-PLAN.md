# JANGH tensor index admission

This experimental Foundation-only component validates bank locations before any
payload read or module construction. It does not enable a model factory or select
an architecture. Seven exact-source XCTest methods passed with zero failures in a standalone
Foundation-only module and test bundle. Production and test files were compiled
unchanged; the harness supplies a minimal module rather than the full package.
Full-package integration remains pending. A separate Foundation-only probe also admitted
real GLM metadata (126 projections, 252 banks) and Naive metadata (141 projections,
282 banks), with dimensions derived from architecture configuration and audited
projection definitions rather than packed shapes. Only config, index, and shard
headers were read; this was not a model load.

The caller supplies a validated `JANGHFormatContract`, an exact architecture-owned
projection dimension map, the safetensors index, and complete header summaries for
all indexed shards. Summaries exclude the safetensors `__metadata__` entry. Header
parsing and collection are separate responsibilities; this component cannot prove
that a caller supplied authentic or complete file contents. A future loader must
obtain summaries from open files and preserve file identity through mapping.

Admission checks exact custom-projection coverage, shard ownership and inventory,
relative shard filenames, bounded nonoverlapping tensor ranges, checked byte counts,
dtype alignment, U32/F16 shapes, and conflicting legacy representations. Unknown
custom tensors and missing banks fail explicitly. Ordinary tensors participate in
range-overlap checks but their architecture, dtype and index consistency remain the
generic loader's responsibility. Returned locations retain shard name, tensor name,
absolute file offset and byte count. No tensor is allocated or evaluated.

The bounded fixtures include mixed 2/3/6-bit banks, 96-channel rows, output tails,
missing/duplicate/unowned banks, unsafe paths, wrong dtype/geometry, integer overflow,
alignment, overlapping ranges, and legacy representation collisions.

Remaining work: reconcile config and sidecar declaration owners and both quantization
aliases; parse authentic index/header summaries; supply audited GLM dimensions and
canonical names; construct mapped custom modules without dense placeholders or
repacking; establish prefill and decode precision contracts; validate real model
memory, cache, multi-turn behavior and performance. Naive requires its own model
architecture. Configuration or location validation is not model execution support.
