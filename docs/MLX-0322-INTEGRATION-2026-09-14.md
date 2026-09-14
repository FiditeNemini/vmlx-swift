# MLX 0.32.2 integration checkpoint — PARTIAL

NOW: Integrate pinned core0.32.2, the matching C ABI and Swift compatibility in
`feat/mlx0322-integration`. Preserve the JANG/mmap fork and the existing Qwen
AR optimizations. No application speed claim yet.
DO NOT: Release, change native sampling/precision, replace the fork with stock
MLX, or mix a larger prefill chunk into the dependency-control measurement.
BATCH OWNER: Core/C/Swift compatibility, then local development-app evidence.
NEXT: Checkpoint the tested sources, repin all six Osaurus locations and rebuild
the local development bundle before promotion.

## Source lineage

- Engine starting point: `9e48d907e45f3d721b3faa42d9f6500bab16224b`.
- Core fork starting point: `be526f81f5534b3447d4882871cac279af2ad14a`.
- Core release integrated: `1f8e74e3f12f31365464a6867c6579f0e9b29d85`
  ([MLX0.32.2](https://github.com/ml-explore/mlx/releases/tag/v0.32.2)).
- C ABI starting point: `7bede8f1491384bb580f87d1d510f80cbd53660d`;
  integration target `c74db5307cc8ce122f48d97ef951b30578674e7f`.
- Integrated core: `1c4fcf773c00aa544bcddade6281fbd311a599c7`;
  C ABI: `2d783ac38713458eae2067ffff9ef8ebbff2ec70`.
- Upstream Swift comparison: [#450](https://github.com/ml-explore/mlx-swift/pull/450).
  Keep this engine's Swift-tools6.1 manifest and integrated runtime products.

## Required adaptations

| Boundary | Adaptation |
|---|---|
| Quantized kernels | Preserve affine1 inference packing, mixed q4/q8 metadata, raw-F32 q6 and Metal-only admission. Preserve the32-lane exact q6 reduction for eligible promoted verifier rows; leave other wide and matrix paths intact. |
| mmap | Retain the fork's regions/advice, excluded keys, safe unaligned handling and array-owned mappings. No model-weight rewrite. |
| Stream lifetime | Use upstream cross-thread stream handles under the Swift evaluation lock; native defaults are now thread-local. Restore the target device's previous default. |
| Compiler cache | Acquire evalLock before the recursive instance lock. Record stable identities and weak handles for every native worker cache used, then erase the function from those caches on destruction. Preserve the compile trust policy. |
| Custom kernels | Adopt source/options-hashed libraries; retain bound buffers through command completion and retired library/pipeline owners. Preserve output shapes in both common Metal and CUDA factories. CUDA is not execution-qualified here. |
| C/Swift surface | Adapt cumulative-axis, FFT normalization, SDPA forceFused defaultfalse, median/trace and Data SEEK_END. Update distributed-group output-parameter ABI with owned typed handles; multi-host execution remains untested. |
| Generated sources | Regenerate copied headers, embedded JIT sources and fresh AOT shaders from these sources. Discover headers with the installed Metal toolchain, but keep the macOS14 app target. Exclude optional newer-Metal fast-fence AOT; do not enable MLX_METAL_FAST_SYNCH=1 with it. |
| Alternate builds | CMake consumes the same pinned core/C ABI, not stock mlx-c0.6.0 or the removed stream shim. Framework membership/public headers regenerate from actual SwiftPM sources, with the framework owning its distributed C entry points directly. Linux CMake CI initializes the pinned submodules. |

## Live numerical evidence so far

Private evidence root:
`/Users/eric/vmlx-private-evidence/mtp-swift-2026-09-04/logs`.

`SWIFTTEST_MLX0322IntegrationTestsF0914__041253.log` records a freshly built
test executable SHA256
`ec5377ef1b3e9fb9d63e6eb28a34789293d4341b555b0f3907d55369d9a8b9d9`
and metallib SHA256
`02ea075fab6e847ba1f5cc8c410657506b63309257a24efbc7eb6f69380dbe6f`.
Its51 XCTest cases had zero failures. The new384-case quantization matrix
ran in two modes: default backend with an explicit reduced-precision budget
for48eligible gathered-M33 cases, and strict2e-4 throughout with
`MLX_ENABLE_TF32=0`. The largest default-mode absolute difference was0.0007005632.
App TF32 policy is unchanged; see the [MLX precision contract](https://ml-explore.github.io/mlx/build/html/usage/precision.html).

Existing exact q6 equality tests were not loosened. RMSNorm, shared-mask
attention partitions, mmap GPU computation, FFT/Data, cumulative scans,
custom-kernel identity/shapes and cross-worker compiler-cache deletion have
focused execution receipts. This is not full model quality or speed proof.

The extended F selection had one stale group128 rejection test, inherited from
before production support in c6ae0e85/#446. The integration corrects that
assertion to group256 and adds positive group128 numerical/row-isolation cases.
`SWIFTTEST_MLX0322IntegrationTestsG0914__042059.log` ended exit0. It reran the
core matrices and recorded zero issues across the21-test HC/GDN/PLE/expert/rotary
and24-test cache selections. Exact checks included GDN336,HC768,rotary72 and
QSA sequence180. Both added group128 layouts had zero batched-versus-single-row
difference for rows2/3/4. Tiny Ling3 disk restore37+suffix5 matched one-shot42;
this is not full-Qwen restore proof. Real installed-model PLE tests remain pending.

Final G test executable SHA256:
`d495c5a622ad6acdff1f7a451b592985a3285c7b46f028b26f2ab2e67f42ffda`.
The fresh metallib hash is unchanged. The bounded supervisor recorded4.20GiB
peak tracked footprint, flat0.49GiB swap and zero owned survivors.

## Remaining acceptance gates

1. Retain the bounded extended test receipt above; no test driver remains active.
2. Source-checkpoint and consume exact core/C/engine revisions in all six app pins.
   No upstream submission or public release is intended.
3. Fresh local optimized development build, actual dependency checkout identities,
   binary UUID/hash, metallib hash and new ABI symbol presence. Observe actual
   compiler thread count; xcodebuild jobs alone did not bound previous WMO memory.
4. Local app AR/MTP-Off controls using identical bundles, tokens and native sampling:
   short and long contexts, repeated runs,1s/5s min/median/peak and longest gaps,
   footprint/read pressure, coherent natural-stop multi-turn/media/cache rows.
   Account individually for JANG1L/2L/4S/4M/6S. Do not transfer Python timings.
5. Only then separately qualify masked D256 attention and bounded larger prefill
   chunks against512, preserving cache/frontier/media positions. Sustained MTP
   remains a subsequent unit, not proof furnished by this dependency update.
