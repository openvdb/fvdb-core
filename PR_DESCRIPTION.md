# PR: Functional Gaussian Splatting API

## Summary

This PR migrates Gaussian splatting from a monolithic C++ class to a layered
Python architecture, and decomposes the C++ op layer for compilation isolation:

- **C++ ops** (individual `.h/.cu` pairs) -- each op exposes a non-template
  wrapper function that handles device dispatch internally. No dispatch
  templates leak into headers. The monolithic `GaussianSplatOps.{h,cpp}`
  aggregation file is eliminated.
- **Python autograd** -- 6 `torch.autograd.Function` subclasses that wrap the
  C++ forward/backward dispatch, replacing the 5 C++ autograd classes.
- **`fvdb.functional.splat`** -- pure-functional pipeline stages that compose
  into custom rendering pipelines from raw tensors.
- **`GaussianSplat3d`** -- unchanged public API, now a thin wrapper that
  delegates to the functional stages.

The public API of `GaussianSplat3d` is **unchanged**. All 138 tests pass
(135 existing + 3 new). No kernel code was modified.

## What to review and how

This is a large diff (~8k lines added, ~7k removed across 60+ files), but much
of it is mechanical. Here's a guide to focus review time:

### 1. Start here: the functional pipeline stages (small, new, important)

These are the new public API surface. Each is a short, self-contained pure
function:

| File | Lines | What it does |
|------|-------|-------------|
| `fvdb/functional/splat/_projection.py` | `RawProjection` + `project_to_2d` (~80 lines) | Raw geometric projection, returns frozen dataclass |
| `fvdb/functional/splat/_opacities.py` | ~30 lines | `sigmoid(logit_opacities) * compensations` |
| `fvdb/functional/splat/_sh.py` | `prepare_render_features` (~40 lines) | SH eval or depth features based on render mode |
| `fvdb/functional/splat/_tile_intersection.py` | `TileIntersection` + `intersect_tiles` (~40 lines) | Tile-Gaussian culling |
| `fvdb/functional/splat/_rasterize.py` | `rasterize_dense` (~30 lines) | Rasterize from raw tensors |

### 2. Then: the autograd functions (medium, correctness-critical)

These replace 5 C++ autograd classes (~1,500 lines of C++) with 6 Python
`torch.autograd.Function` subclasses. The forward/backward logic is
identical -- same dispatch functions, same argument order, same accumulator
pattern.

| File | Class | C++ class it replaces |
|------|-------|-----------------------|
| `_projection.py` | `_ProjectGaussiansFn` | `ProjectGaussians` |
| `_projection.py` | `_ProjectGaussiansJaggedFn` | `ProjectGaussiansJagged` |
| `_sh.py` | `_EvalSHFn` | `EvaluateSphericalHarmonics` |
| `_rasterize.py` | `_RasterizeDenseFn` | `RasterizeGaussiansToPixels` |
| `_rasterize_sparse.py` | `_RasterizeSparseFn` | `RasterizeGaussiansSparseToPixels` |
| `_rasterize_from_world.py` | `_RasterizeFromWorldFn` | `RasterizeFromWorld3DGS` |

**Key review focus**: backward methods -- verify return count matches forward
input count, `ctx.save_for_backward` covers all needed tensors, and
contiguity/assert patterns are correct.

### 3. Then: the C++ op decomposition (mechanical, high confidence)

The monolithic `GaussianSplatOps.{h,cpp}` is eliminated. Each function is
distributed to the `.{h,cu}` file pair of the dispatch backend it corresponds
to. Headers expose only non-template wrapper functions; dispatch templates are
internal to `.cu` files. The pybind file contains zero logic -- just bindings.

**New file pairs** (pure CPU orchestration, no CUDA):
- `GaussianCameraValidation.{h,cpp}` -- camera arg validation
- `GaussianDeduplicatePixels.{h,cpp}` -- pixel deduplication for sparse render
- `GaussianProjectionPipeline.{h,cpp}` -- projection orchestration (analytic/UT
  dispatch, SH eval, tile intersection, sparse render)

**Modified `.h/.cu` pairs** (non-template wrapper added, dispatch template
internalized):
- `GaussianProjectionForward` -- `gaussianProjectionForward`
- `GaussianProjectionBackward` -- `gaussianProjectionBackward`
- `GaussianProjectionUT` -- `gaussianProjectionForwardUT`
- `GaussianProjectionJaggedForward` -- `gaussianProjectionJaggedForward`
- `GaussianProjectionJaggedBackward` -- `gaussianProjectionJaggedBackward`
- `GaussianSphericalHarmonicsForward` -- `sphericalHarmonicsForward` + `evalSphericalHarmonics`
- `GaussianSphericalHarmonicsBackward` -- `sphericalHarmonicsBackward`
- `GaussianTileIntersection` -- `gaussianTileIntersection` + `gaussianSparseTileIntersection`
- `GaussianRasterizeForward` -- `gaussianRasterizeForward` + `gaussianSparseRasterizeForward` + `renderCropFromProjected`
- `GaussianRasterizeBackward` -- `gaussianRasterizeBackward` + `gaussianSparseRasterizeBackward`
- `GaussianRasterizeFromWorldForward` -- `gaussianRasterizeFromWorldForward` + `rasterizeFromWorld`
- `GaussianRasterizeFromWorldBackward` -- `gaussianRasterizeFromWorldBackward`
- `GaussianRasterizeNumContributingGaussians` -- wrappers + `renderNumContributing` + `sparseRenderNumContributing`
- `GaussianRasterizeContributingGaussianIds` -- wrappers + `renderContributingIds` + `sparseRenderContributingIds`
- `GaussianRasterizeTopContributingGaussianIds` -- wrappers
- `GaussianMCMCRelocation` -- `gaussianRelocation`
- `GaussianMCMCAddNoise` -- `gaussianMCMCAddNoise`
- `GaussianComputeNanInfMask` -- `gaussianNanInfMask`

**No kernel code was modified.** The `.cu` kernel implementations are untouched.

### 4. Then: `gaussian_splatting.py` (large, but now just orchestration)

The Python `GaussianSplat3d` class grew from ~2,800 to ~4,200 lines because
it absorbed the orchestration that was previously in C++. However, the core
render pipeline (`_project_decomposed`) is now just ~15 lines composing the
functional stages. The rest is validation, accumulator management, and public
method boilerplate.

### 5. Skip or skim: deleted C++ code

- `GaussianSplat3d.{h,cpp}` -- 3,200 lines deleted (the monolithic class)
- `GaussianSplatBinding.cpp` -- 560 lines deleted (its pybind binding)
- `GaussianSplatOps.{h,cpp}` -- 1,430 lines deleted (the aggregation layer)
- 5 C++ autograd files -- 1,500 lines deleted
- `GaussianSplat3dCameraApiTest.cpp` -- 514 lines deleted (covered by Python tests)

These deletions are the payoff. No review needed beyond confirming nothing
was accidentally dropped.

### 6. Type stubs and docs

- `_fvdb_cpp.pyi` -- updated to reflect new pybind surface (deleted stale
  C++ `GaussianSplat3d` class, added `RenderSettings`, `RenderMode`,
  `SparseProjectedGaussianSplats`, ~30 `gsplat_*` function stubs).
- `functional_splat.rst` -- Sphinx docs for the functional API with pipeline
  diagram and usage example.
- All files pass pyright with 0 new errors (3 pre-existing in JaggedTensor stubs).

## Test plan

- [x] All 135 existing tests pass (no regressions)
- [x] All 36 C++ gtests pass
- [x] `test_functional_forward_matches_oo` -- functional stages produce
      pixel-identical output to `GaussianSplat3d.render_images`
- [x] `test_functional_backward_matches_oo` -- all 6 parameter gradients
      match between functional and OO paths
- [x] `test_functional_training_loop` -- 5 Adam steps, loss decreases,
      all gradients finite and non-zero
- [x] pyright clean (0 new errors, `basic` mode)
- [x] black + clang-format clean

## Breaking changes

**None.** The public API of `GaussianSplat3d` is identical to `main`. The
functional API in `fvdb.functional.splat` is entirely new (additive).
