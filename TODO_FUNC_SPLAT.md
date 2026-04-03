# Functional Gaussian Splatting API - Progress Tracker

## Build Instructions
```bash
conda activate fvdb
./build.sh install editor_skip -e          # standard build
./build.sh install gtests editor_skip -e   # with gtests
```

## Notes
- **kPrivateUse1 / Universal Memory / MultiGPU:** The splatting code uses these intentionally. Do not break them, but do not test them (WSL can't use universal memory).
- **No kernel changes:** CUDA kernel bodies are never touched.
- **Tests must not change:** `pytest tests/unit/test_gaussian_splat_3d.py` must pass after every milestone.
- **Accumulator mutability:** The projection backward CUDA kernel mutates three accumulator tensors in-place (gradient norms, max 2D radii, step counts) via atomicAdd. These are used by Gaussian densification (split/clone/prune). The **forward** pass is pure. In the C++ free functions (`GaussianSplatOps.h`), accumulators are passed as mutable `std::optional<torch::Tensor>&` refs. The pybind layer (`GaussianSplatOps.cpp`) wraps these in lambdas that hide the mutability, presenting a **strictly functional interface** to Python. When autograd moves to Python (M5-7), the Python `GaussianSplat3d` class will own the accumulators and pass them through to the C++ backward dispatch.

---

## Milestones

### Milestone 1: Extract Non-Differentiable Helpers into Free Functions -- DONE
- [x] Move `validateCameraProjectionArgs()`, `resolveProjectionMethod()`, `usesOpenCVDistortion()` to shared header
- [x] Move `checkState()` to shared header (as `checkGaussianState()`)
- [x] Move `deduplicatePixels()` to shared header
- [x] `GaussianSplat3d.cpp` calls the new free functions via `gsplat::` namespace alias
- [x] Update `CMakeLists.txt`
- [x] Update `DeduplicatePixelsTest.cpp` to use new namespace
- [x] Build passes, all 135 tests pass

**New files:** `src/fvdb/detail/ops/gsplat/GaussianSplatOps.h`, `src/fvdb/detail/ops/gsplat/GaussianSplatOps.cpp`

### Milestone 2: Extract Projection Pipeline into Free Functions -- DONE
- [x] `gsplat::evalSphericalHarmonics(...)` free function
- [x] `gsplat::projectGaussiansAnalytic(...)` free function (with accumulator handling)
- [x] `gsplat::projectGaussiansUT(...)` free function (non-differentiable UT path)
- [x] `gsplat::projectGaussiansForCamera(...)` free function (dispatches between analytic/UT)
- [x] Helper `computeRenderQuantity()` and `prepareAccumulators()` factored out
- [x] Tile intersection integrated into projection functions (no separate wrapper needed)
- [x] `evalSphericalHarmonicsImpl`, `projectGaussiansImpl`, `projectGaussiansForCameraImpl` now thin wrappers
- [x] Build passes, all 135 tests pass

### Milestone 3: Extract Rendering, Queries, and Utilities into Free Functions -- DONE
- [x] Dense rasterization: `renderCropFromProjected()` free function
- [x] From-world rasterization: `rasterizeFromWorld()` free function (used by all 3 from-world methods)
- [x] Sparse projection: `sparseProjectGaussiansAnalytic()`, `sparseProjectGaussiansUT()`, `sparseProjectGaussiansForCamera()`
- [x] Sparse rendering: `sparseRender()` free function
- [x] Query ops: `renderNumContributing()`, `sparseRenderNumContributing()`, `renderContributingIds()`, `sparseRenderContributingIds()`
- [x] All class impl methods are now thin wrappers
- [x] MCMC ops (`relocateGaussians`, `addNoiseToMeans`) already thin dispatches -- left as-is
- [x] Indexing ops and PLY I/O -- left as class methods (pure tensor/IO ops, will move to Python in M7)
- [x] Build passes, all 135 tests pass

### Milestone 4: Expose Free-Function Ops via pybind11 -- DONE
- [x] New `src/python/GaussianSplatOps.cpp` with `bind_gaussian_splat_ops()`
- [x] All free functions exposed as `_fvdb_cpp.gsplat_*` pure functional (no mutable accumulator refs)
- [x] Lambda wrappers for functions with accumulator args (accumulators hidden, not exposed to Python)
- [x] `RenderSettings`, `RenderMode`, `SparseProjectedGaussianSplats` bound as Python types
- [x] Update `Bindings.cpp` (declare + call `bind_gaussian_splat_ops`)
- [x] Update `CMakeLists.txt` (add `GaussianSplatOps.cpp` to `FVDB_BINDINGS_CPP_FILES`)
- [x] All 135 tests pass, all new bindings importable from `_fvdb_cpp`

### Milestone 5: Create `fvdb.functional.splat` - Projection and SH -- DONE
- [x] `fvdb/functional/splat/` directory and `__init__.py`
- [x] `_projection.py` with `project_gaussians()` and `project_gaussians_for_camera()` (delegates to pybind, C++ autograd still handles differentiation)
- [x] `_sh.py` with `evaluate_spherical_harmonics()` (delegates to pybind)
- [x] `_tile_intersection.py` with `build_render_settings()` helper
- [x] `fvdb/functional/__init__.py` updated to expose `splat` submodule
- [x] Build passes, all 135 tests pass, all imports verified

### Milestone 6: Create Python Rasterization and Pipeline Functions -- DONE
- [x] `_rasterize.py` - `rasterize_from_projected()` (dense)
- [x] `_rasterize_from_world.py` - `rasterize_from_world()` (world-space geometry grads)
- [x] `_rasterize_sparse.py` - `sparse_render()` (sparse pixel rendering)
- [x] `_queries.py` - `render_num_contributing_gaussians`, `render_contributing_gaussian_ids` + sparse variants
- [x] `_render.py` - 6 composite pipeline functions: `render_images`, `render_depths`, `render_images_and_depths`, + 3 from-world variants
- [x] `_mcmc.py` - placeholder (MCMC ops still on class, will become free functions in M9)
- [x] `__init__.py` updated with all 16 public exports
- [x] Build passes, all 135 tests pass

### Milestone 7: Rewire `GaussianSplat3d` Python Class to Delegate to Functional
- [ ] Store tensors directly as Python attributes
- [ ] `ProjectedGaussianSplats` becomes pure-Python dataclass
- [ ] All render methods delegate to `fvdb.functional.splat`
- [ ] Indexing, cat, detach, to become pure-Python
- [ ] State dict and PLY I/O delegate to functional
- [ ] Remove dependency on `GaussianSplat3dCpp`
- [ ] **CRITICAL GATE:** All tests pass with zero test changes

### Milestone 8: Remove C++ Autograd Classes for Gaussian Ops
- [ ] Delete `GaussianProjection.h/.cpp`
- [ ] Delete `GaussianRasterize.h/.cpp`
- [ ] Delete `GaussianRasterizeFromWorld.h/.cpp`
- [ ] Delete `GaussianRasterizeSparse.h/.cpp`
- [ ] Delete `EvaluateSphericalHarmonics.h/.cpp`
- [ ] Update `CMakeLists.txt`
- [ ] Build passes, all tests pass

### Milestone 9: Remove C++ `GaussianSplat3d` Class
- [ ] Delete `GaussianSplat3d.h` and `.cpp`
- [ ] Move enums and free functions from `GaussianSplatBinding.cpp`
- [ ] Delete `GaussianSplatBinding.cpp`
- [ ] Move `gaussianRenderJagged` to own file
- [ ] Refactor `GaussianPlyIO` to not depend on class
- [ ] Update `Bindings.cpp`, `CMakeLists.txt`, `_fvdb_cpp.pyi`
- [ ] Build passes, all tests pass

### Milestone 10: Public API Polish and Documentation
- [ ] Complete `fvdb/functional/splat/__init__.py` with `__all__`
- [ ] Update `fvdb/functional/__init__.py` exports
- [ ] Update `fvdb/__init__.py` re-exports
- [ ] Add `docs/api/functional_splat.rst`
- [ ] Update `_fvdb_cpp.pyi`
- [ ] Full test suite passes
