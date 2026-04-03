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

### Milestone 2: Extract Projection Pipeline into Free Functions
- [ ] `gsplat::projectGaussiansAnalytic(...)` free function
- [ ] `gsplat::projectGaussiansUT(...)` free function
- [ ] `gsplat::evalSphericalHarmonics(...)` free function
- [ ] `gsplat::tileIntersection(...)` free function
- [ ] Class methods become thin wrappers
- [ ] Build passes, all tests pass

### Milestone 3: Extract Rendering, Queries, and Utilities into Free Functions
- [ ] Dense rendering free function
- [ ] From-world rendering free function
- [ ] Sparse projection + rendering free functions
- [ ] Query ops (contributing gaussians) free functions
- [ ] MCMC ops free functions
- [ ] Indexing ops free functions
- [ ] PLY I/O refactored to accept raw tensors
- [ ] Build passes, all tests pass

### Milestone 4: Expose Free-Function Ops via pybind11
- [ ] New `GaussianSplatOps.cpp` binding file
- [ ] Expose fwd/bwd dispatch functions separately
- [ ] Update `Bindings.cpp` and `CMakeLists.txt`
- [ ] Update `_fvdb_cpp.pyi`
- [ ] Build passes, all tests pass

### Milestone 5: Create `fvdb.functional.splat` - Projection and SH Autograd
- [ ] `fvdb/functional/splat/` directory and `__init__.py`
- [ ] `_projection.py` with `_ProjectGaussiansFn` autograd
- [ ] `_projection_ut.py` (non-differentiable)
- [ ] `_sh.py` with `_EvalSHFn` autograd
- [ ] `_tile_intersection.py` wrappers
- [ ] Build passes, all tests pass

### Milestone 6: Create Python Autograd for Rasterization
- [ ] `_rasterize.py` - dense rasterization autograd
- [ ] `_rasterize_from_world.py` - world-space rasterization autograd
- [ ] `_rasterize_sparse.py` - sparse rasterization autograd
- [ ] `_sparse_utils.py` - dedup, sparse info
- [ ] `_queries.py` - contributing gaussian queries
- [ ] `_mcmc.py` - relocate, add noise
- [ ] `_io.py` - PLY I/O
- [ ] Composite pipeline functions in `__init__.py`
- [ ] Build passes, all tests pass

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
