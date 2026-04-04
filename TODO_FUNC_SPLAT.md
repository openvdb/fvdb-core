# Functional Gaussian Splatting API - Progress Tracker -- COMPLETE

## CRITICAL: Build Environment

**Always activate the conda environment first:**
```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate fvdb
```

**Always use the build script — NEVER call cmake, ninja, or pip install directly:**
```bash
./build.sh install editor_skip -e          # standard editable build (fastest)
./build.sh install gtests editor_skip -e   # with C++ gtests
./build.sh install benchmarks editor_skip -e  # with benchmarks
```

The build script handles CUDA detection, architecture flags, parallel build settings,
memory constraints, and proper pip invocation. Calling cmake/ninja/pip directly WILL
produce broken builds.

**Run tests from the `tests/` directory** (to use the installed package, not the source tree):
```bash
cd tests && pytest unit/test_gaussian_splat_3d.py -v
```

---

## Context
- **Branch:** `feature/functional-gaussian` on `blackencino/fvdb-core` fork
- **Predecessor:** PR #582 did the same transformation for GridBatch ops. The pattern in
  `fvdb/functional/_transforms.py` (e.g., `_VoxelToWorldFn`) is the reference for Python
  autograd wrapping pybind fwd/bwd dispatch functions.
- **Issue:** https://github.com/openvdb/fvdb-core/issues/459
- **Commit style:** `Short description (milestone N/10)` with `Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>`

## Notes
- **kPrivateUse1 / Universal Memory / MultiGPU:** The splatting code uses these intentionally. Do not break them, but do not test them (WSL can't use universal memory).
- **No kernel changes:** CUDA kernel bodies are never touched.
- **Tests must not change:** `pytest tests/unit/test_gaussian_splat_3d.py` must pass after every milestone (135 tests).

---

## Architecture Summary (after M9)

### What was done (Milestones 1-9)

The C++ `GaussianSplat3d` class (~3,200 lines), its pybind binding (~560 lines), and 5 C++
`torch::autograd::Function` subclasses (~1,500 lines) have been **fully removed**. All
Gaussian splatting functionality is now:

1. **CUDA dispatch functions** (unchanged) — forward/backward kernels in `src/fvdb/detail/ops/gsplat/*.cu`
2. **C++ free functions** in `GaussianSplatOps.cpp` — call dispatch directly (non-differentiable at C++ level)
3. **Pybind bindings** in `src/python/GaussianSplatOps.cpp` — expose both high-level ops and raw fwd/bwd dispatch
4. **Python autograd** in `fvdb/functional/splat/` — 6 `torch.autograd.Function` subclasses:
   - `_ProjectGaussiansFn` (fresh-zeros accumulator pattern)
   - `_ProjectGaussiansJaggedFn`
   - `_EvalSHFn`
   - `_RasterizeDenseFn`
   - `_RasterizeSparseFn`
   - `_RasterizeFromWorldFn`
5. **Python `GaussianSplat3d`** in `fvdb/gaussian_splatting.py` — pure Python class that composes the autograd functions

### Key design: Fresh-zeros accumulator pattern

The projection backward CUDA kernel fuses 3 accumulator updates (gradient norms, max 2D radii,
step counts) via `atomicAdd`/`atomicMax`. Rather than passing live accumulators to the kernel
(hidden side effect), each backward call creates fresh zero tensors, passes them to the kernel,
then Python explicitly accumulates deltas into persistent state. See `_ProjectGaussiansFn` in
`fvdb/functional/splat/_projection.py`.

### Key files

| Purpose | File |
|---|---|
| Python GaussianSplat3d class | `fvdb/gaussian_splatting.py` |
| Python autograd Functions | `fvdb/functional/splat/_projection.py`, `_sh.py`, `_rasterize.py`, `_rasterize_sparse.py`, `_rasterize_from_world.py` |
| Python render pipeline helpers | `fvdb/functional/splat/_render.py`, `_tile_intersection.py`, `_queries.py` |
| Standalone types (C++) | `src/fvdb/detail/ops/gsplat/GaussianProjectionTypes.h` |
| C++ free function ops | `src/fvdb/detail/ops/gsplat/GaussianSplatOps.h/.cpp` |
| Pybind bindings (all gsplat) | `src/python/GaussianSplatOps.cpp` |
| CUDA dispatch (forward) | `src/fvdb/detail/ops/gsplat/Gaussian*Forward.{h,cu}` |
| CUDA dispatch (backward) | `src/fvdb/detail/ops/gsplat/Gaussian*Backward.{h,cu}` |
| PLY I/O (raw tensors) | `src/fvdb/detail/io/GaussianPlyIO.{h,cpp}` |

---

## Milestones

### Milestone 1: Extract Non-Differentiable Helpers into Free Functions -- DONE
### Milestone 2: Extract Projection Pipeline into Free Functions -- DONE
### Milestone 3: Extract Rendering, Queries, and Utilities into Free Functions -- DONE
### Milestone 4: Expose Free-Function Ops via pybind11 -- DONE
### Milestone 5: Create `fvdb.functional.splat` - Projection and SH -- DONE
### Milestone 6: Create Python Rasterization and Pipeline Functions -- DONE
### Milestone 7: Rewire `GaussianSplat3d` Python Class to Delegate to Functional -- DONE
### Milestone 8: Eliminate GaussianSplat3dCpp from PLY I/O and MCMC -- DONE

### Milestone 8.5: Move C++ Autograd to Python -- DONE
- [x] Exposed 12 raw fwd/bwd dispatch functions + tile intersection via pybind
- [x] Created 6 Python `torch.autograd.Function` subclasses with fresh-zeros accumulator pattern
- [x] Decomposed GaussianSplat3d render pipeline (dense, sparse, jagged) to use Python autograd
- [x] Updated C++ free functions to call dispatch directly (non-differentiable)
- [x] Deleted 5 C++ autograd file pairs (10 files, ~1,500 lines)
- [x] Rewrote `fvdb.gaussian_render_jagged` and `fvdb.evaluate_spherical_harmonics` in Python
- [x] All 135 tests pass

### Milestone 9: Remove C++ `GaussianSplat3d` Class -- DONE
- [x] Extracted `ProjectedGaussianSplats`, `SparseProjectedGaussianSplats`, `PlyMetadataTypes` into standalone `GaussianProjectionTypes.h`
- [x] Moved enum/type pybind bindings from `GaussianSplatBinding.cpp` to `GaussianSplatOps.cpp`
- [x] Refactored PLY I/O to accept/return raw tensors (no class dependency)
- [x] Decoupled viewer from the class (accepts raw tensors)
- [x] Eliminated `_make_cpp_impl` — migrated 3 query ops to pybind free functions
- [x] Deleted `GaussianSplat3d.h` (1,553 lines), `GaussianSplat3d.cpp` (1,674 lines), `GaussianSplatBinding.cpp` (460 lines)
- [x] Deleted `GaussianSplat3dCameraApiTest.cpp` (covered by Python tests)
- [x] All 135 tests pass

### Milestone 10: Public API Polish and Documentation -- DONE
- [x] Complete `fvdb/functional/splat/__init__.py` with `__all__` (export all public functions)
- [x] Update `fvdb/functional/__init__.py` exports
- [x] Update `fvdb/__init__.py` re-exports
- [x] Update `_fvdb_cpp.pyi` type stubs (removed stale C++ GaussianSplat3d class, added RenderSettings, RenderMode, SparseProjectedGaussianSplats, ~30 gsplat_* function stubs, updated Viewer signature)
- [x] Add `docs/api/functional_splat.rst` documentation
- [x] Enforce `near`/`far` param naming from GaussianSplat3d into functional API and pybind dispatch
- [x] Run `black` and `clang-format` on all changed files
- [x] All 135 tests pass
