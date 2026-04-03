# Functional Gaussian Splatting API - Progress Tracker

## Build & Test Instructions
```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate fvdb
./build.sh install editor_skip -e          # standard build
./build.sh install gtests editor_skip -e   # with gtests
pytest tests/unit/test_gaussian_splat_3d.py -v   # must pass after every milestone
```

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

### Milestone 7: Rewire `GaussianSplat3d` Python Class to Delegate to Functional -- DONE
- [x] Store tensors directly as Python attributes (`self._means`, etc.)
- [x] `ProjectedGaussianSplats` still wraps C++ `ProjectedGaussianSplatsCpp` (returned by pybind)
- [x] All render methods delegate to pybind free functions (`gsplat_project_gaussians_for_camera_with_accum`, `gsplat_render_crop_from_projected`, etc.)
- [x] Added `gsplat_project_gaussians_for_camera_with_accum` pybind binding for accumulator support
- [x] Indexing uses direct tensor slicing/index_put (out-of-place for grad safety)
- [x] cat, detach, to, state_dict are pure-Python tensor operations
- [x] PLY I/O and MCMC still use temporary `GaussianSplat3dCpp` wrappers (will be removed in M9)
- [x] **CRITICAL GATE PASSED:** All 135 tests pass with zero test changes

### Milestone 8: Eliminate GaussianSplat3dCpp from PLY I/O and MCMC -- DONE
**Scope change:** C++ autograd classes remain (they're still called by the free functions).
Instead, this milestone exposed MCMC and PLY I/O as pybind free functions, eliminating
`GaussianSplat3dCpp` usage from PLY I/O, MCMC, and `render_num_contributing_gaussians`.
The `_make_cpp_impl` helper remains only for 3 sparse query ops.
- [x] Added `gsplat_relocate_gaussians` pybind binding (thin dispatch)
- [x] Added `gsplat_add_noise_to_means` pybind binding (thin dispatch, in-place)
- [x] Added `gsplat_save_ply` pybind binding (creates temp C++ object internally)
- [x] Added `gsplat_load_ply` pybind binding (returns raw tensors + metadata)
- [x] `from_ply` uses `gsplat_load_ply` directly (no `GaussianSplat3dCpp`)
- [x] `save_ply` uses `gsplat_save_ply` directly
- [x] `relocate_gaussians` uses `gsplat_relocate_gaussians` directly
- [x] `add_noise_to_means` uses `gsplat_add_noise_to_means` directly
- [x] `render_num_contributing_gaussians` uses `_project_with_accum` + `gsplat_render_num_contributing`
- [x] Build passes, all 135 tests pass

### Milestone 8.5: Move C++ Autograd to Python (NOT YET DONE)

**Why this matters:** The primary goal of this refactor is to reduce C++ compilation surface.
The 5 C++ `torch::autograd::Function` subclasses are among the most compilation-heavy files
because they include torch autograd headers. Moving autograd to Python eliminates them from C++.

**Current state:** The C++ free functions in `GaussianSplatOps.cpp` still call the C++ autograd
classes internally. The call chain is:

```
Python GaussianSplat3d.render_images(...)
  -> _C.gsplat_project_gaussians_for_camera_with_accum(...)  [pybind lambda]
    -> ops::projectGaussiansForCamera(...)                    [C++ free function]
      -> ops::projectGaussiansAnalytic(...)                   [C++ free function]
        -> autograd::ProjectGaussians::apply(...)             [C++ autograd - STILL HERE]
          -> dispatchGaussianProjectionForward<DeviceTag>     [CUDA kernel dispatch]
        -> autograd::EvaluateSphericalHarmonics::apply(...)   [C++ autograd - STILL HERE]
          -> dispatchSphericalHarmonicsForward(...)            [CUDA kernel dispatch]
  -> _C.gsplat_render_crop_from_projected(...)                [pybind]
    -> ops::renderCropFromProjected(...)                      [C++ free function]
      -> autograd::RasterizeGaussiansToPixels::apply(...)     [C++ autograd - STILL HERE]
        -> dispatchGaussianRasterizeForward(...)               [CUDA kernel dispatch]
```

**Target state:** The C++ autograd wrappers are removed. The C++ free functions call the
dispatch (kernel) functions directly for forward only. Python `torch.autograd.Function`
subclasses handle saving context and calling backward kernels.

```
Python GaussianSplat3d.render_images(...)
  -> Python _ProjectGaussiansFn.apply(...)                    [Python autograd]
    -> forward: _C.gsplat_projection_fwd(...)                 [pybind -> dispatch forward kernel]
    -> backward: _C.gsplat_projection_bwd(...)                [pybind -> dispatch backward kernel]
  -> Python _EvalSHFn.apply(...)                              [Python autograd]
    -> forward: _C.gsplat_sh_eval_fwd(...)                    [pybind -> dispatch forward kernel]
    -> backward: _C.gsplat_sh_eval_bwd(...)                   [pybind -> dispatch backward kernel]
  -> Python _RasterizeDenseFn.apply(...)                      [Python autograd]
    -> forward: _C.gsplat_rasterize_fwd(...)                  [pybind -> dispatch forward kernel]
    -> backward: _C.gsplat_rasterize_bwd(...)                 [pybind -> dispatch backward kernel]
```

**What needs to happen:**

1. **Expose raw forward/backward dispatch functions via pybind.** The dispatch functions
   already exist as template functions in the `fvdb::detail::ops` namespace. They need to
   be wrapped in non-template functions (with `FVDB_DISPATCH_KERNEL` for device dispatch)
   and exposed via pybind. The functions to expose are:

   | C++ dispatch function | Header | Purpose |
   |---|---|---|
   | `dispatchGaussianProjectionForward<D>` | `GaussianProjectionForward.h` | Projection fwd |
   | `dispatchGaussianProjectionBackward<D>` | `GaussianProjectionBackward.h` | Projection bwd (includes accum mutation) |
   | `dispatchSphericalHarmonicsForward` | `GaussianSphericalHarmonicsForward.h` | SH eval fwd |
   | `dispatchSphericalHarmonicsBackward` | `GaussianSphericalHarmonicsBackward.h` | SH eval bwd |
   | `dispatchGaussianRasterizeForward<D>` | `GaussianRasterizeForward.h` | Dense rasterize fwd |
   | `dispatchGaussianRasterizeBackward<D>` | `GaussianRasterizeBackward.h` | Dense rasterize bwd |
   | `dispatchGaussianSparseRasterizeForward<D>` | `GaussianRasterizeForward.h` | Sparse rasterize fwd |
   | `dispatchGaussianSparseRasterizeBackward<D>` | `GaussianRasterizeBackward.h` | Sparse rasterize bwd |
   | `dispatchGaussianRasterizeFromWorld3DGSForward<D>` | `GaussianRasterizeFromWorldForward.h` | From-world fwd |
   | `dispatchGaussianRasterizeFromWorld3DGSBackward<D>` | `GaussianRasterizeFromWorldBackward.h` | From-world bwd |
   | `dispatchGaussianProjectionJaggedForward<D>` | `GaussianProjectionJaggedForward.h` | Jagged projection fwd |
   | `dispatchGaussianProjectionJaggedBackward<D>` | `GaussianProjectionJaggedBackward.h` | Jagged projection bwd |

   Each needs a non-template wrapper that does `FVDB_DISPATCH_KERNEL(device, ...)` and a
   pybind `m.def(...)` in `GaussianSplatOps.cpp`.

2. **Create Python `torch.autograd.Function` subclasses** in `fvdb/functional/splat/`.
   These mirror the C++ autograd classes. For each:
   - `forward()` calls the pybind forward dispatch, saves tensors for backward in `ctx`
   - `backward()` calls the pybind backward dispatch

   The key classes are:
   - `_ProjectGaussiansFn` — replaces `autograd::ProjectGaussians`
   - `_EvalSHFn` — replaces `autograd::EvaluateSphericalHarmonics`
   - `_RasterizeDenseFn` — replaces `autograd::RasterizeGaussiansToPixels`
   - `_RasterizeSparseFn` — replaces `autograd::RasterizeGaussiansToPixelsSparse`
   - `_RasterizeFromWorldFn` — replaces `autograd::RasterizeGaussiansToPixelsFromWorld3DGS`
   - `_ProjectGaussiansJaggedFn` — replaces `autograd::ProjectGaussiansJagged`

3. **Handle the accumulator mutation in Python.** The projection backward kernel
   (`dispatchGaussianProjectionBackward`) accepts 3 optional accumulator tensors as the last
   3 parameters and atomicAdds into them during backward. In the Python autograd Function:
   - Pass the accumulator tensors as inputs to `forward()` (they're not differentiable)
   - Save them in `ctx` alongside the other saved tensors
   - In `backward()`, pass them to the backward dispatch function
   - The kernel mutates them in-place (same behavior as now, just the wrapper is in Python)
   - Return `None` gradients for the accumulator inputs

4. **Update `GaussianSplatOps.cpp` (the C++ free functions)** to call the dispatch functions
   directly instead of `autograd::*::apply()`. This makes them non-differentiable at the
   C++ level — differentiation is handled by the Python autograd Functions wrapping the
   raw dispatch calls.

5. **Update `GaussianSplatOps.cpp` (the pybind file)** to replace the high-level bindings
   with raw forward/backward bindings. The Python functional layer (`fvdb/functional/splat/`)
   then calls these raw bindings from within `torch.autograd.Function.forward()` and
   `.backward()`.

6. **Update `fvdb/gaussian_splatting.py`** to use the new Python autograd-based functional
   functions instead of the pybind high-level functions.

7. **Delete the C++ autograd files:**
   - `src/fvdb/detail/autograd/GaussianProjection.h` and `.cpp`
   - `src/fvdb/detail/autograd/GaussianRasterize.h` and `.cpp`
   - `src/fvdb/detail/autograd/GaussianRasterizeFromWorld.h` and `.cpp`
   - `src/fvdb/detail/autograd/GaussianRasterizeSparse.h` and `.cpp`
   - `src/fvdb/detail/autograd/EvaluateSphericalHarmonics.h` and `.cpp`
   Keep `VolumeRender.h/.cpp` and `JaggedReduce.h/.cpp` (not gsplat-specific).
   Update `src/CMakeLists.txt`.

8. **Handle `gaussianRenderJagged`** which is a standalone function in
   `GaussianSplat3d.cpp` that also calls `autograd::ProjectGaussians::apply`,
   `autograd::ProjectGaussiansJagged::apply`, `autograd::EvaluateSphericalHarmonics::apply`,
   and `autograd::RasterizeGaussiansToPixels::apply`. This function needs to either:
   - Be moved to Python using the new Python autograd functions, OR
   - Be updated to call the dispatch functions directly (losing differentiability in C++)
     and have a Python wrapper that provides autograd

**Reference implementation:** Look at how the GridBatch refactor (PR #582) handled this for
the grid ops. The pattern in `fvdb/functional/_transforms.py` shows `_VoxelToWorldFn`
calling `_fvdb_cpp.voxel_to_world()` (forward) and `_fvdb_cpp.voxel_to_world_bwd()`
(backward) as separate pybind functions.

**Key files to read:**
- `src/fvdb/detail/autograd/GaussianProjection.cpp` — the C++ autograd to replicate in Python (338 lines, already read earlier; shows exactly what's saved in ctx and how backward calls the dispatch)
- `src/fvdb/detail/autograd/GaussianRasterize.cpp` — dense rasterize autograd
- `src/fvdb/detail/autograd/GaussianRasterizeSparse.cpp` — sparse rasterize autograd
- `src/fvdb/detail/autograd/GaussianRasterizeFromWorld.cpp` — from-world autograd
- `src/fvdb/detail/autograd/EvaluateSphericalHarmonics.cpp` — SH eval autograd
- `src/fvdb/detail/ops/gsplat/GaussianProjectionForward.h` — dispatch function signatures
- `src/fvdb/detail/ops/gsplat/GaussianProjectionBackward.h` — backward dispatch signatures
- `fvdb/functional/_transforms.py` — reference pattern for Python autograd wrapping pybind fwd/bwd

**Estimated scope:** ~500 lines of new pybind bindings, ~600 lines of Python autograd Functions,
~500 lines of C++ deletions. This is roughly the size of Milestones 2+3 combined.

---

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
