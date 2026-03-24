# Milestones 9-12: Complete the GridBatch Retirement

## Status

Milestones 1-8.75 are complete. The branch has these commits:

```
31699a5 Milestone 8.75: Rename GridBatchImpl -> GridBatchData, _impl -> data
cd328a1 Milestone 8.5:  Align C++ op names with Python API
986c664 Milestones 3-8: Expose raw C++ ops and create fvdb.functional API
e4f862c Milestone 2:    Remove Vec3*OrScalar type system from C++
9e37a3f Milestone 1:    Bind GridBatchData to Python via pybind11
```

### What exists now

- `GridBatchData` (C++) is the core grid data type, bound to Python with properties,
  factories, topology methods, and pickle support (`GridBatchDataBinding.cpp`).
- 39 raw C++ ops are bound as `_fvdb_cpp.<op>` in `GridBatchOps.cpp`.
- `fvdb.functional` has 42 public functions with `@overload` for Grid/GridBatch dispatch.
  Currently these delegate to `grid.data.<method>()` (the C++ `GridBatch` wrapper).
- `Grid.data` and `GridBatch.data` both store a `_fvdb_cpp.GridBatch` (C++ wrapper).
- C++ op names match the Python API (Milestone 8.5 alignment).
- Naming is clean: `GridBatchData` in C++, `.data` attribute in Python (Milestone 8.75).

---

## Milestone 9: Make Grid/GridBatch Thin Wrappers

The largest milestone. The goal is to cut the dependency on the C++ `GridBatch` wrapper
and C++ autograd layer, replacing them with direct calls to raw ops + Python autograd.

### Pre-requisite: `batched` flag on GridBatchData

A Python-only `batched` boolean on `GridBatchData` resolves an ambiguity: when
`grid_count == 1`, there is no way to tell whether data arguments should be
`JaggedTensor` (batched) or `torch.Tensor` (unbatched) based on the grid alone.

- Create a Python wrapper class (e.g. `fvdb/grid_data.py`) that holds the C++ `_fvdb_cpp.GridBatchData`
  plus an immutable `batched: bool`.
- `__getattr__` delegates all C++ properties/methods transparently.
- Batched and unbatched factory methods set the flag at construction time.
- `Grid` always creates `GridBatchData(cpp, batched=False)`.
- `GridBatch` always creates `GridBatchData(cpp, batched=True)`.
- Topology-deriving ops preserve the flag: `coarsened_grid(gd)` returns a new
  `GridBatchData` with the same `batched` value.

### 9a. Implement Python autograd Functions (~16 classes)

Add `torch.autograd.Function` subclasses to each functional module, following the
`convolution_plan.py` pattern. The tracked tensor (the one autograd differentiates through)
is passed as the first positional arg to `.apply()`; non-tensor args (GridBatchData, C++ JaggedTensor)
go through `ctx`.

#### Adjoint pairs (backward of one IS the forward of the other)

| Autograd class         | Forward op                    | Backward op (adjoint)         |
|------------------------|-------------------------------|-------------------------------|
| `_SampleTrilinearFn`   | `_fvdb_cpp.sample_trilinear`  | `_fvdb_cpp.splat_trilinear`   |
| `_SplatTrilinearFn`    | `_fvdb_cpp.splat_trilinear`   | `_fvdb_cpp.sample_trilinear`  |
| `_SampleBezierFn`      | `_fvdb_cpp.sample_bezier`     | `_fvdb_cpp.splat_bezier`      |
| `_SplatBezierFn`       | `_fvdb_cpp.splat_bezier`      | `_fvdb_cpp.sample_bezier`     |
| `_InjectFromDenseCminorFn` | `inject_from_dense_cminor` | `inject_to_dense_cminor`      |
| `_InjectFromDenseCmajorFn` | `inject_from_dense_cmajor` | `inject_to_dense_cmajor`      |
| `_InjectToDenseCminorFn`   | `inject_to_dense_cminor`   | `inject_from_dense_cminor`    |
| `_InjectToDenseCmajorFn`   | `inject_to_dense_cmajor`   | `inject_from_dense_cmajor`    |

#### Explicit backward ops

| Autograd class                  | Forward op                          | Backward op                              | Saved tensors     |
|---------------------------------|-------------------------------------|------------------------------------------|-------------------|
| `_SampleTrilinearWithGradFn`    | `sample_trilinear_with_grad`        | `sample_trilinear_with_grad_bwd`         | `data`            |
| `_SampleBezierWithGradFn`       | `sample_bezier_with_grad`           | `sample_bezier_with_grad_bwd`            | `data`            |
| `_VoxelToWorldFn`               | `voxel_to_world`                    | `voxel_to_world_bwd`                     | (none)            |
| `_WorldToVoxelFn`               | `world_to_voxel`                    | `world_to_voxel_bwd`                     | (none)            |
| `_MaxPoolFn`                    | `max_pool`                          | `max_pool_bwd`                           | input `data`      |
| `_AvgPoolFn`                    | `avg_pool`                          | `avg_pool_bwd`                           | input `data`      |
| `_RefineFn`                     | `refine`                            | `refine_bwd`                             | coarse `data`     |

#### Self-adjoint

| Autograd class | Forward op            | Backward                               |
|----------------|-----------------------|----------------------------------------|
| `_InjectFn`    | `_fvdb_cpp.inject_op` | inject gradient in the reverse direction |

### 9b. Update functional/ to call raw ops + GridBatchData

Each functional module switches from `grid.data.<method>(...)` (calling C++ GridBatch)
to raw C++ ops via `_fvdb_cpp.<op>(grid.data._cpp, ...)`.

Pattern for differentiable ops:
```python
def sample_trilinear(grid_data, points, voxel_data):
    if isinstance(points, torch.Tensor):
        # Unbatched (Grid) path
        jt_pts = JaggedTensor(points)
        result = _SampleTrilinearFn.apply(voxel_data, grid_data._cpp, jt_pts._impl)
        return result
    else:
        # Batched (GridBatch) path
        result = _SampleTrilinearFn.apply(voxel_data.jdata, grid_data._cpp, points._impl)
        return points.jagged_like(result)
```

For topology ops, call GridBatchData methods directly:
- `coarsened_grid` -> `grid_data.coarsen(factor)`, wrap in new `GridBatchData`
- `refined_grid` -> `grid_data.upsample(factor, mask)`
- `dual_grid` -> `grid_data.dual(exclude_border)`
- `dilated_grid` -> `grid_data.dilate(dilation)`
- `merged_grid` -> `grid_data.merge(other._cpp)`
- `pruned_grid` -> `grid_data.prune(mask._impl)`
- `conv_grid` -> `grid_data.convolution_output(ks, st)`
- `conv_transpose_grid` -> `grid_data.convolution_transpose_output(ks, st)`

For non-differentiable ops (queries, rays, meshing), call `_fvdb_cpp.<op>` directly.

For space-filling curves, call `_fvdb_cpp.serialize_encode(grid_data._cpp, order, offset)`.

### 9c. Make Grid/GridBatch methods one-liner delegations

Every operation method becomes a delegation to `fvdb.functional`. Docstrings stay.

```python
# grid_batch.py
def sample_trilinear(self, points: JaggedTensor, voxel_data: JaggedTensor) -> JaggedTensor:
    """... existing docstring ..."""
    return fvdb.functional.sample_trilinear(self.data, points, voxel_data)

# grid.py
def sample_trilinear(self, points: torch.Tensor, voxel_data: torch.Tensor) -> torch.Tensor:
    """... existing docstring ..."""
    return fvdb.functional.sample_trilinear(self.data, points, voxel_data)
```

Property accessors that read grid metadata stay as direct `self.data.<property>` reads.

### 9d. Switch Grid/GridBatch.data to GridBatchData

1. Change import: `from .grid_data import GridBatchData`
2. Change `__init__`: stores `GridBatchData` (Python wrapper with `batched` flag)
3. Update ALL factory classmethods:
   - `from_ijk` -> use `_fvdb_cpp.GridBatchData.create_from_ijk(...)`, wrap in Python `GridBatchData`
   - `from_points` -> `_fvdb_cpp.GridBatchData.create_from_points(...)`
   - `from_mesh` -> `_fvdb_cpp.GridBatchData.create_from_mesh(...)`
   - `from_nearest_voxels_to_points` -> `_fvdb_cpp.GridBatchData.create_from_nearest_voxels_to_points(...)`
   - `from_dense` -> `_fvdb_cpp.GridBatchData.create_dense(...)`
   - `from_zero_voxels` -> `_fvdb_cpp.GridBatchData.create_from_empty(device, voxel_size, origin)`
   - `from_zero_grids` -> `_fvdb_cpp.GridBatchData(device)` (default constructor)
   - `from_cat` -> `_fvdb_cpp.GridBatchData.concatenate(impls)`
   - `from_nanovdb` -> bridge through `_load()` (may need Milestone 10 for full switch)
4. Remove `_grid_batch_data` bridge accessor (now redundant)
5. Remove `_gridbatch` property

### 9e. Update convolution_plan.py

1. Replace `GridBatch(data=source_grid.data)` patterns
2. Update `_fvdb_cpp.GridBatch` type hints to `_fvdb_cpp.GridBatchData`
3. Update `gs_build_topology` / `gs_build_transpose_topology` bindings to accept
   `GridBatchData` instead of `GridBatch`
4. Update `pred_gather_igemm_conv` binding similarly
5. In `Bindings.cpp`: update convolution lambdas to accept
   `const fvdb::detail::GridBatchData &` and call ops directly

### Files touched

- `fvdb/grid_data.py` (NEW)
- `fvdb/functional/_interpolation.py`, `_transforms.py`, `_pooling.py`, `_dense.py`
- `fvdb/functional/_query.py`, `_ray.py`, `_meshing.py`, `_topology.py`
- `fvdb/functional/__init__.py`
- `fvdb/grid_batch.py`, `fvdb/grid.py`
- `fvdb/convolution_plan.py`
- `fvdb/__init__.py`
- `src/python/Bindings.cpp` (convolution bindings only)

---

## Milestone 10: Update IO Layer and FVDB Free Functions

Change C++ IO and FVDB free functions to use `GridBatchData` / `c10::intrusive_ptr<GridBatchData>`.

### Tasks

- [ ] `src/fvdb/detail/io/LoadNanovdb.h/.cpp`:
      `fromNVDB` / `loadNVDB` return `tuple<c10::intrusive_ptr<GridBatchData>, ...>`
- [ ] `src/fvdb/detail/io/SaveNanoVDB.h/.cpp`:
      `toNVDB` / `saveNVDB` take `const GridBatchData &`
- [ ] `src/fvdb/FVDB.h/.cpp`:
      All factory functions return `c10::intrusive_ptr<GridBatchData>`.
      `jcat`, `save`, `load` switch to `GridBatchData`.
      Remove `#include <fvdb/GridBatch.h>`.
- [ ] `src/python/Bindings.cpp`:
      Update `gridbatch_from_*`, `load`, `save`, `jcat` bindings
- [ ] `fvdb/grid_batch.py` / `fvdb/grid.py`:
      Update `from_nanovdb` factories if deferred from Milestone 9

---

## Milestone 11: Delete C++ GridBatch Wrapper and Autograd Layer

Remove all code that is no longer called. Net deletion: ~3,000+ lines of C++.

### Tasks

- [ ] Delete `src/fvdb/GridBatch.h` (~950 lines)
- [ ] Delete `src/fvdb/GridBatch.cpp` (~1200 lines)
- [ ] Delete `src/python/GridBatchBinding.cpp` (~750 lines)
- [ ] Delete C++ autograd files replaced by Python autograd:
  - `src/fvdb/detail/autograd/SampleGrid.h` + `.cpp`
  - `src/fvdb/detail/autograd/SplatIntoGrid.h` + `.cpp`
  - `src/fvdb/detail/autograd/TransformPoints.h` + `.cpp`
  - `src/fvdb/detail/autograd/MaxPoolGrid.h` + `.cpp`
  - `src/fvdb/detail/autograd/AvgPoolGrid.h` + `.cpp`
  - `src/fvdb/detail/autograd/UpsampleGrid.h` + `.cpp`
  - `src/fvdb/detail/autograd/Inject.h` + `.cpp`
  - `src/fvdb/detail/autograd/ReadFromDense.h` + `.cpp`
  - `src/fvdb/detail/autograd/ReadIntoDense.h` + `.cpp`
  - (18 files total)
  - **Keep**: `VolumeRender`, `Gaussian*`, `JaggedReduce`, `EvaluateSphericalHarmonics`
- [ ] Update `src/CMakeLists.txt`: remove GridBatch.cpp + deleted autograd .cpp files
- [ ] Update root `CMakeLists.txt`: remove `GridBatchBinding.cpp` from `FVDB_BINDINGS_CPP_FILES`
- [ ] Remove `m.class_<fvdb::GridBatch>("GridBatch")` from `TORCH_LIBRARY` in `Bindings.cpp`
- [ ] Remove dead `#include <fvdb/GridBatch.h>` from `Viewer.h` and anywhere else
- [ ] Update `fvdb/_fvdb_cpp.pyi`: remove `GridBatch` class, remove autograd entries

---

## Milestone 12: Enforce Immutability on GridBatchData

Make `GridBatchData` (C++) a truly immutable value type after construction.

### Tasks

- [ ] Delete public mutators from `src/fvdb/detail/GridBatchData.h`:
  - `setGlobalVoxelSize`
  - `setGlobalVoxelOrigin`
  - `setGlobalPrimalTransform`
  - `setGlobalDualTransform`
  - `setGlobalVoxelSizeAndOrigin`
- [ ] Make private (used internally by topology-deriving methods):
  - `setFineTransformFromCoarseGrid`
  - `setCoarseTransformFromFineGrid`
  - `setPrimalTransformFromDualGrid`
  - `setGrid`
- [ ] Remove `set_global_origin` and `set_global_voxel_size` from pybind bindings
      (if still exposed after Milestone 11)
- [ ] Remove or make private `nanoGridHandleMut()`
- [ ] Update `fvdb/_fvdb_cpp.pyi`
- [ ] Build and verify all tests pass

---

## Risk Notes

- **Milestone 9** is the riskiest: Python autograd must exactly reproduce C++ autograd
  behavior. Test each autograd Function incrementally if possible.
- **Milestone 10** is moderate risk: IO type changes cascade through several files.
- **Milestones 11-12** are low risk: deleting dead code, restricting access.
- Milestone 9 is mostly Python-only (except convolution bindings in `Bindings.cpp`).
  Milestones 10-12 require C++ rebuilds.
