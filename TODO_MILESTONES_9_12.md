# Milestones 9-12: Complete the GridBatch Retirement

## Build & Environment

- **Conda environment:** `conda activate fvdb`
- **Build command:** `./build.sh install editor_skip -e`
  - C++ build takes **7-10 minutes**. Only rebuild after C++ changes (Milestones 10-11).
  - Phase 1 and Phase 3 are Python-only; no rebuild needed.
- **Run tests:** `cd tests && pytest unit -v`
  - Full suite takes **~5 minutes**.
  - Exclude known pre-existing failures: `--ignore=unit/test_gaussian_splat_3d.py`
    (imports retired `Grid` class) and `--ignore=unit/test_jagged_tensor.py`
    (`torch_scatter` segfault).
- **Quick smoke test:** `cd tests && pytest unit/test_basic_ops.py unit/test_inject.py
  unit/test_empty_grids.py unit/test_io.py -q --tb=line` (~3 minutes)

## Status

Milestones 1-9c are complete. Milestone 9d (switch `self.data` storage) is next,
followed by Milestone 10 (IO layer) and Milestone 11 (deletion).

```
4338af2 Milestone 9c (cont): Fix remaining Phase 1 test failures
43e6474 Milestone 9c (cont): Fix functional layer bugs exposed by delegation
329cbc1 Milestone 9c:        Make GridBatch methods delegate to functional/raw ops
94f94eb Update milestone plan to reflect completed 9a-9g work
f412ab7 Milestone 9f:        Update convolution bindings to accept GridBatchData
6960a43 Milestone 9g:        Update tests from Grid to GridBatch
aa0d804 Milestone 9b+9e:     Python autograd + raw ops in functional layer
09f1bba Milestone 9a-d:      Retire Grid class, add _prepare/_unwrap dispatch helpers
06d2723 Update TODO milestones to reflect frozen-struct refactor
61cce44 Refactor GridBatchData into frozen struct with single constructor
b67c353 Add detailed TODO for Milestones 9-12
31699a5 Milestone 8.75:      Rename GridBatchImpl -> GridBatchData, _impl -> data
cd328a1 Milestone 8.5:       Align C++ op names with Python API
986c664 Milestones 3-8:      Expose raw C++ ops and create fvdb.functional API
e4f862c Milestone 2:         Remove Vec3*OrScalar type system from C++
9e37a3f Milestone 1:         Bind GridBatchData to Python via pybind11
```

### Current architecture

- **`GridBatch` is the sole user-facing grid type.** `fvdb.Grid` was deleted.
- **`GridBatch.data` stores `_fvdb_cpp.GridBatch`** (the old C++ wrapper). This is
  what Milestone 9d will change to `_fvdb_cpp.GridBatchData`.
- **All 61 instance methods on `GridBatch`** now delegate to either `fvdb.functional`
  (~36 methods) or raw `_fvdb_cpp.<op>()` calls (~15 methods) instead of calling
  `self.data.<method>(...)`. The `_get_grid_data()` bridge in `_dispatch.py`
  transparently extracts `GridBatchData` from the old wrapper.
- **Properties** (voxel_sizes, origins, num_voxels, etc.) still read from
  `self.data.<property>` but now normalize dtype (float32) and device to ensure
  consistency whether `self.data` is `GridBatch` or `GridBatchData`.
- **`voxel_to_world` / `world_to_voxel`** are implemented in pure PyTorch using the
  grid's 4x4 affine matrices (row-major convention: `result = input @ M[:3,:3] + M[3,:3]`).
  The C++ raw ops `_fvdb_cpp.voxel_to_world` / `world_to_voxel` use NanoVDB's internal
  transform which has inverted scale vs. fvdb convention -- do NOT use them.
- **`_InjectFn` autograd** uses `mark_dirty` for in-place inject with gradient tracking.
  Works for contiguous tensors; 12 tests fail for non-contiguous strided views (PyTorch
  limitation with in-place autograd on views).
- **Factory classmethods** (`from_ijk`, `from_points`, etc.) still use
  `GridBatchCpp(device)` + `set_from_*()`. Milestone 9d will switch to
  `_fvdb_cpp.create_from_*()` free functions.
- **IO methods** (`save_nanovdb`, `from_nanovdb`) still pass `self.data` (C++ GridBatch)
  to `_fvdb_cpp.save`/`load`. Milestone 10 will update these to accept GridBatchData.
- **`from_cat`** already uses `_fvdb_cpp.concatenate_grids` (accepts GridBatchData).

### Test results

**707/719 tests pass** across the core test files (test_basic_ops, test_basic_ops_single,
test_accessors, test_inject, test_dual, test_empty_grids, test_morton_hilbert,
test_dense_interface).

**12 remaining failures:** All `test_inject_in_place_non_contiguous_subset_into_non_contiguous_superset_backprop`
-- PyTorch does not fully support `mark_dirty` in-place autograd on non-contiguous
strided view tensors. The gradient chain through strided views breaks. This edge case
may require a different approach (e.g., making the tensor contiguous before inject,
or using a custom view-aware backward).

### Key files modified in Milestone 9c

- `fvdb/grid_batch.py` -- method delegations, property normalization, pure-PyTorch
  transforms
- `fvdb/functional/_interpolation.py` -- fixed splat unwrap (grid structure, not points)
- `fvdb/functional/_pooling.py` -- fixed pool/refine unwrap, strided coarse grid
- `fvdb/functional/_dense.py` -- fixed origins shape, feature reshape, `_InjectFn`
  autograd
- `fvdb/functional/_query.py` -- fixed scalar-to-list for cube ops
- `fvdb/functional/_topology.py` -- fixed dilate dilation vector
- `fvdb/functional/_meshing.py` -- fixed marching cubes / TSDF output wrapping

### Known issues / gotchas

1. **GridBatchData properties return CPU tensors** for metadata (num_voxels, cum_voxels,
   voxel_sizes, origins, bboxes) and float64 for coordinates. The GridBatch Python
   wrapper normalizes these with `.to(device=self.device)` and `.float()`.
2. **`_fvdb_cpp.voxel_to_world` raw op is BROKEN** for fvdb use -- it uses NanoVDB's
   internal transform (scale = 1/voxel_size). Do not delegate to it. Use the
   pure-PyTorch matrix implementation in `grid_batch.py` instead.
3. **`to_Vec3fBroadcastable(scalar).tolist()`** returns a scalar, not a 3-element list.
   Always expand scalar results before passing to C++ ops that expect sequences.
4. **`GridBatchData.num_leaves`** vs **`GridBatch.num_leaf_nodes`** -- different
   attribute name. The property in `grid_batch.py` handles both via `hasattr`.
5. **`ijk_to_inv_index`** only exists on the C++ `GridBatch` wrapper, not as a raw op
   and not on `GridBatchData`. It still calls `self.data.ijk_to_inv_index()` and will
   need attention in Milestone 9d/11 (add raw op or implement in Python).

---

## ~~Milestone 9a-c: Python autograd, functional layer, method delegation~~ (DONE)

Completed in commits `09f1bba` through `4338af2`.

- 9a: Python autograd Functions (~16 classes) in each functional module
- 9b: Functional layer calls raw `_fvdb_cpp.<op>()` ops
- 9c: GridBatch methods delegate to functional or raw ops
- 9e: Convolution plan uses `_get_grid_data()` bridge
- 9f: Convolution bindings accept `GridBatchData`
- 9g: Tests updated from Grid to GridBatch

---

## Milestone 9d: Switch GridBatch.data to GridBatchData

**Goal:** `GridBatch.data` stores `_fvdb_cpp.GridBatchData` directly instead of
`_fvdb_cpp.GridBatch`. Factory classmethods use `_fvdb_cpp.create_from_*` free
functions. The `_get_grid_data` bridge is eliminated.

**No `grid_data.py` wrapper needed** -- the `batched` flag concept from the original
plan is obsolete since `Grid` was retired.

### Factory classmethod migration

| Current pattern | New pattern |
|---|---|
| `GridBatchCpp(device)` + `set_from_ijk(...)` | `_fvdb_cpp.create_from_ijk(ijk, voxel_sizes, origins)` |
| `GridBatchCpp(device)` + `set_from_points(...)` | `_fvdb_cpp.create_from_points(points, voxel_sizes, origins)` |
| `GridBatchCpp(device)` + `set_from_mesh(...)` | `_fvdb_cpp.create_from_mesh(vertices, faces, voxel_sizes, origins)` |
| `GridBatchCpp(device)` + `set_from_nearest_voxels_to_points(...)` | `_fvdb_cpp.create_from_nearest_voxels_to_points(...)` |
| `GridBatchCpp(device)` + `set_from_dense_grid(...)` | `_fvdb_cpp.create_dense(...)` |
| `GridBatchCpp(voxel_sizes=..., origins=..., device=...)` | `_fvdb_cpp.create_from_empty(device, voxel_size, origin)` |
| `GridBatchCpp(device=device)` (zero grids) | `_fvdb_cpp.create_from_empty(device)` or add new binding |
| `jcat_cpp(grid_impls)` | `_fvdb_cpp.concatenate_grids(grid_datas)` (already done) |
| `_load(path, ...)` | Returns `GridBatchData` after Milestone 10 |

### Tasks

- [ ] Change `__init__` to accept `GridBatchData` instead of `GridBatchCpp`
- [ ] Rewrite all 10 factory classmethods to use `_fvdb_cpp.create_from_*`
- [ ] Remove `from ._fvdb_cpp import GridBatch as GridBatchCpp` import
- [ ] Remove `_gridbatch` property
- [ ] Simplify `_dispatch.py::_get_grid_data()` -- just `return grid.data`
- [ ] Update `convolution_plan.py` type hints
- [ ] Update `_fvdb_cpp.pyi` -- add `GridBatchData` class, `create_from_*` stubs
- [ ] Handle `ijk_to_inv_index` (add raw op or implement in Python)
- [ ] Handle `from_zero_grids` (needs empty GridBatchData with 0 grids)

### Dependencies

Milestone 9d depends on Milestone 10 for IO methods: `save_nanovdb` passes `self.data`
to `_fvdb_cpp.save` (currently expects GridBatch). Either:
- Do Milestone 10 first (update IO to accept GridBatchData), then 9d
- Or add a temporary bridge in `save_nanovdb`/`from_nanovdb`

---

## Milestone 10: Update IO Layer

Change C++ IO and FVDB free functions to use `GridBatchData`.

### Tasks

- [ ] `src/fvdb/detail/io/LoadNanovdb.h/.cpp`:
      `fromNVDB` / `loadNVDB` return `c10::intrusive_ptr<GridBatchData>`
- [ ] `src/fvdb/detail/io/SaveNanoVDB.h/.cpp`:
      `toNVDB` / `saveNVDB` take `const GridBatchData &`
- [ ] `src/fvdb/FVDB.h/.cpp`:
      All factory functions return `c10::intrusive_ptr<GridBatchData>`.
      `jcat`, `save`, `load` switch to `GridBatchData`.
      Remove `#include <fvdb/GridBatch.h>`.
- [ ] `src/python/Bindings.cpp`:
      Update `load`, `save`, `jcat` lambda bindings
- [ ] `fvdb/grid_batch.py`:
      Update `save_nanovdb` / `from_nanovdb` to pass GridBatchData

**Requires C++ rebuild** (`./build.sh install editor_skip -e`, ~7-10 min).

---

## Milestone 11: Delete C++ GridBatch Wrapper and Autograd Layer

Remove all code that is no longer called. Net deletion: ~3,000+ lines of C++.

### Files to delete

- `src/fvdb/GridBatch.h` (~943 lines)
- `src/fvdb/GridBatch.cpp` (~1193 lines)
- `src/python/GridBatchBinding.cpp` (~728 lines)
- 9 C++ autograd file pairs (18 files) in `src/fvdb/detail/autograd/`:
  `SampleGrid`, `SplatIntoGrid`, `TransformPoints`, `MaxPoolGrid`, `AvgPoolGrid`,
  `UpsampleGrid`, `Inject`, `ReadFromDense`, `ReadIntoDense`
- **Keep**: `VolumeRender`, `Gaussian*`, `JaggedReduce`, `EvaluateSphericalHarmonics`

### Build system updates

- `src/CMakeLists.txt`: remove `fvdb/GridBatch.cpp` from `FVDB_CPP_FILES`,
  remove 9 autograd `.cpp` files
- Root `CMakeLists.txt`: remove `src/python/GridBatchBinding.cpp` from
  `FVDB_BINDINGS_CPP_FILES`
- `src/python/Bindings.cpp`: remove `bind_grid_batch(m)` call, remove
  `m.class_<fvdb::GridBatch>("GridBatch")` from `TORCH_LIBRARY`

### Cleanup

- Remove dead `#include <fvdb/GridBatch.h>` from remaining files
- Update `fvdb/_fvdb_cpp.pyi`: remove `GridBatch` class and autograd stubs

**Requires C++ rebuild.**

---

## ~~Milestone 12: Enforce Immutability on GridBatchData~~ (DONE)

Completed in commit `61cce44`.

---

## Recommended execution order

```
Phase 1 (DONE):  Milestone 9c -- delegate methods (Python-only, no rebuild)
Phase 2 (NEXT):  Milestone 10 -- update IO layer (C++ changes, rebuild required)
Phase 3:         Milestone 9d -- switch self.data storage (Python-only)
Phase 4:         Milestone 11 -- delete dead code (C++ changes, rebuild required)
```

Phase 2 must come before Phase 3 because `save_nanovdb`/`from_nanovdb` pass `self.data`
to `_fvdb_cpp.save`/`load` which currently accept `GridBatch`. The IO bindings must
accept `GridBatchData` before we switch `self.data`.

---

## Risk Notes

- **Milestone 9d** has subtle issues: factory method argument translation must be exact
  (voxel_sizes/origins broadcasting, device resolution). Validate against existing
  `set_from_*` call patterns.
- **Milestone 10** is moderate risk: IO type changes cascade through several C++ files.
- **Milestone 11** is low risk: deleting dead code. Verify clean build and tests.
