# Proposal: Functional Refactor of the GridBatch Pipeline

> **TL;DR** — Eliminate two entire C++ abstraction layers (~2,600 lines) by moving validation and ergonomic logic to Python — where it already largely exists — and replacing the wrapper code with ~25 thin, strict free functions that do only what C++ must do: autograd tape registration and CPU/CUDA template dispatch. **The public-facing Python API does not change. Existing Python tests should pass without modification.**

---

**Contents**

- [Motivation](#motivation)
- [What Changes](#what-changes)
- [What the Thin C++ Functions Look Like](#what-the-thin-c-functions-look-like)
- [What Python Gains](#what-python-gains)
- [What GridBatchImpl Becomes](#what-gridbatchimpl-becomes)
- [Design Principle: Strict in C++, Ergonomic in Python](#design-principle-strict-in-c-ergonomic-in-python)
- [Why This Is Safe](#why-this-is-safe)
- [Why This Can Be Largely Automated](#why-this-can-be-largely-automated)
- [Immutability Cleanup in C++](#immutability-cleanup-in-c)
- [Functional Python API](#functional-python-api)
- [Phasing](#phasing)
- [Metrics](#metrics)

---

## Motivation

### The current pipeline has four layers for every operation

```
Python  (fvdb/grid_batch.py, fvdb/grid.py)
  -> pybind binding  (src/python/GridBatchBinding.cpp)
    -> C++ wrapper class  (src/fvdb/GridBatch.h, src/fvdb/GridBatch.cpp)
      -> C++ core  (src/fvdb/detail/GridBatchImpl.h, src/fvdb/detail/GridBatchImpl.cu)
        -> dispatch / ops / autograd  (actual kernels)
```

Most operations pass through all four layers with each one doing progressively less work. By the time you reach `GridBatchImpl`, the actual computation is typically a single dispatch call. The two middle layers — `GridBatch` and the binding — exist primarily to convert types and re-validate inputs that Python already validated.

### The wrapper layer is already bypassed by the code that matters

The autograd functions — the differentiable operations that are the heart of the library — **already work directly with `GridBatchImpl`**. Every single autograd function in [`src/fvdb/detail/autograd/`](src/fvdb/detail/autograd/) saves and restores `c10::intrusive_ptr<GridBatchImpl>` in the autograd context. The `GridBatch` wrapper class is invisible to them.

The ops layer is the same story. Every dispatch function in [`src/fvdb/detail/ops/`](src/fvdb/detail/ops/) takes `const GridBatchImpl&`. No op or autograd function includes `GridBatch.h` or references the `GridBatch` class.

### The C++ duck-typing system duplicates Python's job

The `Vec3iOrScalar`, `Vec3dBatch`, `Vec3dBatchOrScalar` type system ([`TypesImpl.h`](src/fvdb/detail/TypesImpl.h), [`Types.h`](src/fvdb/Types.h), [`TypeCasters.h`](src/python/TypeCasters.h) — ~500 lines of template metaprogramming) exists to accept flexible inputs from Python (`1.0`, `[1,2,3]`, `torch.tensor([1,2,3])`) and convert them to strict NanoVDB types. But Python's [`fvdb/types.py`](fvdb/types.py) already provides `to_Vec3iBroadcastable`, `to_Vec3fBatchBroadcastable`, etc. — doing the same conversion with better error messages and simpler code. Every call site in [`grid_batch.py`](fvdb/grid_batch.py) already invokes the Python converter before crossing into C++, making the C++ conversion redundant.

### Real cost

The redundancy has concrete costs:

- **Compile time**: [`GridBatch.h`](src/fvdb/GridBatch.h) pulls in [`GridBatchImpl.h`](src/fvdb/detail/GridBatchImpl.h), all of `Types.h`/`TypesImpl.h`, and transitively much of NanoVDB. Every translation unit that touches this pays the price.
- **Comprehension time**: A new contributor tracing `sample_trilinear` must read through four files and two type-conversion systems to find the actual dispatch call.
- **Maintenance cost**: Adding a new parameter to an operation requires changes in `GridBatchImpl`, `GridBatch`, the binding, and the Python wrapper — four coordinated edits for one logical change.

---

## What Changes

### Files deleted

| File | Lines | Role |
|:-----|------:|:-----|
| [`src/fvdb/GridBatch.h`](src/fvdb/GridBatch.h) | 912 | Wrapper class declaration |
| [`src/fvdb/GridBatch.cpp`](src/fvdb/GridBatch.cpp) | 1,232 | Wrapper class implementation |
| [`src/fvdb/detail/TypesImpl.h`](src/fvdb/detail/TypesImpl.h) | 281 | `Vec3*OrScalar` / `Vec3*Batch` templates |
| [`src/fvdb/Types.h`](src/fvdb/Types.h) | ~40 of 49 | Type aliases (keep `SpaceFillingCurveType` enum) |
| [`src/python/TypeCasters.h`](src/python/TypeCasters.h) | ~120 of 165 | pybind type casters for the above (keep `ScalarType` caster if needed) |
| **Total removed** | **~2,585** | |

### Files added or modified

| File | Change |
|:-----|:-------|
| **New:** `src/python/GridBatchOps.cpp` (~200 lines) | ~25 thin free functions bound via pybind11 ([see below](#what-the-thin-c-functions-look-like)) |
| [`src/python/GridBatchBinding.cpp`](src/python/GridBatchBinding.cpp) | Simplified — binds `GridBatchImpl` directly + the free functions |
| [`src/python/Bindings.cpp`](src/python/Bindings.cpp) | Remove `m.class_<fvdb::GridBatch>` registration |
| [`fvdb/grid_batch.py`](fvdb/grid_batch.py) | Calls thin C++ functions directly; gains validation that was in `GridBatch.cpp` |
| [`fvdb/_fvdb_cpp.pyi`](fvdb/_fvdb_cpp.pyi) | Updated stubs to reflect new binding surface |
| [`src/fvdb/detail/ops/CubesInGrid.h`](src/fvdb/detail/ops/CubesInGrid.h), [`.cu`](src/fvdb/detail/ops/CubesInGrid.cu) | Change `Vec3dOrScalar` params to strict `nanovdb::Vec3d` |
| [`src/fvdb/detail/ops/convolution/pack_info/ConvolutionKernelMap.h`](src/fvdb/detail/ops/convolution/pack_info/ConvolutionKernelMap.h), [`.cu`](src/fvdb/detail/ops/convolution/pack_info/ConvolutionKernelMap.cu) | Change `Vec3iOrScalar` params to strict `nanovdb::Coord` |
| [`src/fvdb/detail/autograd/ReadFromDense.h`](src/fvdb/detail/autograd/ReadFromDense.h), [`.cpp`](src/fvdb/detail/autograd/ReadFromDense.cpp) | Change `Vec3iBatch` params to `std::vector<nanovdb::Coord>` |
| [`src/fvdb/detail/autograd/ReadIntoDense.h`](src/fvdb/detail/autograd/ReadIntoDense.h), [`.cpp`](src/fvdb/detail/autograd/ReadIntoDense.cpp) | Change `Vec3iBatch` params to `std::vector<nanovdb::Coord>` |
| [`src/fvdb/detail/viewer/Viewer.h`](src/fvdb/detail/viewer/Viewer.h) | Remove unused `#include <fvdb/GridBatch.h>` |

### Files untouched

- **All ops** ([`src/fvdb/detail/ops/`](src/fvdb/detail/ops/)) — already depend only on `GridBatchImpl`
- **All autograd functions** ([`src/fvdb/detail/autograd/`](src/fvdb/detail/autograd/)) — already depend only on `GridBatchImpl`
- **`GridBatchImpl`** ([`.h`](src/fvdb/detail/GridBatchImpl.h), [`.cu`](src/fvdb/detail/GridBatchImpl.cu)) — the core implementation is unchanged
- **The viewer** ([`src/fvdb/detail/viewer/`](src/fvdb/detail/viewer/)) — one dead include removed, no API change
- **Python tests** — the public API is identical; tests should pass as-is

---

## What the Thin C++ Functions Look Like

There are exactly two reasons code must stay in C++:

1. **`torch::autograd::Function::apply()`** — registers nodes in the autograd tape. This is a C++ API.
2. **`FVDB_DISPATCH_KERNEL_DEVICE`** — compile-time template instantiation for CPU/CUDA code paths.

Each thin wrapper does one of these and nothing else. No validation, no type conversion, no default handling.

### Autograd wrapper example

```cpp
// ~5 lines. No validation, no type conversion, no GridBatch class.
std::vector<torch::Tensor>
sample_trilinear_autograd(c10::intrusive_ptr<GridBatchImpl> grid,
                          JaggedTensor points,
                          torch::Tensor data,
                          bool return_grad) {
    return autograd::SampleGridTrilinear::apply(grid, points, data, return_grad);
}
```

### Device dispatch wrapper example

```cpp
// ~6 lines. Just the template dispatch that Python can't do.
std::vector<JaggedTensor>
marching_cubes_dispatch(const GridBatchImpl &grid,
                        torch::Tensor field,
                        double level) {
    return FVDB_DISPATCH_KERNEL_DEVICE(grid.device(), [&]() {
        return ops::dispatchMarchingCubes<DeviceTag>(grid, field, level);
    });
}
```

These are bound as module-level functions:

```cpp
m.def("sample_trilinear", &sample_trilinear_autograd, ...);
m.def("marching_cubes", &marching_cubes_dispatch, ...);
```

---

## What Python Gains

Python takes ownership of everything that doesn't require C++ compilation.

**Before** — Python delegates everything to the C++ wrapper:

```python
# fvdb/grid_batch.py (current)
def sample_trilinear(self, points: JaggedTensor, voxel_data: JaggedTensor) -> JaggedTensor:
    return JaggedTensor(impl=self._impl.sample_trilinear(points._impl, voxel_data._impl))
```

**After** — Python owns validation, calls C++ only for the autograd apply:

```python
# fvdb/grid_batch.py (proposed)
def sample_trilinear(self, points: JaggedTensor, voxel_data: JaggedTensor) -> JaggedTensor:
    if points.ldim() != 1:
        raise ValueError("Expected points to have 1 list dimension")
    if voxel_data.ldim() != 1:
        raise ValueError("Expected voxel_data to have 1 list dimension")
    result = _fvdb_cpp.sample_trilinear(
        self._grid_batch_impl, points._impl, voxel_data.jdata, False
    )
    return points.jagged_like(result[0])
```

For more complex operations like `max_pool`, Python absorbs the default-stride logic and coarse-grid creation that currently lives in [`GridBatch.cpp`](src/fvdb/GridBatch.cpp):

```python
# fvdb/grid_batch.py (proposed)
def max_pool(self, pool_factor, data, stride=0, coarse_grid=None):
    pool_factor = to_Vec3iBroadcastable(pool_factor, value_constraint=ValueConstraint.POSITIVE)
    stride = to_Vec3iBroadcastable(stride, value_constraint=ValueConstraint.NON_NEGATIVE)
    if (stride == 0).all():
        stride = pool_factor
    if data.ldim() != 1:
        raise ValueError("Expected data to have 1 list dimension")
    if coarse_grid is None:
        coarse_grid = self.coarsened_grid(stride)
    result = _fvdb_cpp.max_pool_autograd(
        self._grid_batch_impl, coarse_grid._grid_batch_impl,
        pool_factor.tolist(), stride.tolist(), data.jdata
    )
    return coarse_grid.jagged_like(result[0]), coarse_grid
```

This is the same logic that `GridBatch.cpp` currently contains — moved to where it's easier to read, test, and modify.

---

## What GridBatchImpl Becomes

[`GridBatchImpl`](src/fvdb/detail/GridBatchImpl.h) is already the right shape. It holds the core data (NanoVDB grid handle, metadata arrays, batch tensors), provides the `Accessor` for CUDA kernels, and supports indexing/slicing as views.

The derived-grid methods currently on `GridBatchImpl` (`coarsen`, `upsample`, `dual`, `clip`, `dilate`, `merge`, `prune`, `convolutionOutput`) can optionally be extracted as free functions in a later phase, since they follow a pure functional pattern: read from `const GridBatchImpl&`, create a new `GridBatchImpl`, return it. But this is not required for the initial refactor — they already work correctly as methods and the ops/autograd layer doesn't care either way.

`GridBatchImpl` retains its `CustomClassHolder` base and `TORCH_LIBRARY` registration, which is required by the autograd saved-variable mechanism.

---

## Immutability Cleanup in C++

The library has formally embraced an immutable design on the Python side — grids are created, never mutated. The C++ layer hasn't fully caught up. [`GridBatchImpl`](src/fvdb/detail/GridBatchImpl.h) currently exposes several public mutation methods that are artifacts of an older design:

| Method | Current callers | After refactor |
|:-------|:----------------|:---------------|
| `setGlobalVoxelSize` | [`GridBatch.cpp`](src/fvdb/GridBatch.cpp) only | **Delete** — `GridBatch.cpp` is removed, no callers remain |
| `setGlobalVoxelOrigin` | `GridBatch.cpp` only | **Delete** |
| `setGlobalPrimalTransform` | `GridBatch.cpp` only | **Delete** |
| `setGlobalDualTransform` | `GridBatch.cpp` only | **Delete** |
| `setGlobalVoxelSizeAndOrigin` | `GridBatch.cpp` only | **Delete** |
| `setFineTransformFromCoarseGrid` | Internal: called by `upsample()` on a freshly-created object | **Make private** |
| `setCoarseTransformFromFineGrid` | Internal: called by `coarsen()` on a freshly-created object | **Make private** |
| `setPrimalTransformFromDualGrid` | Internal: called by `dual()` on a freshly-created object | **Make private** |
| `setGrid` | Internal: construction only | **Make private** |

These are already inaccessible from the Python API — neither [`grid_batch.py`](fvdb/grid_batch.py) nor [`grid.py`](fvdb/grid.py) expose any of them. The `setGlobal*` methods are bound via pybind in [`GridBatchBinding.cpp`](src/python/GridBatchBinding.cpp) but that binding is deleted along with `GridBatch.cpp`.

The three `set*TransformFrom*Grid` methods are used internally by `coarsen()`, `upsample()`, and `dual()` to adjust transforms on a freshly-constructed `GridBatchImpl` before returning it. Making them private preserves this internal use while ensuring the public C++ interface is fully const after construction. In Phase 4, if these derived-grid methods are extracted as free functions, the transform adjustment would be folded into the construction path itself, eliminating these setters entirely.

After this cleanup, `GridBatchImpl`'s public interface is **entirely read-only after construction**: create it, query it, build an accessor from it, index into it, serialize it. No public mutation. This matches the library's design contract and removes a class of potential misuse from the C++ API.

---

## Functional Python API

> *This section is de-emphasized for the initial pitch — it does not block or complicate the core refactor. It describes a natural follow-on that the refactor enables.*

The refactor naturally produces a set of Python functions that take a grid and data as arguments and return results — the same pattern as `torch.nn.functional` relative to `torch.nn`. Once the class-based methods in `grid_batch.py` are calling thin C++ functions directly, we can surface those same calls as a public functional API:

```python
# fvdb/functional.py (new module)

def sample_trilinear(grid: GridBatch, points: JaggedTensor, voxel_data: JaggedTensor) -> JaggedTensor:
    """Sample voxel data at world-space points using trilinear interpolation."""
    if points.ldim() != 1:
        raise ValueError("Expected points to have 1 list dimension")
    if voxel_data.ldim() != 1:
        raise ValueError("Expected voxel_data to have 1 list dimension")
    result = _fvdb_cpp.sample_trilinear(grid._grid_batch_impl, points._impl, voxel_data.jdata, False)
    return points.jagged_like(result[0])

def coarsened_grid(grid: GridBatch, coarsening_factor: NumericMaxRank1) -> GridBatch:
    """Return a coarsened copy of the grid."""
    coarsening_factor = to_Vec3iBroadcastable(coarsening_factor, value_constraint=ValueConstraint.POSITIVE)
    return GridBatch(impl=grid._grid_batch_impl.coarsen(coarsening_factor.tolist()))
```

The class-based API then becomes a thin delegation layer:

```python
# fvdb/grid_batch.py
class GridBatch:
    def sample_trilinear(self, points, voxel_data):
        return fvdb.functional.sample_trilinear(self, points, voxel_data)

    def coarsened_grid(self, coarsening_factor):
        return fvdb.functional.coarsened_grid(self, coarsening_factor)
```

This gives users a choice of style — `grid.sample_trilinear(pts, data)` or `fvdb.functional.sample_trilinear(grid, pts, data)` — without maintaining two implementations. The class methods are one-liners that delegate to the functional versions. The `Grid` single-grid class follows the same pattern.

This is a natural byproduct of the refactor, not an additional effort: the validation and dispatch logic that moves into Python *is* the functional implementation. The class methods just call it.

---

## Design Principle: Strict in C++, Ergonomic in Python

This refactor enshrines a clear boundary:

| Responsibility | Where it lives |
|:---|:---|
| Accept flexible user input (scalars, lists, tensors, mixed types) | **Python** |
| Validate shapes, dtypes, list dimensions, value ranges | **Python** |
| Handle default parameters and optional arguments | **Python** |
| Provide docstrings and type annotations | **Python** |
| Convert to strict, unambiguous types | **Python** |
| Register autograd tape nodes | **C++** (thin function) |
| Dispatch CPU/CUDA template instantiation | **C++** (thin function) |
| Core data storage, accessor, NanoVDB integration | **C++** (`GridBatchImpl`) |
| CUDA kernels and dispatch functions | **C++** (ops layer, unchanged) |

C++ function signatures become strict and unambiguous — they take exactly the types the kernel needs. No duck-typing, no template metaprogramming for input flexibility, no multi-constructor overload resolution. If a kernel needs `nanovdb::Coord`, the C++ function takes `std::array<int32_t, 3>` or `nanovdb::Coord`. Python has already done the conversion before the call crosses the boundary.

---

## Why This Is Safe

1. **The public Python API is unchanged.** [`grid_batch.py`](fvdb/grid_batch.py) and [`grid.py`](fvdb/grid.py) present the same classes, methods, properties, and type signatures. User code does not change.

2. **The ops and autograd layers are untouched.** They already work with `GridBatchImpl` exclusively. The refactor only changes how they are *called*, not what they *do*.

3. **The viewer is unaffected.** [`Viewer.h`](src/fvdb/detail/viewer/Viewer.h) has one unused include of `GridBatch.h` and no actual dependency on the wrapper class. The viewer team's code, data model, and API are not disrupted.

4. **The testing infrastructure is already in place.** Python tests exercise the public API end-to-end. Since the API doesn't change, existing tests validate the refactored code paths. The C++ layers being removed are purely intermediary — they have no behavior that isn't already tested through the Python surface.

5. **`GridBatchImpl` is stable.** It's not being restructured. Its public interface, accessor, memory management, and serialization format are all unchanged.

---

## Why This Can Be Largely Automated

The wrapper methods in [`GridBatch.cpp`](src/fvdb/GridBatch.cpp) follow a small number of mechanical patterns:

| Pattern | Count | Transformation |
|:--------|------:|:---------------|
| **Pure forwarding**: `return mImpl->someMethod(args...)` | ~40 | Becomes direct Python call to `GridBatchImpl` binding |
| **Validation + autograd dispatch**: checks + `autograd::Fn::apply(mImpl, ...)` | ~12 | Validation moves to Python; `::apply()` becomes a thin bound function |
| **Validation + device dispatch**: checks + `FVDB_DISPATCH_KERNEL_DEVICE(...)` | ~15 | Same — validation to Python, dispatch stays in thin C++ function |
| **Type conversion + forwarding**: `Vec3iOrScalar` -> `nanovdb::Coord` -> forward | ~20 | C++ conversion deleted; Python's existing `to_Vec3iBroadcastable` is kept |

Each pattern is identifiable by inspection and transformable by a consistent recipe. The existing test suite serves as the correctness oracle at every step.

---

## Phasing

The refactor can be done incrementally, with tests passing at every intermediate state.

### Phase 1: Bind `GridBatchImpl` and thin functions alongside existing `GridBatch`

Add pybind bindings for `GridBatchImpl` properties/methods and the ~25 thin wrapper functions. Both old and new bindings coexist. Wire up a few Python methods to use the new path. Run tests.

### Phase 2: Migrate `grid_batch.py` method by method

For each method in `grid_batch.py`, switch from calling `self._impl.method(...)` (which goes through `GridBatch`) to calling `_fvdb_cpp.thin_function(self._grid_batch_impl, ...)` directly. Move validation from `GridBatch.cpp` to Python where it doesn't already exist. Run tests after each method.

### Phase 3: Remove `GridBatch.h/cpp` and the `Vec3*` type system

Once no Python code references the old `GridBatch` bindings, delete the files listed in [Files deleted](#files-deleted). Update [`Bindings.cpp`](src/python/Bindings.cpp) to remove the `GridBatch` class registration. Clean up the 4 ops/autograd files that use `Vec3*OrScalar`/`Vec3*Batch`. Remove the dead include from `Viewer.h`. Run tests.

### Phase 4: Enforce immutability in `GridBatchImpl`

Delete the `setGlobal*` public mutation methods from `GridBatchImpl` (no remaining callers after Phase 3). Make `setFineTransformFromCoarseGrid`, `setCoarseTransformFromFineGrid`, `setPrimalTransformFromDualGrid`, and `setGrid` private. This makes `GridBatchImpl`'s public interface fully read-only after construction, matching the library's immutability contract. See [Immutability Cleanup in C++](#immutability-cleanup-in-c) for details.

### Phase 5: Surface `fvdb.functional` module

Extract the validation and dispatch logic in `grid_batch.py` into a public `fvdb/functional.py` module of free functions. Rewrite the `GridBatch` and `Grid` class methods as one-liner delegations to the functional versions. This is a natural byproduct — the functional implementations already exist at this point as the method bodies; they just need to be lifted into a module. See [Functional Python API](#functional-python-api) for details.

### Phase 6 (optional): Extract derived-grid ops from `GridBatchImpl`

Move `coarsen`, `upsample`, `dual`, `clip`, `dilate`, `merge`, `prune`, `convolutionOutput` from `GridBatchImpl` methods to free functions. Fold the private transform-setter calls into the construction path, eliminating those methods entirely. This further slims the core class but is a lower-priority cleanup.

---

## Metrics

| Metric | Before | After | Delta |
|:-------|-------:|------:|------:|
| C++ lines in wrapper layer | ~2,585 | 0 | **-2,585** |
| C++ lines in thin functions | 0 | ~200 | +200 |
| Python lines in `grid_batch.py` | ~2,900 | ~3,100 | +200 |
| Total C++ files for GridBatch pipeline | 6 | 2 | **-4 files** |
| Layers in the call chain | 4 | 2 | **-2 layers** |
| C++ template metaprogramming for type flexibility | ~500 lines | 0 | **-500 lines** |
| Places to edit when adding a new operation | 4 files | 2 files | **-2 files** |

---

## Conclusion

The `GridBatch` wrapper and `Vec3*` type system were reasonable abstractions when the C++ layer was the primary API surface. Now that Python is the API surface — with its own type conversion, validation, docstrings, and static typing — these layers are pure overhead. Every operation is already functional in nature (const input -> new output), the autograd and ops layers already depend only on `GridBatchImpl`, and the Python layer already wraps everything.

This refactor removes the empty middle, enshrines the boundary between Python ergonomics and C++ strictness, and makes the codebase smaller, faster to compile, and easier to understand — without changing a single user-facing behavior. It also brings the C++ layer into alignment with the library's immutability contract by eliminating public mutation methods that are already inaccessible from Python, and it naturally enables a `torch.nn.functional`-style functional Python API as a first-class alternative to the existing class-based interface.
