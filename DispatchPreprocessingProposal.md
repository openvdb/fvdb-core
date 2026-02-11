# Proposal: Dispatch Preprocessing Pass

> **TL;DR** — Before the [functional GridBatch refactor](FunctionalRefactorProposal.md), create a new `fvdb/detail/dispatch/` module with device-unified forEach tools aligned with the new dispatch framework, then port ops one-by-one to use them and internalize their own device dispatch. After this pass, every op header exposes a type-erased function — no templates leak to callers. The first PR introduces the new tools and ports a minimum set of ops to prove the pattern. Subsequent PRs are mechanical "more of the same."

---

**Contents**

- [Motivation](#motivation)
- [Problems with the Current forEach Infrastructure](#problems-with-the-current-foreach-infrastructure)
- [The New `fvdb/detail/dispatch/` Module](#the-new-fvdbdetaildispatch-module)
- [How Ops Change](#how-ops-change)
- [Deprecating SimpleOpHelper](#deprecating-simpleophelper)
- [Op Inventory](#op-inventory)
- [PR Strategy](#pr-strategy)
- [Relationship to the Functional Refactor](#relationship-to-the-functional-refactor)

---

## Motivation

Currently, most ops expose a `template <torch::DeviceType>` function in their header (see [How Ops Change](#how-ops-change) for a concrete before/after). Every caller must wrap the call in `FVDB_DISPATCH_KERNEL_DEVICE` to select the device at runtime. This has three costs:

1. **Template leakage**: The `DeviceTag` template propagates from the op through the caller. Callers must be compiled in a CUDA-aware context even if they contain no device code.
2. **Scattered dispatch**: `FVDB_DISPATCH_KERNEL_DEVICE` appears in [`GridBatchImpl.cu`](src/fvdb/detail/GridBatchImpl.cu) (~8 sites) and [`GridBatch.cpp`](src/fvdb/GridBatch.cpp) (~16 sites). The [functional refactor](FunctionalRefactorProposal.md) deletes `GridBatch.cpp`, so these sites need a home. If they live inside the ops, the functional refactor becomes purely a Python-side change.
3. **Inconsistency**: The newer ops (morton/hilbert) already internalize dispatch with type-erased entry points. Most ops don't.

But more fundamentally, the reason each op has to do per-device dispatch manually is because the underlying forEach tools are per-device. Fixing the tools fixes everything downstream.

---

## Problems with the Current forEach Infrastructure

The current iteration primitives ([`ForEachCPU.h`](src/fvdb/detail/utils/ForEachCPU.h), [`ForEachCUDA.cuh`](src/fvdb/detail/utils/cuda/ForEachCUDA.cuh), [`ForEachPrivateUse1.cuh`](src/fvdb/detail/utils/cuda/ForEachPrivateUse1.cuh)) have several design problems:

### 1. Per-device triplication

There are three separate functions for every iteration pattern:

```cpp
forEachVoxelCPU(numChannels, batchHdl, func, args...);
forEachVoxelCUDA(numThreads, numChannels, batchHdl, func, args...);
forEachVoxelPrivateUse1(numChannels, batchHdl, func, args...);
```

Every op must choose the right one at every call site, leading to the pervasive `if constexpr (DeviceTag == torch::kCUDA) { ... } else if ...` pattern — even when the callback is `__hostdev__` and is identical across all devices.

### 2. Incompatible signatures

The three variants don't even have the same parameter lists. CUDA takes `numThreads` and `stream`; CPU doesn't. This means you can't write a single call that works on all devices.

### 3. Callback signature overload

Callbacks receive raw indices `(bidx, lidx, vidx, cidx, accessor, args...)` — a positional parade of integers with no type safety. Each op must manually decode these into meaningful voxel coordinates and feature indices.

### 4. Channel parallelism entangled with iteration

The `numChannels` / `channelsPerLeaf` parameter is baked into the iteration loop, mixing parallelism strategy with element traversal. This creates complex index arithmetic in every callback.

### 5. Template parameter explosion

Every forEach function is templated on `<ScalarT, NDIMS, Func, Args...>`. This interacts with the per-device triplication to create a combinatorial explosion of template instantiations.

[`SimpleOpHelper.h`](src/fvdb/detail/utils/SimpleOpHelper.h) was a step in the right direction — it introduced CRTP base classes (`BasePerActiveVoxelProcessor`, `BasePerElementProcessor`) that handle output allocation and hide the per-device dispatch behind `if constexpr`. But it still wraps the old forEach tools and inherits their problems. And CRTP is needless ceremony for what are conceptually just callbacks.

---

## The New `fvdb/detail/dispatch/` Module

The new module provides **device-unified** iteration tools aligned with the new dispatch framework's design philosophy: write `__hostdev__` working code once, let the framework handle device dispatch.

### Core design principles

1. **Single call site**: One function call iterates over the grid/tensor. The function handles CPU/CUDA/PrivateUse1 internally based on the input's device. No `if constexpr` in op code.
2. **`__hostdev__` callbacks**: If the callback is `__hostdev__`, it runs on any device. No per-device lambda variants.
3. **No CRTP**: Callbacks are plain callables (lambdas, structs with `operator()`). No base classes to inherit from.
4. **Separated concerns**: Iteration (what to visit) is separate from parallelism (how many threads/channels). Sensible defaults for thread counts; explicit override when needed.

### `forEachActiveVoxel`

Iterates over active voxels in a `GridBatchImpl`, calling a `__hostdev__` callback for each:

```cpp
// fvdb/detail/dispatch/ForEachActiveVoxel.cuh
namespace fvdb::detail::dispatch {

// Callback: void(nanovdb::Coord ijk, int64_t voxel_index, GridBatchImpl::Accessor acc, Args...)
// where voxel_index is the linear feature index for this voxel
template <typename Callback, typename... Args>
void forEachActiveVoxel(const GridBatchImpl &grid, Callback &&callback, Args&&... args);

}
```

An op using this:

```cpp
// No device template. No CRTP. Just a __hostdev__ callable.
struct WriteCoords {
    __hostdev__ void operator()(nanovdb::Coord ijk, int64_t idx,
                                GridBatchImpl::Accessor, auto out) const {
        out[idx][0] = ijk[0];
        out[idx][1] = ijk[1];
        out[idx][2] = ijk[2];
    }
};

JaggedTensor activeGridCoords(const GridBatchImpl &grid) {
    auto out = allocateOutput(grid, FixedShape<int32_t, 3>{});
    auto acc = makeAccessor(out);
    dispatch::forEachActiveVoxel(grid, WriteCoords{}, acc);
    return grid.jaggedTensor(out);
}
```

Compare this to the current version in [`ActiveGridGoords.cu`](src/fvdb/detail/ops/ActiveGridGoords.cu), which requires a CRTP base class, explicit template instantiations for three device types, and a templated `dispatch*` entry point.

### `forEachLeaf`

For leaf-level iteration:

```cpp
// Callback: void(int64_t batch_idx, int64_t leaf_idx, GridBatchImpl::Accessor acc, Args...)
template <typename Callback, typename... Args>
void forEachLeaf(const GridBatchImpl &grid, Callback &&callback, Args&&... args);
```

### `forEachJaggedElement`

For iteration over JaggedTensor elements. Note: `ScalarT` and `NDIMS` template parameters are retained because they determine the accessor type for type-safe tensor access — this is unavoidable without runtime type erasure on the tensor side.

```cpp
// Callback: void(int64_t batch_idx, int64_t element_idx, JaggedAccessor acc, Args...)
template <typename ScalarT, int32_t NDIMS, typename Callback, typename... Args>
void forEachJaggedElement(const JaggedTensor &jt, Callback &&callback, Args&&... args);
```

### `forEachTensorElement`

For plain tensor iteration:

```cpp
// Callback: void(int64_t element_idx, TensorAccessor acc, Args...)
template <typename ScalarT, int32_t NDIMS, typename Callback, typename... Args>
void forEachTensorElement(const torch::Tensor &tensor, Callback &&callback, Args&&... args);
```

### With-channels variants

For ops that need explicit channel parallelism (e.g., trilinear interpolation across feature channels), there would be `WithChannels` variants that expose `channel_idx` in the callback. But the default is no channel dimension — most ops don't need it.

### Internal implementation

Each dispatch tool has a `.cuh` header (since it instantiates CUDA kernels internally) that:
1. Checks `grid.device()` / `tensor.device()`
2. Dispatches to the appropriate existing low-level forEach (CPU loop, CUDA kernel launch, PrivateUse1 kernel launch)
3. Provides sensible defaults for thread count and shared memory

The low-level per-device forEach functions ([`ForEachCPU.h`](src/fvdb/detail/utils/ForEachCPU.h), [`ForEachCUDA.cuh`](src/fvdb/detail/utils/cuda/ForEachCUDA.cuh), [`ForEachPrivateUse1.cuh`](src/fvdb/detail/utils/cuda/ForEachPrivateUse1.cuh)) remain as the backend — they're correct and tested. The new dispatch module is a unifying layer on top, not a rewrite of the kernel launch mechanics.

---

## How Ops Change

### Before (old-style header)

```cpp
// src/fvdb/detail/ops/CoordsInGrid.h (current)
#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/GridBatchImpl.h>    // full include

template <torch::DeviceType>              // template leaks to caller
JaggedTensor dispatchCoordsInGrid(const GridBatchImpl &batchHdl, const JaggedTensor &coords);
```

### After (type-erased header)

```cpp
// src/fvdb/detail/ops/CoordsInGrid.h (target)
#include <fvdb/JaggedTensor.h>

namespace fvdb::detail { class GridBatchImpl; }  // forward declaration only

namespace fvdb::detail::ops {
JaggedTensor coordsInGrid(const GridBatchImpl &grid, const JaggedTensor &coords);
}
```

### The `.cu` file

The internal logic is unchanged. What changes:
1. The device dispatch moves from the caller into the op's `.cu` file
2. The templated `dispatch*` function becomes `static` (file-internal) or is replaced entirely by using the new dispatch forEach tools
3. A non-templated wrapper is the public entry point
4. Explicit template instantiations at the bottom of the file are removed

For ops that currently use `BasePerActiveVoxelProcessor`, the CRTP base class is replaced by a direct call to `dispatch::forEachActiveVoxel` with a plain callback.

For complex ops (trilinear sampling, ray ops) that have custom iteration logic, the change is minimal: the existing templated function becomes `static`, and a ~5-line type-erased wrapper calls `FVDB_DISPATCH_KERNEL` to select the device and calls the static function.

---

## Deprecating SimpleOpHelper

[`SimpleOpHelper.h`](src/fvdb/detail/utils/SimpleOpHelper.h) was a useful stepping stone. It introduced several good ideas:
- Element type descriptors (`FixedElementType`, `DynamicElementType`, `ScalarElementType`)
- Output tensor allocation helpers (`makeOutTensorFromGridBatch`, `makeOutTensorFromTensor`)
- Accessor creation helpers (`makeAccessor`)

These utilities are worth keeping (or evolving) as standalone helpers in `fvdb/detail/dispatch/`. What gets deprecated is the CRTP base class pattern (`BasePerActiveVoxelProcessor`, `BasePerElementProcessor`), which is unnecessary once the dispatch forEach tools unify device dispatch. A plain `__hostdev__` callable + `dispatch::forEachActiveVoxel` is simpler and more composable than inheriting from a CRTP base.

The output allocation and accessor helpers can live in `fvdb/detail/dispatch/OutputHelpers.h` or similar, available to all ops without requiring CRTP.

---

## Op Inventory

~50 ops need migration. They fall into categories by how much the new forEach tools can simplify them.

### High simplification (currently use or can use per-voxel/per-element iteration)

These benefit most from `forEachActiveVoxel` / `forEachTensorElement`. The CRTP base class and explicit template instantiations are eliminated entirely.

| Op | Current pattern | Notes |
|:---|:----------------|:------|
| [`ActiveGridGoords`](src/fvdb/detail/ops/ActiveGridGoords.h) | `BasePerActiveVoxelProcessor` | Poster child for the new pattern |
| [`SerializeEncode`](src/fvdb/detail/ops/SerializeEncode.h) | `BasePerActiveVoxelProcessor` | |
| [`CoordsInGrid`](src/fvdb/detail/ops/CoordsInGrid.h) | Custom per-voxel | |
| [`IjkToIndex`](src/fvdb/detail/ops/IjkToIndex.h) | Custom per-voxel | |
| [`IjkToInvIndex`](src/fvdb/detail/ops/IjkToInvIndex.h) | Custom per-voxel | |
| [`PointsInGrid`](src/fvdb/detail/ops/PointsInGrid.h) | Custom per-element | |
| [`CubesInGrid`](src/fvdb/detail/ops/CubesInGrid.h) | Custom per-element | Also needs `Vec3dOrScalar` cleanup |
| [`JIdxForGrid`](src/fvdb/detail/ops/JIdxForGrid.h) | Custom per-voxel | |
| [`GridEdgeNetwork`](src/fvdb/detail/ops/GridEdgeNetwork.h) | Custom per-voxel | |
| [`Inject`](src/fvdb/detail/ops/Inject.h) | Custom per-voxel | |
| [`MortonHilbertFromIjk`](src/fvdb/detail/ops/MortonHilbertFromIjk.h) | `BasePerElementProcessor` | Already has type-erased entry point |

### Medium simplification (custom loops, use `forEachJaggedElement`)

These use `forEachJaggedElementChannel*` with custom callbacks. They benefit from `dispatch::forEachJaggedElement` (single call site) but keep their custom callback logic.

| Op | Notes |
|:---|:------|
| [`SampleGridTrilinear`](src/fvdb/detail/ops/SampleGridTrilinear.h) | `AT_DISPATCH_V2` + float4 vectorization |
| [`SampleGridTrilinearWithGrad`](src/fvdb/detail/ops/SampleGridTrilinearWithGrad.h) | |
| [`SampleGridTrilinearWithGradBackward`](src/fvdb/detail/ops/SampleGridTrilinearWithGradBackward.h) | |
| [`SampleGridBezier`](src/fvdb/detail/ops/SampleGridBezier.h) | |
| [`SampleGridBezierWithGrad`](src/fvdb/detail/ops/SampleGridBezierWithGrad.h) | |
| [`SampleGridBezierWithGradBackward`](src/fvdb/detail/ops/SampleGridBezierWithGradBackward.h) | |
| [`SplatIntoGridTrilinear`](src/fvdb/detail/ops/SplatIntoGridTrilinear.h) | |
| [`SplatIntoGridBezier`](src/fvdb/detail/ops/SplatIntoGridBezier.h) | |
| [`TransformPointToGrid`](src/fvdb/detail/ops/TransformPointToGrid.cu) | |
| [`DownsampleGridAvgPool`](src/fvdb/detail/ops/DownsampleGridAvgPool.h) | |
| [`DownsampleGridMaxPool`](src/fvdb/detail/ops/DownsampleGridMaxPool.cu) | |
| [`UpsampleGridNearest`](src/fvdb/detail/ops/UpsampleGridNearest.h) | |
| [`VoxelNeighborhood`](src/fvdb/detail/ops/VoxelNeighborhood.h) | |
| [`ReadFromDense`](src/fvdb/detail/ops/ReadFromDense.h) | Also needs `Vec3iBatch` cleanup |
| [`ReadIntoDense`](src/fvdb/detail/ops/ReadIntoDense.h) | Also needs `Vec3iBatch` cleanup |
| [`NearestIjkForPoints`](src/fvdb/detail/ops/NearestIjkForPoints.h) | |
| [`CoarseIjkForFineGrid`](src/fvdb/detail/ops/CoarseIjkForFineGrid.h) | |
| [`ActiveVoxelsInBoundsMask`](src/fvdb/detail/ops/ActiveVoxelsInBoundsMask.h) | |

### Minimal change (complex internals, just add type-erased wrapper)

These have complex multi-pass logic. The main change is adding a type-erased wrapper; internal forEach migration is optional and can be done later.

| Op | Notes |
|:---|:------|
| [`VoxelsAlongRays`](src/fvdb/detail/ops/VoxelsAlongRays.h) | Multi-output |
| [`SegmentsAlongRays`](src/fvdb/detail/ops/SegmentsAlongRays.h) | |
| [`SampleRaysUniform`](src/fvdb/detail/ops/SampleRaysUniform.h) | Many parameters |
| [`RayImplicitIntersection`](src/fvdb/detail/ops/RayImplicitIntersection.h) | |
| [`MarchingCubes`](src/fvdb/detail/ops/MarchingCubes.h) | Multi-output |
| [`IntegrateTSDF`](src/fvdb/detail/ops/IntegrateTSDF.h) | Complex multi-output |
| [`VolumeRender`](src/fvdb/detail/ops/VolumeRender.h) | |
| [`ConvolutionKernelMap`](src/fvdb/detail/ops/convolution/pack_info/ConvolutionKernelMap.h) | Also needs `Vec3iOrScalar` cleanup |

### Grid-building ops (return `GridHandle`)

| Op | Notes |
|:---|:------|
| [`BuildGridFromIjk`](src/fvdb/detail/ops/BuildGridFromIjk.h) | |
| [`BuildGridFromPoints`](src/fvdb/detail/ops/BuildGridFromPoints.h) | |
| [`BuildGridFromMesh`](src/fvdb/detail/ops/BuildGridFromMesh.h) | |
| [`BuildGridFromNearestVoxelsToPoints`](src/fvdb/detail/ops/BuildGridFromNearestVoxelsToPoints.h) | |
| [`BuildCoarseGridFromFine`](src/fvdb/detail/ops/BuildCoarseGridFromFine.h) | |
| [`BuildFineGridFromCoarse`](src/fvdb/detail/ops/BuildFineGridFromCoarse.h) | |
| [`BuildDenseGrid`](src/fvdb/detail/ops/BuildDenseGrid.h) | |
| [`BuildDilatedGrid`](src/fvdb/detail/ops/BuildDilatedGrid.h) | |
| [`BuildMergedGrids`](src/fvdb/detail/ops/BuildMergedGrids.h) | |
| [`BuildPaddedGrid`](src/fvdb/detail/ops/BuildPaddedGrid.h) | |
| [`BuildPrunedGrid`](src/fvdb/detail/ops/BuildPrunedGrid.h) | |
| [`BuildGridForConv`](src/fvdb/detail/ops/BuildGridForConv.h) | |
| [`PopulateGridMetadata`](src/fvdb/detail/ops/PopulateGridMetadata.h) | |

### JaggedTensor-only ops

| Op | Notes |
|:---|:------|
| [`JaggedTensorIndex`](src/fvdb/detail/ops/JaggedTensorIndex.h) | |
| [`JCat0`](src/fvdb/detail/ops/JCat0.h) | |
| [`JIdxForJOffsets`](src/fvdb/detail/ops/JIdxForJOffsets.h) | |
| [`JOffsetsFromJIdx`](src/fvdb/detail/ops/JOffsetsFromJIdx.h) | |
| [`IjkForMesh`](src/fvdb/detail/ops/IjkForMesh.h) | |

---

## PR Strategy

The key insight for review is: **the new dispatch tools are the only novel code**. Every subsequent op migration is a mechanical application of a proven recipe. So the PR strategy front-loads the scrutiny.

### PR 1: New dispatch tools + minimum viable op ports (small, reviewable)

**What's in it:**
- New `src/fvdb/detail/dispatch/` directory with:
  - `ForEachActiveVoxel.cuh` — device-unified active-voxel iteration over `GridBatchImpl`
  - `ForEachLeaf.cuh` — device-unified leaf iteration over `GridBatchImpl`
  - `ForEachJaggedElement.cuh` — device-unified element iteration over `JaggedTensor`
  - `ForEachTensorElement.cuh` — device-unified element iteration over `torch::Tensor`
  - `OutputHelpers.h` — output allocation and accessor utilities (extracted from `SimpleOpHelper.h`)
- Port of **3-4 ops** that cover the main patterns:
  - `ActiveGridGoords` — proves `forEachActiveVoxel` (currently `BasePerActiveVoxelProcessor`)
  - `MortonHilbertFromIjk` — proves `forEachTensorElement` (currently `BasePerElementProcessor`); already has type-erased entry point, just needs to drop `SimpleOpHelper` dependency
  - `CoordsInGrid` — proves `forEachActiveVoxel` for an op that takes additional input (currently custom per-voxel)
  - `SampleGridTrilinear` — proves the "just add a type-erased wrapper" pattern for a complex op
- Removal of `FVDB_DISPATCH_KERNEL_DEVICE` from the call sites of these 4 ops in `GridBatchImpl.cu` and `GridBatch.cpp`
- Old forEach tools remain untouched — no breakage

**What reviewers scrutinize:**
- The design of the dispatch forEach tools (the only new code)
- Whether the ported ops maintain identical behavior (tests pass)
- Whether the pattern is clear enough to replicate mechanically

**What reviewers don't need to worry about:**
- Changes to GridBatchImpl's interface
- Changes to the Python API
- Changes to the autograd layer

### PR 2-N: Bulk op ports (large, mechanical, low-scrutiny)

Once the pattern from PR 1 is approved, the remaining ~45 ops are ported in batches. Each PR:
- Picks a group of ops (5-10 at a time, grouped by pattern similarity)
- Applies the same recipe
- Removes `FVDB_DISPATCH_KERNEL_DEVICE` from corresponding call sites
- Runs tests

These PRs are "another of" — the reviewers have already approved the pattern.

### Final PR: Cleanup

- Mark `SimpleOpHelper.h`'s CRTP bases as deprecated (or delete if all consumers are ported)
- Verify no op header contains `template <torch::DeviceType>`
- Verify `FVDB_DISPATCH_KERNEL` / `FVDB_DISPATCH_KERNEL_DEVICE` no longer appears outside of `src/fvdb/detail/dispatch/` and individual op `.cu` files that need it for scalar-type dispatch
- Old per-device forEach tools can be deprecated (they're still correct, just no longer the recommended path)

---

## Relationship to the Functional Refactor

This preprocessing pass is designed to make the [functional GridBatch refactor](FunctionalRefactorProposal.md) dramatically simpler.

**Before this pass**: The functional refactor needs ~25 thin C++ wrapper functions. About 15 of them exist solely to do `FVDB_DISPATCH_KERNEL_DEVICE` because the ops expose templated headers.

**After this pass**: Ops expose type-erased functions. The functional refactor's thin C++ wrappers are only needed for the ~12 `autograd::Function::apply()` calls. For everything else, Python can call the type-erased op function through a direct pybind binding — no C++ wrapper needed at all.

Additionally, `GridBatchImpl.cu`'s derived-grid methods (`coarsen`, `upsample`, etc.) stop using `FVDB_DISPATCH_KERNEL` — they just call the type-erased op functions. This simplifies `GridBatchImpl` even before the functional refactor touches it.
