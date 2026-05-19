# fvdb-local overrides for nanoVDB headers

This directory holds **modified copies of a small number of upstream nanoVDB
headers**. The build prepends this directory to the include search path before
the upstream nanoVDB source tree, so any `#include <nanovdb/...>` that matches
a file in this tree resolves here. Everything else falls through to upstream.

The wiring lives in `src/cmake/get_nanovdb.cmake`.

## Why not just patch the upstream checkout?

We previously tried `CPM`'s `PATCH_COMMAND` to fix a specific bug (nanoVDB
scratch allocations going through `cudaMallocAsync` instead of PyTorch's
caching allocator, which fragments into two pools and OOMs). Patch-based
approaches are fragile: they silently stop applying if upstream moves the
surrounding lines, and they make it hard to edit nanoVDB during development.

Forking the exact headers we care about into this tree gives us full edit
access with a narrow, reviewable patch surface and a clean diff-against-upstream
workflow.

## Layout

The directory structure **mirrors upstream** starting from the `nanovdb/`
include root:

```
nanovdb_overrides/
  nanovdb/
    cuda/
      DeviceBuffer.h     # forked: device-handle alloc -> c10::cuda::CUDACachingAllocator
      DeviceResource.h   # forked: scratch alloc      -> c10::cuda::CUDACachingAllocator
    tools/
      cuda/
        TopologyBuilder.cuh  # forked: opt-in scratch-size trace (FVDB_NANOVDB_TRACE_ALLOCS)
    ...                  # add more as needed
```

## Adding a new override

1. Copy the upstream header from
   `build/<...>/_deps/nanovdb-src/nanovdb/nanovdb/<rel_path>` into
   `nanovdb_overrides/nanovdb/<rel_path>`, preserving the relative path.
2. Add a short `FVDB FORK:` banner at the top documenting *what* diverges from
   upstream and *why*. Keep the rest of the file byte-identical so a future
   resync with upstream is a clean 3-way merge.
3. Every non-trivial code change inside the file should be tagged with an
   inline `// FVDB FORK:` comment pointing at the banner, so `git blame` and
   text searches make the delta obvious.

## Resyncing with upstream

When bumping the nanoVDB pin in `get_nanovdb.cmake`:

1. `diff` each file in this directory against its upstream counterpart.
2. Port the upstream changes over, keeping the `FVDB FORK` deltas.
3. Rebuild + rerun the fvdb test suite.

## Current overrides

| File                                       | Reason                                                                                                       |
|--------------------------------------------|--------------------------------------------------------------------------------------------------------------|
| `nanovdb/cuda/DeviceBuffer.h`              | Route device-handle allocations through PyTorch's caching allocator (avoids dual-pool OOM).                  |
| `nanovdb/cuda/DeviceResource.h`            | Same allocator routing for the per-point scratch buffers used by `PointsToGrid` / `DilateGrid` / `MergeGrids`. |
| `nanovdb/tools/cuda/TopologyBuilder.cuh`   | Opt-in `FVDB_NANOVDB_TRACE_ALLOCS` print of tile count + scratch size from `allocateInternalMaskBuffers`.    |
