// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/GridBatchDataFactory.h>
#include <fvdb/detail/ops/BuildGridFromIjk.h>
#include <fvdb/detail/ops/BuildMergedGrids.h>
#include <fvdb/detail/ops/Inject.h>
#include <fvdb/detail/ops/PersistentTSDFState.h>

#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>

#include <vector>

namespace fvdb::detail::ops {

namespace {

// Allocate a freshly-zeroed sidecar tensor shaped `[numRows]` or
// `[numRows, trailingDim]` with the same dtype / device as `templateT`.
// Trailing dim == 0 collapses to the 1-D case (features-off path).
torch::Tensor
allocateZeroSidecar(int64_t numRows, int64_t trailingDim, const torch::Tensor &templateT) {
    std::vector<int64_t> shape;
    if (trailingDim > 0) {
        shape = {numRows, trailingDim};
    } else {
        shape = {numRows};
    }
    return torch::zeros(shape, templateT.options());
}

// Copy `src` (indexed by `srcGrid`) into a freshly-zeroed tensor
// `dst` (indexed by `dstGrid`) at the ijk-overlapping positions.
// Slots in `dst` for voxels absent from `srcGrid` are left at their
// zero-init value. Wraps `ops::inject` with the JaggedTensor plumbing
// the op expects.
void
injectSidecar(const GridBatchData &dstGrid,
              const GridBatchData &srcGrid,
              torch::Tensor &dst,
              const torch::Tensor &src) {
    TORCH_CHECK(dst.size(0) == dstGrid.totalVoxels(),
                "dst size mismatch (expected ",
                dstGrid.totalVoxels(),
                " rows, got ",
                dst.size(0),
                ")");
    TORCH_CHECK(src.size(0) == srcGrid.totalVoxels(),
                "src size mismatch (expected ",
                srcGrid.totalVoxels(),
                " rows, got ",
                src.size(0),
                ")");
    JaggedTensor dstJt = dstGrid.jaggedTensor(dst);
    JaggedTensor srcJt = srcGrid.jaggedTensor(src);
    ops::inject(dstGrid, srcGrid, dstJt, srcJt);
    // `ops::inject` may swap the underlying tensor inside dstJt; pull the
    // (possibly-new) tensor back out into our output reference.
    dst = dstJt.jdata();
}

} // namespace

PersistentTSDFState::PersistentTSDFState(c10::intrusive_ptr<GridBatchData> grid,
                                         torch::Tensor tsdf,
                                         torch::Tensor weights,
                                         std::optional<torch::Tensor> features)
    : mGrid(std::move(grid)), mTsdf(std::move(tsdf)), mWeights(std::move(weights)) {
    TORCH_CHECK(mGrid != nullptr, "PersistentTSDFState requires a non-null grid");
    TORCH_CHECK_VALUE(mTsdf.size(0) == mGrid->totalVoxels(),
                      "tsdf size(0) (",
                      mTsdf.size(0),
                      ") must equal grid.totalVoxels() (",
                      mGrid->totalVoxels(),
                      ")");
    TORCH_CHECK_VALUE(mWeights.size(0) == mGrid->totalVoxels(),
                      "weights size(0) (",
                      mWeights.size(0),
                      ") must equal grid.totalVoxels() (",
                      mGrid->totalVoxels(),
                      ")");
    TORCH_CHECK_TYPE(mWeights.scalar_type() == mTsdf.scalar_type(),
                     "weights dtype (",
                     mWeights.scalar_type(),
                     ") must match tsdf dtype (",
                     mTsdf.scalar_type(),
                     ")");
    if (features.has_value() && features.value().defined() &&
        features.value().numel() > 0) {
        mHasFeatures = true;
        mFeatures    = features.value();
        TORCH_CHECK_VALUE(mFeatures.dim() == 2,
                          "features must be 2-D [totalVoxels, featureDim]");
        TORCH_CHECK_VALUE(mFeatures.size(0) == mGrid->totalVoxels(),
                          "features size(0) (",
                          mFeatures.size(0),
                          ") must equal grid.totalVoxels() (",
                          mGrid->totalVoxels(),
                          ")");
    } else {
        mHasFeatures = false;
        // Maintain a well-shaped `[totalVoxels, 0]` placeholder so that
        // `grid().jaggedTensor(features())` works uniformly and so callers
        // can pass `features()` into the `GridBatchData::jaggedTensor`
        // size check even when features are disabled. Matches the
        // placeholder convention already used in `IntegrateTSDF.cu`
        // (`torch::empty({0, 0}, opts)` + `GridBatchData::jaggedTensor`
        // size-check pitfall, documented in paper_extractions impl-notes
        // entry #12).
        mFeatures = torch::empty({mGrid->totalVoxels(), 0}, mTsdf.options());
    }
}

void
PersistentTSDFState::grow(const JaggedTensor &newVoxelIjks) {
    TORCH_CHECK_VALUE(newVoxelIjks.rdim() == 2 && newVoxelIjks.rsize(-1) == 3,
                      "grow(ijks): ijks must have element shape [-1, 3]");
    TORCH_CHECK_VALUE(newVoxelIjks.num_outer_lists() == mGrid->batchSize(),
                      "grow(ijks): batch size mismatch (ijks.num_outer_lists=",
                      newVoxelIjks.num_outer_lists(),
                      " grid.batchSize=",
                      mGrid->batchSize(),
                      ")");
    if (newVoxelIjks.jdata().size(0) == 0) {
        // Zero-voxel shell: nothing to merge in.
        return;
    }
    std::vector<nanovdb::Vec3d> voxelSizes;
    std::vector<nanovdb::Vec3d> origins;
    mGrid->gridVoxelSizesAndOrigins(voxelSizes, origins);
    auto shellGrid = createNanoGridFromIJK(newVoxelIjks, voxelSizes, origins);
    growFromGrid(*shellGrid);
}

void
PersistentTSDFState::growFromGrid(const GridBatchData &shellGrid) {
    if (shellGrid.totalVoxels() == 0) {
        return;
    }
    TORCH_CHECK_VALUE(shellGrid.batchSize() == mGrid->batchSize(),
                      "growFromGrid: shell batchSize (",
                      shellGrid.batchSize(),
                      ") must equal live batchSize (",
                      mGrid->batchSize(),
                      ")");
    TORCH_CHECK_VALUE(shellGrid.device() == mGrid->device(),
                      "growFromGrid: shell/live must be on the same device");

    const c10::cuda::OptionalCUDAGuard device_guard(
        mGrid->device().is_cuda() ? std::optional<torch::Device>(mGrid->device()) : std::nullopt);

    // `mergeGrids` builds the set-union of the two input grids' active
    // voxels. When the shell is a strict subset of the live grid the
    // merged grid is structurally identical to the live grid (same
    // ordered active voxel set) and `totalVoxels()` matches, which we
    // use as the no-op fast path below. This is the hot steady-state
    // case on long trajectories: after the first ~50-100 frames the
    // truncation shell stops introducing novel voxels and we skip both
    // the realloc and the inject pass entirely.
    //
    // Argument order matters: `mergeGrids(shellGrid, mGrid)` iterates
    // the shell's voxels first in the output (per-leaf) ordering, which
    // matches the single-frame `integrateTSDFImpl` path's
    // `ops::mergeGrids(*pointGrid, *grid)` convention. This keeps the
    // batched path bit-identical to the sequential one --
    // `test_integrate_tsdf_frames_matches_sequential` fails (at the
    // ~1e-7 atol level, so order-of-sum sensitivity of the weighted
    // TSDF update) if we swap it to `(mGrid, shell)`.
    auto mergedGrid = mergeGrids(shellGrid, *mGrid);

    // The "overlap-only fast path" -- return early when the merged
    // grid's voxel set exactly matches the live grid's -- is a
    // tempting optimization (avoid the realloc + inject) but in
    // practice introduces a semantic divergence with the sequential-
    // path TSDF output: weight sidecars end up with absolute errors
    // of up to one frame's worth of `new_observation_weight` on
    // multiple-percent of voxels.
    //
    // Hypothesis: when the fast path fires, `state.grid()` retains
    // the *previous* merged-grid `GridBatchData` object, whereas the
    // sequential path constructs a fresh `mergeGrids(shell, base)`
    // result every frame. Even when both produce the same voxel set
    // and enumeration order, there is an internal `GridBatchData`
    // bookkeeping difference that affects what
    // `grid.deviceAccessor().getValue(ijk)` returns for specific
    // voxels in specific frames, causing shell voxels to look up to
    // the wrong linear index and either miss the update or double-
    // count.
    //
    // Disabling the fast path costs us the steady-state speedup on
    // bounded trajectories but keeps the output bit-identical to the
    // sequential reference. TODO: revisit when we have a cheap way
    // to detect "merged grid is structurally identical to base in
    // ALL respects, including internal bookkeeping" -- likely needs
    // a deeper look at `nanovdb::tools::cuda::MergeGrids`'s output
    // layout vs a grid's original construction.
    if (false && mergedGrid->totalVoxels() == mGrid->totalVoxels()) {
        return;
    }

    const int64_t newTotal    = mergedGrid->totalVoxels();
    const int64_t featureDim  = mHasFeatures ? mFeatures.size(1) : 0;

    torch::Tensor newTsdf    = allocateZeroSidecar(newTotal, 0, mTsdf);
    torch::Tensor newWeights = allocateZeroSidecar(newTotal, 0, mWeights);

    injectSidecar(*mergedGrid, *mGrid, newTsdf, mTsdf);
    injectSidecar(*mergedGrid, *mGrid, newWeights, mWeights);

    torch::Tensor newFeatures;
    if (mHasFeatures) {
        newFeatures = allocateZeroSidecar(newTotal, featureDim, mFeatures);
        injectSidecar(*mergedGrid, *mGrid, newFeatures, mFeatures);
    } else {
        // Keep the `[totalVoxels, 0]` placeholder aligned with the new grid
        // so the `jaggedTensor` size check continues to pass.
        newFeatures = torch::empty({newTotal, 0}, mTsdf.options());
    }

    mGrid     = mergedGrid;
    mTsdf     = newTsdf;
    mWeights  = newWeights;
    mFeatures = newFeatures;
}

void
PersistentTSDFState::reset() {
    std::vector<nanovdb::Vec3d> voxelSizes;
    std::vector<nanovdb::Vec3d> origins;
    mGrid->gridVoxelSizesAndOrigins(voxelSizes, origins);
    const auto device = mGrid->device();
    if (voxelSizes.empty()) {
        mGrid = makeEmptyGridBatchData(device);
    } else {
        mGrid = makeEmptyGridBatchData(device, voxelSizes, origins);
    }
    mTsdf    = torch::empty({0}, mTsdf.options());
    mWeights = torch::empty({0}, mWeights.options());
    if (mHasFeatures) {
        mFeatures = torch::empty({0, mFeatures.size(1)}, mFeatures.options());
    } else {
        mFeatures = torch::empty({0, 0}, mTsdf.options());
    }
}

} // namespace fvdb::detail::ops
