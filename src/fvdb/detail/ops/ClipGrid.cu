// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/GridBatchDataFactory.h>
#include <fvdb/detail/ops/ActiveVoxelsInBoundsMask.h>
#include <fvdb/detail/ops/BuildPrunedGrid.h>
#include <fvdb/detail/ops/ClipGrid.h>

namespace fvdb {
namespace detail {
namespace ops {

std::tuple<c10::intrusive_ptr<GridBatchData>, JaggedTensor>
clipGridWithMask(const GridBatchData &grid,
                 const std::vector<nanovdb::Coord> &ijkMin,
                 const std::vector<nanovdb::Coord> &ijkMax) {
    JaggedTensor activeVoxelMask = activeVoxelsInBoundsMask(grid, ijkMin, ijkMax);

    if (grid.batchSize() == 0) {
        return std::make_tuple(makeEmptyGridBatchData(grid.device()), activeVoxelMask);
    }

    // Clipping keeps a subset of the source voxels, so prune the grid to the per-voxel in-bounds
    // mask directly (leaf-mask morphology) rather than materializing every active coordinate and
    // rebuilding via a radix sort. pruneGrid preserves the source transform and the canonical
    // voxel order, so the features rmask'd in clipGridFeaturesWithMask stay row-aligned with the
    // clipped grid. (Both this and activeVoxelsInBoundsMask are CPU+CUDA only; clip never
    // supported PrivateUse1.)
    auto clippedGridPtr = pruneGrid(grid, activeVoxelMask);
    return std::make_tuple(clippedGridPtr, activeVoxelMask);
}

c10::intrusive_ptr<GridBatchData>
clipGrid(const GridBatchData &grid,
         const std::vector<nanovdb::Coord> &ijkMin,
         const std::vector<nanovdb::Coord> &ijkMax) {
    auto [clippedGridPtr, activeVoxelMask] = clipGridWithMask(grid, ijkMin, ijkMax);
    return clippedGridPtr;
}

std::pair<JaggedTensor, c10::intrusive_ptr<GridBatchData>>
clipGridFeaturesWithMask(const GridBatchData &grid,
                         const JaggedTensor &features,
                         const std::vector<nanovdb::Coord> &ijkMin,
                         const std::vector<nanovdb::Coord> &ijkMax) {
    TORCH_CHECK_VALUE(features.ldim() == 1,
                      "Expected features to have 1 list dimension, i.e. be a single list of "
                      "coordinate values, but got",
                      features.ldim(),
                      "list dimensions");
    grid.checkDevice(features);
    TORCH_CHECK(features.rsize(0) == grid.totalVoxels(),
                "Value count of features does not match grid");
    TORCH_CHECK(features.num_outer_lists() == grid.batchSize(),
                "Batch size of features does not match grid.");
    TORCH_CHECK(torch::equal(features.joffsets(), grid.voxelOffsets()),
                "Offsets of features does not match grid.");

    auto [clippedGridPtr, activeVoxelMask] = clipGridWithMask(grid, ijkMin, ijkMax);
    JaggedTensor clippedFeatures           = features.rmask(activeVoxelMask.jdata());
    return {clippedFeatures, clippedGridPtr};
}

} // namespace ops
} // namespace detail
} // namespace fvdb
