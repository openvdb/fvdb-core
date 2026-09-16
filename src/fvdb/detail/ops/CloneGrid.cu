// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/GridBatchDataFactory.h>
#include <fvdb/detail/ops/CloneGrid.h>
#include <fvdb/detail/ops/MakeContiguous.h>

namespace fvdb {
namespace detail {
namespace ops {

c10::intrusive_ptr<GridBatchData>
cloneGrid(const GridBatchData &grid, const torch::Device &device, bool blocking) {
    if (grid.batchSize() == 0) {
        return makeEmptyGridBatchData(device);
    }

    // Compact the (possibly sliced/non-contiguous) selected grids into fresh contiguous storage.
    // Copying the whole storage would copy *every physical grid* it shares -- wrong (and a
    // voxelSizes/gridCount mismatch) for an indexed batch, where gridCount() > batchSize().
    GridStorage cloned = contiguousGridStorage(grid);
    if (cloned.device() != device) {
        // A different target device: the storage is now contiguous, so moving it moves exactly
        // the selected grids. Complete on return for a host destination; for a device destination
        // ordered on its current torch stream, which is where the batch's later work runs.
        cloned = cloned.to(device);
    }
    std::vector<nanovdb::Vec3d> voxelSizes, voxelOrigins;
    grid.gridVoxelSizesAndOrigins(voxelSizes, voxelOrigins);
    return makeGridBatchData(std::move(cloned), voxelSizes, voxelOrigins);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
