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
cloneGrid(const GridBatchData &grid, const torch::Device &requested, bool blocking) {
    // `torch.device("cuda")` arrives index-less; it means the current CUDA device.
    const torch::Device device = GridStorage::resolveDevice(requested);
    if (grid.batchSize() == 0) {
        return makeEmptyGridBatchData(device);
    }

    // Compact the (possibly sliced/non-contiguous) selected grids into fresh contiguous storage.
    // Copying the whole storage would copy *every physical grid* it shares -- wrong (and a
    // voxelSizes/gridCount mismatch) for an indexed batch, where gridCount() > batchSize().
    // A contiguous batch's storage already holds exactly the logical grids, so one GridStorage::to
    // (a deep copy on the same device, a move otherwise) is the clone. A view is compacted on its
    // own device first and moved if the target differs: two passes until compaction takes a
    // destination device.
    GridStorage cloned =
        grid.isContiguous() ? grid.gridStorage().to(device) : contiguousGridStorage(grid);
    if (cloned.device() != device) {
        cloned = cloned.to(device);
    }
    std::vector<nanovdb::Vec3d> voxelSizes, voxelOrigins;
    grid.gridVoxelSizesAndOrigins(voxelSizes, voxelOrigins);
    return makeGridBatchData(std::move(cloned), voxelSizes, voxelOrigins);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
