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
    // One pass either way. A contiguous batch's storage already holds exactly the logical grids,
    // so GridStorage::to (a deep copy on the same device, a move otherwise) is the clone; a view
    // is compacted straight onto the target device.
    GridStorage cloned =
        grid.isContiguous() ? grid.gridStorage().to(device) : contiguousGridStorage(grid, device);
    std::vector<nanovdb::Vec3d> voxelSizes, voxelOrigins;
    grid.gridVoxelSizesAndOrigins(voxelSizes, voxelOrigins);
    return makeGridBatchData(std::move(cloned), voxelSizes, voxelOrigins);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
