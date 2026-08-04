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

    // Compact the (possibly sliced/non-contiguous) selected grids into a fresh contiguous handle.
    // nanoGridHandle().copy() would copy *every physical grid* in the shared handle -- wrong (and a
    // voxelSizes/gridCount mismatch) for an indexed batch, where gridCount() > batchSize().
    nanovdb::GridHandle<TorchDeviceBuffer> clonedHdl = contiguousGridHandle(grid);
    if (clonedHdl.buffer().device() != device) {
        // Requested a different target device: the handle is now contiguous, so a whole-handle copy
        // to `device` moves exactly the selected grids.
        TorchDeviceBuffer guide(0, device);
        clonedHdl = clonedHdl.copy<TorchDeviceBuffer>(guide);
    }

    std::vector<nanovdb::Vec3d> voxelSizes, voxelOrigins;
    grid.gridVoxelSizesAndOrigins(voxelSizes, voxelOrigins);

    return makeContiguous(makeGridBatchData(std::move(clonedHdl), voxelSizes, voxelOrigins));
}

} // namespace ops
} // namespace detail
} // namespace fvdb
