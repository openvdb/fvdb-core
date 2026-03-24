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

    TorchDeviceBuffer guide(0, device);
    nanovdb::GridHandle<TorchDeviceBuffer> clonedHdl =
        grid.nanoGridHandle().copy<TorchDeviceBuffer>(guide);

    std::vector<nanovdb::Vec3d> voxelSizes, voxelOrigins;
    grid.gridVoxelSizesAndOrigins(voxelSizes, voxelOrigins);

    return makeContiguous(
        makeGridBatchData(std::move(clonedHdl), voxelSizes, voxelOrigins));
}

} // namespace ops
} // namespace detail
} // namespace fvdb
