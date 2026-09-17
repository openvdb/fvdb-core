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

    // The selected grids as fresh storage on the target device, one pass either way: a
    // contiguous batch's storage is copied whole, a sliced or non-contiguous view (whose storage
    // holds more physical grids than the batch selects) is compacted straight onto the device.
    GridStorage cloned = contiguousGridStorage(grid, device);
    std::vector<nanovdb::Vec3d> voxelSizes, voxelOrigins;
    grid.gridVoxelSizesAndOrigins(voxelSizes, voxelOrigins);
    return makeGridBatchData(std::move(cloned), voxelSizes, voxelOrigins);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
