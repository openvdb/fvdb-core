// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/GridBatchDataFactory.h>
#include <fvdb/detail/ops/MakeContiguous.h>
#include <fvdb/detail/utils/cuda/StreamOrdering.h>
#include <fvdb/detail/utils/nanovdb/DeviceGridHandleUtils.cuh>

namespace fvdb {
namespace detail {
namespace ops {

namespace {

// The batch's logical grids as spans of its storage, from the side the copy will read.
GridSpanSource
logicalGridSpans(const GridBatchData &input, int64_t first, int64_t count) {
    const bool onHost = input.device().is_cpu();
    GridSpanSource source{input.device(), input.storageStream(), {}};
    source.spans.reserve(count);
    for (int64_t i = first; i < first + count; ++i) {
        const void *grid = onHost ? input.hostGridPtrAt(i) : input.deviceGridPtrAt(i);
        source.spans.push_back(GridSpan{grid, input.numBytesAt(i), nanovdb::GridType::OnIndex});
    }
    return source;
}

} // namespace

GridStorage
contiguousGridStorage(const GridBatchData &input, std::optional<torch::Device> device) {
    const torch::Device target = GridStorage::resolveDevice(device.value_or(input.device()));
    const cudaStream_t stream  = target.is_cpu() ? input.storageStream() : storageStream(target);
    if (input.isContiguous() || input.batchSize() == 0) {
        // The storage holds exactly the logical grids with validated headers (for an empty batch,
        // its one voxel-less grid): one copy, no header rewrite.
        return input.gridStorage().to(target, stream);
    }
    return assembleGridStorage({logicalGridSpans(input, 0, input.batchSize())}, target, stream);
}

GridStorage
cloneGridStorageAt(const GridBatchData &input, int64_t i) {
    const torch::Device device = input.device();
    const cudaStream_t stream  = device.is_cpu() ? input.storageStream() : storageStream(device);
    return assembleGridStorage({logicalGridSpans(input, i, 1)}, device, stream);
}

c10::intrusive_ptr<GridBatchData>
makeContiguous(c10::intrusive_ptr<GridBatchData> input) {
    if (input->isContiguous()) {
        return input;
    }
    std::vector<nanovdb::Vec3d> voxelSizes, voxelOrigins;
    input->gridVoxelSizesAndOrigins(voxelSizes, voxelOrigins);
    return makeGridBatchData(contiguousGridStorage(*input), voxelSizes, voxelOrigins);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
