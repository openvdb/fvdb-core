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
    const GridStorage &storage = input.gridStorage();
    const auto *base           = static_cast<const uint8_t *>(
        input.device().is_cpu() ? storage.hostBytes() : storage.deviceBytes());
    GridSpanSource source{input.device(), storage.stream(), {}};
    source.spans.reserve(count);
    for (int64_t i = first; i < first + count; ++i) {
        source.spans.push_back(
            GridSpan{base + input.cumBytesAt(i), input.numBytesAt(i), nanovdb::GridType::OnIndex});
    }
    return source;
}

} // namespace

GridStorage
contiguousGridStorage(const GridBatchData &input, std::optional<torch::Device> device) {
    const torch::Device target = GridStorage::resolveDevice(device.value_or(input.device()));
    const cudaStream_t stream =
        target.is_cpu() ? input.gridStorage().stream() : storageStream(target);
    return assembleGridStorage({logicalGridSpans(input, 0, input.batchSize())}, target, stream);
}

GridStorage
cloneGridStorageAt(const GridBatchData &input, int64_t i) {
    const torch::Device device = input.device();
    const cudaStream_t stream =
        device.is_cpu() ? input.gridStorage().stream() : storageStream(device);
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
