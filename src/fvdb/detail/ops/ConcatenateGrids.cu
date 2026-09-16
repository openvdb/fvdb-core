// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/GridBatchDataFactory.h>
#include <fvdb/detail/ops/ConcatenateGrids.h>
#include <fvdb/detail/utils/cuda/StreamOrdering.h>
#include <fvdb/detail/utils/nanovdb/DeviceGridHandleUtils.cuh>

#include <nanovdb/NanoVDB.h>

#include <c10/cuda/CUDAGuard.h>

namespace fvdb {
namespace detail {
namespace ops {

c10::intrusive_ptr<GridBatchData>
concatenateGrids(const std::vector<c10::intrusive_ptr<GridBatchData>> &elements) {
    TORCH_CHECK_VALUE(elements.size() > 0, "Must provide at least one grid for concatenate!")

    torch::Device device = elements[0]->device();
    // Every element's logical grids, as spans of its storage: the assembler orders once per
    // element stream, copies, renumbers and wraps.
    std::vector<GridSpanSource> sources;
    sources.reserve(elements.size());
    std::vector<nanovdb::Vec3d> voxelSizes, voxelOrigins;
    for (const auto &element: elements) {
        TORCH_CHECK(element->device() == device, "All grid batches must be on the same device!");
        if (element->batchSize() == 0) {
            continue;
        }
        const GridStorage &storage = element->gridStorage();
        const auto *base = static_cast<const uint8_t *>(device.is_cpu() ? storage.hostBytes()
                                                                        : storage.deviceBytes());
        GridSpanSource source{device, storage.stream(), {}};
        source.spans.reserve(element->batchSize());
        for (int64_t j = 0; j < element->batchSize(); j += 1) {
            voxelSizes.push_back(element->voxelSizeAt(j));
            voxelOrigins.push_back(element->voxelOriginAt(j));
            source.spans.push_back(GridSpan{
                base + element->cumBytesAt(j), element->numBytesAt(j), nanovdb::GridType::OnIndex});
        }
        sources.push_back(std::move(source));
    }
    if (sources.empty()) {
        return makeEmptyGridBatchData(device);
    }
    const cudaStream_t stream =
        device.is_cpu() ? elements[0]->gridStorage().stream() : storageStream(device);
    return makeGridBatchData(
        assembleGridStorage(sources, device, stream), voxelSizes, voxelOrigins);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
