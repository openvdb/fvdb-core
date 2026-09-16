// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/GridBatchDataFactory.h>
#include <fvdb/detail/ops/ConcatenateGrids.h>
#include <fvdb/detail/utils/cuda/StreamOrdering.h>
#include <fvdb/detail/utils/nanovdb/DeviceGridHandleUtils.cuh>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/GridChecksum.h>

#include <c10/cuda/CUDAGuard.h>

namespace fvdb {
namespace detail {
namespace ops {

c10::intrusive_ptr<GridBatchData>
concatenateGrids(const std::vector<c10::intrusive_ptr<GridBatchData>> &elements) {
    TORCH_CHECK_VALUE(elements.size() > 0, "Must provide at least one grid for concatenate!")

    torch::Device device = elements[0]->device();

    std::vector<std::vector<int64_t>> byteSizes;
    std::vector<std::vector<int64_t>> readByteOffsets;
    std::vector<std::vector<int64_t>> writeByteOffsets;
    int64_t totalByteSize     = 0;
    int64_t totalGrids        = 0;
    int64_t nonEmptyCount_pre = 0;
    byteSizes.reserve(elements.size());
    readByteOffsets.reserve(elements.size());
    writeByteOffsets.reserve(elements.size());

    std::vector<nanovdb::Vec3d> voxelSizes, voxelOrigins;

    for (size_t i = 0; i < elements.size(); i += 1) {
        TORCH_CHECK(elements[i]->device() == device,
                    "All grid batches must be on the same device!");

        if (elements[i]->batchSize() == 0) {
            continue;
        }

        readByteOffsets.push_back(std::vector<int64_t>());
        writeByteOffsets.push_back(std::vector<int64_t>());
        byteSizes.push_back(std::vector<int64_t>());
        readByteOffsets.back().reserve(elements[i]->batchSize());
        writeByteOffsets.back().reserve(elements[i]->batchSize());
        byteSizes.back().reserve(elements[i]->batchSize());

        totalGrids += elements[i]->batchSize();

        for (int64_t j = 0; j < elements[i]->batchSize(); j += 1) {
            voxelSizes.push_back(elements[i]->voxelSizeAt(j));
            voxelOrigins.push_back(elements[i]->voxelOriginAt(j));

            readByteOffsets.back().push_back(elements[i]->cumBytesAt(j));
            byteSizes.back().push_back(elements[i]->numBytesAt(j));
            writeByteOffsets.back().push_back(totalByteSize);
            totalByteSize += elements[i]->numBytesAt(j);
        }
        nonEmptyCount_pre += 1;
    }
    if (nonEmptyCount_pre == 0) {
        return makeEmptyGridBatchData(device);
    }

    int count         = 0;
    int nonEmptyCount = 0;
    if (device.is_cpu()) {
        nanovdb::HostBuffer buffer(totalByteSize);
        for (size_t i = 0; i < elements.size(); i += 1) {
            if (elements[i]->batchSize() == 0) {
                continue;
            }
            const auto *srcBase =
                static_cast<const uint8_t *>(elements[i]->gridStorage().hostBytes());
            for (int64_t j = 0; j < elements[i]->batchSize(); j += 1) {
                const int64_t readOffset  = readByteOffsets[nonEmptyCount][j];
                const int64_t writeOffset = writeByteOffsets[nonEmptyCount][j];
                const int64_t numBytes    = byteSizes[nonEmptyCount][j];
                nanovdb::GridData *dst    = reinterpret_cast<nanovdb::GridData *>(
                    static_cast<uint8_t *>(buffer.data()) + writeOffset);
                memcpy((void *)dst, (const void *)(srcBase + readOffset), numBytes);
                nanovdb::tools::updateGridCount(dst, count++, totalGrids);
            }
            nonEmptyCount += 1;
        }
        // Parsing the chain validates the headers just written.
        return makeGridBatchData(
            GridStorage(GridStorage::HostHandle(std::move(buffer))), voxelSizes, voxelOrigins);
    }

    TORCH_CHECK(!device.is_cuda() || device.has_index(),
                "Device must have an index for CUDA operations");
    c10::DeviceGuard deviceGuard(device);
    const cudaStream_t stream = storageStream(device);
    DeviceGridBuffer buffer(
        stream, TorchDeviceResource(device), totalByteSize, nanovdb::cuda::noInit);
    std::vector<nanovdb::GridHandleMetaData> meta;
    meta.reserve(totalGrids);
    for (size_t i = 0; i < elements.size(); i += 1) {
        if (elements[i]->batchSize() == 0) {
            continue;
        }
        // Order the copies after this element's writers (its storage's retained stream), and that
        // stream after the copies, so the element may be released as soon as this returns.
        const cudaStream_t srcStream = elements[i]->gridStorage().stream();
        orderStreamAfter(stream, device, srcStream, device);
        const auto *srcBase =
            static_cast<const uint8_t *>(elements[i]->gridStorage().deviceBytes());
        for (int64_t j = 0; j < elements[i]->batchSize(); j += 1) {
            const int64_t readOffset  = readByteOffsets[nonEmptyCount][j];
            const int64_t writeOffset = writeByteOffsets[nonEmptyCount][j];
            const int64_t numBytes    = byteSizes[nonEmptyCount][j];
            nanovdb::GridData *dst =
                reinterpret_cast<nanovdb::GridData *>(buffer.data() + writeOffset);
            C10_CUDA_CHECK(cudaMemcpyAsync(
                (uint8_t *)dst, srcBase + readOffset, numBytes, cudaMemcpyDeviceToDevice, stream));
            updateGridCountAndZeroChecksum<<<1, 1, 0, stream>>>(dst, count++, totalGrids);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            meta.push_back(nanovdb::GridHandleMetaData{static_cast<uint64_t>(writeOffset),
                                                       static_cast<uint64_t>(numBytes),
                                                       nanovdb::GridType::OnIndex});
        }
        orderStreamAfter(srcStream, device, stream, device);
        nonEmptyCount += 1;
    }
    if (device.is_privateuseone()) {
        // Unified memory read by host code straight away (populateGridMetadata runs on the host
        // for PrivateUse1): the copies and header kernels must have landed.
        synchronizeStream(stream, device);
    }
    // The layout is known, so the handle adopts it: no parse kernel, no synchronization.
    return makeGridBatchData(
        GridStorage(makeDeviceHandleFromLayout(std::move(buffer), std::move(meta)), device),
        voxelSizes,
        voxelOrigins);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
