// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/GridBatchDataFactory.h>
#include <fvdb/detail/ops/ConcatenateGrids.h>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/CreateNanoGrid.h>

#include <c10/cuda/CUDAGuard.h>

namespace {

__global__ void
updateGridCountAndZeroChecksum(nanovdb::GridData *d_data, uint32_t gridIndex, uint32_t gridCount) {
    NANOVDB_ASSERT(gridIndex < gridCount);
    if (d_data->mGridIndex != gridIndex || d_data->mGridCount != gridCount) {
        d_data->mGridIndex = gridIndex;
        d_data->mGridCount = gridCount;
    }
    d_data->mChecksum.disable();
}

} // namespace

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

    TorchDeviceBuffer buffer(totalByteSize, device);

    int count         = 0;
    int nonEmptyCount = 0;
    if (device.is_cpu()) {
        for (size_t i = 0; i < elements.size(); i += 1) {
            if (elements[i]->batchSize() == 0) {
                continue;
            }
            for (int64_t j = 0; j < elements[i]->batchSize(); j += 1) {
                const int64_t readOffset  = readByteOffsets[nonEmptyCount][j];
                const int64_t writeOffset = writeByteOffsets[nonEmptyCount][j];
                const int64_t numBytes    = byteSizes[nonEmptyCount][j];

                nanovdb::GridData *dst =
                    reinterpret_cast<nanovdb::GridData *>(buffer.data() + writeOffset);
                const uint8_t *src =
                    elements[i]->nanoGridHandle().buffer().data() + readOffset;
                memcpy((void *)dst, (void *)src, numBytes);
                nanovdb::tools::updateGridCount(dst, count++, totalGrids);
            }
            nonEmptyCount += 1;
        }
    } else {
        TORCH_CHECK(device.has_index(), "Device must have an index for CUDA operations");
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(device.index());

        for (size_t i = 0; i < elements.size(); i += 1) {
            if (elements[i]->batchSize() == 0) {
                continue;
            }
            for (int64_t j = 0; j < elements[i]->batchSize(); j += 1) {
                const int64_t readOffset  = readByteOffsets[nonEmptyCount][j];
                const int64_t writeOffset = writeByteOffsets[nonEmptyCount][j];
                const int64_t numBytes    = byteSizes[nonEmptyCount][j];

                c10::cuda::CUDAGuard deviceGuard(device.index());
                nanovdb::GridData *dst =
                    reinterpret_cast<nanovdb::GridData *>(buffer.deviceData() + writeOffset);
                const uint8_t *src =
                    elements[i]->nanoGridHandle().buffer().deviceData() + readOffset;
                cudaMemcpyAsync((uint8_t *)dst, src, numBytes, cudaMemcpyDeviceToDevice, stream);

                updateGridCountAndZeroChecksum<<<1, 1, 0, stream>>>(dst, count++, totalGrids);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
            nonEmptyCount += 1;
        }
    }
    nanovdb::GridHandle<TorchDeviceBuffer> gridHdl =
        nanovdb::GridHandle<TorchDeviceBuffer>(std::move(buffer));
    return makeGridBatchData(std::move(gridHdl), voxelSizes, voxelOrigins);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
