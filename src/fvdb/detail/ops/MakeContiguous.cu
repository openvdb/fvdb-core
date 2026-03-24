// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/GridBatchDataFactory.h>
#include <fvdb/detail/ops/MakeContiguous.h>

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
makeContiguous(c10::intrusive_ptr<GridBatchData> input) {
    c10::DeviceGuard guard(input->device());
    if (input->isContiguous()) {
        return input;
    }

    const int64_t totalGrids = input->batchSize();

    int64_t totalByteSize = 0;
    for (int64_t i = 0; i < input->batchSize(); i += 1) {
        totalByteSize += input->numBytesAt(i);
    }

    TorchDeviceBuffer buffer(totalByteSize, input->device());

    int64_t writeOffset = 0;
    std::vector<nanovdb::Vec3d> voxelSizes, voxelOrigins;
    voxelSizes.reserve(input->batchSize());
    voxelOrigins.reserve(input->batchSize());

    if (input->device().is_cpu()) {
        for (int64_t i = 0; i < input->batchSize(); i += 1) {
            voxelSizes.push_back(input->voxelSizeAt(i));
            voxelOrigins.push_back(input->voxelOriginAt(i));

            nanovdb::GridData *dst =
                reinterpret_cast<nanovdb::GridData *>(buffer.data() + writeOffset);
            const uint8_t *src = input->nanoGridHandle().buffer().data() + input->cumBytesAt(i);
            memcpy((void *)dst, (void *)src, input->numBytesAt(i));
            nanovdb::tools::updateGridCount(dst, i, totalGrids);
            writeOffset += input->numBytesAt(i);
        }
    } else {
        TORCH_CHECK(input->device().has_index(), "Device must have an index for CUDA operations");
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(input->device().index());

        for (int64_t i = 0; i < input->batchSize(); i += 1) {
            voxelSizes.push_back(input->voxelSizeAt(i));
            voxelOrigins.push_back(input->voxelOriginAt(i));

            c10::cuda::CUDAGuard deviceGuard(input->device().index());
            nanovdb::GridData *dst =
                reinterpret_cast<nanovdb::GridData *>(buffer.deviceData() + writeOffset);
            const uint8_t *src =
                input->nanoGridHandle().buffer().deviceData() + input->cumBytesAt(i);
            cudaMemcpyAsync(
                (uint8_t *)dst, src, input->numBytesAt(i), cudaMemcpyDeviceToDevice, stream);

            updateGridCountAndZeroChecksum<<<1, 1, 0, stream>>>(dst, i, totalGrids);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            writeOffset += input->numBytesAt(i);
        }
    }

    return makeGridBatchData(
        nanovdb::GridHandle<TorchDeviceBuffer>(std::move(buffer)), voxelSizes, voxelOrigins);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
