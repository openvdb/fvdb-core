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

// Copy one grid's `nbytes` from src to dst (host or device) and rewrite its (gridIndex, gridCount)
// header so the copy is a valid standalone member of a `gridCount`-grid handle.
void
copyAndFixGrid(nanovdb::GridData *dst,
               const uint8_t *src,
               int64_t nbytes,
               uint32_t gridIndex,
               uint32_t gridCount,
               bool isCpu,
               cudaStream_t stream) {
    if (isCpu) {
        memcpy((void *)dst, (const void *)src, nbytes);
        nanovdb::tools::updateGridCount(dst, gridIndex, gridCount);
    } else {
        cudaMemcpyAsync((uint8_t *)dst, src, nbytes, cudaMemcpyDeviceToDevice, stream);
        updateGridCountAndZeroChecksum<<<1, 1, 0, stream>>>(dst, gridIndex, gridCount);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

} // namespace

namespace fvdb {
namespace detail {
namespace ops {

nanovdb::GridHandle<TorchDeviceBuffer>
contiguousGridHandle(const GridBatchData &input) {
    c10::DeviceGuard guard(input.device());

    const int64_t totalGrids = input.batchSize();
    int64_t totalByteSize    = 0;
    for (int64_t i = 0; i < totalGrids; i += 1) {
        totalByteSize += input.numBytesAt(i);
    }

    TorchDeviceBuffer buffer(totalByteSize, input.device());
    const bool isCpu = input.device().is_cpu();
    cudaStream_t stream =
        isCpu ? cudaStream_t(0) : at::cuda::getCurrentCUDAStream(input.device().index()).stream();
    uint8_t *dstBase       = isCpu ? buffer.data() : buffer.deviceData();
    const uint8_t *srcBase = isCpu ? input.nanoGridHandle().buffer().data()
                                   : input.nanoGridHandle().buffer().deviceData();

    int64_t writeOffset = 0;
    for (int64_t i = 0; i < totalGrids; i += 1) {
        copyAndFixGrid(reinterpret_cast<nanovdb::GridData *>(dstBase + writeOffset),
                       srcBase + input.cumBytesAt(i),
                       input.numBytesAt(i),
                       static_cast<uint32_t>(i),
                       static_cast<uint32_t>(totalGrids),
                       isCpu,
                       stream);
        writeOffset += input.numBytesAt(i);
    }

    return nanovdb::GridHandle<TorchDeviceBuffer>(std::move(buffer));
}

nanovdb::GridHandle<TorchDeviceBuffer>
cloneGridHandleAt(const GridBatchData &input, int64_t i) {
    c10::DeviceGuard guard(input.device());

    const int64_t nbytes = input.numBytesAt(i);
    TorchDeviceBuffer buffer(nbytes, input.device());
    const bool isCpu = input.device().is_cpu();
    cudaStream_t stream =
        isCpu ? cudaStream_t(0) : at::cuda::getCurrentCUDAStream(input.device().index()).stream();
    uint8_t *dst       = isCpu ? buffer.data() : buffer.deviceData();
    const uint8_t *src = (isCpu ? input.nanoGridHandle().buffer().data()
                                : input.nanoGridHandle().buffer().deviceData()) +
                         input.cumBytesAt(i);

    copyAndFixGrid(reinterpret_cast<nanovdb::GridData *>(dst),
                   src,
                   nbytes,
                   /*gridIndex=*/0,
                   /*gridCount=*/1,
                   isCpu,
                   stream);

    return nanovdb::GridHandle<TorchDeviceBuffer>(std::move(buffer));
}

c10::intrusive_ptr<GridBatchData>
makeContiguous(c10::intrusive_ptr<GridBatchData> input) {
    if (input->isContiguous()) {
        return input;
    }

    std::vector<nanovdb::Vec3d> voxelSizes, voxelOrigins;
    voxelSizes.reserve(input->batchSize());
    voxelOrigins.reserve(input->batchSize());
    for (int64_t i = 0; i < input->batchSize(); i += 1) {
        voxelSizes.push_back(input->voxelSizeAt(i));
        voxelOrigins.push_back(input->voxelOriginAt(i));
    }

    return makeGridBatchData(contiguousGridHandle(*input), voxelSizes, voxelOrigins);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
