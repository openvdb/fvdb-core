// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/GridBatchDataFactory.h>
#include <fvdb/detail/ops/MakeContiguous.h>
#include <fvdb/detail/utils/cuda/StreamOrdering.h>
#include <fvdb/detail/utils/nanovdb/DeviceGridHandleUtils.cuh>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/GridChecksum.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

namespace {

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
        C10_CUDA_CHECK(
            cudaMemcpyAsync((uint8_t *)dst, src, nbytes, cudaMemcpyDeviceToDevice, stream));
        fvdb::detail::updateGridCountAndZeroChecksum<<<1, 1, 0, stream>>>(
            dst, gridIndex, gridCount);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

// The bytes of the batch as seen from the side the copy runs on.
const uint8_t *
sourceBytes(const fvdb::GridBatchData &input) {
    const auto &s = input.gridStorage();
    return static_cast<const uint8_t *>(input.device().is_cpu() ? s.hostBytes() : s.deviceBytes());
}

// Device copies out of a batch run on `stream`, which need not be the stream its storage retains
// (its writers, and where its block returns when it is freed). Order the copy after the writers
// first, and the storage's stream after the copy last, so the batch may be released as soon as
// the caller returns. PrivateUse1 storage is not stream-ordered and its bytes are read by host
// code straight away, so there the streams are synchronized instead. CPU storage needs neither.
struct SourceStreamOrder {
    SourceStreamOrder(const fvdb::GridBatchData &input, cudaStream_t stream)
        : mDevice(input.device()), mSrcStream(input.gridStorage().stream()), mStream(stream) {
        if (mDevice.is_cuda()) {
            fvdb::detail::orderStreamAfter(mStream, mDevice, mSrcStream, mDevice);
        } else if (mDevice.is_privateuseone()) {
            fvdb::detail::synchronizeStream(mSrcStream, mDevice);
        }
    }
    void
    finish() {
        if (mDevice.is_cuda()) {
            fvdb::detail::orderStreamAfter(mSrcStream, mDevice, mStream, mDevice);
        } else if (mDevice.is_privateuseone()) {
            fvdb::detail::synchronizeStream(mStream, mDevice);
        }
    }
    torch::Device mDevice;
    cudaStream_t mSrcStream;
    cudaStream_t mStream;
};

} // namespace

namespace fvdb {
namespace detail {
namespace ops {

GridStorage
contiguousGridStorage(const GridBatchData &input) {
    c10::DeviceGuard guard(input.device());
    const torch::Device device = input.device();
    const int64_t totalGrids   = input.batchSize();
    int64_t totalByteSize      = 0;
    for (int64_t i = 0; i < totalGrids; i += 1) {
        totalByteSize += input.numBytesAt(i);
    }
    const uint8_t *srcBase = sourceBytes(input);

    if (device.is_cpu()) {
        nanovdb::HostBuffer buffer(totalByteSize);
        uint8_t *dstBase    = static_cast<uint8_t *>(buffer.data());
        int64_t writeOffset = 0;
        for (int64_t i = 0; i < totalGrids; i += 1) {
            copyAndFixGrid(reinterpret_cast<nanovdb::GridData *>(dstBase + writeOffset),
                           srcBase + input.cumBytesAt(i),
                           input.numBytesAt(i),
                           static_cast<uint32_t>(i),
                           static_cast<uint32_t>(totalGrids),
                           /*isCpu=*/true,
                           cudaStream_t{});
            writeOffset += input.numBytesAt(i);
        }
        // Parsing the chain validates the headers just written (index, count, size).
        return GridStorage(GridStorage::HostHandle(std::move(buffer)));
    }

    const cudaStream_t stream = fvdb::detail::storageStream(device);
    SourceStreamOrder order(input, stream);
    DeviceGridBuffer buffer(
        stream, TorchDeviceResource(device), totalByteSize, nanovdb::cuda::noInit);
    std::vector<nanovdb::GridHandleMetaData> meta;
    meta.reserve(totalGrids);
    int64_t writeOffset = 0;
    for (int64_t i = 0; i < totalGrids; i += 1) {
        copyAndFixGrid(reinterpret_cast<nanovdb::GridData *>(buffer.data() + writeOffset),
                       srcBase + input.cumBytesAt(i),
                       input.numBytesAt(i),
                       static_cast<uint32_t>(i),
                       static_cast<uint32_t>(totalGrids),
                       /*isCpu=*/false,
                       stream);
        meta.push_back(nanovdb::GridHandleMetaData{
            static_cast<uint64_t>(writeOffset), input.numBytesAt(i), nanovdb::GridType::OnIndex});
        writeOffset += input.numBytesAt(i);
    }
    order.finish();
    // The layout is known, so the handle adopts it: no parse kernel, no synchronization.
    return GridStorage(makeDeviceHandleFromLayout(std::move(buffer), std::move(meta)), device);
}

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
    const uint8_t *srcBase = sourceBytes(input);
    SourceStreamOrder order(input, stream);
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
    order.finish();
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
    const uint8_t *src = sourceBytes(input) + input.cumBytesAt(i);
    SourceStreamOrder order(input, stream);
    copyAndFixGrid(reinterpret_cast<nanovdb::GridData *>(dst),
                   src,
                   nbytes,
                   /*gridIndex=*/0,
                   /*gridCount=*/1,
                   isCpu,
                   stream);
    order.finish();
    return nanovdb::GridHandle<TorchDeviceBuffer>(std::move(buffer));
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
