// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/GridBatchDataFactory.h>
#include <fvdb/detail/ops/MakeContiguous.h>
#include <fvdb/detail/utils/cuda/StreamOrdering.h>
#include <fvdb/detail/utils/nanovdb/DeviceGridHandleUtils.cuh>

#include <nanovdb/tools/GridChecksum.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

// Legacy TorchDeviceBuffer helpers for the builders that still merge per-item handles of that
// type; they go with the builders' conversion (#770 step 5).
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
    // Copies may already be enqueued when a later CUDA call throws, so the source's stream must
    // be ordered after them on that path too; the destructor does it, swallowing a secondary
    // error so the original one propagates.
    ~SourceStreamOrder() {
        if (mFinished) {
            return;
        }
        try {
            finish();
        } catch (...) {}
    }
    void
    finish() {
        mFinished = true;
        if (mDevice.is_cuda()) {
            fvdb::detail::orderStreamAfter(mSrcStream, mDevice, mStream, mDevice);
        } else if (mDevice.is_privateuseone()) {
            fvdb::detail::synchronizeStream(mStream, mDevice);
        }
    }
    torch::Device mDevice;
    cudaStream_t mSrcStream;
    cudaStream_t mStream;
    bool mFinished = false;
};

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
