// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/GridStorage.h>
#include <fvdb/detail/utils/cuda/StreamOrdering.h>

#include <nanovdb/cuda/GridHandle.cuh>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Exception.h>

#include <utility>

namespace fvdb {

namespace {

using HandleFactory = nanovdb::cuda::detail::HandleFactory;

bool
isDeviceKind(const torch::Device &device) {
    return device.is_cuda() || device.is_privateuseone();
}

void
checkStorageDevice(const torch::Device &device) {
    TORCH_CHECK(device.is_cpu() || isDeviceKind(device),
                "GridStorage: device must be CPU, CUDA or PrivateUse1, got ",
                device);
    TORCH_CHECK(!device.is_cuda() || device.has_index(),
                "GridStorage: CUDA device must carry an index, got ",
                device);
}

// The current torch stream of a CUDA device, or the legacy default stream for PrivateUse1, whose
// unified memory is not stream-ordered.
cudaStream_t
currentStreamOf(const torch::Device &device) {
    if (device.is_cuda()) {
        return c10::cuda::getCurrentCUDAStream(device.index()).stream();
    }
    return cudaStream_t{};
}

// An empty handle whose buffer already carries the right resource, so anything that reads the
// device off the buffer (a later GridStorage(DeviceHandle&&), a prototype) agrees with the storage.
GridStorage::DeviceHandle
emptyDeviceHandle(const torch::Device &device, cudaStream_t stream) {
    return HandleFactory::make(
        DeviceGridBuffer(stream, TorchDeviceResource(device), 0, nanovdb::cuda::noInit),
        std::vector<nanovdb::GridHandleMetaData>{});
}

} // namespace

GridStorage::GridStorage() : mHandle(HostHandle()), mDevice(torch::kCPU) {}

GridStorage::GridStorage(HostHandle &&handle) : mHandle(std::move(handle)), mDevice(torch::kCPU) {}

GridStorage::GridStorage(DeviceHandle &&handle)
    : mHandle(std::in_place_index<1>, std::move(handle)),
      mDevice(std::get<1>(mHandle).buffer().resource().device()) {}

GridStorage::GridStorage(DeviceHandle &&handle, const torch::Device &device)
    : mHandle(std::in_place_index<1>, std::move(handle)), mDevice(device) {
    checkStorageDevice(device);
    TORCH_CHECK(isDeviceKind(device), "GridStorage: a device handle cannot live on ", device);
    const auto &h = std::get<1>(mHandle);
    TORCH_CHECK(h.isEmpty() || h.buffer().resource().device() == device,
                "GridStorage: handle is on ",
                h.buffer().resource().device(),
                " but the storage was declared on ",
                device);
}

GridStorage
GridStorage::empty(const torch::Device &device) {
    checkStorageDevice(device);
    if (device.is_cpu()) {
        return GridStorage();
    }
    return GridStorage(emptyDeviceHandle(device, cudaStream_t{}), device);
}

DeviceGridBuffer
GridStorage::deviceProto(const torch::Device &device, cudaStream_t stream) {
    checkStorageDevice(device);
    TORCH_CHECK(isDeviceKind(device), "GridStorage::deviceProto: not a device: ", device);
    // Allocations through the proto are device-correct on their own (the resource guards), but
    // the kernels the NanoVDB builders launch next to them run on whatever device is current.
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        !device.is_cuda() || c10::cuda::current_device() == device.index(),
        "GridStorage::deviceProto: ",
        device,
        " is not the current CUDA device; the builder's kernels would run on the wrong device");
    return DeviceGridBuffer(stream, TorchDeviceResource(device), 0, nanovdb::cuda::noInit);
}

bool
GridStorage::isEmpty() const {
    return std::visit([](const auto &h) { return h.isEmpty(); }, mHandle);
}

uint64_t
GridStorage::bufferSize() const {
    return std::visit([](const auto &h) { return h.bufferSize(); }, mHandle);
}

const std::vector<nanovdb::GridHandleMetaData> &
GridStorage::metadata() const {
    return std::visit(
        [](const auto &h) -> const std::vector<nanovdb::GridHandleMetaData> & {
            return HandleFactory::meta(h);
        },
        mHandle);
}

uint32_t
GridStorage::gridCount() const {
    return static_cast<uint32_t>(metadata().size());
}

uint64_t
GridStorage::gridSize(uint32_t i) const {
    TORCH_CHECK(i < gridCount(), "GridStorage: grid index ", i, " out of range ", gridCount());
    return metadata()[i].size;
}

uint64_t
GridStorage::gridOffset(uint32_t i) const {
    TORCH_CHECK(i < gridCount(), "GridStorage: grid index ", i, " out of range ", gridCount());
    return metadata()[i].offset;
}

nanovdb::GridType
GridStorage::gridType(uint32_t i) const {
    TORCH_CHECK(i < gridCount(), "GridStorage: grid index ", i, " out of range ", gridCount());
    return metadata()[i].gridType;
}

const void *
GridStorage::hostBytes() const {
    if (isHost()) {
        return std::get<0>(mHandle).data();
    }
    // PrivateUse1 is unified memory: the device pointer is a host pointer too.
    return mDevice.is_privateuseone() ? std::get<1>(mHandle).deviceData() : nullptr;
}

void *
GridStorage::hostBytes() {
    return const_cast<void *>(static_cast<const GridStorage *>(this)->hostBytes());
}

const void *
GridStorage::deviceBytes() const {
    return isDevice() ? std::get<1>(mHandle).deviceData() : nullptr;
}

void *
GridStorage::deviceBytes() {
    return const_cast<void *>(static_cast<const GridStorage *>(this)->deviceBytes());
}

cudaStream_t
GridStorage::stream() const {
    return isDevice() ? std::get<1>(mHandle).buffer().stream() : cudaStream_t{};
}

const GridStorage::HostHandle &
GridStorage::hostHandle() const {
    TORCH_CHECK(isHost(), "GridStorage: storage on ", mDevice, " has no host handle");
    return std::get<0>(mHandle);
}

GridStorage::HostHandle &
GridStorage::hostHandle() {
    TORCH_CHECK(isHost(), "GridStorage: storage on ", mDevice, " has no host handle");
    return std::get<0>(mHandle);
}

const GridStorage::DeviceHandle &
GridStorage::deviceHandle() const {
    TORCH_CHECK(isDevice(), "GridStorage: storage on ", mDevice, " has no device handle");
    return std::get<1>(mHandle);
}

GridStorage::DeviceHandle &
GridStorage::deviceHandle() {
    TORCH_CHECK(isDevice(), "GridStorage: storage on ", mDevice, " has no device handle");
    return std::get<1>(mHandle);
}

GridStorage
GridStorage::to(const torch::Device &device) const {
    return to(device, device.is_cpu() ? stream() : currentStreamOf(device));
}

GridStorage
GridStorage::to(const torch::Device &device, cudaStream_t stream) const {
    checkStorageDevice(device);
    if (isEmpty()) {
        // cuda::copyTo would return a default-constructed handle here, whose resource is the
        // *current* CUDA device and whose stream is the default, not the requested ones.
        if (device.is_cpu()) {
            return GridStorage();
        }
        return GridStorage(emptyDeviceHandle(device, stream), device);
    }

    if (isHost()) {
        const auto &src = std::get<0>(mHandle);
        if (device.is_cpu()) {
            return GridStorage(src.copy<nanovdb::HostBuffer>());
        }
        // Host to device: the pageable source makes the copy synchronous on the host side, and
        // the destination retains `stream`; nothing to order.
        c10::OptionalDeviceGuard deviceGuard(device.is_cuda() ? std::optional<torch::Device>(device)
                                                              : std::nullopt);
        const DeviceGridBuffer proto = deviceProto(device, stream);
        return GridStorage(nanovdb::cuda::copyTo<DeviceGridBuffer>(src, stream, &proto), device);
    }

    // Device source. Its writers are queued on its retained stream, which is also the stream its
    // block is keyed to; the copy runs on `stream`, a stream of the destination device.
    const auto &src              = std::get<1>(mHandle);
    const cudaStream_t srcStream = src.buffer().stream();
    detail::orderStreamAfter(stream, device.is_cpu() ? mDevice : device, srcStream, mDevice);

    if (device.is_cpu()) {
        // copyTo synchronizes `stream` before returning a host-readable handle, so the source is
        // no longer read once this returns and needs no ordering of its own.
        c10::OptionalDeviceGuard srcGuard(mDevice.is_cuda() ? std::optional<torch::Device>(mDevice)
                                                            : std::nullopt);
        return GridStorage(nanovdb::cuda::copyTo<nanovdb::HostBuffer>(src, stream));
    }

    DeviceHandle dst;
    {
        c10::OptionalDeviceGuard dstGuard(device.is_cuda() ? std::optional<torch::Device>(device)
                                                           : std::nullopt);
        if (mDevice.is_cuda() && device.is_cuda() && mDevice != device) {
            // Across CUDA devices the copy has to be a peer copy: torch's allocator backends
            // (expandable segments, cudaMallocAsync) do not support a plain memcpy between
            // devices without peer access, and c10 itself selects cudaMemcpyPeerAsync here.
            DeviceGridBuffer buf(
                stream, TorchDeviceResource(device), src.bufferSize(), nanovdb::cuda::noInit);
            C10_CUDA_CHECK(cudaMemcpyPeerAsync(buf.data(),
                                               device.index(),
                                               src.deviceData(),
                                               mDevice.index(),
                                               src.bufferSize(),
                                               stream));
            dst = HandleFactory::make(std::move(buf), HandleFactory::meta(src));
        } else {
            const DeviceGridBuffer proto = deviceProto(device, stream);
            dst = nanovdb::cuda::copyTo<DeviceGridBuffer>(src, stream, &proto);
        }
    }
    // The copy is still in flight on `stream`. Destroying the source frees its block on
    // `srcStream`, where torch may hand it straight to the next allocation on that stream; make
    // that stream wait for the copy so the block cannot be rewritten underneath it.
    detail::orderStreamAfter(srcStream, mDevice, stream, device);
    return GridStorage(std::move(dst), device);
}

} // namespace fvdb
