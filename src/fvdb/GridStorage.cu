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

void
checkStorageDevice(const torch::Device &device) {
    TORCH_CHECK(device.is_cpu() || device.is_cuda() || device.is_privateuseone(),
                "GridStorage: device must be CPU, CUDA or PrivateUse1, got ",
                device);
    TORCH_CHECK(!device.is_cuda() || device.has_index(),
                "GridStorage: CUDA device must carry an index, got ",
                device);
}

// An empty handle whose buffer already carries the right resource, so anything that reads the
// device off the buffer (a later GridStorage(DeviceHandle&&), a prototype) agrees with the storage.
GridStorage::DeviceHandle
emptyCudaHandle(const torch::Device &device, cudaStream_t stream) {
    return GridStorage::DeviceHandle::adopt(
        DeviceGridBuffer(stream, TorchDeviceResource(device), 0, nanovdb::cuda::noInit),
        std::vector<nanovdb::GridHandleMetaData>{});
}

// Deep-copies device or unified storage held in @p src, whose writers are queued on @p srcStream
// of @p srcDevice, onto @p device on @p stream.
template <typename HandleT>
GridStorage
copyFromDevice(const HandleT &src,
               cudaStream_t srcStream,
               const torch::Device &srcDevice,
               const torch::Device &device,
               cudaStream_t stream) {
    detail::orderStreamAfter(stream, device.is_cpu() ? srcDevice : device, srcStream, srcDevice);

    if (device.is_cpu()) {
        // copyTo synchronizes `stream` before returning a host-readable handle, so the source is
        // no longer read once this returns and needs no ordering of its own.
        c10::OptionalDeviceGuard srcGuard(
            srcDevice.is_cuda() ? std::optional<torch::Device>(srcDevice) : std::nullopt);
        return GridStorage(nanovdb::cuda::copyTo<nanovdb::HostBuffer>(src, stream));
    }

    // A unified destination is host-readable, so copyTo synchronizes `stream` before returning.
    if (device.is_privateuseone()) {
        return GridStorage(nanovdb::cuda::copyTo<UnifiedGridBuffer>(src, stream), device);
    }

    GridStorage::DeviceHandle dst;
    {
        c10::cuda::CUDAGuard dstGuard(device);
        // Only a copy between two different CUDA devices needs the peer path: torch's allocator
        // backends (expandable segments, cudaMallocAsync) do not support a plain memcpy between
        // devices without peer access, and c10 itself selects cudaMemcpyPeerAsync there. Every
        // other pairing (same CUDA device, or a PrivateUse1 source, whose unified memory any
        // device addresses) is an ordinary copyTo with cudaMemcpyDefault.
        if (srcDevice.is_cuda() && srcDevice != device) {
            DeviceGridBuffer buf(
                stream, TorchDeviceResource(device), src.bufferSize(), nanovdb::cuda::noInit);
            C10_CUDA_CHECK(cudaMemcpyPeerAsync(buf.data(),
                                               device.index(),
                                               src.deviceData(),
                                               srcDevice.index(),
                                               src.bufferSize(),
                                               stream));
            dst = GridStorage::DeviceHandle::adopt(std::move(buf), src.metaData());
        } else {
            const DeviceGridBuffer proto = GridStorage::deviceProto(device, stream);
            dst = nanovdb::cuda::copyTo<DeviceGridBuffer>(src, stream, &proto);
        }
    }
    // The copy is still in flight on `stream`. Destroying the source frees its block on
    // `srcStream`, where torch may hand it straight to the next allocation on that stream; make
    // that stream wait for the copy so the block cannot be rewritten underneath it. A PrivateUse1
    // source synchronizes `stream` instead.
    detail::orderStreamAfter(srcStream, srcDevice, stream, device);
    return GridStorage(std::move(dst), device);
}

} // namespace

GridStorage::GridStorage() : mHandle(HostHandle()), mDevice(torch::kCPU) {}

GridStorage::GridStorage(HostHandle &&handle) : mHandle(std::move(handle)), mDevice(torch::kCPU) {}

GridStorage::GridStorage(DeviceHandle &&handle)
    : mHandle(std::in_place_type<DeviceHandle>, std::move(handle)),
      mDevice(std::get<DeviceHandle>(mHandle).buffer().resource().device()) {}

GridStorage::GridStorage(DeviceHandle &&handle, const torch::Device &device)
    : mHandle(std::in_place_type<DeviceHandle>, std::move(handle)), mDevice(resolveDevice(device)) {
    checkStorageDevice(mDevice);
    TORCH_CHECK(mDevice.is_cuda(), "GridStorage: a device handle cannot live on ", mDevice);
    auto &h = std::get<DeviceHandle>(mHandle);
    if (h.isEmpty()) {
        // An empty handle carries whatever resource it was default-constructed with (the current
        // CUDA device, e.g. cuda::copyTo's result for a zero-byte source); re-home it so the
        // buffer agrees with device() for anyone who re-wraps it or uses it as a prototype.
        if (h.buffer().resource().device() != mDevice) {
            h = emptyCudaHandle(mDevice, h.buffer().stream());
        }
        return;
    }
    TORCH_CHECK(h.buffer().resource().device() == mDevice,
                "GridStorage: handle is on ",
                h.buffer().resource().device(),
                " but the storage was declared on ",
                mDevice);
}

GridStorage::GridStorage(UnifiedHandle &&handle, const torch::Device &device)
    : mHandle(std::in_place_type<UnifiedHandle>, std::move(handle)), mDevice(device) {
    TORCH_CHECK(mDevice.is_privateuseone(),
                "GridStorage: a unified handle must live on PrivateUse1, got ",
                mDevice);
}

torch::Device
GridStorage::resolveDevice(const torch::Device &device) {
    if (device.is_cuda() && !device.has_index()) {
        return torch::Device(torch::kCUDA, c10::cuda::current_device());
    }
    return device;
}

GridStorage
GridStorage::empty(const torch::Device &requested, cudaStream_t stream) {
    const torch::Device device = resolveDevice(requested);
    checkStorageDevice(device);
    if (device.is_cpu()) {
        return GridStorage();
    }
    if (device.is_privateuseone()) {
        return GridStorage(UnifiedHandle(), device);
    }
    return GridStorage(emptyCudaHandle(device, stream), device);
}

DeviceGridBuffer
GridStorage::deviceProto(const torch::Device &requested, cudaStream_t stream) {
    const torch::Device device = resolveDevice(requested);
    checkStorageDevice(device);
    TORCH_CHECK(device.is_cuda(), "GridStorage::deviceProto: not a CUDA device: ", device);
    // Allocations through the proto are device-correct on their own (the resource guards), but
    // the kernels the NanoVDB builders launch next to them run on whatever device is current.
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        c10::cuda::current_device() == device.index(),
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
            return h.metaData();
        },
        mHandle);
}

uint32_t
GridStorage::gridCount() const {
    return static_cast<uint32_t>(metadata().size());
}

const nanovdb::GridHandleMetaData &
GridStorage::metaAt(uint32_t i) const {
    const auto &meta = metadata();
    TORCH_CHECK(i < meta.size(), "GridStorage: grid index ", i, " out of range ", meta.size());
    return meta[i];
}

uint64_t
GridStorage::gridSize(uint32_t i) const {
    return metaAt(i).size;
}

uint64_t
GridStorage::gridOffset(uint32_t i) const {
    return metaAt(i).offset;
}

nanovdb::GridType
GridStorage::gridType(uint32_t i) const {
    return metaAt(i).gridType;
}

const void *
GridStorage::hostBytes() const {
    if (const auto *h = std::get_if<HostHandle>(&mHandle)) {
        return h->data();
    }
    // Unified memory: the device pointer is a host pointer too.
    if (const auto *h = std::get_if<UnifiedHandle>(&mHandle)) {
        return h->deviceData();
    }
    return nullptr;
}

void *
GridStorage::hostBytes() {
    return const_cast<void *>(static_cast<const GridStorage *>(this)->hostBytes());
}

const void *
GridStorage::deviceBytes() const {
    if (const auto *h = std::get_if<DeviceHandle>(&mHandle)) {
        return h->deviceData();
    }
    if (const auto *h = std::get_if<UnifiedHandle>(&mHandle)) {
        return h->deviceData();
    }
    return nullptr;
}

void *
GridStorage::deviceBytes() {
    return const_cast<void *>(static_cast<const GridStorage *>(this)->deviceBytes());
}

cudaStream_t
GridStorage::stream() const {
    const auto *h = std::get_if<DeviceHandle>(&mHandle);
    return h ? h->buffer().stream() : cudaStream_t{};
}

const GridStorage::HostHandle &
GridStorage::hostHandle() const {
    TORCH_CHECK(isHost(), "GridStorage: storage on ", mDevice, " has no host handle");
    return std::get<HostHandle>(mHandle);
}

GridStorage::HostHandle &
GridStorage::hostHandle() {
    TORCH_CHECK(isHost(), "GridStorage: storage on ", mDevice, " has no host handle");
    return std::get<HostHandle>(mHandle);
}

const GridStorage::DeviceHandle &
GridStorage::deviceHandle() const {
    TORCH_CHECK(std::holds_alternative<DeviceHandle>(mHandle),
                "GridStorage: storage on ",
                mDevice,
                " has no device handle");
    return std::get<DeviceHandle>(mHandle);
}

GridStorage::DeviceHandle &
GridStorage::deviceHandle() {
    TORCH_CHECK(std::holds_alternative<DeviceHandle>(mHandle),
                "GridStorage: storage on ",
                mDevice,
                " has no device handle");
    return std::get<DeviceHandle>(mHandle);
}

const GridStorage::UnifiedHandle &
GridStorage::unifiedHandle() const {
    TORCH_CHECK(std::holds_alternative<UnifiedHandle>(mHandle),
                "GridStorage: storage on ",
                mDevice,
                " has no unified handle");
    return std::get<UnifiedHandle>(mHandle);
}

GridStorage::UnifiedHandle &
GridStorage::unifiedHandle() {
    TORCH_CHECK(std::holds_alternative<UnifiedHandle>(mHandle),
                "GridStorage: storage on ",
                mDevice,
                " has no unified handle");
    return std::get<UnifiedHandle>(mHandle);
}

GridStorage
GridStorage::to(const torch::Device &requested) const {
    const torch::Device device = resolveDevice(requested);
    return to(device, device.is_cpu() ? stream() : detail::storageStream(device));
}

GridStorage
GridStorage::to(const torch::Device &requested, cudaStream_t stream) const {
    const torch::Device device = resolveDevice(requested);
    checkStorageDevice(device);
    if (isEmpty()) {
        // cuda::copyTo would return a default-constructed handle here, whose resource is the
        // *current* CUDA device and whose stream is the default, not the requested ones.
        return empty(device, stream);
    }

    if (const auto *src = std::get_if<HostHandle>(&mHandle)) {
        if (device.is_cpu()) {
            return GridStorage(src->copy<nanovdb::HostBuffer>());
        }
        // Host to device: the pageable source makes the copy synchronous on the source side (the
        // bytes are staged before the call returns), so neither destination needs ordering. A
        // CUDA destination retains `stream`; a unified destination is host-readable, so copyTo
        // synchronizes `stream` before returning.
        if (device.is_privateuseone()) {
            return GridStorage(nanovdb::cuda::copyTo<UnifiedGridBuffer>(*src, stream), device);
        }
        c10::cuda::CUDAGuard deviceGuard(device);
        const DeviceGridBuffer proto = deviceProto(device, stream);
        return GridStorage(nanovdb::cuda::copyTo<DeviceGridBuffer>(*src, stream, &proto), device);
    }

    // Device source. Its writers are queued on its retained stream, which is also the stream its
    // block is keyed to; the copy runs on `stream`, a stream of the destination device. A unified
    // source has no retained stream and orders against the legacy default stream.
    if (const auto *src = std::get_if<DeviceHandle>(&mHandle)) {
        return copyFromDevice(*src, src->buffer().stream(), mDevice, device, stream);
    }
    return copyFromDevice(
        std::get<UnifiedHandle>(mHandle), cudaStream_t{}, mDevice, device, stream);
}

} // namespace fvdb
