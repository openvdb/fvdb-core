// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/TorchDeviceBuffer.h>
#include <fvdb/TorchResource.h>

#include <nanovdb/GridHandle.h>
#include <nanovdb/cuda/DeviceBuffer.h>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <utility>

namespace nanovdb {

// This is a dirty hack to clone the buffer accross devices
// guide is an empty buffer with the target device of the copy
// TODO: Pass in synchronous option
template <>
template <>
GridHandle<fvdb::TorchDeviceBuffer>
GridHandle<fvdb::TorchDeviceBuffer>::copy(const fvdb::TorchDeviceBuffer &guide) const {
    if (mBuffer.isEmpty()) {
        fvdb::TorchDeviceBuffer retbuf(0, guide.device());
        return GridHandle<fvdb::TorchDeviceBuffer>(std::move(retbuf)); // return an empty handle
    }

    const bool guideIsHost   = guide.device().is_cpu();
    const bool iAmHost       = mBuffer.device().is_cpu();
    const bool guideIsDevice = !guideIsHost;
    const bool iAmDevice     = !iAmHost;

    auto buffer = fvdb::TorchDeviceBuffer::create(mBuffer.size(), &guide);

    if (iAmHost && guideIsHost) {
        std::memcpy(buffer.data(),
                    mBuffer.data(),
                    mBuffer.size()); // deep copy of buffer in CPU RAM
        return GridHandle<fvdb::TorchDeviceBuffer>(std::move(buffer));
    } else if (iAmHost && guideIsDevice) {
        const at::cuda::CUDAGuard device_guard{guide.device()};
        cudaCheck(cudaMemcpy(
            buffer.deviceData(), mBuffer.data(), mBuffer.size(), cudaMemcpyHostToDevice));
        return GridHandle<fvdb::TorchDeviceBuffer>(std::move(buffer));
    } else if (iAmDevice && guideIsHost) {
        const at::cuda::CUDAGuard device_guard{mBuffer.device()};
        cudaCheck(cudaMemcpy(
            buffer.data(), mBuffer.deviceData(), mBuffer.size(), cudaMemcpyDeviceToHost));
        return GridHandle<fvdb::TorchDeviceBuffer>(std::move(buffer));
    } else if (iAmDevice && guideIsDevice) {
        const at::cuda::CUDAGuard device_guard{guide.device()};
        cudaCheck(cudaMemcpy(
            buffer.deviceData(), mBuffer.deviceData(), mBuffer.size(), cudaMemcpyDeviceToDevice));
        return GridHandle<fvdb::TorchDeviceBuffer>(std::move(buffer));
    } else {
        TORCH_CHECK(false, "All host/device combos exhausted. This should never happen.");
    }
}

} // namespace nanovdb

namespace fvdb {

namespace {
// Device memory for the copies in to(), which run on the current device's current torch stream.
void *
deviceAlloc(uint64_t bytes, cudaStream_t stream) {
    return TorchStorageResource{}.allocate_async(
        bytes, TorchStorageResource::DEFAULT_ALIGNMENT, stream);
}

void
deviceFree(void *p) {
    TorchStorageResource{}.deallocate_async(p, 0, TorchStorageResource::DEFAULT_ALIGNMENT, nullptr);
}
} // namespace

TorchDeviceBuffer::TorchDeviceBuffer(uint64_t size /* = 0*/,
                                     const torch::Device &device /* = torch::kCPU*/,
                                     void *stream /* = nullptr*/)
    : mSize(size), mData(nullptr), mDevice(device) {
    if (!mSize) {
        return;
    }

    if (mDevice.is_cpu()) {
        // Initalize on the host
        mData = reinterpret_cast<uint8_t *>(malloc(size));
    } else if (mDevice.is_cuda()) {
        // Initialize on the device through the storage resource. It picks the allocation stream
        // (the device's current torch stream) and orders the stream the caller will write on
        // after it; see TorchStorageResource.
        c10::cuda::CUDAGuard deviceGuard(mDevice);
        mData = reinterpret_cast<uint8_t *>(TorchStorageResource{}.allocate_async(
            size, TorchStorageResource::DEFAULT_ALIGNMENT, static_cast<cudaStream_t>(stream)));
        checkPtr(mData, "failed to allocate device data");
    } else if (mDevice.is_privateuseone()) {
        auto allocator = c10::GetAllocator(c10::DeviceType::PrivateUse1);
        mData          = reinterpret_cast<uint8_t *>(allocator->raw_allocate(size));
    }
}

TorchDeviceBuffer::TorchDeviceBuffer(TorchDeviceBuffer &&other) noexcept
    : mSize(other.mSize), mData(other.mData), mDevice(other.mDevice) {
    other.mSize = 0;
    other.mData = nullptr;
}

TorchDeviceBuffer &
TorchDeviceBuffer::operator=(TorchDeviceBuffer &&other) noexcept {
    clear();
    mSize       = other.mSize;
    mData       = other.mData;
    mDevice     = other.mDevice;
    other.mSize = 0;
    other.mData = nullptr;
    return *this;
}

TorchDeviceBuffer::~TorchDeviceBuffer() {
    this->clear();
};

const torch::Device &
TorchDeviceBuffer::device() const {
    return mDevice;
}

void
TorchDeviceBuffer::to(const torch::Device &device) {
    if (mDevice == device) {
        TORCH_CHECK(
            mData && (mSize > 0),
            "Source device matches destination device but existing data pointer is invalid");
        return;
    }

    TORCH_CHECK(!device.is_cuda() || device.has_index(), "CUDA devices must specify an index");

    // The CUDA copies run on the torch stream of the device they write, which is also the stream
    // the destination block belongs to, and the stream is synchronized before this returns. A
    // copy on the legacy default stream would be neither ordered after the source's writers on
    // a non-blocking torch stream nor guaranteed complete before the block could be reused.
    // Until the copy has succeeded the destination is owned by `dst`, so a failed copy or
    // synchronization releases it and leaves this buffer as it was.
    struct Destination {
        uint8_t *p = nullptr;
        torch::Device device;
        ~Destination() {
            if (!p) {
                return;
            }
            if (!device.is_cuda()) {
                free(p);
                return;
            }
            // Best effort: this runs while a copy failure is propagating, and a sticky CUDA
            // error can make the guard or the free throw too. Swallowing that keeps the
            // original exception, at the cost of the block, rather than terminating.
            try {
                c10::cuda::CUDAGuard deviceGuard(device);
                deviceFree(p);
            } catch (...) {}
        }
        uint8_t *
        release() {
            return std::exchange(p, nullptr);
        }
    } dst{nullptr, device};

    uint8_t *data = nullptr;
    if (mDevice.is_cpu() && device.is_cuda()) { // CPU -> CUDA
        c10::cuda::CUDAGuard deviceGuard(device);
        const auto stream = c10::cuda::getCurrentCUDAStream(device.index());
        dst.p             = reinterpret_cast<uint8_t *>(deviceAlloc(mSize, stream.stream()));
        C10_CUDA_CHECK(cudaMemcpyAsync(dst.p, mData, mSize, cudaMemcpyHostToDevice, stream));
        C10_CUDA_CHECK(cudaStreamSynchronize(stream));
        data = dst.release();
        free(mData);
    } else if (mDevice.is_cuda() && device.is_cpu()) {
        dst.p = reinterpret_cast<uint8_t *>(malloc(mSize));
        c10::cuda::CUDAGuard deviceGuard(mDevice);
        const auto stream = c10::cuda::getCurrentCUDAStream(mDevice.index());
        C10_CUDA_CHECK(cudaMemcpyAsync(dst.p, mData, mSize, cudaMemcpyDeviceToHost, stream));
        C10_CUDA_CHECK(cudaStreamSynchronize(stream));
        data = dst.release();
        deviceFree(mData);
    } else if (mDevice.is_cuda() && device.is_cuda()) {
        {
            // Finish the source's writers before another device's stream reads it.
            c10::cuda::CUDAGuard deviceGuard(mDevice);
            C10_CUDA_CHECK(cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream(mDevice.index())));
        }
        {
            c10::cuda::CUDAGuard deviceGuard(device);
            const auto stream = c10::cuda::getCurrentCUDAStream(device.index());
            dst.p             = reinterpret_cast<uint8_t *>(deviceAlloc(mSize, stream.stream()));
            // Peer copy: correct whether or not peer access is enabled between the two devices'
            // pools, which a plain device-to-device memcpy is not.
            C10_CUDA_CHECK(
                cudaMemcpyPeerAsync(dst.p, device.index(), mData, mDevice.index(), mSize, stream));
            C10_CUDA_CHECK(cudaStreamSynchronize(stream));
            data = dst.release();
        }
        {
            c10::cuda::CUDAGuard deviceGuard(mDevice);
            deviceFree(mData);
        }
    } else if (mDevice.is_cpu() && device.is_privateuseone()) {
        auto allocator = c10::GetAllocator(c10::DeviceType::PrivateUse1);
        data           = reinterpret_cast<uint8_t *>(allocator->raw_allocate(mSize));
        cudaMemcpy(data, mData, mSize, cudaMemcpyDefault);
        free(mData);
    } else if (mDevice.is_privateuseone() && device.is_cpu()) {
        auto allocator = c10::GetAllocator(c10::DeviceType::PrivateUse1);
        data           = reinterpret_cast<uint8_t *>(malloc(mSize));
        cudaMemcpy(data, mData, mSize, cudaMemcpyDefault);
        allocator->raw_deallocate(mData);
    } else {
        TORCH_CHECK(false, "Unsupported source and destination device combination");
    }

    mData   = data;
    mDevice = device;
}

uint8_t *
TorchDeviceBuffer::data() const {
    return (mDevice.is_cpu() || mDevice.is_privateuseone()) ? mData : nullptr;
}

uint8_t *
TorchDeviceBuffer::deviceData() const {
    return (mDevice.is_cuda() || mDevice.is_privateuseone()) ? mData : nullptr;
}

uint64_t
TorchDeviceBuffer::size() const {
    return mSize;
}

bool
TorchDeviceBuffer::empty() const {
    return mSize == 0;
}

bool
TorchDeviceBuffer::isEmpty() const {
    return this->empty();
}

void
TorchDeviceBuffer::clear() {
    if (!mData) {
        mSize = 0;
        return;
    }

    if (mDevice.is_privateuseone()) {
        auto allocator = c10::GetAllocator(c10::DeviceType::PrivateUse1);
        allocator->raw_deallocate(mData);
    } else if (mDevice.is_cuda()) {
        c10::cuda::CUDAGuard deviceGuard(mDevice);
        deviceFree(mData);
    } else if (mDevice.is_cpu()) {
        free(mData);
    }

    mData = nullptr;
    mSize = 0;
}

TorchDeviceBuffer
TorchDeviceBuffer::create(uint64_t size, const TorchDeviceBuffer *proto, int device, void *stream) {
    // The stream a nanovdb builder passes here is the one it will write the buffer on. It does
    // not choose the allocation stream; TorchStorageResource orders it after the allocation.
    if (proto) {
        // This is a hack to pass in the device index when creating grids from nanovdb. Since we
        // can't pass arguments through nanovdb creation functions, we use a prototype grid to pass
        // in the device index.
        return TorchDeviceBuffer(size, proto->device(), stream);
    } else if (device == cudaCpuDeviceId) {
        return TorchDeviceBuffer(size, torch::kCPU);
    } else if (device > cudaCpuDeviceId) {
        return TorchDeviceBuffer(size, torch::Device(torch::kCUDA, device), stream);
    } else {
        TORCH_CHECK(false, "Invalid parameters specified for TorchDeviceBuffer::create");
    }
}

} // namespace fvdb
