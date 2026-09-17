// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_TORCHDEVICERESOURCE_H
#define FVDB_TORCHDEVICERESOURCE_H

#include <fvdb/TorchResource.h>

#include <nanovdb/HostBuffer.h> // BufferHasDeviceSingle and the other BufferTraits detectors
#include <nanovdb/cuda/Buffer.h>
#include <nanovdb/cuda/DeviceResource.h>

#include <c10/core/Allocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <torch/types.h>

#include <cstddef>
#include <stdexcept>

namespace fvdb {

/// @brief Memory resource for grid storage held in a nanovdb::cuda::Buffer, bound to one torch
///        device. This is the resource behind DeviceGridBuffer, the single-space buffer that
///        GridStorage keeps device-resident grids in.
///
///        NanoVDB picks host vs. device storage per buffer *type* at compile time, so the device a
///        buffer lives on has to travel in its resource, not in the buffer: a prototype buffer's
///        resource() is what createDeviceStorage and cuda::copyTo allocate the output from, which
///        makes every allocation device-correct even when the caller forgot a device guard.
///
///        Stream contract. A cuda::Buffer retains the stream it was allocated on and orders its
///        writes and its eventual free on that stream; that is the whole of NanoVDB's ordering
///        model. Torch's caching allocator files a block under the stream it was allocated on and
///        hands it back only to later allocations on that same stream. Keying the block to the
///        buffer's retained stream (raw_alloc_with_stream, as the scratch resource TorchResource
///        does) therefore makes the two models agree: the next tenant of the block is ordered after
///        this buffer's free by stream order, with no events. Two obligations follow for the
///        caller. The retained stream must outlive the block, which torch's pool streams and the
///        legacy default stream do; do not construct one of these buffers on a stream that will be
///        destroyed (a nanovdb DeviceMesh stream), or the block is orphaned in a dead stream's
///        free list. And the retained stream must not change after allocation: torch returns the
///        block to the stream it was keyed to whatever stream the free names, so
///        Buffer::set_stream and Buffer::resize with a different stream would free to a stream the
///        new work was never ordered against. GridStorage never does either; a buffer whose work
///        must move to another stream is copied (GridStorage::to). PrivateUse1 is the exception
///        on both counts: its allocator ignores the stream, which is what lets the
///        DistributedPointsToGrid builders retarget a result allocated on a DeviceMesh stream
///        (set_stream) onto the storage stream before the mesh is destroyed. (Keying to torch's
///        current stream instead, as the deprecated dual-space buffer's resource does, would be
///        wrong here: it would let a buffer retained on another stream be freed to a stream its
///        work was never ordered against.)
///
///        PrivateUse1, fvdb's unified-memory device for multi-GPU builds, allocates through the
///        allocator registered for it; that memory is not stream-ordered and is host- and
///        device-accessible.
///
///        The resource is a cheap value type (a torch::Device), copyable as nanovdb's resource
///        concept requires, and default-constructible to the current CUDA device so that
///        DeviceGridBuffer, its GridHandle, and the copies cuda::copyTo makes of empty handles
///        stay default-constructible.
struct TorchDeviceResource {
    static constexpr size_t DEFAULT_ALIGNMENT = TorchResource::DEFAULT_ALIGNMENT;

    /// @brief Binds to the current CUDA device.
    TorchDeviceResource() : mDevice(torch::kCUDA, c10::cuda::current_device()) {}

    /// @brief Binds to @p device, which must be an indexed CUDA device or PrivateUse1.
    explicit TorchDeviceResource(const torch::Device &device) : mDevice(device) {
        TORCH_CHECK(device.is_cuda() || device.is_privateuseone(),
                    "TorchDeviceResource: device must be CUDA or PrivateUse1, got ",
                    device);
        TORCH_CHECK(!device.is_cuda() || device.has_index(),
                    "TorchDeviceResource: CUDA device must carry an index");
    }

    const torch::Device &
    device() const {
        return mDevice;
    }

    /// @brief Allocates @p bytes on the bound device, keyed to @p stream, the buffer's retained
    ///        stream (see the class comment). Ignored for PrivateUse1.
    void *
    allocate_async(size_t bytes, size_t alignment, cudaStream_t stream) {
        if (mDevice.is_privateuseone()) {
            void *p = c10::GetAllocator(c10::DeviceType::PrivateUse1)->raw_allocate(bytes);
            if (!p && bytes) {
                throw std::runtime_error("fvdb: TorchDeviceResource::allocate_async failed");
            }
            return p;
        }
        c10::cuda::CUDAGuard deviceGuard(mDevice);
        return TorchResource{}.allocate_async(bytes, alignment, stream);
    }

    /// @brief Frees @p p on the bound device. The block returns to the stream it was allocated
    ///        on; @p stream is ignored, so it must be that same stream (see the class comment).
    void
    deallocate_async(void *p, size_t bytes, size_t alignment, cudaStream_t stream) {
        if (p == nullptr) {
            return;
        }
        if (mDevice.is_privateuseone()) {
            c10::GetAllocator(c10::DeviceType::PrivateUse1)->raw_deallocate(p);
            return;
        }
        c10::cuda::CUDAGuard deviceGuard(mDevice);
        TorchResource{}.deallocate_async(p, bytes, alignment, stream);
    }

    /// @brief Synchronous allocation: the block is immediately valid on every stream of the bound
    ///        device. Spelled out rather than inherited from SyncFromAsync so that the
    ///        synchronization happens under the bound device's guard, not the current device's.
    void *
    allocate(size_t bytes, size_t alignment) {
        void *p = allocate_async(bytes, alignment, cudaStream_t{});
        if (mDevice.is_cuda()) {
            c10::cuda::CUDAGuard deviceGuard(mDevice);
            C10_CUDA_CHECK(cudaStreamSynchronize(cudaStream_t{}));
        }
        return p;
    }

    /// @brief Synchronous deallocation; the caller guarantees the memory is no longer in use.
    void
    deallocate(void *p, size_t bytes, size_t alignment) {
        deallocate_async(p, bytes, alignment, cudaStream_t{});
    }

  private:
    torch::Device mDevice;
};

static_assert(nanovdb::cuda::is_async_resource<TorchDeviceResource>::value,
              "TorchDeviceResource must model nanoVDB's stream-ordered AsyncResource concept");

/// @brief The single-space device buffer fvdb keeps grid storage in. Byte-addressed, as
///        nanovdb::GridHandle requires of single-space storage; the device lives in the resource.
using DeviceGridBuffer = nanovdb::cuda::Buffer<std::byte, TorchDeviceResource>;

static_assert(nanovdb::BufferHasDeviceSingle<DeviceGridBuffer>::value,
              "DeviceGridBuffer must be a single-space device buffer");

} // namespace fvdb

#endif // FVDB_TORCHDEVICERESOURCE_H
