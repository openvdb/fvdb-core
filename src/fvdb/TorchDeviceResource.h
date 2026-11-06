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
///        must move to another stream is copied (GridStorage::to). Keying to torch's current
///        stream instead would let a buffer retained on another stream be freed to a stream its
///        work was never ordered against.
///
///        The resource serves CUDA devices only. PrivateUse1, fvdb's unified-memory device for
///        multi-GPU builds, is not stream-ordered and is host- and device-accessible, so its grids
///        live in a UnifiedGridBuffer over TorchUnifiedResource instead.
///
///        The resource is a cheap value type (a torch::Device), copyable as nanovdb's resource
///        concept requires, and default-constructible to the current CUDA device so that
///        DeviceGridBuffer, its GridHandle, and the copies cuda::copyTo makes of empty handles
///        stay default-constructible.
struct TorchDeviceResource {
    static constexpr size_t DEFAULT_ALIGNMENT = TorchResource::DEFAULT_ALIGNMENT;

    /// @brief Binds to the current CUDA device.
    TorchDeviceResource() : mDevice(torch::kCUDA, c10::cuda::current_device()) {}

    /// @brief Binds to @p device, which must be an indexed CUDA device.
    explicit TorchDeviceResource(const torch::Device &device) : mDevice(device) {
        TORCH_CHECK(device.is_cuda() && device.has_index(),
                    "TorchDeviceResource: device must be an indexed CUDA device, got ",
                    device);
    }

    const torch::Device &
    device() const {
        return mDevice;
    }

    /// @brief Allocates @p bytes on the bound device, keyed to @p stream, the buffer's retained
    ///        stream (see the class comment).
    void *
    allocate_async(size_t bytes, size_t alignment, cudaStream_t stream) {
        c10::cuda::CUDAGuard deviceGuard(mDevice);
        return TorchResource{}.allocate_async(bytes, alignment, stream);
    }

    /// @brief Frees @p p. The block returns to the stream it was allocated
    ///        on; @p stream is ignored, so it must be that same stream (see the class comment).
    /// @note No device guard: torch's allocator finds the block's device itself, and setting a
    ///       device fails in a child forked after CUDA initialized, where Python's cycle collector
    ///       can free inherited grids.
    void
    deallocate_async(void *p, size_t bytes, size_t alignment, cudaStream_t stream) {
        if (p == nullptr) {
            return;
        }
        TorchResource{}.deallocate_async(p, bytes, alignment, stream);
    }

    /// @brief Synchronous allocation: the block is immediately valid on every stream of the bound
    ///        device. Spelled out rather than inherited from SyncFromAsync so that the
    ///        synchronization happens under the bound device's guard, not the current device's.
    void *
    allocate(size_t bytes, size_t alignment) {
        void *p = allocate_async(bytes, alignment, cudaStream_t{});
        c10::cuda::CUDAGuard deviceGuard(mDevice);
        C10_CUDA_CHECK(cudaStreamSynchronize(cudaStream_t{}));
        return p;
    }

    /// @brief Synchronous deallocation; the caller guarantees the memory is no longer in use.
    void
    deallocate(void *p, size_t bytes, size_t alignment) {
        deallocate_async(p, bytes, alignment, cudaStream_t{});
    }

    /// @brief Equal when memory from one can be freed through the other. Torch's CUDA allocator
    ///        frees a block on whichever device holds it, so any two resources are equal.
    friend bool
    operator==(const TorchDeviceResource &, const TorchDeviceResource &) {
        return true;
    }
    friend bool
    operator!=(const TorchDeviceResource &, const TorchDeviceResource &) {
        return false;
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

/// @brief Memory resource over the PrivateUse1 allocator, fvdb's unified memory, and the resource
///        behind every PrivateUse1 grid. It declares host and device accessibility so NanoVDB
///        treats its buffers as both-space storage, which DistributedPointsToGrid requires because
///        every device in the mesh writes the grid.
///
///        Like NanoVDB's ManagedResource it is synchronous, because the allocator is not
///        stream-ordered. The caller ensures device work on an allocation has finished before
///        freeing it.
struct TorchUnifiedResource {
    static constexpr size_t DEFAULT_ALIGNMENT = TorchResource::DEFAULT_ALIGNMENT;
    static constexpr bool HOST_ACCESSIBLE     = true;
    static constexpr bool DEVICE_ACCESSIBLE   = true;

    void *
    allocate(size_t bytes, size_t /*alignment*/) {
        void *p = c10::GetAllocator(c10::DeviceType::PrivateUse1)->raw_allocate(bytes);
        if (!p && bytes) {
            throw std::runtime_error("fvdb: TorchUnifiedResource::allocate failed");
        }
        return p;
    }

    void
    deallocate(void *p, size_t /*bytes*/, size_t /*alignment*/) {
        if (p != nullptr) {
            c10::GetAllocator(c10::DeviceType::PrivateUse1)->raw_deallocate(p);
        }
    }

    /// @brief Stateless, so any two instances are equal.
    friend bool
    operator==(const TorchUnifiedResource &, const TorchUnifiedResource &) {
        return true;
    }
    friend bool
    operator!=(const TorchUnifiedResource &, const TorchUnifiedResource &) {
        return false;
    }
};

/// @brief Byte-addressed both-space buffer over PrivateUse1 unified memory.
using UnifiedGridBuffer = nanovdb::cuda::Buffer<std::byte, TorchUnifiedResource>;

static_assert(nanovdb::BufferHasHostSingle<UnifiedGridBuffer>::value &&
                  nanovdb::BufferHasDeviceSingle<UnifiedGridBuffer>::value,
              "UnifiedGridBuffer must be host- and device-accessible single-space storage");

} // namespace fvdb

#endif // FVDB_TORCHDEVICERESOURCE_H
