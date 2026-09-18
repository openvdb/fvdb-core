// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_TORCHDEVICEBUFFER_H
#define FVDB_TORCHDEVICEBUFFER_H

#include <fvdb/TorchResource.h>

#include <nanovdb/GridHandle.h>
#include <nanovdb/HostBuffer.h> // for BufferTraits

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/types.h>

// Everything in this header is deprecated. fvdb no longer uses it; it stays for one release as
// a shim for downstream code that took TorchDeviceBuffer from the public headers (#632), and is
// deleted with the NanoVDB pin bump past upstream's removal of dual-space buffers (openvdb #2232).
// The deprecation warnings this header's own definitions would raise are silenced here; a user of
// the header gets them at every use.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

namespace fvdb {

/// @brief Torch-allocator resource for device storage that torch owns after the op that
///        produced it: grid buffers.
///
///        The block is always taken on the current device's current torch stream, never on
///        the caller's, because that is the only stream torch will reuse it from and it is a
///        stream that outlives the block. A nanovdb builder may write storage on a stream
///        of its own (DistributedPointsToGrid uses its DeviceMesh stream, which dies with the
///        mesh); a block keyed to such a stream would never be reused, would make torch
///        synchronize on a dead handle when releasing an expandable segment, and would be
///        cudaFreeAsync'd on it under that backend.
///
///        @p stream is the stream the caller will write the memory on. When it differs from
///        the allocation stream it is made to wait on an event recorded right after the
///        allocation, so the caller's writes are ordered behind the block's previous tenant,
///        whose work may still be queued on the torch stream. Nothing orders the free: as
///        for a tensor, the block returns to the torch stream on release, so a caller whose
///        writer stream differs must synchronize it before releasing the storage. The
///        NanoVDB builders synchronize before returning a handle over this buffer.
///
///        Only TorchDeviceBuffer uses this, and it is deprecated with it. TorchDeviceResource,
///        the resource behind GridStorage, deliberately keys the block to the buffer's retained
///        stream instead (see TorchDeviceResource.h).
struct [[deprecated(
    "TorchStorageResource served TorchDeviceBuffer; grid storage is GridStorage over "
    "TorchDeviceResource")]] TorchStorageResource
    : nanovdb::cuda::SyncFromAsync<TorchStorageResource> {
    static constexpr size_t DEFAULT_ALIGNMENT = TorchResource::DEFAULT_ALIGNMENT;

    void *
    allocate_async(size_t bytes, size_t alignment, cudaStream_t stream) {
        const cudaStream_t allocStream = c10::cuda::getCurrentCUDAStream().stream();
        void *p = TorchResource{}.allocate_async(bytes, alignment, allocStream);
        if (stream == allocStream) {
            return p;
        }
        cudaEvent_t ready = nullptr;
        cudaError_t err   = cudaEventCreateWithFlags(&ready, cudaEventDisableTiming);
        if (err == cudaSuccess) {
            err = cudaEventRecord(ready, allocStream);
        }
        if (err == cudaSuccess) {
            err = cudaStreamWaitEvent(stream, ready, 0);
        }
        if (ready) {
            const cudaError_t destroyErr = cudaEventDestroy(ready);
            if (err == cudaSuccess) {
                err = destroyErr;
            }
        }
        if (err != cudaSuccess) {
            c10::cuda::CUDACachingAllocator::raw_delete(p);
            C10_CUDA_CHECK(err);
        }
        return p;
    }

    void
    deallocate_async(void *p, size_t bytes, size_t alignment, cudaStream_t stream) {
        TorchResource{}.deallocate_async(p, bytes, alignment, stream);
    }
};

static_assert(nanovdb::cuda::is_async_resource<TorchStorageResource>::value,
              "TorchStorageResource must model nanoVDB's stream-ordered AsyncResource concept");

/// @brief Deprecated: the dual-space buffer fvdb's grid storage used to be a nanovdb::GridHandle
///        over. Grid storage is GridStorage (GridStorage.h), a single-space HostBuffer or
///        nanovdb::cuda::Buffer<std::byte, TorchDeviceResource> handle; the NanoVDB builders take
///        their output storage from GridStorage::deviceProto. Nothing in fvdb constructs or
///        accepts a TorchDeviceBuffer any more.
class [[deprecated("fvdb grid storage is GridStorage; TorchDeviceBuffer is unused and will be "
                   "removed")]] TorchDeviceBuffer {
    uint64_t mSize; // total number of bytes for the NanoVDB grid.
    uint8_t *mData; // raw buffer for the NanoVDB grid.
    torch::Device mDevice{torch::kCPU};

  public:
    /// @brief Default constructor initializes a buffer with the given size and device specified by
    /// host and deviceIndex.
    /// @note This has a weird API because it has to match other buffer classes in nanovdb like
    /// nanovdb::HostBuffer
    /// @param size The size (in bytes to allocate for this buffer)
    /// @param device Specifies the device to use for the buffer
    /// @param stream For a CUDA device, the stream the caller will write the buffer on. The
    /// allocation is made through fvdb::TorchStorageResource, which always takes it on the
    /// device's current torch stream and orders @p stream after that if the two differ (see
    /// TorchStorageResource for why). Null, the default, is the legacy default stream
    /// and is ordered like any other. Ignored for CPU and PrivateUse1 devices.
    TorchDeviceBuffer(uint64_t size               = 0,
                      const torch::Device &device = torch::kCPU,
                      void *stream                = nullptr);

    /// @brief Disallow copy-construction
    TorchDeviceBuffer(const TorchDeviceBuffer &) = delete;

    /// @brief Move copy-constructor
    TorchDeviceBuffer(TorchDeviceBuffer &&other) noexcept;

    /// @brief Disallow copy assignment operation
    TorchDeviceBuffer &operator=(const TorchDeviceBuffer &) = delete;

    /// @brief Move copy assignment operation
    TorchDeviceBuffer &operator=(TorchDeviceBuffer &&other) noexcept;

    /// @brief Destructor frees memory on specified device
    ~TorchDeviceBuffer();

    /// @brief Returns the device used by this buffer
    /// @return The device used by this buffer
    const torch::Device &device() const;

    /// @brief Moves the buffer to the specified device
    void to(const torch::Device &device);

    /// @brief Returns a pointer to the CPU memory buffer managed by this allocator if the device is
    /// torch::kCPU or torch::kPrivateUse1, nullptr otherwise.
    uint8_t *data() const;

    /// @brief Returns a pointer to the GPU memory buffer managed by this allocator if the device is
    /// torch::kCUDA or torch::kPrivateUse1, nullptr otherwise.
    uint8_t *deviceData() const;

    /// @brief Returns the size in bytes of the raw memory buffer managed by this allocator.
    uint64_t size() const;

    /// @brief Returns true if this allocator is empty, i.e. has no allocated memory
    bool empty() const;

    /// @copydoc empty()
    bool isEmpty() const;

    /// @brief De-allocate all memory managed by this allocator and set all pointer to NULL
    void clear();

    /// @brief Static factory method that return an instance of this buffer
    /// @param size byte size of buffer to be initialized
    /// @param guide this argument is there to match the signature of the other create() methods
    /// (e.g. nanovdb::HostBuffer) and to provide a way to specify the device to be used for the
    /// buffer. i.e. if guide is non-null, the created buffer will be on the same device as guide!
    /// note you must also set the device argument to match the guide buffer device
    /// @param device Device index for the buffer. If you passed in a guide buffer, then this must
    /// match the device of the guide buffer!
    /// @return An instance of this class using move semantics
    static TorchDeviceBuffer create(uint64_t size,
                                    const TorchDeviceBuffer *guide = nullptr,
                                    int device                     = -1, // cudaCpuDeviceId
                                    void *stream                   = nullptr);

}; // TorchDeviceBuffer class

} // namespace fvdb

namespace nanovdb {
template <> struct BufferTraits<fvdb::TorchDeviceBuffer> {
    static const bool hasDeviceDual = true;
};

template <>
template <>
GridHandle<fvdb::TorchDeviceBuffer>
GridHandle<fvdb::TorchDeviceBuffer>::copy<fvdb::TorchDeviceBuffer>(
    const fvdb::TorchDeviceBuffer &guide) const;

} // namespace nanovdb

#pragma GCC diagnostic pop

#endif // FVDB_TORCHDEVICEBUFFER_H
