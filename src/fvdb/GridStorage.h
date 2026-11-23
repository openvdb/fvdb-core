// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_GRIDSTORAGE_H
#define FVDB_GRIDSTORAGE_H

#include <fvdb/TorchDeviceResource.h>

#include <nanovdb/GridHandle.h>
#include <nanovdb/HostBuffer.h>
#include <nanovdb/cuda/HandleStorage.h>

#include <torch/types.h>

#include <cstdint>
#include <variant>
#include <vector>

namespace fvdb {

/// @brief The bytes of a batch of NanoVDB grids, on one torch device, with the per-grid
///        metadata that locates each grid inside them.
///
///        This is the storage behind GridBatchData. It holds one of three single-space NanoVDB
///        handles, and the storage's device decides which:
///        - CPU: HostHandle (HostBuffer).
///        - CUDA: DeviceHandle (DeviceGridBuffer, over TorchDeviceResource). The device travels
///          in the buffer's resource, and the buffer retains the stream it was made on.
///        - PrivateUse1: UnifiedHandle (UnifiedGridBuffer, over TorchUnifiedResource). Unified
///          memory is host- and device-accessible and has no retained stream.
///
///        isDevice() is true for CUDA and PrivateUse1 alike, so it does not mean deviceHandle()
///        is valid. Code that serves both should use deviceBytes() and deviceGridAt(), which
///        cover either handle.
///
///        Everything a handle exposes about its grids (count, sizes, types, typed grid pointers)
///        is answered from adopted metadata without touching device memory. Grid indices here are
///        *physical* positions in the storage; a GridBatchData view over a subset of the batch
///        maps its logical indices onto these.
///
///        This header is host-includable: constructing a device handle from raw bytes (which
///        parses on the device) is the one operation that needs a CUDA translation unit, and it
///        lives with the tools that produce such bytes, not here.
class GridStorage {
  public:
    using HostHandle    = nanovdb::GridHandle<nanovdb::HostBuffer>;
    using DeviceHandle  = nanovdb::GridHandle<DeviceGridBuffer>;
    using UnifiedHandle = nanovdb::GridHandle<UnifiedGridBuffer>;

    /// @brief Empty storage on the CPU.
    GridStorage();

    /// @brief Adopts a host handle. The storage's device is kCPU.
    explicit GridStorage(HostHandle &&handle);

    /// @brief Adopts a device handle. The storage's device is the buffer's resource device (for an
    ///        empty handle, whose default resource is the current CUDA device, prefer the overload
    ///        that names the device).
    explicit GridStorage(DeviceHandle &&handle);

    /// @brief Adopts a device handle for storage that lives on @p device, which must be a CUDA
    ///        device that agrees with the handle's resource unless the handle is empty.
    GridStorage(DeviceHandle &&handle, const torch::Device &device);

    /// @brief Adopts a unified handle for storage on @p device, which must be PrivateUse1.
    GridStorage(UnifiedHandle &&handle, const torch::Device &device);

    GridStorage(GridStorage &&) noexcept            = default;
    GridStorage &operator=(GridStorage &&) noexcept = default;
    GridStorage(const GridStorage &)                = delete;
    GridStorage &operator=(const GridStorage &)     = delete;

    /// @brief The device a caller means: an index-less CUDA device (`torch::device("cuda")`)
    ///        resolves to the current CUDA device, as the storage resource defaults to it; anything
    ///        else is returned as given. Every entry point taking a device applies this first.
    static torch::Device resolveDevice(const torch::Device &device);

    /// @brief Empty storage on @p device (kCPU, a CUDA device, or PrivateUse1). Empty CUDA
    ///        storage retains @p stream.
    static GridStorage empty(const torch::Device &device, cudaStream_t stream = cudaStream_t{});

    /// @brief A zero-byte DeviceGridBuffer carrying @p device in its resource and @p stream as its
    ///        retained stream: the prototype the NanoVDB builders and cuda::copyTo take the output
    ///        resource and stream from. @p device must be a CUDA device.
    static DeviceGridBuffer deviceProto(const torch::Device &device, cudaStream_t stream);

    const torch::Device &
    device() const {
        return mDevice;
    }
    bool
    isHost() const {
        return std::holds_alternative<HostHandle>(mHandle);
    }
    /// @brief True for CUDA and PrivateUse1 storage. Only CUDA storage has a deviceHandle().
    bool
    isDevice() const {
        return !isHost();
    }

    /// @brief True if no bytes are held. Empty storage still has a device.
    bool isEmpty() const;
    /// @brief Bytes held, including any padding after the last grid.
    uint64_t bufferSize() const;
    /// @brief Number of grids in the storage.
    uint32_t gridCount() const;
    /// @brief Byte size of the @p i'th grid.
    uint64_t gridSize(uint32_t i) const;
    /// @brief Byte offset of the @p i'th grid from the start of the storage.
    uint64_t gridOffset(uint32_t i) const;
    /// @brief Value type of the @p i'th grid.
    nanovdb::GridType gridType(uint32_t i) const;

    //@{
    /// @brief Start of the bytes as seen from the host: CPU storage, or PrivateUse1 storage, whose
    ///        unified memory is host-accessible. nullptr for CUDA storage and for empty storage.
    const void *hostBytes() const;
    void *hostBytes();
    //@}
    //@{
    /// @brief Start of the bytes as seen from the device: CUDA or PrivateUse1 storage. nullptr for
    ///        CPU storage and for empty storage.
    const void *deviceBytes() const;
    void *deviceBytes();
    //@}

    //@{
    /// @brief Host pointer to the @p i'th grid, or nullptr if the bytes are not host-accessible,
    ///        @p i is out of range, or the grid's value type is not @p ValueT. PrivateUse1 storage
    ///        is unified memory, so its device pointer serves.
    template <typename ValueT>
    const nanovdb::NanoGrid<ValueT> *
    hostGridAt(uint32_t i) const {
        if (const auto *h = std::get_if<HostHandle>(&mHandle)) {
            return h->template grid<ValueT>(i);
        }
        if (const auto *h = std::get_if<UnifiedHandle>(&mHandle)) {
            return h->template deviceGrid<ValueT>(i);
        }
        return nullptr;
    }
    template <typename ValueT>
    nanovdb::NanoGrid<ValueT> *
    hostGridAt(uint32_t i) {
        return const_cast<nanovdb::NanoGrid<ValueT> *>(
            static_cast<const GridStorage *>(this)->hostGridAt<ValueT>(i));
    }
    //@}
    //@{
    /// @brief Device pointer to the @p i'th grid, or nullptr if the bytes are not
    ///        device-accessible, @p i is out of range, or the grid's value type is not @p ValueT.
    template <typename ValueT>
    const nanovdb::NanoGrid<ValueT> *
    deviceGridAt(uint32_t i) const {
        if (const auto *h = std::get_if<DeviceHandle>(&mHandle)) {
            return h->template deviceGrid<ValueT>(i);
        }
        if (const auto *h = std::get_if<UnifiedHandle>(&mHandle)) {
            return h->template deviceGrid<ValueT>(i);
        }
        return nullptr;
    }
    template <typename ValueT>
    nanovdb::NanoGrid<ValueT> *
    deviceGridAt(uint32_t i) {
        return const_cast<nanovdb::NanoGrid<ValueT> *>(
            static_cast<const GridStorage *>(this)->deviceGridAt<ValueT>(i));
    }
    //@}

    /// @brief The stream CUDA storage frees on and orders its work against (the buffer's
    ///        retained stream); the legacy default stream for CPU and PrivateUse1 storage.
    cudaStream_t stream() const;

    /// @brief A deep copy of the storage on @p device, made on @p stream. Empty storage yields
    ///        empty storage on @p device without allocating. For a device destination the copy is
    ///        ordered on @p stream and the result retains it; a host destination is complete on
    ///        return. When the source is device storage, @p stream is ordered after the source's
    ///        retained stream before the copy (so the source's writers are done), and the source's
    ///        stream is ordered after the copy (so destroying the source right after this returns
    ///        cannot let torch reuse its bytes under the copy).
    GridStorage to(const torch::Device &device, cudaStream_t stream) const;

    /// @brief to() on the destination device's current torch stream (the source's retained
    ///        stream when the destination is the host).
    GridStorage to(const torch::Device &device) const;

    //@{
    /// @brief The underlying handle: hostHandle() for CPU, deviceHandle() for CUDA and
    ///        unifiedHandle() for PrivateUse1 storage. Throws for any other device.
    const HostHandle &hostHandle() const;
    HostHandle &hostHandle();
    const DeviceHandle &deviceHandle() const;
    DeviceHandle &deviceHandle();
    const UnifiedHandle &unifiedHandle() const;
    UnifiedHandle &unifiedHandle();
    //@}

    /// @brief The adopted per-grid metadata, in physical order.
    const std::vector<nanovdb::GridHandleMetaData> &metadata() const;

  private:
    const nanovdb::GridHandleMetaData &metaAt(uint32_t i) const;

    std::variant<HostHandle, DeviceHandle, UnifiedHandle> mHandle;
    torch::Device mDevice;
};

} // namespace fvdb

#endif // FVDB_GRIDSTORAGE_H
