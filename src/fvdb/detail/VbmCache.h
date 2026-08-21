// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_VBMCACHE_H
#define FVDB_DETAIL_VBMCACHE_H

#include <nanovdb/HostBuffer.h> // for nanovdb::BufferTraits

#include <c10/core/Device.h>
#include <torch/types.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace fvdb {

struct GridBatchData;

namespace detail {

/// @brief Non-owning device-buffer view satisfying the NanoVDB buffer concept
/// (data()/deviceData()/clear() returning void*). Used only to run the in-place
/// buildVoxelBlockManager over memory whose ownership and stream-safety are managed by the
/// PyTorch caching allocator (torch::Tensor storage held by VbmCache::Entry).
class VbmBufferView {
    void *mDevicePtr = nullptr;

  public:
    VbmBufferView() = default;
    explicit VbmBufferView(void *devicePtr) : mDevicePtr(devicePtr) {}
    VbmBufferView(VbmBufferView &&)            = default;
    VbmBufferView &operator=(VbmBufferView &&) = default;

    void *
    data() const {
        return nullptr; // host side unused: the cache is CUDA-only
    }
    void *
    deviceData() const {
        return mDevicePtr;
    }
    void
    clear() {
        mDevicePtr = nullptr; // non-owning
    }
};

} // namespace detail
} // namespace fvdb

namespace nanovdb {
template <> struct BufferTraits<fvdb::detail::VbmBufferView> {
    static const bool hasDeviceDual = true;
};
} // namespace nanovdb

namespace fvdb {
namespace detail {

/// @brief Lazily-built, per-grid NanoVDB VoxelBlockManager (VBM) handles for a GridBatchData.
///
/// The VBM partitions a grid's active voxels into fixed-width blocks of BlockWidth sequential
/// value indices and stores, per block, the ID of the first leaf overlapping the block plus a
/// bitmask (jumpMap) of the in-block positions where subsequent leaves begin. Kernels can then
/// decode any active voxel's (leafIndex, voxelOffset) in registers via
/// nanovdb::tools::cuda::VoxelBlockManager::decodeInverseMap, giving occupancy-independent
/// per-active-voxel iteration (one decode per active voxel instead of one thread per 512-slot
/// leaf position).
///
/// Grid topology is immutable after GridBatchData construction (every topology op returns a new
/// GridBatchData), so entries never need invalidation; the cache is pure derived state and must
/// never be serialized. It is shared between a GridBatchData and any sliced/indexed views of it
/// (which share the same underlying grid buffer); entries are keyed by the grid's byte offset
/// into that shared buffer, which is the only view-stable per-grid identity.
///
/// Stream safety: entry buffers are torch::Tensor storage from the CUDA caching allocator.
/// get() makes a consumer stream that differs from the build stream wait on the recorded build
/// event (execution ordering) and record_stream()s the buffers for it (lifetime ordering), so
/// destroying the owning GridBatchData while a cross-stream consumer kernel is still in flight
/// cannot recycle the buffers under it. Callers must invoke get() on the same current stream
/// they subsequently launch consuming kernels on.
///
/// CUDA-only: grids on other devices must use the legacy leaf-scan iteration paths.
class VbmCache {
  public:
    static constexpr int kLog2BlockWidth   = 7; // 128 active voxels per VBM block
    static constexpr int kBlockWidth       = 1 << kLog2BlockWidth;
    static constexpr int kJumpMapWordCount = kBlockWidth / 64;

    /// @brief POD view of one grid's VBM metadata. All pointers are device pointers valid for
    /// the lifetime of the owning GridBatchData (and of any views sharing its grid buffer).
    struct GridVbm {
        const uint32_t *firstLeafID = nullptr; // [blockCount]
        const uint64_t *jumpMap     = nullptr; // [blockCount * kJumpMapWordCount]
        uint32_t blockCount         = 0;
        uint64_t firstOffset        = 0;       // always 1 when built (value index 0 = background)
        uint64_t lastOffset         = 0;       // == number of active voxels in the grid
    };

    VbmCache() = default;
    ~VbmCache();

    VbmCache(const VbmCache &)            = delete;
    VbmCache &operator=(const VbmCache &) = delete;
    VbmCache(VbmCache &&)                 = delete;
    VbmCache &operator=(VbmCache &&)      = delete;

    /// @brief Return the VBM for logical grid @p bi of @p batch, building it on the current
    /// CUDA stream of the batch's device on first access. Thread-safe. If a later call arrives
    /// on a different stream than the one the entry was built on, that stream is made to wait
    /// on the recorded build event and is registered with the caching allocator as a user of
    /// the entry's buffers before this returns.
    /// @return The grid's VBM view, or a zero GridVbm (blockCount == 0) for an empty grid.
    GridVbm get(const GridBatchData &batch, int64_t bi);

  private:
    struct Entry {
        torch::Tensor firstLeafID; // int32 [blockCount] (holds uint32 values)
        torch::Tensor jumpMap;     // int64 [blockCount * kJumpMapWordCount] (holds uint64 bits)
        GridVbm view;
        cudaEvent_t builtOn        = nullptr; // recorded on the build stream
        cudaStream_t builtStream   = nullptr;
        c10::DeviceIndex deviceIdx = -1;
    };

    std::mutex mMutex;
    std::unordered_map<uint64_t, Entry> mEntries; // keyed by GridBatchData::cumBytesAt(bi)
};

} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_VBMCACHE_H
