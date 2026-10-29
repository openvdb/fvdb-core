// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_VBMCACHE_H
#define FVDB_DETAIL_VBMCACHE_H

#include <nanovdb/tools/cuda/VoxelBlockManager.cuh>

#include <c10/cuda/CUDAStream.h>
#include <torch/types.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <mutex>
#include <vector>

namespace fvdb {

struct GridBatchData;

namespace detail {

/// @brief Device view of every VoxelBlockManager (VBM) of a batch, built together into shared
/// arrays. Grid g's VBM covers its active voxels [1, entries[g].voxelCount] and occupies blocks
/// [entries[g].blockBase, ...) of firstLeafID and jumpMap. All pointers are device pointers
/// valid for the lifetime of the owning GridBatchData.
struct BatchVbm {
    using Entry = nanovdb::tools::cuda::VoxelBlockManagerBatchEntry;

    const uint32_t *firstLeafID = nullptr; // [blockCount]
    const uint64_t *jumpMap     = nullptr; // [blockCount * VbmCache::kJumpMapWordCount]
    const Entry *entries        = nullptr; // [numGrids], ordered by blockBase
    int64_t numGrids            = 0;
    uint64_t blockCount         = 0;
};

/// @brief Device view of one grid's VBM: its slice of the batch's arrays.
struct GridVbm {
    const uint32_t *firstLeafID = nullptr; // [blockCount]
    const uint64_t *jumpMap     = nullptr; // [blockCount * VbmCache::kJumpMapWordCount]
    uint32_t blockCount         = 0;
    uint64_t firstOffset        = 0;       // always 1 when built (value index 0 = background)
    uint64_t lastOffset         = 0;       // == number of active voxels in the grid
};

/// @brief Lazily-built VoxelBlockManagers for every grid of one GridBatchData.
///
/// The VBM partitions a grid's active voxels into fixed-width blocks of kBlockWidth sequential
/// value indices and stores, per block, the ID of the first leaf overlapping the block plus a
/// bitmask (jumpMap) of the in-block positions where subsequent leaves begin. Kernels decode any
/// active voxel's (leafIndex, voxelOffset) in registers via
/// nanovdb::tools::cuda::VoxelBlockManager::decodeInverseMap, which gives occupancy-independent
/// per-active-voxel iteration.
///
/// The first acquire() builds all of the batch's VBMs on the current stream with one upload,
/// one memset and one kernel launch, sized from host metadata, so it never synchronizes the
/// host. Topology is immutable after construction, so the VBMs never need invalidation. The
/// cache belongs to exactly one GridBatchData, views included, and is never serialized.
///
/// Stream safety: the buffers are torch::Tensor storage from the CUDA caching allocator. The
/// first acquire() from a stream other than the build stream makes that stream wait on the
/// build and record_stream()s the buffers for it, so destroying the owner while that stream's
/// kernels are in flight cannot recycle the buffers under them. Callers must acquire() on the
/// stream they launch consuming kernels on.
class VbmCache {
  public:
    static constexpr int kLog2BlockWidth   = 7; // 128 active voxels per VBM block
    static constexpr int kBlockWidth       = 1 << kLog2BlockWidth;
    static constexpr int kJumpMapWordCount = kBlockWidth / 64;

    VbmCache() = default;
    ~VbmCache();

    VbmCache(const VbmCache &)            = delete;
    VbmCache &operator=(const VbmCache &) = delete;
    VbmCache(VbmCache &&)                 = delete;
    VbmCache &operator=(VbmCache &&)      = delete;

    /// @brief Return the VBMs of @p owner, building them on first call. Thread-safe. Every call
    /// must pass the same GridBatchData; GridBatchData::vbm() is the usual entry point.
    const BatchVbm &acquire(const GridBatchData &owner);

    /// @brief One grid's slice of a built BatchVbm, or a zero GridVbm for an empty grid.
    GridVbm gridVbm(int64_t bi) const;

  private:
    void build(const GridBatchData &owner, const c10::cuda::CUDAStream &stream);

    std::mutex mMutex;
    const GridBatchData *mOwner = nullptr;    // set by the build; acquire() checks it
    BatchVbm mVbm;
    std::vector<BatchVbm::Entry> mHostEntries;
    torch::Tensor mFirstLeafID;               // uint32 [blockCount]
    torch::Tensor mJumpMap;                   // uint64 [blockCount * kJumpMapWordCount]
    torch::Tensor mEntries;                   // bytes of numGrids BatchVbm::Entry
    cudaEvent_t mBuiltOn      = nullptr;      // recorded on mBuiltStream after the build
    cudaStream_t mBuiltStream = nullptr;
    std::vector<cudaStream_t> mSyncedStreams; // other streams already ordered after the build
};

} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_VBMCACHE_H
