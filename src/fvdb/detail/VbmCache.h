// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_VBMCACHE_H
#define FVDB_DETAIL_VBMCACHE_H

#include <fvdb/GridBatchData.h>

#include <nanovdb/tools/cuda/VoxelBlockManager.cuh>

#include <c10/cuda/CUDAStream.h>
#include <torch/types.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <mutex>
#include <vector>

namespace fvdb {
namespace detail {

/// @brief Device view of every VoxelBlockManager (VBM) of a batch, built together into shared
/// arrays. Grid g's VBM covers its active voxels and starts at block VbmCache::blockBase(g) of
/// firstLeafID and jumpMap (see BatchVbmLayout). All pointers are device pointers valid for the
/// lifetime of the owning GridBatchData.
struct BatchVbm {
    const uint32_t *firstLeafID = nullptr; // [blockCount]
    const uint64_t *jumpMap     = nullptr; // [blockCount * VbmCache::kJumpMapWordCount]
    int64_t numGrids            = 0;
    uint64_t blockCount         = 0;       // includes the unused blocks between grids
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
/// The first acquire() builds all of the batch's VBMs on the current stream into one buffer
/// with one memset and one kernel, which derives each grid's placement from the batch's device
/// metadata (BatchVbmLayout). Sizes come from host metadata, so the build never synchronizes
/// the host.
/// Topology is immutable after construction, so the VBMs never need invalidation. The cache belongs
/// to exactly one GridBatchData, views included, and is never serialized.
///
/// Stream safety: the buffer is torch::Tensor storage from the CUDA caching allocator. The
/// first acquire() from a stream other than the build stream makes that stream wait on the
/// build and record_stream()s the buffer for it, so destroying the owner while that stream's
/// kernels are in flight cannot recycle the buffer under them. Callers must acquire() on the
/// stream they launch consuming kernels on.
class VbmCache {
  public:
    static constexpr int kLog2BlockWidth   = 7; // 128 active voxels per VBM block
    static constexpr int kBlockWidth       = 1 << kLog2BlockWidth;
    static constexpr int kJumpMapWordCount = kBlockWidth / 64;

    /// @brief First block of grid g's VBM, given the batch's voxels before it. A grid spans at
    /// most one block more than its share of the batch's voxels, so this leaves room for every
    /// grid without a scan over the grids' block counts. The unused blocks are never decoded.
    __hostdev__ static constexpr uint64_t
    blockBase(int64_t cumVoxels, int64_t g) {
        return (uint64_t(cumVoxels) >> kLog2BlockWidth) + uint64_t(g);
    }

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
    std::vector<nanovdb::tools::cuda::VoxelBlockManagerBatchEntry> mHostEntries;
    torch::Tensor mBuffer;                    // bytes: jumpMap, entries, firstLeafID
    cudaEvent_t mBuiltOn      = nullptr;      // created when a second stream first acquires
    cudaStream_t mBuiltStream = nullptr;
    std::vector<cudaStream_t> mSyncedStreams; // other streams already ordered after the build
};

/// @brief Each grid's VBM placement in its batch's shared arrays, derived from the batch's
/// device-side voxel offsets and grid metadata. Serves as buildVoxelBlockManagerBatch's entry
/// source and as the iteration kernel's grid lookup. The lookup binary-searches voxelOffsets,
/// which is compact (8 bytes per grid), rather than the much larger per-grid metadata.
struct BatchVbmLayout {
    GridBatchData::Accessor acc;
    const int64_t *voxelOffsets; // [numGrids + 1], device

    static BatchVbmLayout
    of(const GridBatchData &batch) {
        const torch::Tensor &offsets = batch.voxelOffsets();
        TORCH_CHECK(offsets.is_contiguous() && offsets.scalar_type() == torch::kInt64,
                    "VbmCache expects contiguous int64 voxel offsets");
        return {batch.deviceAccessor(), offsets.const_data_ptr<int64_t>()};
    }

    __hostdev__ uint64_t
    voxelCount(int64_t g) const {
        return uint64_t(voxelOffsets[g + 1] - voxelOffsets[g]);
    }

    __hostdev__ uint64_t
    blockBase(int64_t g) const {
        return VbmCache::blockBase(voxelOffsets[g], g);
    }

    __hostdev__ nanovdb::tools::cuda::VoxelBlockManagerBatchEntry
    operator()(uint32_t g) const {
        return {acc.grid(g), voxelCount(g), blockBase(g), uint32_t(acc.leafOffset(g))};
    }
};

} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_VBMCACHE_H
