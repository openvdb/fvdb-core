// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_CUDA_FOREACHVBMCUDA_CUH
#define FVDB_DETAIL_UTILS_CUDA_FOREACHVBMCUDA_CUH

#include <fvdb/GridBatchData.h>
#include <fvdb/detail/VbmCache.h>

#include <nanovdb/tools/cuda/VoxelBlockManager.cuh>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

namespace fvdb {

namespace _private {

constexpr int kVbmForEachBlockDim = 256;

/// Grid-stride over one grid's active voxels: each iteration decodes one voxel's
/// (leafIndex, voxelOffset) in registers via the VoxelBlockManager's rank+select inverse-map
/// decode -- no shared memory, no barriers, and no threads wasted on inactive leaf slots
/// (unlike the totalLeaves*512 leaf scan in forEachVoxelCUDAKernel). decodeInverseMap takes
/// the in-block slot position explicitly, so the launch shape is decoupled from the VBM's
/// 512-slot blocks; a 256-thread grid-stride kernel leaves the per-thread register budget to
/// the callback. (A one-thread-per-slot <<<blockCount, 512>>> shape with
/// __launch_bounds__(512) was observed to silently drop work from a register-heavy callback.)
template <typename Func, typename... Args>
__global__ void __launch_bounds__(kVbmForEachBlockDim)
forEachActiveVoxelVbmKernel(const nanovdb::OnIndexGrid *__restrict__ grid,
                            const uint32_t *__restrict__ firstLeafID,
                            const uint64_t *__restrict__ jumpMap,
                            uint64_t firstOffset,
                            uint64_t lastOffset,
                            int64_t baseVoxelOffset, // cumVoxelsAt(bi) of this grid in the batch
                            Func func,
                            Args... args) {
    using VbmT = nanovdb::tools::cuda::VoxelBlockManager<detail::VbmCache::kLog2BlockWidth>;

    const uint64_t numSlots = lastOffset - firstOffset + 1;
    const uint64_t stride   = uint64_t(gridDim.x) * blockDim.x;

    for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < numSlots; i += stride) {
        const uint32_t vbmBlock = uint32_t(i >> detail::VbmCache::kLog2BlockWidth);
        const int blockOffset   = int(i & (detail::VbmCache::kBlockWidth - 1));
        const uint64_t blockFirstOffset =
            firstOffset + uint64_t(vbmBlock) * detail::VbmCache::kBlockWidth;
        uint32_t leafIndex;
        uint16_t voxelOffset;
        VbmT::decodeInverseMap(grid,
                               firstLeafID[vbmBlock],
                               jumpMap + uint64_t(vbmBlock) * detail::VbmCache::kJumpMapWordCount,
                               blockFirstOffset,
                               blockOffset,
                               leafIndex,
                               voxelOffset);
        if (leafIndex == VbmT::UnusedLeafIndex) {
            continue; // defensive; i < numSlots makes this unreachable
        }
        const auto &leaf         = grid->tree().template getFirstNode<0>()[leafIndex];
        const nanovdb::Coord ijk = leaf.offsetToGlobalCoord(voxelOffset);
        // Sequential OnIndex invariant: the decoded voxel's value index equals its slot
        // (firstOffset + i), so the batch-wide feature index is baseVoxelOffset +
        // (firstOffset + i) - 1 (matching the legacy path's baseOffset + getValue - 1).
        const int64_t featureIdx = baseVoxelOffset + int64_t(firstOffset + i) - 1;
        func.perActiveVoxel(ijk, featureIdx, args...);
    }
}

} // namespace _private

/// @brief Run func.perActiveVoxel(ijk, featureIdx, args...) for every active voxel of every
/// grid in the batch, using the batch's cached per-grid VoxelBlockManagers: exactly one
/// grid-stride iteration per active voxel, independent of leaf occupancy.
///
/// The callback contract matches BasePerActiveVoxelProcessor::perActiveVoxel:
///     void perActiveVoxel(nanovdb::Coord const &ijk, int64_t featureIdx, Args...) const
/// where featureIdx is the batch-wide linear voxel index (cumVoxelsAt(bi) + in-grid index).
///
/// One kernel launch per grid in the batch. CUDA-only; callers must fall back to the legacy
/// leaf-scan paths for CPU and PrivateUse1 grids.
template <typename Func, typename... Args>
void
forEachActiveVoxelVbmCUDA(const fvdb::GridBatchData &batchHdl, Func func, Args... args) {
    TORCH_CHECK(batchHdl.device().is_cuda(), "Grid batch must be on a CUDA device");
    TORCH_CHECK(batchHdl.device().has_index(), "Grid batch device must have an index");
    c10::cuda::CUDAGuard deviceGuard(batchHdl.device());
    const at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(batchHdl.device().index());

    for (int64_t bi = 0; bi < batchHdl.batchSize(); ++bi) {
        const auto vbm = batchHdl.vbmCache().get(batchHdl, bi);
        if (vbm.blockCount == 0) {
            continue; // empty grid
        }
        const uint64_t numSlots  = vbm.lastOffset - vbm.firstOffset + 1;
        const uint32_t numBlocks = uint32_t((numSlots + _private::kVbmForEachBlockDim - 1) /
                                            _private::kVbmForEachBlockDim);
        _private::forEachActiveVoxelVbmKernel<<<numBlocks,
                                                _private::kVbmForEachBlockDim,
                                                0,
                                                stream.stream()>>>(batchHdl.deviceGridPtrAt(bi),
                                                                   vbm.firstLeafID,
                                                                   vbm.jumpMap,
                                                                   vbm.firstOffset,
                                                                   vbm.lastOffset,
                                                                   batchHdl.cumVoxelsAt(bi),
                                                                   func,
                                                                   args...);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

} // namespace fvdb

#endif // FVDB_DETAIL_UTILS_CUDA_FOREACHVBMCUDA_CUH
