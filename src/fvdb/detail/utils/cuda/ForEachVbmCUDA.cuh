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

#include <algorithm>
#include <limits>

namespace fvdb {

namespace _private {

constexpr int kVbmForEachBlockDim    = 256;
constexpr int kVbmBlocksPerCudaBlock = kVbmForEachBlockDim / detail::VbmCache::kBlockWidth;
static_assert(kVbmForEachBlockDim % detail::VbmCache::kBlockWidth == 0,
              "each CUDA block must cover whole VBM blocks");

/// One launch over every active voxel of a batch. Each group of kBlockWidth threads takes one
/// VBM block of the batch (grid-stride) and finds its grid through the batch view's
/// lastGridAtOrBefore, a warp-uniform binary search over the grids' block bases. Each thread
/// decodes one voxel's (leafIndex, voxelOffset) in registers via the VoxelBlockManager's
/// rank+select inverse-map decode.
///
/// Keep this kernel at 256 threads. A one-thread-per-slot <<<blockCount, 512>>> shape with
/// __launch_bounds__(512) was observed to silently drop work from a register-heavy callback.
template <typename Func, typename... Args>
__global__ void __launch_bounds__(kVbmForEachBlockDim)
forEachActiveVoxelVbmKernel(const detail::BatchVbm vbm,
                            const detail::BatchVbmLayout layout,
                            Func func,
                            Args... args) {
    using VbmCache = detail::VbmCache;
    using VbmT     = nanovdb::tools::cuda::VoxelBlockManager<VbmCache::kLog2BlockWidth>;
    using ViewT    = detail::BatchVbmView;

    const ViewT view(layout, uint32_t(vbm.numGrids), vbm.blockCount, vbm.firstLeafID, vbm.jumpMap);
    const int slot        = int(threadIdx.x) & (VbmCache::kBlockWidth - 1);
    const uint64_t stride = uint64_t(gridDim.x) * kVbmBlocksPerCudaBlock;
    for (uint64_t vb =
             uint64_t(blockIdx.x) * kVbmBlocksPerCudaBlock + threadIdx.x / VbmCache::kBlockWidth;
         vb < vbm.blockCount;
         vb += stride) {
        // Blocks between grids map to the grid before them and fall past its voxels below
        const uint32_t bi = view.lastGridAtOrBefore(vb);
        if (bi == ViewT::UnusedGridIndex) {
            continue;
        }
        const uint64_t b = vb - view.blockBase(bi);          // VBM block within the grid
        const uint64_t i = b * VbmCache::kBlockWidth + slot; // 0-based voxel in the grid
        if (i >= view.voxelCount(bi)) {
            continue;
        }
        const nanovdb::OnIndexGrid *grid = layout.acc.grid(bi);
        uint32_t leafIndex;
        uint16_t voxelOffset;
        VbmT::decodeInverseMap(grid,
                               view.firstLeafID()[vb],
                               view.jumpMap() + vb * VbmCache::kJumpMapWordCount,
                               1 + b * VbmCache::kBlockWidth,
                               slot,
                               leafIndex,
                               voxelOffset);
        if (leafIndex == VbmT::UnusedLeafIndex) {
            continue; // defensive; i < voxelCount makes this unreachable
        }
        const auto &leaf         = grid->tree().template getFirstNode<0>()[leafIndex];
        const nanovdb::Coord ijk = leaf.offsetToGlobalCoord(voxelOffset);
        // Sequential OnIndex invariant: the decoded voxel's value index equals its slot (i + 1),
        // so the batch-wide feature index matches the leaf scan's baseOffset + getValue - 1.
        func.perActiveVoxel(ijk, layout.voxelOffsets[bi] + int64_t(i), args...);
    }
}

} // namespace _private

/// @brief Run func.perActiveVoxel(ijk, featureIdx, args...) for every active voxel of every
/// grid in the batch, using the batch's VoxelBlockManagers: exactly one thread per active voxel
/// (plus each grid's partial last VBM block), independent of leaf occupancy.
///
/// The callback contract matches BasePerActiveVoxelProcessor::perActiveVoxel:
///     void perActiveVoxel(nanovdb::Coord const &ijk, int64_t featureIdx, Args...) const
/// where featureIdx is the batch-wide linear voxel index (cumVoxelsAt(bi) + in-grid index).
///
/// One kernel launch per batch; the first call on a batch also builds its VBMs. CUDA-only;
/// callers must fall back to the leaf-scan paths for CPU and PrivateUse1 grids.
template <typename Func, typename... Args>
void
forEachActiveVoxelVbmCUDA(const fvdb::GridBatchData &batchHdl, Func func, Args... args) {
    TORCH_CHECK(batchHdl.device().is_cuda(), "Grid batch must be on a CUDA device");
    TORCH_CHECK(batchHdl.device().has_index(), "Grid batch device must have an index");
    c10::cuda::CUDAGuard deviceGuard(batchHdl.device());
    const auto stream = at::cuda::getCurrentCUDAStream(batchHdl.device().index());

    const detail::BatchVbm &vbm = batchHdl.vbm();
    if (vbm.blockCount == 0) {
        return;
    }
    const uint64_t cudaBlocks =
        (vbm.blockCount + _private::kVbmBlocksPerCudaBlock - 1) / _private::kVbmBlocksPerCudaBlock;
    const auto numBlocks =
        uint32_t(std::min<uint64_t>(cudaBlocks, std::numeric_limits<int32_t>::max()));
    _private::forEachActiveVoxelVbmKernel<<<numBlocks, _private::kVbmForEachBlockDim, 0, stream>>>(
        vbm, detail::BatchVbmLayout::of(batchHdl), func, args...);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace fvdb

#endif // FVDB_DETAIL_UTILS_CUDA_FOREACHVBMCUDA_CUH
