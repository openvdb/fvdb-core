// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// ForEachActiveVoxel.cuh — Device-unified active-voxel iteration.
//
// Iterates over every active voxel in a GridBatchData, calling a callback
// for each one.  The leaf scan dispatches by device (CPU thread pool / CUDA
// grid-stride kernel / PrivateUse1 multi-GPU) via dispatch::for_each; the VBM
// path launches its own CUDA kernel (launchBatchVbmKernel).
//
// Two iteration strategies (ActiveVoxelIteration):
//   LeafScan  One thread per leaf slot (totalLeaves * 512), skipping inactive
//             slots. Every device.
//   Vbm       On CUDA, one thread per active voxel through the batch's cached
//             VoxelBlockManagers (see detail/VbmCache.h), independent of leaf
//             occupancy. Other devices fall back to LeafScan.
// LeafScan is the default; callers opt into Vbm per call site after measuring
// it. Before opting in, check the callback for contended global atomics: the
// VBM packs every lane of a warp with an active voxel, so an atomic on a few
// shared addresses that the leaf scan's inactive lanes thinned out collides
// 32 ways (aggregate it, e.g. dispatch::atomic_fetch_increment_i32).
//
// Callback signature:
//
//     void(Tag tg, JIdxType batchIdx, nanovdb::Coord ijk,
//          int64_t voxel_index, GridBatchData::Accessor acc)
//
// The Tag is forwarded so that callbacks can use concept-constrained
// overloads for device-specific specialization when needed.
//
// The isActive check is performed inside the tool — the callback is only
// invoked for truly active voxels, matching the contract of
// BasePerActiveVoxelProcessor.
//
#ifndef FVDB_DETAIL_DISPATCH_FOREACHACTIVEVOXEL_CUH
#define FVDB_DETAIL_DISPATCH_FOREACHACTIVEVOXEL_CUH

#include "dispatch/torch/for_each.h"
#include "dispatch/with_value.h"

#include <fvdb/GridBatchData.h>
#include <fvdb/detail/dispatch/GridAccessor.h>
#include <fvdb/detail/utils/cuda/ForEachVbmCUDA.cuh>

#include <nanovdb/NanoVDB.h>

#include <concepts>
#include <cstdint>

namespace fvdb {
namespace detail {
namespace dispatch {

/// @brief How forEachActiveVoxel walks the active voxels. See the file comment.
enum class ActiveVoxelIteration { LeafScan, Vbm };

namespace _private {

template <typename Tag, typename Func>
    requires ::dispatch::with_type<Tag, torch::DeviceType> &&
             std::invocable<Func &, Tag, JIdxType, nanovdb::Coord, int64_t, GridBatchData::Accessor>
__global__ void __launch_bounds__(::fvdb::_private::kVbmForEachBlockDim)
forEachActiveVoxelVbmKernel(BatchVbm vbm, BatchVbmLayout layout, Tag tg, Func func) {
    ::fvdb::_private::forEachBatchVbmVoxel(
        vbm, layout, [&](uint32_t gridIdx, nanovdb::Coord const &ijk, int64_t voxelIndex) {
            func(tg, JIdxType(gridIdx), ijk, voxelIndex, layout.acc);
        });
}

} // namespace _private

/// @brief Iterate over every active voxel in a grid batch.
///
/// @tparam Tag   A dispatch::tag carrying at least a torch::DeviceType.
/// @tparam Func  Callable with signature
///               void(Tag, JIdxType batchIdx, nanovdb::Coord, int64_t, GridBatchData::Accessor).
///
/// @param tg         Tag instance, forwarded to the callback (and to dispatch::for_each on the
///                   leaf-scan path).
/// @param grid       The grid batch to iterate over.
/// @param func       Callback invoked once per active voxel.
/// @param iteration  LeafScan (default) or Vbm. Vbm applies on CUDA only and builds the batch's
///                   VoxelBlockManagers on first use. Both call func once per active voxel with
///                   the same arguments; neither promises an order across threads.
template <typename Tag, typename Func>
    requires ::dispatch::with_type<Tag, torch::DeviceType>
void
forEachActiveVoxel(Tag tg,
                   GridBatchData const &grid,
                   Func func,
                   ActiveVoxelIteration iteration = ActiveVoxelIteration::LeafScan) {
    if constexpr (::dispatch::tag_get<torch::DeviceType>(Tag{}) == torch::kCUDA) {
        if (iteration == ActiveVoxelIteration::Vbm) {
            ::fvdb::launchBatchVbmKernel(
                grid, &_private::forEachActiveVoxelVbmKernel<Tag, Func>, tg, func);
            return;
        }
    }

    constexpr int64_t VOXELS_PER_LEAF =
        static_cast<int64_t>(nanovdb::OnIndexTree::LeafNodeType::NUM_VALUES);

    auto acc            = make_grid_accessor(tg, grid);
    int64_t const total = grid.totalLeaves() * VOXELS_PER_LEAF;

    // dispatch::for_each handles CPU / CUDA / PrivateUse1 internally.
    // The lambda parameter Tag is left unnamed to avoid shadowing the
    // captured `tg`; both are equivalent for stateless tag types.
    ::dispatch::for_each(tg, total, [=] __hostdev__(Tag /*tg*/, int64_t flat_idx) {
        int64_t const voxelIdx   = flat_idx % VOXELS_PER_LEAF;
        int64_t const cumLeafIdx = flat_idx / VOXELS_PER_LEAF;
        auto const batchIdx      = acc.leafBatchIndex(cumLeafIdx);
        int64_t const leafIdx    = cumLeafIdx - acc.leafOffset(batchIdx);

        auto const *grid_ptr = acc.grid(batchIdx);
        auto const &leaf     = grid_ptr->tree().template getFirstNode<0>()[leafIdx];

        if (leaf.isActive(voxelIdx)) {
            auto const ijk           = leaf.offsetToGlobalCoord(voxelIdx);
            int64_t const voxelIndex = acc.voxelOffset(batchIdx) + leaf.getValue(voxelIdx) - 1;
            func(tg, batchIdx, ijk, voxelIndex, acc);
        }
    });
}

} // namespace dispatch
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_DISPATCH_FOREACHACTIVEVOXEL_CUH
