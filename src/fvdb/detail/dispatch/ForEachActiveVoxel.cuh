// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// ForEachActiveVoxel.cuh — Device-unified active-voxel iteration.
//
// Iterates over every active voxel in a GridBatchImpl, calling a callback
// for each one.  Device dispatch (CPU thread pool / CUDA grid-stride kernel /
// PrivateUse1 multi-GPU) is handled internally via dispatch::for_each.
//
// Callback signature:
//
//     void(Tag tg, nanovdb::Coord ijk, int64_t voxel_index,
//          GridBatchImpl::Accessor acc)
//
// The Tag is forwarded so that callbacks can use concept-constrained
// overloads for device-specific specialization when needed.
//
// The isActive check is performed inside the tool — the callback is only
// invoked for truly active voxels, matching the contract of the former
// BasePerActiveVoxelProcessor.
//
#ifndef FVDB_DETAIL_DISPATCH_FOREACHACTIVEVOXEL_CUH
#define FVDB_DETAIL_DISPATCH_FOREACHACTIVEVOXEL_CUH

#include "dispatch/torch/for_each.h"
#include "dispatch/with_value.h"

#include <fvdb/detail/GridBatchImpl.h>
#include <fvdb/detail/dispatch/GridAccessor.h>

#include <nanovdb/NanoVDB.h>

#include <cstdint>

namespace fvdb {
namespace detail {
namespace dispatch {

/// @brief Iterate over every active voxel in a grid batch.
///
/// @tparam Tag   A dispatch::tag carrying at least a torch::DeviceType.
/// @tparam Func  Callable with signature
///               void(Tag, nanovdb::Coord, int64_t, GridBatchImpl::Accessor).
///
/// @param tg    Tag instance (forwarded to both dispatch::for_each and the callback).
/// @param grid  The grid batch to iterate over.
/// @param func  Callback invoked once per active voxel.
template <typename Tag, typename Func>
    requires ::dispatch::with_type<Tag, torch::DeviceType>
void
forEachActiveVoxel(Tag tg, GridBatchImpl const &grid, Func func) {
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
            func(tg, ijk, voxelIndex, acc);
        }
    });
}

} // namespace dispatch
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_DISPATCH_FOREACHACTIVEVOXEL_CUH
