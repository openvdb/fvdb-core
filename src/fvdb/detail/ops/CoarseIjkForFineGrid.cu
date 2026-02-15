// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include "dispatch/dispatch_table.h"
#include "dispatch/torch/dispatch.h"
#include "dispatch/torch/views.h"

#include <fvdb/detail/GridBatchImpl.h>
#include <fvdb/detail/dispatch/ForEachActiveVoxel.cuh>
#include <fvdb/detail/ops/CoarseIjkForFineGrid.h>

namespace fvdb {
namespace detail {
namespace ops {

namespace {

// ---------------------------------------------------------------------------
// Dispatch table wiring
// ---------------------------------------------------------------------------
// Only dispatches on device -- output is always int32, freshly allocated
// (contiguous).  No user-supplied tensors.
//
// Iteration pattern: forEachActiveVoxel (one thread per leaf-voxel slot).
//
// The old code used forEachVoxelCUDA with numChannels=1, which is the same
// thread decomposition.  The callback body (integer floor-division of ijk
// by the coarsening factor) is unchanged.  The old code lacked a CPU path
// entirely; forEachActiveVoxel provides one for free via dispatch::for_each.

struct coarse_ijk_for_fine_grid_op {
    template <typename Tag>
    static JaggedTensor
    op(Tag tg, GridBatchImpl const &fineGrid, nanovdb::Coord coarseningFactor) {
        constexpr auto dev = ::dispatch::tag_get<torch::DeviceType>(tg);

        auto out =
            torch::empty({static_cast<int64_t>(fineGrid.totalVoxels()), 3},
                         torch::TensorOptions().dtype(torch::kInt32).device(fineGrid.device()));
        auto out_v =
            ::dispatch::tensor_out<dev, torch::kInt32, 2, ::dispatch::contiguity::contiguous>(out);

        ::fvdb::detail::dispatch::forEachActiveVoxel(
            tg,
            fineGrid,
            [=] __hostdev__(Tag /*tg*/,
                            JIdxType /*batchIdx*/,
                            nanovdb::Coord fineIjk,
                            int64_t voxelIndex,
                            GridBatchImpl::Accessor /*acc*/) {
                nanovdb::Coord const coarseIjk =
                    (fineIjk.asVec3d() / coarseningFactor.asVec3d()).floor();
                out_v(voxelIndex, 0) = coarseIjk[0];
                out_v(voxelIndex, 1) = coarseIjk[1];
                out_v(voxelIndex, 2) = coarseIjk[2];
            });

        return fineGrid.jaggedTensor(out);
    }

    using space     = ::dispatch::axes<::dispatch::torch_full_device_axis>;
    using subspaces = ::dispatch::coverage<space>;
    using dispatcher =
        ::dispatch::dispatch_table<space, JaggedTensor(GridBatchImpl const &, nanovdb::Coord)>;
};

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

JaggedTensor
coarseIjkForFineGrid(GridBatchImpl const &fineGrid, nanovdb::Coord coarseningFactor) {
    c10::DeviceGuard guard(fineGrid.device());
    fineGrid.checkNonEmptyGrid();

    static auto const table =
        ::dispatch::dispatch_table_from_op<coarse_ijk_for_fine_grid_op>("coarseIjkForFineGrid");
    return table.select(::dispatch::dispatch_set{fineGrid.device().type()})(fineGrid,
                                                                            coarseningFactor);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
