// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include "dispatch/dispatch_table.h"
#include "dispatch/torch/dispatch.h"
#include "dispatch/torch/views.h"

#include <fvdb/detail/GridBatchImpl.h>
#include <fvdb/detail/dispatch/ForEachActiveVoxel.cuh>
#include <fvdb/detail/ops/ActiveGridGoords.h>

namespace fvdb {
namespace detail {
namespace ops {

namespace {

// ---------------------------------------------------------------------------
// Dispatch table wiring
// ---------------------------------------------------------------------------
// Only dispatches on device — there are no user-supplied tensors whose
// contiguity could vary.  The output is always freshly allocated (contiguous).

struct active_grid_coords_op {
    template <typename Tag>
    static JaggedTensor
    op(Tag tg, GridBatchImpl const &grid) {
        constexpr auto dev = ::dispatch::tag_get<torch::DeviceType>(tg);

        auto out = torch::empty({static_cast<int64_t>(grid.totalVoxels()), 3},
                                torch::TensorOptions().dtype(torch::kInt32).device(grid.device()));
        auto out_v =
            ::dispatch::tensor_out<dev, torch::kInt32, 2, ::dispatch::contiguity::contiguous>(out);

        ::fvdb::detail::dispatch::forEachActiveVoxel(
            tg,
            grid,
            [=] __hostdev__(Tag /*tg*/,
                            nanovdb::Coord ijk,
                            int64_t idx,
                            GridBatchImpl::Accessor /*acc*/) {
                out_v(idx, 0) = static_cast<int32_t>(ijk[0]);
                out_v(idx, 1) = static_cast<int32_t>(ijk[1]);
                out_v(idx, 2) = static_cast<int32_t>(ijk[2]);
            });

        return grid.jaggedTensor(out);
    }

    using space      = ::dispatch::axes<::dispatch::torch_full_device_axis>;
    using subspaces  = ::dispatch::coverage<space>;
    using dispatcher = ::dispatch::dispatch_table<space, JaggedTensor(GridBatchImpl const &)>;
};

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

JaggedTensor
activeGridCoords(GridBatchImpl const &gridBatch) {
    static auto const table =
        ::dispatch::dispatch_table_from_op<active_grid_coords_op>("activeGridCoords");
    return table.select(::dispatch::dispatch_set{gridBatch.device().type()})(gridBatch);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
