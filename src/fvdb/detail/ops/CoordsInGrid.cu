// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include "dispatch/dispatch_table.h"
#include "dispatch/torch/dispatch.h"
#include "dispatch/torch/for_each.h"
#include "dispatch/torch/views.h"

#include <fvdb/detail/GridBatchImpl.h>
#include <fvdb/detail/dispatch/GridAccessor.h>
#include <fvdb/detail/dispatch/JaggedView.h>
#include <fvdb/detail/dispatch/TensorChecks.h>
#include <fvdb/detail/ops/CoordsInGrid.h>

#include <nanovdb/NanoVDB.h>

namespace fvdb {
namespace detail {
namespace ops {

namespace {

// ---------------------------------------------------------------------------
// Dispatch table wiring
// ---------------------------------------------------------------------------
// Axes: device x integral scalar type x contiguity.
// The scalar type axis covers the input ijk tensor (int8..int64).
// The output is always bool (contiguous).

struct coords_in_grid_op {
    template <typename Tag>
    static JaggedTensor
    op(Tag tg, GridBatchImpl const &grid, JaggedTensor ijk) {
        constexpr auto dev    = ::dispatch::tag_get<torch::DeviceType>(tg);
        constexpr auto stype  = ::dispatch::tag_get<torch::ScalarType>(tg);
        constexpr auto contig = ::dispatch::tag_get<::dispatch::contiguity>(tg);

        int64_t const n   = ijk.element_count();
        auto const device = ijk.device();

        auto out   = torch::empty({n}, torch::TensorOptions().dtype(torch::kBool).device(device));
        auto ijk_v = ::fvdb::detail::dispatch::jagged_in<dev, stype, 2, contig>(ijk);
        auto out_v =
            ::dispatch::tensor_out<dev, torch::kBool, 1, ::dispatch::contiguity::contiguous>(out);
        auto grid_acc = ::fvdb::detail::dispatch::make_grid_accessor(tg, grid);

        ::dispatch::for_each(tg, n, [=] __hostdev__(Tag /*tg*/, int64_t eidx) {
            auto const bidx = ijk_v.batchIdx(eidx);

            auto const *grid_ptr = grid_acc.grid(bidx);
            auto const acc       = grid_ptr->getAccessor();

            nanovdb::Coord const vox(static_cast<int32_t>(ijk_v(eidx, 0)),
                                     static_cast<int32_t>(ijk_v(eidx, 1)),
                                     static_cast<int32_t>(ijk_v(eidx, 2)));

            out_v(eidx) = acc.isActive(vox);
        });

        return ijk.jagged_like(out);
    }

    using space     = ::dispatch::axes<::dispatch::torch_full_device_axis,
                                       ::dispatch::torch_full_signed_int_stype_axis,
                                       ::dispatch::full_contiguity_axis>;
    using subspaces = ::dispatch::coverage<space>;
    using dispatcher =
        ::dispatch::dispatch_table<space, JaggedTensor(GridBatchImpl const &, JaggedTensor)>;
};

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

JaggedTensor
coordsInGrid(GridBatchImpl const &grid, JaggedTensor ijk) {
    c10::DeviceGuard guard(grid.device());
    grid.checkNonEmptyGrid();
    grid.checkDevice(ijk);

    TORCH_CHECK_VALUE(
        ijk.ldim() == 1, "Expected ijk to have 1 list dimension, but got ", ijk.ldim());
    namespace dc = ::fvdb::detail::dispatch;
    dc::check_rank(ijk.jdata(), 2, "ijk");
    dc::check_dim_size(ijk.jdata(), 1, 3, "ijk");
    dc::check_non_empty(ijk.jdata(), "ijk");

    static auto const table = ::dispatch::dispatch_table_from_op<coords_in_grid_op>("coordsInGrid");
    return table.select(::dispatch::dispatch_set{
        ijk.device().type(), ijk.scalar_type(), ::dispatch::torch_get_contiguity(ijk.jdata())})(
        grid, ijk);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
