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
#include <fvdb/detail/ops/PointsInGrid.h>

#include <nanovdb/NanoVDB.h>

namespace fvdb {
namespace detail {
namespace ops {

namespace {

// ---------------------------------------------------------------------------
// Dispatch table wiring
// ---------------------------------------------------------------------------
// Axes: device x floating scalar type x contiguity.
// The scalar type axis covers the input points tensor (half, float, double).
// The output is always bool (contiguous).

struct points_in_grid_op {
    template <typename Tag>
    static JaggedTensor
    op(Tag tg, GridBatchImpl const &grid, JaggedTensor points) {
        constexpr auto dev    = ::dispatch::tag_get<torch::DeviceType>(tg);
        constexpr auto stype  = ::dispatch::tag_get<torch::ScalarType>(tg);
        constexpr auto contig = ::dispatch::tag_get<::dispatch::contiguity>(tg);

        using scalar_t = ::dispatch::torch_scalar_cpp_type_t<stype>;

        int64_t const n   = points.element_count();
        auto const device = points.device();

        auto out   = torch::empty({n}, torch::TensorOptions().dtype(torch::kBool).device(device));
        auto pts_v = ::fvdb::detail::dispatch::jagged_in<dev, stype, 2, contig>(points);
        auto out_v =
            ::dispatch::tensor_out<dev, torch::kBool, 1, ::dispatch::contiguity::contiguous>(out);
        auto grid_acc = ::fvdb::detail::dispatch::make_grid_accessor(tg, grid);

        ::dispatch::for_each(tg, n, [=] __hostdev__(Tag /*tg*/, int64_t eidx) {
            auto const bidx = pts_v.batchIdx(eidx);

            auto const *grid_ptr  = grid_acc.grid(bidx);
            auto const acc        = grid_ptr->getAccessor();
            auto const &transform = grid_acc.primalTransform(bidx);

            nanovdb::Coord const vox = transform
                                           .apply(static_cast<scalar_t>(pts_v(eidx, 0)),
                                                  static_cast<scalar_t>(pts_v(eidx, 1)),
                                                  static_cast<scalar_t>(pts_v(eidx, 2)))
                                           .round();

            out_v(eidx) = acc.isActive(vox);
        });

        return points.jagged_like(out);
    }

    // float32, float64, float16 — no bfloat16 (VoxelCoordTransform doesn't support it).
    // Matches the old AT_DISPATCH_V2(AT_FLOATING_TYPES, c10::kHalf) coverage.
    using stype_axis = ::dispatch::axis<torch::kFloat16, torch::kFloat32, torch::kFloat64>;
    using space      = ::dispatch::
        axes<::dispatch::torch_full_device_axis, stype_axis, ::dispatch::full_contiguity_axis>;
    using subspaces = ::dispatch::coverage<space>;
    using dispatcher =
        ::dispatch::dispatch_table<space, JaggedTensor(GridBatchImpl const &, JaggedTensor)>;
};

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

JaggedTensor
pointsInGrid(GridBatchImpl const &grid, JaggedTensor points) {
    grid.checkNonEmptyGrid();
    grid.checkDevice(points);

    namespace dc = ::fvdb::detail::dispatch;
    dc::check_rank(points.jdata(), 2, "points");
    dc::check_dim_size(points.jdata(), 1, 3, "points");
    dc::check_non_empty(points.jdata(), "points");

    static auto const table = ::dispatch::dispatch_table_from_op<points_in_grid_op>("pointsInGrid");
    return table.select(::dispatch::dispatch_set{points.device().type(),
                                                 points.scalar_type(),
                                                 ::dispatch::torch_get_contiguity(points.jdata())})(
        grid, points);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
