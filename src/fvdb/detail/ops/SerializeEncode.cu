// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include "dispatch/dispatch_table.h"
#include "dispatch/torch/dispatch.h"
#include "dispatch/torch/views.h"

#include <fvdb/detail/GridBatchImpl.h>
#include <fvdb/detail/dispatch/ForEachActiveVoxel.cuh>
#include <fvdb/detail/ops/SerializeEncode.h>
#include <fvdb/detail/utils/HilbertCode.h>
#include <fvdb/detail/utils/MortonCode.h>

namespace fvdb {
namespace detail {
namespace ops {

namespace {

// ---------------------------------------------------------------------------
// Dispatch table wiring
// ---------------------------------------------------------------------------
// Device-only dispatch (no scalar type or contiguity axes — output is always
// freshly allocated int64, and there are no user-supplied tensors).

struct serialize_encode_op {
    template <typename Tag>
    static JaggedTensor
    op(Tag tg, GridBatchImpl const &grid, SpaceFillingCurveType order_type, nanovdb::Coord offset) {
        constexpr auto dev = ::dispatch::tag_get<torch::DeviceType>(tg);

        auto out = torch::empty({static_cast<int64_t>(grid.totalVoxels())},
                                torch::TensorOptions().dtype(torch::kInt64).device(grid.device()));
        auto out_v =
            ::dispatch::tensor_out<dev, torch::kInt64, 1, ::dispatch::contiguity::contiguous>(out);

        ::fvdb::detail::dispatch::forEachActiveVoxel(
            tg,
            grid,
            [=] __hostdev__(Tag /*tg*/,
                            JIdxType /*batchIdx*/,
                            nanovdb::Coord ijk,
                            int64_t idx,
                            GridBatchImpl::Accessor /*acc*/) {
                auto const i = static_cast<uint32_t>(ijk[0] + offset[0]);
                auto const j = static_cast<uint32_t>(ijk[1] + offset[1]);
                auto const k = static_cast<uint32_t>(ijk[2] + offset[2]);

                uint64_t code;
                switch (order_type) {
                case SpaceFillingCurveType::ZOrder: code = utils::morton(i, j, k); break;
                case SpaceFillingCurveType::ZOrderTransposed: code = utils::morton(k, j, i); break;
                case SpaceFillingCurveType::Hilbert: code = utils::hilbert(i, j, k); break;
                case SpaceFillingCurveType::HilbertTransposed:
                    code = utils::hilbert(k, j, i);
                    break;
                default: code = 0; break;
                }

                out_v(idx) = static_cast<int64_t>(code);
            });

        return grid.jaggedTensor(out);
    }

    using space      = ::dispatch::axes<::dispatch::torch_full_device_axis>;
    using subspaces  = ::dispatch::coverage<space>;
    using dispatcher = ::dispatch::dispatch_table<
        space,
        JaggedTensor(GridBatchImpl const &, SpaceFillingCurveType, nanovdb::Coord)>;
};

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

JaggedTensor
serializeEncode(GridBatchImpl const &grid,
                SpaceFillingCurveType order_type,
                nanovdb::Coord const &offset) {
    c10::DeviceGuard guard(grid.device());

    static auto const table =
        ::dispatch::dispatch_table_from_op<serialize_encode_op>("serializeEncode");
    return table.select(::dispatch::dispatch_set{grid.device().type()})(grid, order_type, offset);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
