// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include "dispatch/dispatch_table.h"
#include "dispatch/torch/dispatch.h"
#include "dispatch/torch/for_each.h"
#include "dispatch/torch/views.h"

#include <fvdb/detail/dispatch/TensorChecks.h>
#include <fvdb/detail/ops/MortonHilbertFromIjk.h>
#include <fvdb/detail/utils/HilbertCode.h>
#include <fvdb/detail/utils/MortonCode.h>

namespace fvdb {
namespace detail {
namespace ops {

namespace {

// ---------------------------------------------------------------------------
// Dispatch table wiring
// ---------------------------------------------------------------------------

struct morton_from_ijk_op {
    template <typename Tag>
    static torch::Tensor
    op(Tag tg, torch::Tensor ijk) {
        constexpr auto dev    = ::dispatch::tag_get<torch::DeviceType>(tg);
        constexpr auto contig = ::dispatch::tag_get<::dispatch::contiguity>(tg);

        int64_t const n   = ijk.size(0);
        auto const device = ijk.device();

        auto out   = torch::empty({n}, torch::TensorOptions().dtype(torch::kInt64).device(device));
        auto ijk_v = ::dispatch::tensor_in<dev, torch::kInt32, 2, contig>(ijk);
        auto out_v =
            ::dispatch::tensor_out<dev, torch::kInt64, 1, ::dispatch::contiguity::contiguous>(out);

        ::dispatch::for_each(tg, n, [=] __hostdev__(Tag, int64_t idx) {
            auto const i = static_cast<uint32_t>(ijk_v(idx, 0));
            auto const j = static_cast<uint32_t>(ijk_v(idx, 1));
            auto const k = static_cast<uint32_t>(ijk_v(idx, 2));
            out_v(idx)   = static_cast<int64_t>(utils::morton(i, j, k));
        });

        return out;
    }

    using space =
        ::dispatch::axes<::dispatch::torch_full_device_axis, ::dispatch::full_contiguity_axis>;
    using subspaces  = ::dispatch::coverage<space>;
    using dispatcher = ::dispatch::dispatch_table<space, torch::Tensor(torch::Tensor)>;
};

struct hilbert_from_ijk_op {
    template <typename Tag>
    static torch::Tensor
    op(Tag tg, torch::Tensor ijk) {
        constexpr auto dev    = ::dispatch::tag_get<torch::DeviceType>(tg);
        constexpr auto contig = ::dispatch::tag_get<::dispatch::contiguity>(tg);

        int64_t const n   = ijk.size(0);
        auto const device = ijk.device();

        auto out   = torch::empty({n}, torch::TensorOptions().dtype(torch::kInt64).device(device));
        auto ijk_v = ::dispatch::tensor_in<dev, torch::kInt32, 2, contig>(ijk);
        auto out_v =
            ::dispatch::tensor_out<dev, torch::kInt64, 1, ::dispatch::contiguity::contiguous>(out);

        ::dispatch::for_each(tg, n, [=] __hostdev__(Tag, int64_t idx) {
            auto const i = static_cast<uint32_t>(ijk_v(idx, 0));
            auto const j = static_cast<uint32_t>(ijk_v(idx, 1));
            auto const k = static_cast<uint32_t>(ijk_v(idx, 2));
            out_v(idx)   = static_cast<int64_t>(utils::hilbert(i, j, k));
        });

        return out;
    }

    using space =
        ::dispatch::axes<::dispatch::torch_full_device_axis, ::dispatch::full_contiguity_axis>;
    using subspaces  = ::dispatch::coverage<space>;
    using dispatcher = ::dispatch::dispatch_table<space, torch::Tensor(torch::Tensor)>;
};

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Common precondition check for ijk coordinate tensors: [N, 3] int32.
static void
check_ijk(torch::Tensor const &ijk) {
    namespace dc = ::fvdb::detail::dispatch;
    dc::check_rank(ijk, 2, "ijk");
    dc::check_dim_size(ijk, 1, 3, "ijk");
    dc::check_dtype<torch::kInt32>(ijk, "ijk");
}

torch::Tensor
mortonFromIjk(torch::Tensor ijk) {
    c10::DeviceGuard guard(ijk.device());
    check_ijk(ijk);
    static auto const table =
        ::dispatch::dispatch_table_from_op<morton_from_ijk_op>("mortonFromIjk");
    return table.select(
        ::dispatch::dispatch_set{ijk.device().type(), ::dispatch::torch_get_contiguity(ijk)})(ijk);
}

torch::Tensor
hilbertFromIjk(torch::Tensor ijk) {
    c10::DeviceGuard guard(ijk.device());
    check_ijk(ijk);
    static auto const table =
        ::dispatch::dispatch_table_from_op<hilbert_from_ijk_op>("hilbertFromIjk");
    return table.select(
        ::dispatch::dispatch_set{ijk.device().type(), ::dispatch::torch_get_contiguity(ijk)})(ijk);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
