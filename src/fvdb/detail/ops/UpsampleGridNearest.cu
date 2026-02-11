// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include "dispatch/dispatch_table.h"
#include "dispatch/torch/dispatch.h"
#include "dispatch/torch/views.h"

#include <fvdb/detail/GridBatchImpl.h>
#include <fvdb/detail/dispatch/AtomicAdd.cuh>
#include <fvdb/detail/dispatch/ForEachActiveVoxel.cuh>
#include <fvdb/detail/dispatch/TensorChecks.h>
#include <fvdb/detail/ops/UpsampleGridNearest.h>
#include <fvdb/detail/utils/Utils.h>

namespace fvdb {
namespace detail {
namespace ops {

namespace {

// Scalar type axis: float16, bfloat16, float32, float64
using upsample_stype_axis = ::dispatch::axis<torch::kFloat16, torch::kBFloat16,
                                             torch::kFloat32, torch::kFloat64>;

// ---------------------------------------------------------------------------
// Forward: copy coarse features to fine voxels
// ---------------------------------------------------------------------------
// The entry point coalesces coarseData to [N, C] before dispatching.
// The op receives the coalesced 2D tensor directly — no redundant reshape.
//
// Contiguity: dispatched on the coalesced coarseData (the only user-supplied
// tensor we build a view over). The output is freshly allocated, so its view
// is hardcoded contiguous.

struct upsample_grid_nearest_op {
    template <typename Tag>
    static torch::Tensor
    op(Tag tg,
       GridBatchImpl const &coarseGrid,
       GridBatchImpl const &fineGrid,
       torch::Tensor coarseDataCoalesced,    // already [N, C] from entry point
       torch::Tensor originalCoarseData,     // original shape for spliceShape
       nanovdb::Coord upsamplingFactor) {
        constexpr auto dev    = ::dispatch::tag_get<torch::DeviceType>(tg);
        constexpr auto stype  = ::dispatch::tag_get<torch::ScalarType>(tg);
        constexpr auto contig = ::dispatch::tag_get<::dispatch::contiguity>(tg);

        using scalar_t = ::dispatch::torch_scalar_cpp_type_t<stype>;

        int64_t const numFineVoxels = fineGrid.totalVoxels();
        int64_t const numChannels   = coarseDataCoalesced.size(1);

        torch::Tensor outFineData = torch::zeros(
            spliceShape({numFineVoxels}, originalCoarseData),
            torch::TensorOptions().dtype(originalCoarseData.dtype())
                                  .device(originalCoarseData.device()));
        torch::Tensor outFineCoalesced = featureCoalescedView(outFineData);

        // coarse_v: dispatched contiguity — coalesced input may be strided
        auto coarse_v = ::dispatch::tensor_in<dev, stype, 2, contig>(coarseDataCoalesced);
        // fine_out: always contiguous — freshly allocated
        auto fine_out = ::dispatch::tensor_out<dev, stype, 2,
                                               ::dispatch::contiguity::contiguous>(outFineCoalesced);
        auto coarse_acc = ::fvdb::detail::dispatch::make_grid_accessor(tg, coarseGrid);

        ::fvdb::detail::dispatch::forEachActiveVoxel(
            tg, fineGrid,
            [=] __hostdev__(Tag /*tg*/, JIdxType batchIdx, nanovdb::Coord fineIjk,
                            int64_t fineIndex, GridBatchImpl::Accessor /*fineAcc*/) {
                nanovdb::Coord const coarseIjk =
                    nanovdb::math::Vec3<scalar_t>(
                        static_cast<scalar_t>(fineIjk[0]) / static_cast<scalar_t>(upsamplingFactor[0]),
                        static_cast<scalar_t>(fineIjk[1]) / static_cast<scalar_t>(upsamplingFactor[1]),
                        static_cast<scalar_t>(fineIjk[2]) / static_cast<scalar_t>(upsamplingFactor[2]))
                        .floor();

                auto const coarseGridAcc = coarse_acc.grid(batchIdx)->getAccessor();
                if (coarseGridAcc.isActive(coarseIjk)) {
                    int64_t const ci =
                        static_cast<int64_t>(coarseGridAcc.getValue(coarseIjk)) - 1
                        + coarse_acc.voxelOffset(batchIdx);
                    for (int64_t c = 0; c < numChannels; ++c) {
                        fine_out(fineIndex, c) = coarse_v(ci, c);
                    }
                }
            });

        return outFineData;
    }

    using space     = ::dispatch::axes<::dispatch::torch_full_device_axis,
                                       upsample_stype_axis,
                                       ::dispatch::full_contiguity_axis>;
    using subspaces = ::dispatch::coverage<space>;
    using dispatcher = ::dispatch::dispatch_table<
        space,
        torch::Tensor(GridBatchImpl const &, GridBatchImpl const &,
                      torch::Tensor, torch::Tensor, nanovdb::Coord)>;
};

// ---------------------------------------------------------------------------
// Backward: accumulate fine gradients into coarse (with atomics)
// ---------------------------------------------------------------------------
// The entry point coalesces gradOut and coarseData before dispatching.
//
// Contiguity: dispatched on the coalesced gradOut (the only user-supplied
// tensor we read via a view). The output accumulator is freshly allocated,
// so its view is hardcoded contiguous.

struct upsample_grid_nearest_backward_op {
    template <typename Tag>
    static torch::Tensor
    op(Tag tg,
       GridBatchImpl const &fineGrid,
       GridBatchImpl const &coarseGrid,
       torch::Tensor gradOutCoalesced,       // already [M, C] from entry point
       torch::Tensor coarseDataCoalesced,    // already [N, C] from entry point
       torch::Tensor originalGradOut,        // original shape for spliceShape
       nanovdb::Coord upsamplingFactor) {
        constexpr auto dev    = ::dispatch::tag_get<torch::DeviceType>(tg);
        constexpr auto stype  = ::dispatch::tag_get<torch::ScalarType>(tg);
        constexpr auto contig = ::dispatch::tag_get<::dispatch::contiguity>(tg);

        using scalar_t = ::dispatch::torch_scalar_cpp_type_t<stype>;

        torch::Tensor outGradInCoalesced = torch::zeros_like(coarseDataCoalesced);
        int64_t const numChannels        = coarseDataCoalesced.size(1);

        // grad_v: dispatched contiguity — coalesced grad input may be strided
        auto grad_v   = ::dispatch::tensor_in<dev, stype, 2, contig>(gradOutCoalesced);
        // out_grad: always contiguous — freshly allocated via zeros_like
        auto out_grad = ::dispatch::tensor_out<dev, stype, 2,
                                               ::dispatch::contiguity::contiguous>(outGradInCoalesced);
        auto coarse_acc = ::fvdb::detail::dispatch::make_grid_accessor(tg, coarseGrid);

        ::fvdb::detail::dispatch::forEachActiveVoxel(
            tg, fineGrid,
            [=] __hostdev__(Tag tg, JIdxType batchIdx, nanovdb::Coord fineIjk,
                            int64_t fineIndex, GridBatchImpl::Accessor /*fineAcc*/) {
                nanovdb::Coord const coarseIjk =
                    nanovdb::math::Vec3<scalar_t>(
                        static_cast<scalar_t>(fineIjk[0]) / static_cast<scalar_t>(upsamplingFactor[0]),
                        static_cast<scalar_t>(fineIjk[1]) / static_cast<scalar_t>(upsamplingFactor[1]),
                        static_cast<scalar_t>(fineIjk[2]) / static_cast<scalar_t>(upsamplingFactor[2]))
                        .floor();

                auto const coarseGridAcc = coarse_acc.grid(batchIdx)->getAccessor();
                if (coarseGridAcc.isActive(coarseIjk)) {
                    int64_t const ci =
                        static_cast<int64_t>(coarseGridAcc.getValue(coarseIjk)) - 1
                        + coarse_acc.voxelOffset(batchIdx);
                    for (int64_t c = 0; c < numChannels; ++c) {
                        ::fvdb::detail::dispatch::atomic_add(
                            tg, &out_grad(ci, c), grad_v(fineIndex, c));
                    }
                }
            });

        return outGradInCoalesced.reshape(
            spliceShape({coarseDataCoalesced.size(0)}, originalGradOut));
    }

    using space     = ::dispatch::axes<::dispatch::torch_full_device_axis,
                                       upsample_stype_axis,
                                       ::dispatch::full_contiguity_axis>;
    using subspaces = ::dispatch::coverage<space>;
    using dispatcher = ::dispatch::dispatch_table<
        space,
        torch::Tensor(GridBatchImpl const &, GridBatchImpl const &,
                      torch::Tensor, torch::Tensor, torch::Tensor, nanovdb::Coord)>;
};

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------
// featureCoalescedView is called once here. The coalesced tensor is passed
// into the dispatch table so the op() never re-coalesces.  Contiguity is
// determined from the coalesced tensor — this is the tensor we build a
// tensor_in view over, so its contiguity is what matters for dispatch.

torch::Tensor
upsampleGridNearest(GridBatchImpl const &coarseGrid,
                    GridBatchImpl const &fineGrid,
                    torch::Tensor coarseData,
                    nanovdb::Coord upsamplingFactor) {
    coarseGrid.checkNonEmptyGrid();
    fineGrid.checkNonEmptyGrid();
    for (int i = 0; i < 3; ++i) {
        TORCH_CHECK(upsamplingFactor[i] > 0, "upsampling_factor must be greater than 0");
    }
    TORCH_CHECK(coarseData.dim() > 1,
                "coarse_data must have more than one dimension. i.e. have shape (num_voxels, *)");
    TORCH_CHECK(coarseData.size(0) == static_cast<int64_t>(coarseGrid.totalVoxels()),
                "coarse_data must have the same number of voxels as coarse_grid");

    torch::Tensor coarseDataCoalesced = featureCoalescedView(coarseData);

    static auto const table =
        ::dispatch::dispatch_table_from_op<upsample_grid_nearest_op>("upsampleGridNearest");
    return table.select(
        ::dispatch::dispatch_set{coarseData.device().type(),
                                 coarseData.scalar_type(),
                                 ::dispatch::torch_get_contiguity(coarseDataCoalesced)})(
        coarseGrid, fineGrid, coarseDataCoalesced, coarseData, upsamplingFactor);
}

torch::Tensor
upsampleGridNearestBackward(GridBatchImpl const &fineGrid,
                            GridBatchImpl const &coarseGrid,
                            torch::Tensor gradOut,
                            torch::Tensor coarseData,
                            nanovdb::Coord upsamplingFactor) {
    for (int i = 0; i < 3; ++i) {
        TORCH_CHECK(upsamplingFactor[i] > 0, "upsampling_factor must be greater than 0");
    }

    torch::Tensor gradOutCoalesced    = featureCoalescedView(gradOut);
    torch::Tensor coarseDataCoalesced = featureCoalescedView(coarseData);

    static auto const table =
        ::dispatch::dispatch_table_from_op<upsample_grid_nearest_backward_op>(
            "upsampleGridNearestBackward");
    return table.select(
        ::dispatch::dispatch_set{gradOut.device().type(),
                                 gradOut.scalar_type(),
                                 ::dispatch::torch_get_contiguity(gradOutCoalesced)})(
        fineGrid, coarseGrid, gradOutCoalesced, coarseDataCoalesced, gradOut, upsamplingFactor);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
