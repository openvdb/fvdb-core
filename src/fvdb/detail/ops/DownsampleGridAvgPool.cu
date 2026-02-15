// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include "dispatch/dispatch_table.h"
#include "dispatch/torch/dispatch.h"
#include "dispatch/torch/views.h"

#include <fvdb/detail/GridBatchImpl.h>
#include <fvdb/detail/dispatch/ForEachActiveVoxel.cuh>
#include <fvdb/detail/dispatch/TensorChecks.h>
#include <fvdb/detail/ops/DownsampleGridAvgPool.h>
#include <fvdb/detail/utils/Utils.h>

#include <ATen/AccumulateType.h>

namespace fvdb {
namespace detail {
namespace ops {

namespace {

// Scalar type axis: float16, bfloat16, float32, float64
using downsample_stype_axis =
    ::dispatch::axis<torch::kFloat16, torch::kBFloat16, torch::kFloat32, torch::kFloat64>;

// ---------------------------------------------------------------------------
// Forward: average-pool fine features into coarse voxels
// ---------------------------------------------------------------------------
// The entry point coalesces fineData to [N, C] before dispatching.
// Contiguity: dispatched on the coalesced fineData (the only user-supplied
// tensor we build a view over).  The output is freshly allocated, so its view
// is hardcoded contiguous.
//
// Iteration pattern: forEachActiveVoxel over the coarse grid, with a
// sequential channel loop inside the callback.
//
// The old code used forEachVoxelCUDA with numChannels = C, which launched
// totalLeaves * 512 * C threads -- one per (voxel-slot, channel) pair.
// Each thread independently walked the 3D pooling window (up to
// poolingFactor^3 NanoVDB tree lookups) to read a single channel value.
// This meant redundant tree traversals: C threads per voxel each repeated
// the same isActive/getValue sequence for every (i,j,k) in the window.
//
// The new code launches totalLeaves * 512 threads (one per voxel-slot)
// and each thread walks the pooling window once, reading all C channels
// per fine voxel found.  This trades per-channel GPU parallelism for
// fewer tree traversals.  The tree lookups (getAccessor, isActive,
// getValue) dominate cost; the subsequent channel reads are sequential
// on a contiguous [N, C] tensor and are fast in L1/L2.  This is the
// same pattern used by UpsampleGridNearest.

struct downsample_grid_avg_pool_op {
    template <typename Tag>
    static torch::Tensor
    op(Tag tg,
       GridBatchImpl const &fineGrid,
       GridBatchImpl const &coarseGrid,
       torch::Tensor fineDataCoalesced, // already [N, C] from entry point
       torch::Tensor originalFineData,  // original shape for spliceShape
       nanovdb::Coord poolingFactor,
       nanovdb::Coord stride) {
        constexpr auto dev    = ::dispatch::tag_get<torch::DeviceType>(tg);
        constexpr auto stype  = ::dispatch::tag_get<torch::ScalarType>(tg);
        constexpr auto contig = ::dispatch::tag_get<::dispatch::contiguity>(tg);

        using scalar_t    = ::dispatch::torch_scalar_cpp_type_t<stype>;
        using accscalar_t = at::acc_type<scalar_t, /*is_cuda=*/true>;

        int64_t const numCoarseVoxels = coarseGrid.totalVoxels();
        int64_t const numChannels     = fineDataCoalesced.size(1);

        torch::Tensor outCoarseData = torch::zeros(spliceShape({numCoarseVoxels}, originalFineData),
                                                   torch::TensorOptions()
                                                       .dtype(originalFineData.dtype())
                                                       .device(originalFineData.device()));
        torch::Tensor outCoarseCoalesced = featureCoalescedView(outCoarseData);

        // fine_v: dispatched contiguity -- coalesced input may be strided
        auto fine_v = ::dispatch::tensor_in<dev, stype, 2, contig>(fineDataCoalesced);
        // coarse_out: always contiguous -- freshly allocated
        auto coarse_out = ::dispatch::tensor_out<dev, stype, 2, ::dispatch::contiguity::contiguous>(
            outCoarseCoalesced);
        auto fine_acc = ::fvdb::detail::dispatch::make_grid_accessor(tg, fineGrid);

        scalar_t const avgFactor =
            static_cast<scalar_t>(1.0 / (poolingFactor[0] * poolingFactor[1] * poolingFactor[2]));

        // Iterate over coarse voxels, aggregate fine features
        ::fvdb::detail::dispatch::forEachActiveVoxel(
            tg,
            coarseGrid,
            [=] __hostdev__(Tag /*tg*/,
                            JIdxType batchIdx,
                            nanovdb::Coord coarseIjk,
                            int64_t coarseIndex,
                            GridBatchImpl::Accessor /*coarseAcc*/) {
                nanovdb::Coord const fineIjk0(
                    coarseIjk[0] * stride[0], coarseIjk[1] * stride[1], coarseIjk[2] * stride[2]);

                int64_t const fineBaseOffset = fine_acc.voxelOffset(batchIdx);
                auto const fineGridAcc       = fine_acc.grid(batchIdx)->getAccessor();

                for (int64_t c = 0; c < numChannels; ++c) {
                    accscalar_t avgValue = static_cast<accscalar_t>(0);

                    for (nanovdb::Coord::ValueType i = 0; i < poolingFactor[0]; ++i) {
                        for (nanovdb::Coord::ValueType j = 0; j < poolingFactor[1]; ++j) {
                            for (nanovdb::Coord::ValueType k = 0; k < poolingFactor[2]; ++k) {
                                nanovdb::Coord const fineIjk = fineIjk0 + nanovdb::Coord(i, j, k);
                                if (fineGridAcc.isActive(fineIjk)) {
                                    int64_t const fi =
                                        static_cast<int64_t>(fineGridAcc.getValue(fineIjk)) +
                                        fineBaseOffset - 1;
                                    avgValue += static_cast<accscalar_t>(fine_v(fi, c));
                                }
                            }
                        }
                    }

                    coarse_out(coarseIndex, c) = static_cast<scalar_t>(avgValue) * avgFactor;
                }
            });

        return outCoarseData;
    }

    using space      = ::dispatch::axes<::dispatch::torch_full_device_axis,
                                        downsample_stype_axis,
                                        ::dispatch::full_contiguity_axis>;
    using subspaces  = ::dispatch::coverage<space>;
    using dispatcher = ::dispatch::dispatch_table<space,
                                                  torch::Tensor(GridBatchImpl const &,
                                                                GridBatchImpl const &,
                                                                torch::Tensor,
                                                                torch::Tensor,
                                                                nanovdb::Coord,
                                                                nanovdb::Coord)>;
};

// ---------------------------------------------------------------------------
// Backward: scatter coarse gradients to fine voxels
// ---------------------------------------------------------------------------
// The entry point coalesces coarseGradOut before dispatching.
// Contiguity: dispatched on the coalesced coarseGradOut.
//
// Same iteration change as forward: one thread per coarse voxel-slot
// with an inner channel loop, replacing the old per-(voxel, channel)
// thread grid.  No atomics are needed here because stride >= poolingFactor
// guarantees each fine voxel is written by at most one coarse voxel.

struct downsample_grid_avg_pool_backward_op {
    template <typename Tag>
    static torch::Tensor
    op(Tag tg,
       GridBatchImpl const &coarseGrid,
       GridBatchImpl const &fineGrid,
       torch::Tensor coarseGradOutCoalesced, // already [M, C] from entry point
       torch::Tensor fineDataCoalesced,      // already [N, C]; only for shape
       torch::Tensor originalCoarseGradOut,  // original shape for spliceShape
       nanovdb::Coord poolingFactor,
       nanovdb::Coord stride) {
        constexpr auto dev    = ::dispatch::tag_get<torch::DeviceType>(tg);
        constexpr auto stype  = ::dispatch::tag_get<torch::ScalarType>(tg);
        constexpr auto contig = ::dispatch::tag_get<::dispatch::contiguity>(tg);

        using scalar_t = ::dispatch::torch_scalar_cpp_type_t<stype>;

        torch::Tensor outFineGradIn = torch::zeros_like(fineDataCoalesced);
        int64_t const numChannels   = fineDataCoalesced.size(1);

        // coarse_grad: dispatched contiguity
        auto coarse_grad = ::dispatch::tensor_in<dev, stype, 2, contig>(coarseGradOutCoalesced);
        // fine_grad_out: always contiguous -- freshly allocated via zeros_like
        auto fine_grad_out =
            ::dispatch::tensor_out<dev, stype, 2, ::dispatch::contiguity::contiguous>(
                outFineGradIn);
        auto fine_acc = ::fvdb::detail::dispatch::make_grid_accessor(tg, fineGrid);

        scalar_t const avgFactor =
            static_cast<scalar_t>(1.0 / (poolingFactor[0] * poolingFactor[1] * poolingFactor[2]));

        // Iterate over coarse voxels, write gradient to each fine voxel in the
        // pooling window.
        ::fvdb::detail::dispatch::forEachActiveVoxel(
            tg,
            coarseGrid,
            [=] __hostdev__(Tag /*tg*/,
                            JIdxType batchIdx,
                            nanovdb::Coord coarseIjk,
                            int64_t coarseIndex,
                            GridBatchImpl::Accessor /*coarseAcc*/) {
                nanovdb::Coord const fineIjk0(
                    coarseIjk[0] * stride[0], coarseIjk[1] * stride[1], coarseIjk[2] * stride[2]);

                int64_t const fineBaseOffset = fine_acc.voxelOffset(batchIdx);
                auto const fineGridAcc       = fine_acc.grid(batchIdx)->getAccessor();

                for (nanovdb::Coord::ValueType i = 0; i < poolingFactor[0]; ++i) {
                    for (nanovdb::Coord::ValueType j = 0; j < poolingFactor[1]; ++j) {
                        for (nanovdb::Coord::ValueType k = 0; k < poolingFactor[2]; ++k) {
                            nanovdb::Coord const fineIjk = fineIjk0 + nanovdb::Coord(i, j, k);
                            if (fineGridAcc.isActive(fineIjk)) {
                                int64_t const fi =
                                    static_cast<int64_t>(fineGridAcc.getValue(fineIjk)) +
                                    fineBaseOffset - 1;
                                for (int64_t c = 0; c < numChannels; ++c) {
                                    fine_grad_out(fi, c) = coarse_grad(coarseIndex, c) * avgFactor;
                                }
                            }
                        }
                    }
                }
            });

        return outFineGradIn.reshape(
            spliceShape({fineDataCoalesced.size(0)}, originalCoarseGradOut));
    }

    using space      = ::dispatch::axes<::dispatch::torch_full_device_axis,
                                        downsample_stype_axis,
                                        ::dispatch::full_contiguity_axis>;
    using subspaces  = ::dispatch::coverage<space>;
    using dispatcher = ::dispatch::dispatch_table<space,
                                                  torch::Tensor(GridBatchImpl const &,
                                                                GridBatchImpl const &,
                                                                torch::Tensor,
                                                                torch::Tensor,
                                                                torch::Tensor,
                                                                nanovdb::Coord,
                                                                nanovdb::Coord)>;
};

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

torch::Tensor
downsampleGridAvgPool(GridBatchImpl const &fineGrid,
                      GridBatchImpl const &coarseGrid,
                      torch::Tensor fineData,
                      nanovdb::Coord poolingFactor,
                      nanovdb::Coord stride) {
    c10::DeviceGuard guard(fineGrid.device());
    TORCH_CHECK(fineGrid.device() == coarseGrid.device(),
                "coarse_grid and fine_grid must be on the same device");
    for (int i = 0; i < 3; ++i) {
        TORCH_CHECK_VALUE(poolingFactor[i] > 0, "pooling_factor must be greater than 0");
        TORCH_CHECK_VALUE(stride[i] >= 0, "stride must be greater than or equal to 0");
        if (stride[i] == 0) {
            stride[i] = poolingFactor[i];
        }
    }
    coarseGrid.checkNonEmptyGrid();
    fineGrid.checkNonEmptyGrid();
    coarseGrid.checkDevice(fineData);
    TORCH_CHECK(fineData.dim() > 1,
                "fine_data must have more than one dimension. i.e. have shape (num_voxels, *)");
    TORCH_CHECK(fineData.size(0) == static_cast<int64_t>(fineGrid.totalVoxels()),
                "fine_data must have the same number of voxels as fine_grid");

    torch::Tensor fineDataCoalesced = featureCoalescedView(fineData);

    static auto const table =
        ::dispatch::dispatch_table_from_op<downsample_grid_avg_pool_op>("downsampleGridAvgPool");
    return table.select(
        ::dispatch::dispatch_set{fineData.device().type(),
                                 fineData.scalar_type(),
                                 ::dispatch::torch_get_contiguity(fineDataCoalesced)})(
        fineGrid, coarseGrid, fineDataCoalesced, fineData, poolingFactor, stride);
}

torch::Tensor
downsampleGridAvgPoolBackward(GridBatchImpl const &coarseGrid,
                              GridBatchImpl const &fineGrid,
                              torch::Tensor fineData,
                              torch::Tensor coarseGradOut,
                              nanovdb::Coord poolingFactor,
                              nanovdb::Coord stride) {
    c10::DeviceGuard guard(coarseGrid.device());
    for (int i = 0; i < 3; ++i) {
        TORCH_CHECK_VALUE(poolingFactor[i] > 0, "pooling_factor must be greater than 0");
        TORCH_CHECK_VALUE(stride[i] >= 0, "stride must be greater than or equal to 0");
        if (stride[i] == 0) {
            stride[i] = poolingFactor[i];
        }
    }

    torch::Tensor fineDataCoalesced      = featureCoalescedView(fineData);
    torch::Tensor coarseGradOutCoalesced = featureCoalescedView(coarseGradOut);

    static auto const table =
        ::dispatch::dispatch_table_from_op<downsample_grid_avg_pool_backward_op>(
            "downsampleGridAvgPoolBackward");
    return table.select(::dispatch::dispatch_set{
        coarseGradOut.device().type(),
        coarseGradOut.scalar_type(),
        ::dispatch::torch_get_contiguity(coarseGradOutCoalesced)})(coarseGrid,
                                                                   fineGrid,
                                                                   coarseGradOutCoalesced,
                                                                   fineDataCoalesced,
                                                                   coarseGradOut,
                                                                   poolingFactor,
                                                                   stride);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
