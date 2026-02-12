// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// GatherScatterFused.cu -- Fused gather-scatter sparse convolution implementation.
//
// Single-kernel path optimized for small channel counts.  Probes the src grid
// directly for each kernel offset -- no precomputed topology or intermediate
// gather buffers are needed.
//
#include "dispatch/detail/core_types.h"
#include "dispatch/dispatch_table.h"
#include "dispatch/torch/dispatch.h"
#include "dispatch/torch/for_each.h"
#include "dispatch/torch/views.h"

#include <fvdb/detail/dispatch/AtomicAdd.cuh>
#include <fvdb/detail/dispatch/ForEachActiveVoxel.cuh>
#include <fvdb/detail/dispatch/GridAccessor.h>
#include <fvdb/detail/dispatch/TensorChecks.h>
#include <fvdb/detail/ops/convolution/GatherScatterFused.h>

#include <torch/types.h>

#include <cstdint>
#include <tuple>

namespace fvdb {
namespace detail {
namespace ops {

using namespace ::dispatch;

// =============================================================================
// Fused forward convolution (small-C optimized)
// =============================================================================
//
// Single-kernel path: for each active dst voxel, probe the src grid at every
// kernel offset and multiply-accumulate features * weights directly into the
// output.  No intermediate gather buffer, no cuBLAS, no precomputed topology.
//
// Optimal when C_in and C_out are small (say <= 32-64) so that the per-voxel
// dot products fit comfortably in registers and cuBLAS launch overhead would
// otherwise dominate.

struct fused_conv_op {
    template <typename Tag>
        requires with_type<Tag, torch::DeviceType> && with_type<Tag, torch::ScalarType>
    static torch::Tensor
    op(Tag tg,
       torch::Tensor features, // [S, C_in]
       torch::Tensor weights,  // [C_out, C_in, k0, k1, k2]
       GridBatchImpl const &src,
       GridBatchImpl const &dst,
       nanovdb::Coord kernel_size,
       nanovdb::Coord stride) {
        constexpr auto dev        = tag_get<torch::DeviceType>(tg);
        constexpr auto feat_stype = tag_get<torch::ScalarType>(tg);
        using scalar_t            = torch_scalar_cpp_type_t<feat_stype>;

        auto guard = make_device_guard(tag<dev>{}, features);

        int64_t const ks0 = kernel_size[0];
        int64_t const ks1 = kernel_size[1];
        int64_t const ks2 = kernel_size[2];
        int64_t const K   = ks0 * ks1 * ks2;

        int64_t const D     = dst.totalVoxels();
        int64_t const C_out = weights.size(0);
        int64_t const C_in  = weights.size(1);

        // Reshape weights from [C_out, C_in, k0, k1, k2] to [K, C_out, C_in]
        // so that for a given (k, cout) the C_in weights are contiguous -- the
        // layout that makes the inner dot-product loop cache-friendly.
        //
        // Cost: one contiguous copy of K * C_out * C_in elements.  For a 3x3x3
        // kernel with C_in = C_out = 32 this is 27K floats (108 KB), which
        // copies in ~1 us on a modern GPU -- well under a single kernel launch
        // overhead (~5-10 us) and negligible against the fused kernel itself
        // (typically 50-500+ us for grids of practical size).  Accepting
        // pre-reshaped weights would save this microsecond but break the
        // standard PyTorch conv3d [C_out, C_in, k0, k1, k2] layout contract.
        auto W = weights.permute({2, 3, 4, 0, 1}).reshape({K, C_out, C_in}).contiguous();

        // Cast weights to feature dtype if they differ (mixed precision).
        if (W.scalar_type() != feat_stype) {
            W = W.to(feat_stype);
        }

        // Output: [D, C_out], pre-zeroed, same dtype/device as features.
        auto output = torch::zeros({D, C_out}, features.options());

        if (D == 0 || K == 0) {
            return output;
        }

        // Raw pointers for the fused kernel.
        auto feat_ptr = features.data_ptr<scalar_t>();
        auto W_ptr    = W.data_ptr<scalar_t>();
        auto out_ptr  = output.data_ptr<scalar_t>();

        // Kernel start offsets (centered kernel, matching existing convention).
        nanovdb::Coord const kernel_start(static_cast<int>(std::floor(-ks0 / 2.0 + 1)),
                                          static_cast<int>(std::floor(-ks1 / 2.0 + 1)),
                                          static_cast<int>(std::floor(-ks2 / 2.0 + 1)));

        auto src_acc = dispatch::make_grid_accessor(tg, src);

        // One invocation per active dst voxel.  Each voxel-thread loops over
        // all K kernel offsets and all C_out output channels, accumulating
        // the full convolution result with no intermediate buffers.
        dispatch::forEachActiveVoxel(
            tg,
            dst,
            [=] __hostdev__(Tag,
                            JIdxType batch_idx,
                            nanovdb::Coord dst_ijk,
                            int64_t dst_voxel_index,
                            GridBatchImpl::Accessor /*dst_acc*/) {
                nanovdb::Coord const src_base(
                    dst_ijk[0] * stride[0], dst_ijk[1] * stride[1], dst_ijk[2] * stride[2]);

                auto const *src_grid          = src_acc.grid(batch_idx);
                auto src_tree_acc             = src_grid->getAccessor();
                int64_t const src_base_offset = src_acc.voxelOffset(batch_idx);

                int64_t k_idx = 0;
                for (int k0 = kernel_start[0]; k0 < kernel_start[0] + static_cast<int>(ks0); ++k0) {
                    for (int k1 = kernel_start[1]; k1 < kernel_start[1] + static_cast<int>(ks1);
                         ++k1) {
                        for (int k2 = kernel_start[2]; k2 < kernel_start[2] + static_cast<int>(ks2);
                             ++k2, ++k_idx) {
                            nanovdb::Coord const probe = src_base + nanovdb::Coord(k0, k1, k2);
                            if (!src_tree_acc.isActive(probe)) {
                                continue;
                            }
                            int64_t const src_flat =
                                src_base_offset + src_tree_acc.getValue(probe) - 1;

                            // W is [K, C_out, C_in] contiguous.
                            // feat is [S, C_in] contiguous.
                            // out is [D, C_out] contiguous.
                            scalar_t const *f_row = feat_ptr + src_flat * C_in;
                            scalar_t const *w_row = W_ptr + k_idx * (C_out * C_in);

                            for (int64_t co = 0; co < C_out; ++co) {
                                scalar_t acc          = scalar_t(0);
                                scalar_t const *w_col = w_row + co * C_in;
                                for (int64_t ci = 0; ci < C_in; ++ci) {
                                    acc += f_row[ci] * w_col[ci];
                                }
                                out_ptr[dst_voxel_index * C_out + co] += acc;
                            }
                        }
                    }
                }
            });

        return output;
    }

    // Dispatch space: device x feature scalar type (same as gather-scatter path).
    using space = axes<torch_full_device_axis, torch_full_float_stype_axis>;

    using subspaces = coverage<space>;

    using dispatcher = dispatch_table<space,
                                      torch::Tensor(torch::Tensor,
                                                    torch::Tensor,
                                                    GridBatchImpl const &,
                                                    GridBatchImpl const &,
                                                    nanovdb::Coord,
                                                    nanovdb::Coord)>;
};

// Type-erased entry point for fused convolution
torch::Tensor
gatherScatterSparseConvFused(torch::Tensor features,
                             torch::Tensor weights,
                             GridBatchImpl const &src,
                             GridBatchImpl const &dst,
                             nanovdb::Coord kernel_size,
                             nanovdb::Coord stride) {
    // Precondition checks
    TORCH_CHECK(features.dim() == 2,
                "features must be 2D [total_src_voxels, C_in], got ",
                features.dim(),
                "D");
    TORCH_CHECK(features.size(0) == static_cast<int64_t>(src.totalVoxels()),
                "features.size(0)=",
                features.size(0),
                " must match src totalVoxels=",
                src.totalVoxels());
    TORCH_CHECK(features.is_floating_point(), "features must be floating point");

    TORCH_CHECK(weights.dim() == 5,
                "weights must be 5D [C_out, C_in, k0, k1, k2], got ",
                weights.dim(),
                "D");
    TORCH_CHECK(weights.size(1) == features.size(1),
                "weights C_in=",
                weights.size(1),
                " must match features C_in=",
                features.size(1));
    TORCH_CHECK(weights.size(2) == kernel_size[0] && weights.size(3) == kernel_size[1] &&
                    weights.size(4) == kernel_size[2],
                "weights spatial dims [",
                weights.size(2),
                ",",
                weights.size(3),
                ",",
                weights.size(4),
                "] must match kernel_size [",
                kernel_size[0],
                ",",
                kernel_size[1],
                ",",
                kernel_size[2],
                "]");
    TORCH_CHECK(weights.is_floating_point(), "weights must be floating point");

    TORCH_CHECK(features.device() == weights.device(),
                "features and weights must be on the same device");
    TORCH_CHECK(src.device() == dst.device(),
                "src and dst grids must be on the same device, got ",
                src.device(),
                " and ",
                dst.device());
    TORCH_CHECK(features.device() == src.device(), "features and grids must be on the same device");

    for (int d = 0; d < 3; ++d) {
        TORCH_CHECK(
            kernel_size[d] > 0, "kernel_size[", d, "] must be positive, got ", kernel_size[d]);
        TORCH_CHECK(stride[d] > 0, "stride[", d, "] must be positive, got ", stride[d]);
    }

    static auto const table =
        dispatch_table_from_op<fused_conv_op>("gather_scatter_sparse_conv_fused");

    auto const dev  = features.device().type();
    auto const f_st = features.scalar_type();

    return table.select(dispatch_set{dev, f_st})(features, weights, src, dst, kernel_size, stride);
}

// =============================================================================
// Fused backward convolution (small-C optimized)
// =============================================================================
//
// Single-kernel backward path: for each active dst voxel, probe the src grid
// at every kernel offset and accumulate gradients for both features and weights
// using atomic_add.  No intermediate buffers, no cuBLAS.
//
// The weight gradient accumulates across all dst voxels into [K, C_out, C_in],
// so there is contention on atomicAdd for grad_weights.  This is acceptable
// because this path targets small C where the contention is limited and cuBLAS
// launch overhead would otherwise dominate.

struct fused_conv_backward_op {
    template <typename Tag>
        requires with_type<Tag, torch::DeviceType> && with_type<Tag, torch::ScalarType>
    static std::tuple<torch::Tensor, torch::Tensor>
    op(Tag tg,
       torch::Tensor grad_output, // [D, C_out]
       torch::Tensor features,    // [S, C_in]
       torch::Tensor weights,     // [C_out, C_in, k0, k1, k2]
       GridBatchImpl const &src,
       GridBatchImpl const &dst,
       nanovdb::Coord kernel_size,
       nanovdb::Coord stride) {
        constexpr auto dev        = tag_get<torch::DeviceType>(tg);
        constexpr auto feat_stype = tag_get<torch::ScalarType>(tg);
        using scalar_t            = torch_scalar_cpp_type_t<feat_stype>;

        auto guard = make_device_guard(tag<dev>{}, features);

        int64_t const ks0 = kernel_size[0];
        int64_t const ks1 = kernel_size[1];
        int64_t const ks2 = kernel_size[2];
        int64_t const K   = ks0 * ks1 * ks2;

        int64_t const S     = static_cast<int64_t>(src.totalVoxels());
        int64_t const D     = dst.totalVoxels();
        int64_t const C_out = weights.size(0);
        int64_t const C_in  = weights.size(1);

        // Reshape weights: [C_out, C_in, k0, k1, k2] -> [K, C_out, C_in]
        // Same layout as fused forward.
        auto W = weights.permute({2, 3, 4, 0, 1}).reshape({K, C_out, C_in}).contiguous();
        if (W.scalar_type() != feat_stype) {
            W = W.to(feat_stype);
        }

        auto grad_features         = torch::zeros({S, C_in}, features.options());
        auto grad_weights_reshaped = torch::zeros({K, C_out, C_in}, features.options());

        if (D == 0 || K == 0) {
            auto grad_weights = grad_weights_reshaped.reshape({ks0, ks1, ks2, C_out, C_in})
                                    .permute({3, 4, 0, 1, 2})
                                    .contiguous();
            return {grad_features, grad_weights};
        }

        auto feat_ptr   = features.data_ptr<scalar_t>();
        auto grad_o_ptr = grad_output.data_ptr<scalar_t>();
        auto W_ptr      = W.data_ptr<scalar_t>();
        auto grad_f_ptr = grad_features.data_ptr<scalar_t>();
        auto grad_W_ptr = grad_weights_reshaped.data_ptr<scalar_t>();

        nanovdb::Coord const kernel_start(static_cast<int>(std::floor(-ks0 / 2.0 + 1)),
                                          static_cast<int>(std::floor(-ks1 / 2.0 + 1)),
                                          static_cast<int>(std::floor(-ks2 / 2.0 + 1)));

        auto src_acc = dispatch::make_grid_accessor(tg, src);

        dispatch::forEachActiveVoxel(
            tg,
            dst,
            [=] __hostdev__(Tag tg_inner,
                            JIdxType batch_idx,
                            nanovdb::Coord dst_ijk,
                            int64_t dst_voxel_index,
                            GridBatchImpl::Accessor /*dst_acc*/) {
                nanovdb::Coord const src_base(
                    dst_ijk[0] * stride[0], dst_ijk[1] * stride[1], dst_ijk[2] * stride[2]);

                auto const *src_grid          = src_acc.grid(batch_idx);
                auto src_tree_acc             = src_grid->getAccessor();
                int64_t const src_base_offset = src_acc.voxelOffset(batch_idx);

                int64_t k_idx = 0;
                for (int k0 = kernel_start[0]; k0 < kernel_start[0] + static_cast<int>(ks0); ++k0) {
                    for (int k1 = kernel_start[1]; k1 < kernel_start[1] + static_cast<int>(ks1);
                         ++k1) {
                        for (int k2 = kernel_start[2]; k2 < kernel_start[2] + static_cast<int>(ks2);
                             ++k2, ++k_idx) {
                            nanovdb::Coord const probe = src_base + nanovdb::Coord(k0, k1, k2);
                            if (!src_tree_acc.isActive(probe)) {
                                continue;
                            }
                            int64_t const src_flat =
                                src_base_offset + src_tree_acc.getValue(probe) - 1;

                            // Pointers into the current rows.
                            scalar_t const *go_row = grad_o_ptr + dst_voxel_index * C_out;
                            scalar_t const *f_row  = feat_ptr + src_flat * C_in;
                            scalar_t const *w_row  = W_ptr + k_idx * (C_out * C_in);

                            // grad_features[src_flat, ci] += sum_co(grad_out[d,co] * W[k,co,ci])
                            for (int64_t ci = 0; ci < C_in; ++ci) {
                                scalar_t acc = scalar_t(0);
                                for (int64_t co = 0; co < C_out; ++co) {
                                    acc += go_row[co] * w_row[co * C_in + ci];
                                }
                                dispatch::atomic_add(
                                    tg_inner, &grad_f_ptr[src_flat * C_in + ci], acc);
                            }

                            // grad_weights[k, co, ci] += features[src, ci] * grad_out[d, co]
                            scalar_t *gw_row = grad_W_ptr + k_idx * (C_out * C_in);
                            for (int64_t co = 0; co < C_out; ++co) {
                                for (int64_t ci = 0; ci < C_in; ++ci) {
                                    dispatch::atomic_add(
                                        tg_inner, &gw_row[co * C_in + ci], f_row[ci] * go_row[co]);
                                }
                            }
                        }
                    }
                }
            });

        // Reshape grad_weights from [K, C_out, C_in] to [C_out, C_in, k0, k1, k2]
        auto grad_weights = grad_weights_reshaped.reshape({ks0, ks1, ks2, C_out, C_in})
                                .permute({3, 4, 0, 1, 2})
                                .contiguous();

        return {grad_features, grad_weights};
    }

    using space     = axes<torch_full_device_axis, torch_full_float_stype_axis>;
    using subspaces = coverage<space>;
    using dispatcher =
        dispatch_table<space,
                       std::tuple<torch::Tensor, torch::Tensor>(torch::Tensor,
                                                                torch::Tensor,
                                                                torch::Tensor,
                                                                GridBatchImpl const &,
                                                                GridBatchImpl const &,
                                                                nanovdb::Coord,
                                                                nanovdb::Coord)>;
};

// Type-erased entry point for fused backward convolution
std::tuple<torch::Tensor, torch::Tensor>
gatherScatterSparseConvFusedBackward(torch::Tensor grad_output,
                                     torch::Tensor features,
                                     torch::Tensor weights,
                                     GridBatchImpl const &src,
                                     GridBatchImpl const &dst,
                                     nanovdb::Coord kernel_size,
                                     nanovdb::Coord stride) {
    // Precondition checks
    TORCH_CHECK(grad_output.dim() == 2,
                "grad_output must be 2D [dst_total_voxels, C_out], got ",
                grad_output.dim(),
                "D");
    TORCH_CHECK(grad_output.size(0) == static_cast<int64_t>(dst.totalVoxels()),
                "grad_output.size(0)=",
                grad_output.size(0),
                " must match dst totalVoxels=",
                dst.totalVoxels());
    TORCH_CHECK(grad_output.is_floating_point(), "grad_output must be floating point");

    TORCH_CHECK(features.dim() == 2,
                "features must be 2D [total_src_voxels, C_in], got ",
                features.dim(),
                "D");
    TORCH_CHECK(features.size(0) == static_cast<int64_t>(src.totalVoxels()),
                "features.size(0)=",
                features.size(0),
                " must match src totalVoxels=",
                src.totalVoxels());
    TORCH_CHECK(features.is_floating_point(), "features must be floating point");

    TORCH_CHECK(weights.dim() == 5,
                "weights must be 5D [C_out, C_in, k0, k1, k2], got ",
                weights.dim(),
                "D");
    TORCH_CHECK(weights.size(0) == grad_output.size(1),
                "weights C_out=",
                weights.size(0),
                " must match grad_output C_out=",
                grad_output.size(1));
    TORCH_CHECK(weights.size(1) == features.size(1),
                "weights C_in=",
                weights.size(1),
                " must match features C_in=",
                features.size(1));
    TORCH_CHECK(weights.size(2) == kernel_size[0] && weights.size(3) == kernel_size[1] &&
                    weights.size(4) == kernel_size[2],
                "weights spatial dims must match kernel_size");
    TORCH_CHECK(weights.is_floating_point(), "weights must be floating point");

    TORCH_CHECK(grad_output.device() == features.device(),
                "grad_output and features must be on the same device");
    TORCH_CHECK(features.device() == weights.device(),
                "features and weights must be on the same device");
    TORCH_CHECK(src.device() == dst.device(), "src and dst grids must be on the same device");
    TORCH_CHECK(features.device() == src.device(), "features and grids must be on the same device");

    for (int d = 0; d < 3; ++d) {
        TORCH_CHECK(
            kernel_size[d] > 0, "kernel_size[", d, "] must be positive, got ", kernel_size[d]);
        TORCH_CHECK(stride[d] > 0, "stride[", d, "] must be positive, got ", stride[d]);
    }

    static auto const table =
        dispatch_table_from_op<fused_conv_backward_op>("gather_scatter_sparse_conv_fused_backward");

    auto const dev  = features.device().type();
    auto const f_st = features.scalar_type();

    return table.select(dispatch_set{dev, f_st})(
        grad_output, features, weights, src, dst, kernel_size, stride);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
