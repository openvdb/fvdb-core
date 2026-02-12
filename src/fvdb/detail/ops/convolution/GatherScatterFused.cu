// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// GatherScatterFused.cu -- Fused gather-scatter sparse convolution implementation.
//
// Single-kernel path optimized for small channel counts.  Probes the feature
// grid directly for each kernel offset -- no precomputed topology or
// intermediate gather buffers are needed.
//
// Supports both forward and transposed convolution via ConvDirection.
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
#include <fvdb/detail/ops/convolution/GatherScatter.h>
#include <fvdb/detail/ops/convolution/GatherScatterFused.h>

#include <torch/types.h>

#include <cstdint>
#include <tuple>

namespace fvdb {
namespace detail {
namespace ops {

using namespace ::dispatch;

// =============================================================================
// Fused forward convolution (parameterized by ConvDirection)
// =============================================================================
//
// For each active output voxel, probe the feature grid at every kernel offset
// and multiply-accumulate features * weights directly into the output.
//
// Forward:    probe = output_ijk * stride + kernel_offset
//             W reshaped [K, C_out, C_in]: output[o, co] += sum_ci(feat[f, ci] * W[k, co, ci])
//
// Transposed: probe = (output_ijk - kernel_offset) / stride  (when divisible)
//             W reshaped [K, C_in, C_out]: output[o, ci] += sum_co(feat[f, co] * W[k, ci, co])

struct fused_conv_op {
    template <typename Tag>
        requires with_type<Tag, torch::DeviceType> && with_type<Tag, torch::ScalarType>
    static torch::Tensor
    op(Tag tg,
       torch::Tensor features,
       torch::Tensor weights,
       GridBatchImpl const &feature_grid,
       GridBatchImpl const &output_grid,
       nanovdb::Coord kernel_size,
       nanovdb::Coord stride,
       ConvDirection direction) {
        constexpr auto dev        = tag_get<torch::DeviceType>(tg);
        constexpr auto feat_stype = tag_get<torch::ScalarType>(tg);
        using scalar_t            = torch_scalar_cpp_type_t<feat_stype>;

        auto guard = make_device_guard(tag<dev>{}, features);

        int64_t const ks0 = kernel_size[0];
        int64_t const ks1 = kernel_size[1];
        int64_t const ks2 = kernel_size[2];
        int64_t const K   = ks0 * ks1 * ks2;

        int64_t const O = output_grid.totalVoxels();

        bool const is_transposed = (direction == ConvDirection::Transposed);

        // Weights are always [C_out, C_in, k0, k1, k2].
        // Features always have C_in channels; output always has C_out channels.
        int64_t const C_out = weights.size(0);
        int64_t const C_in  = weights.size(1);

        // Reshape to [K, C_out, C_in] so the inner dot product iterates over C_in.
        // output[o, co] += sum_ci(feat[f, ci] * W[k, co, ci])
        auto W = weights.permute({2, 3, 4, 0, 1}).reshape({K, C_out, C_in}).contiguous();

        if (W.scalar_type() != feat_stype) {
            W = W.to(feat_stype);
        }

        auto output = torch::zeros({O, C_out}, features.options());

        if (O == 0 || K == 0) {
            return output;
        }

        auto feat_ptr = features.data_ptr<scalar_t>();
        auto W_ptr    = W.data_ptr<scalar_t>();
        auto out_ptr  = output.data_ptr<scalar_t>();

        nanovdb::Coord const kernel_start(static_cast<int>(std::floor(-ks0 / 2.0 + 1)),
                                          static_cast<int>(std::floor(-ks1 / 2.0 + 1)),
                                          static_cast<int>(std::floor(-ks2 / 2.0 + 1)));

        auto feat_acc = dispatch::make_grid_accessor(tg, feature_grid);

        dispatch::forEachActiveVoxel(
            tg,
            output_grid,
            [=] __hostdev__(Tag,
                            JIdxType batch_idx,
                            nanovdb::Coord output_ijk,
                            int64_t output_voxel_index,
                            GridBatchImpl::Accessor /*output_acc*/) {
                auto const *fg         = feat_acc.grid(batch_idx);
                auto feat_tree_acc     = fg->getAccessor();
                int64_t const f_offset = feat_acc.voxelOffset(batch_idx);

                int64_t k_idx = 0;
                for (int k0 = kernel_start[0]; k0 < kernel_start[0] + static_cast<int>(ks0); ++k0) {
                    for (int k1 = kernel_start[1]; k1 < kernel_start[1] + static_cast<int>(ks1);
                         ++k1) {
                        for (int k2 = kernel_start[2]; k2 < kernel_start[2] + static_cast<int>(ks2);
                             ++k2, ++k_idx) {
                            nanovdb::Coord probe;
                            if (is_transposed) {
                                int const r0 = output_ijk[0] - k0;
                                int const r1 = output_ijk[1] - k1;
                                int const r2 = output_ijk[2] - k2;
                                if (r0 % stride[0] != 0 || r1 % stride[1] != 0 ||
                                    r2 % stride[2] != 0) {
                                    continue;
                                }
                                probe =
                                    nanovdb::Coord(r0 / stride[0], r1 / stride[1], r2 / stride[2]);
                            } else {
                                nanovdb::Coord const base(output_ijk[0] * stride[0],
                                                          output_ijk[1] * stride[1],
                                                          output_ijk[2] * stride[2]);
                                probe = base + nanovdb::Coord(k0, k1, k2);
                            }

                            if (!feat_tree_acc.isActive(probe)) {
                                continue;
                            }
                            int64_t const feat_flat = f_offset + feat_tree_acc.getValue(probe) - 1;

                            // W is [K, C_out, C_in] contiguous.
                            // feat is [F, C_in] contiguous.
                            // out is [O, C_out] contiguous.
                            scalar_t const *f_row = feat_ptr + feat_flat * C_in;
                            scalar_t const *w_row = W_ptr + k_idx * (C_out * C_in);

                            for (int64_t co = 0; co < C_out; ++co) {
                                scalar_t acc          = scalar_t(0);
                                scalar_t const *w_col = w_row + co * C_in;
                                for (int64_t ci = 0; ci < C_in; ++ci) {
                                    acc += f_row[ci] * w_col[ci];
                                }
                                out_ptr[output_voxel_index * C_out + co] += acc;
                            }
                        }
                    }
                }
            });

        return output;
    }

    using space      = axes<torch_full_device_axis, torch_full_float_stype_axis>;
    using subspaces  = coverage<space>;
    using dispatcher = dispatch_table<space,
                                      torch::Tensor(torch::Tensor,
                                                    torch::Tensor,
                                                    GridBatchImpl const &,
                                                    GridBatchImpl const &,
                                                    nanovdb::Coord,
                                                    nanovdb::Coord,
                                                    ConvDirection)>;
};

// =============================================================================
// Fused backward convolution (parameterized by ConvDirection)
// =============================================================================

struct fused_conv_backward_op {
    template <typename Tag>
        requires with_type<Tag, torch::DeviceType> && with_type<Tag, torch::ScalarType>
    static std::tuple<torch::Tensor, torch::Tensor>
    op(Tag tg,
       torch::Tensor grad_output,
       torch::Tensor features,
       torch::Tensor weights,
       GridBatchImpl const &feature_grid,
       GridBatchImpl const &output_grid,
       nanovdb::Coord kernel_size,
       nanovdb::Coord stride,
       ConvDirection direction) {
        constexpr auto dev        = tag_get<torch::DeviceType>(tg);
        constexpr auto feat_stype = tag_get<torch::ScalarType>(tg);
        using scalar_t            = torch_scalar_cpp_type_t<feat_stype>;

        auto guard = make_device_guard(tag<dev>{}, features);

        int64_t const ks0 = kernel_size[0];
        int64_t const ks1 = kernel_size[1];
        int64_t const ks2 = kernel_size[2];
        int64_t const K   = ks0 * ks1 * ks2;

        bool const is_transposed = (direction == ConvDirection::Transposed);

        int64_t const F     = static_cast<int64_t>(feature_grid.totalVoxels());
        int64_t const O     = output_grid.totalVoxels();
        int64_t const C_out = weights.size(0);
        int64_t const C_in  = weights.size(1);

        // Same reshape as forward: always [K, C_out, C_in]
        auto W = weights.permute({2, 3, 4, 0, 1}).reshape({K, C_out, C_in}).contiguous();
        if (W.scalar_type() != feat_stype) {
            W = W.to(feat_stype);
        }

        auto grad_features         = torch::zeros({F, C_in}, features.options());
        auto grad_weights_reshaped = torch::zeros({K, C_out, C_in}, features.options());

        if (O == 0 || K == 0) {
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

        auto feat_acc = dispatch::make_grid_accessor(tg, feature_grid);

        dispatch::forEachActiveVoxel(
            tg,
            output_grid,
            [=] __hostdev__(Tag tg_inner,
                            JIdxType batch_idx,
                            nanovdb::Coord output_ijk,
                            int64_t output_voxel_index,
                            GridBatchImpl::Accessor /*output_acc*/) {
                auto const *fg         = feat_acc.grid(batch_idx);
                auto feat_tree_acc     = fg->getAccessor();
                int64_t const f_offset = feat_acc.voxelOffset(batch_idx);

                int64_t k_idx = 0;
                for (int k0 = kernel_start[0]; k0 < kernel_start[0] + static_cast<int>(ks0); ++k0) {
                    for (int k1 = kernel_start[1]; k1 < kernel_start[1] + static_cast<int>(ks1);
                         ++k1) {
                        for (int k2 = kernel_start[2]; k2 < kernel_start[2] + static_cast<int>(ks2);
                             ++k2, ++k_idx) {
                            nanovdb::Coord probe;
                            if (is_transposed) {
                                int const r0 = output_ijk[0] - k0;
                                int const r1 = output_ijk[1] - k1;
                                int const r2 = output_ijk[2] - k2;
                                if (r0 % stride[0] != 0 || r1 % stride[1] != 0 ||
                                    r2 % stride[2] != 0) {
                                    continue;
                                }
                                probe =
                                    nanovdb::Coord(r0 / stride[0], r1 / stride[1], r2 / stride[2]);
                            } else {
                                nanovdb::Coord const base(output_ijk[0] * stride[0],
                                                          output_ijk[1] * stride[1],
                                                          output_ijk[2] * stride[2]);
                                probe = base + nanovdb::Coord(k0, k1, k2);
                            }

                            if (!feat_tree_acc.isActive(probe)) {
                                continue;
                            }
                            int64_t const feat_flat = f_offset + feat_tree_acc.getValue(probe) - 1;

                            // W is [K, C_out, C_in], grad_output is [O, C_out]
                            scalar_t const *go_row = grad_o_ptr + output_voxel_index * C_out;
                            scalar_t const *f_row  = feat_ptr + feat_flat * C_in;
                            scalar_t const *w_row  = W_ptr + k_idx * (C_out * C_in);

                            // grad_features[feat_flat, ci] += sum_co(grad_out[o,co] * W[k,co,ci])
                            for (int64_t ci = 0; ci < C_in; ++ci) {
                                scalar_t acc = scalar_t(0);
                                for (int64_t co = 0; co < C_out; ++co) {
                                    acc += go_row[co] * w_row[co * C_in + ci];
                                }
                                dispatch::atomic_add(
                                    tg_inner, &grad_f_ptr[feat_flat * C_in + ci], acc);
                            }

                            // grad_weights[k, co, ci] += feat[f, ci] * grad_out[o, co]
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
                                                                nanovdb::Coord,
                                                                ConvDirection)>;
};

// =============================================================================
// Shared precondition checks for fused ops
// =============================================================================

static void
checkFusedPreconditions(torch::Tensor features,
                        torch::Tensor weights,
                        GridBatchImpl const &feature_grid,
                        GridBatchImpl const &output_grid,
                        nanovdb::Coord kernel_size,
                        nanovdb::Coord stride,
                        bool is_transposed,
                        char const *name) {
    TORCH_CHECK(features.dim() == 2, name, ": features must be 2D, got ", features.dim(), "D");
    TORCH_CHECK(features.size(0) == static_cast<int64_t>(feature_grid.totalVoxels()),
                name,
                ": features.size(0)=",
                features.size(0),
                " must match feature_grid totalVoxels=",
                feature_grid.totalVoxels());
    TORCH_CHECK(features.is_floating_point(), name, ": features must be floating point");

    TORCH_CHECK(weights.dim() == 5, name, ": weights must be 5D, got ", weights.dim(), "D");

    // Features always have C_in channels = weights.size(1), regardless of direction.
    TORCH_CHECK(features.size(1) == weights.size(1),
                name,
                ": features channels=",
                features.size(1),
                " must match weights C_in=",
                weights.size(1));

    TORCH_CHECK(weights.size(2) == kernel_size[0] && weights.size(3) == kernel_size[1] &&
                    weights.size(4) == kernel_size[2],
                name,
                ": weights spatial dims must match kernel_size");
    TORCH_CHECK(weights.is_floating_point(), name, ": weights must be floating point");

    TORCH_CHECK(features.device() == weights.device(),
                name,
                ": features and weights must be on the same device");
    TORCH_CHECK(
        feature_grid.device() == output_grid.device(), name, ": grids must be on the same device");
    TORCH_CHECK(features.device() == feature_grid.device(),
                name,
                ": features and grids must be on the same device");

    for (int d = 0; d < 3; ++d) {
        TORCH_CHECK(kernel_size[d] > 0, name, ": kernel_size[", d, "] must be positive");
        TORCH_CHECK(stride[d] > 0, name, ": stride[", d, "] must be positive");
    }
}

// =============================================================================
// Type-erased entry points
// =============================================================================

// Forward fused
torch::Tensor
gatherScatterSparseConvFused(torch::Tensor features,
                             torch::Tensor weights,
                             GridBatchImpl const &feature_grid,
                             GridBatchImpl const &output_grid,
                             nanovdb::Coord kernel_size,
                             nanovdb::Coord stride) {
    checkFusedPreconditions(features,
                            weights,
                            feature_grid,
                            output_grid,
                            kernel_size,
                            stride,
                            false,
                            "gatherScatterSparseConvFused");

    static auto const table =
        dispatch_table_from_op<fused_conv_op>("gather_scatter_sparse_conv_fused");

    auto const dev  = features.device().type();
    auto const f_st = features.scalar_type();

    return table.select(dispatch_set{dev, f_st})(
        features, weights, feature_grid, output_grid, kernel_size, stride, ConvDirection::Forward);
}

// Forward fused backward
std::tuple<torch::Tensor, torch::Tensor>
gatherScatterSparseConvFusedBackward(torch::Tensor grad_output,
                                     torch::Tensor features,
                                     torch::Tensor weights,
                                     GridBatchImpl const &feature_grid,
                                     GridBatchImpl const &output_grid,
                                     nanovdb::Coord kernel_size,
                                     nanovdb::Coord stride) {
    checkFusedPreconditions(features,
                            weights,
                            feature_grid,
                            output_grid,
                            kernel_size,
                            stride,
                            false,
                            "gatherScatterSparseConvFusedBackward");
    TORCH_CHECK(grad_output.dim() == 2 &&
                    grad_output.size(0) == static_cast<int64_t>(output_grid.totalVoxels()),
                "grad_output shape mismatch");
    TORCH_CHECK(grad_output.is_floating_point(), "grad_output must be floating point");
    TORCH_CHECK(grad_output.device() == features.device(),
                "grad_output and features must be on the same device");

    static auto const table =
        dispatch_table_from_op<fused_conv_backward_op>("gather_scatter_sparse_conv_fused_backward");

    auto const dev  = features.device().type();
    auto const f_st = features.scalar_type();

    return table.select(dispatch_set{dev, f_st})(grad_output,
                                                 features,
                                                 weights,
                                                 feature_grid,
                                                 output_grid,
                                                 kernel_size,
                                                 stride,
                                                 ConvDirection::Forward);
}

// Transposed fused
torch::Tensor
gatherScatterSparseConvFusedTranspose(torch::Tensor features,
                                      torch::Tensor weights,
                                      GridBatchImpl const &feature_grid,
                                      GridBatchImpl const &output_grid,
                                      nanovdb::Coord kernel_size,
                                      nanovdb::Coord stride) {
    checkFusedPreconditions(features,
                            weights,
                            feature_grid,
                            output_grid,
                            kernel_size,
                            stride,
                            true,
                            "gatherScatterSparseConvFusedTranspose");

    static auto const table =
        dispatch_table_from_op<fused_conv_op>("gather_scatter_sparse_conv_fused_transpose");

    auto const dev  = features.device().type();
    auto const f_st = features.scalar_type();

    return table.select(dispatch_set{dev, f_st})(features,
                                                 weights,
                                                 feature_grid,
                                                 output_grid,
                                                 kernel_size,
                                                 stride,
                                                 ConvDirection::Transposed);
}

// Transposed fused backward
std::tuple<torch::Tensor, torch::Tensor>
gatherScatterSparseConvFusedTransposeBackward(torch::Tensor grad_output,
                                              torch::Tensor features,
                                              torch::Tensor weights,
                                              GridBatchImpl const &feature_grid,
                                              GridBatchImpl const &output_grid,
                                              nanovdb::Coord kernel_size,
                                              nanovdb::Coord stride) {
    checkFusedPreconditions(features,
                            weights,
                            feature_grid,
                            output_grid,
                            kernel_size,
                            stride,
                            true,
                            "gatherScatterSparseConvFusedTransposeBackward");
    TORCH_CHECK(grad_output.dim() == 2 &&
                    grad_output.size(0) == static_cast<int64_t>(output_grid.totalVoxels()),
                "grad_output shape mismatch");
    TORCH_CHECK(grad_output.is_floating_point(), "grad_output must be floating point");
    TORCH_CHECK(grad_output.device() == features.device(),
                "grad_output and features must be on the same device");

    static auto const table = dispatch_table_from_op<fused_conv_backward_op>(
        "gather_scatter_sparse_conv_fused_transpose_backward");

    auto const dev  = features.device().type();
    auto const f_st = features.scalar_type();

    return table.select(dispatch_set{dev, f_st})(grad_output,
                                                 features,
                                                 weights,
                                                 feature_grid,
                                                 output_grid,
                                                 kernel_size,
                                                 stride,
                                                 ConvDirection::Transposed);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
