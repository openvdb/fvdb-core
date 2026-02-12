// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// GatherScatter.cu -- Gather-scatter sparse convolution implementation (GEMM path).
//
// Topology precomputation: forEachActiveVoxel over the output grid, probing the
// feature grid's nanovdb accessor for each kernel offset.
//
// Forward/transposed convolution: per-kernel-weight loop of gather -> GEMM -> accumulate.
// Uses the dispatch table with device and feature scalar type axes.
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

#include <torch/types.h>

#include <cstdint>
#include <tuple>

namespace fvdb {
namespace detail {
namespace ops {

using namespace ::dispatch;

// =============================================================================
// Tensor options helpers
// =============================================================================

template <torch::ScalarType Stype>
inline torch::TensorOptions
opts_on(torch::Device device) {
    return torch::dtype(Stype).device(device);
}

// =============================================================================
// Topology implementation via dispatch table
// =============================================================================
//
// Parameterized by ConvDirection.  For each active voxel in output_grid, probe
// the feature_grid at every kernel offset and record the flat feature voxel
// index (or -1).
//
// Forward:    probe = output_ijk * stride + kernel_offset
// Transposed: probe = (output_ijk - kernel_offset) / stride
//             (only when divisible by stride in all 3 dimensions)

struct topology_op {
    template <typename Tag>
        requires with_type<Tag, torch::DeviceType>
    static GatherScatterTopology
    op(Tag tg,
       GridBatchImpl const &feature_grid,
       GridBatchImpl const &output_grid,
       nanovdb::Coord kernel_size,
       nanovdb::Coord stride,
       ConvDirection direction) {
        constexpr auto dev = tag_get<torch::DeviceType>(Tag{});

        int64_t const ks0           = kernel_size[0];
        int64_t const ks1           = kernel_size[1];
        int64_t const ks2           = kernel_size[2];
        int64_t const kernel_volume = ks0 * ks1 * ks2;

        int64_t const output_total  = output_grid.totalVoxels();
        int64_t const feature_total = feature_grid.totalVoxels();

        // Kernel start offsets (centered kernel, matching existing convention)
        nanovdb::Coord const kernel_start(static_cast<int>(std::floor(-ks0 / 2.0 + 1)),
                                          static_cast<int>(std::floor(-ks1 / 2.0 + 1)),
                                          static_cast<int>(std::floor(-ks2 / 2.0 + 1)));

        // Allocate output kernel map: [output_total, kernel_volume], filled with -1
        auto kmap = torch::full(
            {output_total, kernel_volume}, -1, opts_on<torch::kInt32>(output_grid.device()));
        auto kmap_out = tensor_out<dev, torch::kInt32, 2, contiguity::contiguous>(kmap);

        auto guard       = make_device_guard(tg, kmap);
        auto feature_acc = dispatch::make_grid_accessor(tg, feature_grid);

        bool const is_transposed = (direction == ConvDirection::Transposed);

        // Iterate over every active output voxel
        dispatch::forEachActiveVoxel(
            tg,
            output_grid,
            [=] __hostdev__(Tag,
                            JIdxType batch_idx,
                            nanovdb::Coord output_ijk,
                            int64_t output_voxel_index,
                            GridBatchImpl::Accessor /*output_acc*/) {
                auto const *feat_grid          = feature_acc.grid(batch_idx);
                auto feat_tree_acc             = feat_grid->getAccessor();
                int64_t const feat_base_offset = feature_acc.voxelOffset(batch_idx);

                int64_t k_idx = 0;
                for (int k0 = kernel_start[0]; k0 < kernel_start[0] + static_cast<int>(ks0); ++k0) {
                    for (int k1 = kernel_start[1]; k1 < kernel_start[1] + static_cast<int>(ks1);
                         ++k1) {
                        for (int k2 = kernel_start[2]; k2 < kernel_start[2] + static_cast<int>(ks2);
                             ++k2, ++k_idx) {
                            nanovdb::Coord probe;
                            if (is_transposed) {
                                // Transposed: probe = (output_ijk - offset) / stride
                                int const r0 = output_ijk[0] - k0;
                                int const r1 = output_ijk[1] - k1;
                                int const r2 = output_ijk[2] - k2;
                                // Check divisibility by stride
                                if (r0 % stride[0] != 0 || r1 % stride[1] != 0 ||
                                    r2 % stride[2] != 0) {
                                    continue; // stays -1
                                }
                                probe =
                                    nanovdb::Coord(r0 / stride[0], r1 / stride[1], r2 / stride[2]);
                            } else {
                                // Forward: probe = output_ijk * stride + offset
                                nanovdb::Coord const base(output_ijk[0] * stride[0],
                                                          output_ijk[1] * stride[1],
                                                          output_ijk[2] * stride[2]);
                                probe = base + nanovdb::Coord(k0, k1, k2);
                            }

                            if (feat_tree_acc.isActive(probe)) {
                                int32_t const feat_flat = static_cast<int32_t>(
                                    feat_base_offset + feat_tree_acc.getValue(probe) - 1);
                                kmap_out(output_voxel_index, k_idx) = feat_flat;
                            }
                            // else: stays -1 from initialization
                        }
                    }
                }
            });

        // Check if the center kernel offset is an identity mapping
        bool center_identity = false;
        if (kernel_volume % 2 == 1) {
            int64_t const mid = kernel_volume / 2;
            auto mid_col      = kmap.select(1, mid);
            auto expected =
                torch::arange(output_total, opts_on<torch::kInt32>(output_grid.device()));
            center_identity = mid_col.equal(expected);
        }

        return GatherScatterTopology{
            /*.kernel_map             =*/kmap,
            /*.feature_total_voxels   =*/feature_total,
            /*.output_total_voxels    =*/output_total,
            /*.kernel_volume          =*/kernel_volume,
            /*.kernel_size            =*/kernel_size,
            /*.stride                 =*/stride,
            /*.direction              =*/direction,
            /*.center_is_identity     =*/center_identity,
        };
    }

    // Single axis: device.  ConvDirection is a runtime parameter.
    using space      = axes<torch_full_device_axis>;
    using subspaces  = coverage<space>;
    using dispatcher = dispatch_table<space,
                                      GatherScatterTopology(GridBatchImpl const &,
                                                            GridBatchImpl const &,
                                                            nanovdb::Coord,
                                                            nanovdb::Coord,
                                                            ConvDirection)>;
};

// Shared precondition checks for topology builders
static void
checkTopologyPreconditions(GridBatchImpl const &feature_grid,
                           GridBatchImpl const &output_grid,
                           nanovdb::Coord kernel_size,
                           nanovdb::Coord stride) {
    TORCH_CHECK(feature_grid.device() == output_grid.device(),
                "feature_grid and output_grid must be on the same device, got ",
                feature_grid.device(),
                " and ",
                output_grid.device());
    for (int d = 0; d < 3; ++d) {
        TORCH_CHECK(
            kernel_size[d] > 0, "kernel_size[", d, "] must be positive, got ", kernel_size[d]);
        TORCH_CHECK(stride[d] > 0, "stride[", d, "] must be positive, got ", stride[d]);
    }
}

// Type-erased entry point for forward topology
GatherScatterTopology
gatherScatterSparseConvTopology(GridBatchImpl const &feature_grid,
                                GridBatchImpl const &output_grid,
                                nanovdb::Coord kernel_size,
                                nanovdb::Coord stride) {
    checkTopologyPreconditions(feature_grid, output_grid, kernel_size, stride);

    static auto const table =
        dispatch_table_from_op<topology_op>("gather_scatter_sparse_conv_topology");

    auto const dev = feature_grid.device().type();
    return table.select(dispatch_set{dev})(
        feature_grid, output_grid, kernel_size, stride, ConvDirection::Forward);
}

// Type-erased entry point for transposed topology
GatherScatterTopology
gatherScatterSparseConvTransposeTopology(GridBatchImpl const &feature_grid,
                                         GridBatchImpl const &output_grid,
                                         nanovdb::Coord kernel_size,
                                         nanovdb::Coord stride) {
    checkTopologyPreconditions(feature_grid, output_grid, kernel_size, stride);

    static auto const table =
        dispatch_table_from_op<topology_op>("gather_scatter_sparse_conv_transpose_topology");

    auto const dev = feature_grid.device().type();
    return table.select(dispatch_set{dev})(
        feature_grid, output_grid, kernel_size, stride, ConvDirection::Transposed);
}

// =============================================================================
// Gather / scatter helpers (direction-agnostic)
// =============================================================================

// Gather: for a given kernel index k, copy feature-grid features into a dense buffer.
template <typename Tag>
    requires with_type<Tag, torch::DeviceType> && with_type<Tag, torch::ScalarType>
void
conv_gather(Tag tg,
            torch::Tensor features, // [F, C]
            torch::Tensor buf_in,   // [O, C]
            torch::Tensor kmap_col, // [O] int32
            int64_t O,
            int64_t C) {
    constexpr auto dev   = tag_get<torch::DeviceType>(tg);
    constexpr auto stype = tag_get<torch::ScalarType>(tg);
    using scalar_t       = torch_scalar_cpp_type_t<stype>;

    auto feat_ptr = features.data_ptr<scalar_t>();
    auto buf_ptr  = buf_in.data_ptr<scalar_t>();
    auto kmap_ptr = kmap_col.data_ptr<int32_t>();

    for_each(tg, O * C, [=] __hostdev__(Tag, int64_t idx) {
        int64_t const o        = idx / C;
        int64_t const c        = idx % C;
        int32_t const feat_idx = kmap_ptr[o];
        buf_ptr[o * C + c] =
            (feat_idx >= 0) ? feat_ptr[static_cast<int64_t>(feat_idx) * C + c] : scalar_t(0);
    });
}

// Scatter-add: atomically accumulate buf values back into grad_features.
template <typename Tag>
    requires with_type<Tag, torch::DeviceType> && with_type<Tag, torch::ScalarType>
void
conv_scatter_add(Tag tg,
                 torch::Tensor buf,       // [O, C]
                 torch::Tensor grad_feat, // [F, C] destination
                 torch::Tensor kmap_col,  // [O] int32
                 int64_t O,
                 int64_t C) {
    constexpr auto stype = tag_get<torch::ScalarType>(tg);
    using scalar_t       = torch_scalar_cpp_type_t<stype>;

    auto buf_ptr       = buf.data_ptr<scalar_t>();
    auto grad_feat_ptr = grad_feat.data_ptr<scalar_t>();
    auto kmap_ptr      = kmap_col.data_ptr<int32_t>();

    for_each(tg, O * C, [=] __hostdev__(Tag tg_inner, int64_t idx) {
        int64_t const o        = idx / C;
        int64_t const c        = idx % C;
        int32_t const feat_idx = kmap_ptr[o];
        if (feat_idx >= 0) {
            dispatch::atomic_add(tg_inner,
                                 &grad_feat_ptr[static_cast<int64_t>(feat_idx) * C + c],
                                 buf_ptr[o * C + c]);
        }
    });
}

// =============================================================================
// Forward convolution implementation
// =============================================================================
//
// Works for both forward and transposed convolution.  The topology encodes
// the direction (probe coordinate math) and the weight reshape changes:
//
// Forward:    weights [C_out, C_in, K] -> [K, C_in, C_out]
//             features [F, C_in] @ W[k] [C_in, C_out] -> output [O, C_out]
//
// Transposed: weights [C_out, C_in, K] -> [K, C_out, C_in]
//             features [F, C_out] @ W[k] [C_out, C_in] -> output [O, C_in]

struct gather_scatter_conv_op {
    template <typename Tag>
        requires with_type<Tag, torch::DeviceType> && with_type<Tag, torch::ScalarType>
    static torch::Tensor
    op(Tag tg,
       torch::Tensor features, // [F, C_feat]
       torch::Tensor weights,  // [C_out, C_in, k0, k1, k2]
       GatherScatterTopology const &topo) {
        constexpr auto dev        = tag_get<torch::DeviceType>(tg);
        constexpr auto feat_stype = tag_get<torch::ScalarType>(tg);

        auto guard = make_device_guard(tag<dev>{}, features);

        int64_t const O = topo.output_total_voxels;
        int64_t const K = topo.kernel_volume;

        // Weights are always [C_out, C_in, k0, k1, k2].
        // Features always have C_in channels; output always has C_out channels.
        // The only difference for transposed is in the topology (negated probe
        // offsets) and the weight permutation.
        int64_t const C_out = weights.size(0);
        int64_t const C_in  = weights.size(1);

        // Reshape weights to [K, C_in, C_out]:
        //   Forward:    permute {2,3,4,1,0} -> [K, C_in, C_out]
        //   Transposed: permute {2,3,4,0,1} -> [K, C_out, C_in], then we
        //               transpose each slice in the GEMM via W[k].T.
        //               Equivalently: just use the same [K, C_in, C_out] layout.
        //
        // The old codebase always reshapes to [K, inC, outC] for both directions.
        // We follow the same pattern.
        auto weights_reshaped =
            weights.permute({2, 3, 4, 1, 0}).reshape({K, C_in, C_out}).contiguous();

        if (weights_reshaped.scalar_type() != feat_stype) {
            weights_reshaped = weights_reshaped.to(feat_stype);
        }

        // GEMM: gathered_features [O, C_in] @ W[k] [C_in, C_out] -> [O, C_out]
        // This is the same for both forward and transposed -- the topology
        // handles the direction difference.
        auto output = torch::zeros({O, C_out}, features.options());

        if (O == 0 || K == 0) {
            return output;
        }

        auto buf_in  = torch::empty({O, C_in}, features.options());
        auto buf_out = torch::empty({O, C_out}, features.options());

        int64_t const mid_k     = K / 2;
        bool const middle_accel = topo.center_is_identity;

        if (middle_accel) {
            torch::mm_out(output, features, weights_reshaped[mid_k]);
        }

        auto feat_tag = tag<dev, feat_stype>{};
        auto kmap_t   = topo.kernel_map.t().contiguous();

        for (int64_t k = 0; k < K; ++k) {
            if (middle_accel && k == mid_k) {
                continue;
            }

            auto kmap_col = kmap_t[k];
            conv_gather(feat_tag, features, buf_in, kmap_col, O, C_in);
            torch::mm_out(buf_out, buf_in, weights_reshaped[k]);
            output.add_(buf_out);
        }

        return output;
    }

    using space = axes<torch_full_device_axis, torch_full_float_stype_axis>;

    using subspaces = coverage<space>;

    using dispatcher =
        dispatch_table<space,
                       torch::Tensor(torch::Tensor, torch::Tensor, GatherScatterTopology const &)>;
};

// =============================================================================
// Backward convolution implementation (GEMM path)
// =============================================================================
//
// Works for both forward and transposed backward.  Uses the same
// gather + GEMM + scatter pattern; the weight reshape matches the
// forward direction stored in topo.direction.

struct gather_scatter_conv_backward_op {
    template <typename Tag>
        requires with_type<Tag, torch::DeviceType> && with_type<Tag, torch::ScalarType>
    static std::tuple<torch::Tensor, torch::Tensor>
    op(Tag tg,
       torch::Tensor grad_output, // [O, C_out]
       torch::Tensor features,    // [F, C_in]
       torch::Tensor weights,     // [C_out, C_in, k0, k1, k2]
       GatherScatterTopology const &topo) {
        constexpr auto dev        = tag_get<torch::DeviceType>(tg);
        constexpr auto feat_stype = tag_get<torch::ScalarType>(tg);

        auto guard = make_device_guard(tag<dev>{}, features);

        int64_t const F     = topo.feature_total_voxels;
        int64_t const O     = topo.output_total_voxels;
        int64_t const K     = topo.kernel_volume;
        int64_t const C_out = weights.size(0);
        int64_t const C_in  = weights.size(1);

        // Same reshape as forward: always [K, C_in, C_out]
        auto weights_reshaped =
            weights.permute({2, 3, 4, 1, 0}).reshape({K, C_in, C_out}).contiguous();
        if (weights_reshaped.scalar_type() != feat_stype) {
            weights_reshaped = weights_reshaped.to(feat_stype);
        }

        auto grad_features         = torch::zeros({F, C_in}, features.options());
        auto grad_weights_reshaped = torch::zeros({K, C_in, C_out}, features.options());

        if (O == 0 || K == 0) {
            // [K, C_in, C_out] -> [k0,k1,k2, C_in, C_out] -> [C_out, C_in, k0,k1,k2]
            auto ks           = topo.kernel_size;
            auto grad_weights = grad_weights_reshaped.reshape({ks[0], ks[1], ks[2], C_in, C_out})
                                    .permute({4, 3, 0, 1, 2})
                                    .contiguous();
            return {grad_features, grad_weights};
        }

        auto buf_feat    = torch::empty({O, C_in}, features.options());
        auto buf_grad_in = torch::empty({O, C_in}, features.options());

        int64_t const mid_k     = K / 2;
        bool const middle_accel = topo.center_is_identity;

        if (middle_accel) {
            // grad_features += grad_output @ W[mid].T
            grad_features.addmm_(grad_output, weights_reshaped[mid_k].t());
            // grad_weights[mid] = features.T @ grad_output
            auto gw_mid = grad_weights_reshaped[mid_k];
            torch::mm_out(gw_mid, features.t(), grad_output);
        }

        auto feat_tag = tag<dev, feat_stype>{};
        auto kmap_t   = topo.kernel_map.t().contiguous();

        for (int64_t k = 0; k < K; ++k) {
            if (middle_accel && k == mid_k) {
                continue;
            }

            auto kmap_col = kmap_t[k];

            // Gather features for this kernel offset
            conv_gather(feat_tag, features, buf_feat, kmap_col, O, C_in);
            // grad_features: [O, C_out] @ [C_out, C_in] -> [O, C_in]
            torch::mm_out(buf_grad_in, grad_output, weights_reshaped[k].t());
            conv_scatter_add(feat_tag, buf_grad_in, grad_features, kmap_col, O, C_in);
            // grad_weights[k] = features.T @ grad_output: [C_in, O] @ [O, C_out] -> [C_in, C_out]
            auto gw_k = grad_weights_reshaped[k];
            torch::mm_out(gw_k, buf_feat.t(), grad_output);
        }

        // Reshape grad_weights from [K, C_in, C_out] back to [C_out, C_in, k0, k1, k2]
        auto ks           = topo.kernel_size;
        auto grad_weights = grad_weights_reshaped.reshape({ks[0], ks[1], ks[2], C_in, C_out})
                                .permute({4, 3, 0, 1, 2})
                                .contiguous();

        return {grad_features, grad_weights};
    }

    using space      = axes<torch_full_device_axis, torch_full_float_stype_axis>;
    using subspaces  = coverage<space>;
    using dispatcher = dispatch_table<
        space,
        std::tuple<torch::Tensor, torch::Tensor>(
            torch::Tensor, torch::Tensor, torch::Tensor, GatherScatterTopology const &)>;
};

// =============================================================================
// Shared precondition checks
// =============================================================================

static void
checkConvPreconditions(torch::Tensor features,
                       torch::Tensor weights,
                       GatherScatterTopology const &topo,
                       char const *name) {
    TORCH_CHECK(features.dim() == 2, name, ": features must be 2D, got ", features.dim(), "D");
    TORCH_CHECK(features.size(0) == topo.feature_total_voxels,
                name,
                ": features.size(0)=",
                features.size(0),
                " must match feature_total_voxels=",
                topo.feature_total_voxels);
    TORCH_CHECK(features.is_floating_point(), name, ": features must be floating point");

    TORCH_CHECK(weights.dim() == 5, name, ": weights must be 5D, got ", weights.dim(), "D");

    // Features always have C_in channels = weights.size(1), regardless of direction.
    TORCH_CHECK(features.size(1) == weights.size(1),
                name,
                ": features channels=",
                features.size(1),
                " must match weights C_in=",
                weights.size(1));

    TORCH_CHECK(weights.size(2) == topo.kernel_size[0] && weights.size(3) == topo.kernel_size[1] &&
                    weights.size(4) == topo.kernel_size[2],
                name,
                ": weights spatial dims must match topology kernel_size");
    TORCH_CHECK(weights.is_floating_point(), name, ": weights must be floating point");

    TORCH_CHECK(features.device() == weights.device(),
                name,
                ": features and weights must be on the same device");
    TORCH_CHECK(features.device() == topo.kernel_map.device(),
                name,
                ": features and kernel_map must be on the same device");
}

// =============================================================================
// Type-erased entry points
// =============================================================================

// Forward convolution
torch::Tensor
gatherScatterSparseConv(torch::Tensor features,
                        torch::Tensor weights,
                        GatherScatterTopology const &topo) {
    checkConvPreconditions(features, weights, topo, "gatherScatterSparseConv");
    TORCH_CHECK(topo.direction == ConvDirection::Forward,
                "gatherScatterSparseConv requires topology with direction=Forward");

    static auto const table =
        dispatch_table_from_op<gather_scatter_conv_op>("gather_scatter_sparse_conv");

    auto const dev  = features.device().type();
    auto const f_st = features.scalar_type();

    return table.select(dispatch_set{dev, f_st})(features, weights, topo);
}

// Forward backward
std::tuple<torch::Tensor, torch::Tensor>
gatherScatterSparseConvBackward(torch::Tensor grad_output,
                                torch::Tensor features,
                                torch::Tensor weights,
                                GatherScatterTopology const &topo) {
    checkConvPreconditions(features, weights, topo, "gatherScatterSparseConvBackward");
    TORCH_CHECK(topo.direction == ConvDirection::Forward,
                "gatherScatterSparseConvBackward requires topology with direction=Forward");
    TORCH_CHECK(grad_output.dim() == 2 && grad_output.size(0) == topo.output_total_voxels,
                "grad_output shape mismatch");
    TORCH_CHECK(grad_output.is_floating_point(), "grad_output must be floating point");
    TORCH_CHECK(grad_output.device() == features.device(),
                "grad_output and features must be on the same device");

    static auto const table = dispatch_table_from_op<gather_scatter_conv_backward_op>(
        "gather_scatter_sparse_conv_backward");

    auto const dev  = features.device().type();
    auto const f_st = features.scalar_type();

    return table.select(dispatch_set{dev, f_st})(grad_output, features, weights, topo);
}

// Transposed convolution
torch::Tensor
gatherScatterSparseConvTranspose(torch::Tensor features,
                                 torch::Tensor weights,
                                 GatherScatterTopology const &topo) {
    checkConvPreconditions(features, weights, topo, "gatherScatterSparseConvTranspose");
    TORCH_CHECK(topo.direction == ConvDirection::Transposed,
                "gatherScatterSparseConvTranspose requires topology with direction=Transposed");

    static auto const table =
        dispatch_table_from_op<gather_scatter_conv_op>("gather_scatter_sparse_conv_transpose");

    auto const dev  = features.device().type();
    auto const f_st = features.scalar_type();

    return table.select(dispatch_set{dev, f_st})(features, weights, topo);
}

// Transposed backward
std::tuple<torch::Tensor, torch::Tensor>
gatherScatterSparseConvTransposeBackward(torch::Tensor grad_output,
                                         torch::Tensor features,
                                         torch::Tensor weights,
                                         GatherScatterTopology const &topo) {
    checkConvPreconditions(features, weights, topo, "gatherScatterSparseConvTransposeBackward");
    TORCH_CHECK(
        topo.direction == ConvDirection::Transposed,
        "gatherScatterSparseConvTransposeBackward requires topology with direction=Transposed");
    TORCH_CHECK(grad_output.dim() == 2 && grad_output.size(0) == topo.output_total_voxels,
                "grad_output shape mismatch");
    TORCH_CHECK(grad_output.is_floating_point(), "grad_output must be floating point");
    TORCH_CHECK(grad_output.device() == features.device(),
                "grad_output and features must be on the same device");

    static auto const table = dispatch_table_from_op<gather_scatter_conv_backward_op>(
        "gather_scatter_sparse_conv_transpose_backward");

    auto const dev  = features.device().type();
    auto const f_st = features.scalar_type();

    return table.select(dispatch_set{dev, f_st})(grad_output, features, weights, topo);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
