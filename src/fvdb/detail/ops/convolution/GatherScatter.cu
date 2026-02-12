// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// GatherScatter.cu -- Gather-scatter sparse convolution implementation.
//
// Topology precomputation: forEachActiveVoxel over the dst grid, probing the
// src grid's nanovdb accessor for each kernel offset.
//
// Forward convolution: per-kernel-weight loop of gather -> GEMM -> accumulate.
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
//
// Reduce the torch::TensorOptions().dtype(...).device(...) boilerplate.
//
//   opts_on<kInt32>(device)  -- compile-time dtype, runtime device
//   tensor.options()         -- same dtype + device as an existing tensor (builtin)

template <torch::ScalarType Stype>
inline torch::TensorOptions
opts_on(torch::Device device) {
    return torch::dtype(Stype).device(device);
}

// =============================================================================
// Topology implementation via dispatch table
// =============================================================================
//
// For each active voxel in the dst grid, probe the src grid at every kernel
// offset and record the flat src voxel index (or -1).

struct topology_op {
    template <typename Tag>
        requires with_type<Tag, torch::DeviceType>
    static GatherScatterTopology
    op(Tag tg,
       GridBatchImpl const &src,
       GridBatchImpl const &dst,
       nanovdb::Coord kernel_size,
       nanovdb::Coord stride) {
        constexpr auto dev = tag_get<torch::DeviceType>(Tag{});

        int64_t const ks0           = kernel_size[0];
        int64_t const ks1           = kernel_size[1];
        int64_t const ks2           = kernel_size[2];
        int64_t const kernel_volume = ks0 * ks1 * ks2;

        int64_t const dst_total = dst.totalVoxels();
        int64_t const src_total = src.totalVoxels();

        // Kernel start offsets (centered kernel, matching existing convention)
        nanovdb::Coord const kernel_start(static_cast<int>(std::floor(-ks0 / 2.0 + 1)),
                                          static_cast<int>(std::floor(-ks1 / 2.0 + 1)),
                                          static_cast<int>(std::floor(-ks2 / 2.0 + 1)));

        // Allocate output kernel map: [dst_total, kernel_volume], filled with -1
        auto kmap =
            torch::full({dst_total, kernel_volume}, -1, opts_on<torch::kInt32>(dst.device()));
        auto kmap_out = tensor_out<dev, torch::kInt32, 2, contiguity::contiguous>(kmap);

        auto guard   = make_device_guard(tg, kmap);
        auto src_acc = dispatch::make_grid_accessor(tg, src);

        // Iterate over every active dst voxel
        dispatch::forEachActiveVoxel(
            tg,
            dst,
            [=] __hostdev__(Tag,
                            JIdxType batch_idx,
                            nanovdb::Coord dst_ijk,
                            int64_t dst_voxel_index,
                            GridBatchImpl::Accessor dst_acc) {
                // Compute source coordinate (dst_ijk * stride)
                nanovdb::Coord const src_base(
                    dst_ijk[0] * stride[0], dst_ijk[1] * stride[1], dst_ijk[2] * stride[2]);

                // Get the src grid's nanovdb accessor for this batch element
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
                            if (src_tree_acc.isActive(probe)) {
                                int32_t const src_flat = static_cast<int32_t>(
                                    src_base_offset + src_tree_acc.getValue(probe) - 1);
                                kmap_out(dst_voxel_index, k_idx) = src_flat;
                            }
                            // else: stays -1 from initialization
                        }
                    }
                }
            });

        // Check if the center kernel offset is an identity mapping
        // (kmap[:, K/2] == arange(D)).  This enables the "middle acceleration"
        // in the forward pass -- verified once here so the hot path is free.
        bool center_identity = false;
        if (kernel_volume % 2 == 1) {
            int64_t const mid = kernel_volume / 2;
            auto mid_col      = kmap.select(1, mid);
            auto expected     = torch::arange(dst_total, opts_on<torch::kInt32>(dst.device()));
            center_identity   = mid_col.equal(expected);
        }

        return GatherScatterTopology{
            /*.kernel_map          =*/kmap,
            /*.src_total_voxels    =*/src_total,
            /*.dst_total_voxels    =*/dst_total,
            /*.kernel_volume       =*/kernel_volume,
            /*.kernel_size         =*/kernel_size,
            /*.stride              =*/stride,
            /*.center_is_identity  =*/center_identity,
        };
    }

    // Single axis: device
    using space      = axes<torch_full_device_axis>;
    using subspaces  = coverage<space>;
    using dispatcher = dispatch_table<
        space,
        GatherScatterTopology(
            GridBatchImpl const &, GridBatchImpl const &, nanovdb::Coord, nanovdb::Coord)>;
};

// Type-erased entry point for topology
GatherScatterTopology
gatherScatterSparseConvTopology(GridBatchImpl const &src,
                                GridBatchImpl const &dst,
                                nanovdb::Coord kernel_size,
                                nanovdb::Coord stride) {
    TORCH_CHECK(src.device() == dst.device(),
                "src and dst grids must be on the same device, got ",
                src.device(),
                " and ",
                dst.device());
    for (int d = 0; d < 3; ++d) {
        TORCH_CHECK(
            kernel_size[d] > 0, "kernel_size[", d, "] must be positive, got ", kernel_size[d]);
        TORCH_CHECK(stride[d] > 0, "stride[", d, "] must be positive, got ", stride[d]);
    }

    static auto const table =
        dispatch_table_from_op<topology_op>("gather_scatter_sparse_conv_topology");

    auto const dev = src.device().type();
    return table.select(dispatch_set{dev})(src, dst, kernel_size, stride);
}

// =============================================================================
// Forward convolution implementation
// =============================================================================

// Gather: for a given kernel index k, copy src features into a dense buffer.
//   buf_in[d, c] = features[kmap[d, k], c]   if kmap[d,k] >= 0
//                  0                           otherwise
//
// Inactive entries are explicitly zeroed so callers need not pre-zero buf_in.
template <typename Tag>
    requires with_type<Tag, torch::DeviceType> && with_type<Tag, torch::ScalarType>
void
conv_gather(Tag tg,
            torch::Tensor features, // [S, C_in]
            torch::Tensor buf_in,   // [D, C_in]
            torch::Tensor kmap_col, // [D] int32, the k-th column of kernel_map
            int64_t D,
            int64_t C_in) {
    constexpr auto dev   = tag_get<torch::DeviceType>(tg);
    constexpr auto stype = tag_get<torch::ScalarType>(tg);
    using scalar_t       = torch_scalar_cpp_type_t<stype>;

    auto feat_ptr = features.data_ptr<scalar_t>();
    auto buf_ptr  = buf_in.data_ptr<scalar_t>();
    auto kmap_ptr = kmap_col.data_ptr<int32_t>();

    for_each(tg, D * C_in, [=] __hostdev__(Tag, int64_t idx) {
        int64_t const d       = idx / C_in;
        int64_t const c       = idx % C_in;
        int32_t const src_idx = kmap_ptr[d];
        buf_ptr[d * C_in + c] =
            (src_idx >= 0) ? feat_ptr[static_cast<int64_t>(src_idx) * C_in + c] : scalar_t(0);
    });
}

// Scatter-add: for a given kernel index k, atomically accumulate buf values
// back into the gradient features tensor.
//   grad_feat[kmap[d, k], c] += buf[d, c]   if kmap[d,k] >= 0
//
// Uses dispatch::atomic_add for thread-safe accumulation on both CPU and GPU.
template <typename Tag>
    requires with_type<Tag, torch::DeviceType> && with_type<Tag, torch::ScalarType>
void
conv_scatter_add(Tag tg,
                 torch::Tensor buf,       // [D, C]
                 torch::Tensor grad_feat, // [S, C] destination (accumulated into)
                 torch::Tensor kmap_col,  // [D] int32, the k-th column of kernel_map
                 int64_t D,
                 int64_t C) {
    constexpr auto stype = tag_get<torch::ScalarType>(tg);
    using scalar_t       = torch_scalar_cpp_type_t<stype>;

    auto buf_ptr       = buf.data_ptr<scalar_t>();
    auto grad_feat_ptr = grad_feat.data_ptr<scalar_t>();
    auto kmap_ptr      = kmap_col.data_ptr<int32_t>();

    for_each(tg, D * C, [=] __hostdev__(Tag tg_inner, int64_t idx) {
        int64_t const d       = idx / C;
        int64_t const c       = idx % C;
        int32_t const src_idx = kmap_ptr[d];
        if (src_idx >= 0) {
            dispatch::atomic_add(tg_inner,
                                 &grad_feat_ptr[static_cast<int64_t>(src_idx) * C + c],
                                 buf_ptr[d * C + c]);
        }
    });
}

// The dispatch op struct for the forward convolution.
// Dispatch axes: device x feature_stype.
//
// Mixed feature/weight dtypes are supported: weights are cast to the feature
// dtype after reshape.  Output dtype follows the feature dtype.
//
// The actual per-kernel-weight loop lives in the op method.  Each iteration:
//   1. Gather src features for this kernel offset (zeros for inactive entries)
//   2. GEMM: buf_in [D, C_in] x weight_k [C_in, C_out] -> buf_out [D, C_out]
//   3. output += buf_out
struct gather_scatter_conv_op {
    template <typename Tag>
        requires with_type<Tag, torch::DeviceType> && with_type<Tag, torch::ScalarType>
    static torch::Tensor
    op(Tag tg,
       torch::Tensor features, // [S, C_in]
       torch::Tensor weights,  // [C_out, C_in, k0, k1, k2]
       GatherScatterTopology const &topo) {
        constexpr auto dev        = tag_get<torch::DeviceType>(tg);
        constexpr auto feat_stype = tag_get<torch::ScalarType>(tg);

        auto guard = make_device_guard(tag<dev>{}, features);

        int64_t const D     = topo.dst_total_voxels;
        int64_t const K     = topo.kernel_volume;
        int64_t const C_out = weights.size(0);
        int64_t const C_in  = weights.size(1);

        // Reshape weights from [C_out, C_in, k0, k1, k2] to [K, C_in, C_out]
        // where K = k0*k1*k2, and the per-weight slice is [C_in, C_out] for mm.
        //
        // Cost: one contiguous copy of K * C_in * C_out elements.  For a 3x3x3
        // kernel with C_in = C_out = 128 this is 442K floats (1.7 MB), which
        // copies in ~10 us on a modern GPU -- well under a single cuBLAS launch
        // and negligible against the K GEMM calls that follow.  Accepting
        // pre-reshaped weights would save this but break the standard PyTorch
        // conv3d [C_out, C_in, k0, k1, k2] layout contract.
        auto weights_reshaped =
            weights.permute({2, 3, 4, 1, 0}).reshape({K, C_in, C_out}).contiguous();

        // Cast weights to feature dtype if they differ (mixed precision support).
        if (weights_reshaped.scalar_type() != feat_stype) {
            weights_reshaped = weights_reshaped.to(feat_stype);
        }

        // Output: [D, C_out], same dtype and device as features
        auto output = torch::zeros({D, C_out}, features.options());

        if (D == 0 || K == 0) {
            return output;
        }

        // Reusable buffers for gather and GEMM output
        auto buf_in  = torch::empty({D, C_in}, features.options());
        auto buf_out = torch::empty({D, C_out}, features.options());

        // Middle acceleration: if the center kernel offset is a verified
        // identity mapping (precomputed in topology), skip gather and mm
        // features directly against the center weight.
        int64_t const mid_k     = K / 2;
        bool const middle_accel = topo.center_is_identity;

        if (middle_accel) {
            torch::mm_out(output, features, weights_reshaped[mid_k]);
        }

        // Tag for dispatching gather (uses feature scalar type only)
        auto feat_tag = tag<dev, feat_stype>{};

        // Transpose kernel_map once: [D, K] -> [K, D] contiguous, so
        // kmap_t[k] is a contiguous row (avoids per-iteration allocation).
        auto kmap_t = topo.kernel_map.t().contiguous();

        for (int64_t k = 0; k < K; ++k) {
            if (middle_accel && k == mid_k) {
                continue;
            }

            auto kmap_col = kmap_t[k]; // [D], contiguous row

            // Gather src features for this kernel offset
            conv_gather(feat_tag, features, buf_in, kmap_col, D, C_in);

            // GEMM: [D, C_in] x [C_in, C_out] -> [D, C_out]
            torch::mm_out(buf_out, buf_in, weights_reshaped[k]);

            // Accumulate into output
            output.add_(buf_out);
        }

        return output;
    }

    // Dispatch space: device x feature scalar type
    using space = axes<torch_full_device_axis, torch_full_float_stype_axis>;

    using subspaces = coverage<space>;

    using dispatcher =
        dispatch_table<space,
                       torch::Tensor(torch::Tensor, torch::Tensor, GatherScatterTopology const &)>;
};

// Type-erased entry point for forward convolution
torch::Tensor
gatherScatterSparseConv(torch::Tensor features,
                        torch::Tensor weights,
                        GatherScatterTopology const &topo) {
    // Precondition checks
    TORCH_CHECK(features.dim() == 2,
                "features must be 2D [total_src_voxels, C_in], got ",
                features.dim(),
                "D");
    TORCH_CHECK(features.size(0) == topo.src_total_voxels,
                "features.size(0)=",
                features.size(0),
                " must match src_total_voxels=",
                topo.src_total_voxels);
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
    TORCH_CHECK(weights.size(2) == topo.kernel_size[0] && weights.size(3) == topo.kernel_size[1] &&
                    weights.size(4) == topo.kernel_size[2],
                "weights spatial dims [",
                weights.size(2),
                ",",
                weights.size(3),
                ",",
                weights.size(4),
                "] must match topology kernel_size [",
                topo.kernel_size[0],
                ",",
                topo.kernel_size[1],
                ",",
                topo.kernel_size[2],
                "]");
    TORCH_CHECK(weights.is_floating_point(), "weights must be floating point");

    TORCH_CHECK(features.device() == weights.device(),
                "features and weights must be on the same device");
    TORCH_CHECK(features.device() == topo.kernel_map.device(),
                "features and kernel_map must be on the same device");

    static auto const table =
        dispatch_table_from_op<gather_scatter_conv_op>("gather_scatter_sparse_conv");

    auto const dev  = features.device().type();
    auto const f_st = features.scalar_type();

    return table.select(dispatch_set{dev, f_st})(features, weights, topo);
}

// =============================================================================
// Backward convolution implementation (GEMM path)
// =============================================================================
//
// Computes gradients for features and weights given grad_output from the
// forward pass.  Uses the same gather + GEMM + scatter pattern:
//
// For each kernel offset k:
//   1. Gather: buf_feat = gather(features, kmap[:,k])         [D, C_in]
//   2. GEMM:  buf_grad = grad_output @ weights[k].T           [D, C_in]
//   3. Scatter-add: grad_features[kmap[:,k]] += buf_grad       atomically
//   4. GEMM:  grad_weights[k] = buf_feat.T @ grad_output      [C_in, C_out]

struct gather_scatter_conv_backward_op {
    template <typename Tag>
        requires with_type<Tag, torch::DeviceType> && with_type<Tag, torch::ScalarType>
    static std::tuple<torch::Tensor, torch::Tensor>
    op(Tag tg,
       torch::Tensor grad_output, // [D, C_out]
       torch::Tensor features,    // [S, C_in]
       torch::Tensor weights,     // [C_out, C_in, k0, k1, k2]
       GatherScatterTopology const &topo) {
        constexpr auto dev        = tag_get<torch::DeviceType>(tg);
        constexpr auto feat_stype = tag_get<torch::ScalarType>(tg);

        auto guard = make_device_guard(tag<dev>{}, features);

        int64_t const S     = topo.src_total_voxels;
        int64_t const D     = topo.dst_total_voxels;
        int64_t const K     = topo.kernel_volume;
        int64_t const C_out = weights.size(0);
        int64_t const C_in  = weights.size(1);

        // Reshape weights: [C_out, C_in, k0, k1, k2] -> [K, C_in, C_out]
        // Same layout as forward so weights_reshaped[k] is [C_in, C_out].
        auto weights_reshaped =
            weights.permute({2, 3, 4, 1, 0}).reshape({K, C_in, C_out}).contiguous();
        if (weights_reshaped.scalar_type() != feat_stype) {
            weights_reshaped = weights_reshaped.to(feat_stype);
        }

        // Outputs: grad_features [S, C_in], grad_weights_reshaped [K, C_in, C_out]
        auto grad_features         = torch::zeros({S, C_in}, features.options());
        auto grad_weights_reshaped = torch::zeros({K, C_in, C_out}, features.options());

        if (D == 0 || K == 0) {
            // Reshape grad_weights to [C_out, C_in, k0, k1, k2]
            auto ks           = topo.kernel_size;
            auto grad_weights = grad_weights_reshaped.reshape({ks[0], ks[1], ks[2], C_in, C_out})
                                    .permute({4, 3, 0, 1, 2})
                                    .contiguous();
            return {grad_features, grad_weights};
        }

        // Reusable buffers
        auto buf_feat    = torch::empty({D, C_in}, features.options());
        auto buf_grad_in = torch::empty({D, C_in}, features.options());

        // Middle acceleration
        int64_t const mid_k     = K / 2;
        bool const middle_accel = topo.center_is_identity;

        if (middle_accel) {
            // grad_features += grad_output @ weights[mid_k].T
            // Since center is identity, dst voxel d maps to src voxel d directly.
            grad_features.addmm_(grad_output, weights_reshaped[mid_k].t());
            // grad_weights[mid_k] = features.T @ grad_output
            auto gw_mid = grad_weights_reshaped[mid_k];
            torch::mm_out(gw_mid, features.t(), grad_output);
        }

        auto feat_tag = tag<dev, feat_stype>{};
        auto kmap_t   = topo.kernel_map.t().contiguous();

        for (int64_t k = 0; k < K; ++k) {
            if (middle_accel && k == mid_k) {
                continue;
            }

            auto kmap_col = kmap_t[k]; // [D], contiguous row

            // 1. Gather features for this kernel offset
            conv_gather(feat_tag, features, buf_feat, kmap_col, D, C_in);

            // 2. GEMM for grad_features: [D, C_out] x [C_out, C_in] -> [D, C_in]
            torch::mm_out(buf_grad_in, grad_output, weights_reshaped[k].t());

            // 3. Scatter-add into grad_features
            conv_scatter_add(feat_tag, buf_grad_in, grad_features, kmap_col, D, C_in);

            // 4. GEMM for grad_weights: [C_in, D] x [D, C_out] -> [C_in, C_out]
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

// Type-erased entry point for backward convolution (GEMM path)
std::tuple<torch::Tensor, torch::Tensor>
gatherScatterSparseConvBackward(torch::Tensor grad_output,
                                torch::Tensor features,
                                torch::Tensor weights,
                                GatherScatterTopology const &topo) {
    // Precondition checks
    TORCH_CHECK(grad_output.dim() == 2,
                "grad_output must be 2D [dst_total_voxels, C_out], got ",
                grad_output.dim(),
                "D");
    TORCH_CHECK(grad_output.size(0) == topo.dst_total_voxels,
                "grad_output.size(0)=",
                grad_output.size(0),
                " must match dst_total_voxels=",
                topo.dst_total_voxels);
    TORCH_CHECK(grad_output.is_floating_point(), "grad_output must be floating point");

    TORCH_CHECK(features.dim() == 2,
                "features must be 2D [src_total_voxels, C_in], got ",
                features.dim(),
                "D");
    TORCH_CHECK(features.size(0) == topo.src_total_voxels,
                "features.size(0)=",
                features.size(0),
                " must match src_total_voxels=",
                topo.src_total_voxels);
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
    TORCH_CHECK(weights.size(2) == topo.kernel_size[0] && weights.size(3) == topo.kernel_size[1] &&
                    weights.size(4) == topo.kernel_size[2],
                "weights spatial dims must match topology kernel_size");
    TORCH_CHECK(weights.is_floating_point(), "weights must be floating point");

    TORCH_CHECK(grad_output.device() == features.device(),
                "grad_output and features must be on the same device");
    TORCH_CHECK(features.device() == weights.device(),
                "features and weights must be on the same device");
    TORCH_CHECK(features.device() == topo.kernel_map.device(),
                "features and kernel_map must be on the same device");

    static auto const table = dispatch_table_from_op<gather_scatter_conv_backward_op>(
        "gather_scatter_sparse_conv_backward");

    auto const dev  = features.device().type();
    auto const f_st = features.scalar_type();

    return table.select(dispatch_set{dev, f_st})(grad_output, features, weights, topo);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
