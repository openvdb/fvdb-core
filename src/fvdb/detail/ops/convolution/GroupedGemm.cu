// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// GroupedGemm.cu -- Compacted Gather-Scatter sparse convolution implementation.
//
// This approach maintains the algorithmic performance characteristics of a Grouped GEMM
// (only computing on valid (voxel, kernel_offset) pairs to avoid zero-padding)
// but leverages PyTorch's highly optimized, robust cuBLAS `torch::mm` over sliced buffers.
//
// Three phases per convolution:
//   1. Gather:       features[gather_indices[i]] -> contiguous buffer       (1 launch)
//   2. Compact GEMM: Loop over K, slice buffer, torch::mm                   (K launches)
//   3. Scatter-add:  result buffer -> output[scatter_indices[i]] (atomic)   (1 launch)
//

#include "dispatch/detail/core_types.h"
#include "dispatch/dispatch_table.h"
#include "dispatch/torch/dispatch.h"
#include "dispatch/torch/for_each.h"

#include <fvdb/detail/dispatch/AtomicAdd.cuh>
#include <fvdb/detail/ops/convolution/GatherScatter.h>
#include <fvdb/detail/ops/convolution/GroupedGemm.h>

#include <torch/torch.h>

#include <cstdint>
#include <tuple>

namespace fvdb {
namespace detail {
namespace ops {

using namespace ::dispatch;

// =============================================================================
// Gather / scatter-add helpers (tag-dispatched via for_each)
// =============================================================================

// Gather: dst[i * C + c] = src[indices[i] * C + c]
template <typename Tag>
    requires with_type<Tag, torch::DeviceType> && with_type<Tag, torch::ScalarType>
void
grouped_gemm_gather(Tag tg,
                    torch::Tensor src,
                    torch::Tensor dst,
                    torch::Tensor indices,
                    int64_t total_pairs,
                    int64_t C) {
    if (total_pairs == 0)
        return;

    constexpr auto stype = tag_get<torch::ScalarType>(tg);
    using scalar_t       = torch_scalar_cpp_type_t<stype>;

    auto const *src_ptr = src.data_ptr<scalar_t>();
    auto *dst_ptr       = dst.data_ptr<scalar_t>();
    auto const *idx_ptr = indices.data_ptr<int32_t>();

    for_each(tg, total_pairs * C, [=] __hostdev__(Tag, int64_t idx) {
        int64_t const pair    = idx / C;
        int64_t const c       = idx % C;
        int32_t const src_row = idx_ptr[pair];
        dst_ptr[pair * C + c] = src_ptr[static_cast<int64_t>(src_row) * C + c];
    });
}

// Scatter-add: dst[indices[i] * C + c] += src[i * C + c]
template <typename Tag>
    requires with_type<Tag, torch::DeviceType> && with_type<Tag, torch::ScalarType>
void
grouped_gemm_scatter_add(Tag tg,
                         torch::Tensor src,
                         torch::Tensor dst,
                         torch::Tensor indices,
                         int64_t total_pairs,
                         int64_t C) {
    if (total_pairs == 0)
        return;

    constexpr auto stype = tag_get<torch::ScalarType>(tg);
    using scalar_t       = torch_scalar_cpp_type_t<stype>;

    auto const *src_ptr = src.data_ptr<scalar_t>();
    auto *dst_ptr       = dst.data_ptr<scalar_t>();
    auto const *idx_ptr = indices.data_ptr<int32_t>();

    for_each(tg, total_pairs * C, [=] __hostdev__(Tag tg_inner, int64_t idx) {
        int64_t const pair    = idx / C;
        int64_t const c       = idx % C;
        int32_t const dst_row = idx_ptr[pair];
        dispatch::atomic_add(
            tg_inner, &dst_ptr[static_cast<int64_t>(dst_row) * C + c], src_ptr[pair * C + c]);
    });
}

// =============================================================================
// Topology compaction
// =============================================================================

static GroupedGemmTopology
compactTopology(GatherScatterTopology const &dense_topo) {
    auto const &kmap = dense_topo.kernel_map; // [O, K] int32 on device
    int64_t const O  = dense_topo.output_total_voxels;
    int64_t const K  = dense_topo.kernel_volume;

    // Transpose to [K, O] so nonzero() iterates k-major (groups are contiguous)
    auto kmap_t = kmap.t().contiguous(); // [K, O]
    auto mask   = kmap_t != -1;          // [K, O] bool

    // Per-offset pair counts and cumulative offsets
    auto sizes = torch::sum(mask, /*dim=*/-1, /*keepdim=*/false, torch::kInt64); // [K] on device
    auto sizes_cpu = sizes.cpu().contiguous();                         // [K] int64 on host

    auto offsets = torch::zeros({K + 1}, torch::dtype(torch::kInt64)); // host
    auto off_acc = offsets.accessor<int64_t, 1>();
    auto sz_acc  = sizes_cpu.accessor<int64_t, 1>();
    for (int64_t k = 0; k < K; ++k) {
        off_acc[k + 1] = off_acc[k] + sz_acc[k];
    }
    int64_t total_pairs = off_acc[K];

    if (total_pairs == 0) {
        return GroupedGemmTopology{
            torch::empty({0}, torch::dtype(torch::kInt32).device(kmap.device())),
            torch::empty({0}, torch::dtype(torch::kInt32).device(kmap.device())),
            offsets,
            dense_topo.feature_total_voxels,
            dense_topo.output_total_voxels,
            K,
            0,
            dense_topo.kernel_size,
            dense_topo.stride,
            dense_topo.direction,
        };
    }

    // Find all active (k, o) pairs -- sorted by k because nonzero iterates row-major on [K, O]
    auto pairs = torch::nonzero(mask).contiguous(); // [total_pairs, 2] int64
    auto k_col = pairs.select(1, 0);                // [total_pairs] -- kernel offset indices
    auto o_col = pairs.select(1, 1);                // [total_pairs] -- output voxel indices

    // Look up the feature voxel index for each pair from kmap_t[k, o]
    auto flat_idx        = k_col * O + o_col;
    auto gather_indices  = kmap_t.reshape({-1}).index({flat_idx}).to(torch::kInt32).contiguous();
    auto scatter_indices = o_col.to(torch::kInt32).contiguous();

    return GroupedGemmTopology{
        gather_indices,
        scatter_indices,
        offsets,
        dense_topo.feature_total_voxels,
        dense_topo.output_total_voxels,
        K,
        total_pairs,
        dense_topo.kernel_size,
        dense_topo.stride,
        dense_topo.direction,
    };
}

// =============================================================================
// Topology builder entry points
// =============================================================================

GroupedGemmTopology
groupedGemmSparseConvTopology(GridBatchImpl const &feature_grid,
                              GridBatchImpl const &output_grid,
                              nanovdb::Coord kernel_size,
                              nanovdb::Coord stride) {
    auto dense = gatherScatterSparseConvTopology(feature_grid, output_grid, kernel_size, stride);
    return compactTopology(dense);
}

GroupedGemmTopology
groupedGemmSparseConvTransposeTopology(GridBatchImpl const &feature_grid,
                                       GridBatchImpl const &output_grid,
                                       nanovdb::Coord kernel_size,
                                       nanovdb::Coord stride) {
    auto dense =
        gatherScatterSparseConvTransposeTopology(feature_grid, output_grid, kernel_size, stride);
    return compactTopology(dense);
}

// =============================================================================
// Precondition checks
// =============================================================================

static void
checkGroupedGemmPreconditions(torch::Tensor features,
                              torch::Tensor weights,
                              GroupedGemmTopology const &topo,
                              char const *name) {
    TORCH_CHECK(features.dim() == 2, name, ": features must be 2D");
    TORCH_CHECK(features.size(0) == topo.feature_total_voxels,
                name,
                ": features.size(0)=",
                features.size(0),
                " must match feature_total_voxels=",
                topo.feature_total_voxels);
    TORCH_CHECK(features.is_floating_point(), name, ": features must be floating point");
    TORCH_CHECK(features.is_contiguous(), name, ": features must be contiguous");

    TORCH_CHECK(weights.dim() == 5, name, ": weights must be 5D [C_out, C_in, k0, k1, k2]");
    TORCH_CHECK(weights.is_floating_point(), name, ": weights must be floating point");
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

    TORCH_CHECK(features.device() == weights.device(),
                name,
                ": features and weights must be on the same device");
}

// =============================================================================
// Forward convolution (shared by forward and transposed)
// =============================================================================

struct grouped_gemm_conv_op {
    template <typename Tag>
        requires with_type<Tag, torch::DeviceType> && with_type<Tag, torch::ScalarType>
    static torch::Tensor
    op(Tag tg, torch::Tensor features, torch::Tensor weights, GroupedGemmTopology const &topo) {
        constexpr auto dev = tag_get<torch::DeviceType>(Tag{});

        auto guard = make_device_guard(tag<dev>{}, features);

        int64_t const O     = topo.output_total_voxels;
        int64_t const K     = topo.kernel_volume;
        int64_t const C_in  = weights.size(1);
        int64_t const C_out = weights.size(0);
        int64_t const TP    = topo.total_pairs;

        // Reshape weights: [C_out, C_in, k0, k1, k2] -> [K, C_in, C_out] RowMajor contiguous
        auto W = weights.permute({2, 3, 4, 1, 0}).reshape({K, C_in, C_out}).contiguous();
        if (W.scalar_type() != features.scalar_type()) {
            W = W.to(features.scalar_type());
        }

        auto output = torch::zeros({O, C_out}, features.options());

        if (O == 0 || K == 0 || TP == 0)
            return output;

        // Phase 1: Gather features into contiguous buffer [total_pairs, C_in]
        auto buf_A = torch::empty({TP, C_in}, features.options());
        grouped_gemm_gather(tg, features, buf_A, topo.gather_indices, TP, C_in);

        // Phase 2: PyTorch mm_out over K valid slice blocks
        auto buf_D   = torch::empty({TP, C_out}, features.options());
        auto off_acc = topo.offsets.accessor<int64_t, 1>();

        for (int64_t k = 0; k < K; ++k) {
            int64_t start = off_acc[k];
            int64_t end   = off_acc[k + 1];

            // Fully bypass boundary matrices where problem boundaries evaluate to 0
            if (start == end)
                continue;

            auto A_slice = buf_A.slice(0, start, end);
            auto D_slice = buf_D.slice(0, start, end);

            torch::mm_out(D_slice, A_slice, W[k]);
        }

        // Phase 3: Scatter-add result into output
        grouped_gemm_scatter_add(tg, buf_D, output, topo.scatter_indices, TP, C_out);

        return output;
    }

    using space     = axes<torch_full_device_axis, torch_builtin_float_stype_axis>;
    using subspaces = coverage<space>;
    using dispatcher =
        dispatch_table<space,
                       torch::Tensor(torch::Tensor, torch::Tensor, GroupedGemmTopology const &)>;
};

// =============================================================================
// Backward convolution (forward direction)
// =============================================================================

struct grouped_gemm_conv_backward_op {
    template <typename Tag>
        requires with_type<Tag, torch::DeviceType> && with_type<Tag, torch::ScalarType>
    static std::tuple<torch::Tensor, torch::Tensor>
    op(Tag tg,
       torch::Tensor grad_output,
       torch::Tensor features,
       torch::Tensor weights,
       GroupedGemmTopology const &topo) {
        constexpr auto dev = tag_get<torch::DeviceType>(Tag{});

        auto guard = make_device_guard(tag<dev>{}, features);

        int64_t const F     = topo.feature_total_voxels;
        int64_t const O     = topo.output_total_voxels;
        int64_t const K     = topo.kernel_volume;
        int64_t const C_in  = weights.size(1);
        int64_t const C_out = weights.size(0);
        int64_t const TP    = topo.total_pairs;

        // Reshape weights: [C_out, C_in, k0, k1, k2] -> [K, C_in, C_out]
        auto W = weights.permute({2, 3, 4, 1, 0}).reshape({K, C_in, C_out}).contiguous();
        if (W.scalar_type() != features.scalar_type()) {
            W = W.to(features.scalar_type());
        }

        auto grad_features = torch::zeros({F, C_in}, features.options());

        // Zeros initialization correctly accommodates skipped 0-bound operations natively
        auto grad_W_flat = torch::zeros({K, C_in, C_out}, features.options());

        if (O == 0 || K == 0 || TP == 0) {
            auto ks           = topo.kernel_size;
            auto grad_weights = grad_W_flat.reshape({ks[0], ks[1], ks[2], C_in, C_out})
                                    .permute({4, 3, 0, 1, 2})
                                    .contiguous();
            return {grad_features, grad_weights};
        }

        auto off_acc = topo.offsets.accessor<int64_t, 1>();

        // Gather features and grad_output into contiguous buffers
        auto feat_buf = torch::empty({TP, C_in}, features.options());
        auto grad_buf = torch::empty({TP, C_out}, features.options());
        grouped_gemm_gather(tg, features, feat_buf, topo.gather_indices, TP, C_in);
        grouped_gemm_gather(tg, grad_output, grad_buf, topo.scatter_indices, TP, C_out);

        // --- grad_features: grad_buf @ W[k]^T -> grad_feat_buf ---
        auto grad_feat_buf = torch::empty({TP, C_in}, features.options());
        for (int64_t k = 0; k < K; ++k) {
            int64_t start = off_acc[k];
            int64_t end   = off_acc[k + 1];
            if (start == end)
                continue;

            auto grad_slice = grad_buf.slice(0, start, end);
            auto gf_slice   = grad_feat_buf.slice(0, start, end);
            torch::mm_out(gf_slice, grad_slice, W[k].t());
        }

        // Scatter-add grad_feat_buf into grad_features using gather_indices
        grouped_gemm_scatter_add(tg, grad_feat_buf, grad_features, topo.gather_indices, TP, C_in);

        // --- grad_weights: feat_buf^T @ grad_buf -> grad_W[k] ---
        for (int64_t k = 0; k < K; ++k) {
            int64_t start = off_acc[k];
            int64_t end   = off_acc[k + 1];
            if (start == end)
                continue;

            auto feat_slice = feat_buf.slice(0, start, end);
            auto grad_slice = grad_buf.slice(0, start, end);
            auto gw_slice   = grad_W_flat[k]; // shape [C_in, C_out]
            torch::mm_out(gw_slice, feat_slice.t(), grad_slice);
        }

        // Reshape grad_W from [K, C_in, C_out] to [C_out, C_in, k0, k1, k2]
        auto ks           = topo.kernel_size;
        auto grad_weights = grad_W_flat.reshape({ks[0], ks[1], ks[2], C_in, C_out})
                                .permute({4, 3, 0, 1, 2})
                                .contiguous();

        return {grad_features, grad_weights};
    }

    using space      = axes<torch_full_device_axis, torch_builtin_float_stype_axis>;
    using subspaces  = coverage<space>;
    using dispatcher = dispatch_table<
        space,
        std::tuple<torch::Tensor, torch::Tensor>(
            torch::Tensor, torch::Tensor, torch::Tensor, GroupedGemmTopology const &)>;
};

// =============================================================================
// Backward convolution (transposed direction)
// =============================================================================
//
// Identical to the forward backward except that both grad_features and
// grad_weights GEMMs are fused into a single loop over K.

struct grouped_gemm_conv_transpose_backward_op {
    template <typename Tag>
        requires with_type<Tag, torch::DeviceType> && with_type<Tag, torch::ScalarType>
    static std::tuple<torch::Tensor, torch::Tensor>
    op(Tag tg,
       torch::Tensor grad_output,
       torch::Tensor features,
       torch::Tensor weights,
       GroupedGemmTopology const &topo) {
        constexpr auto dev = tag_get<torch::DeviceType>(Tag{});

        auto guard = make_device_guard(tag<dev>{}, features);

        int64_t const F     = topo.feature_total_voxels;
        int64_t const O     = topo.output_total_voxels;
        int64_t const K     = topo.kernel_volume;
        int64_t const C_in  = weights.size(1);
        int64_t const C_out = weights.size(0);
        int64_t const TP    = topo.total_pairs;

        auto W = weights.permute({2, 3, 4, 1, 0}).reshape({K, C_in, C_out}).contiguous();
        if (W.scalar_type() != features.scalar_type()) {
            W = W.to(features.scalar_type());
        }

        auto grad_features = torch::zeros({F, C_in}, features.options());
        auto grad_W_flat   = torch::zeros({K, C_in, C_out}, features.options());

        if (O == 0 || K == 0 || TP == 0) {
            auto ks           = topo.kernel_size;
            auto grad_weights = grad_W_flat.reshape({ks[0], ks[1], ks[2], C_in, C_out})
                                    .permute({4, 3, 0, 1, 2})
                                    .contiguous();
            return {grad_features, grad_weights};
        }

        auto off_acc = topo.offsets.accessor<int64_t, 1>();

        auto feat_buf = torch::empty({TP, C_in}, features.options());
        auto grad_buf = torch::empty({TP, C_out}, features.options());
        grouped_gemm_gather(tg, features, feat_buf, topo.gather_indices, TP, C_in);
        grouped_gemm_gather(tg, grad_output, grad_buf, topo.scatter_indices, TP, C_out);

        auto grad_feat_buf = torch::empty({TP, C_in}, features.options());

        for (int64_t k = 0; k < K; ++k) {
            int64_t start = off_acc[k];
            int64_t end   = off_acc[k + 1];
            if (start == end)
                continue;

            auto gd_slice = grad_buf.slice(0, start, end);
            auto gf_slice = grad_feat_buf.slice(0, start, end);
            torch::mm_out(gf_slice, gd_slice, W[k].t());

            auto fb_slice = feat_buf.slice(0, start, end);
            auto gw_slice = grad_W_flat[k];
            torch::mm_out(gw_slice, fb_slice.t(), gd_slice);
        }

        grouped_gemm_scatter_add(tg, grad_feat_buf, grad_features, topo.gather_indices, TP, C_in);

        auto ks           = topo.kernel_size;
        auto grad_weights = grad_W_flat.reshape({ks[0], ks[1], ks[2], C_in, C_out})
                                .permute({4, 3, 0, 1, 2})
                                .contiguous();

        return {grad_features, grad_weights};
    }

    using space      = axes<torch_full_device_axis, torch_builtin_float_stype_axis>;
    using subspaces  = coverage<space>;
    using dispatcher = dispatch_table<
        space,
        std::tuple<torch::Tensor, torch::Tensor>(
            torch::Tensor, torch::Tensor, torch::Tensor, GroupedGemmTopology const &)>;
};

// =============================================================================
// Type-erased entry points
// =============================================================================

torch::Tensor
groupedGemmSparseConv(torch::Tensor features,
                      torch::Tensor weights,
                      GroupedGemmTopology const &topo) {
    checkGroupedGemmPreconditions(features, weights, topo, "groupedGemmSparseConv");
    TORCH_CHECK(topo.direction == ConvDirection::Forward,
                "groupedGemmSparseConv requires topology with direction=Forward");

    static auto const table =
        dispatch_table_from_op<grouped_gemm_conv_op>("grouped_gemm_sparse_conv");

    auto const dev  = features.device().type();
    auto const f_st = features.scalar_type();

    return table.select(dispatch_set{dev, f_st})(features, weights, topo);
}

std::tuple<torch::Tensor, torch::Tensor>
groupedGemmSparseConvBackward(torch::Tensor grad_output,
                              torch::Tensor features,
                              torch::Tensor weights,
                              GroupedGemmTopology const &topo) {
    checkGroupedGemmPreconditions(features, weights, topo, "groupedGemmSparseConvBackward");
    TORCH_CHECK(topo.direction == ConvDirection::Forward,
                "groupedGemmSparseConvBackward requires topology with direction=Forward");
    TORCH_CHECK(grad_output.dim() == 2 && grad_output.size(0) == topo.output_total_voxels,
                "grad_output shape mismatch");
    TORCH_CHECK(grad_output.is_contiguous(), "grad_output must be contiguous");
    TORCH_CHECK(grad_output.is_floating_point(), "grad_output must be floating point");

    static auto const table =
        dispatch_table_from_op<grouped_gemm_conv_backward_op>("grouped_gemm_sparse_conv_backward");

    auto const dev  = features.device().type();
    auto const f_st = features.scalar_type();

    return table.select(dispatch_set{dev, f_st})(grad_output, features, weights, topo);
}

torch::Tensor
groupedGemmSparseConvTranspose(torch::Tensor features,
                               torch::Tensor weights,
                               GroupedGemmTopology const &topo) {
    checkGroupedGemmPreconditions(features, weights, topo, "groupedGemmSparseConvTranspose");
    TORCH_CHECK(topo.direction == ConvDirection::Transposed,
                "groupedGemmSparseConvTranspose requires topology with direction=Transposed");

    static auto const table =
        dispatch_table_from_op<grouped_gemm_conv_op>("grouped_gemm_sparse_conv_transpose");

    auto const dev  = features.device().type();
    auto const f_st = features.scalar_type();

    return table.select(dispatch_set{dev, f_st})(features, weights, topo);
}

std::tuple<torch::Tensor, torch::Tensor>
groupedGemmSparseConvTransposeBackward(torch::Tensor grad_output,
                                       torch::Tensor features,
                                       torch::Tensor weights,
                                       GroupedGemmTopology const &topo) {
    checkGroupedGemmPreconditions(
        features, weights, topo, "groupedGemmSparseConvTransposeBackward");
    TORCH_CHECK(topo.direction == ConvDirection::Transposed,
                "groupedGemmSparseConvTransposeBackward requires direction=Transposed");
    TORCH_CHECK(grad_output.dim() == 2 && grad_output.size(0) == topo.output_total_voxels,
                "grad_output shape mismatch");
    TORCH_CHECK(grad_output.is_contiguous(), "grad_output must be contiguous");
    TORCH_CHECK(grad_output.is_floating_point(), "grad_output must be floating point");

    static auto const table = dispatch_table_from_op<grouped_gemm_conv_transpose_backward_op>(
        "grouped_gemm_sparse_conv_transpose_backward");

    auto const dev  = features.device().type();
    auto const f_st = features.scalar_type();

    return table.select(dispatch_set{dev, f_st})(grad_output, features, weights, topo);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
