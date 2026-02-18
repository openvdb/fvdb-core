// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// GatherScatterDefault.cu -- Default gather-scatter sparse convolution.
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
#include <fvdb/detail/dispatch/ForEachActiveVoxel.cuh>
#include <fvdb/detail/dispatch/GridAccessor.h>
#include <fvdb/detail/dispatch/TensorChecks.h>
#include <fvdb/detail/ops/convolution/GatherScatterDefault.h>

#include <torch/torch.h>

#include <cmath>
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

// =============================================================================
// Two-pass topology builder via dispatch framework (CPU + CUDA)
// =============================================================================
//
// Two sweeps over active output voxels via forEachActiveVoxel:
//   Sweep 1: Count active pairs per kernel offset k using atomic counters.
//   Prefix sums: Compute CSR offsets via torch::cumsum.
//   Sweep 2: Fill gather/scatter indices using atomic position assignment.

struct twopass_topology_op {
    template <typename Tag>
        requires with_type<Tag, torch::DeviceType>
    static GatherScatterDefaultTopology
    op(Tag tg,
       GridBatchImpl const &feature_grid,
       GridBatchImpl const &output_grid,
       nanovdb::Coord kernel_size,
       nanovdb::Coord stride,
       ConvDirection direction) {
        int64_t const ks0 = kernel_size[0];
        int64_t const ks1 = kernel_size[1];
        int64_t const ks2 = kernel_size[2];
        int64_t const K   = ks0 * ks1 * ks2;

        int64_t const feature_total = feature_grid.totalVoxels();
        int64_t const output_total  = output_grid.totalVoxels();

        nanovdb::Coord const kernel_start(static_cast<int>(std::floor(-ks0 / 2.0 + 1)),
                                          static_cast<int>(std::floor(-ks1 / 2.0 + 1)),
                                          static_cast<int>(std::floor(-ks2 / 2.0 + 1)));

        bool const is_transposed = (direction == ConvDirection::Transposed);
        auto const device        = output_grid.device();

        // ---- Sweep 1: count pairs per offset k ----
        auto counts = torch::zeros({K}, opts_on<torch::kInt32>(device));
        auto guard  = make_device_guard(tg, counts);

        auto feature_acc = dispatch::make_grid_accessor(tg, feature_grid);
        auto *counts_ptr = counts.data_ptr<int32_t>();

        dispatch::forEachActiveVoxel(
            tg,
            output_grid,
            [=] __hostdev__(Tag tg_inner,
                            JIdxType batch_idx,
                            nanovdb::Coord ijk,
                            int64_t /*voxel_idx*/,
                            GridBatchImpl::Accessor /*output_acc*/) {
                auto const *feat_grid = feature_acc.grid(batch_idx);
                auto feat_tree_acc    = feat_grid->getAccessor();

                for (int64_t k = 0; k < K; ++k) {
                    int32_t const k0 = kernel_start[0] + static_cast<int32_t>(k / (ks1 * ks2));
                    int32_t const k1 = kernel_start[1] + static_cast<int32_t>((k / ks2) % ks1);
                    int32_t const k2 = kernel_start[2] + static_cast<int32_t>(k % ks2);

                    nanovdb::Coord probe;
                    if (is_transposed) {
                        int32_t const r0 = ijk[0] - k0;
                        int32_t const r1 = ijk[1] - k1;
                        int32_t const r2 = ijk[2] - k2;
                        if (r0 % stride[0] != 0 || r1 % stride[1] != 0 || r2 % stride[2] != 0)
                            continue;
                        probe = nanovdb::Coord(r0 / stride[0], r1 / stride[1], r2 / stride[2]);
                    } else {
                        probe = nanovdb::Coord(ijk[0] * stride[0] + k0,
                                               ijk[1] * stride[1] + k1,
                                               ijk[2] * stride[2] + k2);
                    }

                    if (feat_tree_acc.isActive(probe)) {
                        dispatch::atomic_fetch_add_i32(tg_inner, &counts_ptr[k], 1);
                    }
                }
            });

        // ---- Prefix sums (device-generic torch ops) ----
        auto offsets_dev = torch::zeros({K + 1}, opts_on<torch::kInt64>(device));
        if (K > 0) {
            offsets_dev.slice(0, 1, K + 1).copy_(torch::cumsum(counts.to(torch::kInt64), 0));
        }
        int64_t total_pairs = (K > 0) ? offsets_dev[K].item<int64_t>() : 0;
        auto offsets_host   = offsets_dev.cpu();

        if (total_pairs == 0) {
            return GatherScatterDefaultTopology{
                torch::empty({0}, opts_on<torch::kInt32>(device)),
                torch::empty({0}, opts_on<torch::kInt32>(device)),
                offsets_host,
                feature_total,
                output_total,
                K,
                0,
                kernel_size,
                stride,
                direction,
            };
        }

        // ---- Sweep 2: fill gather/scatter indices ----
        auto gather_indices  = torch::empty({total_pairs}, opts_on<torch::kInt32>(device));
        auto scatter_indices = torch::empty({total_pairs}, opts_on<torch::kInt32>(device));
        auto counters        = torch::zeros({K}, opts_on<torch::kInt32>(device));

        auto *counters_ptr = counters.data_ptr<int32_t>();
        auto *offsets_ptr  = offsets_dev.data_ptr<int64_t>();
        auto *gather_ptr   = gather_indices.data_ptr<int32_t>();
        auto *scatter_ptr  = scatter_indices.data_ptr<int32_t>();

        dispatch::forEachActiveVoxel(
            tg,
            output_grid,
            [=] __hostdev__(Tag tg_inner,
                            JIdxType batch_idx,
                            nanovdb::Coord ijk,
                            int64_t voxel_idx,
                            GridBatchImpl::Accessor /*output_acc*/) {
                auto const *feat_grid   = feature_acc.grid(batch_idx);
                auto feat_tree_acc      = feat_grid->getAccessor();
                int64_t const feat_base = feature_acc.voxelOffset(batch_idx);

                for (int64_t k = 0; k < K; ++k) {
                    int32_t const k0 = kernel_start[0] + static_cast<int32_t>(k / (ks1 * ks2));
                    int32_t const k1 = kernel_start[1] + static_cast<int32_t>((k / ks2) % ks1);
                    int32_t const k2 = kernel_start[2] + static_cast<int32_t>(k % ks2);

                    nanovdb::Coord probe;
                    if (is_transposed) {
                        int32_t const r0 = ijk[0] - k0;
                        int32_t const r1 = ijk[1] - k1;
                        int32_t const r2 = ijk[2] - k2;
                        if (r0 % stride[0] != 0 || r1 % stride[1] != 0 || r2 % stride[2] != 0)
                            continue;
                        probe = nanovdb::Coord(r0 / stride[0], r1 / stride[1], r2 / stride[2]);
                    } else {
                        probe = nanovdb::Coord(ijk[0] * stride[0] + k0,
                                               ijk[1] * stride[1] + k1,
                                               ijk[2] * stride[2] + k2);
                    }

                    if (feat_tree_acc.isActive(probe)) {
                        int32_t const feat_flat =
                            static_cast<int32_t>(feat_base + feat_tree_acc.getValue(probe) - 1);
                        int32_t const pos =
                            dispatch::atomic_fetch_add_i32(tg_inner, &counters_ptr[k], 1);
                        int64_t const write_pos = offsets_ptr[k] + pos;
                        gather_ptr[write_pos]   = feat_flat;
                        scatter_ptr[write_pos]  = static_cast<int32_t>(voxel_idx);
                    }
                }
            });

        return GatherScatterDefaultTopology{
            gather_indices,
            scatter_indices,
            offsets_host,
            feature_total,
            output_total,
            K,
            total_pairs,
            kernel_size,
            stride,
            direction,
        };
    }

    using space      = axes<torch_full_device_axis>;
    using subspaces  = coverage<space>;
    using dispatcher = dispatch_table<space,
                                      GatherScatterDefaultTopology(GridBatchImpl const &,
                                                                   GridBatchImpl const &,
                                                                   nanovdb::Coord,
                                                                   nanovdb::Coord,
                                                                   ConvDirection)>;
};

static GatherScatterDefaultTopology
buildTopologyTwoPass(GridBatchImpl const &feature_grid,
                     GridBatchImpl const &output_grid,
                     nanovdb::Coord kernel_size,
                     nanovdb::Coord stride,
                     ConvDirection direction) {
    checkTopologyPreconditions(feature_grid, output_grid, kernel_size, stride);

    static auto const table =
        dispatch_table_from_op<twopass_topology_op>("gather_scatter_default_twopass_topology");

    auto const dev = feature_grid.device().type();
    return table.select(dispatch_set{dev})(
        feature_grid, output_grid, kernel_size, stride, direction);
}

// =============================================================================
// Topology builder entry points (two-pass for all devices)
// =============================================================================

GatherScatterDefaultTopology
gatherScatterDefaultSparseConvTopology(GridBatchImpl const &feature_grid,
                                       GridBatchImpl const &output_grid,
                                       nanovdb::Coord kernel_size,
                                       nanovdb::Coord stride) {
    return buildTopologyTwoPass(
        feature_grid, output_grid, kernel_size, stride, ConvDirection::Forward);
}

GatherScatterDefaultTopology
gatherScatterDefaultSparseConvTransposeTopology(GridBatchImpl const &feature_grid,
                                                GridBatchImpl const &output_grid,
                                                nanovdb::Coord kernel_size,
                                                nanovdb::Coord stride) {
    return buildTopologyTwoPass(
        feature_grid, output_grid, kernel_size, stride, ConvDirection::Transposed);
}

// =============================================================================
// Gather / scatter-add helpers (tag-dispatched via for_each)
// =============================================================================

template <typename Tag>
    requires with_type<Tag, torch::DeviceType> && with_type<Tag, torch::ScalarType>
void
gs_default_gather(Tag tg,
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

template <typename Tag>
    requires with_type<Tag, torch::DeviceType> && with_type<Tag, torch::ScalarType>
void
gs_default_scatter_add(Tag tg,
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
// Type promotion
// =============================================================================

static torch::ScalarType
promoteFloatTypes(torch::ScalarType a, torch::ScalarType b) {
    return at::result_type(torch::empty({0}, torch::dtype(a)), torch::empty({0}, torch::dtype(b)));
}

// =============================================================================
// CPU-safe matrix multiply
// =============================================================================
//
// torch::mm on CPU does not support float16 or bfloat16.  On those types we
// promote to float32, multiply, and demote.  On CUDA the fast path is always
// taken (cuBLAS supports half types natively via tensor cores).

static void
mm_out_safe(torch::Tensor &out, torch::Tensor const &a, torch::Tensor const &b) {
    if (a.is_cpu() && (a.scalar_type() == torch::kFloat16 || a.scalar_type() == torch::kBFloat16)) {
        auto a_f = a.to(torch::kFloat32);
        auto b_f = b.to(torch::kFloat32);
        auto o_f = torch::mm(a_f, b_f);
        out.copy_(o_f.to(out.scalar_type()));
    } else {
        torch::mm_out(out, a, b);
    }
}

// =============================================================================
// Precondition checks
// =============================================================================

static void
checkConvPreconditions(torch::Tensor features,
                       torch::Tensor weights,
                       GatherScatterDefaultTopology const &topo,
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

struct gs_default_conv_op {
    template <typename Tag>
        requires with_type<Tag, torch::DeviceType> && with_type<Tag, torch::ScalarType>
    static torch::Tensor
    op(Tag tg,
       torch::Tensor features,
       torch::Tensor weights,
       GatherScatterDefaultTopology const &topo) {
        constexpr auto dev = tag_get<torch::DeviceType>(Tag{});

        auto guard = make_device_guard(tag<dev>{}, features);

        int64_t const O     = topo.output_total_voxels;
        int64_t const K     = topo.kernel_volume;
        int64_t const C_in  = weights.size(1);
        int64_t const C_out = weights.size(0);
        int64_t const TP    = topo.total_pairs;

        auto W = weights.permute({2, 3, 4, 1, 0}).reshape({K, C_in, C_out}).contiguous();
        if (W.scalar_type() != features.scalar_type()) {
            W = W.to(features.scalar_type());
        }

        auto output = torch::zeros({O, C_out}, features.options());

        if (O == 0 || K == 0 || TP == 0)
            return output;

        auto buf_A = torch::empty({TP, C_in}, features.options());
        gs_default_gather(tg, features, buf_A, topo.gather_indices, TP, C_in);

        auto buf_D   = torch::empty({TP, C_out}, features.options());
        auto off_acc = topo.offsets.accessor<int64_t, 1>();

        for (int64_t k = 0; k < K; ++k) {
            int64_t start = off_acc[k];
            int64_t end   = off_acc[k + 1];

            if (start == end)
                continue;

            auto A_slice = buf_A.slice(0, start, end);
            auto D_slice = buf_D.slice(0, start, end);

            mm_out_safe(D_slice, A_slice, W[k]);
        }

        gs_default_scatter_add(tg, buf_D, output, topo.scatter_indices, TP, C_out);

        return output;
    }

    using space     = axes<torch_full_device_axis, torch_full_float_stype_axis>;
    using subspaces = coverage<space>;
    using dispatcher =
        dispatch_table<space,
                       torch::Tensor(
                           torch::Tensor, torch::Tensor, GatherScatterDefaultTopology const &)>;
};

// =============================================================================
// Backward convolution (forward direction)
// =============================================================================

struct gs_default_conv_backward_op {
    template <typename Tag>
        requires with_type<Tag, torch::DeviceType> && with_type<Tag, torch::ScalarType>
    static std::tuple<torch::Tensor, torch::Tensor>
    op(Tag tg,
       torch::Tensor grad_output,
       torch::Tensor features,
       torch::Tensor weights,
       GatherScatterDefaultTopology const &topo) {
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

        auto grad_W_flat = torch::zeros({K, C_in, C_out}, features.options());

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
        gs_default_gather(tg, features, feat_buf, topo.gather_indices, TP, C_in);
        gs_default_gather(tg, grad_output, grad_buf, topo.scatter_indices, TP, C_out);

        auto grad_feat_buf = torch::empty({TP, C_in}, features.options());
        for (int64_t k = 0; k < K; ++k) {
            int64_t start = off_acc[k];
            int64_t end   = off_acc[k + 1];
            if (start == end)
                continue;

            auto grad_slice = grad_buf.slice(0, start, end);
            auto gf_slice   = grad_feat_buf.slice(0, start, end);
            mm_out_safe(gf_slice, grad_slice, W[k].t());
        }

        gs_default_scatter_add(tg, grad_feat_buf, grad_features, topo.gather_indices, TP, C_in);

        for (int64_t k = 0; k < K; ++k) {
            int64_t start = off_acc[k];
            int64_t end   = off_acc[k + 1];
            if (start == end)
                continue;

            auto feat_slice = feat_buf.slice(0, start, end);
            auto grad_slice = grad_buf.slice(0, start, end);
            auto gw_slice   = grad_W_flat[k];
            mm_out_safe(gw_slice, feat_slice.t(), grad_slice);
        }

        auto ks           = topo.kernel_size;
        auto grad_weights = grad_W_flat.reshape({ks[0], ks[1], ks[2], C_in, C_out})
                                .permute({4, 3, 0, 1, 2})
                                .contiguous();

        return {grad_features, grad_weights};
    }

    using space      = axes<torch_full_device_axis, torch_full_float_stype_axis>;
    using subspaces  = coverage<space>;
    using dispatcher = dispatch_table<
        space,
        std::tuple<torch::Tensor, torch::Tensor>(
            torch::Tensor, torch::Tensor, torch::Tensor, GatherScatterDefaultTopology const &)>;
};

// =============================================================================
// Backward convolution (transposed direction)
// =============================================================================

struct gs_default_conv_transpose_backward_op {
    template <typename Tag>
        requires with_type<Tag, torch::DeviceType> && with_type<Tag, torch::ScalarType>
    static std::tuple<torch::Tensor, torch::Tensor>
    op(Tag tg,
       torch::Tensor grad_output,
       torch::Tensor features,
       torch::Tensor weights,
       GatherScatterDefaultTopology const &topo) {
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
        gs_default_gather(tg, features, feat_buf, topo.gather_indices, TP, C_in);
        gs_default_gather(tg, grad_output, grad_buf, topo.scatter_indices, TP, C_out);

        auto grad_feat_buf = torch::empty({TP, C_in}, features.options());

        for (int64_t k = 0; k < K; ++k) {
            int64_t start = off_acc[k];
            int64_t end   = off_acc[k + 1];
            if (start == end)
                continue;

            auto gd_slice = grad_buf.slice(0, start, end);
            auto gf_slice = grad_feat_buf.slice(0, start, end);
            mm_out_safe(gf_slice, gd_slice, W[k].t());

            auto fb_slice = feat_buf.slice(0, start, end);
            auto gw_slice = grad_W_flat[k];
            mm_out_safe(gw_slice, fb_slice.t(), gd_slice);
        }

        gs_default_scatter_add(tg, grad_feat_buf, grad_features, topo.gather_indices, TP, C_in);

        auto ks           = topo.kernel_size;
        auto grad_weights = grad_W_flat.reshape({ks[0], ks[1], ks[2], C_in, C_out})
                                .permute({4, 3, 0, 1, 2})
                                .contiguous();

        return {grad_features, grad_weights};
    }

    using space      = axes<torch_full_device_axis, torch_full_float_stype_axis>;
    using subspaces  = coverage<space>;
    using dispatcher = dispatch_table<
        space,
        std::tuple<torch::Tensor, torch::Tensor>(
            torch::Tensor, torch::Tensor, torch::Tensor, GatherScatterDefaultTopology const &)>;
};

// =============================================================================
// Type-erased entry points
// =============================================================================

torch::Tensor
gatherScatterDefaultSparseConv(torch::Tensor features,
                               torch::Tensor weights,
                               GatherScatterDefaultTopology const &topo) {
    checkConvPreconditions(features, weights, topo, "gatherScatterDefaultSparseConv");
    TORCH_CHECK(topo.direction == ConvDirection::Forward,
                "gatherScatterDefaultSparseConv requires topology with direction=Forward");

    auto working_st = promoteFloatTypes(features.scalar_type(), weights.scalar_type());
    if (features.scalar_type() != working_st)
        features = features.to(working_st);

    static auto const table =
        dispatch_table_from_op<gs_default_conv_op>("gather_scatter_default_sparse_conv");

    auto const dev = features.device().type();
    return table.select(dispatch_set{dev, working_st})(features, weights, topo);
}

std::tuple<torch::Tensor, torch::Tensor>
gatherScatterDefaultSparseConvBackward(torch::Tensor grad_output,
                                       torch::Tensor features,
                                       torch::Tensor weights,
                                       GatherScatterDefaultTopology const &topo) {
    checkConvPreconditions(features, weights, topo, "gatherScatterDefaultSparseConvBackward");
    TORCH_CHECK(topo.direction == ConvDirection::Forward,
                "gatherScatterDefaultSparseConvBackward requires topology with direction=Forward");
    TORCH_CHECK(grad_output.dim() == 2 && grad_output.size(0) == topo.output_total_voxels,
                "grad_output shape mismatch");
    TORCH_CHECK(grad_output.is_contiguous(), "grad_output must be contiguous");
    TORCH_CHECK(grad_output.is_floating_point(), "grad_output must be floating point");

    auto working_st = promoteFloatTypes(features.scalar_type(), weights.scalar_type());
    if (features.scalar_type() != working_st)
        features = features.to(working_st);
    if (grad_output.scalar_type() != working_st)
        grad_output = grad_output.to(working_st);

    static auto const table = dispatch_table_from_op<gs_default_conv_backward_op>(
        "gather_scatter_default_sparse_conv_backward");

    auto const dev = features.device().type();
    return table.select(dispatch_set{dev, working_st})(grad_output, features, weights, topo);
}

torch::Tensor
gatherScatterDefaultSparseConvTranspose(torch::Tensor features,
                                        torch::Tensor weights,
                                        GatherScatterDefaultTopology const &topo) {
    checkConvPreconditions(features, weights, topo, "gatherScatterDefaultSparseConvTranspose");
    TORCH_CHECK(
        topo.direction == ConvDirection::Transposed,
        "gatherScatterDefaultSparseConvTranspose requires topology with direction=Transposed");

    auto working_st = promoteFloatTypes(features.scalar_type(), weights.scalar_type());
    if (features.scalar_type() != working_st)
        features = features.to(working_st);

    static auto const table =
        dispatch_table_from_op<gs_default_conv_op>("gather_scatter_default_sparse_conv_transpose");

    auto const dev = features.device().type();
    return table.select(dispatch_set{dev, working_st})(features, weights, topo);
}

std::tuple<torch::Tensor, torch::Tensor>
gatherScatterDefaultSparseConvTransposeBackward(torch::Tensor grad_output,
                                                torch::Tensor features,
                                                torch::Tensor weights,
                                                GatherScatterDefaultTopology const &topo) {
    checkConvPreconditions(
        features, weights, topo, "gatherScatterDefaultSparseConvTransposeBackward");
    TORCH_CHECK(topo.direction == ConvDirection::Transposed,
                "gatherScatterDefaultSparseConvTransposeBackward requires direction=Transposed");
    TORCH_CHECK(grad_output.dim() == 2 && grad_output.size(0) == topo.output_total_voxels,
                "grad_output shape mismatch");
    TORCH_CHECK(grad_output.is_contiguous(), "grad_output must be contiguous");
    TORCH_CHECK(grad_output.is_floating_point(), "grad_output must be floating point");

    auto working_st = promoteFloatTypes(features.scalar_type(), weights.scalar_type());
    if (features.scalar_type() != working_st)
        features = features.to(working_st);
    if (grad_output.scalar_type() != working_st)
        grad_output = grad_output.to(working_st);

    static auto const table = dispatch_table_from_op<gs_default_conv_transpose_backward_op>(
        "gather_scatter_default_sparse_conv_transpose_backward");

    auto const dev = features.device().type();
    return table.select(dispatch_set{dev, working_st})(grad_output, features, weights, topo);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
