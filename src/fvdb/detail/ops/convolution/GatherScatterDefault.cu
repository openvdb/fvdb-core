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
#include "dispatch/torch/views.h"

#include <fvdb/detail/dispatch/AtomicAdd.cuh>
#include <fvdb/detail/dispatch/ForEachActiveVoxel.cuh>
#include <fvdb/detail/dispatch/GridAccessor.h>
#include <fvdb/detail/dispatch/TensorChecks.h>
#include <fvdb/detail/ops/convolution/GatherScatterDefault.h>

#include <torch/torch.h>

#include <cstdint>
#include <tuple>

namespace fvdb {
namespace detail {
namespace ops {

using namespace ::dispatch;

// =============================================================================
// Internal dense topology (intermediate representation for compaction)
// =============================================================================

struct DenseTopology {
    torch::Tensor kernel_map; // [output_total_voxels, kernel_volume] int32
    int64_t feature_total_voxels;
    int64_t output_total_voxels;
    int64_t kernel_volume;
    nanovdb::Coord kernel_size;
    nanovdb::Coord stride;
    ConvDirection direction;
};

// =============================================================================
// Tensor options helpers
// =============================================================================

template <torch::ScalarType Stype>
inline torch::TensorOptions
opts_on(torch::Device device) {
    return torch::dtype(Stype).device(device);
}

// =============================================================================
// Dense topology builder via dispatch table
// =============================================================================
//
// For each active voxel in output_grid, probe the feature_grid at every kernel
// offset and record the flat feature voxel index (or -1).
//
// Forward:    probe = output_ijk * stride + kernel_offset
// Transposed: probe = (output_ijk - kernel_offset) / stride
//             (only when divisible by stride in all 3 dimensions)

struct dense_topology_op {
    template <typename Tag>
        requires with_type<Tag, torch::DeviceType>
    static DenseTopology
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

        nanovdb::Coord const kernel_start(static_cast<int>(std::floor(-ks0 / 2.0 + 1)),
                                          static_cast<int>(std::floor(-ks1 / 2.0 + 1)),
                                          static_cast<int>(std::floor(-ks2 / 2.0 + 1)));

        auto kmap = torch::full(
            {output_total, kernel_volume}, -1, opts_on<torch::kInt32>(output_grid.device()));
        auto kmap_out = tensor_out<dev, torch::kInt32, 2, contiguity::contiguous>(kmap);

        auto guard       = make_device_guard(tg, kmap);
        auto feature_acc = dispatch::make_grid_accessor(tg, feature_grid);

        bool const is_transposed = (direction == ConvDirection::Transposed);

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

                            if (feat_tree_acc.isActive(probe)) {
                                int32_t const feat_flat = static_cast<int32_t>(
                                    feat_base_offset + feat_tree_acc.getValue(probe) - 1);
                                kmap_out(output_voxel_index, k_idx) = feat_flat;
                            }
                        }
                    }
                }
            });

        return DenseTopology{
            /*.kernel_map             =*/kmap,
            /*.feature_total_voxels   =*/feature_total,
            /*.output_total_voxels    =*/output_total,
            /*.kernel_volume          =*/kernel_volume,
            /*.kernel_size            =*/kernel_size,
            /*.stride                 =*/stride,
            /*.direction              =*/direction,
        };
    }

    using space      = axes<torch_full_device_axis>;
    using subspaces  = coverage<space>;
    using dispatcher = dispatch_table<space,
                                      DenseTopology(GridBatchImpl const &,
                                                    GridBatchImpl const &,
                                                    nanovdb::Coord,
                                                    nanovdb::Coord,
                                                    ConvDirection)>;
};

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

static DenseTopology
buildDenseTopology(GridBatchImpl const &feature_grid,
                   GridBatchImpl const &output_grid,
                   nanovdb::Coord kernel_size,
                   nanovdb::Coord stride,
                   ConvDirection direction) {
    checkTopologyPreconditions(feature_grid, output_grid, kernel_size, stride);

    static auto const table =
        dispatch_table_from_op<dense_topology_op>("gather_scatter_default_dense_topology");

    auto const dev = feature_grid.device().type();
    return table.select(dispatch_set{dev})(
        feature_grid, output_grid, kernel_size, stride, direction);
}

// =============================================================================
// Topology compaction (DenseTopology -> GatherScatterDefaultTopology)
// =============================================================================

static GatherScatterDefaultTopology
compactTopology(DenseTopology const &dense_topo) {
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
        return GatherScatterDefaultTopology{
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

    return GatherScatterDefaultTopology{
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

GatherScatterDefaultTopology
gatherScatterDefaultSparseConvTopology(GridBatchImpl const &feature_grid,
                                       GridBatchImpl const &output_grid,
                                       nanovdb::Coord kernel_size,
                                       nanovdb::Coord stride) {
    auto dense =
        buildDenseTopology(feature_grid, output_grid, kernel_size, stride, ConvDirection::Forward);
    return compactTopology(dense);
}

GatherScatterDefaultTopology
gatherScatterDefaultSparseConvTransposeTopology(GridBatchImpl const &feature_grid,
                                                GridBatchImpl const &output_grid,
                                                nanovdb::Coord kernel_size,
                                                nanovdb::Coord stride) {
    auto dense = buildDenseTopology(
        feature_grid, output_grid, kernel_size, stride, ConvDirection::Transposed);
    return compactTopology(dense);
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

            torch::mm_out(D_slice, A_slice, W[k]);
        }

        gs_default_scatter_add(tg, buf_D, output, topo.scatter_indices, TP, C_out);

        return output;
    }

    using space     = axes<torch_full_device_axis, torch_builtin_float_stype_axis>;
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
            torch::mm_out(gf_slice, grad_slice, W[k].t());
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
            torch::mm_out(gw_slice, feat_slice.t(), grad_slice);
        }

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
            torch::mm_out(gf_slice, gd_slice, W[k].t());

            auto fb_slice = feat_buf.slice(0, start, end);
            auto gw_slice = grad_W_flat[k];
            torch::mm_out(gw_slice, fb_slice.t(), gd_slice);
        }

        gs_default_scatter_add(tg, grad_feat_buf, grad_features, topo.gather_indices, TP, C_in);

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

    static auto const table =
        dispatch_table_from_op<gs_default_conv_op>("gather_scatter_default_sparse_conv");

    auto const dev  = features.device().type();
    auto const f_st = features.scalar_type();

    return table.select(dispatch_set{dev, f_st})(features, weights, topo);
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

    static auto const table = dispatch_table_from_op<gs_default_conv_backward_op>(
        "gather_scatter_default_sparse_conv_backward");

    auto const dev  = features.device().type();
    auto const f_st = features.scalar_type();

    return table.select(dispatch_set{dev, f_st})(grad_output, features, weights, topo);
}

torch::Tensor
gatherScatterDefaultSparseConvTranspose(torch::Tensor features,
                                        torch::Tensor weights,
                                        GatherScatterDefaultTopology const &topo) {
    checkConvPreconditions(features, weights, topo, "gatherScatterDefaultSparseConvTranspose");
    TORCH_CHECK(
        topo.direction == ConvDirection::Transposed,
        "gatherScatterDefaultSparseConvTranspose requires topology with direction=Transposed");

    static auto const table =
        dispatch_table_from_op<gs_default_conv_op>("gather_scatter_default_sparse_conv_transpose");

    auto const dev  = features.device().type();
    auto const f_st = features.scalar_type();

    return table.select(dispatch_set{dev, f_st})(features, weights, topo);
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

    static auto const table = dispatch_table_from_op<gs_default_conv_transpose_backward_op>(
        "gather_scatter_default_sparse_conv_transpose_backward");

    auto const dev  = features.device().type();
    auto const f_st = features.scalar_type();

    return table.select(dispatch_set{dev, f_st})(grad_output, features, weights, topo);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
