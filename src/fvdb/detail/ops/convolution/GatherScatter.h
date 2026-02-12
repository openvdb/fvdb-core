// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// GatherScatter.h -- Gather-scatter sparse convolution op (GEMM path).
//
// Type-erased entry points for:
//   1. Topology precomputation (kernel map between src and dst grids)
//   2. Forward sparse convolution using the precomputed topology
//   3. Backward sparse convolution using the precomputed topology
//
// The fused (small-C) path lives in GatherScatterFused.h/.cu.
//
// All dispatch, precondition checking, and device guards are contained
// within these functions. Callers do nothing but a straight call.
//
#ifndef FVDB_DETAIL_OPS_CONVOLUTION_GATHERSCATTER_H
#define FVDB_DETAIL_OPS_CONVOLUTION_GATHERSCATTER_H

#include <fvdb/detail/GridBatchImpl.h>

#include <nanovdb/NanoVDB.h>

#include <torch/types.h>

#include <cstdint>
#include <tuple>

namespace fvdb {
namespace detail {
namespace ops {

// =============================================================================
// GatherScatterTopology -- precomputed neighborhood structure
// =============================================================================
//
// Stores the mapping from each dst voxel to its src neighbors for every
// kernel offset. This is independent of feature data and can be reused
// across multiple convolutions with different features/weights on the
// same grid pair.

struct GatherScatterTopology {
    // kernel_map[d, k] = flat src voxel index if the k-th kernel offset from
    // dst voxel d lands on an active src voxel, else -1.
    // Shape: [dst_total_voxels, kernel_volume], dtype int32, same device as grids.
    torch::Tensor kernel_map;

    int64_t src_total_voxels;
    int64_t dst_total_voxels;
    int64_t kernel_volume; // product of kernel_size components

    // Kept for validation against weights shape.
    nanovdb::Coord kernel_size;
    nanovdb::Coord stride;

    // True iff kmap[:, K/2] == arange(dst_total_voxels) when kernel_volume is
    // odd.  Enables the "middle acceleration" fast-path in the forward pass
    // (skip gather for the center kernel offset).  Computed once during
    // topology precomputation so the forward op pays no verification cost.
    bool center_is_identity;
};

// =============================================================================
// Topology precomputation
// =============================================================================

/// Build the kernel map between src and dst grids.
///
/// For every active voxel in dst, probes the src grid at each kernel offset
/// and records the flat src voxel index (or -1 if inactive).
///
/// @param src          Source grid batch (features live here).
/// @param dst          Destination grid batch (output lives here).
/// @param kernel_size  Per-axis kernel size (e.g. {3,3,3}).
/// @param stride       Per-axis stride (e.g. {1,1,1}).
/// @return             Precomputed topology structure.
GatherScatterTopology gatherScatterSparseConvTopology(GridBatchImpl const &src,
                                                      GridBatchImpl const &dst,
                                                      nanovdb::Coord kernel_size,
                                                      nanovdb::Coord stride);

// =============================================================================
// Forward sparse convolution
// =============================================================================

/// Gather-scatter forward sparse convolution.
///
/// @param features  Input features, shape [src_total_voxels, C_in].
/// @param weights   Kernel weights, shape [C_out, C_in, k0, k1, k2]
///                  (PyTorch conv3d layout).
/// @param topo      Precomputed topology from gatherScatterSparseConvTopology.
/// @return          Output features, shape [dst_total_voxels, C_out],
///                  same dtype as features.
torch::Tensor gatherScatterSparseConv(torch::Tensor features,
                                      torch::Tensor weights,
                                      GatherScatterTopology const &topo);

// =============================================================================
// Backward sparse convolution (GEMM path)
// =============================================================================

/// Backward pass for gather-scatter sparse convolution.
///
/// Computes gradients for both input features and kernel weights given
/// the incoming gradient from the forward pass output.
///
/// @param grad_output  Gradient of the loss w.r.t. forward output,
///                     shape [dst_total_voxels, C_out].
/// @param features     Input features from the forward pass,
///                     shape [src_total_voxels, C_in].
/// @param weights      Kernel weights from the forward pass,
///                     shape [C_out, C_in, k0, k1, k2].
/// @param topo         Precomputed topology from gatherScatterSparseConvTopology.
/// @return             Tuple of (grad_features [src_total_voxels, C_in],
///                     grad_weights [C_out, C_in, k0, k1, k2]),
///                     same dtype as features.
std::tuple<torch::Tensor, torch::Tensor>
gatherScatterSparseConvBackward(torch::Tensor grad_output,
                                torch::Tensor features,
                                torch::Tensor weights,
                                GatherScatterTopology const &topo);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_CONVOLUTION_GATHERSCATTER_H
