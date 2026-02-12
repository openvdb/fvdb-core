// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// GatherScatterFused.h -- Fused gather-scatter sparse convolution op.
//
// Single-kernel path optimized for small channel counts (C_in/C_out <= 32-64).
// Probes the src grid directly for each kernel offset -- no precomputed
// topology or intermediate gather buffers are needed.
//
#ifndef FVDB_DETAIL_OPS_CONVOLUTION_GATHERSCATTERFUSED_H
#define FVDB_DETAIL_OPS_CONVOLUTION_GATHERSCATTERFUSED_H

#include <fvdb/detail/GridBatchImpl.h>

#include <nanovdb/NanoVDB.h>

#include <torch/types.h>

#include <cstdint>
#include <tuple>

namespace fvdb {
namespace detail {
namespace ops {

// =============================================================================
// Fused forward sparse convolution (small-C optimized)
// =============================================================================

/// Single-kernel fused sparse convolution optimized for small channel counts.
///
/// Probes the src grid directly for each kernel offset -- no precomputed
/// topology or intermediate gather buffers are needed.  Each active dst
/// voxel accumulates across all K kernel offsets in a single pass, avoiding
/// cuBLAS launch overhead that dominates when C_in/C_out are small.
///
/// @param features     Input features, shape [src_total_voxels, C_in].
/// @param weights      Kernel weights, shape [C_out, C_in, k0, k1, k2]
///                     (PyTorch conv3d layout).
/// @param src          Source grid batch (features live here).
/// @param dst          Destination grid batch (output lives here).
/// @param kernel_size  Per-axis kernel size (e.g. {3,3,3}).
/// @param stride       Per-axis stride (e.g. {1,1,1}).
/// @return             Output features, shape [dst_total_voxels, C_out],
///                     same dtype as features.
torch::Tensor gatherScatterSparseConvFused(torch::Tensor features,
                                           torch::Tensor weights,
                                           GridBatchImpl const &src,
                                           GridBatchImpl const &dst,
                                           nanovdb::Coord kernel_size,
                                           nanovdb::Coord stride);

// =============================================================================
// Fused backward sparse convolution (small-C optimized)
// =============================================================================

/// Backward pass for the fused sparse convolution.
///
/// Single-kernel path that probes the src grid directly for each kernel
/// offset and accumulates gradients for features and weights via atomic
/// adds.  No precomputed topology or intermediate buffers are needed.
///
/// @param grad_output   Gradient of the loss w.r.t. forward output,
///                      shape [dst_total_voxels, C_out].
/// @param features      Input features from the forward pass,
///                      shape [src_total_voxels, C_in].
/// @param weights       Kernel weights from the forward pass,
///                      shape [C_out, C_in, k0, k1, k2].
/// @param src           Source grid batch (features live here).
/// @param dst           Destination grid batch (output lives here).
/// @param kernel_size   Per-axis kernel size (e.g. {3,3,3}).
/// @param stride        Per-axis stride (e.g. {1,1,1}).
/// @return              Tuple of (grad_features [src_total_voxels, C_in],
///                      grad_weights [C_out, C_in, k0, k1, k2]),
///                      same dtype as features.
std::tuple<torch::Tensor, torch::Tensor>
gatherScatterSparseConvFusedBackward(torch::Tensor grad_output,
                                     torch::Tensor features,
                                     torch::Tensor weights,
                                     GridBatchImpl const &src,
                                     GridBatchImpl const &dst,
                                     nanovdb::Coord kernel_size,
                                     nanovdb::Coord stride);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_CONVOLUTION_GATHERSCATTERFUSED_H
