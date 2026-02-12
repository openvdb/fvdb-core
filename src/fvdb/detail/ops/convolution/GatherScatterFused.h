// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// GatherScatterFused.h -- Fused gather-scatter sparse convolution op.
//
// Single-kernel path optimized for small channel counts (C_in/C_out <= 32-64).
// Probes the feature grid directly for each kernel offset -- no precomputed
// topology or intermediate gather buffers are needed.
//
// Supports both forward and transposed convolution via separate entry points.
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

/// Single-kernel fused forward sparse convolution.
///
/// Probes the feature grid directly for each kernel offset.
/// Probe coordinate: output_ijk * stride + kernel_offset.
///
/// @param features      Input features, shape [feature_total_voxels, C_in].
/// @param weights       Kernel weights, shape [C_out, C_in, k0, k1, k2].
/// @param feature_grid  Grid batch where input features live.
/// @param output_grid   Grid batch where convolution output lives.
/// @param kernel_size   Per-axis kernel size (e.g. {3,3,3}).
/// @param stride        Per-axis stride (e.g. {1,1,1}).
/// @return              Output features, shape [output_total_voxels, C_out].
torch::Tensor gatherScatterSparseConvFused(torch::Tensor features,
                                           torch::Tensor weights,
                                           GridBatchImpl const &feature_grid,
                                           GridBatchImpl const &output_grid,
                                           nanovdb::Coord kernel_size,
                                           nanovdb::Coord stride);

/// Backward pass for fused forward sparse convolution.
std::tuple<torch::Tensor, torch::Tensor>
gatherScatterSparseConvFusedBackward(torch::Tensor grad_output,
                                     torch::Tensor features,
                                     torch::Tensor weights,
                                     GridBatchImpl const &feature_grid,
                                     GridBatchImpl const &output_grid,
                                     nanovdb::Coord kernel_size,
                                     nanovdb::Coord stride);

// =============================================================================
// Fused transposed sparse convolution (small-C optimized)
// =============================================================================

/// Single-kernel fused transposed sparse convolution.
///
/// Probes the feature grid with negated offsets and stride division.
/// Probe coordinate: (output_ijk - kernel_offset) / stride, valid only
/// when (output_ijk - kernel_offset) is divisible by stride.
///
/// @param features      Input features, shape [feature_total_voxels, C_out].
///                      (For transposed conv, feature channels = C_out.)
/// @param weights       Kernel weights, shape [C_out, C_in, k0, k1, k2].
/// @param feature_grid  Grid batch where input features live.
/// @param output_grid   Grid batch where convolution output lives.
/// @param kernel_size   Per-axis kernel size (e.g. {3,3,3}).
/// @param stride        Per-axis stride (e.g. {1,1,1}).
/// @return              Output features, shape [output_total_voxels, C_in].
torch::Tensor gatherScatterSparseConvFusedTranspose(torch::Tensor features,
                                                    torch::Tensor weights,
                                                    GridBatchImpl const &feature_grid,
                                                    GridBatchImpl const &output_grid,
                                                    nanovdb::Coord kernel_size,
                                                    nanovdb::Coord stride);

/// Backward pass for fused transposed sparse convolution.
std::tuple<torch::Tensor, torch::Tensor>
gatherScatterSparseConvFusedTransposeBackward(torch::Tensor grad_output,
                                              torch::Tensor features,
                                              torch::Tensor weights,
                                              GridBatchImpl const &feature_grid,
                                              GridBatchImpl const &output_grid,
                                              nanovdb::Coord kernel_size,
                                              nanovdb::Coord stride);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_CONVOLUTION_GATHERSCATTERFUSED_H
