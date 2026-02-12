// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// GatherScatter.h -- Gather-scatter sparse convolution op (GEMM path).
//
// Type-erased entry points for:
//   1. Topology precomputation (kernel map between feature and output grids)
//   2. Forward / transposed forward sparse convolution using precomputed topology
//   3. Backward / transposed backward sparse convolution using precomputed topology
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
// ConvDirection -- forward vs transposed convolution
// =============================================================================

enum class ConvDirection { Forward, Transposed };

// =============================================================================
// GatherScatterTopology -- precomputed neighborhood structure
// =============================================================================
//
// Stores the mapping from each output voxel to its feature-grid neighbors for
// every kernel offset.  This is independent of feature data and can be reused
// across multiple convolutions with different features/weights on the same
// grid pair.
//
// Naming: "feature_grid" is the grid whose voxels carry input features;
// "output_grid" is the grid whose voxels carry the convolution output.
// These names are stable across forward and transposed convolution.

struct GatherScatterTopology {
    // kernel_map[o, k] = flat feature-grid voxel index if the k-th kernel
    // offset from output voxel o lands on an active feature voxel, else -1.
    // Shape: [output_total_voxels, kernel_volume], dtype int32, same device as grids.
    torch::Tensor kernel_map;

    int64_t feature_total_voxels;
    int64_t output_total_voxels;
    int64_t kernel_volume; // product of kernel_size components

    // Kept for validation against weights shape.
    nanovdb::Coord kernel_size;
    nanovdb::Coord stride;

    // Which direction was used to build this topology.
    ConvDirection direction;

    // True iff kmap[:, K/2] == arange(output_total_voxels) when kernel_volume
    // is odd.  Enables the "middle acceleration" fast-path in the forward pass
    // (skip gather for the center kernel offset).  Computed once during
    // topology precomputation so the forward op pays no verification cost.
    bool center_is_identity;
};

// =============================================================================
// Topology precomputation
// =============================================================================

/// Build the forward kernel map between feature and output grids.
///
/// For every active voxel in output_grid, probes the feature_grid at each
/// kernel offset and records the flat feature voxel index (or -1 if inactive).
///
/// Probe coordinate: output_ijk * stride + kernel_offset.
///
/// @param feature_grid  Grid batch where input features live.
/// @param output_grid   Grid batch where convolution output lives.
/// @param kernel_size   Per-axis kernel size (e.g. {3,3,3}).
/// @param stride        Per-axis stride (e.g. {1,1,1}).
/// @return              Precomputed topology structure with direction=Forward.
GatherScatterTopology gatherScatterSparseConvTopology(GridBatchImpl const &feature_grid,
                                                      GridBatchImpl const &output_grid,
                                                      nanovdb::Coord kernel_size,
                                                      nanovdb::Coord stride);

/// Build the transposed kernel map between feature and output grids.
///
/// For every active voxel in output_grid, probes the feature_grid at each
/// kernel offset with negated (flipped) offsets and stride division.
///
/// Probe coordinate: (output_ijk - kernel_offset) / stride, valid only when
/// (output_ijk - kernel_offset) is divisible by stride in all 3 dimensions.
///
/// @param feature_grid  Grid batch where input features live.
/// @param output_grid   Grid batch where convolution output lives.
/// @param kernel_size   Per-axis kernel size (e.g. {3,3,3}).
/// @param stride        Per-axis stride (e.g. {1,1,1}).
/// @return              Precomputed topology structure with direction=Transposed.
GatherScatterTopology gatherScatterSparseConvTransposeTopology(GridBatchImpl const &feature_grid,
                                                               GridBatchImpl const &output_grid,
                                                               nanovdb::Coord kernel_size,
                                                               nanovdb::Coord stride);

// =============================================================================
// Forward sparse convolution
// =============================================================================

/// Gather-scatter forward sparse convolution.
///
/// @param features  Input features, shape [feature_total_voxels, C_in].
/// @param weights   Kernel weights, shape [C_out, C_in, k0, k1, k2]
///                  (PyTorch conv3d layout).
/// @param topo      Precomputed topology (direction=Forward).
/// @return          Output features, shape [output_total_voxels, C_out],
///                  same dtype as features.
torch::Tensor gatherScatterSparseConv(torch::Tensor features,
                                      torch::Tensor weights,
                                      GatherScatterTopology const &topo);

/// Backward pass for gather-scatter forward sparse convolution.
///
/// @param grad_output  Gradient w.r.t. forward output,
///                     shape [output_total_voxels, C_out].
/// @param features     Input features from the forward pass,
///                     shape [feature_total_voxels, C_in].
/// @param weights      Kernel weights, shape [C_out, C_in, k0, k1, k2].
/// @param topo         Precomputed topology (direction=Forward).
/// @return             Tuple of (grad_features, grad_weights).
std::tuple<torch::Tensor, torch::Tensor>
gatherScatterSparseConvBackward(torch::Tensor grad_output,
                                torch::Tensor features,
                                torch::Tensor weights,
                                GatherScatterTopology const &topo);

// =============================================================================
// Transposed sparse convolution
// =============================================================================

/// Gather-scatter transposed sparse convolution.
///
/// @param features  Input features, shape [feature_total_voxels, C_in].
///                  (Same channel convention as forward.)
/// @param weights   Kernel weights, shape [C_out, C_in, k0, k1, k2]
///                  (same layout as forward).
/// @param topo      Precomputed topology (direction=Transposed).
/// @return          Output features, shape [output_total_voxels, C_out],
///                  same dtype as features.
torch::Tensor gatherScatterSparseConvTranspose(torch::Tensor features,
                                               torch::Tensor weights,
                                               GatherScatterTopology const &topo);

/// Backward pass for gather-scatter transposed sparse convolution.
///
/// @param grad_output  Gradient w.r.t. transposed output,
///                     shape [output_total_voxels, C_out].
/// @param features     Input features from the forward pass,
///                     shape [feature_total_voxels, C_in].
/// @param weights      Kernel weights, shape [C_out, C_in, k0, k1, k2].
/// @param topo         Precomputed topology (direction=Transposed).
/// @return             Tuple of (grad_features, grad_weights).
std::tuple<torch::Tensor, torch::Tensor>
gatherScatterSparseConvTransposeBackward(torch::Tensor grad_output,
                                         torch::Tensor features,
                                         torch::Tensor weights,
                                         GatherScatterTopology const &topo);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_CONVOLUTION_GATHERSCATTER_H
