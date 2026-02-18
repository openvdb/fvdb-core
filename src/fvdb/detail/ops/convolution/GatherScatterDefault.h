// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// GatherScatterDefault.h -- Default gather-scatter sparse convolution op.
//
// Compacts a dense kernel_map into per-offset CSR segments and executes:
//   1. Single gather into a contiguous [total_pairs, C_in] buffer
//   2. Per-offset torch::mm over sliced buffer segments
//   3. Single scatter-add from [total_pairs, C_out] into output
//
// Only active (voxel, kernel_offset) pairs are gathered and computed on,
// eliminating zero-padding waste present in the dense kernel map.
//
// Supports CPU and CUDA.  Dispatches via the dispatch table framework
// for device type and scalar type (float32, float64).
//
// Forward, backward, transposed forward, and transposed backward are all
// supported through separate entry points.  The transposed variants differ
// only in the topology (probe formula); the GEMM structure is identical.
//
#ifndef FVDB_DETAIL_OPS_CONVOLUTION_GATHERSCATTERDEFAULT_H
#define FVDB_DETAIL_OPS_CONVOLUTION_GATHERSCATTERDEFAULT_H

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
// GatherScatterDefaultTopology -- compacted CSR topology
// =============================================================================
//
// For each kernel offset k (0 .. kernel_volume-1), stores the list of active
// (feature_voxel, output_voxel) pairs as contiguous segments in flat arrays.
//
// Layout:
//   gather_indices[offsets[k] .. offsets[k+1])  = feature voxel flat indices
//   scatter_indices[offsets[k] .. offsets[k+1]) = output voxel flat indices
//
// This is built once and reused across multiple convolutions on the same
// grid pair.

struct GatherScatterDefaultTopology {
    torch::Tensor gather_indices;  // [total_pairs] int32, on device
    torch::Tensor scatter_indices; // [total_pairs] int32, on device

    // Segment boundaries: offsets[k] to offsets[k+1] are the pairs for offset k.
    // Shape [kernel_volume + 1], int64, ON HOST (small, iterated for buffer slicing).
    torch::Tensor offsets;

    int64_t feature_total_voxels;
    int64_t output_total_voxels;
    int64_t kernel_volume;
    int64_t total_pairs;

    nanovdb::Coord kernel_size;
    nanovdb::Coord stride;

    ConvDirection direction;
};

// =============================================================================
// Topology builders
// =============================================================================

/// Build a compacted forward topology from source and output grids.
GatherScatterDefaultTopology
gatherScatterDefaultSparseConvTopology(GridBatchImpl const &feature_grid,
                                       GridBatchImpl const &output_grid,
                                       nanovdb::Coord kernel_size,
                                       nanovdb::Coord stride);

/// Build a compacted transposed topology from source and output grids.
GatherScatterDefaultTopology
gatherScatterDefaultSparseConvTransposeTopology(GridBatchImpl const &feature_grid,
                                                GridBatchImpl const &output_grid,
                                                nanovdb::Coord kernel_size,
                                                nanovdb::Coord stride);

// =============================================================================
// Forward sparse convolution
// =============================================================================

/// @param features  Input features, shape [feature_total_voxels, C_in].
/// @param weights   Kernel weights, shape [C_out, C_in, k0, k1, k2].
/// @param topo      Precomputed compacted topology (direction=Forward).
/// @return          Output features, shape [output_total_voxels, C_out].
torch::Tensor gatherScatterDefaultSparseConv(torch::Tensor features,
                                             torch::Tensor weights,
                                             GatherScatterDefaultTopology const &topo);

/// Backward pass for forward sparse convolution.
/// @return Tuple of (grad_features, grad_weights).
std::tuple<torch::Tensor, torch::Tensor>
gatherScatterDefaultSparseConvBackward(torch::Tensor grad_output,
                                       torch::Tensor features,
                                       torch::Tensor weights,
                                       GatherScatterDefaultTopology const &topo);

// =============================================================================
// Transposed sparse convolution
// =============================================================================

torch::Tensor gatherScatterDefaultSparseConvTranspose(torch::Tensor features,
                                                      torch::Tensor weights,
                                                      GatherScatterDefaultTopology const &topo);

std::tuple<torch::Tensor, torch::Tensor>
gatherScatterDefaultSparseConvTransposeBackward(torch::Tensor grad_output,
                                                torch::Tensor features,
                                                torch::Tensor weights,
                                                GatherScatterDefaultTopology const &topo);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_CONVOLUTION_GATHERSCATTERDEFAULT_H
