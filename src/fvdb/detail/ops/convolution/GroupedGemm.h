// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// GroupedGemm.h -- Compacted gather-scatter sparse convolution op.
//
// Compacts the dense kernel_map into per-offset CSR segments and executes:
//   1. Single gather into a contiguous [total_pairs, C_in] buffer
//   2. Per-offset torch::mm over sliced buffer segments
//   3. Single scatter-add from [total_pairs, C_out] into output
//
// Only active (voxel, kernel_offset) pairs are gathered and computed on,
// eliminating zero-padding waste present in the dense GatherScatter path.
//
// Supports CPU and CUDA.  Dispatches via the dispatch table framework
// for device type and scalar type (float32, float64).
//
// Forward, backward, transposed forward, and transposed backward are all
// supported through separate entry points.  The transposed variants differ
// only in the topology (probe formula); the GEMM structure is identical.
//
#ifndef FVDB_DETAIL_OPS_CONVOLUTION_GROUPEDGEMM_H
#define FVDB_DETAIL_OPS_CONVOLUTION_GROUPEDGEMM_H

#include <fvdb/detail/GridBatchImpl.h>
#include <fvdb/detail/ops/convolution/GatherScatter.h>

#include <nanovdb/NanoVDB.h>

#include <torch/types.h>

#include <cstdint>
#include <tuple>

namespace fvdb {
namespace detail {
namespace ops {

// =============================================================================
// GroupedGemmTopology -- compacted CSR topology for grouped GEMM
// =============================================================================
//
// For each kernel offset k (0 .. kernel_volume-1), stores the list of active
// (feature_voxel, output_voxel) pairs as contiguous segments in flat arrays.
//
// Layout:
//   gather_indices[offsets[k] .. offsets[k+1])  = feature voxel flat indices
//   scatter_indices[offsets[k] .. offsets[k+1]) = output voxel flat indices
//
// This is built once from a GatherScatterTopology::kernel_map and reused
// across multiple convolutions on the same grid pair.

struct GroupedGemmTopology {
    // Flat arrays of active pairs, grouped by kernel offset.
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
///
/// Internally builds a GatherScatterTopology (dense kernel_map) and then
/// compacts it into the CSR format needed by the grouped GEMM.
GroupedGemmTopology groupedGemmSparseConvTopology(GridBatchImpl const &feature_grid,
                                                  GridBatchImpl const &output_grid,
                                                  nanovdb::Coord kernel_size,
                                                  nanovdb::Coord stride);

/// Build a compacted transposed topology from source and output grids.
GroupedGemmTopology groupedGemmSparseConvTransposeTopology(GridBatchImpl const &feature_grid,
                                                           GridBatchImpl const &output_grid,
                                                           nanovdb::Coord kernel_size,
                                                           nanovdb::Coord stride);

// =============================================================================
// Forward sparse convolution
// =============================================================================

/// Compacted gather-scatter forward sparse convolution.
///
/// @param features  Input features, shape [feature_total_voxels, C_in].
/// @param weights   Kernel weights, shape [C_out, C_in, k0, k1, k2].
/// @param topo      Precomputed compacted topology (direction=Forward).
/// @return          Output features, shape [output_total_voxels, C_out].
torch::Tensor groupedGemmSparseConv(torch::Tensor features,
                                    torch::Tensor weights,
                                    GroupedGemmTopology const &topo);

/// Backward pass for compacted gather-scatter forward sparse convolution.
///
/// @param grad_output  Gradient w.r.t. forward output.
/// @param features     Input features from the forward pass.
/// @param weights      Kernel weights, shape [C_out, C_in, k0, k1, k2].
/// @param topo         Precomputed compacted topology (direction=Forward).
/// @return             Tuple of (grad_features, grad_weights).
std::tuple<torch::Tensor, torch::Tensor>
groupedGemmSparseConvBackward(torch::Tensor grad_output,
                              torch::Tensor features,
                              torch::Tensor weights,
                              GroupedGemmTopology const &topo);

// =============================================================================
// Transposed sparse convolution
// =============================================================================

/// Compacted gather-scatter transposed sparse convolution.
torch::Tensor groupedGemmSparseConvTranspose(torch::Tensor features,
                                             torch::Tensor weights,
                                             GroupedGemmTopology const &topo);

/// Backward pass for compacted gather-scatter transposed sparse convolution.
std::tuple<torch::Tensor, torch::Tensor>
groupedGemmSparseConvTransposeBackward(torch::Tensor grad_output,
                                       torch::Tensor features,
                                       torch::Tensor weights,
                                       GroupedGemmTopology const &topo);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_CONVOLUTION_GROUPEDGEMM_H
