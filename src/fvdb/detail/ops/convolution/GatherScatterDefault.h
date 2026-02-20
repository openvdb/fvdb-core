// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file GatherScatterDefault.h
/// @brief Default gather-scatter sparse convolution op.
///
/// Compacts a dense kernel_map into per-offset CSR segments and executes:
///   1. Single gather into a contiguous [total_pairs, C_in] buffer
///   2. Per-offset torch::mm over sliced buffer segments
///   3. Single scatter-add from [total_pairs, C_out] into output
///
/// Only active (voxel, kernel_offset) pairs are gathered and computed on,
/// eliminating zero-padding waste present in the dense kernel map.
///
/// Supports CPU and CUDA.  Dispatches via the dispatch table framework
/// for device type and scalar type (float32, float64).
///
/// Forward, backward, transposed forward, and transposed backward are all
/// supported through separate entry points.  The transposed variants differ
/// only in the topology (probe formula); the GEMM structure is identical.
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

/// @brief Selects forward vs transposed convolution topology.
enum class ConvDirection { Forward, Transposed };

/// @brief Compacted CSR topology for gather-scatter sparse convolution.
///
/// For each kernel offset k (0 .. kernel_volume-1), stores the list of active
/// (feature_voxel, output_voxel) pairs as contiguous segments in flat arrays.
///
/// Layout:
///   - @c gather_indices[offsets[k] .. offsets[k+1])  = feature voxel flat indices
///   - @c scatter_indices[offsets[k] .. offsets[k+1]) = output voxel flat indices
///
/// Built once and reused across multiple convolutions on the same grid pair.
struct GatherScatterDefaultTopology {
    /// @brief Feature-side flat voxel indices, shape [total_pairs], int32, on device.
    torch::Tensor gather_indices;
    /// @brief Output-side flat voxel indices, shape [total_pairs], int32, on device.
    torch::Tensor scatter_indices;

    /// @brief Segment boundaries per kernel offset.
    ///
    /// @c offsets[k] to @c offsets[k+1] delimit the pairs for offset k.
    /// Shape [kernel_volume + 1], int64, stored on host (small, iterated for
    /// buffer slicing).
    torch::Tensor offsets;

    int64_t feature_total_voxels; ///< Total voxels in the feature grid.
    int64_t output_total_voxels;  ///< Total voxels in the output grid.
    int64_t kernel_volume;        ///< Product of kernel spatial dimensions (k0 * k1 * k2).
    int64_t total_pairs;        ///< Total active (feature, output) voxel pairs across all offsets.

    nanovdb::Coord kernel_size; ///< Spatial kernel dimensions [k0, k1, k2].
    nanovdb::Coord stride;      ///< Convolution stride [s0, s1, s2].

    ConvDirection direction;    ///< Whether this topology is for forward or transposed convolution.
};

/// @brief Build a compacted forward topology via two-pass atomic counting.
/// @param feature_grid  Grid batch containing the input feature voxels.
/// @param output_grid   Grid batch containing the output voxels.
/// @param kernel_size   Spatial kernel dimensions [k0, k1, k2].
/// @param stride        Convolution stride [s0, s1, s2].
/// @return Topology with direction=Forward.
GatherScatterDefaultTopology
gatherScatterDefaultSparseConvTopology(GridBatchImpl const &feature_grid,
                                       GridBatchImpl const &output_grid,
                                       nanovdb::Coord kernel_size,
                                       nanovdb::Coord stride);

/// @brief Build a compacted transposed topology via two-pass atomic counting.
/// @param feature_grid  Grid batch containing the input feature voxels.
/// @param output_grid   Grid batch containing the output voxels.
/// @param kernel_size   Spatial kernel dimensions [k0, k1, k2].
/// @param stride        Convolution stride [s0, s1, s2].
/// @return Topology with direction=Transposed.
GatherScatterDefaultTopology
gatherScatterDefaultSparseConvTransposeTopology(GridBatchImpl const &feature_grid,
                                                GridBatchImpl const &output_grid,
                                                nanovdb::Coord kernel_size,
                                                nanovdb::Coord stride);

/// @brief Forward pass of sparse convolution.
/// @param features  Input features, shape [feature_total_voxels, C_in].
/// @param weights   Kernel weights, shape [C_out, C_in, k0, k1, k2].
/// @param topo      Precomputed compacted topology (direction=Forward).
/// @return          Output features, shape [output_total_voxels, C_out].
torch::Tensor gatherScatterDefaultSparseConv(torch::Tensor features,
                                             torch::Tensor weights,
                                             GatherScatterDefaultTopology const &topo);

/// @brief Backward pass of forward sparse convolution.
/// @param grad_output  Gradient w.r.t. the convolution output.
/// @param features     Input features used in the forward pass.
/// @param weights      Kernel weights used in the forward pass.
/// @param topo         Precomputed compacted topology (direction=Forward).
/// @return Tuple of (grad_features, grad_weights).
std::tuple<torch::Tensor, torch::Tensor>
gatherScatterDefaultSparseConvBackward(torch::Tensor grad_output,
                                       torch::Tensor features,
                                       torch::Tensor weights,
                                       GatherScatterDefaultTopology const &topo);

/// @brief Forward pass of transposed sparse convolution.
/// @param features  Input features, shape [feature_total_voxels, C_in].
/// @param weights   Kernel weights, shape [C_out, C_in, k0, k1, k2].
/// @param topo      Precomputed compacted topology (direction=Transposed).
/// @return          Output features, shape [output_total_voxels, C_out].
torch::Tensor gatherScatterDefaultSparseConvTranspose(torch::Tensor features,
                                                      torch::Tensor weights,
                                                      GatherScatterDefaultTopology const &topo);

/// @brief Backward pass of transposed sparse convolution.
/// @param grad_output  Gradient w.r.t. the transposed convolution output.
/// @param features     Input features used in the forward pass.
/// @param weights      Kernel weights used in the forward pass.
/// @param topo         Precomputed compacted topology (direction=Transposed).
/// @return Tuple of (grad_features, grad_weights).
std::tuple<torch::Tensor, torch::Tensor>
gatherScatterDefaultSparseConvTransposeBackward(torch::Tensor grad_output,
                                                torch::Tensor features,
                                                torch::Tensor weights,
                                                GatherScatterDefaultTopology const &topo);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_CONVOLUTION_GATHERSCATTERDEFAULT_H
