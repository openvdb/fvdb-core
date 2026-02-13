// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// CutlassGroupedGemm.h -- CUTLASS-accelerated sparse 3D convolution.
//
// GPU-only implementation using:
//   - Two-pass GPU topology builder (count + fill via NanoVDB tree accessors)
//   - CUTLASS grouped GEMM (fp16 tensor cores, fp32 accumulate)
//   - Collision-free scatter (each output voxel unique per offset, sequential over k)
//
// Supports arbitrary (non-uniform) kernel sizes and strides.
// Channel counts (Cin, Cout) must be multiples of 32.
//
// Forward, backward, transposed forward, and transposed backward are all
// implemented.  Transposed differs only in topology (probe formula);
// the GEMM structure is shared.
//
#ifndef FVDB_DETAIL_OPS_CONVOLUTION_CUTLASSGROUPEDGEMM_H
#define FVDB_DETAIL_OPS_CONVOLUTION_CUTLASSGROUPEDGEMM_H

#include <fvdb/detail/GridBatchImpl.h>

#include <nanovdb/NanoVDB.h>

#include <torch/types.h>

#include <cstdint>
#include <tuple>

namespace fvdb {
namespace detail {
namespace ops {

// =============================================================================
// CutlassConvTopology -- GPU-built CSR topology for CUTLASS grouped GEMM
// =============================================================================
//
// For each kernel offset k (0 .. kernel_volume-1), stores the list of active
// (feature_voxel, output_voxel) pairs as contiguous segments in flat arrays.
//
// Within each segment, each output voxel index appears at most once (since a
// given offset probes exactly one input coordinate per output voxel). This
// property allows collision-free scatter without atomics when processing
// offsets sequentially.
//
// Layout:
//   gather_indices[offsets[k] .. offsets[k+1])  = feature voxel flat indices
//   scatter_indices[offsets[k] .. offsets[k+1]) = output voxel flat indices

struct CutlassConvTopology {
    torch::Tensor gather_indices;  // [total_pairs] int32, CUDA
    torch::Tensor scatter_indices; // [total_pairs] int32, CUDA

    // Segment boundaries: offsets[k] to offsets[k+1] are the pairs for offset k.
    // Shape [kernel_volume + 1], int64, ON HOST (small, iterated for GEMM setup).
    torch::Tensor offsets;

    int64_t feature_total_voxels;
    int64_t output_total_voxels;
    int64_t kernel_volume;
    int64_t total_pairs;

    nanovdb::Coord kernel_size; // non-uniform, e.g. {3, 5, 3}
    nanovdb::Coord stride;      // non-uniform, e.g. {1, 2, 1}
};

// =============================================================================
// Topology builder (GPU-native two-pass: count + fill)
// =============================================================================

/// Build a forward convolution topology entirely on GPU.
CutlassConvTopology cutlassConvTopology(GridBatchImpl const &feature_grid,
                                        GridBatchImpl const &output_grid,
                                        nanovdb::Coord kernel_size,
                                        nanovdb::Coord stride);

// =============================================================================
// Forward sparse convolution
// =============================================================================

/// CUTLASS-accelerated forward sparse convolution (fp16 in, fp16 out).
///
/// @param features  Input features, shape [feature_total_voxels, Cin], fp16.
/// @param weights   Kernel weights, shape [Cout, Cin, k0, k1, k2], fp16.
/// @param topo      Precomputed topology from cutlassConvTopology.
/// @return          Output features, shape [output_total_voxels, Cout], fp16.
torch::Tensor
cutlassConv(torch::Tensor features, torch::Tensor weights, CutlassConvTopology const &topo);

// =============================================================================
// Backward sparse convolution
// =============================================================================

/// CUTLASS-accelerated backward pass for forward sparse convolution.
///
/// @param grad_output  Gradient w.r.t. forward output, shape [output_total_voxels, Cout], fp16.
/// @param features     Input features from the forward pass, shape [feature_total_voxels, Cin], fp16.
/// @param weights      Kernel weights, shape [Cout, Cin, k0, k1, k2], fp16.
/// @param topo         Precomputed topology from cutlassConvTopology (same as forward).
/// @return             Tuple of (grad_features [feature_total_voxels, Cin] fp16,
///                               grad_weights [Cout, Cin, k0, k1, k2] fp16).
std::tuple<torch::Tensor, torch::Tensor>
cutlassConvBackward(torch::Tensor grad_output,
                    torch::Tensor features,
                    torch::Tensor weights,
                    CutlassConvTopology const &topo);

// =============================================================================
// Transposed sparse convolution
// =============================================================================

/// Build a transposed convolution topology entirely on GPU.
/// Uses probe = (output_ijk - kernel_offset) / stride with divisibility check.
CutlassConvTopology cutlassConvTransposeTopology(GridBatchImpl const &feature_grid,
                                                 GridBatchImpl const &output_grid,
                                                 nanovdb::Coord kernel_size,
                                                 nanovdb::Coord stride);

/// CUTLASS-accelerated transposed forward sparse convolution (fp16 in, fp16 out).
/// GEMM structure is identical to forward; only the topology differs.
torch::Tensor
cutlassConvTranspose(torch::Tensor features, torch::Tensor weights, CutlassConvTopology const &topo);

/// CUTLASS-accelerated backward pass for transposed sparse convolution.
/// GEMM structure is identical to forward backward; only the topology differs.
std::tuple<torch::Tensor, torch::Tensor>
cutlassConvTransposeBackward(torch::Tensor grad_output,
                             torch::Tensor features,
                             torch::Tensor weights,
                             CutlassConvTopology const &topo);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_CONVOLUTION_CUTLASSGROUPEDGEMM_H
