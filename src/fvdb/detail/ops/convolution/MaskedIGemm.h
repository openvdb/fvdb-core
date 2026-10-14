// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// MaskedIGemm.h -- Masked implicit-GEMM submanifold sparse convolution.
//
// Output-stationary implicit GEMM over a voxel-major neighbor map. Each CTA owns
// a tile of output rows, stages their neighbor rows in shared memory, and
// accumulates every tap any of its rows uses in registers. Forward and input
// gradient need no atomics. Rows are grouped by the Gray rank of their tap
// mask so tiles skip the taps none of their rows use. The weight gradient
// reduces over per-tap (input, output) pair lists.
//
// Supports any stride and kernel size with volume <= 32, and the transposed
// direction of each, on CUDA with float16, bfloat16 and float32 (computed as TF32)
// features. The input gradient and the transpose run the same kernel over the
// reverse map; a submanifold map (same grid, stride 1, odd kernel) is its own
// reverse with mirrored taps, so it stores none.
//

#ifndef FVDB_DETAIL_OPS_CONVOLUTION_MASKEDIGEMM_H
#define FVDB_DETAIL_OPS_CONVOLUTION_MASKEDIGEMM_H

#include <fvdb/GridBatchData.h>

#include <nanovdb/NanoVDB.h>

#include <torch/types.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>

namespace fvdb {
namespace detail {
namespace ops {

struct MaskedIGemmTopology {
    // Row o of neighborMap holds output voxel o's K input rows (-1 if absent).
    torch::Tensor neighborMap; ///< int32 [N, K]
    torch::Tensor sortedRows;  ///< int32 [N], rows ordered by Gray rank of their tap mask
    /// sortedRows regrouped into windows of nearby rows; built on first use for large inputs.
    mutable torch::Tensor localRows;
    torch::Tensor rowMasks;    ///< int32 [N], bit k set when the row has a tap-k neighbor
    torch::Tensor pairIn;      ///< int32 [P], per tap the input row of each active pair
    torch::Tensor pairOut;     ///< int32 [P], per tap the output row, ascending
    torch::Tensor pairOffsets; ///< int32 [K + 1], CSR offsets of each tap's pairs (device)
    /// int32 [K, ceil(N / 256)], first pair of each tap at or after each 256-row boundary
    torch::Tensor pairBlockOffsets;

    int64_t totalVoxels;       ///< Output rows (rows of neighborMap)
    int64_t inputVoxels;       ///< Input rows (the values neighborMap holds)
    int64_t kernelVolume;      ///< k0 * k1 * k2
    nanovdb::Coord kernelSize; ///< [k0, k1, k2]

    /// Map from input rows to the output rows reading them per tap, with its row order and
    /// masks; no pair lists. Null for submanifold maps.
    std::shared_ptr<MaskedIGemmTopology const> reverse;
};

/// Build the map from @p outputGrid's voxels to @p featureGrid's voxels for convolution with
/// @p kernelSize and @p stride (fine = stride * coarse + tap - paddingBefore).
MaskedIGemmTopology maskedIGemmTopology(GridBatchData const &featureGrid,
                                        GridBatchData const &outputGrid,
                                        nanovdb::Coord kernelSize,
                                        nanovdb::Coord stride);

/// Forward: weights are a (C_out, C_in, k0, k1, k2) tensor, bias an optional [C_out] tensor
/// added in the epilogue. Weights and bias must match the features' device and dtype.
/// @p transposed runs the transposed convolution, from the topology's output rows to its
/// input rows. Returns [N_out, C_out].
torch::Tensor maskedIGemmConv(torch::Tensor features,
                              torch::Tensor weights,
                              std::optional<torch::Tensor> bias,
                              MaskedIGemmTopology const &topo,
                              bool transposed = false);

/// Backward of maskedIGemmConv with the same @p transposed: returns (grad_features [N, C_in],
/// grad_weights shaped like weights). Each is an undefined tensor when its need flag is false.
std::tuple<torch::Tensor, torch::Tensor> maskedIGemmConvBackward(torch::Tensor gradOutput,
                                                                 torch::Tensor features,
                                                                 torch::Tensor weights,
                                                                 MaskedIGemmTopology const &topo,
                                                                 bool needFeatures,
                                                                 bool needWeights,
                                                                 bool transposed = false);

/// True when maskedIGemmConv can run these channel counts and dtype on @p device.
bool maskedIGemmSupports(int64_t inChannels,
                         int64_t outChannels,
                         torch::ScalarType dtype,
                         torch::Device device);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_CONVOLUTION_MASKEDIGEMM_H
