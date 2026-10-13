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
// Supports submanifold convolution (output grid == feature grid, stride 1,
// odd kernel sizes) on CUDA with float16, bfloat16 and float32 (computed as TF32)
// features.
//

#ifndef FVDB_DETAIL_OPS_CONVOLUTION_MASKEDIGEMM_H
#define FVDB_DETAIL_OPS_CONVOLUTION_MASKEDIGEMM_H

#include <fvdb/GridBatchData.h>

#include <nanovdb/NanoVDB.h>

#include <torch/types.h>

#include <cstdint>
#include <optional>
#include <tuple>

namespace fvdb {
namespace detail {
namespace ops {

struct MaskedIGemmTopology {
    // Row o of neighborMap holds output voxel o's K input rows (-1 if absent).
    torch::Tensor neighborMap; ///< int32 [N, K]
    torch::Tensor sortedRows;  ///< int32 [N], rows ordered by Gray rank of their tap mask
    torch::Tensor rowMasks;    ///< int32 [N], bit k set when the row has a tap-k neighbor
    torch::Tensor pairIn;      ///< int32 [P], per tap the input row of each active pair
    torch::Tensor pairOut;     ///< int32 [P], per tap the output row, ascending
    torch::Tensor pairOffsets; ///< int32 [K + 1], CSR offsets of each tap's pairs (device)

    int64_t totalVoxels;       ///< Rows in the feature (and output) grid
    int64_t kernelVolume;      ///< k0 * k1 * k2
    nanovdb::Coord kernelSize; ///< [k0, k1, k2]
};

/// Build the neighbor map and row ordering for submanifold convolution on @p grid.
MaskedIGemmTopology maskedIGemmTopology(GridBatchData const &grid, nanovdb::Coord kernelSize);

/// Forward: weights are a (C_out, C_in, k0, k1, k2) tensor, bias an optional [C_out] tensor
/// added in the epilogue. Returns [N, C_out].
torch::Tensor maskedIGemmConv(torch::Tensor features,
                              torch::Tensor weights,
                              std::optional<torch::Tensor> bias,
                              MaskedIGemmTopology const &topo);

/// Backward: returns (grad_features [N, C_in], grad_weights shaped like weights). Each is an
/// undefined tensor when its need flag is false.
std::tuple<torch::Tensor, torch::Tensor> maskedIGemmConvBackward(torch::Tensor gradOutput,
                                                                 torch::Tensor features,
                                                                 torch::Tensor weights,
                                                                 MaskedIGemmTopology const &topo,
                                                                 bool needFeatures,
                                                                 bool needWeights);

/// True when maskedIGemmConv can run these channel counts and dtype.
bool maskedIGemmSupports(int64_t inChannels, int64_t outChannels, torch::ScalarType dtype);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_CONVOLUTION_MASKEDIGEMM_H
