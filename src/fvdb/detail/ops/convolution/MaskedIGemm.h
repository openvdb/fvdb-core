// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// MaskedIGemm.h -- Masked implicit-GEMM submanifold sparse convolution.
//
// Output-stationary implicit GEMM over a voxel-major neighbor map. Each CTA owns
// a tile of output rows, stages their neighbor rows in shared memory, and
// accumulates every tap any of its rows uses in registers. Forward and input
// gradient need no atomics. Rows are grouped by the Gray rank of their tap
// mask so tiles skip the taps none of their rows use.
//
// Supports submanifold convolution (output grid == feature grid, stride 1,
// odd kernel sizes) on CUDA with float16 and bfloat16 features.
//

#ifndef FVDB_DETAIL_OPS_CONVOLUTION_MASKEDIGEMM_H
#define FVDB_DETAIL_OPS_CONVOLUTION_MASKEDIGEMM_H

#include <fvdb/GridBatchData.h>

#include <nanovdb/NanoVDB.h>

#include <torch/types.h>

#include <cstdint>
#include <tuple>

namespace fvdb {
namespace detail {
namespace ops {

struct MaskedIGemmTopology {
    // Row o of neighborMap holds output voxel o's K input rows (-1 if absent).
    torch::Tensor neighborMap;     ///< int32 [N, K]
    torch::Tensor sortedRows;      ///< int32 [N], rows ordered by Gray rank of their tap mask
    torch::Tensor chunkTaps;       ///< int32, per tap the 32-row chunks of sortedRows using it
    torch::Tensor chunkTapOffsets; ///< int32 [K + 1], CSR offsets into chunkTaps (device)

    int64_t totalVoxels;           ///< Rows in the feature (and output) grid
    int64_t kernelVolume;          ///< k0 * k1 * k2
    nanovdb::Coord kernelSize;     ///< [k0, k1, k2]
};

/// Rows per weight-gradient chunk.
constexpr int kMaskedIGemmChunkRows = 32;

/// Build the neighbor map and row ordering for submanifold convolution on @p grid.
MaskedIGemmTopology maskedIGemmTopology(GridBatchData const &grid, nanovdb::Coord kernelSize);

/// Forward: weights are a (C_out, C_in, k0, k1, k2) tensor. Returns [N, C_out].
torch::Tensor
maskedIGemmConv(torch::Tensor features, torch::Tensor weights, MaskedIGemmTopology const &topo);

/// Backward: returns (grad_features [N, C_in], grad_weights shaped like weights).
std::tuple<torch::Tensor, torch::Tensor> maskedIGemmConvBackward(torch::Tensor gradOutput,
                                                                 torch::Tensor features,
                                                                 torch::Tensor weights,
                                                                 MaskedIGemmTopology const &topo);

/// True when maskedIGemmConv can run these channel counts and dtype.
bool maskedIGemmSupports(int64_t inChannels, int64_t outChannels, torch::ScalarType dtype);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_CONVOLUTION_MASKEDIGEMM_H
