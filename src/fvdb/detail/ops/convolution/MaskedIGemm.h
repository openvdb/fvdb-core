// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// MaskedIGemm.h -- Masked implicit-GEMM sparse convolution.
//
// Output-stationary implicit GEMM over a voxel-major neighbor map. Each CTA owns
// a tile of output rows, stages their neighbor rows in shared memory, and
// accumulates every tap any of its rows uses in registers. Forward and input
// gradient need no atomics, except that split-K (several CTAs per tile, chosen
// to fill the last wave) adds float32 partials atomically. Rows are grouped by
// the Gray rank of their tap mask so tiles skip the taps none of their rows
// use. The weight gradient reduces over per-tap (input, output) pair lists,
// split across CTAs with the same atomics. Under
// torch.use_deterministic_algorithms(True) each split writes its own float32
// slice instead (bounded scratch) and the slices are summed in a fixed order,
// so every result is bitwise repeatable.
//
// Supports any stride and any kernel size whose per-tap state fits the device's
// shared memory (several thousand taps), and the transposed direction of each, on
// CUDA SM80+ with float16, bfloat16 and float32 features (float32 computed as TF32,
// or as 3xTF32 when precise). The input gradient and the transpose run the same kernel over the
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
    // Row o of neighborMap holds output voxel o's K input rows (-1 if absent); see rowWords for
    // the compact map of kernels of more than 32 taps.
    torch::Tensor neighborMap; ///< int32 [N, K]
    torch::Tensor sortedRows;  ///< int32 [N], rows ordered by Gray rank of their tap mask
    /// sortedRows regrouped into windows of nearby rows, used when the gathered features are
    /// much larger than L2; undefined for maps of at most one window.
    torch::Tensor localRows;
    torch::Tensor rowMasks;    ///< int32 [N], bit k set when the row has a tap-k neighbor
    torch::Tensor pairIn;      ///< int32 [P], per tap the input row of each active pair
    torch::Tensor pairOut;     ///< int32 [P], per tap the output row, ascending
    torch::Tensor pairOffsets; ///< int32 [K + 1], CSR offsets of each tap's pairs (device)
    /// int32 [K, ceil(N / 256)], first pair of each tap at or after each 256-row boundary
    torch::Tensor pairBlockOffsets;

    int64_t outputTotalVoxels;  ///< Output rows (rows of neighborMap)
    int64_t featureTotalVoxels; ///< Input rows (the values neighborMap holds)
    int64_t kernelVolume;       ///< k0 * k1 * k2
    nanovdb::Coord kernelSize;  ///< [k0, k1, k2]

    /// Map from input rows to the output rows reading them per tap, with its row order and
    /// masks; no pair lists. Null for submanifold maps.
    std::shared_ptr<MaskedIGemmTopology const> reverse;
    // Kernels of more than 32 taps keep a compact map: neighborMap is int32 [P], each row's present
    // neighbors in tap order, and rowMasks is int32 [N, W] with W = ceil(K / 32) mask words.
    torch::Tensor
        neighborOffsets; ///< int32 [N + 1], start of each row in neighborMap (large kernels only)
    /// int32 [N, W, 2], (mask, base) per 32-tap word: tap t's neighbor is
    /// neighborMap[base + popc(mask & below(t % 32))] when its bit is set (large kernels only)
    torch::Tensor rowWords;
};

/// Build the map from @p outputGrid's voxels to @p featureGrid's voxels for convolution with
/// @p kernelSize and @p stride (fine = stride * coarse + tap - paddingBefore).
MaskedIGemmTopology maskedIGemmSparseConvTopology(GridBatchData const &featureGrid,
                                                  GridBatchData const &outputGrid,
                                                  nanovdb::Coord kernelSize,
                                                  nanovdb::Coord stride);

/// Forward: weights are a (C_out, C_in, k0, k1, k2) tensor, bias an optional [C_out] tensor
/// added in the epilogue. Weights and bias must match the features' device and dtype.
/// @p transposed runs the transposed convolution, from the topology's output rows to its
/// input rows. @p precise computes float32 as 3xTF32 (near-float32 accuracy) instead of TF32;
/// other dtypes ignore it. Returns [N_out, C_out].
torch::Tensor maskedIGemmSparseConv(torch::Tensor features,
                                    torch::Tensor weights,
                                    std::optional<torch::Tensor> bias,
                                    MaskedIGemmTopology const &topo,
                                    bool transposed = false,
                                    bool precise    = false);

/// Backward of maskedIGemmSparseConv with the same @p transposed and @p precise: returns
/// (grad_features [N, C_in], grad_weights shaped like weights). Each is an undefined tensor when
/// its need flag is false. @p gradOutput is cast to the features' dtype. grad_weights follows the
/// weights' layout: a view for module weights stored (k2, k1, k0, C_in, C_out), contiguous
/// otherwise.
std::tuple<torch::Tensor, torch::Tensor>
maskedIGemmSparseConvBackward(torch::Tensor gradOutput,
                              torch::Tensor features,
                              torch::Tensor weights,
                              MaskedIGemmTopology const &topo,
                              bool needFeatures,
                              bool needWeights,
                              bool transposed = false,
                              bool precise    = false);

/// Choices that are otherwise made from the grid size and the device, forced so tests can reach
/// every code path on any GPU. Unset fields leave the heuristics in charge; set counts must be at
/// least 1. For tests only.
struct MaskedIGemmTestOverrides {
    std::optional<int> splits;         ///< split-K count of the forward and input-gradient kernel
    std::optional<int>
        weightRowSplits;               ///< split the weight gradient by output rows, this many ways
    std::optional<bool> stagedBuilder; ///< leaf-staged neighbor-map builder where it applies
    std::optional<bool> localRowOrder; ///< Gray order windowed to nearby rows instead of global
};
void setMaskedIGemmTestOverrides(MaskedIGemmTestOverrides const &overrides);

/// True when maskedIGemmSparseConv can run these channel counts and dtype on @p device.
bool maskedIGemmSupports(int64_t inChannels,
                         int64_t outChannels,
                         torch::ScalarType dtype,
                         torch::Device device);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_CONVOLUTION_MASKEDIGEMM_H
