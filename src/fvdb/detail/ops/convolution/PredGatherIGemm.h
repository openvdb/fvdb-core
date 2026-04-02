// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file PredGatherIGemm.h
/// @brief Predicated-gather implicit-GEMM sparse convolution backend.
///
/// SM80 (Ampere) CUTLASS/CuTe IGEMM that processes one output leaf node per
/// CTA, using predicated cp.async gather loads for the activation (B) matrix.
/// Activations and weights use TF32 tensor-core arithmetic; output is FP32.
///
/// Constraints:
///   - Forward pass only (no transpose, no gradient)
///   - Input and output channel counts must be multiples of 32
///   - Float32 tensors only (internally promoted to TF32)
///   - Uniform kernel sizes {3, 5, 7} and uniform strides {1, 2}
#ifndef FVDB_DETAIL_OPS_CONVOLUTION_PREDGATHERIGEMM_H
#define FVDB_DETAIL_OPS_CONVOLUTION_PREDGATHERIGEMM_H

#include <fvdb/detail/GridBatchData.h>

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

/// @brief Forward pass of predicated-gather IGEMM sparse convolution.
///
/// Accepts tensors in standard fVDB layout (0-indexed features, weights as
/// [K, C, T, R, S]).  Index padding and weight permutation are handled
/// internally.
///
/// @param features      Input features, shape [N_in, C], float32, on CUDA.
/// @param weights       Kernel weights, shape [K, C, ks, ks, ks], float32.
/// @param feature_grid  NanoVDB grid batch for the input (feature) voxels.
/// @param output_grid   NanoVDB grid batch for the output voxels.
/// @param kernel_size   Uniform spatial kernel extent (3, 5, or 7).
/// @param stride        Uniform convolution stride (1 or 2).
/// @return              Output features, shape [N_out, K], float32.
torch::Tensor predGatherIGemmSparseConv(torch::Tensor features,
                                        torch::Tensor weights,
                                        GridBatchData const &feature_grid,
                                        GridBatchData const &output_grid,
                                        int kernel_size,
                                        int stride);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_CONVOLUTION_PREDGATHERIGEMM_H
