// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Old-style GELU implementation using fvdb's ForEach utilities.
// This demonstrates the traditional approach for comparison with the new dispatch framework.
//
#ifndef FVDB_BENCHMARKS_GELU_COMPARISON_GELU_OLD_CUH
#define FVDB_BENCHMARKS_GELU_COMPARISON_GELU_OLD_CUH

#include <torch/types.h>

namespace gelu_comparison {

/// @brief Old-style GELU implementation using forEachTensorElementChannelCUDA.
///
/// This demonstrates the traditional fvdb pattern which requires:
/// - AT_DISPATCH_FLOATING_TYPES_AND_HALF for dtype dispatch
/// - Manual device dispatch (CUDA vs CPU)
/// - packed_accessor32 for tensor access
/// - Functor with signature (elementIdx, channelIdx, accessor, ...)
/// - Only handles contiguous tensors (no stride optimization)
///
/// @param input Input tensor (must be contiguous, 1D)
/// @return Output tensor with GELU applied
torch::Tensor gelu_old(torch::Tensor input);

/// @brief In-place old-style GELU
torch::Tensor gelu_old_(torch::Tensor input);

} // namespace gelu_comparison

#endif // FVDB_BENCHMARKS_GELU_COMPARISON_GELU_OLD_CUH
