// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef DISPATCH_EXAMPLES_SOFTPLUS_H
#define DISPATCH_EXAMPLES_SOFTPLUS_H

#include <torch/types.h>

namespace dispatch_examples {

/// Softplus: out[i] = log(1 + exp(beta * x[i])) / beta, or x[i] if beta * x[i] > threshold.
///
/// Demonstrates the for_each + views pattern:
///   - One __hostdev__ scalar function, all devices
///   - One kernel function, all devices / scalar types / contiguities
///   - No manual CUDA kernels, no manual device guard, no manual grid/block
///   - Views handle strided tensors transparently
///   - Contiguity resolved at dispatch time
torch::Tensor example_softplus(torch::Tensor input, double beta = 1.0, double threshold = 20.0);

/// Pre-allocated output variant (for benchmarking without allocation overhead)
void example_softplus_out(torch::Tensor input,
                          torch::Tensor output,
                          double beta      = 1.0,
                          double threshold = 20.0);

} // namespace dispatch_examples

#endif // DISPATCH_EXAMPLES_SOFTPLUS_H
