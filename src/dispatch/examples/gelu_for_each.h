// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef DISPATCH_EXAMPLES_GELU_FOR_EACH_H
#define DISPATCH_EXAMPLES_GELU_FOR_EACH_H

#include <torch/types.h>

namespace dispatch_examples {

// out-of-place
torch::Tensor example_gelu_for_each(torch::Tensor input);

// out-of-place with pre-allocated output (for benchmarking without allocation)
void example_gelu_for_each_out(torch::Tensor input, torch::Tensor output);

// in-place
torch::Tensor example_gelu_for_each_(torch::Tensor input);

} // namespace dispatch_examples

#endif // DISPATCH_EXAMPLES_GELU_FOR_EACH_H
