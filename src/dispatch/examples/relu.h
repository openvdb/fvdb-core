// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef DISPATCH_EXAMPLES_RELU_H
#define DISPATCH_EXAMPLES_RELU_H

#include <torch/types.h>

namespace dispatch_examples {

// out-of-place
torch::Tensor relu(torch::Tensor input);

// in-place
torch::Tensor relu_(torch::Tensor input);

} // namespace dispatch_examples

#endif // DISPATCH_EXAMPLES_RELU_H
