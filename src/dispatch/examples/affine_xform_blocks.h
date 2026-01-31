// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef DISPATCH_EXAMPLES_AFFINE_XFORM_BLOCKS_H
#define DISPATCH_EXAMPLES_AFFINE_XFORM_BLOCKS_H

#include <torch/extension.h>

namespace dispatch_examples {

// Block-based affine transformation: y = R @ x + T
//
// Inputs:
//   R: (N, 3, 3) or (1, 3, 3) - rotation matrices
//   T: (N, 3) or (1, 3) - translation vectors
//   x: (N, 3) - input positions
//
// Output:
//   y: (N, 3) - transformed positions
//
// This example demonstrates the block-based accessor framework.
torch::Tensor example_affine_xform_blocks(torch::Tensor R, torch::Tensor T, torch::Tensor x);

} // namespace dispatch_examples

#endif // DISPATCH_EXAMPLES_AFFINE_XFORM_BLOCKS_H
