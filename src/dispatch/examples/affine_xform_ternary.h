// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Ternary affine transformation example: y = R @ x + T
//
// This example tests the dispatch framework's ability to handle:
//   - Multiple input tensors with different ranks
//   - Broadcasting on the iteration dimension
//   - Views from packed storage (strided, non-contiguous)
//
// Terminology:
//   - Iteration shape: (N,) - we loop over N points
//   - Element shapes: R is (3,3), T is (3,), x is (3,), y is (3,)
//
// This is a "pointwise" operation in the geometric sense - transforming
// N 3D points where each point has associated structured elements.
//
#ifndef DISPATCH_EXAMPLES_AFFINE_XFORM_TERNARY_H
#define DISPATCH_EXAMPLES_AFFINE_XFORM_TERNARY_H

#include <torch/types.h>

namespace dispatch_examples {

// Affine transformation: y = R @ x + T
//
// Parameters:
//   R: (N, 3, 3) or (1, 3, 3) rotation matrices
//      If shape is (1, 3, 3), the same rotation is applied to all points.
//   T: (N, 3) or (1, 3) translation vectors
//      If shape is (1, 3), the same translation is applied to all points.
//   x: (N, 3) input positions
//
// Returns:
//   y: (N, 3) transformed positions where y[n] = R[n] @ x[n] + T[n]
//
// All tensors must have the same scalar type (float16, bfloat16, float32, float64).
// Tensors can be strided (non-contiguous).
//
torch::Tensor example_affine_xform(torch::Tensor R, torch::Tensor T, torch::Tensor x);

} // namespace dispatch_examples

#endif // DISPATCH_EXAMPLES_AFFINE_XFORM_TERNARY_H
