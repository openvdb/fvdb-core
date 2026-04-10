// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef FVDB_DETAIL_OPS_ADDNOISETOGAUSSIANMEANS_H
#define FVDB_DETAIL_OPS_ADDNOISETOGAUSSIANMEANS_H

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

/// @brief Add noise to Gaussian means, scaled by opacity and covariance.
///
/// Dispatches to the appropriate device implementation (CPU, CUDA, or PrivateUse1)
/// based on the device of the input tensors.
///
/// @param[in,out] means 3D positions of Gaussians [N, 3] (modified in place)
/// @param[in] logScales Log scale factors of Gaussians [N, 3]
/// @param[in] logitOpacities Logit opacity values of Gaussians [N]
/// @param[in] quats Quaternion rotations of Gaussians [N, 4]
/// @param[in] noiseScale Overall noise magnitude
/// @param[in] t Opacity threshold for noise scaling sigmoid
/// @param[in] k Sharpness of the noise scaling sigmoid
void add_noise_to_gaussian_means(torch::Tensor &means,                // [N, 3] input/output
                                 const torch::Tensor &logScales,      // [N, 3]
                                 const torch::Tensor &logitOpacities, // [N]
                                 const torch::Tensor &quats,          // [N, 4]
                                 const float noiseScale,
                                 const float t,
                                 const float k);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_ADDNOISETOGAUSSIANMEANS_H
