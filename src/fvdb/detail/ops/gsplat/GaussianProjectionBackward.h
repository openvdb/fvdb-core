// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_GAUSSIANPROJECTIONBACKWARD_H
#define FVDB_DETAIL_OPS_GSPLAT_GAUSSIANPROJECTIONBACKWARD_H

#include <torch/types.h>

#include <tuple>

namespace fvdb {
namespace detail {
namespace ops {

/// @brief Calculate gradients for the 3D to 2D Gaussian projection (backward pass)
///
/// This function computes the gradients of the 3D to 2D Gaussian projection with respect to
/// the input parameters: 3D means, quaternions, scales, view matrices, and optionally camera
/// intrinsics. It enables backpropagation through the projection step in the Gaussian Splatting
/// pipeline.
///
/// @tparam DeviceType Device type template parameter (torch::kCUDA or torch::kCPU)
///
/// @param[in] means 3D positions of Gaussians [N, 3]
/// @param[in] quats Quaternion rotations of Gaussians [N, 4] in format (x, y, z, w)
/// @param[in] scales Scale factors of Gaussians [N, 3] representing extent in each dimension
/// @param[in] worldToCamMatrices Camera view matrices [C, 4, 4]
/// @param[in] projectionMatrices Camera intrinsic matrices [C, 3, 3]
/// @param[in] compensations View-dependent compensation factors [N, 6] (optional)
/// @param[in] imageWidth Width of the image in pixels
/// @param[in] imageHeight Height of the image in pixels
/// @param[in] eps2d 2D projection epsilon for numerical stability
/// @param[in] radii Output radii from forward pass [C, N]
/// @param[in] conics Output conics from forward pass [C, N, 3]
/// @param[out] dLossDMeans2d Gradients with respect to projected 2D means [C, N, 2]
/// @param[out] dLossDDepths Gradients with respect to depths [C, N]
/// @param[out] dLossDConics Gradients with respect to conics [C, N, 3]
/// @param[out] dLossDCompensations Gradients with respect to compensations [C, N] (optional)
/// @param[in] worldToCamMatricesRequiresGrad Whether viewmats requires gradient
/// @param[in] ortho Whether orthographic projection was used in forward pass
/// @param[in] outNormalizeddLossdMeans2dNormAccum Optional output for normalized gradients tracked
/// across backward passes
/// @param[in] outNormalizedMaxRadiiAccum Optional output for maximum radii tracked across backward
/// passes
/// @param[in] outGradientStepCounts Optional output for the number of times each gradient was
/// counted tracked across backward passes
///
/// @return std::tuple containing gradients of the loss function with respect to the input
/// parameters:
///         - 3D means [N, 3] - ∂L/∂means
///         - Quaternions [N, 4] - ∂L/∂quats
///         - Scales [N, 3] - ∂L/∂scales
///         - View matrices [C, 4, 4] - ∂L/∂viewmats
///         - Camera intrinsics [C, 3, 3] - ∂L/∂Ks
template <torch::DeviceType>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianProjectionBackward(
    const torch::Tensor &means,                       // [N, 3]
    const torch::Tensor &quats,                       // [N, 4]
    const torch::Tensor &scales,                      // [N, 3]
    const torch::Tensor &worldToCamMatrices,          // [C, 4, 4]
    const torch::Tensor &projectionMatrices,          // [C, 3, 3]
    const at::optional<torch::Tensor> &compensations, // [N, 6] optional
    const uint32_t imageWidth,
    const uint32_t imageHeight,
    const float eps2d,
    const torch::Tensor &radii,                             // [C, N]
    const torch::Tensor &conics,                            // [C, N, 3]
    const torch::Tensor &dLossDMeans2d,                     // [C, N, 2]
    const torch::Tensor &dLossDDepths,                      // [C, N]
    const torch::Tensor &dLossDConics,                      // [C, N, 3]
    const at::optional<torch::Tensor> &dLossDCompensations, // [C, N] optional
    const bool worldToCamMatricesRequiresGrad,
    const bool ortho,
    at::optional<torch::Tensor> outNormalizeddLossdMeans2dNormAccum = std::nullopt,
    at::optional<torch::Tensor> outNormalizedMaxRadiiAccum          = std::nullopt,
    at::optional<torch::Tensor> outGradientStepCounts               = std::nullopt);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_GSPLAT_GAUSSIANPROJECTIONBACKWARD_H
