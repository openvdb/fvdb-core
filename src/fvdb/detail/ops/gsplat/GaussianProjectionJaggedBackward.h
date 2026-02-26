// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_GAUSSIANPROJECTIONJAGGEDBACKWARD_H
#define FVDB_DETAIL_OPS_GSPLAT_GAUSSIANPROJECTIONJAGGEDBACKWARD_H

#include <fvdb/detail/ops/gsplat/GaussianProjection.h>

#include <torch/types.h>

#include <tuple>

namespace fvdb {
namespace detail {
namespace ops {

/// @brief Calculate gradients for the jagged 3D to 2D Gaussian projection (backward pass)
///
/// This function computes the gradients of the 3D to 2D Gaussian projection with respect to
/// the input parameters when using jagged tensors for batch processing. It enables backpropagation
/// through the projection step in the Gaussian Splatting pipeline for scenes with variable
/// numbers of objects and cameras per batch.
///
/// @tparam DeviceType Device type template parameter (torch::kCUDA or torch::kCPU)
///
/// @param[in] gSizes Batch sizes for Gaussians [B]
/// @param[in] means 3D positions of Gaussians [M, 3]
/// @param[in] quats Quaternion rotations of Gaussians [M, 4] in format (x, y, z, w)
/// @param[in] scales Scale factors of Gaussians [M, 3] representing extent in each dimension
/// @param[in] cSizes Batch sizes for cameras [B]
/// @param[in] worldToCamMatrices Camera view matrices [BC, 4, 4]
/// @param[in] projectionMatrices Camera intrinsic matrices [BC, 3, 3]
/// @param[in] imageWidth Width of the output image in pixels
/// @param[in] imageHeight Height of the output image in pixels
/// @param[in] eps2d 2D projection epsilon for numerical stability
/// @param[in] radii Output radii from forward pass [M]
/// @param[in] conics Output conics from forward pass [M, 3]
/// @param[out] dLossDMeans2d Gradients with respect to projected 2D means [M, 2]
/// @param[out] dLossDDepths Gradients with respect to depths [M]
/// @param[out] dLossDConics Gradients with respect to conics [M, 3]
/// @param[in] worldToCamMatricesRequiresGrad Whether viewmats requires gradient
/// @param[in] ortho Whether orthographic projection was used in forward pass
///
/// @return std::tuple containing gradients of the loss function with respect to the input
/// parameters:
///         - 3D means [M, 3] - ∂L/∂means
///         - Quaternions [M, 4] - ∂L/∂quats
///         - Scales [M, 3] - ∂L/∂scales
///         - View matrices [BC, 4, 4] - ∂L/∂viewmats (if viewmats_requires_grad is true, otherwise
/// empty tensor)
///         - Camera intrinsics [BC, 3, 3] - ∂L/∂Ks
template <torch::DeviceType>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianProjectionJaggedBackward(const torch::Tensor &gSizes, // [B] gaussian sizes
                                         const torch::Tensor &means,  // [N, 3]
                                         const torch::Tensor &quats,  // [N, 4] optional
                                         const torch::Tensor &scales, // [N, 3] optional
                                         const torch::Tensor &cSizes, // [B] camera sizes
                                         const torch::Tensor &worldToCamMatrices, // [C, 4, 4]
                                         const torch::Tensor &projectionMatrices, // [C, 3, 3]
                                         const bool ortho,
                                         const uint32_t imageWidth,
                                         const uint32_t imageHeight,
                                         const float eps2d,
                                         const torch::Tensor &radii,         // [N]
                                         const torch::Tensor &conics,        // [N, 3]
                                         const torch::Tensor &dLossDMeans2d, // [N, 2]
                                         const torch::Tensor &dLossDDepths,  // [N]
                                         const torch::Tensor &dLossDConics,  // [N, 3]
                                         const bool worldToCamMatricesRequiresGrad);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_GSPLAT_GAUSSIANPROJECTIONJAGGEDBACKWARD_H
