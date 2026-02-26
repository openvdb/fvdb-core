// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_GAUSSIANPROJECTIONJAGGEDFORWARD_H
#define FVDB_DETAIL_OPS_GSPLAT_GAUSSIANPROJECTIONJAGGEDFORWARD_H

#include <fvdb/detail/ops/gsplat/GaussianProjection.h>

#include <torch/types.h>

#include <tuple>

namespace fvdb {
namespace detail {
namespace ops {

/// @brief Project 3D Gaussians to 2D screen space using jagged tensors for batched processing
///
/// This function transforms batches of 3D Gaussians to 2D screen space by applying camera
/// projections. It handles jagged (variable-sized) inputs for efficient batch processing, where
/// each batch element may contain a different number of Gaussians and cameras.
///
/// @attention The output radii of 3D Gaussians that are discarded (due to clipping or projection
/// too small) are set to zero, but the other output values of discarded Gaussians are uninitialized
/// (undefined).
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
/// @param[in] nearPlane Near clipping plane distance
/// @param[in] farPlane Far clipping plane distance
/// @param[in] minRadius2d Radius clipping value to limit the maximum size of projected Gaussians
/// @param[in] ortho Whether to use orthographic projection instead of perspective
///
/// @return std::tuple containing:
///         - 2D projected Gaussian centers [M, 2]
///         - Depths of Gaussians [M]
///         - Covariance matrices in conic form [M, 3] representing (a, b, c) in ax² + 2bxy + cy²
///         - Radii of 2D Gaussians [M]
///         - Flattened camera indices [M] indicating which camera each projection corresponds to
template <torch::DeviceType>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianProjectionJaggedForward(const torch::Tensor &gSizes, // [B] gaussian sizes
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
                                        const float nearPlane,
                                        const float farPlane,
                                        const float minRadius2d);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_GSPLAT_GAUSSIANPROJECTIONJAGGEDFORWARD_H
