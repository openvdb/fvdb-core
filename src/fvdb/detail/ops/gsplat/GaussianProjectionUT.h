// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_GAUSSIANPROJECTIONUT_H
#define FVDB_DETAIL_OPS_GSPLAT_GAUSSIANPROJECTIONUT_H

#include <ATen/core/TensorBody.h>
#include <torch/types.h>

#include <tuple>

namespace fvdb {
namespace detail {
namespace ops {

enum class RollingShutterType { NONE = 0, VERTICAL = 1, HORIZONTAL = 2 };

struct UTParams {
    float alpha         = 0.1f; // Blending parameter for UT
    float beta          = 2.0f; // Scaling parameter for UT
    float kappa         = 0.0f; // Additional scaling parameter for UT
    // For a 3D Unscented Transform with the standard \(2D+1\) formulation, D=3 => 7 points.
    // This implementation currently supports only this 3D case.
    int numSigmaPoints  = 7;
    float inImageMargin = 0.1f; // Margin for in-image check
    bool requireAllSigmaPointsInImage =
        true; // Require all sigma points to be in image to consider a Gaussian valid
};

/// @brief Project 3D Gaussians to 2D screen space pixel coordinates for rendering
///
/// This function transforms 3D Gaussians to 2D screen space by applying camera projections.
/// It computes the 2D means, depths, 2D covariance matrices (conics), and potentially compensation
/// factors to accurately represent the 3D Gaussians in 2D for later rasterization.
///
/// The origin of the 2D pixel coordinates is the top-left corner of the image, with positive x-axis
/// pointing to the right and positive y-axis pointing downwards.
///
/// @attention The output radii of 3D Gaussians that are discarded (due to clipping or projection
/// too small) are set to zero, but the other output values of discarded Gaussians are uninitialized
/// (undefined).
///
/// @tparam DeviceType Device type template parameter (torch::kCUDA or torch::kCPU)
///
/// @param[in] means 3D positions of Gaussians [N, 3] where N is number of Gaussians
/// @param[in] quats Quaternion rotations of Gaussians [N, 4] in format (w, x, y, z)
/// @param[in] scales Scale factors of Gaussians [N, 3] representing extent in each dimension
/// @param[in] worldToCamMatricesStart Camera view matrices at the start of the frame. Shape [C, 4,
/// 4] where C is number of cameras
/// @param[in] worldToCamMatricesEnd Camera view matrices at the end of the frame. Shape [C, 4, 4]
/// where C is number of cameras
/// @param[in] projectionMatrices Camera intrinsic matrices [C, 3, 3]
/// @param[in] rollingShutterType Type of rolling shutter effect to apply
/// @param[in] utParams Unscented Transform parameters
/// @param[in] radialCoeffs Radial distortion coefficients (OpenCV convention):
///   - Polynomial: k1,k2,k3 in a [C,3] tensor
///   - Rational:   k1..k6 in a [C,6] tensor
///   - None:       [C,0]
/// @param[in] tangentialCoeffs Tangential distortion coefficients (OpenCV convention) p1,p2 in a
///   [C,2] tensor (or [C,0] for none).
/// @param[in] thinPrismCoeffs Thin prism distortion coefficients (OpenCV convention) s1..s4.
///   This kernel accepts [C,4] (full OpenCV) or [C,0] for none.
/// @param[in] imageWidth Width of the output image in pixels
/// @param[in] imageHeight Height of the output image in pixels
/// @param[in] eps2d 2D projection epsilon for numerical stability
/// @param[in] nearPlane Near clipping plane distance
/// @param[in] farPlane Far clipping plane distance
/// @param[in] minRadius2d Radius clipping value to limit the maximum size of projected Gaussians
/// @param[in] calcCompensations Whether to calculate view-dependent compensation factors
/// @param[in] ortho Whether to use orthographic projection instead of perspective
///
/// @return std::tuple containing:
///         - 2D projected Gaussian centers [C, N, 2]
///         - Depths of Gaussians [C, N]
///         - Covariance matrices in conic form [C, N, 3] representing (a, b, c) in ax² + 2bxy + cy²
///         - Radii of 2D Gaussians [C, N]
///         - Compensation factors [C, N] (if calc_compensations is true, otherwise empty tensor)
template <torch::DeviceType>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianProjectionForwardUT(
    const torch::Tensor &means,                   // [N, 3]
    const torch::Tensor &quats,                   // [N, 4]
    const torch::Tensor &scales,                  // [N, 3]
    const torch::Tensor &worldToCamMatricesStart, // [C, 4, 4]
    const torch::Tensor &worldToCamMatricesEnd,   // [C, 4, 4]
    const torch::Tensor &projectionMatrices,      // [C, 3, 3]
    const RollingShutterType rollingShutterType,
    const UTParams &utParams,
    const torch::Tensor &radialCoeffs,     // [C, 6] or [C, 3] or [C, 0] distortion coefficients
    const torch::Tensor &tangentialCoeffs, // [C, 2] or [C, 0] distortion coefficients
    const torch::Tensor &thinPrismCoeffs,  // [C, 4] or [C, 0] distortion coefficients
    const int64_t imageWidth,
    const int64_t imageHeight,
    const float eps2d,
    const float nearPlane,
    const float farPlane,
    const float minRadius2d,
    const bool calcCompensations,
    const bool ortho);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_GSPLAT_GAUSSIANPROJECTIONUT_H
