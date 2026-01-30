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

// Camera model for projection in the UT kernel.
//
// Today, all supported camera models are pinhole intrinsics + optional OpenCV-style distortion.
//
// Distortion coefficients are supplied as a single tensor `distortionCoeffs` and interpreted
// according to this enum for the OpenCV variants.
enum class CameraModel : int32_t {
    // Pinhole intrinsics only (no distortion).
    PINHOLE = 0,

    // OpenCV variants which are just pinhole intrinsics + optional distortion (all of them use the
    // same [C,12] distortion coefficients layout: [k1,k2,k3,k4,k5,k6,p1,p2,s1,s2,s3,s4]).
    OPENCV_RADTAN_5            = 1, // polynomial radial (k1,k2,k3) + tangential (p1,p2)).
    OPENCV_RATIONAL_8          = 2, // rational radial (k1..k6) + tangential (p1,p2)).
    OPENCV_RADTAN_THIN_PRISM_9 = 3, // polynomial radial + tangential + thin-prism (s1..s4)).
    OPENCV_THIN_PRISM_12       = 4, // rational radial + tangential + thin-prism (s1..s4)).
};

struct UTParams {
    float alpha         = 0.1f; // Blending parameter for UT
    float beta          = 2.0f; // Scaling parameter for UT
    float kappa         = 0.0f; // Additional scaling parameter for UT
    float inImageMargin = 0.1f; // Margin for in-image check
    bool requireAllSigmaPointsInImage =
        true; // Require all sigma points to be in image to consider a Gaussian valid
};

/// @brief Project 3D Gaussians to 2D screen space pixel coordinates for rendering using the
/// Unscented Transform (UT) algorithm.
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
/// The UT algorithm is a non-parametric method for approximating the mean and covariance of a
/// probability distribution. It is used to project 3D Gaussians to 2D screen space by applying
/// camera projections.
///
/// The UT algorithm is based on the following steps:
/// 1. Generate a set of sigma points around the mean of the Gaussian.
/// 2. Project the sigma points to 2D screen space.
/// 3. Compute the mean and covariance of the projected sigma points.
/// 4. Use the mean and covariance to compute the 2D means, depths, 2D covariance matrices (conics),
/// and potentially compensation factors.
///
/// @tparam DeviceType Device type template parameter (torch::kCUDA or torch::kCPU)
///
/// @param[in] means 3D positions of Gaussians [N, 3] where N is number of Gaussians
/// @param[in] quats Quaternion rotations of Gaussians [N, 4] in format (w, x, y, z)
/// @param[in] logScales Log-scale factors of Gaussians [N, 3] (natural log), representing extent in
/// each dimension
/// @param[in] worldToCamMatricesStart Camera view matrices at the start of the frame. Shape [C, 4,
/// 4] where C is number of cameras
/// @param[in] worldToCamMatricesEnd Camera view matrices at the end of the frame. Shape [C, 4, 4]
/// where C is number of cameras
/// @param[in] projectionMatrices Camera intrinsic matrices [C, 3, 3]
/// @param[in] rollingShutterType Type of rolling shutter effect to apply
/// @param[in] utParams Unscented Transform parameters
/// @param[in] cameraModel Camera model used to interpret `distortionCoeffs`.
/// @param[in] distortionCoeffs Distortion coefficients for each camera.
///   - CameraModel::PINHOLE: ignored (use [C,0] or [C,K] tensor).
///   - CameraModel::OPENCV_*: expects [C,12] coefficients in the following order:
///       [k1,k2,k3,k4,k5,k6,p1,p2,s1,s2,s3,s4]
///     where k1..k6 are radial (rational), p1,p2 are tangential, and s1..s4 are thin-prism.
/// @param[in] imageWidth Width of the output image in pixels
/// @param[in] imageHeight Height of the output image in pixels
/// @param[in] eps2d 2D projection epsilon for numerical stability
/// @param[in] nearPlane Near clipping plane distance
/// @param[in] farPlane Far clipping plane distance
/// @param[in] minRadius2d Minimum 2D radius threshold; Gaussians with projected radius <= this
/// value are clipped/discarded
/// @param[in] calcCompensations Whether to calculate view-dependent compensation factors
/// @param[in] ortho Whether to use orthographic projection instead of perspective
///
/// @return std::tuple containing:
///         - Radii of 2D Gaussians [C, N]
///         - 2D projected Gaussian centers [C, N, 2]
///         - Depths of Gaussians [C, N]
///         - Covariance matrices in conic form [C, N, 3] representing (a, b, c) in ax² + 2bxy + cy²
///         - Compensation factors [C, N] (if calc_compensations is true, otherwise empty tensor)
template <torch::DeviceType>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianProjectionForwardUT(
    const torch::Tensor &means,                   // [N, 3]
    const torch::Tensor &quats,                   // [N, 4]
    const torch::Tensor &logScales,               // [N, 3]
    const torch::Tensor &worldToCamMatricesStart, // [C, 4, 4]
    const torch::Tensor &worldToCamMatricesEnd,   // [C, 4, 4]
    const torch::Tensor &projectionMatrices,      // [C, 3, 3]
    const RollingShutterType rollingShutterType,
    const UTParams &utParams,
    const CameraModel cameraModel,
    const torch::Tensor &distortionCoeffs, // [C, 12] for OPENCV_*, or [C, 0] for NONE
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
