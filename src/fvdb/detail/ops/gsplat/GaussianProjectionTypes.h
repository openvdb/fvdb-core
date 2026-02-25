// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_GAUSSIANPROJECTIONTYPES_H
#define FVDB_DETAIL_OPS_GSPLAT_GAUSSIANPROJECTIONTYPES_H

#include <cstdint>

namespace fvdb::detail::ops {

/// @brief Rolling shutter policy for camera projection / ray generation.
enum class RollingShutterType : int32_t { NONE = 0, VERTICAL = 1, HORIZONTAL = 2 };

/// @brief Camera model for projection (shared across projection and 3DGS rasterization paths).
///
/// Notes:
/// - `PINHOLE` and `ORTHOGRAPHIC` have no distortion.
/// - `OPENCV_*` variants are pinhole + OpenCV-style distortion, and expect packed coefficients.
enum class CameraModel : int32_t {
    // Pinhole intrinsics only (no distortion).
    PINHOLE = 0,

    // OpenCV variants which are just pinhole intrinsics + optional distortion (all of them use the
    // same [C,12] distortion coefficients layout: [k1,k2,k3,k4,k5,k6,p1,p2,s1,s2,s3,s4]).
    OPENCV_RADTAN_5            = 1, // polynomial radial (k1,k2,k3) + tangential (p1,p2)
    OPENCV_RATIONAL_8          = 2, // rational radial (k1..k6) + tangential (p1,p2)
    OPENCV_RADTAN_THIN_PRISM_9 = 3, // polynomial radial + tangential + thin-prism (s1..s4)
    OPENCV_THIN_PRISM_12       = 4, // rational radial + tangential + thin-prism (s1..s4)

    // Orthographic intrinsics (no distortion).
    ORTHOGRAPHIC = 5,
};

/// @brief Unscented Transform hyperparameters.
///
/// This kernel implements the canonical 3D UT with a fixed \(2D+1\) sigma point set (7 points).
/// The parameters here control the standard UT scaling / weighting.
struct UTParams {
    float alpha         = 0.1f; // Blending parameter for UT
    float beta          = 2.0f; // Scaling parameter for UT
    float kappa         = 0.0f; // Additional scaling parameter for UT
    float inImageMargin = 0.1f; // Margin for in-image check
    bool requireAllSigmaPointsInImage =
        true; // Require all sigma points to be in image to consider a Gaussian valid
};

} // namespace fvdb::detail::ops

#endif // FVDB_DETAIL_OPS_GSPLAT_GAUSSIANPROJECTIONTYPES_H
