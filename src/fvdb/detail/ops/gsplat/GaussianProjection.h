// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_GAUSSIANPROJECTION_H
#define FVDB_DETAIL_OPS_GSPLAT_GAUSSIANPROJECTION_H

#include <cstdint>

namespace fvdb::detail::ops {

/// @brief Rolling shutter policy for camera projection / ray generation.
enum class RollingShutterType : int32_t { NONE = 0, VERTICAL = 1, HORIZONTAL = 2 };

/// @brief Distortion/projection mode selector used by distortion-enabled camera ops.
///
/// Notes:
/// - `PINHOLE` and `ORTHOGRAPHIC` have no distortion.
/// - `OPENCV_*` variants are pinhole + OpenCV-style distortion, and expect packed coefficients.
enum class DistortionModel : int32_t {
    PINHOLE = 0,
    OPENCV_RADTAN_5 = 1,
    OPENCV_RATIONAL_8 = 2,
    OPENCV_RADTAN_THIN_PRISM_9 = 3,
    OPENCV_THIN_PRISM_12 = 4,
    ORTHOGRAPHIC = 5,
};

/// @brief Unscented Transform hyperparameters.
struct UTParams {
    float alpha         = 0.1f;
    float beta          = 2.0f;
    float kappa         = 0.0f;
    float inImageMargin = 0.1f;
    bool requireAllSigmaPointsInImage = true;
};

} // namespace fvdb::detail::ops

#endif // FVDB_DETAIL_OPS_GSPLAT_GAUSSIANPROJECTION_H
