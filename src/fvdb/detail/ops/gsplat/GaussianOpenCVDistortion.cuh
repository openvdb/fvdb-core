// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_GAUSSIANOPENCVDISTORTION_CUH
#define FVDB_DETAIL_OPS_GSPLAT_GAUSSIANOPENCVDISTORTION_CUH

#include <fvdb/detail/ops/gsplat/GaussianCameraModels.h>

#include <nanovdb/math/Math.h>

#include <cstdint>

namespace fvdb::detail::ops {

/// @brief Apply OpenCV-style distortion to normalized camera-plane coordinates.
///
/// This uses FVDB's packed coefficient layout:
/// `[k1,k2,k3,k4,k5,k6,p1,p2,s1,s2,s3,s4]`.
///
/// Notes:
/// - `CameraModel::PINHOLE` and `CameraModel::ORTHOGRAPHIC` ignore distortion.
/// - Unused coefficients for a given model are ignored (but still read from the packed array).
///
/// @param cameraModel Camera model selector.
/// @param p_normalized Normalized camera-plane coordinate.
///   - perspective: (x/z, y/z)
///   - orthographic: (x, y)
/// @param distortionCoeffs Pointer to packed coefficients (may be null for non-OpenCV models).
/// @param numCoeffs Number of coefficients per camera (0 for PINHOLE/ORTHOGRAPHIC, 12 for OPENCV).
template <typename T>
inline __device__ nanovdb::math::Vec2<T>
applyOpenCVDistortionPacked(const CameraModel cameraModel,
                            const nanovdb::math::Vec2<T> &p_normalized,
                            const T *distortionCoeffs,
                            const int64_t numCoeffs) {
    // For pinhole/orthographic, distortion is ignored.
    if (cameraModel == CameraModel::PINHOLE || cameraModel == CameraModel::ORTHOGRAPHIC ||
        numCoeffs == 0 || distortionCoeffs == nullptr) {
        return p_normalized;
    }

    // Packed OpenCV coefficient layout:
    // [k1,k2,k3,k4,k5,k6,p1,p2,s1,s2,s3,s4]
    // NOTE: for RADTAN_5 we use k1,k2,k3 and ignore k4..k6; for rational we use k1..k6.
    const T x  = p_normalized[0];
    const T y  = p_normalized[1];
    const T x2 = x * x;
    const T y2 = y * y;
    const T xy = x * y;
    const T r2 = x2 + y2;
    const T r4 = r2 * r2;
    const T r6 = r4 * r2;

    const T k1 = distortionCoeffs[0];
    const T k2 = distortionCoeffs[1];
    const T k3 = distortionCoeffs[2];
    const T k4 = distortionCoeffs[3];
    const T k5 = distortionCoeffs[4];
    const T k6 = distortionCoeffs[5];
    const T p1 = distortionCoeffs[6];
    const T p2 = distortionCoeffs[7];
    const T s1 = distortionCoeffs[8];
    const T s2 = distortionCoeffs[9];
    const T s3 = distortionCoeffs[10];
    const T s4 = distortionCoeffs[11];

    T radial = T(1);
    if (cameraModel == CameraModel::OPENCV_RATIONAL_8 ||
        cameraModel == CameraModel::OPENCV_THIN_PRISM_12) {
        const T num = T(1) + r2 * (k1 + r2 * (k2 + r2 * k3));
        const T den = T(1) + r2 * (k4 + r2 * (k5 + r2 * k6));
        radial      = (den != T(0)) ? (num / den) : T(0);
    } else if (cameraModel == CameraModel::OPENCV_RADTAN_5 ||
               cameraModel == CameraModel::OPENCV_RADTAN_THIN_PRISM_9) {
        radial = T(1) + k1 * r2 + k2 * r4 + k3 * r6;
    }

    T x_dist = x * radial;
    T y_dist = y * radial;

    // Tangential
    x_dist += T(2) * p1 * xy + p2 * (r2 + T(2) * x2);
    y_dist += p1 * (r2 + T(2) * y2) + T(2) * p2 * xy;

    // Thin prism
    if (cameraModel == CameraModel::OPENCV_THIN_PRISM_12 ||
        cameraModel == CameraModel::OPENCV_RADTAN_THIN_PRISM_9) {
        x_dist += s1 * r2 + s2 * r4;
        y_dist += s3 * r2 + s4 * r4;
    }

    return nanovdb::math::Vec2<T>(x_dist, y_dist);
}

/// @brief Iteratively undistort a packed OpenCV distortion model using fixed-point iteration.
///
/// This solves for `x` such that `applyOpenCVDistortionPacked(x) == p_distorted`.
/// It uses a small fixed iteration count (matching the UT kernel style) and is intended for use in
/// ray generation.
template <typename T>
inline __device__ nanovdb::math::Vec2<T>
undistortOpenCVPackedFixedPoint(const CameraModel cameraModel,
                                const nanovdb::math::Vec2<T> &p_distorted,
                                const T *distortionCoeffs,
                                const int64_t numCoeffs,
                                const int iters = 8) {
    if (cameraModel == CameraModel::PINHOLE || cameraModel == CameraModel::ORTHOGRAPHIC ||
        numCoeffs == 0 || distortionCoeffs == nullptr) {
        return p_distorted;
    }

    nanovdb::math::Vec2<T> x = p_distorted;
    for (int it = 0; it < iters; ++it) {
        const nanovdb::math::Vec2<T> x_dist =
            applyOpenCVDistortionPacked(cameraModel, x, distortionCoeffs, numCoeffs);
        const nanovdb::math::Vec2<T> err = x_dist - p_distorted;
        x[0] -= err[0];
        x[1] -= err[1];
    }
    return x;
}

} // namespace fvdb::detail::ops

#endif // FVDB_DETAIL_OPS_GSPLAT_GAUSSIANOPENCVDISTORTION_CUH
