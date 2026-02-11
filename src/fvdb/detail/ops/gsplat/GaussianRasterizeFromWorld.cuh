// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_GAUSSIANRASTERIZEFROMWORLD_CUH
#define FVDB_DETAIL_OPS_GSPLAT_GAUSSIANRASTERIZEFROMWORLD_CUH

#include <fvdb/detail/ops/gsplat/GaussianCameraModels.h>
#include <fvdb/detail/ops/gsplat/GaussianUtils.cuh>

#include <nanovdb/math/Math.h>

#include <cstdint>

namespace fvdb::detail::ops {

constexpr __device__ float kAlphaThreshold = 0.999f;

template <typename T> struct WorldRay {
    nanovdb::math::Vec3<T> origin;
    nanovdb::math::Vec3<T> dir; // expected normalized for perspective
    bool valid = true;
};

template <typename T>
inline __device__ nanovdb::math::Vec2<T>
applyOpenCVDistortionPacked(const CameraModel cameraModel,
                            const nanovdb::math::Vec2<T> &p_normalized,
                            const T *distortionCoeffs, // [12] packed layout
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

template <typename T>
inline __device__ nanovdb::math::Vec2<T>
undistortOpenCVPackedFixedPoint(const CameraModel cameraModel,
                                const nanovdb::math::Vec2<T> &p_distorted,
                                const T *distortionCoeffs,
                                const int64_t numCoeffs) {
    if (cameraModel == CameraModel::PINHOLE || cameraModel == CameraModel::ORTHOGRAPHIC ||
        numCoeffs == 0 || distortionCoeffs == nullptr) {
        return p_distorted;
    }

    nanovdb::math::Vec2<T> x = p_distorted;
    // Fixed small iteration count (matches UT kernel style).
    constexpr int kIters = 8;
    for (int it = 0; it < kIters; ++it) {
        const nanovdb::math::Vec2<T> x_dist =
            applyOpenCVDistortionPacked(cameraModel, x, distortionCoeffs, numCoeffs);
        const nanovdb::math::Vec2<T> err = x_dist - p_distorted;
        x[0] -= err[0];
        x[1] -= err[1];
    }
    return x;
}

template <typename T>
inline __device__ nanovdb::math::Vec3<T>
normalizeSafe(const nanovdb::math::Vec3<T> &v) {
    const T n2 = v.dot(v);
    if (n2 > T(0)) {
        return v * (T(1) / sqrt(n2));
    }
    return nanovdb::math::Vec3<T>(T(0), T(0), T(0));
}

template <typename T>
inline __device__ nanovdb::math::Vec3<T>
normalizeSafeVJP(const nanovdb::math::Vec3<T> &x, const nanovdb::math::Vec3<T> &v_y) {
    const T n2 = x.dot(x);
    if (!(n2 > T(0))) {
        return nanovdb::math::Vec3<T>(T(0), T(0), T(0));
    }
    const T n    = sqrt(n2);
    const T invn = T(1) / n;
    // v_x = (I/n - x x^T / n^3) v_y
    const T invn3 = invn * invn * invn;
    const T xdotv = x.dot(v_y);
    return v_y * invn - x * (xdotv * invn3);
}

template <typename T>
inline __device__ nanovdb::math::Mat3<T>
mat3FromWorldToCam(const T *m44 /* row-major 4x4 */) {
    return nanovdb::math::Mat3<T>(
        m44[0], m44[1], m44[2], m44[4], m44[5], m44[6], m44[8], m44[9], m44[10]);
}

template <typename T>
inline __device__ nanovdb::math::Vec3<T>
vec3FromWorldToCamT(const T *m44 /* row-major 4x4 */) {
    return nanovdb::math::Vec3<T>(m44[3], m44[7], m44[11]);
}

template <typename T>
inline __device__ void
interpolateWorldToCam(const nanovdb::math::Mat3<T> &R0,
                      const nanovdb::math::Vec3<T> &t0,
                      const nanovdb::math::Mat3<T> &R1,
                      const nanovdb::math::Vec3<T> &t1,
                      const T u,
                      nanovdb::math::Mat3<T> &R_out,
                      nanovdb::math::Vec3<T> &t_out) {
    t_out                           = t0 + u * (t1 - t0);
    const nanovdb::math::Vec4<T> q0 = rotationMatrixToQuaternion<T>(R0);
    const nanovdb::math::Vec4<T> q1 = rotationMatrixToQuaternion<T>(R1);
    const nanovdb::math::Vec4<T> qi = nlerpQuaternionShortestPath<T>(q0, q1, u);
    R_out                           = quaternionToRotationMatrix<T>(qi);
}

template <typename T>
inline __device__ WorldRay<T>
pixelToWorldRay(const uint32_t row,
                const uint32_t col,
                const uint32_t imageWidth,
                const uint32_t imageHeight,
                const uint32_t imageOriginW,
                const uint32_t imageOriginH,
                const nanovdb::math::Mat3<T> &R_wc_start,
                const nanovdb::math::Vec3<T> &t_wc_start,
                const nanovdb::math::Mat3<T> &R_wc_end,
                const nanovdb::math::Vec3<T> &t_wc_end,
                const nanovdb::math::Mat3<T> &K,
                const T *distCoeffs,
                const int64_t numDistCoeffs,
                const RollingShutterType rollingShutterType,
                const CameraModel cameraModel) {
    // Pixel center in crop coordinates.
    const T px = T(col) + T(imageOriginW) + T(0.5);
    const T py = T(row) + T(imageOriginH) + T(0.5);

    // Rolling shutter time based on pixel location.
    T u = T(0);
    if (rollingShutterType == RollingShutterType::VERTICAL) {
        u = floor(py) / max(T(1), T(imageHeight - 1));
    } else if (rollingShutterType == RollingShutterType::HORIZONTAL) {
        u = floor(px) / max(T(1), T(imageWidth - 1));
    }
    u = min(T(1), max(T(0), u));

    nanovdb::math::Mat3<T> R_wc;
    nanovdb::math::Vec3<T> t_wc;
    if (rollingShutterType == RollingShutterType::NONE) {
        R_wc = R_wc_start;
        t_wc = t_wc_start;
    } else {
        interpolateWorldToCam<T>(R_wc_start, t_wc_start, R_wc_end, t_wc_end, u, R_wc, t_wc);
    }

    // Invert rigid transform to get camera->world.
    const nanovdb::math::Mat3<T> R_cw = R_wc.transpose();

    const T fx = K[0][0];
    const T fy = K[1][1];
    const T cx = K[0][2];
    const T cy = K[1][2];
    const nanovdb::math::Vec2<T> p_distorted((px - cx) / fx, (py - cy) / fy);
    const nanovdb::math::Vec2<T> p =
        undistortOpenCVPackedFixedPoint(cameraModel, p_distorted, distCoeffs, numDistCoeffs);

    WorldRay<T> ray;
    if (cameraModel == CameraModel::ORTHOGRAPHIC) {
        // Parallel rays; origin varies with pixel.
        const nanovdb::math::Vec3<T> o_cam(p[0], p[1], T(0));
        const nanovdb::math::Vec3<T> d_cam(T(0), T(0), T(1));
        // p_world = R_cw * (p_cam - t_wc)
        ray.origin = R_cw * (o_cam - t_wc);
        ray.dir    = R_cw * d_cam;
        ray.dir    = normalizeSafe(ray.dir);
        ray.valid  = true;
        return ray;
    }

    // Perspective (pinhole / OpenCV distorted pinhole): origin at camera center.
    const nanovdb::math::Vec3<T> d_cam = normalizeSafe(nanovdb::math::Vec3<T>(p[0], p[1], T(1)));
    ray.origin                         = R_cw * (nanovdb::math::Vec3<T>(T(0), T(0), T(0)) - t_wc);
    ray.dir                            = normalizeSafe(R_cw * d_cam);
    ray.valid                          = true;
    return ray;
}

template <typename T>
inline __device__ nanovdb::math::Vec4<T>
quatLoadWxyz(const T *q) {
    return nanovdb::math::Vec4<T>(q[0], q[1], q[2], q[3]);
}

template <typename T>
inline __device__ nanovdb::math::Mat3<T>
computeIsclRot(const nanovdb::math::Vec4<T> &quat_wxyz, const nanovdb::math::Vec3<T> &scale) {
    const nanovdb::math::Mat3<T> R = quaternionToRotationMatrix<T>(quat_wxyz);
    const nanovdb::math::Mat3<T> S_inv(
        T(1) / scale[0], T(0), T(0), T(0), T(1) / scale[1], T(0), T(0), T(0), T(1) / scale[2]);
    return S_inv * R.transpose();
}

template <typename T>
inline __device__ void
isclRotVectorJacobianProduct(const nanovdb::math::Vec4<T> &quat_wxyz,
                             const nanovdb::math::Vec3<T> &scale,
                             const nanovdb::math::Mat3<T> &dLossDIsclRot,
                             nanovdb::math::Vec4<T> &dLossDQuat,
                             nanovdb::math::Vec3<T> &dLossDLogScale) {
    // TODO(fvdb): Consider returning {dLossDQuat, dLossDLogScale} to match other VJP helpers in
    // this module (e.g. `transformCovarianceWorldToCamVectorJacobianProduct`), rather than using
    // output reference parameters.
    // iscl_rot = S_inv * R^T, with S_inv = diag(1/scale)
    const nanovdb::math::Mat3<T> R = quaternionToRotationMatrix<T>(quat_wxyz);
    const nanovdb::math::Vec3<T> invScale(T(1) / scale[0], T(1) / scale[1], T(1) / scale[2]);

    // For D = A * B, dA = G * B^T, dB = A^T * G.
    // Here A = S_inv (diag), B = R^T.
    const nanovdb::math::Mat3<T> dA = dLossDIsclRot * R; // since (R^T)^T = R
    const nanovdb::math::Mat3<T> A(
        invScale[0], T(0), T(0), T(0), invScale[1], T(0), T(0), T(0), invScale[2]);
    const nanovdb::math::Mat3<T> dB = A * dLossDIsclRot;
    const nanovdb::math::Mat3<T> dR = dB.transpose(); // B = R^T

    dLossDQuat = quaternionToRotationMatrixVectorJacobianProduct<T>(quat_wxyz, dR);

    // Diagonal of dA gives gradients w.r.t invScale.
    const nanovdb::math::Vec3<T> dInvScale(dA[0][0], dA[1][1], dA[2][2]);
    // invScale = 1/scale => dlogScale = dscale * scale = -dInvScale / scale
    dLossDLogScale = nanovdb::math::Vec3<T>(
        -dInvScale[0] * invScale[0], -dInvScale[1] * invScale[1], -dInvScale[2] * invScale[2]);
}

} // namespace fvdb::detail::ops

#endif // FVDB_DETAIL_OPS_GSPLAT_GAUSSIANRASTERIZEFROMWORLD_CUH
