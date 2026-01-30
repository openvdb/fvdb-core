// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#include <fvdb/detail/ops/gsplat/GaussianProjectionUT.h>
#include <fvdb/detail/ops/gsplat/GaussianUtils.cuh>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/Nvtx.h>
#include <fvdb/detail/utils/cuda/GridDim.h>

#include <c10/core/DeviceType.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>

#include <cmath>
#include <tuple>

namespace fvdb::detail::ops {

namespace {

// OpenCV camera distortion conventions:
// https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
// Distortion coefficients are interpreted as:
// - Radial (rational): k1..k6
// - Tangential: p1, p2
// - Thin prism: s1..s4
//
/// @brief OpenCV camera model (pinhole intrinsics + distortion).
///
/// This is an internal helper used by the UT projection kernel.
/// It owns the camera intrinsics pointer `K` and the distortion coefficient pointers and can
/// project `p_cam -> pixel`.
/// @see https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
///
/// @tparam T Scalar type
template <typename T> class OpenCVCameraModel {
  public:
    using Vec2 = nanovdb::math::Vec2<T>;
    using Vec3 = nanovdb::math::Vec3<T>;
    using Mat3 = nanovdb::math::Mat3<T>;

    /// @brief Internal distortion evaluation mode.
    ///
    /// This is intentionally separate from the public API `CameraModel` to keep the math paths
    /// explicit and avoid implying that `CameraModel` is “just distortion” (since it also includes
    /// projection as well).
    enum class Model : uint8_t {
        NONE          = 0,  // no distortion
        RADTAN_5      = 5,  // k1,k2,p1,p2,k3 (polynomial radial up to r^6)
        RATIONAL_8    = 8,  // k1,k2,p1,p2,k3,k4,k5,k6 (rational radial)
        RADTAN_THIN_9 = 9,  // RADTAN_5 + thin prism s1..s4 (polynomial radial + thin-prism)
        THIN_PRISM_12 = 12, // RATIONAL_8 + s1,s2,s3,s4
    };

    /// @brief Construct a camera model for projection.
    ///
    /// For OpenCV models, coefficients are read from a per-camera packed layout:
    /// `[k1,k2,k3,k4,k5,k6,p1,p2,s1,s2,s3,s4]`.
    ///
    /// For `CameraModel::PINHOLE` / `CameraModel::ORTHOGRAPHIC`, `distortionCoeffs` is ignored.
    ///
    /// Preconditions are asserted on-device (trap) rather than returning status codes.
    ///
    /// @param[in] cameraModel Public camera model selector.
    /// @param[in] K_in Pointer to 3x3 intrinsics matrix (typically points into shared memory).
    /// @param[in] distortionCoeffs Pointer to per-camera distortion coefficients (layout depends on
    ///            cameraModel).
    __device__ __forceinline__
    OpenCVCameraModel(const CameraModel cameraModel, const Mat3 *K_in, const T *distortionCoeffs)
        : K(K_in) {
        deviceAssertOrTrap(K != nullptr);

        // ORTHOGRAPHIC is implemented as pinhole intrinsics without the perspective divide.
        // Distortion is intentionally not supported for orthographic projection.
        orthographic = (cameraModel == CameraModel::ORTHOGRAPHIC);

        if (cameraModel == CameraModel::ORTHOGRAPHIC) {
            radial = tangential = thinPrism = nullptr;
            numRadial = numTangential = numThinPrism = 0;
            model                                    = Model::NONE;
            return;
        }

        if (cameraModel == CameraModel::PINHOLE) {
            radial = tangential = thinPrism = nullptr;
            numRadial = numTangential = numThinPrism = 0;
            model                                    = Model::NONE;
            return;
        }

        // OpenCV models require the packed coefficient layout.
        deviceAssertOrTrap(distortionCoeffs != nullptr);
        const T *radial_in = distortionCoeffs + 0; // k1..k6
        const T *tang_in   = distortionCoeffs + 6; // p1,p2
        const T *thin_in   = distortionCoeffs + 8; // s1..s4

        if (cameraModel == CameraModel::OPENCV_RADTAN_5) {
            radial        = radial_in;
            numRadial     = 3; // k1,k2,k3 (polynomial)
            tangential    = tang_in;
            numTangential = 2;
            thinPrism     = nullptr;
            numThinPrism  = 0;
            model         = Model::RADTAN_5;
            return;
        }
        if (cameraModel == CameraModel::OPENCV_RATIONAL_8) {
            radial        = radial_in;
            numRadial     = 6; // k1..k6 (rational)
            tangential    = tang_in;
            numTangential = 2;
            thinPrism     = nullptr;
            numThinPrism  = 0;
            model         = Model::RATIONAL_8;
            return;
        }
        if (cameraModel == CameraModel::OPENCV_THIN_PRISM_12) {
            radial        = radial_in;
            numRadial     = 6; // k1..k6 (rational)
            tangential    = tang_in;
            numTangential = 2;
            thinPrism     = thin_in;
            numThinPrism  = 4; // s1..s4
            model         = Model::THIN_PRISM_12;
            return;
        }
        if (cameraModel == CameraModel::OPENCV_RADTAN_THIN_PRISM_9) {
            // Polynomial radial + thin-prism; ignore k4..k6 by construction.
            radial        = radial_in;
            numRadial     = 3; // k1,k2,k3 (polynomial)
            tangential    = tang_in;
            numTangential = 2;
            thinPrism     = thin_in;
            numThinPrism  = 4; // s1..s4
            model         = Model::RADTAN_THIN_9;
            return;
        }

        // Unknown camera model: should be unreachable if host validation is correct.
        deviceAssertOrTrap(false);
    }

    /// @brief Project a 3D point in camera coordinates to pixel coordinates.
    ///
    /// - Perspective (`PINHOLE`/`OPENCV_*`): normalize by depth \(x/z, y/z\).
    /// - Orthographic (`ORTHOGRAPHIC`): no divide; uses \(x,y\) directly.
    ///
    /// @param[in] p_cam Point in camera coordinates.
    /// @return Pixel coordinates (u,v).
    __device__ Vec2
    project(const Vec3 &p_cam) const {
        // Normalize to camera plane.
        Vec2 p_normalized;
        if (orthographic) {
            p_normalized = Vec2(p_cam[0], p_cam[1]);
        } else {
            const T z_inv = T(1) / max(p_cam[2], T(1e-6));
            p_normalized  = Vec2(p_cam[0] * z_inv, p_cam[1] * z_inv);
        }

        const Vec2 p_distorted = applyDistortion(p_normalized);

        // Project to pixel coordinates.
        const Mat3 &K_ref = *K;
        const T fx        = K_ref[0][0];
        const T fy        = K_ref[1][1];
        const T cx        = K_ref[0][2];
        const T cy        = K_ref[1][2];
        return Vec2(fx * p_distorted[0] + cx, fy * p_distorted[1] + cy);
    }

  private:
    /// @brief Device-side assert that always traps on failure.
    ///
    /// @param[in] cond Condition that must hold.
    __device__ __forceinline__ static void
    deviceAssertOrTrap(const bool cond) {
        if (!cond) {
            // `assert()` is typically compiled out in release builds; use a trap to guarantee a
            // loud failure when invariants are violated.
            asm volatile("trap;");
        }
    }

    // Camera intrinsics pointer (typically points into shared memory).
    // This avoids copying a Mat3 into registers per-thread.
    const Mat3 *K     = nullptr;
    bool orthographic = false;

    // Coefficients for the distortion model.
    const T *radial     = nullptr;     // k1..k6 (but k4..k6 only used in rational model)
    int numRadial       = 0;           // Number of radial coefficients
    const T *tangential = nullptr;     // p1,p2
    int numTangential   = 0;           // Number of tangential coefficients
    const T *thinPrism  = nullptr;     // s1..s4
    int numThinPrism    = 0;           // Number of thin prism coefficients
    Model model         = Model::NONE; // Distortion model

    /// @brief Read a coefficient if present, otherwise return 0.
    ///
    /// @param[in] ptr Coefficient array (may be null).
    /// @param[in] n Number of coefficients in ptr.
    /// @param[in] i Index to read.
    /// @return Coefficient value or 0 if out-of-range / null.
    __host__ __device__ inline T
    coeffOrZero(const T *ptr, const int n, const int i) const {
        return (ptr != nullptr && i >= 0 && i < n) ? ptr[i] : T(0);
    }

    /// @brief Apply OpenCV distortion to a normalized camera-plane point.
    ///
    /// Input coordinates are assumed to already be normalized to the camera plane:
    /// - perspective: \((x/z, y/z)\)
    /// - orthographic: \((x, y)\)
    ///
    /// @param[in] p_normalized Normalized camera-plane coordinates.
    /// @return Distorted normalized coordinates.
    __device__ Vec2
    applyDistortion(const Vec2 &p_normalized) const {
        const T x  = p_normalized[0];
        const T y  = p_normalized[1];
        const T x2 = x * x;
        const T y2 = y * y;
        const T xy = x * y;

        const T r2 = x2 + y2;
        const T r4 = r2 * r2;
        const T r6 = r4 * r2;

        // Radial distortion.
        T radial_dist = T(1);
        if (model == Model::RATIONAL_8 || model == Model::THIN_PRISM_12) {
            const T k1  = coeffOrZero(radial, numRadial, 0);
            const T k2  = coeffOrZero(radial, numRadial, 1);
            const T k3  = coeffOrZero(radial, numRadial, 2);
            const T k4  = coeffOrZero(radial, numRadial, 3);
            const T k5  = coeffOrZero(radial, numRadial, 4);
            const T k6  = coeffOrZero(radial, numRadial, 5);
            const T num = T(1) + r2 * (k1 + r2 * (k2 + r2 * k3));
            const T den = T(1) + r2 * (k4 + r2 * (k5 + r2 * k6));
            radial_dist = (den != T(0)) ? (num / den) : T(0);
        } else if (model == Model::RADTAN_5) {
            // Polynomial radial (up to k3 / r^6).
            const T k1  = coeffOrZero(radial, numRadial, 0);
            const T k2  = coeffOrZero(radial, numRadial, 1);
            const T k3  = coeffOrZero(radial, numRadial, 2);
            radial_dist = T(1) + k1 * r2 + k2 * r4 + k3 * r6;
        } else if (model == Model::RADTAN_THIN_9) {
            // Polynomial radial (same as RADTAN_5). Thin-prism terms are applied below.
            const T k1  = coeffOrZero(radial, numRadial, 0);
            const T k2  = coeffOrZero(radial, numRadial, 1);
            const T k3  = coeffOrZero(radial, numRadial, 2);
            radial_dist = T(1) + k1 * r2 + k2 * r4 + k3 * r6;
        }

        T x_dist = x * radial_dist;
        T y_dist = y * radial_dist;

        // Tangential distortion.
        const T p1 = coeffOrZero(tangential, numTangential, 0);
        const T p2 = coeffOrZero(tangential, numTangential, 1);
        x_dist += T(2) * p1 * xy + p2 * (r2 + T(2) * x2);
        y_dist += p1 * (r2 + T(2) * y2) + T(2) * p2 * xy;

        // Thin-prism distortion.
        if (model == Model::THIN_PRISM_12 || model == Model::RADTAN_THIN_9) {
            const T s1 = coeffOrZero(thinPrism, numThinPrism, 0);
            const T s2 = coeffOrZero(thinPrism, numThinPrism, 1);
            const T s3 = coeffOrZero(thinPrism, numThinPrism, 2);
            const T s4 = coeffOrZero(thinPrism, numThinPrism, 3);
            x_dist += s1 * r2 + s2 * r4;
            y_dist += s3 * r2 + s4 * r4;
        }

        return Vec2(x_dist, y_dist);
    }
};

/// @brief UT-local rigid transform (cached rotation + translation).
///
/// Quaternion is stored as \([w,x,y,z]\) and is assumed to represent a rotation.
/// The corresponding rotation matrix \(R(q)\) is cached to avoid recomputing it for every point
/// transform (UT sigma points, rolling-shutter iterations, depth culls, etc.).
template <typename T> struct RigidTransform {
    nanovdb::math::Mat3<T> R;
    nanovdb::math::Vec4<T> q;
    nanovdb::math::Vec3<T> t;

    /// @brief Default constructor (identity transform).
    ///
    /// Initializes to unit quaternion \([1,0,0,0]\) and zero translation.
    __device__
    RigidTransform()
        : R(nanovdb::math::Mat3<T>(nanovdb::math::Vec3<T>(T(1), T(0), T(0)),
                                   nanovdb::math::Vec3<T>(T(0), T(1), T(0)),
                                   nanovdb::math::Vec3<T>(T(0), T(0), T(1)))),
          q(T(1), T(0), T(0), T(0)), t(T(0), T(0), T(0)) {}

    /// @brief Construct from quaternion and translation.
    /// @param[in] q_in Rotation quaternion \([w,x,y,z]\).
    /// @param[in] t_in Translation vector.
    __device__
    RigidTransform(const nanovdb::math::Vec4<T> &q_in, const nanovdb::math::Vec3<T> &t_in)
        : R(quaternionToRotationMatrix(q_in)), q(q_in), t(t_in) {}

    /// @brief Construct from rotation matrix and translation.
    /// @param[in] R_in Rotation matrix.
    /// @param[in] t_in Translation vector.
    __device__
    RigidTransform(const nanovdb::math::Mat3<T> &R_in, const nanovdb::math::Vec3<T> &t_in)
        : R(R_in), q(rotationMatrixToQuaternion<T>(R_in)), t(t_in) {}

    /// @brief Apply the transform to a 3D point: \(R(q)\,p + t\).
    /// @param[in] p_world Point to transform.
    /// @return Transformed point.
    __device__ __forceinline__ nanovdb::math::Vec3<T>
    apply(const nanovdb::math::Vec3<T> &p_world) const {
        // p_cam = R * p_world + t
        return R * p_world + t;
    }

    /// @brief Interpolate between two rigid transforms.
    ///
    /// Translation is linearly interpolated; rotation uses NLERP along the shortest arc.
    ///
    /// @param[in] u Interpolation parameter in \([0,1]\).
    /// @param[in] start Start transform.
    /// @param[in] end End transform.
    /// @return Interpolated transform.
    static inline __device__ RigidTransform<T>
    interpolate(const T u, const RigidTransform<T> &start, const RigidTransform<T> &end) {
        const nanovdb::math::Vec3<T> t_interp = start.t + u * (end.t - start.t);
        const nanovdb::math::Vec4<T> q_interp = nlerpQuaternionShortestPath<T>(start.q, end.q, u);
        return RigidTransform<T>(q_interp, t_interp);
    }
};

/// @brief Projection status for a single world point.
///
/// The kernel treats `BehindCamera` as a hard failure (discontinuous projection), while
/// `OutOfBounds` may still be usable depending on UTParams.
enum class ProjStatus : uint8_t { BehindCamera, OutOfBounds, InImage };

/// @brief World-space point -> pixel transform with rolling shutter.
///
/// This wraps the camera model, rolling shutter policy, and in-image bounds checks used when
/// projecting UT sigma points.
///
/// @tparam ScalarType Scalar type (float).
template <typename ScalarType> struct WorldToPixelTransform {
    using Vec2 = nanovdb::math::Vec2<ScalarType>;
    using Vec3 = nanovdb::math::Vec3<ScalarType>;
    using Mat3 = nanovdb::math::Mat3<ScalarType>;

    const OpenCVCameraModel<ScalarType> *camera = nullptr;
    RollingShutterType rollingShutterType;
    int64_t imageWidth;
    int64_t imageHeight;
    ScalarType inImageMargin;
    RigidTransform<ScalarType> worldToCamStart;
    RigidTransform<ScalarType> worldToCamEnd;

    /// @brief Helper: whether a projection status is in-image.
    /// @param[in] s Projection status.
    /// @return True if s is InImage.
    __device__ __forceinline__ static bool
    isInImage(const ProjStatus s) {
        return s == ProjStatus::InImage;
    }

    /// @brief Transform a world-space point with a given world->cam transform and project to pixel.
    ///
    /// @param[in] p_world World-space point.
    /// @param[in] xf World->camera transform.
    /// @param[out] out_pix Pixel coordinate output (always written).
    /// @return Projection status.
    __device__ __forceinline__ ProjStatus
    projectWithTransform(const Vec3 &p_world,
                         const RigidTransform<ScalarType> &xf,
                         Vec2 &out_pix) const {
        const Vec3 p_cam = xf.apply(p_world);
        // Reject points with z<=0 to avoid projecting behind the camera. For ORTHOGRAPHIC this is a
        // policy choice (not a mathematical necessity), but we keep it consistent across models.
        if (p_cam[2] <= ScalarType(0)) {
            // Ensure deterministic output to avoid UB on callers that assign/read even on invalid
            // projections. This value is ignored when we treat BehindCamera as a hard reject.
            out_pix = Vec2(ScalarType(0), ScalarType(0));
            return ProjStatus::BehindCamera;
        }

        out_pix                   = camera->project(p_cam);
        const ScalarType margin_x = ScalarType(imageWidth) * inImageMargin;
        const ScalarType margin_y = ScalarType(imageHeight) * inImageMargin;
        const bool in_img =
            (out_pix[0] >= -margin_x) && (out_pix[0] < ScalarType(imageWidth) + margin_x) &&
            (out_pix[1] >= -margin_y) && (out_pix[1] < ScalarType(imageHeight) + margin_y);
        return in_img ? ProjStatus::InImage : ProjStatus::OutOfBounds;
    }

    /// @brief Project a world-space point to pixel coordinates.
    ///
    /// For rolling shutter modes, this uses a small fixed-point iteration that estimates shutter
    /// time from the current pixel coordinate (row/col -> time).
    ///
    /// @param[in] p_world World-space point.
    /// @param[out] out_pixel Pixel coordinate output (always written).
    /// @return Projection status.
    __device__ __forceinline__ ProjStatus
    projectWorldPoint(const Vec3 &p_world, Vec2 &out_pixel) const {
        // Rolling shutter: iterate pose based on the current pixel estimate (row/col -> time).

        // Start/end projections for initialization.
        const RigidTransform<ScalarType> &pose_start = worldToCamStart;
        const RigidTransform<ScalarType> &pose_end   = worldToCamEnd;
        Vec2 pix_start(ScalarType(0), ScalarType(0));
        Vec2 pix_end(ScalarType(0), ScalarType(0));
        const ProjStatus status_start = projectWithTransform(p_world, pose_start, pix_start);
        const ProjStatus status_end   = projectWithTransform(p_world, pose_end, pix_end);

        if (rollingShutterType == RollingShutterType::NONE) {
            out_pixel = pix_start;
            return status_start;
        }

        // If both endpoints are behind the camera, treat as a hard invalid (discontinuous).
        if (status_start == ProjStatus::BehindCamera && status_end == ProjStatus::BehindCamera) {
            out_pixel = pix_end;
            return ProjStatus::BehindCamera;
        }

        // If neither endpoint is in-image (but at least one is in front), treat as invalid.
        // (We require an in-image seed for the fixed-point iteration.)
        if (!isInImage(status_start) && !isInImage(status_end)) {
            out_pixel = (status_end != ProjStatus::BehindCamera) ? pix_end : pix_start;
            return ProjStatus::OutOfBounds;
        }

        Vec2 pix_prev = isInImage(status_start) ? pix_start : pix_end;
        // Fixed small iteration count (good enough for convergence in practice).
        constexpr int kIters = 6;
        for (int it = 0; it < kIters; ++it) {
            ScalarType t_rs = ScalarType(0);
            if (rollingShutterType == RollingShutterType::VERTICAL) {
                t_rs = floor(pix_prev[1]) / max(ScalarType(1), ScalarType(imageHeight - 1));
            } else if (rollingShutterType == RollingShutterType::HORIZONTAL) {
                t_rs = floor(pix_prev[0]) / max(ScalarType(1), ScalarType(imageWidth - 1));
            }
            t_rs = min(ScalarType(1), max(ScalarType(0), t_rs));
            const RigidTransform<ScalarType> pose_rs =
                RigidTransform<ScalarType>::interpolate(t_rs, worldToCamStart, worldToCamEnd);
            Vec2 pix_rs(ScalarType(0), ScalarType(0));
            const ProjStatus status_rs = projectWithTransform(p_world, pose_rs, pix_rs);
            pix_prev                   = pix_rs;
            if (status_rs == ProjStatus::BehindCamera) {
                out_pixel = pix_rs;
                return ProjStatus::BehindCamera;
            }
            if (!isInImage(status_rs)) {
                out_pixel = pix_rs;
                return ProjStatus::OutOfBounds;
            }
        }

        out_pixel = pix_prev;
        return ProjStatus::InImage;
    }
};

/// @brief Generate 3D UT sigma points and weights (fixed 7-point UT in 3D).
///
/// Sigma points are generated in **world space** from \((\mu, R, s)\) where \(R\) comes from the
/// input quaternion and \(s\) are the axis scales. For a 3D UT with the canonical \(2D+1\)
/// formulation, D=3 => 7 sigma points.
///
/// @tparam T Scalar type.
/// @param[in] mean_world Mean in world space.
/// @param[in] quat_wxyz Rotation quaternion \([w,x,y,z]\).
/// @param[in] scale_world Axis-aligned scale in world space (per-axis standard deviation).
/// @param[in] params UT hyperparameters.
/// @param[out] sigma_points Output sigma points (size 7).
/// @param[out] weights_mean UT mean weights (size 7).
/// @param[out] weights_cov UT covariance weights (size 7).
template <typename T>
__device__ void
generateWorldSigmaPoints(const nanovdb::math::Vec3<T> &mean_world,
                         const nanovdb::math::Vec4<T> &quat_wxyz,
                         const nanovdb::math::Vec3<T> &scale_world,
                         const UTParams &params,
                         nanovdb::math::Vec3<T> *sigma_points, // [7]
                         T *weights_mean,                      // [7]
                         T *weights_cov) {                     // [7]
    constexpr int D = 3;
    // This kernel currently supports only the canonical 3D UT with 2D+1 points.
    // (We keep the arrays fixed-size for performance and simplicity.)
    const T alpha  = T(params.alpha);
    const T beta   = T(params.beta);
    const T kappa  = T(params.kappa);
    const T lambda = alpha * alpha * (T(D) + kappa) - T(D);
    const T denom  = T(D) + lambda;

    // Rotation matrix from quaternion. NOTE: `quaternionToRotationMatrix` expects [w,x,y,z].
    const nanovdb::math::Mat3<T> R = quaternionToRotationMatrix<T>(quat_wxyz);

    sigma_points[0] = mean_world;
    weights_mean[0] = lambda / denom;
    weights_cov[0]  = lambda / denom + (T(1) - alpha * alpha + beta);

    const T wi = T(1) / (T(2) * denom);
    for (int i = 0; i < 2 * D; ++i) {
        weights_mean[i + 1] = wi;
        weights_cov[i + 1]  = wi;
    }

    // sqrt(D + lambda) scaling
    const T gamma = sqrt(max(T(0), denom));

    // For covariance C = R * diag(scale^2) * R^T, the columns of R are the singular vectors.
    // Generate sigma points: mean +/- gamma * scale[i] * col_i(R)
    for (int i = 0; i < D; ++i) {
        const nanovdb::math::Vec3<T> col_i(R[0][i], R[1][i], R[2][i]);
        const nanovdb::math::Vec3<T> delta = (gamma * scale_world[i]) * col_i;
        sigma_points[i + 1]                = mean_world + delta;
        sigma_points[i + 1 + D]            = mean_world - delta;
    }
}

/// @brief Reconstruct a 2D covariance matrix from projected sigma points.
///
/// This computes \(\Sigma = \sum_i w_i (x_i-\mu)(x_i-\mu)^T\).
///
/// @tparam T Scalar type.
/// @param[in] projected_points Projected sigma points (size num_points).
/// @param[in] weights_cov Covariance weights (size num_points).
/// @param[in] num_points Number of sigma points.
/// @param[in] mean2d Precomputed 2D mean.
/// @return 2x2 covariance matrix.
template <typename T>
__device__ nanovdb::math::Mat2<T>
reconstructCovarianceFromSigmaPoints(const nanovdb::math::Vec2<T> *projected_points,
                                     const T *weights_cov,
                                     int num_points,
                                     const nanovdb::math::Vec2<T> &mean2d) {
    nanovdb::math::Mat2<T> covar2d(T(0), T(0), T(0), T(0));
    for (int i = 0; i < num_points; ++i) {
        const nanovdb::math::Vec2<T> diff = projected_points[i] - mean2d;
        covar2d[0][0] += weights_cov[i] * diff[0] * diff[0];
        covar2d[0][1] += weights_cov[i] * diff[0] * diff[1];
        covar2d[1][0] += weights_cov[i] * diff[1] * diff[0];
        covar2d[1][1] += weights_cov[i] * diff[1] * diff[1];
    }
    return covar2d;
}

} // namespace

/// @brief CUDA kernel functor for UT forward projection.
///
/// This struct owns tensor accessors, shared memory pointers, and scalar configuration for
/// projecting N gaussians into C camera views.
template <typename ScalarType> struct ProjectionForwardUT {
    using Mat3 = nanovdb::math::Mat3<ScalarType>;
    using Vec3 = nanovdb::math::Vec3<ScalarType>;
    using Vec4 = nanovdb::math::Vec4<ScalarType>;
    using Mat2 = nanovdb::math::Mat2<ScalarType>;
    using Vec2 = nanovdb::math::Vec2<ScalarType>;

    // Scalar Inputs
    const int64_t C;
    const int64_t N;
    const int32_t mImageWidth;
    const int32_t mImageHeight;
    const ScalarType mEps2d;
    const ScalarType mNearPlane;
    const ScalarType mFarPlane;
    const ScalarType mRadiusClip;
    const RollingShutterType mRollingShutterType;
    const UTParams mUTParams;
    const CameraModel mCameraModel;
    const int64_t
        mNumDistortionCoeffs; // Number of distortion coeffs per camera (e.g. 12 for OPENCV)

    // Tensor Inputs
    const fvdb::TorchRAcc64<ScalarType, 2> mMeansAcc;                   // [N, 3]
    const fvdb::TorchRAcc64<ScalarType, 2> mQuatsAcc;                   // [N, 4]
    const fvdb::TorchRAcc64<ScalarType, 2> mLogScalesAcc;               // [N, 3]
    const fvdb::TorchRAcc32<ScalarType, 3> mWorldToCamMatricesStartAcc; // [C, 4, 4]
    const fvdb::TorchRAcc32<ScalarType, 3> mWorldToCamMatricesEndAcc;   // [C, 4, 4]
    const fvdb::TorchRAcc32<ScalarType, 3> mProjectionMatricesAcc;      // [C, 3, 3]
    const fvdb::TorchRAcc64<ScalarType, 2> mDistortionCoeffsAcc;        // [C, K]

    // Outputs
    fvdb::TorchRAcc64<int32_t, 2> mOutRadiiAcc;      // [C, N]
    fvdb::TorchRAcc64<ScalarType, 3> mOutMeans2dAcc; // [C, N, 2]
    fvdb::TorchRAcc64<ScalarType, 2> mOutDepthsAcc;  // [C, N]
    fvdb::TorchRAcc64<ScalarType, 3> mOutConicsAcc;  // [C, N, 3]

    // Optional Outputs (need to be pointers since they may be null)
    ScalarType *__restrict__ mOutCompensationsAcc; // [C, N] optional

    // Shared memory pointers
    Mat3 *__restrict__ projectionMatsShared             = nullptr;
    Mat3 *__restrict__ worldToCamRotMatsStartShared     = nullptr;
    Mat3 *__restrict__ worldToCamRotMatsEndShared       = nullptr;
    Vec3 *__restrict__ worldToCamTranslationStartShared = nullptr;
    Vec3 *__restrict__ worldToCamTranslationEndShared   = nullptr;
    ScalarType *__restrict__ distortionCoeffsShared     = nullptr;

    /// @brief Construct the functor with configuration and tensor references.
    ///
    /// @param[in] imageWidth Image width in pixels.
    /// @param[in] imageHeight Image height in pixels.
    /// @param[in] eps2d Blur epsilon added to covariance for numerical stability.
    /// @param[in] nearPlane Near-plane threshold for depth culling.
    /// @param[in] farPlane Far-plane threshold for depth culling.
    /// @param[in] minRadius2d Minimum radius threshold; smaller gaussians are discarded.
    /// @param[in] rollingShutterType Rolling shutter policy.
    /// @param[in] utParams UT hyperparameters.
    /// @param[in] cameraModel Camera model selector.
    /// @param[in] calcCompensations Whether to compute compensation factors.
    /// @param[in] means [N,3] tensor.
    /// @param[in] quats [N,4] tensor.
    /// @param[in] logScales [N,3] tensor.
    /// @param[in] worldToCamMatricesStart [C,4,4] tensor.
    /// @param[in] worldToCamMatricesEnd [C,4,4] tensor.
    /// @param[in] projectionMatrices [C,3,3] tensor.
    /// @param[in] distortionCoeffs [C,K] tensor (K=0 for PINHOLE/ORTHOGRAPHIC; K=12 for OPENCV).
    /// @param[out] outRadii [C,N] tensor.
    /// @param[out] outMeans2d [C,N,2] tensor.
    /// @param[out] outDepths [C,N] tensor.
    /// @param[out] outConics [C,N,3] tensor.
    /// @param[out] outCompensations [C,N] tensor (optional, may be undefined).
    ProjectionForwardUT(const int64_t imageWidth,
                        const int64_t imageHeight,
                        const ScalarType eps2d,
                        const ScalarType nearPlane,
                        const ScalarType farPlane,
                        const ScalarType minRadius2d,
                        const RollingShutterType rollingShutterType,
                        const UTParams &utParams,
                        const CameraModel cameraModel,
                        const bool calcCompensations,
                        const torch::Tensor &means,                   // [N, 3]
                        const torch::Tensor &quats,                   // [N, 4]
                        const torch::Tensor &logScales,               // [N, 3]
                        const torch::Tensor &worldToCamMatricesStart, // [C, 4, 4]
                        const torch::Tensor &worldToCamMatricesEnd,   // [C, 4, 4]
                        const torch::Tensor &projectionMatrices,      // [C, 3, 3]
                        const torch::Tensor &distortionCoeffs,        // [C, K]
                        torch::Tensor &outRadii,                      // [C, N]
                        torch::Tensor &outMeans2d,                    // [C, N, 2]
                        torch::Tensor &outDepths,                     // [C, N]
                        torch::Tensor &outConics,                     // [C, N, 3]
                        torch::Tensor &outCompensations               // [C, N] optional
                        )
        : C(projectionMatrices.size(0)), N(means.size(0)),
          mImageWidth(static_cast<int32_t>(imageWidth)),
          mImageHeight(static_cast<int32_t>(imageHeight)), mEps2d(eps2d), mNearPlane(nearPlane),
          mFarPlane(farPlane), mRadiusClip(minRadius2d), mRollingShutterType(rollingShutterType),
          mUTParams(utParams), mCameraModel(cameraModel),
          mNumDistortionCoeffs(distortionCoeffs.size(1)),
          mMeansAcc(means.packed_accessor64<ScalarType, 2, torch::RestrictPtrTraits>()),
          mQuatsAcc(quats.packed_accessor64<ScalarType, 2, torch::RestrictPtrTraits>()),
          mLogScalesAcc(logScales.packed_accessor64<ScalarType, 2, torch::RestrictPtrTraits>()),
          mWorldToCamMatricesStartAcc(
              worldToCamMatricesStart.packed_accessor32<ScalarType, 3, torch::RestrictPtrTraits>()),
          mWorldToCamMatricesEndAcc(
              worldToCamMatricesEnd.packed_accessor32<ScalarType, 3, torch::RestrictPtrTraits>()),
          mProjectionMatricesAcc(
              projectionMatrices.packed_accessor32<ScalarType, 3, torch::RestrictPtrTraits>()),
          mDistortionCoeffsAcc(
              distortionCoeffs.packed_accessor64<ScalarType, 2, torch::RestrictPtrTraits>()),
          mOutRadiiAcc(outRadii.packed_accessor64<int32_t, 2, torch::RestrictPtrTraits>()),
          mOutMeans2dAcc(outMeans2d.packed_accessor64<ScalarType, 3, torch::RestrictPtrTraits>()),
          mOutDepthsAcc(outDepths.packed_accessor64<ScalarType, 2, torch::RestrictPtrTraits>()),
          mOutConicsAcc(outConics.packed_accessor64<ScalarType, 3, torch::RestrictPtrTraits>()),
          mOutCompensationsAcc(outCompensations.defined() ? outCompensations.data_ptr<ScalarType>()
                                                          : nullptr) {}

    /// @brief Load per-camera matrices/coeffs into shared memory for faster access.
    ///
    /// Layout is `[K, R_start, R_end, t_start, t_end, distortionCoeffs]` per camera.
    inline __device__ void
    loadCameraInfoIntoSharedMemory() {
        // Load projection matrices and world-to-camera matrices into shared memory
        alignas(Mat3) extern __shared__ char sharedMemory[];

        // Keep a running pointer which we increment to assign shared memory blocks
        uint8_t *pointer = reinterpret_cast<uint8_t *>(sharedMemory);

        projectionMatsShared = reinterpret_cast<Mat3 *>(pointer);
        pointer += C * sizeof(Mat3);

        worldToCamRotMatsStartShared = reinterpret_cast<Mat3 *>(pointer);
        pointer += C * sizeof(Mat3);

        worldToCamRotMatsEndShared = reinterpret_cast<Mat3 *>(pointer);
        pointer += C * sizeof(Mat3);

        worldToCamTranslationStartShared = reinterpret_cast<Vec3 *>(pointer);
        pointer += C * sizeof(Vec3);

        worldToCamTranslationEndShared = reinterpret_cast<Vec3 *>(pointer);
        pointer += C * sizeof(Vec3);

        distortionCoeffsShared =
            mNumDistortionCoeffs > 0 ? reinterpret_cast<ScalarType *>(pointer) : nullptr;
        pointer += C * mNumDistortionCoeffs * sizeof(ScalarType);

        // Layout in element units:
        const int64_t projectionOffset = 0;
        const int64_t rotStartOffset   = projectionOffset + C * 9;
        const int64_t rotEndOffset     = rotStartOffset + C * 9;
        const int64_t transStartOffset = rotEndOffset + C * 9;
        const int64_t transEndOffset   = transStartOffset + C * 3;
        const int64_t distortionOffset = transEndOffset + C * 3;
        const int64_t totalElements    = distortionOffset + C * mNumDistortionCoeffs;

        for (int64_t i = threadIdx.x; i < totalElements; i += blockDim.x) {
            if (i < rotStartOffset) {
                const auto camId   = (i - projectionOffset) / 9;
                const auto entryId = (i - projectionOffset) % 9;
                const auto rowId   = entryId / 3;
                const auto colId   = entryId % 3;
                projectionMatsShared[camId][rowId][colId] =
                    mProjectionMatricesAcc[camId][rowId][colId];
            } else if (i < rotEndOffset) {
                const auto camId   = (i - rotStartOffset) / 9;
                const auto entryId = (i - rotStartOffset) % 9;
                const auto rowId   = entryId / 3;
                const auto colId   = entryId % 3;
                worldToCamRotMatsStartShared[camId][rowId][colId] =
                    mWorldToCamMatricesStartAcc[camId][rowId][colId];
            } else if (i < transStartOffset) {
                const auto camId   = (i - rotEndOffset) / 9;
                const auto entryId = (i - rotEndOffset) % 9;
                const auto rowId   = entryId / 3;
                const auto colId   = entryId % 3;
                worldToCamRotMatsEndShared[camId][rowId][colId] =
                    mWorldToCamMatricesEndAcc[camId][rowId][colId];
            } else if (i < transEndOffset) {
                const auto camId   = (i - transStartOffset) / 3;
                const auto entryId = (i - transStartOffset) % 3;
                worldToCamTranslationStartShared[camId][entryId] =
                    mWorldToCamMatricesStartAcc[camId][entryId][3];
            } else if (i < distortionOffset) {
                const auto camId   = (i - transEndOffset) / 3;
                const auto entryId = (i - transEndOffset) % 3;
                worldToCamTranslationEndShared[camId][entryId] =
                    mWorldToCamMatricesEndAcc[camId][entryId][3];
            } else if (mNumDistortionCoeffs > 0) {
                const auto baseIdx = i - distortionOffset;
                const auto camId   = baseIdx / mNumDistortionCoeffs;
                const auto entryId = baseIdx % mNumDistortionCoeffs;
                distortionCoeffsShared[camId * mNumDistortionCoeffs + entryId] =
                    mDistortionCoeffsAcc[camId][entryId];
            }
        }
    }

    /// @brief Project one gaussian for one camera.
    ///
    /// @param[in] idx Flattened index in \([0, C*N)\) mapping to (camId, gaussianId).
    inline __device__ void
    projectionForward(int64_t idx) {
        if (idx >= C * N) {
            return;
        }

        const int64_t camId      = idx / N;
        const int64_t gaussianId = idx % N;

        // Get camera parameters
        const Mat3 &projectionMatrix     = projectionMatsShared[camId];
        const Mat3 &worldToCamRotStart   = worldToCamRotMatsStartShared[camId];
        const Mat3 &worldToCamRotEnd     = worldToCamRotMatsEndShared[camId];
        const Vec3 &worldToCamTransStart = worldToCamTranslationStartShared[camId];
        const Vec3 &worldToCamTransEnd   = worldToCamTranslationEndShared[camId];
        const ScalarType *distortionCoeffs =
            (mNumDistortionCoeffs > 0) ? &distortionCoeffsShared[camId * mNumDistortionCoeffs]
                                       : nullptr;

        // Define the camera model (projection and distortion) using the shared memory pointers
        OpenCVCameraModel<ScalarType> camera(mCameraModel, &projectionMatrix, distortionCoeffs);

        // Define the world-to-camera transforms using the shared memory pointers at the start and
        // end of the shutter period
        const RigidTransform<ScalarType> worldToCamStart(worldToCamRotStart,
                                                         worldToCamTransStart); // t=0.0
        const RigidTransform<ScalarType> worldToCamEnd(worldToCamRotEnd,
                                                       worldToCamTransEnd);     // t=1.0

        // Get Gaussian parameters
        const Vec3 meanWorldSpace(
            mMeansAcc[gaussianId][0], mMeansAcc[gaussianId][1], mMeansAcc[gaussianId][2]);
        const Vec4 quat_wxyz(mQuatsAcc[gaussianId][0],
                             mQuatsAcc[gaussianId][1],
                             mQuatsAcc[gaussianId][2],
                             mQuatsAcc[gaussianId][3]);
        const Vec3 scale_world(exp(mLogScalesAcc[gaussianId][0]),
                               exp(mLogScalesAcc[gaussianId][1]),
                               exp(mLogScalesAcc[gaussianId][2]));

        // Depth culling should use the same shutter pose as projection:
        // - RollingShutterType::NONE: use start pose (t=0.0), matching
        //   `WorldToPixelTransform::projectWorldPoint` which uses the start transform when NONE.
        // - Rolling shutter modes: use center pose (t=0.5) as a conservative/stable cull.
        {
            const RigidTransform<ScalarType> shutter_pose =
                (mRollingShutterType == RollingShutterType::NONE)
                    ? worldToCamStart
                    : RigidTransform<ScalarType>::interpolate(
                          ScalarType(0.5), worldToCamStart, worldToCamEnd);
            const Vec3 meanCam = shutter_pose.apply(meanWorldSpace);
            if (meanCam[2] < mNearPlane || meanCam[2] > mFarPlane) {
                mOutRadiiAcc[camId][gaussianId] = 0;
                return;
            }
        }

        // Generate world-space sigma points (7) and UT weights (mean/cov).
        nanovdb::math::Vec3<ScalarType> sigma_points_world[7];
        ScalarType weights_mean[7];
        ScalarType weights_cov[7];
        generateWorldSigmaPoints(meanWorldSpace,
                                 quat_wxyz,
                                 scale_world,
                                 mUTParams,
                                 sigma_points_world,
                                 weights_mean,
                                 weights_cov);

        const WorldToPixelTransform<ScalarType> worldToPixel{&camera,
                                                             mRollingShutterType,
                                                             mImageWidth,
                                                             mImageHeight,
                                                             ScalarType(mUTParams.inImageMargin),
                                                             worldToCamStart,
                                                             worldToCamEnd};

        // Project sigma points through camera model
        nanovdb::math::Vec2<ScalarType> projected_points[7];
        bool valid_any                = false;
        constexpr int kNumSigmaPoints = 7;
        for (int i = 0; i < kNumSigmaPoints; ++i) {
            Vec2 pix;
            const ProjStatus status = worldToPixel.projectWorldPoint(sigma_points_world[i], pix);
            // Hard reject if any sigma point is behind the camera since the projection will be
            // discontinuous.
            if (status == ProjStatus::BehindCamera) {
                mOutRadiiAcc[camId][gaussianId] = 0;
                return;
            }
            const bool valid_i  = (status == ProjStatus::InImage);
            projected_points[i] = pix;
            valid_any |= valid_i;
            if (mUTParams.requireAllSigmaPointsInImage && !valid_i) {
                mOutRadiiAcc[camId][gaussianId] = 0;
                return;
            }
        }

        if (!mUTParams.requireAllSigmaPointsInImage && !valid_any) {
            mOutRadiiAcc[camId][gaussianId] = 0;
            return;
        }

        // Compute mean of projected points
        nanovdb::math::Vec2<ScalarType> mean2d(ScalarType(0), ScalarType(0));
        for (int i = 0; i < kNumSigmaPoints; ++i) {
            mean2d[0] += weights_mean[i] * projected_points[i][0];
            mean2d[1] += weights_mean[i] * projected_points[i][1];
        }

        // Reconstruct 2D covariance from projected sigma points
        Mat2 covar2d = reconstructCovarianceFromSigmaPoints(
            projected_points, weights_cov, kNumSigmaPoints, mean2d);

        // Add blur for numerical stability
        ScalarType compensation;
        const ScalarType det = addBlur(mEps2d, covar2d, compensation);
        if (det <= 0.f) {
            mOutRadiiAcc[camId][gaussianId] = 0;
            return;
        }

        const Mat2 covar2dInverse = covar2d.inverse();

        // Compute bounding box radius (similar to standard projection)
        const ScalarType b      = 0.5f * (covar2d[0][0] + covar2d[1][1]);
        const ScalarType tmp    = sqrtf(max(0.01f, b * b - det));
        const ScalarType v1     = b + tmp; // larger eigenvalue
        const ScalarType extend = 3.0f;    // 3 sigma
        ScalarType r1           = extend * sqrtf(v1);
        ScalarType radius_x     = ceilf(min(extend * sqrtf(covar2d[0][0]), r1));
        ScalarType radius_y     = ceilf(min(extend * sqrtf(covar2d[1][1]), r1));

        if (radius_x <= mRadiusClip && radius_y <= mRadiusClip) {
            mOutRadiiAcc[camId][gaussianId] = 0;
            return;
        }

        // Mask out gaussians outside the image region
        if (mean2d[0] + radius_x <= 0 || mean2d[0] - radius_x >= mImageWidth ||
            mean2d[1] + radius_y <= 0 || mean2d[1] - radius_y >= mImageHeight) {
            mOutRadiiAcc[camId][gaussianId] = 0;
            return;
        }

        // Write outputs (using radius_x for compatibility, but could use both)
        mOutRadiiAcc[camId][gaussianId]      = int32_t(max(radius_x, radius_y));
        mOutMeans2dAcc[camId][gaussianId][0] = mean2d[0];
        mOutMeans2dAcc[camId][gaussianId][1] = mean2d[1];
        // For depth we use the same shutter pose as the cull check above.
        {
            const ScalarType t_depth = (mRollingShutterType == RollingShutterType::NONE)
                                           ? ScalarType(0.0)
                                           : ScalarType(0.5);
            const RigidTransform<ScalarType> shutter_pose =
                RigidTransform<ScalarType>::interpolate(t_depth, worldToCamStart, worldToCamEnd);
            const Vec3 meanCam               = shutter_pose.apply(meanWorldSpace);
            mOutDepthsAcc[camId][gaussianId] = meanCam[2];
        }
        mOutConicsAcc[camId][gaussianId][0] = covar2dInverse[0][0];
        mOutConicsAcc[camId][gaussianId][1] = covar2dInverse[0][1];
        mOutConicsAcc[camId][gaussianId][2] = covar2dInverse[1][1];
        if (mOutCompensationsAcc != nullptr) {
            mOutCompensationsAcc[idx] = compensation;
        }
    }
};

/// @brief CUDA kernel wrapper for `ProjectionForwardUT`.
///
/// Each thread processes multiple (camera, gaussian) pairs in a grid-stride loop.
template <typename ScalarType>
__global__ __launch_bounds__(256) void
projectionForwardUTKernel(int64_t offset,
                          int64_t count,
                          ProjectionForwardUT<ScalarType> projectionForward) {
    projectionForward.loadCameraInfoIntoSharedMemory();
    __syncthreads();

    // parallelize over C * N
    for (auto idx = blockIdx.x * blockDim.x + threadIdx.x; idx < count;
         idx += blockDim.x * gridDim.x) {
        projectionForward.projectionForward(idx + offset);
    }
}

/// @brief CUDA specialization for UT forward projection dispatch.
///
/// Performs host-side validation and launches `projectionForwardUTKernel`.
template <>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianProjectionForwardUT<torch::kCUDA>(
    const torch::Tensor &means,                   // [N, 3]
    const torch::Tensor &quats,                   // [N, 4]
    const torch::Tensor &logScales,               // [N, 3]
    const torch::Tensor &worldToCamMatricesStart, // [C, 4, 4]
    const torch::Tensor &worldToCamMatricesEnd,   // [C, 4, 4]
    const torch::Tensor &projectionMatrices,      // [C, 3, 3]
    const RollingShutterType rollingShutterType,
    const UTParams &utParams,
    const CameraModel cameraModel,
    const torch::Tensor &distortionCoeffs, // [C,12] for OPENCV, [C,0] for NONE
    const int64_t imageWidth,
    const int64_t imageHeight,
    const float eps2d,
    const float nearPlane,
    const float farPlane,
    const float minRadius2d,
    const bool calcCompensations) {
    FVDB_FUNC_RANGE();

    TORCH_CHECK_VALUE(means.is_cuda(), "means must be a CUDA tensor");
    TORCH_CHECK_VALUE(quats.is_cuda(), "quats must be a CUDA tensor");
    TORCH_CHECK_VALUE(logScales.is_cuda(), "logScales must be a CUDA tensor");
    TORCH_CHECK_VALUE(worldToCamMatricesStart.is_cuda(),
                      "worldToCamMatricesStart must be a CUDA tensor");
    TORCH_CHECK_VALUE(worldToCamMatricesEnd.is_cuda(),
                      "worldToCamMatricesEnd must be a CUDA tensor");
    TORCH_CHECK_VALUE(projectionMatrices.is_cuda(), "projectionMatrices must be a CUDA tensor");
    TORCH_CHECK_VALUE(distortionCoeffs.is_cuda(), "distortionCoeffs must be a CUDA tensor");
    TORCH_CHECK_VALUE(distortionCoeffs.dim() == 2, "distortionCoeffs must be 2D");
    if (cameraModel == CameraModel::PINHOLE || cameraModel == CameraModel::ORTHOGRAPHIC) {
        // Distortion coefficients are ignored for these camera models.
        // (Intrinsics `projectionMatrices` are always used.)
    } else if (cameraModel == CameraModel::OPENCV_RADTAN_5 ||
               cameraModel == CameraModel::OPENCV_RATIONAL_8 ||
               cameraModel == CameraModel::OPENCV_RADTAN_THIN_PRISM_9 ||
               cameraModel == CameraModel::OPENCV_THIN_PRISM_12) {
        TORCH_CHECK_VALUE(distortionCoeffs.size(1) == 12,
                          "For CameraModel::OPENCV_* , distortionCoeffs must have shape [C,12] "
                          "as [k1,k2,k3,k4,k5,k6,p1,p2,s1,s2,s3,s4]");
    } else {
        TORCH_CHECK_VALUE(false, "Unknown CameraModel for GaussianProjectionForwardUT");
    }

    // This kernel implements only the canonical 3D UT with 2D+1 sigma points (7).

    const at::cuda::OptionalCUDAGuard device_guard(device_of(means));

    const auto N                = means.size(0);              // number of gaussians
    const auto C                = projectionMatrices.size(0); // number of cameras
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(means.device().index());

    TORCH_CHECK_VALUE(distortionCoeffs.size(0) == C,
                      "distortionCoeffs must have shape [C,K] matching projectionMatrices.size(0)");

    torch::Tensor outRadii   = torch::empty({C, N}, means.options().dtype(torch::kInt32));
    torch::Tensor outMeans2d = torch::empty({C, N, 2}, means.options());
    torch::Tensor outDepths  = torch::empty({C, N}, means.options());
    torch::Tensor outConics  = torch::empty({C, N, 3}, means.options());
    torch::Tensor outCompensations;
    if (calcCompensations) {
        outCompensations = torch::zeros({C, N}, means.options());
    }

    if (N == 0 || C == 0) {
        return std::make_tuple(outRadii, outMeans2d, outDepths, outConics, outCompensations);
    }

    using scalar_t = float;

    const size_t NUM_BLOCKS = GET_BLOCKS(C * N, 256);
    // Orthographic is supported only for CameraModel::ORTHOGRAPHIC (undistorted).

    const size_t SHARED_MEM_SIZE = C * (3 * sizeof(nanovdb::math::Mat3<scalar_t>) +
                                        2 * sizeof(nanovdb::math::Vec3<scalar_t>)) +
                                   C * distortionCoeffs.size(1) * sizeof(scalar_t);

    ProjectionForwardUT<scalar_t> projectionForward(imageWidth,
                                                    imageHeight,
                                                    eps2d,
                                                    nearPlane,
                                                    farPlane,
                                                    minRadius2d,
                                                    rollingShutterType,
                                                    utParams,
                                                    cameraModel,
                                                    calcCompensations,
                                                    means,
                                                    quats,
                                                    logScales,
                                                    worldToCamMatricesStart,
                                                    worldToCamMatricesEnd,
                                                    projectionMatrices,
                                                    distortionCoeffs,
                                                    outRadii,
                                                    outMeans2d,
                                                    outDepths,
                                                    outConics,
                                                    outCompensations);

    projectionForwardUTKernel<scalar_t>
        <<<NUM_BLOCKS, 256, SHARED_MEM_SIZE, stream>>>(0, C * N, projectionForward);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(outRadii, outMeans2d, outDepths, outConics, outCompensations);
}

/// @brief CPU specialization (not implemented).
template <>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianProjectionForwardUT<torch::kCPU>(
    const torch::Tensor &means,                   // [N, 3]
    const torch::Tensor &quats,                   // [N, 4]
    const torch::Tensor &logScales,               // [N, 3]
    const torch::Tensor &worldToCamMatricesStart, // [C, 4, 4]
    const torch::Tensor &worldToCamMatricesEnd,   // [C, 4, 4]
    const torch::Tensor &projectionMatrices,      // [C, 3, 3]
    const RollingShutterType rollingShutterType,
    const UTParams &utParams,
    const CameraModel cameraModel,
    const torch::Tensor &distortionCoeffs, // [C,12] for OPENCV, [C,0] for NONE
    const int64_t imageWidth,
    const int64_t imageHeight,
    const float eps2d,
    const float nearPlane,
    const float farPlane,
    const float minRadius2d,
    const bool calcCompensations) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "GaussianProjectionForwardUT not implemented on the CPU");
}

/// @brief PrivateUse1 specialization (not implemented).
template <>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianProjectionForwardUT<torch::kPrivateUse1>(
    const torch::Tensor &means,                   // [N, 3]
    const torch::Tensor &quats,                   // [N, 4]
    const torch::Tensor &logScales,               // [N, 3]
    const torch::Tensor &worldToCamMatricesStart, // [C, 4, 4]
    const torch::Tensor &worldToCamMatricesEnd,   // [C, 4, 4]
    const torch::Tensor &projectionMatrices,      // [C, 3, 3]
    const RollingShutterType rollingShutterType,
    const UTParams &utParams,
    const CameraModel cameraModel,
    const torch::Tensor &distortionCoeffs, // [C,12] for OPENCV, [C,0] for NONE
    const int64_t imageWidth,
    const int64_t imageHeight,
    const float eps2d,
    const float nearPlane,
    const float farPlane,
    const float minRadius2d,
    const bool calcCompensations) {
    TORCH_CHECK_NOT_IMPLEMENTED(false,
                                "GaussianProjectionForwardUT not implemented for this device type");
}

} // namespace fvdb::detail::ops
