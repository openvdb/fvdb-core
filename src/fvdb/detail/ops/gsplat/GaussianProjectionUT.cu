// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#include <fvdb/detail/ops/gsplat/GaussianProjectionUT.h>
#include <fvdb/detail/ops/gsplat/GaussianUtils.cuh>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/cuda/GridDim.h>
#include <fvdb/detail/utils/Nvtx.h>

#include <c10/core/DeviceType.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>

#include <cmath>
#include <tuple>

namespace fvdb::detail::ops {

template <typename T> class Quat;

namespace {

// Helper structure for rolling shutter pose
template <typename T> struct ShutterPose {
    Quat<T> q; // rotation quaternion
    nanovdb::math::Vec3<T> t; // translation
};

// Interpolate between two camera poses for rolling shutter
template <typename T>
__device__ ShutterPose<T>
interpolateShutterPose(T t, const nanovdb::math::Mat3<T> &R_start,
                       const nanovdb::math::Vec3<T> &t_start,
                       const nanovdb::math::Mat3<T> &R_end, const nanovdb::math::Vec3<T> &t_end) {
    // Interpolate translation
    nanovdb::math::Vec3<T> t_interp = t_start + t * (t_end - t_start);

    // Interpolate rotation using SLERP
    Quat<T> q_start = Quat<T>::fromRotationMatrix(R_start);
    Quat<T> q_end   = Quat<T>::fromRotationMatrix(R_end);

    // Simple linear interpolation for quaternions (can be improved with SLERP)
    T dot = q_start.w * q_end.w + q_start.x * q_end.x + q_start.y * q_end.y + q_start.z * q_end.z;
    if (dot < 0) {
        q_end.w = -q_end.w;
        q_end.x = -q_end.x;
        q_end.y = -q_end.y;
        q_end.z = -q_end.z;
        dot     = -dot;
    }

    T s = 1.0 - t;
    Quat<T> q_result(s * q_start.w + t * q_end.w, s * q_start.x + t * q_end.x,
                     s * q_start.y + t * q_end.y, s * q_start.z + t * q_end.z);

    // Normalize
    T norm = sqrt(q_result.w * q_result.w + q_result.x * q_result.x + q_result.y * q_result.y +
                  q_result.z * q_result.z);
    if (norm > 1e-6) {
        q_result.w /= norm;
        q_result.x /= norm;
        q_result.y /= norm;
        q_result.z /= norm;
    }

    return {q_result, t_interp};
}

// Apply camera distortion to a normalized point
template <typename T>
__device__ nanovdb::math::Vec2<T>
applyDistortion(const nanovdb::math::Vec2<T> &p_normalized, const T *radial_coeffs, int num_radial,
                const T *tangential_coeffs, int num_tangential, const T *thin_prism_coeffs,
                int num_thin_prism) {
    T x = p_normalized[0];
    T y = p_normalized[1];
    T x2 = x * x;
    T y2 = y * y;
    T xy = x * y;
    T r2 = x2 + y2;
    T r4 = r2 * r2;
    T r6 = r4 * r2;

    // Default: no radial distortion
    T radial_dist = T(1);
    // Support both:
    // - 4 coeffs: polynomial k1,k2  (1 + k1 r^2 + k2 r^4)
    // - 6 coeffs: OpenCV rational k1..k6  (num/den)
    if (radial_coeffs != nullptr) {
        if (num_radial == 4) {
            radial_dist = T(1) + radial_coeffs[0] * r2 + radial_coeffs[1] * r4;
        } else if (num_radial == 6) {
            const T k1 = radial_coeffs[0], k2 = radial_coeffs[1], k3 = radial_coeffs[2];
            const T k4 = radial_coeffs[3], k5 = radial_coeffs[4], k6 = radial_coeffs[5];
            const T num = T(1) + r2 * (k1 + r2 * (k2 + r2 * k3));
            const T den = T(1) + r2 * (k4 + r2 * (k5 + r2 * k6));
            // Avoid division by 0 / negative flips. Mark invalid later if needed.
            radial_dist = den != T(0) ? (num / den) : T(0);
        } else if (num_radial >= 2) {
            // Best-effort fallback: interpret as polynomial k1,k2,k3...
            radial_dist = T(1) + radial_coeffs[0] * r2;
            if (num_radial > 1) radial_dist += radial_coeffs[1] * r4;
            if (num_radial > 2) radial_dist += radial_coeffs[2] * r6;
        }
    }

    T x_dist = x * radial_dist;
    T y_dist = y * radial_dist;

    if (tangential_coeffs != nullptr && num_tangential >= 2) {
        x_dist += 2 * tangential_coeffs[0] * xy + tangential_coeffs[1] * (r2 + 2 * x2);
        y_dist += tangential_coeffs[0] * (r2 + 2 * y2) + 2 * tangential_coeffs[1] * xy;
    }

    // Thin-prism distortion: OpenCV uses 4 coeffs (s1,s2,s3,s4) as r2 and r4 terms in x/y.
    if (thin_prism_coeffs != nullptr && num_thin_prism >= 2) {
        const T s1 = thin_prism_coeffs[0];
        const T s2 = thin_prism_coeffs[1];
        const T s3 = (num_thin_prism > 2) ? thin_prism_coeffs[2] : T(0);
        const T s4 = (num_thin_prism > 3) ? thin_prism_coeffs[3] : T(0);
        x_dist += s1 * r2 + s2 * r4;
        y_dist += s3 * r2 + s4 * r4;
    }

    return nanovdb::math::Vec2<T>(x_dist, y_dist);
}

// Project a 3D point to 2D with distortion
template <typename T>
__device__ nanovdb::math::Vec2<T>
projectPointWithDistortion(const nanovdb::math::Vec3<T> &p_cam, const nanovdb::math::Mat3<T> &K,
                           const T *radial_coeffs, int num_radial, const T *tangential_coeffs,
                           int num_tangential, const T *thin_prism_coeffs, int num_thin_prism) {
    // Normalize by depth
    T z_inv = T(1) / max(p_cam[2], T(1e-6));
    nanovdb::math::Vec2<T> p_normalized(p_cam[0] * z_inv, p_cam[1] * z_inv);

    // Apply distortion
    nanovdb::math::Vec2<T> p_distorted =
        applyDistortion(p_normalized, radial_coeffs, num_radial, tangential_coeffs,
                       num_tangential, thin_prism_coeffs, num_thin_prism);

    // Project to pixel coordinates
    T fx = K[0][0];
    T fy = K[1][1];
    T cx = K[0][2];
    T cy = K[1][2];

    return nanovdb::math::Vec2<T>(fx * p_distorted[0] + cx, fy * p_distorted[1] + cy);
}

// Generate 3D sigma points and weights for the (scaled) Unscented Transform.
// Mirrors the reference (2*D+1) formulation used in 3DGUT/LichtFeld-Studio.
// Sigma points are generated in WORLD space directly from (mean, scale, quaternion),
// exploiting the closed form SVD of the Gaussian covariance.
template <typename T>
__device__ void
generateWorldSigmaPoints(const nanovdb::math::Vec3<T> &mean_world,
                         const nanovdb::math::Vec4<T> &quat_wxyz,
                         const nanovdb::math::Vec3<T> &scale_world,
                         const UTParams &params,
                         nanovdb::math::Vec3<T> *sigma_points,     // [7]
                         T *weights_mean,                          // [7]
                         T *weights_cov) {                         // [7]
    constexpr int D = 3;
    // This kernel currently supports only the canonical 3D UT with 2D+1 points.
    // (We keep the arrays fixed-size for performance and simplicity.)
    const T alpha   = T(params.alpha);
    const T beta    = T(params.beta);
    const T kappa   = T(params.kappa);
    const T lambda  = alpha * alpha * (T(D) + kappa) - T(D);
    const T denom   = T(D) + lambda;

    // Rotation matrix from quaternion. NOTE: `quaternionToRotationMatrix` expects [w,x,y,z].
    const nanovdb::math::Mat3<T> R = quaternionToRotationMatrix<T>(quat_wxyz);

    sigma_points[0]  = mean_world;
    weights_mean[0]  = lambda / denom;
    weights_cov[0]   = lambda / denom + (T(1) - alpha * alpha + beta);

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
        sigma_points[i + 1]     = mean_world + delta;
        sigma_points[i + 1 + D] = mean_world - delta;
    }
}

// Reconstruct 2D covariance from projected sigma points and a precomputed mean.
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

template <typename T> class Quat {
  public:
    T w, x, y, z;

    __host__ __device__
    Quat()
        : w(1), x(0), y(0), z(0) {}

    __host__ __device__
    Quat(T w_, T x_, T y_, T z_)
        : w(w_), x(x_), y(y_), z(z_) {}

    __host__ __device__ nanovdb::math::Mat3<T>
    toRotationMatrix() const {
        nanovdb::math::Mat3<T> R;
        const T xx = x * x;
        const T yy = y * y;
        const T zz = z * z;
        const T xy = x * y;
        const T xz = x * z;
        const T yz = y * z;
        const T wx = w * x;
        const T wy = w * y;
        const T wz = w * z;

        R[0][0] = 1 - 2 * (yy + zz);
        R[0][1] = 2 * (xy - wz);
        R[0][2] = 2 * (xz + wy);

        R[1][0] = 2 * (xy + wz);
        R[1][1] = 1 - 2 * (xx + zz);
        R[1][2] = 2 * (yz - wx);

        R[2][0] = 2 * (xz - wy);
        R[2][1] = 2 * (yz + wx);
        R[2][2] = 1 - 2 * (xx + yy);

        return R;
    }

    static __host__ __device__ Quat<T>
    fromArray(const T *arr) {
        return Quat(arr[0], arr[1], arr[2], arr[3]);
    }

    static __host__ __device__ Quat<T>
    fromRotationMatrix(const nanovdb::math::Mat3<T> &R) {
        T trace = R[0][0] + R[1][1] + R[2][2];
        T x, y, z, w;

        if (trace > 0) {
            T s = sqrt(trace + 1.0) * 2; // S=4*qw
            w   = 0.25 * s;
            x   = (R[2][1] - R[1][2]) / s;
            y   = (R[0][2] - R[2][0]) / s;
            z   = (R[1][0] - R[0][1]) / s;
        } else if ((R[0][0] > R[1][1]) && (R[0][0] > R[2][2])) {
            T s = sqrt(1.0 + R[0][0] - R[1][1] - R[2][2]) * 2; // S=4*qx
            w   = (R[2][1] - R[1][2]) / s;
            x   = 0.25 * s;
            y   = (R[0][1] + R[1][0]) / s;
            z   = (R[0][2] + R[2][0]) / s;
        } else if (R[1][1] > R[2][2]) {
            T s = sqrt(1.0 + R[1][1] - R[0][0] - R[2][2]) * 2; // S=4*qy
            w   = (R[0][2] - R[2][0]) / s;
            x   = (R[0][1] + R[1][0]) / s;
            y   = 0.25 * s;
            z   = (R[1][2] + R[2][1]) / s;
        } else {
            T s = sqrt(1.0 + R[2][2] - R[0][0] - R[1][1]) * 2; // S=4*qz
            w   = (R[1][0] - R[0][1]) / s;
            x   = (R[0][2] + R[2][0]) / s;
            y   = (R[1][2] + R[2][1]) / s;
            z   = 0.25 * s;
        }
        return Quat(w, x, y, z);
    }

    __host__ __device__ nanovdb::math::Vec3<T>
    rotate(const nanovdb::math::Vec3<T> &v) const {
        // Convert vector to quaternion with w=0
        Quat<T> qv(0, v.x(), v.y(), v.z());

        // Perform quaternion multiplication: q * v * q_conjugate
        Quat<T> q_result = this->multiply(qv).multiply(conjugate());

        return nanovdb::math::Vec3<T>(q_result.x, q_result.y, q_result.z);
    }

    __host__ __device__ Quat<T>
    multiply(const Quat<T> &other) const {
        return Quat<T>(w * other.w - x * other.x - y * other.y - z * other.z,
                       w * other.x + x * other.w + y * other.z - z * other.y,
                       w * other.y - x * other.z + y * other.w + z * other.x,
                       w * other.z + x * other.y - y * other.x + z * other.w);
    }

    __host__ __device__ Quat<T>
    conjugate() const {
        return Quat<T>(w, -x, -y, -z);
    }
};

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
    const int64_t numRadialCoeffs;     // Number of radial distortion coeffs. One of (6, 4, 0).
    const int64_t numTangentialCoeffs; // Number of tangential distortion coeffs. One of (2, 0).
    const int64_t numThinPrismCoeffs;  // Number of thin prism distortion coeffs. One of (3, 0).

    // Tensor Inputs
    const fvdb::TorchRAcc64<ScalarType, 2> mMeansAcc;                   // [N, 3]
    const fvdb::TorchRAcc64<ScalarType, 2> mQuatsAcc;                   // [N, 4]
    const fvdb::TorchRAcc64<ScalarType, 2> mLogScalesAcc;               // [N, 3]
    const fvdb::TorchRAcc32<ScalarType, 3> mWorldToCamMatricesStartAcc; // [C, 4, 4]
    const fvdb::TorchRAcc32<ScalarType, 3> mWorldToCamMatricesEndAcc;   // [C, 4, 4]
    const fvdb::TorchRAcc32<ScalarType, 3> mProjectionMatricesAcc;      // [C, 3, 3]
    const fvdb::TorchRAcc64<ScalarType, 2> mRadialCoeffsAcc;     // [C, 6] or [C, 4] or [C, 0]
    const fvdb::TorchRAcc64<ScalarType, 2> mTangentialCoeffsAcc; // [C, 2] or [C, 0]
    const fvdb::TorchRAcc64<ScalarType, 2> mThinPrismCoeffsAcc;  // [C, 3] or [C, 0]

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
    ScalarType *__restrict__ radialCoeffsShared         = nullptr;
    ScalarType *__restrict__ tangentialCoeffsShared     = nullptr;
    ScalarType *__restrict__ thinPrismCoeffsShared      = nullptr;

    ProjectionForwardUT(
        const int64_t imageWidth,
        const int64_t imageHeight,
        const ScalarType eps2d,
        const ScalarType nearPlane,
        const ScalarType farPlane,
        const ScalarType minRadius2d,
        const RollingShutterType rollingShutterType,
        const UTParams &utParams,
        const bool calcCompensations,
        const torch::Tensor &means,                   // [N, 3]
        const torch::Tensor &quats,                   // [N, 4]
        const torch::Tensor &logScales,               // [N, 3]
        const torch::Tensor &worldToCamMatricesStart, // [C, 4, 4]
        const torch::Tensor &worldToCamMatricesEnd,   // [C, 4, 4]
        const torch::Tensor &projectionMatrices,      // [C, 3, 3]
        const torch::Tensor &radialCoeffs,     // [C, 6] or [C, 4] or [C, 0] distortion coefficients
        const torch::Tensor &tangentialCoeffs, // [C, 2]
        const torch::Tensor &thinPrismCoeffs,  // [C, 3]
        torch::Tensor &outRadii,               // [C, N]
        torch::Tensor &outMeans2d,             // [C, N, 2]
        torch::Tensor &outDepths,              // [C, N]
        torch::Tensor &outConics,              // [C, N, 3]
        torch::Tensor &outCompensations        // [C, N] optional
        )
        : C(projectionMatrices.size(0)), N(means.size(0)),
          mImageWidth(static_cast<int32_t>(imageWidth)),
          mImageHeight(static_cast<int32_t>(imageHeight)), mEps2d(eps2d), mNearPlane(nearPlane),
          mFarPlane(farPlane), mRadiusClip(minRadius2d), mRollingShutterType(rollingShutterType),
          mUTParams(utParams),
          numRadialCoeffs(radialCoeffs.size(1)),
          numTangentialCoeffs(tangentialCoeffs.size(1)),
          numThinPrismCoeffs(thinPrismCoeffs.size(1)),
          mMeansAcc(means.packed_accessor64<ScalarType, 2, torch::RestrictPtrTraits>()),
          mQuatsAcc(quats.packed_accessor64<ScalarType, 2, torch::RestrictPtrTraits>()),
          mLogScalesAcc(logScales.packed_accessor64<ScalarType, 2, torch::RestrictPtrTraits>()),
          mWorldToCamMatricesStartAcc(worldToCamMatricesStart.packed_accessor32<ScalarType, 3, torch::RestrictPtrTraits>()),
          mWorldToCamMatricesEndAcc(worldToCamMatricesEnd.packed_accessor32<ScalarType, 3, torch::RestrictPtrTraits>()),
          mProjectionMatricesAcc(
              projectionMatrices.packed_accessor32<ScalarType, 3, torch::RestrictPtrTraits>()),
          mRadialCoeffsAcc(
              radialCoeffs.packed_accessor64<ScalarType, 2, torch::RestrictPtrTraits>()),
          mTangentialCoeffsAcc(
              tangentialCoeffs.packed_accessor64<ScalarType, 2, torch::RestrictPtrTraits>()),
          mThinPrismCoeffsAcc(
              thinPrismCoeffs.packed_accessor64<ScalarType, 2, torch::RestrictPtrTraits>()),
          mOutRadiiAcc(outRadii.packed_accessor64<int32_t, 2, torch::RestrictPtrTraits>()),
          mOutMeans2dAcc(outMeans2d.packed_accessor64<ScalarType, 3, torch::RestrictPtrTraits>()),
          mOutDepthsAcc(outDepths.packed_accessor64<ScalarType, 2, torch::RestrictPtrTraits>()),
          mOutConicsAcc(outConics.packed_accessor64<ScalarType, 3, torch::RestrictPtrTraits>()),
          mOutCompensationsAcc(outCompensations.defined() ? outCompensations.data_ptr<ScalarType>()
                                                          : nullptr) {}

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

        radialCoeffsShared =
            numRadialCoeffs > 0 ? reinterpret_cast<ScalarType *>(pointer) : nullptr;
        pointer += C * numRadialCoeffs * sizeof(ScalarType);

        tangentialCoeffsShared =
            numTangentialCoeffs > 0 ? reinterpret_cast<ScalarType *>(pointer) : nullptr;
        pointer += C * numTangentialCoeffs * sizeof(ScalarType);

        thinPrismCoeffsShared =
            numThinPrismCoeffs > 0 ? reinterpret_cast<ScalarType *>(pointer) : nullptr;
        pointer += C * numThinPrismCoeffs * sizeof(ScalarType);

        // Layout in element units:
        const int64_t projectionOffset      = 0;
        const int64_t rotStartOffset        = projectionOffset + C * 9;
        const int64_t rotEndOffset          = rotStartOffset + C * 9;
        const int64_t transStartOffset      = rotEndOffset + C * 9;
        const int64_t transEndOffset        = transStartOffset + C * 3;
        const int64_t radialOffset          = transEndOffset + C * 3;
        const int64_t tangentialOffset      = radialOffset + C * numRadialCoeffs;
        const int64_t thinPrismOffset       = tangentialOffset + C * numTangentialCoeffs;
        const int64_t totalElements         = thinPrismOffset + C * numThinPrismCoeffs;

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
            } else if (i < radialOffset) {
                const auto camId   = (i - transEndOffset) / 3;
                const auto entryId = (i - transEndOffset) % 3;
                worldToCamTranslationEndShared[camId][entryId] =
                    mWorldToCamMatricesEndAcc[camId][entryId][3];
            } else if (i < tangentialOffset && numRadialCoeffs > 0) {
                const auto baseIdx = i - radialOffset;
                const auto camId   = baseIdx / numRadialCoeffs;
                const auto entryId = baseIdx % numRadialCoeffs;
                radialCoeffsShared[camId * numRadialCoeffs + entryId] =
                    mRadialCoeffsAcc[camId][entryId];
            } else if (i < thinPrismOffset && numTangentialCoeffs > 0) {
                const auto baseIdx = i - tangentialOffset;
                const auto camId   = baseIdx / numTangentialCoeffs;
                const auto entryId = baseIdx % numTangentialCoeffs;
                tangentialCoeffsShared[camId * numTangentialCoeffs + entryId] =
                    mTangentialCoeffsAcc[camId][entryId];
            } else if (numThinPrismCoeffs > 0) {
                const auto baseIdx = i - thinPrismOffset;
                const auto camId   = baseIdx / numThinPrismCoeffs;
                const auto entryId = baseIdx % numThinPrismCoeffs;
                thinPrismCoeffsShared[camId * numThinPrismCoeffs + entryId] =
                    mThinPrismCoeffsAcc[camId][entryId];
            }
        }
    }

    inline __device__ void
    projectionForward(int idx) {
        if (idx >= C * N) {
            return;
        }

        const int64_t camId      = idx / N;
        const int64_t gaussianId = idx % N;

        // Get camera parameters
        const Mat3 &projectionMatrix = projectionMatsShared[camId];
        const Mat3 &worldToCamRotStart = worldToCamRotMatsStartShared[camId];
        const Mat3 &worldToCamRotEnd   = worldToCamRotMatsEndShared[camId];
        const Vec3 &worldToCamTransStart = worldToCamTranslationStartShared[camId];
        const Vec3 &worldToCamTransEnd   = worldToCamTranslationEndShared[camId];

        // Get distortion coefficients
        const ScalarType *radial_coeffs =
            numRadialCoeffs > 0 ? &radialCoeffsShared[camId * numRadialCoeffs] : nullptr;
        const ScalarType *tangential_coeffs =
            numTangentialCoeffs > 0 ? &tangentialCoeffsShared[camId * numTangentialCoeffs] : nullptr;
        const ScalarType *thin_prism_coeffs =
            numThinPrismCoeffs > 0 ? &thinPrismCoeffsShared[camId * numThinPrismCoeffs] : nullptr;

        // Get Gaussian parameters
        const Vec3 meanWorldSpace(mMeansAcc[gaussianId][0], mMeansAcc[gaussianId][1],
                                  mMeansAcc[gaussianId][2]);
        const auto quatAcc     = mQuatsAcc[gaussianId];
        const auto logScaleAcc = mLogScalesAcc[gaussianId];
        const Vec4 quat_wxyz(quatAcc[0], quatAcc[1], quatAcc[2], quatAcc[3]);
        const Vec3 scale_world(::cuda::std::exp(logScaleAcc[0]), ::cuda::std::exp(logScaleAcc[1]),
                               ::cuda::std::exp(logScaleAcc[2]));

        // Depth culling uses center shutter pose (matches reference kernel).
        {
            const ShutterPose<ScalarType> shutter_pose_center =
                interpolateShutterPose(ScalarType(0.5), worldToCamRotStart, worldToCamTransStart,
                                       worldToCamRotEnd, worldToCamTransEnd);
            const Mat3 R_center = shutter_pose_center.q.toRotationMatrix();
            const Vec3 t_center = shutter_pose_center.t;
            const Vec3 meanCamCenter =
                transformPointWorldToCam(R_center, t_center, meanWorldSpace);
            if (meanCamCenter[2] < mNearPlane || meanCamCenter[2] > mFarPlane) {
                mOutRadiiAcc[camId][gaussianId] = 0;
                return;
            }
        }

        // Generate world-space sigma points (7) and UT weights (mean/cov).
        nanovdb::math::Vec3<ScalarType> sigma_points_world[7];
        ScalarType weights_mean[7];
        ScalarType weights_cov[7];
        // Enforce supported sigma-point count (2*3+1=7).
        if (mUTParams.numSigmaPoints != 7) {
            mOutRadiiAcc[camId][gaussianId] = 0;
            return;
        }
        generateWorldSigmaPoints(meanWorldSpace, quat_wxyz, scale_world, mUTParams, sigma_points_world,
                                 weights_mean, weights_cov);
        constexpr int num_sigma_points = 7;

        // Project sigma points through camera model
        nanovdb::math::Vec2<ScalarType> projected_points[7];
        bool valid_any = false;
        const ScalarType margin_x = ScalarType(mImageWidth) * ScalarType(mUTParams.inImageMargin);
        const ScalarType margin_y = ScalarType(mImageHeight) * ScalarType(mUTParams.inImageMargin);

        auto project_world_point = [&] __device__ (const Vec3 &p_world, Vec2 &out_pixel) -> bool {
            // Rolling shutter projection similar to reference: iterate shutter pose based on the
            // current estimate of pixel coordinate.
            auto project_with_pose = [&] __device__ (const ShutterPose<ScalarType> &pose,
                                                     Vec2 &out_pix) -> bool {
                const Vec3 p_cam = transformPointWorldToCam(pose.q.toRotationMatrix(), pose.t, p_world);
                // Perspective only (ortho is not meaningful for distorted camera models).
                if (p_cam[2] <= ScalarType(0)) {
                    return false;
                }
                out_pix = projectPointWithDistortion(p_cam, projectionMatrix, radial_coeffs,
                                                     (int)numRadialCoeffs, tangential_coeffs,
                                                     (int)numTangentialCoeffs, thin_prism_coeffs,
                                                     (int)numThinPrismCoeffs);
                const bool in_img = (out_pix[0] >= -margin_x) && (out_pix[0] < ScalarType(mImageWidth) + margin_x) &&
                                    (out_pix[1] >= -margin_y) && (out_pix[1] < ScalarType(mImageHeight) + margin_y);
                return in_img;
            };

            // Start/end projections for initialization.
            ShutterPose<ScalarType> pose_start =
                interpolateShutterPose(ScalarType(0.0), worldToCamRotStart, worldToCamTransStart,
                                       worldToCamRotEnd, worldToCamTransEnd);
            ShutterPose<ScalarType> pose_end =
                interpolateShutterPose(ScalarType(1.0), worldToCamRotStart, worldToCamTransStart,
                                       worldToCamRotEnd, worldToCamTransEnd);
            Vec2 pix_start, pix_end;
            const bool valid_start = project_with_pose(pose_start, pix_start);
            const bool valid_end   = project_with_pose(pose_end, pix_end);

            if (mRollingShutterType == RollingShutterType::NONE) {
                if (!valid_start) {
                    out_pixel = pix_start;
                    return false;
                }
                out_pixel = pix_start;
                return true;
            }

            // If neither endpoint is valid, treat as invalid.
            if (!valid_start && !valid_end) {
                out_pixel = pix_end;
                return false;
            }

            Vec2 pix_prev = valid_start ? pix_start : pix_end;
            // Iteration count: small fixed number (reference uses 10).
            constexpr int kIters = 6;
            for (int it = 0; it < kIters; ++it) {
                ScalarType t_rs = ScalarType(0);
                if (mRollingShutterType == RollingShutterType::VERTICAL) {
                    t_rs = floor(pix_prev[1]) / max(ScalarType(1), ScalarType(mImageHeight - 1));
                } else if (mRollingShutterType == RollingShutterType::HORIZONTAL) {
                    t_rs = floor(pix_prev[0]) / max(ScalarType(1), ScalarType(mImageWidth - 1));
                }
                t_rs = min(ScalarType(1), max(ScalarType(0), t_rs));
                ShutterPose<ScalarType> pose_rs =
                    interpolateShutterPose(t_rs, worldToCamRotStart, worldToCamTransStart,
                                           worldToCamRotEnd, worldToCamTransEnd);
                Vec2 pix_rs;
                const bool valid_rs = project_with_pose(pose_rs, pix_rs);
                pix_prev            = pix_rs;
                if (!valid_rs) {
                    out_pixel = pix_rs;
                    return false;
                }
            }

            out_pixel = pix_prev;
            return true;
        };

        for (int i = 0; i < num_sigma_points; ++i) {
            Vec2 pix;
            const bool valid_i = project_world_point(sigma_points_world[i], pix);
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
        for (int i = 0; i < num_sigma_points; ++i) {
            mean2d[0] += weights_mean[i] * projected_points[i][0];
            mean2d[1] += weights_mean[i] * projected_points[i][1];
        }

        // Reconstruct 2D covariance from projected sigma points
        Mat2 covar2d = reconstructCovarianceFromSigmaPoints(
            projected_points, weights_cov, num_sigma_points, mean2d);

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
        // For depth we use the Gaussian mean under the center shutter pose (same as the cull check).
        {
            const ShutterPose<ScalarType> shutter_pose_center =
                interpolateShutterPose(ScalarType(0.5), worldToCamRotStart, worldToCamTransStart,
                                       worldToCamRotEnd, worldToCamTransEnd);
            const Mat3 R_center = shutter_pose_center.q.toRotationMatrix();
            const Vec3 t_center = shutter_pose_center.t;
            const Vec3 meanCamCenter =
                transformPointWorldToCam(R_center, t_center, meanWorldSpace);
            mOutDepthsAcc[camId][gaussianId] = meanCamCenter[2];
        }
        mOutConicsAcc[camId][gaussianId][0]   = covar2dInverse[0][0];
        mOutConicsAcc[camId][gaussianId][1]   = covar2dInverse[0][1];
        mOutConicsAcc[camId][gaussianId][2]   = covar2dInverse[1][1];
        if (mOutCompensationsAcc != nullptr) {
            mOutCompensationsAcc[idx] = compensation;
        }
    }
};

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

template <>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianProjectionForwardUT<torch::kCUDA>(
    const torch::Tensor &means,                   // [N, 3]
    const torch::Tensor &quats,                   // [N, 4]
    const torch::Tensor &scales,                  // [N, 3]
    const torch::Tensor &worldToCamMatricesStart, // [C, 4, 4]
    const torch::Tensor &worldToCamMatricesEnd,   // [C, 4, 4]
    const torch::Tensor &projectionMatrices,      // [C, 3, 3]
    const RollingShutterType rollingShutterType,
    const UTParams &utParams,
    const torch::Tensor &radialCoeffs,     // [C, 6] or [C, 4] or [C, 0] distortion coefficients
    const torch::Tensor &tangentialCoeffs, // [C, 2] or [C, 0] distortion coefficients
    const torch::Tensor &thinPrismCoeffs,  // [C, 3] or [C, 0] distortion coefficients
    const int64_t imageWidth,
    const int64_t imageHeight,
    const float eps2d,
    const float nearPlane,
    const float farPlane,
    const float minRadius2d,
    const bool calcCompensations,
    const bool ortho) {
    FVDB_FUNC_RANGE();

    TORCH_CHECK_VALUE(means.is_cuda(), "means must be a CUDA tensor");
    TORCH_CHECK_VALUE(quats.is_cuda(), "quats must be a CUDA tensor");
    TORCH_CHECK_VALUE(scales.is_cuda(), "scales must be a CUDA tensor");
    TORCH_CHECK_VALUE(worldToCamMatricesStart.is_cuda(),
                      "worldToCamMatricesStart must be a CUDA tensor");
    TORCH_CHECK_VALUE(worldToCamMatricesEnd.is_cuda(),
                      "worldToCamMatricesEnd must be a CUDA tensor");
    TORCH_CHECK_VALUE(projectionMatrices.is_cuda(), "projectionMatrices must be a CUDA tensor");
    TORCH_CHECK_VALUE(radialCoeffs.is_cuda(), "radialCoeffs must be a CUDA tensor");
    TORCH_CHECK_VALUE(tangentialCoeffs.is_cuda(), "tangentialCoeffs must be a CUDA tensor");
    TORCH_CHECK_VALUE(thinPrismCoeffs.is_cuda(), "thinPrismCoeffs must be a CUDA tensor");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(means));

    const auto N                = means.size(0);              // number of gaussians
    const auto C                = projectionMatrices.size(0); // number of cameras
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(means.device().index());

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
    // This kernel currently implements the (distorted) perspective camera model.
    // Keep parity with the reference kernel: orthographic is not supported here.
    TORCH_CHECK_VALUE(!ortho, "GaussianProjectionForwardUT does not support orthographic projection");

    const size_t SHARED_MEM_SIZE =
        C * (3 * sizeof(nanovdb::math::Mat3<scalar_t>) + 2 * sizeof(nanovdb::math::Vec3<scalar_t>)) +
        C * (radialCoeffs.size(1) + tangentialCoeffs.size(1) + thinPrismCoeffs.size(1)) *
            sizeof(scalar_t);

    ProjectionForwardUT<scalar_t> projectionForward(
        imageWidth, imageHeight, eps2d, nearPlane, farPlane, minRadius2d, rollingShutterType,
        utParams, calcCompensations, means, quats, torch::log(scales), worldToCamMatricesStart,
        worldToCamMatricesEnd, projectionMatrices, radialCoeffs, tangentialCoeffs, thinPrismCoeffs,
        outRadii, outMeans2d, outDepths, outConics, outCompensations);

    projectionForwardUTKernel<scalar_t><<<NUM_BLOCKS, 256, SHARED_MEM_SIZE, stream>>>(
        0, C * N, projectionForward);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(outRadii, outMeans2d, outDepths, outConics, outCompensations);
}

template <>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianProjectionForwardUT<torch::kCPU>(
    const torch::Tensor &means,                   // [N, 3]
    const torch::Tensor &quats,                   // [N, 4]
    const torch::Tensor &scales,                  // [N, 3]
    const torch::Tensor &worldToCamMatricesStart, // [C, 4, 4]
    const torch::Tensor &worldToCamMatricesEnd,   // [C, 4, 4]
    const torch::Tensor &projectionMatrices,      // [C, 3, 3]
    const RollingShutterType rollingShutterType,
    const UTParams &utParams,
    const torch::Tensor &radialCoeffs,     // [C, 6] or [C, 4] or [C, 0] distortion coefficients
    const torch::Tensor &tangentialCoeffs, // [C, 2] or [C, 0] distortion coefficients
    const torch::Tensor &thinPrismCoeffs,  // [C, 3] or [C, 0] distortion coefficients
    const int64_t imageWidth,
    const int64_t imageHeight,
    const float eps2d,
    const float nearPlane,
    const float farPlane,
    const float minRadius2d,
    const bool calcCompensations,
    const bool ortho) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "GaussianProjectionForwardUT not implemented on the CPU");
}

template <>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianProjectionForwardUT<torch::kPrivateUse1>(
    const torch::Tensor &means,                   // [N, 3]
    const torch::Tensor &quats,                   // [N, 4]
    const torch::Tensor &scales,                  // [N, 3]
    const torch::Tensor &worldToCamMatricesStart, // [C, 4, 4]
    const torch::Tensor &worldToCamMatricesEnd,   // [C, 4, 4]
    const torch::Tensor &projectionMatrices,      // [C, 3, 3]
    const RollingShutterType rollingShutterType,
    const UTParams &utParams,
    const torch::Tensor &radialCoeffs,     // [C, 6] or [C, 4] or [C, 0] distortion coefficients
    const torch::Tensor &tangentialCoeffs, // [C, 2] or [C, 0] distortion coefficients
    const torch::Tensor &thinPrismCoeffs,  // [C, 3] or [C, 0] distortion coefficients
    const int64_t imageWidth,
    const int64_t imageHeight,
    const float eps2d,
    const float nearPlane,
    const float farPlane,
    const float minRadius2d,
    const bool calcCompensations,
    const bool ortho) {
    TORCH_CHECK_NOT_IMPLEMENTED(false,
                                "GaussianProjectionForwardUT not implemented for this device type");
}

} // namespace fvdb::detail::ops
