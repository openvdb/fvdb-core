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
/// This is an internal helper used by the UT projection kernel. It owns the camera intrinsics
/// `K` and the distortion coefficient pointers and can project `p_cam -> pixel`.
/// @see https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
///
/// @tparam T Scalar type
template <typename T> struct OpenCVCameraModel {
    using Vec2 = nanovdb::math::Vec2<T>;
    using Vec3 = nanovdb::math::Vec3<T>;
    using Mat3 = nanovdb::math::Mat3<T>;

    // Internal "math mode" for OpenCV distortion evaluation.
    // Kept inside the struct to avoid confusion with the public API `DistortionModel`.
    enum class Model : uint8_t {
        NONE          = 0,  // no distortion
        RADTAN_5      = 5,  // k1,k2,p1,p2,k3 (polynomial radial up to r^6)
        RATIONAL_8    = 8,  // k1,k2,p1,p2,k3,k4,k5,k6 (rational radial)
        RADTAN_THIN_9 = 9,  // RADTAN_5 + thin prism s1..s4 (polynomial radial + thin-prism)
        THIN_PRISM_12 = 12, // RATIONAL_8 + s1,s2,s3,s4
    };

    // Camera intrinsics. Must be set by the caller before use.
    Mat3 K = Mat3();

    // Coefficients for the distortion model.
    const T *radial     = nullptr;     // k1..k6 (but k4..k6 only used in rational model)
    int numRadial       = 0;           // Number of radial coefficients
    const T *tangential = nullptr;     // p1,p2
    int numTangential   = 0;           // Number of tangential coefficients
    const T *thinPrism  = nullptr;     // s1..s4 (but s1..s3 only used in thin prism model)
    int numThinPrism    = 0;           // Number of thin prism coefficients
    Model model         = Model::NONE; // Distortion model

    // Initialize this camera model from the public DistortionModel enum and a per-camera
    // coefficient vector in OpenCV's packed layout:
    //   [k1,k2,k3,k4,k5,k6,p1,p2,s1,s2,s3,s4]
    //
    // Returns false if the coefficient layout is not available for the requested model.
    __device__ bool
    init(const DistortionModel distortionModel,
         const Mat3 &K_in,
         const T *coeffs,
         const int nCoeffs) {
        K = K_in;

        if (distortionModel == DistortionModel::NONE) {
            radial = tangential = thinPrism = nullptr;
            numRadial = numTangential = numThinPrism = 0;
            model                                    = Model::NONE;
            return true;
        }

        // OpenCV models require the packed coefficient layout.
        if (coeffs == nullptr || nCoeffs < 12) {
            return false;
        }
        const T *radial_in = coeffs + 0; // k1..k6
        const T *tang_in   = coeffs + 6; // p1,p2
        const T *thin_in   = coeffs + 8; // s1..s4

        if (distortionModel == DistortionModel::OPENCV_RADTAN_5) {
            radial        = radial_in;
            numRadial     = 3; // k1,k2,k3 (polynomial)
            tangential    = tang_in;
            numTangential = 2;
            thinPrism     = nullptr;
            numThinPrism  = 0;
            model         = Model::RADTAN_5;
            return true;
        }
        if (distortionModel == DistortionModel::OPENCV_RATIONAL_8) {
            radial        = radial_in;
            numRadial     = 6; // k1..k6 (rational)
            tangential    = tang_in;
            numTangential = 2;
            thinPrism     = nullptr;
            numThinPrism  = 0;
            model         = Model::RATIONAL_8;
            return true;
        }
        if (distortionModel == DistortionModel::OPENCV_THIN_PRISM_12) {
            radial        = radial_in;
            numRadial     = 6; // k1..k6 (rational)
            tangential    = tang_in;
            numTangential = 2;
            thinPrism     = thin_in;
            numThinPrism  = 4; // s1..s4
            model         = Model::THIN_PRISM_12;
            return true;
        }
        if (distortionModel == DistortionModel::OPENCV_RADTAN_THIN_PRISM_9) {
            // Polynomial radial + thin-prism; ignore k4..k6 by construction.
            radial        = radial_in;
            numRadial     = 3; // k1,k2,k3 (polynomial)
            tangential    = tang_in;
            numTangential = 2;
            thinPrism     = thin_in;
            numThinPrism  = 4; // s1..s4
            model         = Model::RADTAN_THIN_9;
            return true;
        }

        // Unknown distortion model.
        return false;
    }

    /// @brief Get a coefficient or zero if the pointer is null.
    ///
    /// @param[in] ptr Pointer to the coefficients
    /// @param[in] n Number of coefficients
    /// @param[in] i Index of the coefficient
    ///
    /// @return Coefficient or zero if the pointer is null
    __host__ __device__ inline T
    coeffOrZero(const T *ptr, const int n, const int i) const {
        return (ptr != nullptr && i >= 0 && i < n) ? ptr[i] : T(0);
    }

    /// @brief Apply the distortion to a 2D point.
    ///
    /// @param[in] p_normalized Normalized 2D point [x, y] in camera coordinates
    ///
    /// @return Distorted 2D point [x_dist, y_dist] in camera coordinates
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
        // OpenCV: x += 2*p1*x*y + p2*(r^2 + 2*x^2)
        //         y += p1*(r^2 + 2*y^2) + 2*p2*x*y
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

    // Project a 3D point in camera coordinates to pixel coordinates using this camera model
    // (pinhole + distortion + intrinsics).
    __device__ Vec2
    project(const Vec3 &p_cam) const {
        // Normalize by depth.
        const T z_inv = T(1) / max(p_cam[2], T(1e-6));
        const Vec2 p_normalized(p_cam[0] * z_inv, p_cam[1] * z_inv);

        const Vec2 p_distorted = applyDistortion(p_normalized);

        // Project to pixel coordinates.
        const T fx = K[0][0];
        const T fy = K[1][1];
        const T cx = K[0][2];
        const T cy = K[1][2];
        return Vec2(fx * p_distorted[0] + cx, fy * p_distorted[1] + cy);
    }
};

// Generate 3D sigma points and weights for the (scaled) Unscented Transform.
// Sigma points are generated in WORLD space directly from (mean, scale, quaternion),
// exploiting the closed form SVD of the Gaussian covariance.
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
    const DistortionModel mDistortionModel;
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

    ProjectionForwardUT(const int64_t imageWidth,
                        const int64_t imageHeight,
                        const ScalarType eps2d,
                        const ScalarType nearPlane,
                        const ScalarType farPlane,
                        const ScalarType minRadius2d,
                        const RollingShutterType rollingShutterType,
                        const UTParams &utParams,
                        const DistortionModel distortionModel,
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
          mUTParams(utParams), mDistortionModel(distortionModel),
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

        OpenCVCameraModel<ScalarType> camera;
        const ScalarType *coeffs = (distortionCoeffsShared != nullptr)
                                       ? &distortionCoeffsShared[camId * mNumDistortionCoeffs]
                                       : nullptr;
        if (!camera.init(mDistortionModel, projectionMatrix, coeffs, int(mNumDistortionCoeffs))) {
            mOutRadiiAcc[camId][gaussianId] = 0;
            return;
        }

        // Get Gaussian parameters
        const Vec3 meanWorldSpace(
            mMeansAcc[gaussianId][0], mMeansAcc[gaussianId][1], mMeansAcc[gaussianId][2]);
        const auto quatAcc     = mQuatsAcc[gaussianId];
        const auto logScaleAcc = mLogScalesAcc[gaussianId];
        const Vec4 quat_wxyz(quatAcc[0], quatAcc[1], quatAcc[2], quatAcc[3]);
        const Vec3 scale_world(::cuda::std::exp(logScaleAcc[0]),
                               ::cuda::std::exp(logScaleAcc[1]),
                               ::cuda::std::exp(logScaleAcc[2]));

        // Depth culling should use the same shutter pose as projection:
        // - RollingShutterType::NONE: use start pose (t=0.0), matching `project_world_point`.
        // - Rolling shutter modes: use center pose (t=0.5) as a conservative/stable cull.
        {
            const ScalarType t_depth            = (mRollingShutterType == RollingShutterType::NONE)
                                                      ? ScalarType(0.0)
                                                      : ScalarType(0.5);
            const Pose<ScalarType> shutter_pose = interpolatePose(t_depth,
                                                                  worldToCamRotStart,
                                                                  worldToCamTransStart,
                                                                  worldToCamRotEnd,
                                                                  worldToCamTransEnd);
            const Mat3 R_depth                  = quaternionToRotationMatrix(shutter_pose.q);
            const Vec3 t_depth_v                = shutter_pose.t;
            const Vec3 meanCam = transformPointWorldToCam(R_depth, t_depth_v, meanWorldSpace);
            if (meanCam[2] < mNearPlane || meanCam[2] > mFarPlane) {
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
        generateWorldSigmaPoints(meanWorldSpace,
                                 quat_wxyz,
                                 scale_world,
                                 mUTParams,
                                 sigma_points_world,
                                 weights_mean,
                                 weights_cov);
        constexpr int num_sigma_points = 7;

        // Project sigma points through camera model
        nanovdb::math::Vec2<ScalarType> projected_points[7];
        bool valid_any            = false;
        const ScalarType margin_x = ScalarType(mImageWidth) * ScalarType(mUTParams.inImageMargin);
        const ScalarType margin_y = ScalarType(mImageHeight) * ScalarType(mUTParams.inImageMargin);

        enum class ProjStatus : uint8_t { BehindCamera, OutOfBounds, InImage };

        auto project_world_point = [&]
            __device__(const Vec3 &p_world, Vec2 &out_pixel) -> ProjStatus {
            // Rolling shutter projection similar to reference: iterate shutter pose based on the
            // current estimate of pixel coordinate.
            auto isInImage = []
                __device__(const ProjStatus s) -> bool { return s == ProjStatus::InImage; };
            auto project_with_pose = [&]
                __device__(const Pose<ScalarType> &pose, Vec2 &out_pix) -> ProjStatus {
                const Vec3 p_cam =
                    transformPointWorldToCam(quaternionToRotationMatrix(pose.q), pose.t, p_world);
                // Perspective only (ortho is not meaningful for distorted camera models).
                if (p_cam[2] <= ScalarType(0)) {
                    // Ensure deterministic output to avoid UB on callers that assign/read even on
                    // invalid projections. This value is ignored when we treat BehindCamera as a
                    // hard reject (see below).
                    out_pix = Vec2(ScalarType(0), ScalarType(0));
                    return ProjStatus::BehindCamera;
                }
                out_pix           = camera.project(p_cam);
                const bool in_img = (out_pix[0] >= -margin_x) &&
                                    (out_pix[0] < ScalarType(mImageWidth) + margin_x) &&
                                    (out_pix[1] >= -margin_y) &&
                                    (out_pix[1] < ScalarType(mImageHeight) + margin_y);
                return in_img ? ProjStatus::InImage : ProjStatus::OutOfBounds;
            };

            // Start/end projections for initialization.
            Pose<ScalarType> pose_start = interpolatePose(ScalarType(0.0),
                                                          worldToCamRotStart,
                                                          worldToCamTransStart,
                                                          worldToCamRotEnd,
                                                          worldToCamTransEnd);
            Pose<ScalarType> pose_end   = interpolatePose(ScalarType(1.0),
                                                        worldToCamRotStart,
                                                        worldToCamTransStart,
                                                        worldToCamRotEnd,
                                                        worldToCamTransEnd);
            Vec2 pix_start(ScalarType(0), ScalarType(0));
            Vec2 pix_end(ScalarType(0), ScalarType(0));
            const ProjStatus status_start = project_with_pose(pose_start, pix_start);
            const ProjStatus status_end   = project_with_pose(pose_end, pix_end);

            if (mRollingShutterType == RollingShutterType::NONE) {
                out_pixel = pix_start;
                return status_start;
            }

            // If both endpoints are behind the camera, treat as a hard invalid (discontinuous).
            if (status_start == ProjStatus::BehindCamera &&
                status_end == ProjStatus::BehindCamera) {
                out_pixel = pix_end;
                return ProjStatus::BehindCamera;
            }

            // If neither endpoint is in image (but at least one is in front), treat as invalid.
            // (Keep parity with the old behavior which required an in-image endpoint to iterate.)
            if (!isInImage(status_start) && !isInImage(status_end)) {
                out_pixel = (status_end != ProjStatus::BehindCamera) ? pix_end : pix_start;
                return ProjStatus::OutOfBounds;
            }

            Vec2 pix_prev = isInImage(status_start) ? pix_start : pix_end;
            // Iteration count: small fixed number (reference uses 10).
            constexpr int kIters = 6;
            for (int it = 0; it < kIters; ++it) {
                ScalarType t_rs = ScalarType(0);
                if (mRollingShutterType == RollingShutterType::VERTICAL) {
                    t_rs = floor(pix_prev[1]) / max(ScalarType(1), ScalarType(mImageHeight - 1));
                } else if (mRollingShutterType == RollingShutterType::HORIZONTAL) {
                    t_rs = floor(pix_prev[0]) / max(ScalarType(1), ScalarType(mImageWidth - 1));
                }
                t_rs                     = min(ScalarType(1), max(ScalarType(0), t_rs));
                Pose<ScalarType> pose_rs = interpolatePose(t_rs,
                                                           worldToCamRotStart,
                                                           worldToCamTransStart,
                                                           worldToCamRotEnd,
                                                           worldToCamTransEnd);
                Vec2 pix_rs(ScalarType(0), ScalarType(0));
                const ProjStatus status_rs = project_with_pose(pose_rs, pix_rs);
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
        };

        for (int i = 0; i < num_sigma_points; ++i) {
            Vec2 pix;
            const ProjStatus status = project_world_point(sigma_points_world[i], pix);
            // Hard reject if any sigma point is behind the camera (discontinuous projection).
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
        // For depth we use the same shutter pose as the cull check above.
        {
            const ScalarType t_depth            = (mRollingShutterType == RollingShutterType::NONE)
                                                      ? ScalarType(0.0)
                                                      : ScalarType(0.5);
            const Pose<ScalarType> shutter_pose = interpolatePose(t_depth,
                                                                  worldToCamRotStart,
                                                                  worldToCamTransStart,
                                                                  worldToCamRotEnd,
                                                                  worldToCamTransEnd);
            const Mat3 R_depth                  = quaternionToRotationMatrix(shutter_pose.q);
            const Vec3 t_depth_v                = shutter_pose.t;
            const Vec3 meanCam = transformPointWorldToCam(R_depth, t_depth_v, meanWorldSpace);
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
    const torch::Tensor &logScales,               // [N, 3]
    const torch::Tensor &worldToCamMatricesStart, // [C, 4, 4]
    const torch::Tensor &worldToCamMatricesEnd,   // [C, 4, 4]
    const torch::Tensor &projectionMatrices,      // [C, 3, 3]
    const RollingShutterType rollingShutterType,
    const UTParams &utParams,
    const DistortionModel distortionModel,
    const torch::Tensor &distortionCoeffs, // [C,12] for OPENCV, [C,0] for NONE
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
    TORCH_CHECK_VALUE(logScales.is_cuda(), "logScales must be a CUDA tensor");
    TORCH_CHECK_VALUE(worldToCamMatricesStart.is_cuda(),
                      "worldToCamMatricesStart must be a CUDA tensor");
    TORCH_CHECK_VALUE(worldToCamMatricesEnd.is_cuda(),
                      "worldToCamMatricesEnd must be a CUDA tensor");
    TORCH_CHECK_VALUE(projectionMatrices.is_cuda(), "projectionMatrices must be a CUDA tensor");
    TORCH_CHECK_VALUE(distortionCoeffs.is_cuda(), "distortionCoeffs must be a CUDA tensor");
    TORCH_CHECK_VALUE(distortionCoeffs.dim() == 2, "distortionCoeffs must be 2D");
    if (distortionModel == DistortionModel::NONE) {
        // Accept any K (including 0); ignored.
    } else if (distortionModel == DistortionModel::OPENCV_RADTAN_5 ||
               distortionModel == DistortionModel::OPENCV_RATIONAL_8 ||
               distortionModel == DistortionModel::OPENCV_RADTAN_THIN_PRISM_9 ||
               distortionModel == DistortionModel::OPENCV_THIN_PRISM_12) {
        TORCH_CHECK_VALUE(distortionCoeffs.size(1) == 12,
                          "For DistortionModel::OPENCV_* , distortionCoeffs must have shape [C,12] "
                          "as [k1,k2,k3,k4,k5,k6,p1,p2,s1,s2,s3,s4]");
    } else {
        TORCH_CHECK_VALUE(false, "Unknown DistortionModel for GaussianProjectionForwardUT");
    }

    // This kernel currently implements only the canonical 3D UT with 2D+1 sigma points (7).
    // Validate on the host so misconfiguration is reported loudly instead of silently discarding.
    TORCH_CHECK_VALUE(
        utParams.numSigmaPoints == 7,
        "GaussianProjectionForwardUT currently supports only utParams.numSigmaPoints == 7 "
        "(3D UT with 2D+1 sigma points). Got ",
        utParams.numSigmaPoints);

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
    // This kernel currently implements the (distorted) perspective camera model.
    // Keep parity with the reference kernel: orthographic is not supported here.
    TORCH_CHECK_VALUE(!ortho,
                      "GaussianProjectionForwardUT does not support orthographic projection");

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
                                                    distortionModel,
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
    const DistortionModel distortionModel,
    const torch::Tensor &distortionCoeffs, // [C,12] for OPENCV, [C,0] for NONE
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
    const torch::Tensor &logScales,               // [N, 3]
    const torch::Tensor &worldToCamMatricesStart, // [C, 4, 4]
    const torch::Tensor &worldToCamMatricesEnd,   // [C, 4, 4]
    const torch::Tensor &projectionMatrices,      // [C, 3, 3]
    const RollingShutterType rollingShutterType,
    const UTParams &utParams,
    const DistortionModel distortionModel,
    const torch::Tensor &distortionCoeffs, // [C,12] for OPENCV, [C,0] for NONE
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
