// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#include <fvdb/detail/ops/gsplat/GaussianCameraAccessorCopy.cuh>
#include <fvdb/detail/ops/gsplat/GaussianCameras.cuh>
#include <fvdb/detail/ops/gsplat/GaussianProjectionUT.h>
#include <fvdb/detail/ops/gsplat/GaussianRigidTransform.cuh>
#include <fvdb/detail/ops/gsplat/GaussianUtils.cuh>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/Nvtx.h>
#include <fvdb/detail/utils/cuda/GridDim.h>

#include <c10/core/DeviceType.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>

#include <cuda/std/cmath>

#include <cmath>
#include <tuple>

namespace fvdb::detail::ops {

namespace {

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
                         nanovdb::math::Vec3<T> (&sigma_points)[7],
                         T (&weights_mean)[7],
                         T (&weights_cov)[7]) {
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
reconstructCovarianceFromSigmaPoints(const nanovdb::math::Vec2<T> (&projected_points)[7],
                                     const T (&weights_cov)[7],
                                     const nanovdb::math::Vec2<T> &mean2d) {
    nanovdb::math::Mat2<T> covar2d(T(0), T(0), T(0), T(0));
    constexpr int kNumSigmaPoints = 7;
    for (int i = 0; i < kNumSigmaPoints; ++i) {
        const nanovdb::math::Vec2<T> diff = projected_points[i] - mean2d;
        covar2d[0][0] += weights_cov[i] * diff[0] * diff[0];
        covar2d[0][1] += weights_cov[i] * diff[0] * diff[1];
        covar2d[1][0] += weights_cov[i] * diff[1] * diff[0];
        covar2d[1][1] += weights_cov[i] * diff[1] * diff[1];
    }
    return covar2d;
}

/// @brief Enforce a positive-semidefinite 2x2 covariance matrix.
///
/// UT covariance reconstruction can produce an indefinite (or even negative definite) matrix due
/// to negative covariance weights combined with the nonlinear projection. This function clamps the
/// eigenvalues to a minimum threshold and reconstructs the matrix, ensuring downstream operations
/// (sqrt, inverse) remain numerically well-defined.
template <typename T>
__device__ __forceinline__ void
enforcePSD2x2(const T minEigen, nanovdb::math::Mat2<T> &covar2d) {
    using Vec2 = nanovdb::math::Vec2<T>;

    // Symmetrize defensively.
    const T a = covar2d[0][0];
    const T c = covar2d[1][1];
    const T b = T(0.5) * (covar2d[0][1] + covar2d[1][0]);

    const T trace = a + c;
    const T det   = a * c - b * b;

    const T half_trace = T(0.5) * trace;
    T disc             = half_trace * half_trace - det;
    disc               = max(T(0), disc);
    const T s          = sqrt(disc);

    // Eigenvalues (v1 >= v2).
    const T v1 = half_trace + s;
    const T v2 = half_trace - s;

    // Clamp eigenvalues to ensure PSD + invertibility.
    const T v1c = max(v1, minEigen);
    const T v2c = max(v2, minEigen);

    // Eigenvector for v1. For a 2x2 symmetric matrix, we can form a stable vector from either:
    //   [b, v1-a] or [v1-c, b]
    Vec2 u(T(1), T(0));
    const T eps = (sizeof(T) == sizeof(float)) ? T(1e-8) : T(1e-12);
    if (::cuda::std::fabs(b) > eps || ::cuda::std::fabs(v1 - a) > eps ||
        ::cuda::std::fabs(v1 - c) > eps) {
        T ux = b;
        T uy = v1 - a;
        // Prefer the formulation with the larger component to avoid cancellation.
        if (::cuda::std::fabs(v1 - c) > ::cuda::std::fabs(v1 - a)) {
            ux = v1 - c;
            uy = b;
        }
        const T n = sqrt(ux * ux + uy * uy);
        if (n > eps) {
            u = Vec2(ux / n, uy / n);
        }
    } else {
        // Diagonal (or near-diagonal) case.
        u = (a >= c) ? Vec2(T(1), T(0)) : Vec2(T(0), T(1));
    }

    // Orthonormal basis.
    const Vec2 v(-u[1], u[0]);

    // Reconstruct: cov = Q * diag(v1c, v2c) * Q^T
    covar2d[0][0] = v1c * u[0] * u[0] + v2c * v[0] * v[0];
    covar2d[0][1] = v1c * u[0] * u[1] + v2c * v[0] * v[1];
    covar2d[1][0] = covar2d[0][1];
    covar2d[1][1] = v1c * u[1] * u[1] + v2c * v[1] * v[1];
}

} // namespace

/// @brief CUDA kernel functor for UT forward projection.
///
/// This struct owns tensor accessors, shared memory pointers, and scalar configuration for
/// projecting N gaussians into C camera views.
template <typename ScalarType, typename CameraOp> struct ProjectionForwardUT {
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
    const UTParams mUTParams;
    CameraOp mCameraOp;

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

    // Optional Outputs
    //
    // NOTE: This is intentionally a raw pointer to represent optional (nullable) outputs.
    // Required inputs are passed/stored as references where possible to avoid null-deref hazards.
    ScalarType *__restrict__ mOutCompensationsAcc; // [C, N] optional

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
                        const UTParams &utParams,
                        const CameraOp &cameraOp,
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
          mFarPlane(farPlane), mRadiusClip(minRadius2d), mUTParams(utParams), mCameraOp(cameraOp),
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

    /// @brief Project one gaussian for one camera.
    ///
    /// @param[in] idx Flattened index in \([0, C*N)\) mapping to (camId, gaussianId).
    /// @return true if the gaussian is projected successfully, false otherwise.
    inline __device__ void
    projectionForward(int64_t idx) {
        if (idx >= C * N) {
            return;
        }

        const int64_t camId      = idx / N;
        const int64_t gaussianId = idx % N;

        // Get Gaussian parameters
        const Vec3 meanWorldSpace(
            mMeansAcc[gaussianId][0], mMeansAcc[gaussianId][1], mMeansAcc[gaussianId][2]);
        const Vec4 quat_wxyz(mQuatsAcc[gaussianId][0],
                             mQuatsAcc[gaussianId][1],
                             mQuatsAcc[gaussianId][2],
                             mQuatsAcc[gaussianId][3]);
        const Vec3 scale_world(::cuda::std::exp(mLogScalesAcc[gaussianId][0]),
                               ::cuda::std::exp(mLogScalesAcc[gaussianId][1]),
                               ::cuda::std::exp(mLogScalesAcc[gaussianId][2]));

        // Depth culling should use the same shutter pose as projection:
        // - RollingShutterType::NONE: use start pose (t=0.0), matching
        //   `WorldToPixelTransform::projectWorldPoint` which uses the start transform when NONE.
        // - Rolling shutter modes: use center pose (t=0.5) as a conservative/stable cull.
        {
            const ScalarType tDepth = ScalarType(0.5);
            const ScalarType depth  = mCameraOp.cameraDepthAtTime(camId, meanWorldSpace, tDepth);
            if (depth < mNearPlane || depth > mFarPlane) {
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

        // Project sigma points through camera model
        nanovdb::math::Vec2<ScalarType> projected_points[7];
        bool valid_any                = false;
        constexpr int kNumSigmaPoints = 7;
        for (int i = 0; i < kNumSigmaPoints; ++i) {
            Vec2 pix;
            const ProjectWorldToPixelStatus status = mCameraOp.projectWorldPointToPixel(
                camId, sigma_points_world[i], ScalarType(mUTParams.inImageMargin), pix);
            // Hard reject if any sigma point is behind the camera since the projection will be
            // discontinuous.
            if (status == ProjectWorldToPixelStatus::BehindCamera) {
                mOutRadiiAcc[camId][gaussianId] = 0;
                return;
            }
            const bool valid_i  = (status == ProjectWorldToPixelStatus::InImage);
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
        Mat2 covar2d = reconstructCovarianceFromSigmaPoints(projected_points, weights_cov, mean2d);

        // Add blur for numerical stability
        ScalarType compensation;
        const ScalarType det_blur = addBlur(mEps2d, covar2d, compensation);
        if (det_blur <= ScalarType(0)) {
            mOutRadiiAcc[camId][gaussianId] = 0;
            return;
        }

        // Ensure reconstructed covariance is PSD to avoid NaNs when taking square-roots or
        // inverting.
        enforcePSD2x2(mEps2d, covar2d);

        const ScalarType det_psd = covar2d[0][0] * covar2d[1][1] - covar2d[0][1] * covar2d[1][0];
        if (!(det_psd > ScalarType(0))) {
            mOutRadiiAcc[camId][gaussianId] = 0;
            return;
        }

        const Mat2 covar2dInverse = covar2d.inverse();

        // Compute bounding box radius (similar to standard projection)
        const ScalarType b      = 0.5f * (covar2d[0][0] + covar2d[1][1]);
        const ScalarType tmp    = sqrtf(max(0.01f, b * b - det_psd));
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
        mOutDepthsAcc[camId][gaussianId] =
            mCameraOp.cameraDepthAtTime(camId, meanWorldSpace, ScalarType(0.5));
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
template <typename ScalarType, typename CameraOp>
__global__ __launch_bounds__(256) void
projectionForwardUTKernel(int64_t offset,
                          int64_t count,
                          ProjectionForwardUT<ScalarType, CameraOp> projectionForward) {
    alignas(nanovdb::math::Mat3<ScalarType>) extern __shared__ char sharedMemory[];
    projectionForward.mCameraOp.loadSharedMemory(sharedMemory);
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
    const DistortionModel cameraModel,
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

    // Validate UT hyperparameters on the host to avoid inf/NaNs from invalid scaling/weights.
    // In the 3D UT, D=3 and:
    //   lambda = alpha^2 * (D + kappa) - D
    //   denom  = D + lambda = alpha^2 * (D + kappa)
    // denom must be finite and strictly positive.
    constexpr float kUtDim = 3.0f;
    TORCH_CHECK_VALUE(std::isfinite(utParams.alpha), "utParams.alpha must be finite");
    TORCH_CHECK_VALUE(std::isfinite(utParams.beta), "utParams.beta must be finite");
    TORCH_CHECK_VALUE(std::isfinite(utParams.kappa), "utParams.kappa must be finite");
    TORCH_CHECK_VALUE(utParams.alpha > 0.0f, "utParams.alpha must be > 0");
    TORCH_CHECK_VALUE(kUtDim + utParams.kappa > 0.0f,
                      "utParams.kappa must satisfy (D + kappa) > 0 for the 3D UT (D=3)");
    const float denom = utParams.alpha * utParams.alpha * (kUtDim + utParams.kappa);
    TORCH_CHECK_VALUE(std::isfinite(denom) && denom > 0.0f,
                      "Invalid UTParams: expected denom = alpha^2*(D+kappa) to be finite and > 0");

    if (cameraModel == DistortionModel::PINHOLE || cameraModel == DistortionModel::ORTHOGRAPHIC) {
        // Distortion coefficients are ignored for these camera models.
        // (Intrinsics `projectionMatrices` are always used.)
    } else if (cameraModel == DistortionModel::OPENCV_RADTAN_5 ||
               cameraModel == DistortionModel::OPENCV_RATIONAL_8 ||
               cameraModel == DistortionModel::OPENCV_RADTAN_THIN_PRISM_9 ||
               cameraModel == DistortionModel::OPENCV_THIN_PRISM_12) {
        TORCH_CHECK_VALUE(distortionCoeffs.size(1) == 12,
                          "For DistortionModel::OPENCV_* , distortionCoeffs must have shape [C,12] "
                          "as [k1,k2,k3,k4,k5,k6,p1,p2,s1,s2,s3,s4]");
    } else {
        TORCH_CHECK_VALUE(false, "Unknown DistortionModel for GaussianProjectionForwardUT");
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
    if (cameraModel == DistortionModel::ORTHOGRAPHIC) {
        OrthographicWithDistortionCameraOp<scalar_t> cameraOp(worldToCamMatricesStart,
                                                              worldToCamMatricesEnd,
                                                              projectionMatrices,
                                                              static_cast<uint32_t>(C),
                                                              static_cast<int32_t>(imageWidth),
                                                              static_cast<int32_t>(imageHeight),
                                                              0,
                                                              0,
                                                              rollingShutterType);
        const size_t SHARED_MEM_SIZE = cameraOp.numSharedMemBytes();
        ProjectionForwardUT<scalar_t, OrthographicWithDistortionCameraOp<scalar_t>>
            projectionForward(imageWidth,
                              imageHeight,
                              eps2d,
                              nearPlane,
                              farPlane,
                              minRadius2d,
                              utParams,
                              cameraOp,
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
        projectionForwardUTKernel<scalar_t, OrthographicWithDistortionCameraOp<scalar_t>>
            <<<NUM_BLOCKS, 256, SHARED_MEM_SIZE, stream>>>(0, C * N, projectionForward);
    } else {
        PerspectiveWithDistortionCameraOp<scalar_t> cameraOp(worldToCamMatricesStart,
                                                             worldToCamMatricesEnd,
                                                             projectionMatrices,
                                                             distortionCoeffs,
                                                             static_cast<uint32_t>(C),
                                                             distortionCoeffs.size(1),
                                                             static_cast<int32_t>(imageWidth),
                                                             static_cast<int32_t>(imageHeight),
                                                             0,
                                                             0,
                                                             rollingShutterType,
                                                             cameraModel);
        const size_t SHARED_MEM_SIZE = cameraOp.numSharedMemBytes();
        ProjectionForwardUT<scalar_t, PerspectiveWithDistortionCameraOp<scalar_t>>
            projectionForward(imageWidth,
                              imageHeight,
                              eps2d,
                              nearPlane,
                              farPlane,
                              minRadius2d,
                              utParams,
                              cameraOp,
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
        projectionForwardUTKernel<scalar_t, PerspectiveWithDistortionCameraOp<scalar_t>>
            <<<NUM_BLOCKS, 256, SHARED_MEM_SIZE, stream>>>(0, C * N, projectionForward);
    }
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
    const DistortionModel cameraModel,
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
    const DistortionModel cameraModel,
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
