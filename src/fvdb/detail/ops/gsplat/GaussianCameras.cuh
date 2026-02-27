// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_GAUSSIANCAMERAS_CUH
#define FVDB_DETAIL_OPS_GSPLAT_GAUSSIANCAMERAS_CUH

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
    PINHOLE                    = 0,
    OPENCV_RADTAN_5            = 1,
    OPENCV_RATIONAL_8          = 2,
    OPENCV_RADTAN_THIN_PRISM_9 = 3,
    OPENCV_THIN_PRISM_12       = 4,
    ORTHOGRAPHIC               = 5,
};

/// @brief Unscented Transform hyperparameters.
struct UTParams {
    float alpha                       = 0.1f;
    float beta                        = 2.0f;
    float kappa                       = 0.0f;
    float inImageMargin               = 0.1f;
    bool requireAllSigmaPointsInImage = true;
};

} // namespace fvdb::detail::ops

#if defined(__CUDACC__)

#include <fvdb/detail/ops/gsplat/GaussianCameraAccessorCopy.cuh>
#include <fvdb/detail/ops/gsplat/GaussianCameraMatrixUtils.cuh>
#include <fvdb/detail/ops/gsplat/GaussianRigidTransform.cuh>
#include <fvdb/detail/ops/gsplat/GaussianUtils.cuh>
#include <fvdb/detail/utils/AccessorHelpers.cuh>

#include <nanovdb/math/Math.h>
#include <nanovdb/math/Ray.h>

#include <torch/types.h>

#include <cuda/std/cmath>

namespace fvdb::detail::ops {

/// @brief Return whether a projected Gaussian footprint is completely outside the image bounds.
template <typename T>
inline __device__ bool
isOutsideImageWithRadius(const nanovdb::math::Vec2<T> &mean2d,
                         const T radiusX,
                         const T radiusY,
                         const int32_t imageWidth,
                         const int32_t imageHeight) {
    return (mean2d[0] + radiusX <= T(0) || mean2d[0] - radiusX >= T(imageWidth) ||
            mean2d[1] + radiusY <= T(0) || mean2d[1] - radiusY >= T(imageHeight));
}

/// @brief Result state for world-point to pixel projection.
enum class ProjectWorldToPixelStatus : uint8_t { BehindCamera, OutOfBounds, InImage };

/// @brief Estimate projected radius from a 2x2 covariance using the largest eigenvalue heuristic.
template <typename T>
inline __device__ T
radiusFromCovariance2dDet(const nanovdb::math::Mat2<T> &covar2d,
                          const T det,
                          const T sigma,
                          const T detClamp = T(0.01)) {
    // Matches the classic FVDB/gsplat heuristic: use the largest eigenvalue of the 2x2 covariance
    // (via trace/determinant) and take `sigma` standard deviations.
    const T b  = T(0.5) * (covar2d[0][0] + covar2d[1][1]);
    const T v1 = b + ::cuda::std::sqrt(nanovdb::math::Max(detClamp, b * b - det));
    return ::cuda::std::ceil(sigma * ::cuda::std::sqrt(v1));
}

/// @brief Pack symmetric inverse covariance [[a,b],[b,c]] into row-major-3 [a,b,c].
template <typename T>
inline __device__ nanovdb::math::Vec3<T>
packConicRowMajor3(const nanovdb::math::Mat2<T> &covar2dInverse) {
    return nanovdb::math::Vec3<T>(covar2dInverse[0][0], covar2dInverse[0][1], covar2dInverse[1][1]);
}

/// @brief Unpack row-major-3 conic [a,b,c] into symmetric matrix [[a,b],[b,c]].
template <typename T>
inline __device__ nanovdb::math::Mat2<T>
loadConicRowMajor3(const T *conic3) {
    return nanovdb::math::Mat2<T>(conic3[0], conic3[1], conic3[1], conic3[2]);
}

/// @brief Unpack gradient wrt row-major-3 [a,b,c] into symmetric 2x2 gradient matrix.
template <typename T>
inline __device__ nanovdb::math::Mat2<T>
loadConicGradRowMajor3(const T *dConic3) {
    // Off-diagonal appears once in packed representation but corresponds to two symmetric entries
    // in the full 2x2 matrix.
    return nanovdb::math::Mat2<T>(dConic3[0], dConic3[1] * T(0.5), dConic3[1] * T(0.5), dConic3[2]);
}

/// @brief Unpack packed symmetric 3x3 covariance [xx,xy,xz,yy,yz,zz] into Mat3.
template <typename T>
inline __device__ nanovdb::math::Mat3<T>
loadCovarianceRowMajor6(const T *covars6) {
    // Packed symmetric layout: [xx, xy, xz, yy, yz, zz]
    return nanovdb::math::Mat3<T>(covars6[0],
                                  covars6[1],
                                  covars6[2], // 1st row
                                  covars6[1],
                                  covars6[3],
                                  covars6[4], // 2nd row
                                  covars6[2],
                                  covars6[4],
                                  covars6[5]  // 3rd row
    );
}

/// @brief Load quaternion [w,x,y,z] and scale = exp(logScale) from row-major arrays.
template <typename T>
inline __device__ void
loadQuatScaleFromLogScalesRowMajor(const T *quats4,
                                   const T *logScales3,
                                   nanovdb::math::Vec4<T> &outQuatWxyz,
                                   nanovdb::math::Vec3<T> &outScale) {
    outQuatWxyz = nanovdb::math::Vec4<T>(quats4[0], quats4[1], quats4[2], quats4[3]);
    outScale    = nanovdb::math::Vec3<T>(::cuda::std::exp(logScales3[0]),
                                      ::cuda::std::exp(logScales3[1]),
                                      ::cuda::std::exp(logScales3[2]));
}

/// @brief Load quaternion [w,x,y,z] and linear scale from row-major arrays.
template <typename T>
inline __device__ void
loadQuatScaleFromScalesRowMajor(const T *quats4,
                                const T *scales3,
                                nanovdb::math::Vec4<T> &outQuatWxyz,
                                nanovdb::math::Vec3<T> &outScale) {
    outQuatWxyz = nanovdb::math::Vec4<T>(quats4[0], quats4[1], quats4[2], quats4[3]);
    outScale    = nanovdb::math::Vec3<T>(scales3[0], scales3[1], scales3[2]);
}

/// @brief Pinhole camera model without distortion for projection/rasterization kernels.
template <typename T> struct PerspectiveCameraOp {
  public:
    using Mat3 = nanovdb::math::Mat3<T>;
    using Vec3 = nanovdb::math::Vec3<T>;

    PerspectiveCameraOp(const torch::Tensor &projectionMatrices,
                        const torch::Tensor &worldToCamMatrices,
                        int32_t numCameras,
                        int32_t imageWidth,
                        int32_t imageHeight,
                        T nearPlane,
                        T farPlane)
        : projectionMatricesAcc(
              projectionMatrices.template packed_accessor32<T, 3, torch::RestrictPtrTraits>()),
          worldToCamMatricesAcc(
              worldToCamMatrices.template packed_accessor32<T, 3, torch::RestrictPtrTraits>()),
          numCameras(numCameras), imageWidth(imageWidth), imageHeight(imageHeight),
          nearPlane(nearPlane), farPlane(farPlane) {}

  private:
    fvdb::TorchRAcc32<T, 3> projectionMatricesAcc; // [C,3,3]
    fvdb::TorchRAcc32<T, 3> worldToCamMatricesAcc; // [C,4,4]
    int32_t numCameras  = 0;
    int32_t imageWidth  = 0;
    int32_t imageHeight = 0;
    T nearPlane         = T(-1e10);
    T farPlane          = T(1e10);

    Mat3 *__restrict__ projectionMatsShared        = nullptr; // [C,3,3], optional
    Mat3 *__restrict__ worldToCamRotMatsShared     = nullptr; // [C,3,3], optional
    Vec3 *__restrict__ worldToCamTranslationShared = nullptr; // [C,3], optional

  public:
    /// @brief Bytes of dynamic shared memory needed to cache camera matrices.
    inline __host__ __device__ size_t
    numSharedMemBytes() const {
        return static_cast<size_t>(2 * numCameras) * sizeof(Mat3) +
               static_cast<size_t>(numCameras) * sizeof(Vec3);
    }

    /// @brief Load per-camera intrinsics/extrinsics into dynamic shared memory.
    inline __device__ void
    loadSharedMemory(void *sharedMemory) {
        const int64_t C             = numCameras;
        projectionMatsShared        = reinterpret_cast<Mat3 *>(sharedMemory);
        worldToCamRotMatsShared     = projectionMatsShared + C;
        worldToCamTranslationShared = reinterpret_cast<Vec3 *>(worldToCamRotMatsShared + C);
        copyMat3Accessor<T>(C, projectionMatsShared, projectionMatricesAcc);
        copyMat3Accessor<T>(C, worldToCamRotMatsShared, worldToCamMatricesAcc);
        copyWorldToCamTranslation<T>(C, worldToCamTranslationShared, worldToCamMatricesAcc);
    }

    /// @brief Check whether a camera-space depth lies in the clipping range.
    inline __device__ bool
    isDepthVisible(const T depth) const {
        return depth >= nearPlane && depth <= farPlane;
    }

    inline __host__ __device__ int32_t
    imageWidthPx() const {
        return imageWidth;
    }

    inline __host__ __device__ int32_t
    imageHeightPx() const {
        return imageHeight;
    }

    inline __device__ std::tuple<Mat3, Vec3>
    worldToCamRt(const int64_t cid) const {
        if (worldToCamRotMatsShared != nullptr) {
            return std::make_tuple(worldToCamRotMatsShared[cid], worldToCamTranslationShared[cid]);
        }
        const auto W = worldToCamMatricesAcc[cid];
        return std::make_tuple(
            Mat3(W[0][0], W[0][1], W[0][2], W[1][0], W[1][1], W[1][2], W[2][0], W[2][1], W[2][2]),
            Vec3(W[0][3], W[1][3], W[2][3]));
    }

    inline __device__ std::tuple<nanovdb::math::Mat2<T>, nanovdb::math::Vec2<T>>
    projectTo2DGaussian(const int64_t cid,
                        const nanovdb::math::Vec3<T> &meansCamSpace,
                        const nanovdb::math::Mat3<T> &covarCamSpace) const {
        using Mat2x3 = nanovdb::math::Mat2x3<T>;
        using Mat2   = nanovdb::math::Mat2<T>;
        using Vec2   = nanovdb::math::Vec2<T>;

        T fx, fy, cx, cy;
        if (projectionMatsShared != nullptr) {
            const Mat3 &K = projectionMatsShared[cid];
            fx            = K[0][0];
            fy            = K[1][1];
            cx            = K[0][2];
            cy            = K[1][2];
        } else {
            const auto K = projectionMatricesAcc[cid];
            fx           = K[0][0];
            fy           = K[1][1];
            cx           = K[0][2];
            cy           = K[1][2];
        }
        const T x = meansCamSpace[0];
        const T y = meansCamSpace[1];
        const T z = meansCamSpace[2];

        const T tanFovX = T(0.5) * T(imageWidth) / fx;
        const T tanFovY = T(0.5) * T(imageHeight) / fy;
        const T limXPos = (T(imageWidth) - cx) / fx + T(0.3) * tanFovX;
        const T limXNeg = cx / fx + T(0.3) * tanFovX;
        const T limYPos = (T(imageHeight) - cy) / fy + T(0.3) * tanFovY;
        const T limYNeg = cy / fy + T(0.3) * tanFovY;

        const T rz  = T(1) / z;
        const T rz2 = rz * rz;
        const T tx  = z * nanovdb::math::Min(limXPos, nanovdb::math::Max(-limXNeg, x * rz));
        const T ty  = z * nanovdb::math::Min(limYPos, nanovdb::math::Max(-limYNeg, y * rz));

        const Mat2x3 J(fx * rz,
                       T(0),
                       -fx * tx * rz2, // 1st row
                       T(0),
                       fy * rz,
                       -fy * ty * rz2  // 2nd row
        );
        const Mat2 cov2d = J * covarCamSpace * J.transpose();
        const Vec2 mean2d({fx * x * rz + cx, fy * y * rz + cy});
        return {cov2d, mean2d};
    }

    inline __device__ std::tuple<nanovdb::math::Mat3<T>, nanovdb::math::Vec3<T>>
    projectTo2DGaussianVJP(const int64_t cid,
                           const nanovdb::math::Vec3<T> &meansCamSpace,
                           const nanovdb::math::Mat3<T> &covarCamSpace,
                           const nanovdb::math::Mat2<T> &dLossDCovar2d,
                           const nanovdb::math::Vec2<T> &dLossDMeans2d) const {
        using Mat2x3 = nanovdb::math::Mat2x3<T>;
        using Mat3   = nanovdb::math::Mat3<T>;
        using Vec3   = nanovdb::math::Vec3<T>;

        T fx, fy, cx, cy;
        if (projectionMatsShared != nullptr) {
            const Mat3 &K = projectionMatsShared[cid];
            fx            = K[0][0];
            fy            = K[1][1];
            cx            = K[0][2];
            cy            = K[1][2];
        } else {
            const auto K = projectionMatricesAcc[cid];
            fx           = K[0][0];
            fy           = K[1][1];
            cx           = K[0][2];
            cy           = K[1][2];
        }
        const T x = meansCamSpace[0];
        const T y = meansCamSpace[1];
        const T z = meansCamSpace[2];

        const T tanFovX = T(0.5) * T(imageWidth) / fx;
        const T tanFovY = T(0.5) * T(imageHeight) / fy;
        const T limXPos = (T(imageWidth) - cx) / fx + T(0.3) * tanFovX;
        const T limXNeg = cx / fx + T(0.3) * tanFovX;
        const T limYPos = (T(imageHeight) - cy) / fy + T(0.3) * tanFovY;
        const T limYNeg = cy / fy + T(0.3) * tanFovY;

        const T rz  = T(1) / z;
        const T rz2 = rz * rz;
        const T tx  = z * nanovdb::math::Min(limXPos, nanovdb::math::Max(-limXNeg, x * rz));
        const T ty  = z * nanovdb::math::Min(limYPos, nanovdb::math::Max(-limYNeg, y * rz));

        const Mat2x3 J(fx * rz,
                       T(0),
                       -fx * tx * rz2, // 1st row
                       T(0),
                       fy * rz,
                       -fy * ty * rz2  // 2nd row
        );

        const Mat3 dLossDCovar3d(J.transpose() * dLossDCovar2d * J);
        Vec3 dLossDMean3d(fx * rz * dLossDMeans2d[0],
                          fy * rz * dLossDMeans2d[1],
                          -(fx * x * dLossDMeans2d[0] + fy * y * dLossDMeans2d[1]) * rz2);

        const T rz3     = rz2 * rz;
        const Mat2x3 dJ = dLossDCovar2d * J * covarCamSpace.transpose() +
                          dLossDCovar2d.transpose() * J * covarCamSpace;

        if (x * rz <= limXPos && x * rz >= -limXNeg) {
            dLossDMean3d[0] += -fx * rz2 * dJ[0][2];
        } else {
            dLossDMean3d[2] += -fx * rz3 * dJ[0][2] * tx;
        }
        if (y * rz <= limYPos && y * rz >= -limYNeg) {
            dLossDMean3d[1] += -fy * rz2 * dJ[1][2];
        } else {
            dLossDMean3d[2] += -fy * rz3 * dJ[1][2] * ty;
        }
        dLossDMean3d[2] += -fx * rz2 * dJ[0][0] - fy * rz2 * dJ[1][1] +
                           T(2) * fx * tx * rz3 * dJ[0][2] + T(2) * fy * ty * rz3 * dJ[1][2];
        return {dLossDCovar3d, dLossDMean3d};
    }
};

/// @brief Orthographic camera model without distortion for projection/rasterization kernels.
template <typename T> struct OrthographicCameraOp {
  public:
    using Mat3 = nanovdb::math::Mat3<T>;
    using Vec3 = nanovdb::math::Vec3<T>;

    OrthographicCameraOp(const torch::Tensor &projectionMatrices,
                         const torch::Tensor &worldToCamMatrices,
                         int32_t numCameras,
                         int32_t imageWidth,
                         int32_t imageHeight,
                         T nearPlane,
                         T farPlane)
        : projectionMatricesAcc(
              projectionMatrices.template packed_accessor32<T, 3, torch::RestrictPtrTraits>()),
          worldToCamMatricesAcc(
              worldToCamMatrices.template packed_accessor32<T, 3, torch::RestrictPtrTraits>()),
          numCameras(numCameras), imageWidth(imageWidth), imageHeight(imageHeight),
          nearPlane(nearPlane), farPlane(farPlane) {}

  private:
    fvdb::TorchRAcc32<T, 3> projectionMatricesAcc; // [C,3,3]
    fvdb::TorchRAcc32<T, 3> worldToCamMatricesAcc; // [C,4,4]
    int32_t numCameras  = 0;
    int32_t imageWidth  = 0;
    int32_t imageHeight = 0;
    T nearPlane         = T(-1e10);
    T farPlane          = T(1e10);

    Mat3 *__restrict__ projectionMatsShared        = nullptr; // [C,3,3], optional
    Mat3 *__restrict__ worldToCamRotMatsShared     = nullptr; // [C,3,3], optional
    Vec3 *__restrict__ worldToCamTranslationShared = nullptr; // [C,3], optional

  public:
    /// @brief Bytes of dynamic shared memory needed to cache camera matrices.
    inline __host__ __device__ size_t
    numSharedMemBytes() const {
        return static_cast<size_t>(2 * numCameras) * sizeof(Mat3) +
               static_cast<size_t>(numCameras) * sizeof(Vec3);
    }

    /// @brief Load per-camera intrinsics/extrinsics into dynamic shared memory.
    inline __device__ void
    loadSharedMemory(void *sharedMemory) {
        const int64_t C             = numCameras;
        projectionMatsShared        = reinterpret_cast<Mat3 *>(sharedMemory);
        worldToCamRotMatsShared     = projectionMatsShared + C;
        worldToCamTranslationShared = reinterpret_cast<Vec3 *>(worldToCamRotMatsShared + C);
        copyMat3Accessor<T>(C, projectionMatsShared, projectionMatricesAcc);
        copyMat3Accessor<T>(C, worldToCamRotMatsShared, worldToCamMatricesAcc);
        copyWorldToCamTranslation<T>(C, worldToCamTranslationShared, worldToCamMatricesAcc);
    }

    /// @brief Check whether a camera-space depth lies in the clipping range.
    inline __device__ bool
    isDepthVisible(const T depth) const {
        return depth >= nearPlane && depth <= farPlane;
    }

    inline __host__ __device__ int32_t
    imageWidthPx() const {
        return imageWidth;
    }

    inline __host__ __device__ int32_t
    imageHeightPx() const {
        return imageHeight;
    }

    inline __device__ std::tuple<Mat3, Vec3>
    worldToCamRt(const int64_t cid) const {
        if (worldToCamRotMatsShared != nullptr) {
            return std::make_tuple(worldToCamRotMatsShared[cid], worldToCamTranslationShared[cid]);
        }
        const auto W = worldToCamMatricesAcc[cid];
        return std::make_tuple(
            Mat3(W[0][0], W[0][1], W[0][2], W[1][0], W[1][1], W[1][2], W[2][0], W[2][1], W[2][2]),
            Vec3(W[0][3], W[1][3], W[2][3]));
    }

    inline __device__ std::tuple<nanovdb::math::Mat2<T>, nanovdb::math::Vec2<T>>
    projectTo2DGaussian(const int64_t cid,
                        const nanovdb::math::Vec3<T> &meansCamSpace,
                        const nanovdb::math::Mat3<T> &covarCamSpace) const {
        using Mat2x3 = nanovdb::math::Mat2x3<T>;
        using Mat2   = nanovdb::math::Mat2<T>;
        using Vec2   = nanovdb::math::Vec2<T>;

        T fx, fy, cx, cy;
        if (projectionMatsShared != nullptr) {
            const Mat3 &K = projectionMatsShared[cid];
            fx            = K[0][0];
            fy            = K[1][1];
            cx            = K[0][2];
            cy            = K[1][2];
        } else {
            const auto K = projectionMatricesAcc[cid];
            fx           = K[0][0];
            fy           = K[1][1];
            cx           = K[0][2];
            cy           = K[1][2];
        }
        const T x = meansCamSpace[0];
        const T y = meansCamSpace[1];
        const Mat2x3 J(fx,
                       T(0),
                       T(0), // 1st row
                       T(0),
                       fy,
                       T(0)  // 2nd row
        );
        const Mat2 cov2d = J * covarCamSpace * J.transpose();
        const Vec2 mean2d({fx * x + cx, fy * y + cy});
        return {cov2d, mean2d};
    }

    inline __device__ std::tuple<nanovdb::math::Mat3<T>, nanovdb::math::Vec3<T>>
    projectTo2DGaussianVJP(const int64_t cid,
                           const nanovdb::math::Vec3<T> &meansCamSpace,
                           const nanovdb::math::Mat3<T> &covarCamSpace,
                           const nanovdb::math::Mat2<T> &dLossDCovar2d,
                           const nanovdb::math::Vec2<T> &dLossDMeans2d) const {
        using Mat2x3 = nanovdb::math::Mat2x3<T>;
        using Mat3   = nanovdb::math::Mat3<T>;
        using Vec3   = nanovdb::math::Vec3<T>;

        T fx, fy, cx, cy;
        if (projectionMatsShared != nullptr) {
            const Mat3 &K = projectionMatsShared[cid];
            fx            = K[0][0];
            fy            = K[1][1];
            cx            = K[0][2];
            cy            = K[1][2];
        } else {
            const auto K = projectionMatricesAcc[cid];
            fx           = K[0][0];
            fy           = K[1][1];
            cx           = K[0][2];
            cy           = K[1][2];
        }
        const Mat2x3 J(fx,
                       T(0),
                       T(0), // 1st row
                       T(0),
                       fy,
                       T(0)  // 2nd row
        );
        const Mat3 dLossDCovar3d(J.transpose() * dLossDCovar2d * J);
        const Vec3 dLossDMean3d(fx * dLossDMeans2d[0], fy * dLossDMeans2d[1], T(0));
        return {dLossDCovar3d, dLossDMean3d};
    }
};

/// @brief Pinhole + OpenCV-distortion camera with optional rolling shutter support.
template <typename T> struct PerspectiveWithDistortionCameraOp {
  public:
    using Mat3 = nanovdb::math::Mat3<T>;
    using Vec3 = nanovdb::math::Vec3<T>;

    PerspectiveWithDistortionCameraOp(const torch::Tensor &worldToCamStart,
                                      const torch::Tensor &worldToCamEnd,
                                      const torch::Tensor &projectionMatrices,
                                      const torch::Tensor &distortionCoeffs,
                                      uint32_t numCameras,
                                      int64_t numDistCoeffs,
                                      int32_t imageWidth,
                                      int32_t imageHeight,
                                      int32_t imageOriginW,
                                      int32_t imageOriginH,
                                      RollingShutterType rollingShutterType,
                                      DistortionModel cameraModel)
        : worldToCamStartAcc(
              worldToCamStart.template packed_accessor32<T, 3, torch::RestrictPtrTraits>()),
          worldToCamEndAcc(
              worldToCamEnd.template packed_accessor32<T, 3, torch::RestrictPtrTraits>()),
          projectionMatricesAcc(
              projectionMatrices.template packed_accessor32<T, 3, torch::RestrictPtrTraits>()),
          distortionCoeffsAcc(
              distortionCoeffs.template packed_accessor32<T, 2, torch::RestrictPtrTraits>()),
          numCameras(numCameras), numDistCoeffs(numDistCoeffs), imageWidth(imageWidth),
          imageHeight(imageHeight), imageOriginW(imageOriginW), imageOriginH(imageOriginH),
          rollingShutterType(rollingShutterType), cameraModel(cameraModel) {}

  private:
    fvdb::TorchRAcc32<T, 3> worldToCamStartAcc;    // [C,4,4]
    fvdb::TorchRAcc32<T, 3> worldToCamEndAcc;      // [C,4,4]
    fvdb::TorchRAcc32<T, 3> projectionMatricesAcc; // [C,3,3]
    fvdb::TorchRAcc32<T, 2> distortionCoeffsAcc;   // [C,K]
    uint32_t numCameras                   = 0;
    int64_t numDistCoeffs                 = 0;
    int32_t imageWidth                    = 0;
    int32_t imageHeight                   = 0;
    int32_t imageOriginW                  = 0;
    int32_t imageOriginH                  = 0;
    RollingShutterType rollingShutterType = RollingShutterType::NONE;
    DistortionModel cameraModel           = DistortionModel::PINHOLE;

    Mat3 *__restrict__ worldToCamStartRotShared   = nullptr; // [C,3,3], optional
    Vec3 *__restrict__ worldToCamStartTransShared = nullptr; // [C,3], optional
    Mat3 *__restrict__ worldToCamEndRotShared     = nullptr; // [C,3,3], optional
    Vec3 *__restrict__ worldToCamEndTransShared   = nullptr; // [C,3], optional
    Mat3 *__restrict__ projectionMatsShared       = nullptr; // [C,3,3], optional
    T *__restrict__ distortionShared              = nullptr; // [C,K], optional

  public:
    /// @brief Bytes of dynamic shared memory needed to cache camera state.
    inline __host__ __device__ size_t
    numSharedMemBytes() const {
        return static_cast<size_t>(numCameras) * (3 * sizeof(Mat3) + 2 * sizeof(Vec3)) +
               static_cast<size_t>(numCameras * numDistCoeffs) * sizeof(T);
    }

    /// @brief Load per-camera state (poses, intrinsics, distortion) into shared memory.
    inline __device__ void
    loadSharedMemory(void *sharedMemory) {
        char *ptr                = reinterpret_cast<char *>(sharedMemory);
        worldToCamStartRotShared = reinterpret_cast<Mat3 *>(ptr);
        ptr += numCameras * sizeof(Mat3);
        worldToCamStartTransShared = reinterpret_cast<Vec3 *>(ptr);
        ptr += numCameras * sizeof(Vec3);
        worldToCamEndRotShared = reinterpret_cast<Mat3 *>(ptr);
        ptr += numCameras * sizeof(Mat3);
        worldToCamEndTransShared = reinterpret_cast<Vec3 *>(ptr);
        ptr += numCameras * sizeof(Vec3);
        projectionMatsShared = reinterpret_cast<Mat3 *>(ptr);
        ptr += numCameras * sizeof(Mat3);
        distortionShared = reinterpret_cast<T *>(ptr);
        copyMat3Accessor<T>(numCameras, worldToCamStartRotShared, worldToCamStartAcc);
        copyWorldToCamTranslation<T>(numCameras, worldToCamStartTransShared, worldToCamStartAcc);
        copyMat3Accessor<T>(numCameras, worldToCamEndRotShared, worldToCamEndAcc);
        copyWorldToCamTranslation<T>(numCameras, worldToCamEndTransShared, worldToCamEndAcc);
        copyMat3Accessor<T>(numCameras, projectionMatsShared, projectionMatricesAcc);
        if (numDistCoeffs > 0) {
            copyDistortionCoeffs<T>(
                numCameras, numDistCoeffs, distortionShared, distortionCoeffsAcc);
        }
    }

    /// @brief Compute camera-space depth at a given rolling-shutter time.
    inline __device__ T
    cameraDepthAtTime(const int64_t cid,
                      const nanovdb::math::Vec3<T> &pointWorld,
                      const T shutterTime) const {
        const auto [R_wc_start, t_wc_start, R_wc_end, t_wc_end] = worldToCamRtStartEnd(cid);
        nanovdb::math::Mat3<T> R_wc;
        nanovdb::math::Vec3<T> t_wc;
        if (rollingShutterType == RollingShutterType::NONE) {
            R_wc = R_wc_start;
            t_wc = t_wc_start;
        } else {
            const RigidTransform<T> worldToCamStart(R_wc_start, t_wc_start);
            const RigidTransform<T> worldToCamEnd(R_wc_end, t_wc_end);
            const RigidTransform<T> worldToCam =
                RigidTransform<T>::interpolate(shutterTime, worldToCamStart, worldToCamEnd);
            R_wc = worldToCam.R;
            t_wc = worldToCam.t;
        }
        const nanovdb::math::Vec3<T> pointCam = R_wc * pointWorld + t_wc;
        return pointCam[2];
    }

    /// @brief Project a world-space point to pixel coordinates and classify visibility.
    inline __device__ ProjectWorldToPixelStatus
    projectWorldPointToPixel(const int64_t cid,
                             const nanovdb::math::Vec3<T> &pointWorld,
                             const T inImageMargin,
                             nanovdb::math::Vec2<T> &outPixel) const {
        const auto [R_wc_start, t_wc_start, R_wc_end, t_wc_end] = worldToCamRtStartEnd(cid);
        const Mat3 K                                            = projectionMatrix(cid);
        const T *distCoeffs                                     = distortionPtr(cid);

        const auto projectWithTransform =
            [&](const RigidTransform<T> &worldToCam,
                nanovdb::math::Vec2<T> &pix) -> ProjectWorldToPixelStatus {
            const nanovdb::math::Vec3<T> pointCam = worldToCam.apply(pointWorld);
            if (pointCam[2] <= T(1e-6)) {
                pix = nanovdb::math::Vec2<T>(T(0), T(0));
                return ProjectWorldToPixelStatus::BehindCamera;
            }
            const nanovdb::math::Vec2<T> pNormalized(pointCam[0] / pointCam[2],
                                                     pointCam[1] / pointCam[2]);
            const nanovdb::math::Vec2<T> pDistorted =
                applyOpenCVDistortion(cameraModel, pNormalized, distCoeffs, numDistCoeffs);
            const T fx = K[0][0];
            const T fy = K[1][1];
            const T cx = K[0][2];
            const T cy = K[1][2];
            pix        = nanovdb::math::Vec2<T>(fx * pDistorted[0] + cx, fy * pDistorted[1] + cy);
            const T marginX  = T(imageWidth) * inImageMargin;
            const T marginY  = T(imageHeight) * inImageMargin;
            const bool inImg = (pix[0] >= -marginX) && (pix[0] < T(imageWidth) + marginX) &&
                               (pix[1] >= -marginY) && (pix[1] < T(imageHeight) + marginY);
            return inImg ? ProjectWorldToPixelStatus::InImage
                         : ProjectWorldToPixelStatus::OutOfBounds;
        };

        const RigidTransform<T> worldToCamStart(R_wc_start, t_wc_start);
        const RigidTransform<T> worldToCamEnd(R_wc_end, t_wc_end);
        nanovdb::math::Vec2<T> pixStart(T(0), T(0));
        nanovdb::math::Vec2<T> pixEnd(T(0), T(0));
        const ProjectWorldToPixelStatus statusStart =
            projectWithTransform(worldToCamStart, pixStart);
        const ProjectWorldToPixelStatus statusEnd = projectWithTransform(worldToCamEnd, pixEnd);

        if (rollingShutterType == RollingShutterType::NONE) {
            outPixel = pixStart;
            return statusStart;
        }
        if (statusStart == ProjectWorldToPixelStatus::BehindCamera &&
            statusEnd == ProjectWorldToPixelStatus::BehindCamera) {
            outPixel = pixEnd;
            return ProjectWorldToPixelStatus::BehindCamera;
        }
        if (statusStart != ProjectWorldToPixelStatus::InImage &&
            statusEnd != ProjectWorldToPixelStatus::InImage) {
            outPixel = (statusEnd != ProjectWorldToPixelStatus::BehindCamera) ? pixEnd : pixStart;
            return ProjectWorldToPixelStatus::OutOfBounds;
        }

        nanovdb::math::Vec2<T> pixPrev =
            (statusStart == ProjectWorldToPixelStatus::InImage) ? pixStart : pixEnd;
        constexpr int kIters = 6;
        for (int it = 0; it < kIters; ++it) {
            const T tRs = rollingShutterTimeFromPixel(
                rollingShutterType, pixPrev[0], pixPrev[1], imageWidth, imageHeight);
            const RigidTransform<T> worldToCam =
                RigidTransform<T>::interpolate(tRs, worldToCamStart, worldToCamEnd);
            nanovdb::math::Vec2<T> pixRs(T(0), T(0));
            const ProjectWorldToPixelStatus statusRs = projectWithTransform(worldToCam, pixRs);
            pixPrev                                  = pixRs;
            if (statusRs != ProjectWorldToPixelStatus::InImage) {
                outPixel = pixRs;
                return statusRs;
            }
        }
        outPixel = pixPrev;
        return ProjectWorldToPixelStatus::InImage;
    }

  private:
    inline __device__ std::tuple<Mat3, Vec3, Mat3, Vec3>
    worldToCamRtStartEnd(const int64_t cid) const {
        if (worldToCamStartRotShared != nullptr) {
            return std::make_tuple(worldToCamStartRotShared[cid],
                                   worldToCamStartTransShared[cid],
                                   worldToCamEndRotShared[cid],
                                   worldToCamEndTransShared[cid]);
        }
        const auto Ws = worldToCamStartAcc[cid];
        const auto We = worldToCamEndAcc[cid];
        return std::make_tuple(Mat3(Ws[0][0],
                                    Ws[0][1],
                                    Ws[0][2],
                                    Ws[1][0],
                                    Ws[1][1],
                                    Ws[1][2],
                                    Ws[2][0],
                                    Ws[2][1],
                                    Ws[2][2]),
                               Vec3(Ws[0][3], Ws[1][3], Ws[2][3]),
                               Mat3(We[0][0],
                                    We[0][1],
                                    We[0][2],
                                    We[1][0],
                                    We[1][1],
                                    We[1][2],
                                    We[2][0],
                                    We[2][1],
                                    We[2][2]),
                               Vec3(We[0][3], We[1][3], We[2][3]));
    }

    inline __device__ Mat3
    projectionMatrix(const int64_t cid) const {
        if (projectionMatsShared != nullptr) {
            return projectionMatsShared[cid];
        }
        const auto K = projectionMatricesAcc[cid];
        return Mat3(
            K[0][0], K[0][1], K[0][2], K[1][0], K[1][1], K[1][2], K[2][0], K[2][1], K[2][2]);
    }

    inline __device__ const T *
    distortionPtr(const int64_t cid) const {
        if (numDistCoeffs <= 0) {
            return nullptr;
        }
        if (distortionShared != nullptr) {
            return distortionShared + cid * numDistCoeffs;
        }
        return &distortionCoeffsAcc[cid][0];
    }

    inline __device__ nanovdb::math::Vec3<T>
    normalizeSafe(const nanovdb::math::Vec3<T> &v) const {
        const T n2 = v.dot(v);
        if (n2 > T(0)) {
            return v * (T(1) / sqrt(n2));
        }
        return nanovdb::math::Vec3<T>(T(0), T(0), T(0));
    }

    inline static __device__ T
    clamp01(const T x) {
        return (x < T(0)) ? T(0) : ((x > T(1)) ? T(1) : x);
    }

    inline static __device__ T
    rollingShutterTimeFromPixel(const RollingShutterType rollingShutterType,
                                const T px,
                                const T py,
                                const int64_t imageWidth,
                                const int64_t imageHeight) {
        if (rollingShutterType == RollingShutterType::NONE) {
            return T(0);
        }
        T u = T(0);
        if (rollingShutterType == RollingShutterType::VERTICAL) {
            const T denom = (imageHeight > 1) ? T(imageHeight - 1) : T(1);
            u             = ::cuda::std::floor(py) / denom;
        } else if (rollingShutterType == RollingShutterType::HORIZONTAL) {
            const T denom = (imageWidth > 1) ? T(imageWidth - 1) : T(1);
            u             = ::cuda::std::floor(px) / denom;
        }
        return clamp01(u);
    }

    inline static __device__ nanovdb::math::Vec2<T>
    applyOpenCVDistortion(const DistortionModel model,
                          const nanovdb::math::Vec2<T> &pNormalized,
                          const T *distCoeffs,
                          const int64_t numCoeffs) {
        if (model == DistortionModel::PINHOLE || model == DistortionModel::ORTHOGRAPHIC ||
            numCoeffs == 0 || distCoeffs == nullptr) {
            return pNormalized;
        }
        const T x  = pNormalized[0];
        const T y  = pNormalized[1];
        const T x2 = x * x;
        const T y2 = y * y;
        const T xy = x * y;
        const T r2 = x2 + y2;
        const T r4 = r2 * r2;
        const T r6 = r4 * r2;
        const T k1 = distCoeffs[0];
        const T k2 = distCoeffs[1];
        const T k3 = distCoeffs[2];
        const T k4 = distCoeffs[3];
        const T k5 = distCoeffs[4];
        const T k6 = distCoeffs[5];
        const T p1 = distCoeffs[6];
        const T p2 = distCoeffs[7];
        const T s1 = distCoeffs[8];
        const T s2 = distCoeffs[9];
        const T s3 = distCoeffs[10];
        const T s4 = distCoeffs[11];
        T radial   = T(1);
        if (model == DistortionModel::OPENCV_RATIONAL_8 ||
            model == DistortionModel::OPENCV_THIN_PRISM_12) {
            const T num = T(1) + r2 * (k1 + r2 * (k2 + r2 * k3));
            const T den = T(1) + r2 * (k4 + r2 * (k5 + r2 * k6));
            radial      = (den != T(0)) ? (num / den) : T(0);
        } else if (model == DistortionModel::OPENCV_RADTAN_5 ||
                   model == DistortionModel::OPENCV_RADTAN_THIN_PRISM_9) {
            radial = T(1) + k1 * r2 + k2 * r4 + k3 * r6;
        }
        T xDist = x * radial;
        T yDist = y * radial;
        xDist += T(2) * p1 * xy + p2 * (r2 + T(2) * x2);
        yDist += p1 * (r2 + T(2) * y2) + T(2) * p2 * xy;
        if (model == DistortionModel::OPENCV_THIN_PRISM_12 ||
            model == DistortionModel::OPENCV_RADTAN_THIN_PRISM_9) {
            xDist += s1 * r2 + s2 * r4;
            yDist += s3 * r2 + s4 * r4;
        }
        return nanovdb::math::Vec2<T>(xDist, yDist);
    }

    inline static __device__ nanovdb::math::Vec2<T>
    undistortOpenCV(const DistortionModel model,
                    const nanovdb::math::Vec2<T> &pDistorted,
                    const T *distCoeffs,
                    const int64_t numCoeffs,
                    const int iters = 8) {
        if (model == DistortionModel::PINHOLE || model == DistortionModel::ORTHOGRAPHIC ||
            numCoeffs == 0 || distCoeffs == nullptr) {
            return pDistorted;
        }
        nanovdb::math::Vec2<T> x = pDistorted;
        for (int it = 0; it < iters; ++it) {
            const nanovdb::math::Vec2<T> xDist =
                applyOpenCVDistortion(model, x, distCoeffs, numCoeffs);
            const nanovdb::math::Vec2<T> err = xDist - pDistorted;
            x[0] -= err[0];
            x[1] -= err[1];
        }
        return x;
    }

  public:
    /// @brief Unproject a pixel center to a world-space ray.
    inline __device__ nanovdb::math::Ray<T>
    projectToRay(const int64_t cid, const uint32_t row, const uint32_t col) const {
        const auto [R_wc_start, t_wc_start, R_wc_end, t_wc_end] = worldToCamRtStartEnd(cid);
        const Mat3 K                                            = projectionMatrix(cid);
        const T *distCoeffs                                     = distortionPtr(cid);
        const T px                                              = T(col) + T(imageOriginW) + T(0.5);
        const T py                                              = T(row) + T(imageOriginH) + T(0.5);

        const T u =
            rollingShutterTimeFromPixel(rollingShutterType, px, py, imageWidth, imageHeight);

        nanovdb::math::Mat3<T> R_wc;
        nanovdb::math::Vec3<T> t_wc;
        if (rollingShutterType == RollingShutterType::NONE) {
            R_wc = R_wc_start;
            t_wc = t_wc_start;
        } else {
            const RigidTransform<T> worldToCamStart(R_wc_start, t_wc_start);
            const RigidTransform<T> worldToCamEnd(R_wc_end, t_wc_end);
            const RigidTransform<T> worldToCam =
                RigidTransform<T>::interpolate(u, worldToCamStart, worldToCamEnd);
            R_wc = worldToCam.R;
            t_wc = worldToCam.t;
        }

        const nanovdb::math::Mat3<T> R_cw = R_wc.transpose();
        const T fx                        = K[0][0];
        const T fy                        = K[1][1];
        const T cx                        = K[0][2];
        const T cy                        = K[1][2];
        const nanovdb::math::Vec2<T> p_distorted((px - cx) / fx, (py - cy) / fy);
        const nanovdb::math::Vec2<T> p =
            undistortOpenCV(cameraModel, p_distorted, distCoeffs, numDistCoeffs);

        const nanovdb::math::Vec3<T> d_cam =
            normalizeSafe(nanovdb::math::Vec3<T>(p[0], p[1], T(1)));
        const nanovdb::math::Vec3<T> origin_w =
            R_cw * (nanovdb::math::Vec3<T>(T(0), T(0), T(0)) - t_wc);
        const nanovdb::math::Vec3<T> dir_w = normalizeSafe(R_cw * d_cam);
        return nanovdb::math::Ray<T>(origin_w, dir_w);
    }
};

/// @brief Orthographic camera with optional rolling shutter support.
template <typename T> struct OrthographicWithDistortionCameraOp {
  public:
    using Mat3 = nanovdb::math::Mat3<T>;
    using Vec3 = nanovdb::math::Vec3<T>;

    OrthographicWithDistortionCameraOp(const torch::Tensor &worldToCamStart,
                                       const torch::Tensor &worldToCamEnd,
                                       const torch::Tensor &projectionMatrices,
                                       uint32_t numCameras,
                                       int32_t imageWidth,
                                       int32_t imageHeight,
                                       int32_t imageOriginW,
                                       int32_t imageOriginH,
                                       RollingShutterType rollingShutterType)
        : worldToCamStartAcc(
              worldToCamStart.template packed_accessor32<T, 3, torch::RestrictPtrTraits>()),
          worldToCamEndAcc(
              worldToCamEnd.template packed_accessor32<T, 3, torch::RestrictPtrTraits>()),
          projectionMatricesAcc(
              projectionMatrices.template packed_accessor32<T, 3, torch::RestrictPtrTraits>()),
          numCameras(numCameras), imageWidth(imageWidth), imageHeight(imageHeight),
          imageOriginW(imageOriginW), imageOriginH(imageOriginH),
          rollingShutterType(rollingShutterType) {}

  private:
    fvdb::TorchRAcc32<T, 3> worldToCamStartAcc;    // [C,4,4]
    fvdb::TorchRAcc32<T, 3> worldToCamEndAcc;      // [C,4,4]
    fvdb::TorchRAcc32<T, 3> projectionMatricesAcc; // [C,3,3]
    uint32_t numCameras                   = 0;
    int32_t imageWidth                    = 0;
    int32_t imageHeight                   = 0;
    int32_t imageOriginW                  = 0;
    int32_t imageOriginH                  = 0;
    RollingShutterType rollingShutterType = RollingShutterType::NONE;

    Mat3 *__restrict__ worldToCamStartRotShared   = nullptr; // [C,3,3], optional
    Vec3 *__restrict__ worldToCamStartTransShared = nullptr; // [C,3], optional
    Mat3 *__restrict__ worldToCamEndRotShared     = nullptr; // [C,3,3], optional
    Vec3 *__restrict__ worldToCamEndTransShared   = nullptr; // [C,3], optional
    Mat3 *__restrict__ projectionMatsShared       = nullptr; // [C,3,3], optional

  public:
    /// @brief Bytes of dynamic shared memory needed to cache camera state.
    inline __host__ __device__ size_t
    numSharedMemBytes() const {
        return static_cast<size_t>(numCameras) * (3 * sizeof(Mat3) + 2 * sizeof(Vec3));
    }

    /// @brief Load per-camera state (poses and intrinsics) into shared memory.
    inline __device__ void
    loadSharedMemory(void *sharedMemory) {
        char *ptr                = reinterpret_cast<char *>(sharedMemory);
        worldToCamStartRotShared = reinterpret_cast<Mat3 *>(ptr);
        ptr += numCameras * sizeof(Mat3);
        worldToCamStartTransShared = reinterpret_cast<Vec3 *>(ptr);
        ptr += numCameras * sizeof(Vec3);
        worldToCamEndRotShared = reinterpret_cast<Mat3 *>(ptr);
        ptr += numCameras * sizeof(Mat3);
        worldToCamEndTransShared = reinterpret_cast<Vec3 *>(ptr);
        ptr += numCameras * sizeof(Vec3);
        projectionMatsShared = reinterpret_cast<Mat3 *>(ptr);
        copyMat3Accessor<T>(numCameras, worldToCamStartRotShared, worldToCamStartAcc);
        copyWorldToCamTranslation<T>(numCameras, worldToCamStartTransShared, worldToCamStartAcc);
        copyMat3Accessor<T>(numCameras, worldToCamEndRotShared, worldToCamEndAcc);
        copyWorldToCamTranslation<T>(numCameras, worldToCamEndTransShared, worldToCamEndAcc);
        copyMat3Accessor<T>(numCameras, projectionMatsShared, projectionMatricesAcc);
    }

    /// @brief Compute camera-space depth at a given rolling-shutter time.
    inline __device__ T
    cameraDepthAtTime(const int64_t cid,
                      const nanovdb::math::Vec3<T> &pointWorld,
                      const T shutterTime) const {
        const auto [R_wc_start, t_wc_start, R_wc_end, t_wc_end] = worldToCamRtStartEnd(cid);
        nanovdb::math::Mat3<T> R_wc;
        nanovdb::math::Vec3<T> t_wc;
        if (rollingShutterType == RollingShutterType::NONE) {
            R_wc = R_wc_start;
            t_wc = t_wc_start;
        } else {
            const RigidTransform<T> worldToCamStart(R_wc_start, t_wc_start);
            const RigidTransform<T> worldToCamEnd(R_wc_end, t_wc_end);
            const RigidTransform<T> worldToCam =
                RigidTransform<T>::interpolate(shutterTime, worldToCamStart, worldToCamEnd);
            R_wc = worldToCam.R;
            t_wc = worldToCam.t;
        }
        const nanovdb::math::Vec3<T> pointCam = R_wc * pointWorld + t_wc;
        return pointCam[2];
    }

    /// @brief Project a world-space point to pixel coordinates and classify visibility.
    inline __device__ ProjectWorldToPixelStatus
    projectWorldPointToPixel(const int64_t cid,
                             const nanovdb::math::Vec3<T> &pointWorld,
                             const T inImageMargin,
                             nanovdb::math::Vec2<T> &outPixel) const {
        const auto [R_wc_start, t_wc_start, R_wc_end, t_wc_end] = worldToCamRtStartEnd(cid);
        const Mat3 K                                            = projectionMatrix(cid);
        const T fx                                              = K[0][0];
        const T fy                                              = K[1][1];
        const T cx                                              = K[0][2];
        const T cy                                              = K[1][2];

        const auto projectWithTransform =
            [&](const RigidTransform<T> &worldToCam,
                nanovdb::math::Vec2<T> &pix) -> ProjectWorldToPixelStatus {
            const nanovdb::math::Vec3<T> pointCam = worldToCam.apply(pointWorld);
            if (pointCam[2] <= T(0)) {
                pix = nanovdb::math::Vec2<T>(T(0), T(0));
                return ProjectWorldToPixelStatus::BehindCamera;
            }
            pix              = nanovdb::math::Vec2<T>(fx * pointCam[0] + cx, fy * pointCam[1] + cy);
            const T marginX  = T(imageWidth) * inImageMargin;
            const T marginY  = T(imageHeight) * inImageMargin;
            const bool inImg = (pix[0] >= -marginX) && (pix[0] < T(imageWidth) + marginX) &&
                               (pix[1] >= -marginY) && (pix[1] < T(imageHeight) + marginY);
            return inImg ? ProjectWorldToPixelStatus::InImage
                         : ProjectWorldToPixelStatus::OutOfBounds;
        };

        const RigidTransform<T> worldToCamStart(R_wc_start, t_wc_start);
        const RigidTransform<T> worldToCamEnd(R_wc_end, t_wc_end);
        nanovdb::math::Vec2<T> pixStart(T(0), T(0));
        nanovdb::math::Vec2<T> pixEnd(T(0), T(0));
        const ProjectWorldToPixelStatus statusStart =
            projectWithTransform(worldToCamStart, pixStart);
        const ProjectWorldToPixelStatus statusEnd = projectWithTransform(worldToCamEnd, pixEnd);

        if (rollingShutterType == RollingShutterType::NONE) {
            outPixel = pixStart;
            return statusStart;
        }
        if (statusStart == ProjectWorldToPixelStatus::BehindCamera &&
            statusEnd == ProjectWorldToPixelStatus::BehindCamera) {
            outPixel = pixEnd;
            return ProjectWorldToPixelStatus::BehindCamera;
        }
        if (statusStart != ProjectWorldToPixelStatus::InImage &&
            statusEnd != ProjectWorldToPixelStatus::InImage) {
            outPixel = (statusEnd != ProjectWorldToPixelStatus::BehindCamera) ? pixEnd : pixStart;
            return ProjectWorldToPixelStatus::OutOfBounds;
        }

        nanovdb::math::Vec2<T> pixPrev =
            (statusStart == ProjectWorldToPixelStatus::InImage) ? pixStart : pixEnd;
        constexpr int kIters = 6;
        for (int it = 0; it < kIters; ++it) {
            const T tRs = rollingShutterTimeFromPixel(
                rollingShutterType, pixPrev[0], pixPrev[1], imageWidth, imageHeight);
            const RigidTransform<T> worldToCam =
                RigidTransform<T>::interpolate(tRs, worldToCamStart, worldToCamEnd);
            nanovdb::math::Vec2<T> pixRs(T(0), T(0));
            const ProjectWorldToPixelStatus statusRs = projectWithTransform(worldToCam, pixRs);
            pixPrev                                  = pixRs;
            if (statusRs != ProjectWorldToPixelStatus::InImage) {
                outPixel = pixRs;
                return statusRs;
            }
        }
        outPixel = pixPrev;
        return ProjectWorldToPixelStatus::InImage;
    }

  private:
    inline __device__ std::tuple<Mat3, Vec3, Mat3, Vec3>
    worldToCamRtStartEnd(const int64_t cid) const {
        if (worldToCamStartRotShared != nullptr) {
            return std::make_tuple(worldToCamStartRotShared[cid],
                                   worldToCamStartTransShared[cid],
                                   worldToCamEndRotShared[cid],
                                   worldToCamEndTransShared[cid]);
        }
        const auto Ws = worldToCamStartAcc[cid];
        const auto We = worldToCamEndAcc[cid];
        return std::make_tuple(Mat3(Ws[0][0],
                                    Ws[0][1],
                                    Ws[0][2],
                                    Ws[1][0],
                                    Ws[1][1],
                                    Ws[1][2],
                                    Ws[2][0],
                                    Ws[2][1],
                                    Ws[2][2]),
                               Vec3(Ws[0][3], Ws[1][3], Ws[2][3]),
                               Mat3(We[0][0],
                                    We[0][1],
                                    We[0][2],
                                    We[1][0],
                                    We[1][1],
                                    We[1][2],
                                    We[2][0],
                                    We[2][1],
                                    We[2][2]),
                               Vec3(We[0][3], We[1][3], We[2][3]));
    }

    inline __device__ Mat3
    projectionMatrix(const int64_t cid) const {
        if (projectionMatsShared != nullptr) {
            return projectionMatsShared[cid];
        }
        const auto K = projectionMatricesAcc[cid];
        return Mat3(
            K[0][0], K[0][1], K[0][2], K[1][0], K[1][1], K[1][2], K[2][0], K[2][1], K[2][2]);
    }

    inline __device__ nanovdb::math::Vec3<T>
    normalizeSafe(const nanovdb::math::Vec3<T> &v) const {
        const T n2 = v.dot(v);
        if (n2 > T(0)) {
            return v * (T(1) / sqrt(n2));
        }
        return nanovdb::math::Vec3<T>(T(0), T(0), T(0));
    }

    inline static __device__ T
    clamp01(const T x) {
        return (x < T(0)) ? T(0) : ((x > T(1)) ? T(1) : x);
    }

    inline static __device__ T
    rollingShutterTimeFromPixel(const RollingShutterType rollingShutterType,
                                const T px,
                                const T py,
                                const int64_t imageWidth,
                                const int64_t imageHeight) {
        if (rollingShutterType == RollingShutterType::NONE) {
            return T(0);
        }
        T u = T(0);
        if (rollingShutterType == RollingShutterType::VERTICAL) {
            const T denom = (imageHeight > 1) ? T(imageHeight - 1) : T(1);
            u             = ::cuda::std::floor(py) / denom;
        } else if (rollingShutterType == RollingShutterType::HORIZONTAL) {
            const T denom = (imageWidth > 1) ? T(imageWidth - 1) : T(1);
            u             = ::cuda::std::floor(px) / denom;
        }
        return clamp01(u);
    }

  public:
    /// @brief Unproject a pixel center to a world-space orthographic ray.
    inline __device__ nanovdb::math::Ray<T>
    projectToRay(const int64_t cid, const uint32_t row, const uint32_t col) const {
        const auto [R_wc_start, t_wc_start, R_wc_end, t_wc_end] = worldToCamRtStartEnd(cid);
        const Mat3 K                                            = projectionMatrix(cid);
        const T px                                              = T(col) + T(imageOriginW) + T(0.5);
        const T py                                              = T(row) + T(imageOriginH) + T(0.5);

        const T u =
            rollingShutterTimeFromPixel(rollingShutterType, px, py, imageWidth, imageHeight);

        nanovdb::math::Mat3<T> R_wc;
        nanovdb::math::Vec3<T> t_wc;
        if (rollingShutterType == RollingShutterType::NONE) {
            R_wc = R_wc_start;
            t_wc = t_wc_start;
        } else {
            const RigidTransform<T> worldToCamStart(R_wc_start, t_wc_start);
            const RigidTransform<T> worldToCamEnd(R_wc_end, t_wc_end);
            const RigidTransform<T> worldToCam =
                RigidTransform<T>::interpolate(u, worldToCamStart, worldToCamEnd);
            R_wc = worldToCam.R;
            t_wc = worldToCam.t;
        }

        const nanovdb::math::Mat3<T> R_cw = R_wc.transpose();
        const T fx                        = K[0][0];
        const T fy                        = K[1][1];
        const T cx                        = K[0][2];
        const T cy                        = K[1][2];
        const nanovdb::math::Vec2<T> p((px - cx) / fx, (py - cy) / fy);

        const nanovdb::math::Vec3<T> o_cam(p[0], p[1], T(0));
        const nanovdb::math::Vec3<T> d_cam(T(0), T(0), T(1));
        const nanovdb::math::Vec3<T> origin_w = R_cw * (o_cam - t_wc);
        const nanovdb::math::Vec3<T> dir_w    = normalizeSafe(R_cw * d_cam);
        return nanovdb::math::Ray<T>(origin_w, dir_w);
    }
};

} // namespace fvdb::detail::ops

#endif // defined(__CUDACC__)

#endif // FVDB_DETAIL_OPS_GSPLAT_GAUSSIANCAMERAS_CUH
