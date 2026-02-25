// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_GAUSSIANPROJECTIONUTILS_CUH
#define FVDB_DETAIL_OPS_GSPLAT_GAUSSIANPROJECTIONUTILS_CUH

#include <fvdb/detail/ops/gsplat/GaussianCameraAccessorCopy.cuh>
#include <fvdb/detail/ops/gsplat/GaussianCameraMatrixUtils.cuh>
#include <fvdb/detail/ops/gsplat/GaussianUtils.cuh>
#include <fvdb/detail/utils/AccessorHelpers.cuh>

#include <nanovdb/math/Math.h>

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

template <typename T> struct PerspectiveCameraOp {
    using Mat3 = nanovdb::math::Mat3<T>;
    using Vec3 = nanovdb::math::Vec3<T>;

    fvdb::TorchRAcc32<T, 3> projectionMatricesAcc; // [C,3,3]
    fvdb::TorchRAcc32<T, 3> worldToCamMatricesAcc; // [C,4,4]
    int32_t imageWidth = 0;
    int32_t imageHeight = 0;
    T nearPlane = T(-1e10);
    T farPlane = T(1e10);

    Mat3 *__restrict__ projectionMatsShared = nullptr;    // [C,3,3], optional
    Mat3 *__restrict__ worldToCamRotMatsShared = nullptr; // [C,3,3], optional
    Vec3 *__restrict__ worldToCamTranslationShared = nullptr; // [C,3], optional

    inline __device__ void
    bindSharedMemory(void *sharedMemory, const int64_t C) {
        projectionMatsShared = reinterpret_cast<Mat3 *>(sharedMemory);
        worldToCamRotMatsShared = projectionMatsShared + C;
        worldToCamTranslationShared = reinterpret_cast<Vec3 *>(worldToCamRotMatsShared + C);
    }

    inline __device__ void
    loadCameraStateToShared(const int64_t C) const {
        if (projectionMatsShared != nullptr) {
            copyMat3Accessor<T>(C, projectionMatsShared, projectionMatricesAcc);
            copyMat3Accessor<T>(C, worldToCamRotMatsShared, worldToCamMatricesAcc);
            copyWorldToCamTranslation<T>(C, worldToCamTranslationShared, worldToCamMatricesAcc);
        }
    }

    inline __device__ bool
    isDepthVisible(const T depth) const {
        return depth >= nearPlane && depth <= farPlane;
    }

    inline __device__ std::tuple<Mat3, Vec3>
    worldToCamRt(const int64_t cid) const {
        if (worldToCamRotMatsShared != nullptr) {
            return std::make_tuple(worldToCamRotMatsShared[cid], worldToCamTranslationShared[cid]);
        }
        const auto W = worldToCamMatricesAcc[cid];
        return std::make_tuple(Mat3(W[0][0], W[0][1], W[0][2], W[1][0], W[1][1], W[1][2], W[2][0], W[2][1], W[2][2]),
                               Vec3(W[0][3], W[1][3], W[2][3]));
    }

    inline __device__ std::tuple<nanovdb::math::Mat2<T>, nanovdb::math::Vec2<T>>
    project(const int64_t cid,
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
    vjp(const int64_t cid,
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

        const T rz3       = rz2 * rz;
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

template <typename T> struct OrthographicCameraOp {
    using Mat3 = nanovdb::math::Mat3<T>;
    using Vec3 = nanovdb::math::Vec3<T>;

    fvdb::TorchRAcc32<T, 3> projectionMatricesAcc; // [C,3,3]
    fvdb::TorchRAcc32<T, 3> worldToCamMatricesAcc; // [C,4,4]
    int32_t imageWidth = 0;
    int32_t imageHeight = 0;
    T nearPlane = T(-1e10);
    T farPlane = T(1e10);

    Mat3 *__restrict__ projectionMatsShared = nullptr;    // [C,3,3], optional
    Mat3 *__restrict__ worldToCamRotMatsShared = nullptr; // [C,3,3], optional
    Vec3 *__restrict__ worldToCamTranslationShared = nullptr; // [C,3], optional

    inline __device__ void
    bindSharedMemory(void *sharedMemory, const int64_t C) {
        projectionMatsShared = reinterpret_cast<Mat3 *>(sharedMemory);
        worldToCamRotMatsShared = projectionMatsShared + C;
        worldToCamTranslationShared = reinterpret_cast<Vec3 *>(worldToCamRotMatsShared + C);
    }

    inline __device__ void
    loadCameraStateToShared(const int64_t C) const {
        if (projectionMatsShared != nullptr) {
            copyMat3Accessor<T>(C, projectionMatsShared, projectionMatricesAcc);
            copyMat3Accessor<T>(C, worldToCamRotMatsShared, worldToCamMatricesAcc);
            copyWorldToCamTranslation<T>(C, worldToCamTranslationShared, worldToCamMatricesAcc);
        }
    }

    inline __device__ bool
    isDepthVisible(const T depth) const {
        return depth >= nearPlane && depth <= farPlane;
    }

    inline __device__ std::tuple<Mat3, Vec3>
    worldToCamRt(const int64_t cid) const {
        if (worldToCamRotMatsShared != nullptr) {
            return std::make_tuple(worldToCamRotMatsShared[cid], worldToCamTranslationShared[cid]);
        }
        const auto W = worldToCamMatricesAcc[cid];
        return std::make_tuple(Mat3(W[0][0], W[0][1], W[0][2], W[1][0], W[1][1], W[1][2], W[2][0], W[2][1], W[2][2]),
                               Vec3(W[0][3], W[1][3], W[2][3]));
    }

    inline __device__ std::tuple<nanovdb::math::Mat2<T>, nanovdb::math::Vec2<T>>
    project(const int64_t cid,
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
    vjp(const int64_t cid,
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

} // namespace fvdb::detail::ops

#endif // FVDB_DETAIL_OPS_GSPLAT_GAUSSIANPROJECTIONUTILS_CUH
