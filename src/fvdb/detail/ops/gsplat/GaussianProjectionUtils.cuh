// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_GAUSSIANPROJECTIONUTILS_CUH
#define FVDB_DETAIL_OPS_GSPLAT_GAUSSIANPROJECTIONUTILS_CUH

#include <fvdb/detail/ops/gsplat/GaussianCameraMatrixUtils.cuh>
#include <fvdb/detail/ops/gsplat/GaussianCameraIntrinsics.cuh>
#include <fvdb/detail/ops/gsplat/GaussianUtils.cuh>

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

} // namespace fvdb::detail::ops

#endif // FVDB_DETAIL_OPS_GSPLAT_GAUSSIANPROJECTIONUTILS_CUH
