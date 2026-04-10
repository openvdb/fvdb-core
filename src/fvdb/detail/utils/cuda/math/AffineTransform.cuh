// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_CUDA_MATH_AFFINETRANSFORM_CUH
#define FVDB_DETAIL_UTILS_CUDA_MATH_AFFINETRANSFORM_CUH

#include <nanovdb/math/Math.h>

#include <tuple>

namespace fvdb {
namespace detail {

/// Transform a point from world to camera coordinates: p_cam = R * p_world + t.
template <typename T>
inline __device__ nanovdb::math::Vec3<T>
transformPointWorldToCam(nanovdb::math::Mat3<T> const &worldToCamRotation,
                         nanovdb::math::Vec3<T> const &worldToCamTranslation,
                         nanovdb::math::Vec3<T> const &worldSpacePoint) {
    return worldToCamRotation * worldSpacePoint + worldToCamTranslation;
}

/// VJP for transformPointWorldToCam. Returns (dL/dR, dL/dt, dL/dp_world).
template <typename T>
inline __device__ std::tuple<nanovdb::math::Mat3<T>, nanovdb::math::Vec3<T>, nanovdb::math::Vec3<T>>
transformPointWorldToCamVectorJacobianProduct(const nanovdb::math::Mat3<T> &worldToCamRotation,
                                              const nanovdb::math::Vec3<T> &worldToCamTranslation,
                                              const nanovdb::math::Vec3<T> &worldSpacePoint,
                                              const nanovdb::math::Vec3<T> &dLossDPointCamera) {
    return {dLossDPointCamera.outer(worldSpacePoint),
            dLossDPointCamera,
            worldToCamRotation.transpose() * dLossDPointCamera};
}

/// Transform a covariance matrix from world to camera: covar_cam = R * covar_world * R^T.
template <typename T>
inline __device__ nanovdb::math::Mat3<T>
transformCovarianceWorldToCam(nanovdb::math::Mat3<T> const &R,
                              nanovdb::math::Mat3<T> const &covar) {
    return R * covar * R.transpose();
}

/// VJP for transformCovarianceWorldToCam. Returns (dL/dR, dL/dcovar_world).
template <typename T>
inline __device__ std::tuple<nanovdb::math::Mat3<T>, nanovdb::math::Mat3<T>>
transformCovarianceWorldToCamVectorJacobianProduct(
    const nanovdb::math::Mat3<T> &R,
    const nanovdb::math::Mat3<T> &covar,
    const nanovdb::math::Mat3<T> &dLossDCovarCamera) {
    return {dLossDCovarCamera * R * covar.transpose() + dLossDCovarCamera.transpose() * R * covar,
            R.transpose() * dLossDCovarCamera * R};
}

/// VJP for matrix inverse: dL/dM = -M^{-1} * dL/dM^{-1} * M^{-1}.
template <typename T>
inline __device__ T
inverseVectorJacobianProduct(const T &MInv, const T &dLossDMInv) {
    return -MInv * dLossDMInv * MInv;
}

} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_UTILS_CUDA_MATH_AFFINETRANSFORM_CUH
