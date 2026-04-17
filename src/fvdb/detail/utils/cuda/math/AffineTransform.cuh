// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_CUDA_MATH_AFFINETRANSFORM_CUH
#define FVDB_DETAIL_UTILS_CUDA_MATH_AFFINETRANSFORM_CUH

#include <nanovdb/math/Math.h>

#include <tuple>

namespace fvdb {
namespace detail {

/// @brief Transform a point from world to camera coordinates: p_cam = R * p_world + t.
///
/// @tparam T Scalar type (float or double).
/// @param worldToCamRotation 3x3 rotation matrix of the world-to-camera transform.
/// @param worldToCamTranslation Translation vector of the world-to-camera transform.
/// @param worldSpacePoint Point in world coordinates.
/// @return Point in camera coordinates.
template <typename T>
inline __device__ nanovdb::math::Vec3<T>
transformPointWorldToCam(nanovdb::math::Mat3<T> const &worldToCamRotation,
                         nanovdb::math::Vec3<T> const &worldToCamTranslation,
                         nanovdb::math::Vec3<T> const &worldSpacePoint) {
    return worldToCamRotation * worldSpacePoint + worldToCamTranslation;
}

/// @brief VJP for transformPointWorldToCam.
///
/// @tparam T Scalar type (float or double).
/// @param worldToCamRotation 3x3 rotation matrix of the world-to-camera transform.
/// @param worldToCamTranslation Translation vector of the world-to-camera transform.
/// @param worldSpacePoint Point in world coordinates (forward-pass input).
/// @param dLossDPointCamera Upstream gradient w.r.t. the camera-space point.
/// @return Tuple of (dL/dR, dL/dt, dL/dp_world).
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

/// @brief Transform a covariance matrix from world to camera frame: covar_cam = R * covar_world *
/// R^T.
///
/// @tparam T Scalar type (float or double).
/// @param R 3x3 rotation matrix of the world-to-camera transform.
/// @param covar 3x3 covariance matrix in world coordinates.
/// @return 3x3 covariance matrix in camera coordinates.
template <typename T>
inline __device__ nanovdb::math::Mat3<T>
transformCovarianceWorldToCam(nanovdb::math::Mat3<T> const &R,
                              nanovdb::math::Mat3<T> const &covar) {
    return R * covar * R.transpose();
}

/// @brief VJP for transformCovarianceWorldToCam.
///
/// @tparam T Scalar type (float or double).
/// @param R 3x3 rotation matrix of the world-to-camera transform (forward-pass input).
/// @param covar 3x3 covariance matrix in world coordinates (forward-pass input).
/// @param dLossDCovarCamera Upstream gradient w.r.t. the camera-space covariance.
/// @return Tuple of (dL/dR, dL/dcovar_world).
template <typename T>
inline __device__ std::tuple<nanovdb::math::Mat3<T>, nanovdb::math::Mat3<T>>
transformCovarianceWorldToCamVectorJacobianProduct(
    const nanovdb::math::Mat3<T> &R,
    const nanovdb::math::Mat3<T> &covar,
    const nanovdb::math::Mat3<T> &dLossDCovarCamera) {
    return {dLossDCovarCamera * R * covar.transpose() + dLossDCovarCamera.transpose() * R * covar,
            R.transpose() * dLossDCovarCamera * R};
}

/// @brief VJP for matrix inverse: dL/dM = -M^{-T} * dL/dM^{-1} * M^{-T}.
///
/// @tparam T Matrix type supporting operator* (e.g. Mat2, Mat3).
/// @param MInv The inverse of the original matrix (forward-pass output).
/// @param dLossDMInv Upstream gradient w.r.t. M^{-1}.
/// @return Gradient w.r.t. the original matrix M.
template <typename T>
inline __device__ T
inverseVectorJacobianProduct(const T &MInv, const T &dLossDMInv) {
    return -MInv * dLossDMInv * MInv;
}

} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_UTILS_CUDA_MATH_AFFINETRANSFORM_CUH
