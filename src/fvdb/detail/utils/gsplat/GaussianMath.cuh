// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_GSPLAT_GAUSSIANMATH_CUH
#define FVDB_DETAIL_UTILS_GSPLAT_GAUSSIANMATH_CUH

#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/cuda/math/Rotation.cuh>

#include <nanovdb/math/Math.h>

#include <cstdint>
#include <tuple>

namespace fvdb {
namespace detail {

/// @brief VJP for quaternion-and-scale to covariance matrix transformation.
///
/// The covariance is C = M * M^T where M = R * S, with R from the quaternion and
/// S = diag(scale). When ApplyLogScaleChainRule is true (default), returns
/// dL/d(log_scale) by multiplying by scale.
///
/// @tparam T Scalar type (float or double).
/// @tparam ApplyLogScaleChainRule When true, applies the chain rule for log-scale
///         parameterization so the returned scale gradient is dL/d(log_scale).
/// @param quat Unit quaternion [w,x,y,z] defining the rotation.
/// @param scale Per-axis scale factors [sx, sy, sz].
/// @param R 3x3 rotation matrix corresponding to @p quat (pre-computed).
/// @param dLossDCovar Upstream gradient w.r.t. the 3x3 covariance matrix.
/// @return Tuple of (dL/d_quat, dL/d_scale) -- or (dL/d_quat, dL/d_log_scale) when
///         ApplyLogScaleChainRule is true.
template <typename T, bool ApplyLogScaleChainRule = true>
inline __device__ std::tuple<nanovdb::math::Vec4<T>, nanovdb::math::Vec3<T>>
quaternionAndScaleToCovarianceVectorJacobianProduct(const nanovdb::math::Vec4<T> &quat,
                                                    const nanovdb::math::Vec3<T> &scale,
                                                    const nanovdb::math::Mat3<T> &R,
                                                    const nanovdb::math::Mat3<T> &dLossDCovar) {
    T w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    T sx = scale[0], sy = scale[1], sz = scale[2];

    const nanovdb::math::Mat3<T> S(sx, 0.f, 0.f, 0.f, sy, 0.f, 0.f, 0.f, sz);
    const nanovdb::math::Mat3<T> M = R * S;

    const nanovdb::math::Mat3<T> dLossDM = (dLossDCovar + dLossDCovar.transpose()) * M;
    const nanovdb::math::Mat3<T> dLossDR = dLossDM * S.transpose();

    const nanovdb::math::Vec4<T> &dLossDQuat =
        quaternionToRotationMatrixVectorJacobianProduct<T>(quat, dLossDR);

    const nanovdb::math::Vec3<T> dLossDScale(
        R[0][0] * dLossDM[0][0] + R[1][0] * dLossDM[1][0] + R[2][0] * dLossDM[2][0],
        R[0][1] * dLossDM[0][1] + R[1][1] * dLossDM[1][1] + R[2][1] * dLossDM[2][1],
        R[0][2] * dLossDM[0][2] + R[1][2] * dLossDM[1][2] + R[2][2] * dLossDM[2][2]);

    if constexpr (ApplyLogScaleChainRule) {
        return {
            dLossDQuat,
            nanovdb::math::Vec3<T>(sx * dLossDScale[0], sy * dLossDScale[1], sz * dLossDScale[2])};
    } else {
        return {dLossDQuat, dLossDScale};
    }
}

/// @brief Compute 3D covariance C = R * S * S^T * R^T from quaternion and scale.
///
/// @tparam T Scalar type (float or double).
/// @param quat Unit quaternion [w,x,y,z] defining the rotation.
/// @param scale Per-axis scale factors [sx, sy, sz].
/// @return 3x3 covariance matrix.
template <typename T>
inline __device__ nanovdb::math::Mat3<T>
quaternionAndScaleToCovariance(const nanovdb::math::Vec4<T> &quat,
                               const nanovdb::math::Vec3<T> &scale) {
    const nanovdb::math::Mat3<T> &R = quaternionToRotationMatrix<T>(quat);
    const nanovdb::math::Mat3<T> S(scale[0], 0.f, 0.f, 0.f, scale[1], 0.f, 0.f, 0.f, scale[2]);
    const nanovdb::math::Mat3<T> M = R * S;
    return M * M.transpose();
}

/// @brief Add isotropic blur to a 2D covariance and compute a compensation factor.
///
/// Adds eps2d to the diagonal of the 2x2 covariance. The compensation factor is
/// sqrt(det_original / det_blurred), which can be used for antialiasing.
///
/// @tparam T Scalar type (float or double).
/// @param eps2d Blur amount added to the diagonal.
/// @param[in,out] outCovar 2x2 covariance matrix; modified in-place with blur added.
/// @param[out] outCompensation Compensation factor sqrt(det_orig / det_blur).
/// @return Determinant of the blurred covariance.
template <typename T>
inline __device__ T
addBlur(const T eps2d, nanovdb::math::Mat2<T> &outCovar, T &outCompensation) {
    const T det_orig = outCovar[0][0] * outCovar[1][1] - outCovar[0][1] * outCovar[1][0];
    outCovar[0][0] += eps2d;
    outCovar[1][1] += eps2d;
    const T det_blur = outCovar[0][0] * outCovar[1][1] - outCovar[0][1] * outCovar[1][0];
    outCompensation  = sqrt(max(0.f, det_orig / det_blur));
    return det_blur;
}

/// @brief VJP for the addBlur operation.
///
/// @tparam T Scalar type (float or double).
/// @param eps2d Blur amount that was added to the diagonal (forward-pass input).
/// @param conic_blur Blurred 2x2 covariance from the forward pass.
/// @param compensation Compensation factor from the forward pass.
/// @param dLossDCompensation Upstream gradient w.r.t. the compensation factor.
/// @return Gradient w.r.t. the original (pre-blur) 2x2 covariance matrix.
template <typename T>
inline __device__ nanovdb::math::Mat2<T>
generateBlurVectorJacobianProduct(const T eps2d,
                                  const nanovdb::math::Mat2<T> &conic_blur,
                                  const T compensation,
                                  const T dLossDCompensation) {
    const T det_conic_blur =
        conic_blur[0][0] * conic_blur[1][1] - conic_blur[0][1] * conic_blur[1][0];
    const T v_sqr_comp         = dLossDCompensation * 0.5 / (compensation + 1e-6);
    const T one_minus_sqr_comp = 1 - compensation * compensation;
    return v_sqr_comp *
           nanovdb::math::Mat2<T>(one_minus_sqr_comp * conic_blur[0][0] - eps2d * det_conic_blur,
                                  one_minus_sqr_comp * conic_blur[0][1],
                                  one_minus_sqr_comp * conic_blur[1][0],
                                  one_minus_sqr_comp * conic_blur[1][1] - eps2d * det_conic_blur);
}

/// Transmittance threshold below which a pixel is considered fully opaque.
/// Matches the 3DGS convention (exclusive comparison: pixel is done when T <= threshold).
constexpr float kTransmittanceThreshold = 1e-4f;

/// Accessor type for per-tile pixel bitmasks (2D array of 64-bit words).
using tilePixelMaskAccessor = fvdb::TorchRAcc64<uint64_t, 2>;
/// Number of bits packed into each bitmask word.
static constexpr uint32_t sTileBitmaskBitsPerWord = 64;

/// @brief Compute the number of 64-bit words needed to represent a tile's pixel bitmask.
///
/// @param tileSideLength Side length of a tile in pixels (e.g. 16).
/// @return Number of uint64_t words needed (ceil(tileSideLength^2 / 64)).
inline uint32_t
numWordsPerTileBitmask(const uint32_t tileSideLength) {
    return (tileSideLength * tileSideLength + sTileBitmaskBitsPerWord - 1) /
           sTileBitmaskBitsPerWord;
}

inline __device__ uint32_t
bitmaskWordIndex(const uint32_t bitIndex) {
    return bitIndex / sTileBitmaskBitsPerWord;
}
inline __device__ uint32_t
bitmaskBitIndex(const uint32_t bitIndex) {
    return bitIndex % sTileBitmaskBitsPerWord;
}

inline __device__ bool
tilePixelActive(tilePixelMaskAccessor const &tilePixelMask,
                const uint32_t tileSideLength,
                const uint32_t tileId,
                const uint32_t iInTile,
                const uint32_t jInTile) {
    const uint32_t bitIndex = iInTile * tileSideLength + jInTile;
    return tilePixelMask[tileId][bitmaskWordIndex(bitIndex)] & (1ull << bitmaskBitIndex(bitIndex));
}

} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_UTILS_GSPLAT_GAUSSIANMATH_CUH
