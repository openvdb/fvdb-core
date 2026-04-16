// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_GSPLAT_GAUSSIANRASTERIZEFROMWORLD_CUH
#define FVDB_DETAIL_UTILS_GSPLAT_GAUSSIANRASTERIZEFROMWORLD_CUH

#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/cuda/BinSearch.cuh>
#include <fvdb/detail/utils/cuda/math/AffineTransform.cuh>
#include <fvdb/detail/utils/cuda/math/Rotation.cuh>
#include <fvdb/detail/utils/gsplat/GaussianCameras.cuh>
#include <fvdb/detail/utils/gsplat/GaussianMath.cuh>

#include <nanovdb/math/Math.h>
#include <nanovdb/math/Ray.h>

#include <cuda/std/tuple>

#include <cstdint>

namespace fvdb::detail::ops {

/// Opacity threshold used by 3DGS alpha compositing.
constexpr __device__ float kAlphaThreshold = 0.999f;

/// Common dense-tile rasterization arguments shared by from-world forward/backward kernels.
struct RasterizeFromWorldCommonArgs {
    using TileOffsetsAccessor     = fvdb::TorchRAcc64<int32_t, 3>;
    using TileGaussianIdsAccessor = fvdb::TorchRAcc64<int32_t, 1>;
    using Acc2f                   = fvdb::TorchRAcc64<float, 2>;
    using Acc3f                   = fvdb::TorchRAcc64<float, 3>;

    uint32_t imageWidth;
    uint32_t imageHeight;
    uint32_t imageOriginW;
    uint32_t imageOriginH;
    uint32_t tileSize;
    uint32_t numTilesW;
    uint32_t numTilesH;
    uint32_t numChannels;
    int32_t totalIntersections;

    TileOffsetsAccessor tileOffsets;         // [C, TH, TW]
    TileGaussianIdsAccessor tileGaussianIds; // [n_isects]
    const float *backgrounds;                // [C, D] or nullptr
    const bool *masks;                       // [C, TH, TW] or nullptr

    // Common from-world inputs shared by forward/backward kernels.
    Acc2f means;     // [N,3]
    Acc2f quats;     // [N,4]
    Acc2f logScales; // [N,3]
    Acc3f features;  // [C,N,D]
    Acc2f opacities; // [C,N]

    inline __device__ void
    denseCoordinates(uint32_t &cameraId,
                     uint32_t &tileRow,
                     uint32_t &tileCol,
                     uint32_t &row,
                     uint32_t &col) const {
        const uint32_t globalLinearBlockIdx = blockIdx.x;
        const uint32_t tileExtentW          = numTilesW;
        const uint32_t tileExtentH          = numTilesH;

        cameraId = globalLinearBlockIdx / (tileExtentH * tileExtentW);
        tileRow  = (globalLinearBlockIdx / tileExtentW) % tileExtentH;
        tileCol  = globalLinearBlockIdx % tileExtentW;

        row = tileRow * tileSize + threadIdx.y;
        col = tileCol * tileSize + threadIdx.x;
    }

    inline __device__ uint32_t
    tileId(const uint32_t cameraId, const uint32_t tileRow, const uint32_t tileCol) const {
        return cameraId * numTilesH * numTilesW + tileRow * numTilesW + tileCol;
    }

    inline __device__ bool
    tileMasked(const uint32_t cameraId, const uint32_t tileRow, const uint32_t tileCol) const {
        return (masks != nullptr) && (!masks[tileId(cameraId, tileRow, tileCol)]);
    }

    inline __device__ cuda::std::tuple<int32_t, int32_t>
    tileGaussianRange(const uint32_t cameraId,
                      const uint32_t tileRow,
                      const uint32_t tileCol) const {
        const uint32_t numCameras            = tileOffsets.size(0);
        const int32_t firstGaussianIdInBlock = tileOffsets[cameraId][tileRow][tileCol];
        auto [nextTileRow, nextTileCol]      = (tileCol < numTilesW - 1)
                                                   ? cuda::std::make_tuple(tileRow, tileCol + 1)
                                                   : cuda::std::make_tuple(tileRow + 1, 0u);
        const int32_t lastGaussianIdInBlock =
            ((cameraId == numCameras - 1) && (nextTileRow == numTilesH))
                ? totalIntersections
                : tileOffsets[cameraId][nextTileRow][nextTileCol];
        return {firstGaussianIdInBlock, lastGaussianIdInBlock};
    }

    inline __device__ float
    backgroundValue(const uint32_t cameraId, const uint32_t channelId) const {
        return (backgrounds != nullptr) ? backgrounds[cameraId * numChannels + channelId] : 0.0f;
    }
};

/// @brief Compute the inverse-scale-rotation-transpose matrix S^{-1} R^T.
///
/// This is used in the world-space rasterizer to transform directions from camera
/// space back to the Gaussian's local coordinate frame. "Iscl" is short for
/// "inverse-scale".
///
/// @tparam T Scalar type (float or double).
/// @param quat_wxyz Unit quaternion [w,x,y,z] defining the Gaussian's rotation.
/// @param scale Per-axis scale factors [sx, sy, sz] of the Gaussian.
/// @return 3x3 matrix S^{-1} R^T.
template <typename T>
inline __device__ nanovdb::math::Mat3<T>
computeIsclRot(const nanovdb::math::Vec4<T> &quat_wxyz, const nanovdb::math::Vec3<T> &scale) {
    const nanovdb::math::Mat3<T> R = quaternionToRotationMatrix<T>(quat_wxyz);
    const nanovdb::math::Mat3<T> S_inv(
        T(1) / scale[0], T(0), T(0), T(0), T(1) / scale[1], T(0), T(0), T(0), T(1) / scale[2]);
    return S_inv * R.transpose();
}

/// @brief VJP for computeIsclRot (inverse-scale-rotation-transpose = S^{-1} R^T).
///
/// Computes gradients of the loss w.r.t. the quaternion and log-scale, given the
/// upstream gradient w.r.t. the S^{-1} R^T matrix.
///
/// @tparam T Scalar type (float or double).
/// @param quat_wxyz Unit quaternion [w,x,y,z] (forward-pass input).
/// @param scale Per-axis scale factors [sx, sy, sz] (forward-pass input).
/// @param dLossDIsclRot Upstream gradient w.r.t. the S^{-1} R^T matrix.
/// @param[out] dLossDQuat Gradient w.r.t. the quaternion.
/// @param[out] dLossDLogScale Gradient w.r.t. log-scale (includes the chain rule for
///             the log parameterization).
template <typename T>
inline __device__ void
isclRotVectorJacobianProduct(const nanovdb::math::Vec4<T> &quat_wxyz,
                             const nanovdb::math::Vec3<T> &scale,
                             const nanovdb::math::Mat3<T> &dLossDIsclRot,
                             nanovdb::math::Vec4<T> &dLossDQuat,
                             nanovdb::math::Vec3<T> &dLossDLogScale) {
    // TODO(fvdb): Consider returning {dLossDQuat, dLossDLogScale} to match other VJP helpers in
    // this module (e.g. `transformCovarianceWorldToCamVectorJacobianProduct`), rather than using
    // output reference parameters.
    // iscl_rot = S_inv * R^T, with S_inv = diag(1/scale)
    const nanovdb::math::Mat3<T> R = quaternionToRotationMatrix<T>(quat_wxyz);
    const nanovdb::math::Vec3<T> invScale(T(1) / scale[0], T(1) / scale[1], T(1) / scale[2]);

    // For D = A * B, dA = G * B^T, dB = A^T * G.
    // Here A = S_inv (diag), B = R^T.
    const nanovdb::math::Mat3<T> dA = dLossDIsclRot * R; // since (R^T)^T = R
    const nanovdb::math::Mat3<T> A(
        invScale[0], T(0), T(0), T(0), invScale[1], T(0), T(0), T(0), invScale[2]);
    const nanovdb::math::Mat3<T> dB = A * dLossDIsclRot;
    const nanovdb::math::Mat3<T> dR = dB.transpose(); // B = R^T

    dLossDQuat = quaternionToRotationMatrixVectorJacobianProduct<T>(quat_wxyz, dR);

    // Diagonal of dA gives gradients w.r.t invScale.
    const nanovdb::math::Vec3<T> dInvScale(dA[0][0], dA[1][1], dA[2][2]);
    // invScale = 1/scale => dlogScale = dscale * scale = -dInvScale / scale
    dLossDLogScale = nanovdb::math::Vec3<T>(
        -dInvScale[0] * invScale[0], -dInvScale[1] * invScale[1], -dInvScale[2] * invScale[2]);
}

} // namespace fvdb::detail::ops

#endif // FVDB_DETAIL_UTILS_GSPLAT_GAUSSIANRASTERIZEFROMWORLD_CUH
