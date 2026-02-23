// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_GAUSSIANRASTERIZEFROMWORLD_CUH
#define FVDB_DETAIL_OPS_GSPLAT_GAUSSIANRASTERIZEFROMWORLD_CUH

#include <fvdb/detail/ops/gsplat/GaussianCameraModels.h>
#include <fvdb/detail/ops/gsplat/GaussianCameraIntrinsics.cuh>
#include <fvdb/detail/ops/gsplat/GaussianOpenCVDistortion.cuh>
#include <fvdb/detail/ops/gsplat/GaussianRigidTransform.cuh>
#include <fvdb/detail/ops/gsplat/GaussianRollingShutter.cuh>
#include <fvdb/detail/ops/gsplat/GaussianUtils.cuh>
#include <fvdb/detail/utils/AccessorHelpers.cuh>

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

    uint32_t numCameras;
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

    inline __device__ uint32_t
    pixelId(const uint32_t row, const uint32_t col) const {
        return row * imageWidth + col;
    }

    inline __device__ uint32_t
    outputPixelBase(const uint32_t cameraId, const uint32_t pixId) const {
        return cameraId * imageHeight * imageWidth + pixId;
    }

    inline __device__ uint32_t
    outputFeatureBase(const uint32_t cameraId, const uint32_t pixId) const {
        return outputPixelBase(cameraId, pixId) * numChannels;
    }

    inline __device__ float
    backgroundValue(const uint32_t cameraId, const uint32_t channelId) const {
        return (backgrounds != nullptr) ? backgrounds[cameraId * numChannels + channelId] : 0.0f;
    }
};

template <typename T>
inline __device__ nanovdb::math::Vec3<T>
normalizeSafe(const nanovdb::math::Vec3<T> &v) {
    const T n2 = v.dot(v);
    if (n2 > T(0)) {
        return v * (T(1) / sqrt(n2));
    }
    return nanovdb::math::Vec3<T>(T(0), T(0), T(0));
}

/// Vector-Jacobian product for y = normalizeSafe(x).
template <typename T>
inline __device__ nanovdb::math::Vec3<T>
normalizeSafeVJP(const nanovdb::math::Vec3<T> &x, const nanovdb::math::Vec3<T> &v_y) {
    const T n2 = x.dot(x);
    if (!(n2 > T(0))) {
        return nanovdb::math::Vec3<T>(T(0), T(0), T(0));
    }
    const T n    = sqrt(n2);
    const T invn = T(1) / n;
    // v_x = (I/n - x x^T / n^3) v_y
    const T invn3 = invn * invn * invn;
    const T xdotv = x.dot(v_y);
    return v_y * invn - x * (xdotv * invn3);
}

template <typename T>
inline __device__ nanovdb::math::Ray<T>
pixelToWorldRay(const uint32_t row,
                const uint32_t col,
                const uint32_t imageWidth,
                const uint32_t imageHeight,
                const uint32_t imageOriginW,
                const uint32_t imageOriginH,
                const nanovdb::math::Mat3<T> &R_wc_start,
                const nanovdb::math::Vec3<T> &t_wc_start,
                const nanovdb::math::Mat3<T> &R_wc_end,
                const nanovdb::math::Vec3<T> &t_wc_end,
                const nanovdb::math::Mat3<T> &K,
                const T *distCoeffs,
                const int64_t numDistCoeffs,
                const RollingShutterType rollingShutterType,
                const CameraModel cameraModel) {
    // Pixel center in crop coordinates.
    const T px = T(col) + T(imageOriginW) + T(0.5);
    const T py = T(row) + T(imageOriginH) + T(0.5);

    // Rolling shutter time based on pixel location.
    const T u = rollingShutterTimeFromPixel<T>(rollingShutterType, px, py, imageWidth, imageHeight);

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

    // Invert rigid transform to get camera->world.
    const nanovdb::math::Mat3<T> R_cw = R_wc.transpose();

    const CameraIntrinsics<T> intrinsics(K);
    const nanovdb::math::Vec2<T> p_distorted((px - intrinsics.cx) / intrinsics.fx,
                                             (py - intrinsics.cy) / intrinsics.fy);
    const nanovdb::math::Vec2<T> p =
        undistortOpenCVPackedFixedPoint(cameraModel, p_distorted, distCoeffs, numDistCoeffs);

    // Note: `nanovdb::math::Ray` is the standard ray type used elsewhere in FVDB.
    // We store world-space `eye` and `dir` (normalized) and leave [t0,t1] at default values.
    if (cameraModel == CameraModel::ORTHOGRAPHIC) {
        // Parallel rays; origin varies with pixel.
        const nanovdb::math::Vec3<T> o_cam(p[0], p[1], T(0));
        const nanovdb::math::Vec3<T> d_cam(T(0), T(0), T(1));
        // p_world = R_cw * (p_cam - t_wc)
        const nanovdb::math::Vec3<T> origin_w = R_cw * (o_cam - t_wc);
        nanovdb::math::Vec3<T> dir_w          = normalizeSafe(R_cw * d_cam);
        return nanovdb::math::Ray<T>(origin_w, dir_w);
    }

    // Perspective (pinhole / OpenCV distorted pinhole): origin at camera center.
    const nanovdb::math::Vec3<T> d_cam = normalizeSafe(nanovdb::math::Vec3<T>(p[0], p[1], T(1)));
    const nanovdb::math::Vec3<T> origin_w =
        R_cw * (nanovdb::math::Vec3<T>(T(0), T(0), T(0)) - t_wc);
    nanovdb::math::Vec3<T> dir_w = normalizeSafe(R_cw * d_cam);
    return nanovdb::math::Ray<T>(origin_w, dir_w);
}

/// Load quaternion in [w,x,y,z] order.
template <typename T>
inline __device__ nanovdb::math::Vec4<T>
quatLoadWxyz(const T *q) {
    return nanovdb::math::Vec4<T>(q[0], q[1], q[2], q[3]);
}

/// Build S^{-1} R^T from quaternion + scale.
template <typename T>
inline __device__ nanovdb::math::Mat3<T>
computeIsclRot(const nanovdb::math::Vec4<T> &quat_wxyz, const nanovdb::math::Vec3<T> &scale) {
    const nanovdb::math::Mat3<T> R = quaternionToRotationMatrix<T>(quat_wxyz);
    const nanovdb::math::Mat3<T> S_inv(
        T(1) / scale[0], T(0), T(0), T(0), T(1) / scale[1], T(0), T(0), T(0), T(1) / scale[2]);
    return S_inv * R.transpose();
}

/// Vector-Jacobian product for isclRot = S^{-1} R^T.
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

#endif // FVDB_DETAIL_OPS_GSPLAT_GAUSSIANRASTERIZEFROMWORLD_CUH
