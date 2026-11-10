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
#include <fvdb/detail/utils/gsplat/GaussianTileRange.cuh>

#include <nanovdb/math/Math.h>
#include <nanovdb/math/Ray.h>

#include <cuda/std/tuple>

#include <cstdint>
#include <type_traits>

namespace fvdb::detail::ops {

/// Opacity threshold used by 3DGS alpha compositing.
constexpr __device__ float kAlphaThreshold = 0.99f;

/// Number of feature channels each rasterization pass holds in registers.
constexpr uint32_t kRasterChannelChunk = 64;

/// Largest channel remainder that joins the last full chunk instead of taking its own pass.
constexpr uint32_t kRasterMaxFoldedRemainder = 8;

/// Number of channels a per-channel loop unrolls when it keeps no per-channel state.
template <uint32_t NUM_CHANNELS>
constexpr uint32_t kRasterChannelUnroll =
    NUM_CHANNELS < kRasterChannelChunk ? NUM_CHANNELS : kRasterChannelChunk;

/// Chunk schedule for NUM_CHANNELS: NUM_FULL passes of kRasterChannelChunk channels, then one
/// pass of TAIL channels (none if TAIL is 0). A remainder of at most kRasterMaxFoldedRemainder
/// channels widens the last full chunk; a larger one gets its own pass.
template <uint32_t NUM_CHANNELS> struct RasterChannelSchedule {
    static constexpr uint32_t CHUNK = kRasterChannelChunk;
    static constexpr uint32_t REM   = NUM_CHANNELS % CHUNK;
    static constexpr bool FOLD_REM =
        (NUM_CHANNELS > CHUNK) && (REM != 0) && (REM <= kRasterMaxFoldedRemainder);
    static constexpr uint32_t NUM_FULL = NUM_CHANNELS / CHUNK - (FOLD_REM ? 1 : 0);
    static constexpr uint32_t TAIL     = FOLD_REM ? CHUNK + REM : REM;
};

// The compiled channel counts only have remainders of 0 or 1 above kRasterChannelChunk, so the
// separate tail pass after full chunks (e.g. 100 = 64 + 36) is never instantiated. These checks
// pin the schedule for that case and the folded ones.
static_assert(RasterChannelSchedule<33>::NUM_FULL == 0 && RasterChannelSchedule<33>::TAIL == 33);
static_assert(RasterChannelSchedule<64>::NUM_FULL == 1 && RasterChannelSchedule<64>::TAIL == 0);
static_assert(RasterChannelSchedule<65>::NUM_FULL == 0 && RasterChannelSchedule<65>::TAIL == 65);
static_assert(RasterChannelSchedule<100>::NUM_FULL == 1 && RasterChannelSchedule<100>::TAIL == 36);
static_assert(RasterChannelSchedule<129>::NUM_FULL == 1 && RasterChannelSchedule<129>::TAIL == 65);
static_assert(RasterChannelSchedule<192>::NUM_FULL == 3 && RasterChannelSchedule<192>::TAIL == 0);

/// Split NUM_CHANNELS into register-resident chunks per RasterChannelSchedule and call
/// `fn(std::integral_constant<uint32_t, WIDTH>{}, channelStart)` once per chunk, in order.
///
/// Every chunk has a compile-time width, so per-channel arrays stay in registers and need no
/// tail predicates.
template <uint32_t NUM_CHANNELS, typename Fn>
inline __device__ void
forEachRasterChannelChunk(Fn &&fn) {
    using Schedule              = RasterChannelSchedule<NUM_CHANNELS>;
    constexpr uint32_t CHUNK    = Schedule::CHUNK;
    constexpr uint32_t NUM_FULL = Schedule::NUM_FULL;
    constexpr uint32_t TAIL     = Schedule::TAIL;

    // Guarded so a zero trip count never emits `chunk < 0`, which older nvcc rejects (#186-D).
    if constexpr (NUM_FULL != 0) {
#pragma unroll 1
        for (uint32_t chunk = 0; chunk < NUM_FULL; ++chunk) {
            fn(std::integral_constant<uint32_t, CHUNK>{}, chunk * CHUNK);
        }
    }
    if constexpr (TAIL != 0) {
        fn(std::integral_constant<uint32_t, TAIL>{}, NUM_FULL * CHUNK);
    }
}

/// Common dense-tile rasterization arguments shared by from-world forward/backward kernels.
struct RasterizeFromWorldCommonArgs {
    using TileOffsetsAccessor     = fvdb::TorchRAcc64<int64_t, 3>;
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
    int64_t totalIntersections;

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
                     uint32_t &col,
                     const uint32_t blockOffset = 0) const {
        const uint32_t globalLinearBlockIdx = blockOffset + blockIdx.x;
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

    inline __device__ cuda::std::tuple<int64_t, int64_t>
    tileGaussianRange(const uint32_t cameraId,
                      const uint32_t tileRow,
                      const uint32_t tileCol) const {
        return fvdb::detail::ops::tileGaussianRange(tileOffsets,
                                                    totalIntersections,
                                                    tileOffsets.size(0),
                                                    numTilesH,
                                                    numTilesW,
                                                    cameraId,
                                                    tileRow,
                                                    tileCol);
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
