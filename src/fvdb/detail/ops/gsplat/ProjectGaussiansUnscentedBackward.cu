// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/gsplat/ProjectGaussiansUnscentedBackward.h>
#include <fvdb/detail/utils/Nvtx.h>
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/cuda/GridDim.h>
#include <fvdb/detail/utils/cuda/Utils.cuh>
#include <fvdb/detail/utils/cuda/math/AffineTransform.cuh>
#include <fvdb/detail/utils/cuda/math/Rotation.cuh>
#include <fvdb/detail/utils/gsplat/GaussianCameras.cuh>
#include <fvdb/detail/utils/gsplat/GaussianMath.cuh>

#include <nanovdb/math/Math.h>

#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAGuard.h>

namespace fvdb {
namespace detail {
namespace ops {

namespace {

/// @brief Per-Gaussian, per-camera backward kernel for the unscented-transform projection.
///
/// See ProjectGaussiansUnscentedBackward.h for the derivation, its scope limitations, and the
/// numerical validation this was checked against before being written here. Recomputes the
/// forward's sigma points / projections / 2D covariance reconstruction locally (cheap: 7 points,
/// a handful of scalar ops), matching the pattern `projectionBackwardKernel` in
/// ProjectGaussiansAnalyticBackward.cu already uses for the analytic path's camera-space
/// mean/covariance, rather than requiring them to be saved from the forward pass.
template <typename T, typename Camera>
__global__ __launch_bounds__(DEFAULT_BLOCK_DIM) void
projectionUnscentedBackwardKernel(const int64_t count,
                                  const uint32_t C,
                                  const uint32_t N,
                                  const T *__restrict__ means,     // [N, 3]
                                  const T *__restrict__ quats,     // [N, 4]
                                  const T *__restrict__ logScales, // [N, 3]
                                  Camera camera,
                                  const T eps2d,
                                  const UTParams utParams,
                                  // fwd outputs
                                  const int32_t *__restrict__ radii,  // [C, N, 2]
                                  const T *__restrict__ conics,       // [C, N, 3]
                                  // grad outputs
                                  const T *__restrict__ dLossDMeans2d, // [C, N, 2]
                                  const T *__restrict__ dLossDDepths,  // [C, N]
                                  const T *__restrict__ dLossDConics,  // [C, N, 3]
                                  // grad inputs
                                  T *__restrict__ outDLossDMeans,     // [N, 3]
                                  T *__restrict__ outDLossDQuats,     // [N, 4]
                                  T *__restrict__ outDLossDLogScales  // [N, 3]
) {
    using Vec2 = nanovdb::math::Vec2<T>;
    using Vec3 = nanovdb::math::Vec3<T>;
    using Vec4 = nanovdb::math::Vec4<T>;
    using Mat2 = nanovdb::math::Mat2<T>;
    using Mat3 = nanovdb::math::Mat3<T>;

    const uint32_t cId = blockIdx.y;
    alignas(Mat3) extern __shared__ char sharedMemory[];
    camera.loadSharedMemory(cId, sharedMemory);
    __syncthreads();

    const uint32_t gId = blockIdx.x * blockDim.x + threadIdx.x;
    if (gId >= count) {
        return;
    }
    const int64_t idx = int64_t(cId) * N + gId;

    const int32_t radiusX = radii[idx * 2 + 0];
    const int32_t radiusY = radii[idx * 2 + 1];
    if (radiusX <= 0 || radiusY <= 0) {
        return;
    }

    // ---------------------------------------------------------------------------------------
    // Recompute the forward pass locally (means, quat, scale -> 7 sigma points -> projected
    // pixels -> mean2d/covar2d -> blur -> PSD-enforce). Mirrors
    // `_gaussian_unscented_torch.py::project_gaussians_unscented_torch` exactly; that Python
    // implementation is itself validated to sub-pixel agreement against this CUDA forward.
    // ---------------------------------------------------------------------------------------
    const Vec3 meanWorld(means[gId * 3 + 0], means[gId * 3 + 1], means[gId * 3 + 2]);
    const Vec4 quat(
        quats[gId * 4 + 0], quats[gId * 4 + 1], quats[gId * 4 + 2], quats[gId * 4 + 3]);
    const Vec3 scale(::cuda::std::exp(logScales[gId * 3 + 0]),
                     ::cuda::std::exp(logScales[gId * 3 + 1]),
                     ::cuda::std::exp(logScales[gId * 3 + 2]));

    const T alpha = T(utParams.alpha), beta = T(utParams.beta), kappa = T(utParams.kappa);
    constexpr int D    = 3;
    const T lambda      = alpha * alpha * (T(D) + kappa) - T(D);
    const T denom        = T(D) + lambda;
    const T gamma        = ::cuda::std::sqrt(nanovdb::math::Max(T(0), denom));
    T wMean[7], wCov[7];
    wMean[0] = lambda / denom;
    wCov[0]  = lambda / denom + (T(1) - alpha * alpha + beta);
    const T wi = T(1) / (T(2) * denom);
#pragma unroll
    for (int i = 1; i < 7; ++i) {
        wMean[i] = wi;
        wCov[i]  = wi;
    }

    const Mat3 R = quaternionToRotationMatrix<T>(quat);

    Vec3 sigma[7];
    sigma[0] = meanWorld;
    Vec3 offset[3]; // offset[axis] = gamma * scale[axis] * R[:, axis]
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        const Vec3 col(R[0][axis], R[1][axis], R[2][axis]);
        offset[axis]        = (gamma * scale[axis]) * col;
        sigma[1 + axis]      = meanWorld + offset[axis];
        sigma[4 + axis]      = meanWorld - offset[axis];
    }

    const auto [Rwc, twc] = camera.worldToCamStartRt(cId); // RollingShutterType::NONE scope only.

    Vec3 pCam[7];
    Vec2 pNormRaw[7], pNorm[7];
    constexpr T kZNear = T(1e-6);
    constexpr T kNormClamp = T(20);
#pragma unroll
    for (int k = 0; k < 7; ++k) {
        pCam[k]    = Rwc * sigma[k] + twc;
        const T z  = nanovdb::math::Max(pCam[k][2], kZNear);
        pNormRaw[k] = Vec2(pCam[k][0] / z, pCam[k][1] / z);
        pNorm[k]    = Vec2(nanovdb::math::Min(kNormClamp, nanovdb::math::Max(-kNormClamp, pNormRaw[k][0])),
                           nanovdb::math::Min(kNormClamp, nanovdb::math::Max(-kNormClamp, pNormRaw[k][1])));
    }

    const T *distCoeffs         = camera.distortionCoeffsPtr(cId);
    const int64_t numDistCoeffs = camera.numDistortionCoeffs();
    const DistortionModel model = camera.distortionModel();
    const Mat3 K                = camera.projectionMatrixPublic(cId);
    const T fx = K[0][0], fy = K[1][1], cx = K[0][2], cy = K[1][2];

    Vec2 pix[7];
#pragma unroll
    for (int k = 0; k < 7; ++k) {
        const Vec2 pd = Camera::applyOpenCVDistortion(model, pNorm[k], distCoeffs, numDistCoeffs);
        pix[k]        = Vec2(fx * pd[0] + cx, fy * pd[1] + cy);
    }

    Vec2 mean2d(T(0), T(0));
#pragma unroll
    for (int k = 0; k < 7; ++k) {
        mean2d[0] += wMean[k] * pix[k][0];
        mean2d[1] += wMean[k] * pix[k][1];
    }
    Vec2 diff[7];
    Mat2 covar2d(T(0), T(0), T(0), T(0));
#pragma unroll
    for (int k = 0; k < 7; ++k) {
        diff[k]      = pix[k] - mean2d;
        covar2d[0][0] += wCov[k] * diff[k][0] * diff[k][0];
        covar2d[0][1] += wCov[k] * diff[k][0] * diff[k][1];
        covar2d[1][1] += wCov[k] * diff[k][1] * diff[k][1];
    }
    covar2d[1][0] = covar2d[0][1];

    const T aBlur = covar2d[0][0] + eps2d;
    const T cBlur = covar2d[1][1] + eps2d;
    const T bBlur = covar2d[0][1];
    const T detBlur = aBlur * cBlur - bBlur * bBlur;

    const T halfTrace = T(0.5) * (aBlur + cBlur);
    const T discRaw    = halfTrace * halfTrace - detBlur;
    const bool discActive = discRaw > T(1e-12);
    const T disc  = nanovdb::math::Max(discRaw, T(1e-12));
    const T s     = ::cuda::std::sqrt(disc);
    const T v1    = halfTrace + s;
    const T v2    = halfTrace - s;
    const T v1c   = nanovdb::math::Max(v1, eps2d);
    const T v2c   = nanovdb::math::Max(v2, eps2d);

    // Eigenvector directions: constants w.r.t. this backward pass -- see the header comment for
    // why (the reconstructed matrix does not depend on the choice when v1==v2, and
    // differentiating the direction is a real divide-by-near-zero hazard for no benefit).
    T ux, uy;
    {
        constexpr T eigEps = T(1e-8);
        const bool useAForm = ::cuda::std::fabs(v1 - cBlur) > ::cuda::std::fabs(v1 - aBlur);
        T uxRaw = useAForm ? (v1 - cBlur) : bBlur;
        T uyRaw = useAForm ? bBlur : (v1 - aBlur);
        const T n = ::cuda::std::sqrt(uxRaw * uxRaw + uyRaw * uyRaw);
        if (n <= eigEps) {
            ux = (aBlur >= cBlur) ? T(1) : T(0);
            uy = T(1) - ux;
        } else {
            ux = uxRaw / n;
            uy = uyRaw / n;
        }
    }
    const T vx = -uy, vy = ux;

    // ---------------------------------------------------------------------------------------
    // Backward. Upstream gradients.
    // ---------------------------------------------------------------------------------------
    const Vec2 gMean2d(dLossDMeans2d[idx * 2 + 0], dLossDMeans2d[idx * 2 + 1]);
    const T gDepth = dLossDDepths[idx];

    // Step 1: inverse VJP dL/dCovarPSD, reusing the existing (already-tested) helpers: unpack
    // the packed conic gradient into a proper full-matrix argument, then the generic 2x2
    // inverse VJP.
    const Mat2 conic          = loadConicRowMajor3<T>(conics + idx * 3);
    const Mat2 dLossDConicMat = loadConicGradRowMajor3<T>(dLossDConics + idx * 3);
    const Mat2 dPSD           = inverseVectorJacobianProduct<Mat2>(conic, dLossDConicMat);
    const T dPSD00 = dPSD[0][0];
    const T dPSD11 = dPSD[1][1];
    // Packed convention (matches how a_blur/b_blur/c_blur are treated as 3 independent scalars
    // below): the off-diagonal gradient is the sum across both symmetric slots.
    const T dPSDOff = dPSD[0][1] + dPSD[1][0];

    // Step 2: project onto the (detached) eigenbasis. covar2d_psd_01 = v1c*ux*uy + v2c*vx*vy has
    // a single (ux*uy) coefficient in the packed-scalar parameterization -- no extra factor of 2.
    T dV1c = ux * ux * dPSD00 + ux * uy * dPSDOff + uy * uy * dPSD11;
    T dV2c = vx * vx * dPSD00 + vx * vy * dPSDOff + vy * vy * dPSD11;

    // Step 3: v1c = clamp(v1, eps2d); v2c = clamp(v2, eps2d).
    T dV1 = (v1 > eps2d) ? dV1c : T(0);
    T dV2 = (v2 > eps2d) ? dV2c : T(0);

    // Step 4-6: v1 = halfTrace + s; v2 = halfTrace - s; s = sqrt(disc);
    //           disc = clamp(halfTrace^2 - detBlur, 1e-12).
    T dHalfTrace = dV1 + dV2;
    T dS         = dV1 - dV2;
    T dDisc      = discActive ? (dS * T(0.5) / nanovdb::math::Max(s, T(1e-20))) : T(0);
    dHalfTrace += dDisc * T(2) * halfTrace;
    const T dDetBlurFromPSD = -dDisc;

    // Step 7-8: halfTrace = 0.5*(aBlur+cBlur); detBlur = aBlur*cBlur - bBlur^2.
    T dABlur = T(0.5) * dHalfTrace + dDetBlurFromPSD * cBlur;
    T dCBlur = T(0.5) * dHalfTrace + dDetBlurFromPSD * aBlur;
    T dBBlur = dDetBlurFromPSD * (T(-2) * bBlur);

    // Blur adds a constant (+eps2d) to the diagonal only: identity gradient onto the pre-blur
    // covariance. No antialiasing-compensation term here (this backward does not yet support
    // `calcCompensations`; `dLossDCompensations` is not exposed as an input -- see the header's
    // scope note, follows the same pattern as `generateBlurVectorJacobianProduct` if added).
    const T dA0 = dABlur, dB0 = dBBlur, dC0 = dCBlur;

    // Step 10: covar2d_ab = sum_k wCov[k] diff_k_a diff_k_b -> dL/d(diff_k) = 2*wCov[k]*(G@diff_k)
    // with G = [[dA0, 0.5*dB0], [0.5*dB0, dC0]] (undoing the packed-vs-full convention once more,
    // for the SAME reason as step 1).
    const T G00 = dA0, G01 = T(0.5) * dB0, G11 = dC0;

    Vec2 dPix[7];
    Vec2 sumDDiff(T(0), T(0));
    Vec2 dDiff[7];
#pragma unroll
    for (int k = 0; k < 7; ++k) {
        const T gx = G00 * diff[k][0] + G01 * diff[k][1];
        const T gy = G01 * diff[k][0] + G11 * diff[k][1];
        dDiff[k]   = Vec2(T(2) * wCov[k] * gx, T(2) * wCov[k] * gy);
        sumDDiff += dDiff[k];
    }
#pragma unroll
    for (int k = 0; k < 7; ++k) {
        dPix[k] = dDiff[k] + wMean[k] * (gMean2d - sumDDiff);
    }

    // Step 11-13: pix = (fx*xDist+cx, fy*yDist+cy); OpenCV distortion polynomial; normalized-
    // coordinate clamp.
    Vec3 dSigma[7];
#pragma unroll
    for (int k = 0; k < 7; ++k) {
        const T dXDist = dPix[k][0] * fx;
        const T dYDist = dPix[k][1] * fy;

        T dX = T(0), dY = T(0);
        Camera::applyOpenCVDistortionVJP(pNorm[k], distCoeffs, dXDist, dYDist, dX, dY);

        // Normalized-coordinate clamp VJP: zero gradient where the (pre-clamp) coordinate was
        // saturated. See the header/module docs for why this clamp exists at all (bounding an
        // otherwise-unbounded intermediate for Gaussians that have drifted near the camera
        // plane, which are always culled anyway but must not be allowed to produce inf/nan that
        // poisons the rest of the backward pass once summed into the loss).
        const bool xActive = pNormRaw[k][0] > -kNormClamp && pNormRaw[k][0] < kNormClamp;
        const bool yActive = pNormRaw[k][1] > -kNormClamp && pNormRaw[k][1] < kNormClamp;
        const T dXNorm = xActive ? dX : T(0);
        const T dYNorm = yActive ? dY : T(0);

        // Step 14: pNormRaw = pCam[:2] / zSafe.
        const T z = nanovdb::math::Max(pCam[k][2], kZNear);
        const T dPCamX = dXNorm / z;
        const T dPCamY = dYNorm / z;
        T dZSafe        = -(dXNorm * pCam[k][0] + dYNorm * pCam[k][1]) / (z * z);
        if (!(pCam[k][2] > kZNear)) {
            dZSafe = T(0);
        }
        T dPCamZ = dZSafe;
        if (k == 0) {
            // depth output == pCam[0][2] (the sigma-point-0/mean camera-space depth).
            dPCamZ += gDepth;
        }

        const Vec3 dPCam(dPCamX, dPCamY, dPCamZ);
        // Step 15: pCam = Rwc @ sigma + twc -> dSigma = Rwc^T @ dPCam. Reuse the existing
        // world-to-cam-point VJP helper (its dL/dR, dL/dt outputs are discarded: no camera-pose
        // gradient on this path yet, see the header's scope note).
        const auto [dR_unused, dt_unused, dSigma_k] =
            transformPointWorldToCamVectorJacobianProduct<T>(Rwc, twc, sigma[k], dPCam);
        (void)dR_unused;
        (void)dt_unused;
        dSigma[k] = dSigma_k;
    }

    // Step 16: sigma[0]=mean; sigma[1..3]=mean+offset[axis]; sigma[4..6]=mean-offset[axis].
    Vec3 dMean = dSigma[0];
    Vec3 dOffset[3];
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        dMean += dSigma[1 + axis];
        dMean += dSigma[4 + axis];
        dOffset[axis] = dSigma[1 + axis] - dSigma[4 + axis];
    }

    // Step 17: offset[axis] = gamma*scale[axis]*R[:,axis].
    Vec3 dScale(T(0), T(0), T(0));
    Mat3 dR(T(0), T(0), T(0), T(0), T(0), T(0), T(0), T(0), T(0));
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        const Vec3 col(R[0][axis], R[1][axis], R[2][axis]);
        dScale[axis] = gamma * dOffset[axis].dot(col);
        const Vec3 dCol = (gamma * scale[axis]) * dOffset[axis];
#pragma unroll
        for (int row = 0; row < 3; ++row) {
            dR[row][axis] += dCol[row];
        }
    }
    const Vec3 dLogScale(dScale[0] * scale[0], dScale[1] * scale[1], dScale[2] * scale[2]);

    // Step 18 (final): dR -> dQuat via the existing, already-tested quaternion VJP (which
    // handles the raw-quat normalization Jacobian internally, matching how R was computed above
    // from a normalized copy of `quat`).
    const Vec4 dQuat = quaternionToRotationMatrixVectorJacobianProduct<T>(quat, dR);

    // ---------------------------------------------------------------------------------------
    // Accumulate.
    // ---------------------------------------------------------------------------------------
    outDLossDMeans += gId * 3;
    gpuAtomicAdd(outDLossDMeans + 0, dMean[0]);
    gpuAtomicAdd(outDLossDMeans + 1, dMean[1]);
    gpuAtomicAdd(outDLossDMeans + 2, dMean[2]);

    outDLossDQuats += gId * 4;
    gpuAtomicAdd(outDLossDQuats + 0, dQuat[0]);
    gpuAtomicAdd(outDLossDQuats + 1, dQuat[1]);
    gpuAtomicAdd(outDLossDQuats + 2, dQuat[2]);
    gpuAtomicAdd(outDLossDQuats + 3, dQuat[3]);

    outDLossDLogScales += gId * 3;
    gpuAtomicAdd(outDLossDLogScales + 0, dLogScale[0]);
    gpuAtomicAdd(outDLossDLogScales + 1, dLogScale[1]);
    gpuAtomicAdd(outDLossDLogScales + 2, dLogScale[2]);
}

} // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
projectGaussiansUnscentedBwd(
    const torch::Tensor &means,
    const torch::Tensor &quats,
    const torch::Tensor &logScales,
    const torch::Tensor &worldToCamMatrices,
    const torch::Tensor &projectionMatrices,
    const DistortionModel cameraModel,
    const torch::Tensor &distortionCoeffs,
    const int64_t imageWidth,
    const int64_t imageHeight,
    const float eps2d,
    const UTParams &utParams,
    const torch::Tensor &radii,
    const torch::Tensor &conics,
    const torch::Tensor &dLossDMeans2d,
    const torch::Tensor &dLossDDepths,
    const torch::Tensor &dLossDConics) {
    FVDB_FUNC_RANGE();

    TORCH_CHECK(means.is_cuda() && means.is_contiguous(), "means must be a contiguous CUDA tensor");
    TORCH_CHECK(quats.is_cuda() && quats.is_contiguous(), "quats must be a contiguous CUDA tensor");
    TORCH_CHECK(logScales.is_cuda() && logScales.is_contiguous(),
                "logScales must be a contiguous CUDA tensor");
    TORCH_CHECK(worldToCamMatrices.is_cuda() && worldToCamMatrices.is_contiguous(),
                "worldToCamMatrices must be a contiguous CUDA tensor");
    TORCH_CHECK(projectionMatrices.is_cuda() && projectionMatrices.is_contiguous(),
                "projectionMatrices must be a contiguous CUDA tensor");
    TORCH_CHECK(radii.is_cuda() && radii.is_contiguous(), "radii must be a contiguous CUDA tensor");
    TORCH_CHECK(conics.is_cuda() && conics.is_contiguous(),
                "conics must be a contiguous CUDA tensor");
    TORCH_CHECK(dLossDMeans2d.is_cuda() && dLossDMeans2d.is_contiguous(),
                "dLossDMeans2d must be a contiguous CUDA tensor");
    TORCH_CHECK(dLossDDepths.is_cuda() && dLossDDepths.is_contiguous(),
                "dLossDDepths must be a contiguous CUDA tensor");
    TORCH_CHECK(dLossDConics.is_cuda() && dLossDConics.is_contiguous(),
                "dLossDConics must be a contiguous CUDA tensor");
    TORCH_CHECK(means.scalar_type() == torch::kFloat32,
                "projectGaussiansUnscentedBwd currently only supports float32 (matching the "
                "forward kernel)");
    // See ProjectGaussiansUnscentedBackward.h and applyOpenCVDistortionVJP's docstring: the
    // OPENCV_RATIONAL_8 / *_THIN_PRISM_* variants' extra terms have not been derived/validated
    // for this backward yet. Reject rather than silently produce wrong gradients for them.
    // PINHOLE/ORTHOGRAPHIC never reach this path at all (routed through the analytic backward).
    TORCH_CHECK(cameraModel == DistortionModel::OPENCV_RADTAN_5,
                "projectGaussiansUnscentedBwd currently only supports DistortionModel::"
                "OPENCV_RADTAN_5 (COLMAP's SIMPLE_RADIAL/RADIAL/OPENCV camera models); "
                "OPENCV_RATIONAL_8 and the *_THIN_PRISM_* variants are not yet implemented here.");

    const uint32_t N = means.size(0);
    const uint32_t C = worldToCamMatrices.size(0);
    const at::cuda::OptionalCUDAGuard deviceGuard(device_of(means));
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(means.device().index());

    torch::Tensor dLossDMeans     = torch::zeros_like(means);
    torch::Tensor dLossDQuats     = torch::zeros_like(quats);
    torch::Tensor dLossDLogScales = torch::zeros_like(logScales);

    if (C && N) {
        const dim3 numBlocks(GET_BLOCKS(N, DEFAULT_BLOCK_DIM), C);
        const auto camera = PerspectiveWithDistortionCamera<float>{
            worldToCamMatrices,
            worldToCamMatrices, // start == end: RollingShutterType::NONE only (see header).
            projectionMatrices,
            distortionCoeffs,
            C,
            distortionCoeffs.size(-1),
            static_cast<int32_t>(imageWidth),
            static_cast<int32_t>(imageHeight),
            0,
            0,
            RollingShutterType::NONE,
            cameraModel};
        projectionUnscentedBackwardKernel<float, PerspectiveWithDistortionCamera<float>>
            <<<numBlocks, DEFAULT_BLOCK_DIM, camera.numSharedMemBytes(), stream>>>(
                N,
                C,
                N,
                means.data_ptr<float>(),
                quats.data_ptr<float>(),
                logScales.data_ptr<float>(),
                camera,
                eps2d,
                utParams,
                radii.data_ptr<int32_t>(),
                conics.data_ptr<float>(),
                dLossDMeans2d.data_ptr<float>(),
                dLossDDepths.data_ptr<float>(),
                dLossDConics.data_ptr<float>(),
                dLossDMeans.data_ptr<float>(),
                dLossDQuats.data_ptr<float>(),
                dLossDLogScales.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return std::make_tuple(dLossDMeans, dLossDQuats, dLossDLogScales);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
