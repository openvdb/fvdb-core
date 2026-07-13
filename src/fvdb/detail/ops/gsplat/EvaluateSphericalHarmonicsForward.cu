// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/gsplat/EvaluateSphericalHarmonicsForward.h>
#include <fvdb/detail/utils/Nvtx.h>
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/cuda/GridDim.h>
#include <fvdb/detail/utils/cuda/Utils.cuh>

#include <c10/cuda/CUDAGuard.h>

namespace fvdb {
namespace detail {
namespace ops {

namespace {

// Evaluate spherical harmonics bases at unit direction for high orders using
// approach described by Efficient Spherical Harmonic Evaluation, Peter-Pike
// Sloan, JCGT 2013 See https://jcgt.org/published/0002/02/06/ for reference
// implementation
template <typename T>
inline __device__ T
evalShFunction(const int64_t degree,  // degree of SH to be evaluated
               const int64_t gi,      // gaussian index
               const int64_t c,       // render channel
               const float3 &viewDir, // [D]
               const torch::PackedTensorAccessor64<T, 3, torch::RestrictPtrTraits> sh0Coeffs,
               const torch::PackedTensorAccessor64<T, 3, torch::RestrictPtrTraits> shNCoeffs) {
    const T cSH0 = sh0Coeffs[gi][0][c];

    T result = T(0.2820947917738781) * cSH0;

    if (degree >= 1) {
        // Normally rsqrt is faster than sqrt, but --use_fast_math will optimize
        // sqrt on single precision, so we use sqrt here.
        const T inorm =
            rsqrtf(viewDir.x * viewDir.x + viewDir.y * viewDir.y + viewDir.z * viewDir.z);
        const T x = viewDir.x * inorm;
        const T y = viewDir.y * inorm;
        const T z = viewDir.z * inorm;

        const T cSH1 = shNCoeffs[gi][0][c];
        const T cSH2 = shNCoeffs[gi][1][c];
        const T cSH3 = shNCoeffs[gi][2][c];

        result += 0.48860251190292f * (-y * cSH1 + z * cSH2 - x * cSH3);

        if (degree >= 2) {
            const T z2 = z * z;

            const T fTmp0B = T(-1.092548430592079) * z;
            const T fC1    = x * x - y * y;
            const T fS1    = 2.f * x * y;
            const T pSH6   = (T(0.9461746957575601) * z2 - T(0.3153915652525201));
            const T pSH7   = fTmp0B * x;
            const T pSH5   = fTmp0B * y;
            const T pSH8   = T(0.5462742152960395) * fC1;
            const T pSH4   = T(0.5462742152960395) * fS1;

            const T cSH4 = shNCoeffs[gi][3][c];
            const T cSH5 = shNCoeffs[gi][4][c];
            const T cSH6 = shNCoeffs[gi][5][c];
            const T cSH7 = shNCoeffs[gi][6][c];
            const T cSH8 = shNCoeffs[gi][7][c];

            result += (pSH4 * cSH4) + (pSH5 * cSH5) + (pSH6 * cSH6) + (pSH7 * cSH7) + (pSH8 * cSH8);

            if (degree >= 3) {
                const T fTmp0C = T(-2.285228997322329) * z2 + T(0.4570457994644658);
                const T fTmp1B = T(1.445305721320277) * z;
                const T fC2    = x * fC1 - y * fS1;
                const T fS2    = x * fS1 + y * fC1;
                const T pSH12  = z * (T(1.865881662950577) * z2 - T(1.119528997770346));
                const T pSH13  = fTmp0C * x;
                const T pSH11  = fTmp0C * y;
                const T pSH14  = fTmp1B * fC1;
                const T pSH10  = fTmp1B * fS1;
                const T pSH15  = T(-0.5900435899266435) * fC2;
                const T pSH9   = T(-0.5900435899266435) * fS2;

                const T cSH9  = shNCoeffs[gi][8][c];
                const T cSH10 = shNCoeffs[gi][9][c];
                const T cSH11 = shNCoeffs[gi][10][c];
                const T cSH12 = shNCoeffs[gi][11][c];
                const T cSH13 = shNCoeffs[gi][12][c];
                const T cSH14 = shNCoeffs[gi][13][c];
                const T cSH15 = shNCoeffs[gi][14][c];

                result += (pSH9 * cSH9) + (pSH10 * cSH10) + (pSH11 * cSH11) + (pSH12 * cSH12) +
                          (pSH13 * cSH13) + (pSH14 * cSH14) + (pSH15 * cSH15);

                if (degree >= 4) {
                    const T fTmp0D = z * (T(-4.683325804901025) * z2 + T(2.007139630671868));
                    const T fTmp1C = T(3.31161143515146) * z2 - T(0.47308734787878);
                    const T fTmp2B = -1.770130769779931f * z;
                    const T fC3    = x * fC2 - y * fS2;
                    const T fS3    = x * fS2 + y * fC2;
                    const T pSH20 =
                        (T(1.984313483298443) * z * pSH12 - T(1.006230589874905) * pSH6);
                    const T pSH21 = fTmp0D * x;
                    const T pSH19 = fTmp0D * y;
                    const T pSH22 = fTmp1C * fC1;
                    const T pSH18 = fTmp1C * fS1;
                    const T pSH23 = fTmp2B * fC2;
                    const T pSH17 = fTmp2B * fS2;
                    const T pSH24 = T(0.6258357354491763) * fC3;
                    const T pSH16 = T(0.6258357354491763) * fS3;

                    const T cSH16 = shNCoeffs[gi][15][c];
                    const T cSH17 = shNCoeffs[gi][16][c];
                    const T cSH18 = shNCoeffs[gi][17][c];
                    const T cSH19 = shNCoeffs[gi][18][c];
                    const T cSH20 = shNCoeffs[gi][19][c];
                    const T cSH21 = shNCoeffs[gi][20][c];
                    const T cSH22 = shNCoeffs[gi][21][c];
                    const T cSH23 = shNCoeffs[gi][22][c];
                    const T cSH24 = shNCoeffs[gi][23][c];

                    result += (pSH16 * cSH16) + (pSH17 * cSH17) + (pSH18 * cSH18) +
                              (pSH19 * cSH19) + (pSH20 * cSH20) + (pSH21 * cSH21) +
                              (pSH22 * cSH22) + (pSH23 * cSH23) + (pSH24 * cSH24);
                }
            }
        }
    }

    return result + T(0.5);
}

// Compute the world-space direction from a camera center to a Gaussian center without
// materializing the full [C, N, 3] direction tensor. World-to-camera matrices are rigid
// transforms [R | t], so the camera center is -R^T t and the direction is mean + R^T t.
template <typename T>
inline __device__ float3
viewDirectionFromViewMatrix(const T *__restrict__ mean, const T *__restrict__ worldToCamMatrix) {
    const T tx = worldToCamMatrix[3];
    const T ty = worldToCamMatrix[7];
    const T tz = worldToCamMatrix[11];
    return make_float3(
        mean[0] + worldToCamMatrix[0] * tx + worldToCamMatrix[4] * ty + worldToCamMatrix[8] * tz,
        mean[1] + worldToCamMatrix[1] * tx + worldToCamMatrix[5] * ty + worldToCamMatrix[9] * tz,
        mean[2] + worldToCamMatrix[2] * tx + worldToCamMatrix[6] * ty + worldToCamMatrix[10] * tz);
}

// Evaluate SH directly from world-space Gaussian means and rigid world-to-camera matrices.
// cameraIds and gaussianIds are both null for a dense [C, N] problem. For an indexed problem,
// the logical shape is [1, N] and the two arrays map each logical Gaussian-camera pair back to
// its camera and world-space Gaussian.
template <typename T>
__global__ __launch_bounds__(DEFAULT_BLOCK_DIM) void
computeSh(
    const int64_t offset,
    const int64_t count,
    const int64_t C,
    const int64_t N,
    const int64_t D,
    const int64_t shDegreeToUse,
    const T *__restrict__ means,                                                   // [N, 3]
    const T *__restrict__ worldToCamMatrices,                                      // [C, 4, 4]
    const int32_t *__restrict__ cameraIds,                                         // [N] optional
    const int32_t *__restrict__ gaussianIds,                                       // [N] optional
    const torch::PackedTensorAccessor64<T, 3, torch::RestrictPtrTraits> sh0Coeffs, // [1, N, D]
    const torch::PackedTensorAccessor64<T, 3, torch::RestrictPtrTraits> shNCoeffs, // [K-1, N, D]
    const int *__restrict__ radii,                                                 // [C, N, 2]
    T *__restrict__ outRenderQuantities                                            // [C, N, D]
) {
    // parallelize over C * N * D
    auto idx = blockIdx.x * blockDim.x + threadIdx.x; // cidx * N * D + gidx * D + kidx
    if (idx >= count * C * D) {
        return;
    }

    const auto eid = idx / D;              // cidx * N + gidx
    const auto cid = eid / count;          // camera index
    const auto gid = eid % count + offset; // gaussian index
    const auto c   = idx % D;              // render channel

    T result           = T(0);
    const bool visible = radii == nullptr ||
                         (radii[(cid * N + gid) * 2 + 0] > 0 && radii[(cid * N + gid) * 2 + 1] > 0);
    if (visible) {
        static_assert(std::is_same<T, float>::value,
                      "SH kernels assume float precision (float3 casts)");
        const int64_t worldCid = cameraIds == nullptr ? cid : cameraIds[gid];
        const int64_t worldGid = gaussianIds == nullptr ? gid : gaussianIds[gid];
        const float3 viewDir =
            viewDirectionFromViewMatrix(means + worldGid * 3, worldToCamMatrices + worldCid * 16);
        result = evalShFunction(shDegreeToUse, gid, c, viewDir, sh0Coeffs, shNCoeffs);
    }
    outRenderQuantities[(cid * N + gid) * D + c] = result;
}

template <typename T>
__global__ __launch_bounds__(DEFAULT_BLOCK_DIM) void
computeShDiffuseOnly(const int64_t offset,
                     const int64_t count,
                     const int64_t C,
                     const int64_t N,
                     const int64_t D,
                     const torch::PackedTensorAccessor64<T, 3, torch::RestrictPtrTraits> sh0Coeffs,
                     const int *__restrict__ radii, // [C, N, 2]
                     T *__restrict__ outRenderQuantities) {
    // parallelize over C * N * D
    auto idx = blockIdx.x * blockDim.x + threadIdx.x; // cidx * N * D + gidx * D + kidx
    if (idx >= count * C * D) {
        return;
    }

    const auto eid = idx / D;              // cidx * N + gidx
    const auto cid = eid / count;          // camera index
    const auto gid = eid % count + offset; // gaussian index
    const auto c   = idx % D;              // render channel
    if (radii != nullptr &&
        (radii[(cid * N + gid) * 2 + 0] <= 0 || radii[(cid * N + gid) * 2 + 1] <= 0)) {
        outRenderQuantities[(cid * N + gid) * D + c] = T(0);
    } else {
        outRenderQuantities[(cid * N + gid) * D + c] =
            T(0.2820947917738781) * sh0Coeffs[gid][0][c] + T(0.5);
    }
}

} // namespace

template <torch::DeviceType>
torch::Tensor
dispatchEvaluateSphericalHarmonicsFwd(const int64_t shDegreeToUse,
                                      const int64_t numCameras,
                                      const torch::Tensor &means,              // [N, 3]
                                      const torch::Tensor &worldToCamMatrices, // [C, 4, 4]
                                      const torch::Tensor &cameraIds,          // [N] optional
                                      const torch::Tensor &gaussianIds,        // [N] optional
                                      const torch::Tensor &sh0Coeffs,          // [N, 1, D]
                                      const torch::Tensor &shNCoeffs,          // [N, K-1, D]
                                      const torch::Tensor &radii               // [C, N, 2]
);

template <>
torch::Tensor
dispatchEvaluateSphericalHarmonicsFwd<torch::kCUDA>(const int64_t shDegreeToUse,
                                                    const int64_t numCameras,
                                                    const torch::Tensor &means,
                                                    const torch::Tensor &worldToCamMatrices,
                                                    const torch::Tensor &cameraIds,
                                                    const torch::Tensor &gaussianIds,
                                                    const torch::Tensor &sh0Coeffs,
                                                    const torch::Tensor &shNCoeffs,
                                                    const torch::Tensor &radii) {
    FVDB_FUNC_RANGE();
    // Valid modes:
    // 0: sh0Coeffs only
    // 1: sh0Coeffs + radii
    // 2: sh0Coeffs + shNCoeffs + world-space geometry
    // 2: sh0Coeffs + shNCoeffs + world-space geometry + radii

    const at::cuda::OptionalCUDAGuard device_guard(at::device_of(sh0Coeffs));

    const bool hasShNCoeffs = shNCoeffs.defined();
    const bool hasMeans     = means.defined();
    const bool hasViewMats  = worldToCamMatrices.defined();
    const bool hasRadii     = radii.defined();

    const int64_t numGaussians = sh0Coeffs.size(0);

    TORCH_CHECK_VALUE(sh0Coeffs.dim() == 3, "sh0Coeffs must have shape [K, N, D]");
    TORCH_CHECK_VALUE(sh0Coeffs.is_cuda(), "sh0Coeffs must be a CUDA tensor");

    if (hasShNCoeffs) {
        TORCH_CHECK_VALUE(hasMeans, "means must be defined if shNCoeffs is defined");
        TORCH_CHECK_VALUE(hasViewMats,
                          "worldToCamMatrices must be defined if shNCoeffs is defined");
        TORCH_CHECK_VALUE(shNCoeffs.is_cuda(), "shNCoeffs must be a CUDA tensor");
        TORCH_CHECK_VALUE(shNCoeffs.dim() == 3, "shNCoeffs must have shape [N, K, D]");
        TORCH_CHECK_VALUE(shNCoeffs.size(0) == numGaussians, "shNCoeffs must have shape [N, K, D]");
    } else {
        TORCH_CHECK_VALUE(shDegreeToUse == 0, "shDegreeToUse must be 0 if no shNCoeffs");
    }

    if (hasRadii) {
        TORCH_CHECK_VALUE(radii.dim() == 3 && radii.size(2) == 2,
                          "radii must have shape [C, N, 2]");
        TORCH_CHECK_VALUE(numGaussians == radii.size(1), "radii must have shape [C, N, 2]");
        TORCH_CHECK_VALUE(radii.size(0) == numCameras,
                          "radii must have shape [C, N, 2] and C must match numCameras");
        TORCH_CHECK_VALUE(radii.is_cuda(), "radii must be a CUDA tensor");
        TORCH_CHECK_VALUE(radii.is_contiguous(), "radii must be a contiguous");
    }

    const int64_t K        = hasShNCoeffs ? shNCoeffs.size(1) + 1 : 1;
    const int64_t N        = sh0Coeffs.size(0);
    const int64_t C        = numCameras;
    const int64_t D        = sh0Coeffs.size(2);
    const auto TOTAL_ELEMS = C * N * D;
    const auto NUM_BLOCKS  = GET_BLOCKS(TOTAL_ELEMS, DEFAULT_BLOCK_DIM);

    const bool hasCameraIds   = cameraIds.defined() && cameraIds.numel() > 0;
    const bool hasGaussianIds = gaussianIds.defined() && gaussianIds.numel() > 0;
    const bool indexed        = hasCameraIds || hasGaussianIds ||
                         (hasMeans && hasViewMats && cameraIds.defined() && gaussianIds.defined() &&
                          (means.size(0) != N || worldToCamMatrices.size(0) != C));

    // View directions are computed in the kernel from the means and world-to-camera matrices.
    if (hasShNCoeffs && K > 0 && shDegreeToUse > 0) {
        TORCH_CHECK_VALUE(means.is_cuda() && means.scalar_type() == torch::kFloat32,
                          "means must be a float32 CUDA tensor");
        TORCH_CHECK_VALUE(means.dim() == 2 && means.size(1) == 3 && means.is_contiguous(),
                          "means must be contiguous with shape [N, 3]");
        TORCH_CHECK_VALUE(worldToCamMatrices.is_cuda() &&
                              worldToCamMatrices.scalar_type() == torch::kFloat32,
                          "worldToCamMatrices must be a float32 CUDA tensor");
        TORCH_CHECK_VALUE(worldToCamMatrices.dim() == 3 && worldToCamMatrices.size(1) == 4 &&
                              worldToCamMatrices.size(2) == 4 && worldToCamMatrices.is_contiguous(),
                          "worldToCamMatrices must be contiguous with shape [C, 4, 4]");
        TORCH_CHECK_VALUE(means.device() == sh0Coeffs.device() &&
                              worldToCamMatrices.device() == sh0Coeffs.device(),
                          "means and worldToCamMatrices must be on the same device as sh0Coeffs");
        TORCH_CHECK_VALUE(
            hasCameraIds == hasGaussianIds,
            "cameraIds and gaussianIds must either both be empty or both be populated");
        if (indexed) {
            TORCH_CHECK_VALUE(C == 1, "indexed SH evaluation must have numCameras == 1");
            TORCH_CHECK_VALUE(cameraIds.is_cuda() && gaussianIds.is_cuda(),
                              "cameraIds and gaussianIds must be CUDA tensors");
            TORCH_CHECK_VALUE(cameraIds.scalar_type() == torch::kInt32 &&
                                  gaussianIds.scalar_type() == torch::kInt32,
                              "cameraIds and gaussianIds must be int32 tensors");
            TORCH_CHECK_VALUE(cameraIds.dim() == 1 && gaussianIds.dim() == 1 &&
                                  cameraIds.size(0) == N && gaussianIds.size(0) == N,
                              "cameraIds and gaussianIds must have shape [N]");
            TORCH_CHECK_VALUE(cameraIds.is_contiguous() && gaussianIds.is_contiguous(),
                              "cameraIds and gaussianIds must be contiguous");
            TORCH_CHECK_VALUE(cameraIds.device() == sh0Coeffs.device() &&
                                  gaussianIds.device() == sh0Coeffs.device(),
                              "cameraIds and gaussianIds must be on the same device as sh0Coeffs");
        } else {
            TORCH_CHECK_VALUE(means.size(0) == N, "means must have shape [N, 3] in dense mode");
            TORCH_CHECK_VALUE(worldToCamMatrices.size(0) == C,
                              "worldToCamMatrices must have shape [C, 4, 4] in dense mode");
        }
    }

    if (N == 0) {
        return torch::empty({int64_t(C), N, D}, sh0Coeffs.options());
    }

    using scalar_t                    = float;
    const at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(sh0Coeffs.device().index());

    const int *radiiPtr            = hasRadii ? radii.data_ptr<int>() : nullptr;
    torch::Tensor renderQuantities = torch::empty({int64_t(C), N, D}, sh0Coeffs.options());
    if (hasShNCoeffs && shDegreeToUse > 0) {
        computeSh<scalar_t><<<NUM_BLOCKS, DEFAULT_BLOCK_DIM, 0, stream>>>(
            0,
            N,
            C,
            N,
            D,
            shDegreeToUse,
            means.data_ptr<scalar_t>(),
            worldToCamMatrices.data_ptr<scalar_t>(),
            indexed ? cameraIds.data_ptr<int32_t>() : nullptr,
            indexed ? gaussianIds.data_ptr<int32_t>() : nullptr,
            sh0Coeffs.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
            shNCoeffs.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
            radiiPtr,
            renderQuantities.data_ptr<scalar_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        computeShDiffuseOnly<scalar_t><<<NUM_BLOCKS, DEFAULT_BLOCK_DIM, 0, stream>>>(
            0,
            N,
            C,
            N,
            D,
            sh0Coeffs.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
            radiiPtr,
            renderQuantities.data_ptr<scalar_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return renderQuantities;
}

template <>
torch::Tensor
dispatchEvaluateSphericalHarmonicsFwd<torch::kPrivateUse1>(const int64_t shDegreeToUse,
                                                           const int64_t numCameras,
                                                           const torch::Tensor &means,
                                                           const torch::Tensor &worldToCamMatrices,
                                                           const torch::Tensor &cameraIds,
                                                           const torch::Tensor &gaussianIds,
                                                           const torch::Tensor &sh0Coeffs,
                                                           const torch::Tensor &shNCoeffs,
                                                           const torch::Tensor &radii) {
    FVDB_FUNC_RANGE();
    // Valid modes:
    // 0: sh0Coeffs only
    // 1: sh0Coeffs + radii
    // 2: sh0Coeffs + shNCoeffs + world-space geometry
    // 2: sh0Coeffs + shNCoeffs + world-space geometry + radii

    const bool hasShNCoeffs = shNCoeffs.defined();
    const bool hasMeans     = means.defined();
    const bool hasViewMats  = worldToCamMatrices.defined();
    const bool hasRadii     = radii.defined();

    const int64_t numGaussians = sh0Coeffs.size(0);

    TORCH_CHECK_VALUE(sh0Coeffs.dim() == 3, "sh0Coeffs must have shape [K, N, D]");
    TORCH_CHECK_VALUE(sh0Coeffs.is_privateuseone(), "sh0Coeffs must be a PrivateUse1 tensor");

    if (hasShNCoeffs) {
        TORCH_CHECK_VALUE(hasMeans, "means must be defined if shNCoeffs is defined");
        TORCH_CHECK_VALUE(hasViewMats,
                          "worldToCamMatrices must be defined if shNCoeffs is defined");
        TORCH_CHECK_VALUE(shNCoeffs.is_privateuseone(), "shNCoeffs must be a PrivateUse1 tensor");
        TORCH_CHECK_VALUE(shNCoeffs.dim() == 3, "shNCoeffs must have shape [N, K, D]");
        TORCH_CHECK_VALUE(shNCoeffs.size(0) == numGaussians, "shNCoeffs must have shape [N, K, D]");
    } else {
        TORCH_CHECK_VALUE(shDegreeToUse == 0, "shDegreeToUse must be 0 if no shNCoeffs");
    }

    if (hasRadii) {
        TORCH_CHECK_VALUE(radii.dim() == 3 && radii.size(2) == 2,
                          "radii must have shape [C, N, 2]");
        TORCH_CHECK_VALUE(numGaussians == radii.size(1), "radii must have shape [C, N, 2]");
        TORCH_CHECK_VALUE(radii.size(0) == numCameras,
                          "radii must have shape [C, N, 2] and C must match numCameras");
        TORCH_CHECK_VALUE(radii.is_privateuseone(), "radii must be a PrivateUse1 tensor");
        TORCH_CHECK_VALUE(radii.is_contiguous(), "radii must be a contiguous");
    }

    const int64_t K = hasShNCoeffs ? shNCoeffs.size(1) + 1 : 1;
    const int64_t N = sh0Coeffs.size(0);
    const int64_t C = numCameras;
    const int64_t D = sh0Coeffs.size(2);

    // View directions are computed in the kernel from the means and world-to-camera matrices.
    if (hasShNCoeffs && K > 0 && shDegreeToUse > 0) {
        const bool hasCameraIds   = cameraIds.defined() && cameraIds.numel() > 0;
        const bool hasGaussianIds = gaussianIds.defined() && gaussianIds.numel() > 0;
        TORCH_CHECK_VALUE(!hasCameraIds && !hasGaussianIds,
                          "indexed SH evaluation is not available on PrivateUse1");
        TORCH_CHECK_VALUE(means.is_privateuseone() && means.scalar_type() == torch::kFloat32,
                          "means must be a float32 PrivateUse1 tensor");
        TORCH_CHECK_VALUE(means.dim() == 2 && means.size(0) == N && means.size(1) == 3 &&
                              means.is_contiguous(),
                          "means must be contiguous with shape [N, 3]");
        TORCH_CHECK_VALUE(worldToCamMatrices.is_privateuseone() &&
                              worldToCamMatrices.scalar_type() == torch::kFloat32,
                          "worldToCamMatrices must be a float32 PrivateUse1 tensor");
        TORCH_CHECK_VALUE(worldToCamMatrices.dim() == 3 && worldToCamMatrices.size(0) == C &&
                              worldToCamMatrices.size(1) == 4 && worldToCamMatrices.size(2) == 4 &&
                              worldToCamMatrices.is_contiguous(),
                          "worldToCamMatrices must be contiguous with shape [C, 4, 4]");
    }

    if (N == 0) {
        return torch::empty({int64_t(C), N, D}, sh0Coeffs.options());
    }

    using scalar_t = float;

    const int *radiiPtr            = hasRadii ? radii.data_ptr<int>() : nullptr;
    torch::Tensor renderQuantities = torch::empty({int64_t(C), N, D}, sh0Coeffs.options());

    for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        auto stream = c10::cuda::getCurrentCUDAStream(deviceId);

        int64_t elementOffset, elementCount;
        std::tie(elementOffset, elementCount) = deviceChunk(N, deviceId);

        const auto NUM_BLOCKS = GET_BLOCKS(C * elementCount * D, DEFAULT_BLOCK_DIM);

        if (hasShNCoeffs && shDegreeToUse > 0) {
            computeSh<scalar_t><<<NUM_BLOCKS, DEFAULT_BLOCK_DIM, 0, stream>>>(
                elementOffset,
                elementCount,
                C,
                N,
                D,
                shDegreeToUse,
                means.data_ptr<scalar_t>(),
                worldToCamMatrices.data_ptr<scalar_t>(),
                nullptr,
                nullptr,
                sh0Coeffs.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                shNCoeffs.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                radiiPtr,
                renderQuantities.data_ptr<scalar_t>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
            computeShDiffuseOnly<scalar_t><<<NUM_BLOCKS, DEFAULT_BLOCK_DIM, 0, stream>>>(
                elementOffset,
                elementCount,
                C,
                N,
                D,
                sh0Coeffs.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                radiiPtr,
                renderQuantities.data_ptr<scalar_t>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }

    mergeStreams();
    return renderQuantities;
}

template <>
torch::Tensor
dispatchEvaluateSphericalHarmonicsFwd<torch::kCPU>(const int64_t shDegreeToUse,
                                                   const int64_t numCameras,
                                                   const torch::Tensor &means,
                                                   const torch::Tensor &worldToCamMatrices,
                                                   const torch::Tensor &cameraIds,
                                                   const torch::Tensor &gaussianIds,
                                                   const torch::Tensor &sh0Coeffs,
                                                   const torch::Tensor &shNCoeffs,
                                                   const torch::Tensor &radii) {
    TORCH_CHECK(false, "CPU implementation not available");
}

torch::Tensor
evaluateSphericalHarmonicsFwd(const int64_t shDegreeToUse,
                              const int64_t numCameras,
                              const torch::Tensor &means,
                              const torch::Tensor &worldToCamMatrices,
                              const torch::Tensor &cameraIds,
                              const torch::Tensor &gaussianIds,
                              const torch::Tensor &sh0Coeffs,
                              const torch::Tensor &shNCoeffs,
                              const torch::Tensor &radii) {
    return FVDB_DISPATCH_KERNEL(sh0Coeffs.device(), [&]() {
        return dispatchEvaluateSphericalHarmonicsFwd<DeviceTag>(shDegreeToUse,
                                                                numCameras,
                                                                means,
                                                                worldToCamMatrices,
                                                                cameraIds,
                                                                gaussianIds,
                                                                sh0Coeffs,
                                                                shNCoeffs,
                                                                radii);
    });
}

} // namespace ops
} // namespace detail
} // namespace fvdb
