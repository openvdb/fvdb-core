// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#include <fvdb/detail/ops/gsplat/GaussianMCMCRelocation.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/Nvtx.h>
#include <fvdb/detail/utils/cuda/GridDim.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <complex.h>

#include <algorithm>

namespace fvdb::detail::ops {
namespace {

template <typename ScalarType>
__global__ void
gaussianRelocationKernel(fvdb::TorchRAcc64<ScalarType, 2> logScales,
                         fvdb::TorchRAcc64<ScalarType, 1> logitOpacities,
                         fvdb::TorchRAcc64<int32_t, 1> ratios,
                         fvdb::TorchRAcc64<ScalarType, 2> binomialCoeffs,
                         fvdb::TorchRAcc64<ScalarType, 2> logScalesNew,
                         fvdb::TorchRAcc64<ScalarType, 1> logitOpacitiesNew,
                         std::size_t nMax,
                         float minOpacity) {
    const auto N = logScales.size(0);
    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N;
         idx += blockDim.x * gridDim.x) {
        const int32_t nIdx = ratios[idx];

        // convert logit opacity to opacity
        auto opacity    = ScalarType(1.0) / (1 + exp(-logitOpacities[idx]));
        auto opacityNew = ScalarType(1.0) - powf(1 - opacity, ScalarType(1.0) / nIdx);
        opacityNew      = std::clamp<ScalarType>(
            opacityNew, minOpacity, 1.0 - std::numeric_limits<ScalarType>::epsilon());
        // logit(x) = log(x / (1-x))
        logitOpacitiesNew[idx] = log(opacityNew / (ScalarType(1.0) - opacityNew));

        // compute new scale
        ScalarType denomSum = 0.0f;
        for (int32_t i = 1; i <= nIdx; ++i) {
            for (int32_t k = 0; k <= (i - 1); ++k) {
                const ScalarType binomialCoefficient = binomialCoeffs[i - 1][k];
                denomSum +=
                    binomialCoefficient * pow(-1, k) * powf(opacityNew, k + 1) * rsqrtf(k + 1);
            }
        }

        const ScalarType coeff = (opacity / denomSum);
        for (int i = 0; i < 3; ++i) {
            // convert log scale to scale, multiply by coeff and convert back to log scale
            logScalesNew[idx][i] = log(coeff * ::cuda::std::exp(logScales[idx][i]));
        }
    }
}

template <typename ScalarType>
std::tuple<torch::Tensor, torch::Tensor>
launchGaussianRelocation(const torch::Tensor &logScales,      // [N, 3]
                         const torch::Tensor &logitOpacities, // [N]
                         const torch::Tensor &ratios,         // [N]
                         const torch::Tensor &binomialCoeffs, // [nMax, nMax]
                         const int nMax,
                         ScalarType minOpacity) {
    const auto N = logitOpacities.size(0);

    auto logitOpacitiesNew = torch::empty_like(logitOpacities);
    auto logScalesNew      = torch::empty_like(logScales);

    const int blockDim                = DEFAULT_BLOCK_DIM;
    const int gridDim                 = fvdb::GET_BLOCKS(N, blockDim);
    const at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    gaussianRelocationKernel<ScalarType><<<gridDim, blockDim, 0, stream>>>(
        logScales.packed_accessor64<ScalarType, 2, torch::RestrictPtrTraits>(),
        logitOpacities.packed_accessor64<ScalarType, 1, torch::RestrictPtrTraits>(),
        ratios.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
        binomialCoeffs.packed_accessor64<ScalarType, 2, torch::RestrictPtrTraits>(),
        logScalesNew.packed_accessor64<ScalarType, 2, torch::RestrictPtrTraits>(),
        logitOpacitiesNew.packed_accessor64<ScalarType, 1, torch::RestrictPtrTraits>(),
        nMax,
        minOpacity);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(logitOpacitiesNew, logScalesNew);
}

} // namespace

template <>
std::tuple<torch::Tensor, torch::Tensor>
dispatchGaussianRelocation<torch::kCUDA>(const torch::Tensor &logScales,      // [N, 3]
                                         const torch::Tensor &logitOpacities, // [N]
                                         const torch::Tensor &ratios,         // [N]
                                         const torch::Tensor &binomialCoeffs, // [nMax, nMax]
                                         const int nMax,
                                         float minOpacity) {
    FVDB_FUNC_RANGE();
    const at::cuda::OptionalCUDAGuard device_guard(device_of(logScales));

    const auto N = logScales.size(0);

    TORCH_CHECK_VALUE(binomialCoeffs.is_cuda(), "binomialCoeffs must be a CUDA tensor");
    TORCH_CHECK_VALUE(binomialCoeffs.dim() == 2, "binomialCoeffs must have shape [nMax, nMax]");
    TORCH_CHECK_VALUE(binomialCoeffs.size(0) == nMax,
                      "binomialCoeffs must have shape [nMax, nMax]");
    TORCH_CHECK_VALUE(binomialCoeffs.size(1) == nMax,
                      "binomialCoeffs must have shape [nMax, nMax]");
    TORCH_CHECK_VALUE(logScales.is_cuda(), "logScales must be a CUDA tensor");
    TORCH_CHECK_VALUE(logScales.dim() == 2, "logScales must have shape [N, 3]");
    TORCH_CHECK_VALUE(logScales.size(0) == N, "logScales must have shape [N, 3]");
    TORCH_CHECK_VALUE(logScales.size(1) == 3, "logScales must have shape [N, 3]");
    TORCH_CHECK_VALUE(logitOpacities.is_cuda(), "logitOpacities must be a CUDA tensor");
    TORCH_CHECK_VALUE(logitOpacities.dim() == 1, "logitOpacities must have shape [N]");
    TORCH_CHECK_VALUE(logitOpacities.size(0) == N, "logitOpacities must have shape [N]");
    TORCH_CHECK_VALUE(ratios.is_cuda(), "ratios must be a CUDA tensor");
    TORCH_CHECK_VALUE(ratios.dim() == 1, "ratios must have shape [N]");
    TORCH_CHECK_VALUE(ratios.size(0) == N, "ratios must have shape [N]");
    TORCH_CHECK_VALUE(ratios.dtype() == torch::kInt32, "ratios must be an int32 tensor");

    // Only float is supported for now, matching the kernel math (powf/rsqrtf).
    return launchGaussianRelocation<float>(logScales.contiguous(),
                                           logitOpacities.contiguous(),
                                           ratios.contiguous(),
                                           binomialCoeffs,
                                           nMax,
                                           minOpacity);
}

template <>
std::tuple<torch::Tensor, torch::Tensor>
dispatchGaussianRelocation<torch::kPrivateUse1>(const torch::Tensor &logScales,      // [N, 3]
                                                const torch::Tensor &logitOpacities, // [N]
                                                const torch::Tensor &ratios,         // [N]
                                                const torch::Tensor &binomialCoeffs, // [nMax, nMax]
                                                const int nMax,
                                                float minOpacity) {
    // TODO: Implement PrivateUse1
    TORCH_CHECK_NOT_IMPLEMENTED(false, "PrivateUse1 implementation not available");
}

template <>
std::tuple<torch::Tensor, torch::Tensor>
dispatchGaussianRelocation<torch::kCPU>(const torch::Tensor &logScales,      // [N, 3]
                                        const torch::Tensor &logitOpacities, // [N]
                                        const torch::Tensor &ratios,         // [N]
                                        const torch::Tensor &binomialCoeffs, // [nMax, nMax]
                                        const int nMax,
                                        float minOpacity) {
    // CPU path intentionally unsupported; keep signature for clearer error messaging in tests.
    TORCH_CHECK_NOT_IMPLEMENTED(false, "GaussianRelocation is not implemented for CPU");
}

} // namespace fvdb::detail::ops
