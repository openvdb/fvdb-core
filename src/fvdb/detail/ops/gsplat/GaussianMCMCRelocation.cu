// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#include <fvdb/detail/ops/gsplat/GaussianRelocation.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/Nvtx.h>
#include <fvdb/detail/utils/cuda/GridDim.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <complex.h>

namespace fvdb::detail::ops {
namespace {

template <typename ScalarType>
__global__ void
gaussianRelocationKernel(fvdb::TorchRAcc64<ScalarType, 1> opacities,
                         fvdb::TorchRAcc64<ScalarType, 2> scales,
                         fvdb::TorchRAcc64<int32_t, 1> ratios,
                         fvdb::TorchRAcc64<ScalarType, 2> binomialCoeffs,
                         fvdb::TorchRAcc64<ScalarType, 1> opacitiesNew,
                         fvdb::TorchRAcc64<ScalarType, 2> scalesNew,
                         std::size_t nMax) {
    const auto N = opacities.size(0);
    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N;
         idx += blockDim.x * gridDim.x) {
        const int32_t nIdx = ratios[idx];

        // compute new opacity
        opacitiesNew[idx] = ScalarType(1.0) - powf(1 - opacities[idx], ScalarType(1.0) / nIdx);

        // compute new scale
        ScalarType denomSum = 0.0f;
        for (int32_t i = 1; i <= nIdx; ++i) {
            for (int32_t k = 0; k <= (i - 1); ++k) {
                const ScalarType binomialCoefficient = binomialCoeffs[i - 1][k];
                denomSum += binomialCoefficient * pow(-1, k) * powf(opacitiesNew[idx], k + 1) *
                            rsqrtf(k + 1);
            }
        }

        const ScalarType coeff = (opacities[idx] / denomSum);
        for (int i = 0; i < 3; ++i) {
            scalesNew[idx][i] = coeff * scales[idx][i];
        }
    }
}

template <typename ScalarType>
std::tuple<torch::Tensor, torch::Tensor>
launchGaussianRelocation(const torch::Tensor &opacities,      // [N]
                         const torch::Tensor &scales,         // [N, 3]
                         const torch::Tensor &ratios,         // [N]
                         const torch::Tensor &binomialCoeffs, // [nMax, nMax]
                         const int nMax) {
    const auto N = opacities.size(0);

    auto opacitiesNew = torch::empty_like(opacities);
    auto scalesNew    = torch::empty_like(scales);

    const int blockDim                = DEFAULT_BLOCK_DIM;
    const int gridDim                 = fvdb::GET_BLOCKS(N, blockDim);
    const at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    gaussianRelocationKernel<ScalarType><<<gridDim, blockDim, 0, stream>>>(
        opacities.packed_accessor64<ScalarType, 1, torch::RestrictPtrTraits>(),
        scales.packed_accessor64<ScalarType, 2, torch::RestrictPtrTraits>(),
        ratios.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
        binomialCoeffs.packed_accessor64<ScalarType, 2, torch::RestrictPtrTraits>(),
        opacitiesNew.packed_accessor64<ScalarType, 1, torch::RestrictPtrTraits>(),
        scalesNew.packed_accessor64<ScalarType, 2, torch::RestrictPtrTraits>(),
        nMax);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(opacitiesNew, scalesNew);
}

} // namespace

template <>
std::tuple<torch::Tensor, torch::Tensor>
dispatchGaussianRelocation<torch::kCUDA>(const torch::Tensor &opacities,      // [N]
                                         const torch::Tensor &scales,         // [N, 3]
                                         const torch::Tensor &ratios,         // [N]
                                         const torch::Tensor &binomialCoeffs, // [nMax, nMax]
                                         const int nMax) {
    FVDB_FUNC_RANGE();
    const at::cuda::OptionalCUDAGuard device_guard(device_of(opacities));

    const auto N = opacities.size(0);

    TORCH_CHECK_VALUE(binomialCoeffs.is_cuda(), "binomialCoeffs must be a CUDA tensor");
    TORCH_CHECK_VALUE(binomialCoeffs.dim() == 2, "binomialCoeffs must have shape [nMax, nMax]");
    TORCH_CHECK_VALUE(binomialCoeffs.size(0) == nMax,
                      "binomialCoeffs must have shape [nMax, nMax]");
    TORCH_CHECK_VALUE(binomialCoeffs.size(1) == nMax,
                      "binomialCoeffs must have shape [nMax, nMax]");
    TORCH_CHECK_VALUE(opacities.is_cuda(), "opacities must be a CUDA tensor");
    TORCH_CHECK_VALUE(opacities.dim() == 1, "opacities must have shape [N]");
    TORCH_CHECK_VALUE(scales.is_cuda(), "scales must be a CUDA tensor");
    TORCH_CHECK_VALUE(scales.dim() == 2, "scales must have shape [N, 3]");
    TORCH_CHECK_VALUE(scales.size(0) == N, "scales must have shape [N, 3]");
    TORCH_CHECK_VALUE(scales.size(1) == 3, "scales must have shape [N, 3]");
    TORCH_CHECK_VALUE(ratios.is_cuda(), "ratios must be a CUDA tensor");
    TORCH_CHECK_VALUE(ratios.dim() == 1, "ratios must have shape [N]");
    TORCH_CHECK_VALUE(ratios.size(0) == N, "ratios must have shape [N]");
    TORCH_CHECK_VALUE(ratios.dtype() == torch::kInt32, "ratios must be an int32 tensor");

    // Only float is supported for now, matching the kernel math (powf/rsqrtf).
    return launchGaussianRelocation<float>(
        opacities.contiguous(), scales.contiguous(), ratios.contiguous(), binomialCoeffs, nMax);
}

template <>
std::tuple<torch::Tensor, torch::Tensor>
dispatchGaussianRelocation<torch::kPrivateUse1>(const torch::Tensor &opacities,      // [N]
                                                const torch::Tensor &scales,         // [N, 3]
                                                const torch::Tensor &ratios,         // [N]
                                                const torch::Tensor &binomialCoeffs, // [nMax, nMax]
                                                const int nMax) {
    // TODO: Implement PrivateUse1
    TORCH_CHECK_NOT_IMPLEMENTED(false, "PrivateUse1 implementation not available");
}

template <>
std::tuple<torch::Tensor, torch::Tensor>
dispatchGaussianRelocation<torch::kCPU>(const torch::Tensor &opacities,      // [N]
                                        const torch::Tensor &scales,         // [N, 3]
                                        const torch::Tensor &ratios,         // [N]
                                        const torch::Tensor &binomialCoeffs, // [nMax, nMax]
                                        const int nMax) {
    // CPU path intentionally unsupported; keep signature for clearer error messaging in tests.
    TORCH_CHECK_NOT_IMPLEMENTED(false, "GaussianRelocation is not implemented for CPU");
}

} // namespace fvdb::detail::ops
