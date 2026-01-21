// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#include <fvdb/detail/ops/gsplat/GaussianMCMCAddNoise.h>
#include <fvdb/detail/ops/gsplat/GaussianUtils.cuh>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/Nvtx.h>
#include <fvdb/detail/utils/cuda/GridDim.h>
#include <fvdb/detail/utils/cuda/Utils.cuh>

#include <nanovdb/math/Math.h>

#include <c10/cuda/CUDAGuard.h>

namespace fvdb::detail::ops {

using fvdb::detail::deviceChunk;
using fvdb::detail::mergeStreams;

template <typename ScalarType>
inline __device__ ScalarType
sigmoid(ScalarType x) {
    return ScalarType(1) / (ScalarType(1) + ::cuda::std::exp(-x));
}

template <typename ScalarType>
__global__ void
gaussianMCMCAddNoiseKernel(int64_t localToGlobalOffset,
                           int64_t localSize,
                           fvdb::TorchRAcc64<ScalarType, 2> outMeans,
                           fvdb::TorchRAcc64<ScalarType, 2> logScales,
                           fvdb::TorchRAcc64<ScalarType, 1> logitOpacities,
                           fvdb::TorchRAcc64<ScalarType, 2> quats,
                           fvdb::TorchRAcc64<ScalarType, 2> baseNoise,
                           const ScalarType noiseScale,
                           const ScalarType t,
                           const ScalarType k) {
    const auto N = outMeans.size(0);
    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x + localToGlobalOffset;
         idx < localSize + localToGlobalOffset;
         idx += blockDim.x * gridDim.x) {
        const auto opacity = sigmoid(logitOpacities[idx]);

        const auto quatAcc     = quats[idx];
        const auto logScaleAcc = logScales[idx];
        const auto covar       = quaternionAndScaleToCovariance<ScalarType>(
            nanovdb::math::Vec4<ScalarType>(quatAcc[0], quatAcc[1], quatAcc[2], quatAcc[3]),
            nanovdb::math::Vec3<ScalarType>(::cuda::std::exp(logScaleAcc[0]),
                                            ::cuda::std::exp(logScaleAcc[1]),
                                            ::cuda::std::exp(logScaleAcc[2])));

        nanovdb::math::Vec3<ScalarType> noise = {
            baseNoise[idx][0], baseNoise[idx][1], baseNoise[idx][2]};

        // The noise term is scaled down based on the opacity of the Gaussian.
        // More opaque Gaussians get less noise added to them.
        // The parameters t and k control the transition point and sharpness
        // of the scaling function.
        noise *= sigmoid(-k * (opacity - t)) * noiseScale;
        noise = covar * noise;
        outMeans[idx][0] += noise[0];
        outMeans[idx][1] += noise[1];
        outMeans[idx][2] += noise[2];
    }
}

template <typename ScalarType>
void
launchGaussianMCMCAddNoise(torch::Tensor &means,                // [N, 3]
                           const torch::Tensor &logScales,      // [N, 3]
                           const torch::Tensor &logitOpacities, // [N]
                           const torch::Tensor &quats,          // [N, 4]
                           const torch::Tensor &baseNoise,      // [N, 3]
                           const ScalarType noiseScale,
                           const ScalarType t,
                           const ScalarType k,
                           int64_t offset,
                           int64_t size,
                           cudaStream_t stream) {
    const int blockDim = DEFAULT_BLOCK_DIM;
    const int gridDim  = fvdb::GET_BLOCKS(size, blockDim);

    gaussianMCMCAddNoiseKernel<ScalarType><<<gridDim, blockDim, 0, stream>>>(
        offset,
        size,
        means.packed_accessor64<ScalarType, 2, torch::RestrictPtrTraits>(),
        logScales.packed_accessor64<ScalarType, 2, torch::RestrictPtrTraits>(),
        logitOpacities.packed_accessor64<ScalarType, 1, torch::RestrictPtrTraits>(),
        quats.packed_accessor64<ScalarType, 2, torch::RestrictPtrTraits>(),
        baseNoise.packed_accessor64<ScalarType, 2, torch::RestrictPtrTraits>(),
        noiseScale,
        t,
        k);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <>
void
dispatchGaussianMCMCAddNoise<torch::kCUDA>(torch::Tensor &means,                // [N, 3]
                                           const torch::Tensor &logScales,      // [N]
                                           const torch::Tensor &logitOpacities, // [N]
                                           const torch::Tensor &quats,          // [N, 4]
                                           const float noiseScale,
                                           const float t,
                                           const float k) {
    FVDB_FUNC_RANGE();
    const at::cuda::OptionalCUDAGuard device_guard(device_of(means));

    const auto N = means.size(0);
    const at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    auto baseNoise = torch::randn_like(means);

    launchGaussianMCMCAddNoise<float>(means, logScales, logitOpacities, quats, baseNoise,
                                      noiseScale, t, k, 0, N, stream);
}

template <>
void
dispatchGaussianMCMCAddNoise<torch::kPrivateUse1>(torch::Tensor &means, // [N, 3] input/output
                                                  const torch::Tensor &logScales,      // [N, 3]
                                                  const torch::Tensor &logitOpacities, // [N]
                                                  const torch::Tensor &quats,          // [N, 4]
                                                  const float noiseScale,
                                                  const float t,
                                                  const float k) {
    FVDB_FUNC_RANGE();

    const auto N = means.size(0);

    // Generate base noise once for all devices (unified memory)
    auto baseNoise = torch::randn_like(means);

    for (const auto deviceId : c10::irange(c10::cuda::device_count())) {
        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        auto stream = c10::cuda::getCurrentCUDAStream(deviceId);

        int64_t deviceOffset, deviceSize;
        std::tie(deviceOffset, deviceSize) = deviceChunk(N, deviceId);

        if (deviceSize > 0) {
            launchGaussianMCMCAddNoise<float>(means, logScales, logitOpacities, quats, baseNoise,
                                              noiseScale, t, k, deviceOffset, deviceSize, stream);
        }
    }

    mergeStreams();
}

template <>
void
dispatchGaussianMCMCAddNoise<torch::kCPU>(torch::Tensor &means,           // [N, 3] input/output
                                          const torch::Tensor &logScales, // [N, 3]
                                          const torch::Tensor &logitOpacities, // [N]
                                          const torch::Tensor &quats,          // [N, 4]
                                          const float noiseScale,
                                          const float t,
                                          const float k) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "GaussianMCMCAddNoise is not implemented for CPU");
}

} // namespace fvdb::detail::ops
