// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/VolumeRender.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/cuda/GridDim.h>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMathCompat.h>

#include <thrust/execution_policy.h>
#include <thrust/scan.h>

namespace fvdb {
namespace detail {
namespace ops {

namespace {

// Maximum number of channels supported by the backward kernel. Bounded so we can
// allocate on-stack scratch arrays inside the device callback without dynamic
// allocation. Bump if a use case legitimately needs more channels.
static constexpr int MAX_VOLUME_RENDER_CHANNELS = 16;

template <typename scalar_t, template <typename T, int32_t D> typename TensorAccessor>
__hostdev__ void
volumeRenderFwdCallback(const TensorAccessor<scalar_t, 1> sigmas,
                        const TensorAccessor<scalar_t, 2> rgbs,
                        const TensorAccessor<scalar_t, 1> deltas,
                        const TensorAccessor<scalar_t, 1> ts,
                        const TensorAccessor<JOffsetsType, 1> jOffsets,
                        scalar_t tsmtThreshold,
                        int32_t numChannels,
                        int32_t rayIdx,
                        TensorAccessor<int64_t, 1> outTotalSamples,
                        TensorAccessor<scalar_t, 1> outOpacity,
                        TensorAccessor<scalar_t, 1> outDepth,
                        TensorAccessor<scalar_t, 2> outRGB,
                        TensorAccessor<scalar_t, 1> outWs) {
    const JOffsetsType sampleStartIdx = jOffsets[rayIdx];
    const JOffsetsType numRaySamples  = jOffsets[rayIdx + 1] - sampleStartIdx;

    // front to back compositing
    JOffsetsType numSamples = 0;
    scalar_t T              = static_cast<scalar_t>(1.0);

    while (numSamples < numRaySamples) {
        const JOffsetsType s = sampleStartIdx + numSamples;
        const scalar_t a =
            static_cast<scalar_t>(1.0) - c10::cuda::compat::exp(-sigmas[s] * deltas[s]);
        const scalar_t w = a * T; // weight of the sample point

        // Forward pass works for arbitrary number of channels
        for (int c = 0; c < numChannels; ++c) {
            outRGB[rayIdx][c] += w * rgbs[s][c];
        }
        outDepth[rayIdx] += w * ts[s];
        // outDepthSq[rayIdx] += w*ts[s]*ts[s];
        outOpacity[rayIdx] += w;
        outWs[s] = w;
        T *= static_cast<scalar_t>(1.0) - a;
        numSamples += 1;

        // ray has enough opacity
        if (T <= tsmtThreshold) {
            break;
        }
    }
    outTotalSamples[rayIdx] = numSamples;
}

template <torch::DeviceType device,
          typename scalar_t,
          template <typename T, int32_t D>
          typename TensorAccessor>
__hostdev__ void
volumeRenderBwdCallback(const TensorAccessor<scalar_t, 1> dLdOpacity,     // [B*R]
                        const TensorAccessor<scalar_t, 1> dLdDepth,       // [B*R]
                        const TensorAccessor<scalar_t, 2> dLdRgb,         // [B*R, C]
                        const TensorAccessor<scalar_t, 1> dLdWs,          // [B*R*S]
                        const TensorAccessor<scalar_t, 1> dLdWs_times_ws, // [B*R*S]
                        const TensorAccessor<scalar_t, 1> sigmas,         // [B*R*S]
                        const TensorAccessor<scalar_t, 2> rgbs,           // [B*R*S, C]
                        const TensorAccessor<scalar_t, 1> deltas,         // [B*R*S]
                        const TensorAccessor<scalar_t, 1> ts,             // [B*R*S]
                        const TensorAccessor<JOffsetsType, 1> jOffsets,   // [B*R, 2]
                        const TensorAccessor<scalar_t, 1> opacity,        // [B*R]
                        const TensorAccessor<scalar_t, 1> depth,          // [B*R]
                        const TensorAccessor<scalar_t, 2> rgb,            // [B*R, C]
                        const scalar_t tsmtThreshold,                     // scalar
                        const int32_t rayIdx,
                        TensorAccessor<scalar_t, 1> out_dL_dsigmas,       // [B*R*S]
                        TensorAccessor<scalar_t, 2> out_dLdRgbs) {        // [B*R*S, C]

    const JOffsetsType sampleStartIdx = jOffsets[rayIdx];
    const JOffsetsType numRaySamples  = jOffsets[rayIdx + 1] - sampleStartIdx;
    const int32_t numChannels         = rgbs.size(1);

    // Empty ray: nothing to scan or accumulate. Leave out_dL_dsigmas and
    // out_dLdRgbs at their zero-initialized values and skip the prefix scan
    // (which would otherwise OOB-read dLdWs_times_ws[sampleStartIdx - 1]).
    if (numRaySamples == 0) {
        return;
    }

    // front to back compositing
    JOffsetsType numSamples = 0;
    // Per-channel final integrated radiance (from fwd pass) and running partial
    // accumulator. Bounded stack storage; host checks numChannels against
    // MAX_VOLUME_RENDER_CHANNELS before launching so overflow is impossible.
    scalar_t finalRgb[MAX_VOLUME_RENDER_CHANNELS];
    scalar_t partialRgb[MAX_VOLUME_RENDER_CHANNELS];
    for (int c = 0; c < numChannels; ++c) {
        finalRgb[c]   = rgb[rayIdx][c];
        partialRgb[c] = static_cast<scalar_t>(0.0);
    }

    scalar_t O = opacity[rayIdx], D = depth[rayIdx]; //, Dsq = depthSq[rayIdx];
    scalar_t T = static_cast<scalar_t>(1.0);
    scalar_t d = static_cast<scalar_t>(0.0);         //, dsq = static_cast<scalar_t>(0.0);
    // compute prefix sum of dLdWs * ws
    // [a0, a1, a2, a3, ...] -> [a0, a0+a1, a0+a1+a2, a0+a1+a2+a3, ...]
    if constexpr (device == torch::kCUDA) {
        thrust::inclusive_scan(thrust::device,
                               dLdWs_times_ws.data() + sampleStartIdx,
                               dLdWs_times_ws.data() + sampleStartIdx + numRaySamples,
                               dLdWs_times_ws.data() + sampleStartIdx);
    } else {
        thrust::inclusive_scan(thrust::seq,
                               dLdWs_times_ws.data() + sampleStartIdx,
                               dLdWs_times_ws.data() + sampleStartIdx + numRaySamples,
                               dLdWs_times_ws.data() + sampleStartIdx);
    }
    scalar_t dLdWs_times_ws_sum = dLdWs_times_ws[sampleStartIdx + numRaySamples - 1];

    while (numSamples < numRaySamples) {
        const JOffsetsType s = sampleStartIdx + numSamples;
        const scalar_t a =
            static_cast<scalar_t>(1.0) - c10::cuda::compat::exp(-sigmas[s] * deltas[s]);
        const scalar_t w = a * T;

        for (int c = 0; c < numChannels; ++c) {
            partialRgb[c] += w * rgbs[s][c];
        }
        d += w * ts[s]; // dsq += w*ts[s]*ts[s];
        T *= static_cast<scalar_t>(1.0) - a;

        // compute gradients by math...
        scalar_t rgbGradSum = static_cast<scalar_t>(0.0);
        for (int c = 0; c < numChannels; ++c) {
            out_dLdRgbs[s][c] = dLdRgb[rayIdx][c] * w;
            rgbGradSum += dLdRgb[rayIdx][c] * (rgbs[s][c] * T - (finalRgb[c] - partialRgb[c]));
        }

        out_dL_dsigmas[s] =
            deltas[s] * (rgbGradSum +                               // gradients from rgb
                         dLdOpacity[rayIdx] * (1 - O) +             // gradient from opacity
                         dLdDepth[rayIdx] * (ts[s] * T - (D - d)) + // gradient from depth
                         // dLdDepthSq[rayIdx]*(ts[s]*ts[s]*T-(Dsq-dsq)) +
                         T * dLdWs[s] - (dLdWs_times_ws_sum - dLdWs_times_ws[s]) // gradient from ws
                        );

        if (T <= tsmtThreshold)
            break; // ray has enough opacity
        numSamples++;
    }
}

template <typename scalar_t>
void
volumeRenderCPU(const TorchAcc<scalar_t, 1> sigmas,       // [B*R*S]
                const TorchAcc<scalar_t, 2> rgbs,         // [B*R*S, C]
                const TorchAcc<scalar_t, 1> deltas,       // [B*R*S]
                const TorchAcc<scalar_t, 1> ts,           // [B*R*S]
                const TorchAcc<JOffsetsType, 1> jOffsets, // [B*R, 2]
                const scalar_t tsmtThreshold,             // scalar
                TorchAcc<int64_t, 1> outTotalSamples,     // [B*R]
                TorchAcc<scalar_t, 1> outOpacity,         // [B*R]
                TorchAcc<scalar_t, 1> outDepth,           // [B*R]
                TorchAcc<scalar_t, 2> outRGB,             // [B*R, C]
                TorchAcc<scalar_t, 1> outWs) {            // [B*R*S]

    const int numChannels = rgbs.size(1);

    for (int rayIdx = 0; rayIdx < (jOffsets.size(0) - 1); rayIdx += 1) {
        volumeRenderFwdCallback<scalar_t, TorchAcc>(sigmas,
                                                    rgbs,
                                                    deltas,
                                                    ts,
                                                    jOffsets,
                                                    tsmtThreshold,
                                                    numChannels,
                                                    rayIdx,
                                                    outTotalSamples,
                                                    outOpacity,
                                                    outDepth,
                                                    outRGB,
                                                    outWs);
    }
}

template <typename scalar_t>
void
volumeRenderBackwardCPU(const TorchAcc<scalar_t, 1> dLdOpacity,     // [B*R]
                        const TorchAcc<scalar_t, 1> dLdDepth,       // [B*R]
                        const TorchAcc<scalar_t, 2> dLdRgb,         // [B*R, C]
                        const TorchAcc<scalar_t, 1> dLdWs,          // [B*R*S]
                        const TorchAcc<scalar_t, 1> dLdWs_times_ws, // [B*R*S]
                        const TorchAcc<scalar_t, 1> sigmas,         // [B*R*S]
                        const TorchAcc<scalar_t, 2> rgbs,           // [B*R*S, C]
                        const TorchAcc<scalar_t, 1> deltas,         // [B*R*S]
                        const TorchAcc<scalar_t, 1> ts,             // [B*R*S]
                        const TorchAcc<JOffsetsType, 1> jOffsets,   // [B*R, 2]
                        const TorchAcc<scalar_t, 1> opacity,        // [B*R]
                        const TorchAcc<scalar_t, 1> depth,          // [B*R]
                        const TorchAcc<scalar_t, 2> rgb,            // [B*R, C]
                        const scalar_t tsmtThreshold,               // scalar
                        TorchAcc<scalar_t, 1> out_dL_dsigmas,       // [B*R*S]
                        TorchAcc<scalar_t, 2> out_dLdRgbs) {        // [B*R*S, C]

    for (int rayIdx = 0; rayIdx < (jOffsets.size(0) - 1); rayIdx += 1) {
        volumeRenderBwdCallback<torch::kCPU, scalar_t, TorchAcc>(dLdOpacity,
                                                                 dLdDepth,
                                                                 dLdRgb,
                                                                 dLdWs,
                                                                 dLdWs_times_ws,
                                                                 sigmas,
                                                                 rgbs,
                                                                 deltas,
                                                                 ts,
                                                                 jOffsets,
                                                                 opacity,
                                                                 depth,
                                                                 rgb,
                                                                 tsmtThreshold,
                                                                 rayIdx,
                                                                 out_dL_dsigmas,
                                                                 out_dLdRgbs);
    }
}

template <typename scalar_t>
__global__ __launch_bounds__(DEFAULT_BLOCK_DIM) void
volumeRender(const TorchRAcc64<scalar_t, 1> sigmas,
             const TorchRAcc64<scalar_t, 2> rgbs,
             const TorchRAcc64<scalar_t, 1> deltas,
             const TorchRAcc64<scalar_t, 1> ts,
             const TorchRAcc64<JOffsetsType, 1> jOffsets,
             const scalar_t tsmtThreshold,
             TorchRAcc64<int64_t, 1> outTotalSamples,
             TorchRAcc64<scalar_t, 1> outOpacity,
             TorchRAcc64<scalar_t, 1> outDepth,
             TorchRAcc64<scalar_t, 2> outRGB,
             TorchRAcc64<scalar_t, 1> outWs) {
    const int rayIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (rayIdx >= outOpacity.size(0)) {
        return;
    }
    const int numChannels = rgbs.size(1);
    volumeRenderFwdCallback<scalar_t, TorchRAcc64>(sigmas,
                                                   rgbs,
                                                   deltas,
                                                   ts,
                                                   jOffsets,
                                                   tsmtThreshold,
                                                   numChannels,
                                                   rayIdx,
                                                   outTotalSamples,
                                                   outOpacity,
                                                   outDepth,
                                                   outRGB,
                                                   outWs);
}

template <typename scalar_t>
__global__ __launch_bounds__(DEFAULT_BLOCK_DIM) void
volumeRenderBackward(const TorchRAcc64<scalar_t, 1> dLdOpacity,
                     const TorchRAcc64<scalar_t, 1> dLdDepth,
                     const TorchRAcc64<scalar_t, 2> dLdRgb,
                     const TorchRAcc64<scalar_t, 1> dLdWs,
                     const TorchRAcc64<scalar_t, 1> dLdWs_times_ws,
                     const TorchRAcc64<scalar_t, 1> sigmas,
                     const TorchRAcc64<scalar_t, 2> rgbs,
                     const TorchRAcc64<scalar_t, 1> deltas,
                     const TorchRAcc64<scalar_t, 1> ts,
                     const TorchRAcc64<JOffsetsType, 1> jOffsets,
                     const TorchRAcc64<scalar_t, 1> opacity,
                     const TorchRAcc64<scalar_t, 1> depth,
                     const TorchRAcc64<scalar_t, 2> rgb,
                     const scalar_t tsmtThreshold,
                     TorchRAcc64<scalar_t, 1> out_dL_dsigmas,
                     TorchRAcc64<scalar_t, 2> out_dLdRgbs) {
    const int rayIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (rayIdx >= opacity.size(0)) {
        return;
    }
    volumeRenderBwdCallback<torch::kCUDA, scalar_t, TorchRAcc64>(dLdOpacity,
                                                                 dLdDepth,
                                                                 dLdRgb,
                                                                 dLdWs,
                                                                 dLdWs_times_ws,
                                                                 sigmas,
                                                                 rgbs,
                                                                 deltas,
                                                                 ts,
                                                                 jOffsets,
                                                                 opacity,
                                                                 depth,
                                                                 rgb,
                                                                 tsmtThreshold,
                                                                 rayIdx,
                                                                 out_dL_dsigmas,
                                                                 out_dLdRgbs);
}

template <torch::DeviceType DeviceTag>
void dispatchVolumeRender(const torch::Tensor sigmas,
                          const torch::Tensor rgbs,
                          const torch::Tensor deltas,
                          const torch::Tensor ts,
                          const torch::Tensor jOffsets,
                          const float tsmtThreshold,
                          torch::Tensor &outOpacity,
                          torch::Tensor &outDepth,
                          torch::Tensor &outRgb,
                          torch::Tensor &outWs,
                          torch::Tensor &outTotalSamples);

template <torch::DeviceType DeviceTag>
void dispatchVolumeRenderBackward(const torch::Tensor dLdOpacity,
                                  const torch::Tensor dLdDepth,
                                  const torch::Tensor dLdRgb,
                                  const torch::Tensor dLdWs,
                                  const torch::Tensor sigmas,
                                  const torch::Tensor rgbs,
                                  const torch::Tensor ws,
                                  const torch::Tensor deltas,
                                  const torch::Tensor ts,
                                  const torch::Tensor jOffsets,
                                  const torch::Tensor opacity,
                                  const torch::Tensor depth,
                                  const torch::Tensor rgb,
                                  const float tsmtThreshold,
                                  torch::Tensor &outDLdSigmas,
                                  torch::Tensor &outDLdRbgs);

template <>
void
dispatchVolumeRender<torch::kCUDA>(
    const torch::Tensor sigmas,       // [B*R*S]
    const torch::Tensor rgbs,         // [B*R*S, C]
    const torch::Tensor deltas,       // [B*R*S]
    const torch::Tensor ts,           // [B*R*S]
    const torch::Tensor jOffsets,     // JaggedTensor joffsets for sigmas, rgbs, deltas, ts [B*R, 2]
    const float tsmtThreshold,
    torch::Tensor &outOpacity,        // [B*R]
    torch::Tensor &outDepth,          // [B*R]
    torch::Tensor &outRgb,            // [B*R, C]
    torch::Tensor &outWs,             // [B*R*S]
    torch::Tensor &outTotalSamples) { // [B*R]
    const int64_t numRays = jOffsets.size(0) - 1;
    const int64_t N       = sigmas.size(0);

    TORCH_CHECK(sigmas.device().is_cuda(), "sigmas must be a CUDA tensor");
    TORCH_CHECK(sigmas.device().has_index(), "sigmas must have CUDA index");
    TORCH_CHECK(sigmas.device() == rgbs.device(), "All tensors must be on the same device");
    TORCH_CHECK(sigmas.device() == deltas.device(), "All tensors must be on the same device");
    TORCH_CHECK(sigmas.device() == ts.device(), "All tensors must be on the same device");
    TORCH_CHECK(sigmas.device() == jOffsets.device(), "All tensors must be on the same device");
    TORCH_CHECK(sigmas.device() == outOpacity.device(), "All tensors must be on the same device");
    TORCH_CHECK(sigmas.device() == outDepth.device(), "All tensors must be on the same device");
    TORCH_CHECK(sigmas.device() == outRgb.device(), "All tensors must be on the same device");
    TORCH_CHECK(sigmas.device() == outWs.device(), "All tensors must be on the same device");
    TORCH_CHECK(sigmas.device() == outTotalSamples.device(),
                "All tensors must be on the same device");

    c10::cuda::CUDAGuard deviceGuard(sigmas.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(sigmas.device().index()).stream();

    // auto opacity = torch::zeros({numRays}, sigmas.options());
    // auto depth = torch::zeros({numRays}, sigmas.options());
    // auto depthSq = torch::zeros({numRays}, sigmas.options());
    // auto rgb = torch::zeros({numRays, 3}, sigmas.options());
    // auto ws = torch::zeros({N}, sigmas.options());
    // auto total_samples = torch::zeros({numRays},
    // torch::dtype(torch::kLong).device(sigmas.device()));

    const int64_t NUM_BLOCKS = GET_BLOCKS(numRays, DEFAULT_BLOCK_DIM);

    AT_DISPATCH_V2(
        sigmas.scalar_type(),
        "volumeRender",
        AT_WRAP([&] {
            volumeRender<scalar_t><<<NUM_BLOCKS, DEFAULT_BLOCK_DIM, 0, stream>>>(
                sigmas.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                rgbs.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                deltas.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                ts.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                jOffsets.packed_accessor64<JOffsetsType, 1, torch::RestrictPtrTraits>(),
                tsmtThreshold,
                outTotalSamples.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                outOpacity.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                outDepth.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                // outDepthSq.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                outRgb.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                outWs.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }),
        AT_EXPAND(AT_FLOATING_TYPES),
        c10::kHalf);

    // return {total_samples, opacity, depth, depthSq, rgb, ws};
}

template <>
void
dispatchVolumeRenderBackward<torch::kCUDA>(const torch::Tensor dLdOpacity,
                                           const torch::Tensor dLdDepth,
                                           // const torch::Tensor dLdDepthSq,
                                           const torch::Tensor dLdRgb,
                                           const torch::Tensor dLdWs,
                                           const torch::Tensor sigmas,
                                           const torch::Tensor rgbs,
                                           const torch::Tensor ws,
                                           const torch::Tensor deltas,
                                           const torch::Tensor ts,
                                           const torch::Tensor jOffsets,
                                           const torch::Tensor opacity,
                                           const torch::Tensor depth,
                                           // const torch::Tensor depthSq,
                                           const torch::Tensor rgb,
                                           const float tsmtThreshold,
                                           torch::Tensor &outDLdSigmas,
                                           torch::Tensor &outDLdRbgs) {
    TORCH_CHECK(dLdOpacity.device().is_cuda(), "dLdOpacity must be a CUDA tensor");
    TORCH_CHECK(dLdOpacity.device().has_index(), "dLdOpacity must have CUDA index");
    TORCH_CHECK(dLdOpacity.device() == dLdDepth.device(), "All tensors must be on the same device");
    TORCH_CHECK(dLdOpacity.device() == dLdRgb.device(), "All tensors must be on the same device");
    TORCH_CHECK(dLdOpacity.device() == dLdWs.device(), "All tensors must be on the same device");
    TORCH_CHECK(dLdOpacity.device() == rgbs.device(), "All tensors must be on the same device");
    TORCH_CHECK(dLdOpacity.device() == ws.device(), "All tensors must be on the same device");
    TORCH_CHECK(dLdOpacity.device() == deltas.device(), "All tensors must be on the same device");
    TORCH_CHECK(dLdOpacity.device() == ts.device(), "All tensors must be on the same device");
    TORCH_CHECK(dLdOpacity.device() == jOffsets.device(), "All tensors must be on the same device");
    TORCH_CHECK(dLdOpacity.device() == opacity.device(), "All tensors must be on the same device");
    TORCH_CHECK(dLdOpacity.device() == depth.device(), "All tensors must be on the same device");
    TORCH_CHECK(dLdOpacity.device() == rgb.device(), "All tensors must be on the same device");
    TORCH_CHECK(dLdOpacity.device() == outDLdSigmas.device(),
                "All tensors must be on the same device");
    TORCH_CHECK(dLdOpacity.device() == outDLdRbgs.device(),
                "All tensors must be on the same device");

    c10::cuda::CUDAGuard deviceGuard(dLdOpacity.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(dLdOpacity.device().index()).stream();

    const int64_t N       = sigmas.size(0);
    const int64_t numRays = jOffsets.size(0) - 1;

    // auto dL_dsigmas = torch::zeros({N}, sigmas.options());
    // auto dLdRgbs = torch::zeros({N, 3}, sigmas.options());

    torch::Tensor dLdWs_times_ws = (dLdWs * ws); // auxiliary input

    const int64_t NUM_BLOCKS = GET_BLOCKS(numRays, DEFAULT_BLOCK_DIM);

    AT_DISPATCH_V2(
        sigmas.scalar_type(),
        "volumeRenderBackward",
        AT_WRAP([&] {
            volumeRenderBackward<scalar_t><<<NUM_BLOCKS, DEFAULT_BLOCK_DIM, 0, stream>>>(
                dLdOpacity.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                dLdDepth.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                // dLdDepthSq.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                dLdRgb.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                dLdWs.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                dLdWs_times_ws.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                sigmas.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                rgbs.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                deltas.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                ts.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                jOffsets.packed_accessor64<JOffsetsType, 1, torch::RestrictPtrTraits>(),
                opacity.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                depth.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                // depthSq.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                rgb.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                tsmtThreshold,
                outDLdSigmas.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                outDLdRbgs.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }),
        AT_EXPAND(AT_FLOATING_TYPES),
        c10::kHalf);

    // return {dL_dsigmas, dLdRgbs};
}

template <>
void
dispatchVolumeRender<torch::kCPU>(const torch::Tensor sigmas,
                                  const torch::Tensor rgbs,
                                  const torch::Tensor deltas,
                                  const torch::Tensor ts,
                                  const torch::Tensor jOffsets,
                                  const float tsmtThreshold,
                                  torch::Tensor &outOpacity,
                                  torch::Tensor &outDepth,
                                  torch::Tensor &outRgb,
                                  torch::Tensor &outWs,
                                  torch::Tensor &outTotalSamples) {
    AT_DISPATCH_V2(sigmas.scalar_type(),
                   "volumeRender",
                   AT_WRAP([&] {
                       volumeRenderCPU<scalar_t>(sigmas.accessor<scalar_t, 1>(),
                                                 rgbs.accessor<scalar_t, 2>(),
                                                 deltas.accessor<scalar_t, 1>(),
                                                 ts.accessor<scalar_t, 1>(),
                                                 jOffsets.accessor<JOffsetsType, 1>(),
                                                 tsmtThreshold,
                                                 outTotalSamples.accessor<int64_t, 1>(),
                                                 outOpacity.accessor<scalar_t, 1>(),
                                                 outDepth.accessor<scalar_t, 1>(),
                                                 outRgb.accessor<scalar_t, 2>(),
                                                 outWs.accessor<scalar_t, 1>());
                   }),
                   AT_EXPAND(AT_FLOATING_TYPES),
                   c10::kHalf);
}

template <>
void
dispatchVolumeRenderBackward<torch::kCPU>(const torch::Tensor dLdOpacity,
                                          const torch::Tensor dLdDepth,
                                          const torch::Tensor dLdRgb,
                                          const torch::Tensor dLdWs,
                                          const torch::Tensor sigmas,
                                          const torch::Tensor rgbs,
                                          const torch::Tensor ws,
                                          const torch::Tensor deltas,
                                          const torch::Tensor ts,
                                          const torch::Tensor jOffsets,
                                          const torch::Tensor opacity,
                                          const torch::Tensor depth,
                                          const torch::Tensor rgb,
                                          const float tsmtThreshold,
                                          torch::Tensor &outDLdSigmas,
                                          torch::Tensor &outDLdRbgs) {
    torch::Tensor dLdWs_times_ws = (dLdWs * ws); // auxiliary input

    AT_DISPATCH_V2(sigmas.scalar_type(),
                   "volumeRenderBackward",
                   AT_WRAP([&] {
                       volumeRenderBackwardCPU<scalar_t>(dLdOpacity.accessor<scalar_t, 1>(),
                                                         dLdDepth.accessor<scalar_t, 1>(),
                                                         dLdRgb.accessor<scalar_t, 2>(),
                                                         dLdWs.accessor<scalar_t, 1>(),
                                                         dLdWs_times_ws.accessor<scalar_t, 1>(),
                                                         sigmas.accessor<scalar_t, 1>(),
                                                         rgbs.accessor<scalar_t, 2>(),
                                                         deltas.accessor<scalar_t, 1>(),
                                                         ts.accessor<scalar_t, 1>(),
                                                         jOffsets.accessor<JOffsetsType, 1>(),
                                                         opacity.accessor<scalar_t, 1>(),
                                                         depth.accessor<scalar_t, 1>(),
                                                         rgb.accessor<scalar_t, 2>(),
                                                         tsmtThreshold,
                                                         outDLdSigmas.accessor<scalar_t, 1>(),
                                                         outDLdRbgs.accessor<scalar_t, 2>());
                   }),
                   AT_EXPAND(AT_FLOATING_TYPES),
                   c10::kHalf);
}

} // anonymous namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
volumeRender(const torch::Tensor &sigmas,
             const torch::Tensor &rgbs,
             const torch::Tensor &deltaTs,
             const torch::Tensor &ts,
             const torch::Tensor &jOffsets,
             double tsmtThreshold) {
    const int64_t numRays = jOffsets.size(0) - 1;
    const int64_t N       = sigmas.size(0);

    TORCH_CHECK(jOffsets.dim() == 1, "jOffsets must have shape (nRays+1,)");
    TORCH_CHECK(sigmas.dim() == 1, "sigmas must have shape (nRays*nSamplesPerRay,)");
    TORCH_CHECK(rgbs.dim() == 2, "rgbs must have shape (nRays*nSamplesPerRay, numChannels)");
    TORCH_CHECK(rgbs.size(1) > 0 && rgbs.size(1) <= MAX_VOLUME_RENDER_CHANNELS,
                "rgbs must have between 1 and ",
                MAX_VOLUME_RENDER_CHANNELS,
                " channels (got ",
                rgbs.size(1),
                ")");
    TORCH_CHECK(deltaTs.dim() == 1, "deltaTs must have shape (nRays*nSamplesPerRay,)");
    TORCH_CHECK(ts.dim() == 1, "ts must have shape (N,)");

    TORCH_CHECK(sigmas.device() == rgbs.device(), "All tensors must be on the same device");
    TORCH_CHECK(sigmas.device() == deltaTs.device(), "All tensors must be on the same device");
    TORCH_CHECK(sigmas.device() == ts.device(), "All tensors must be on the same device");
    TORCH_CHECK(sigmas.device() == jOffsets.device(), "All tensors must be on the same device");

    TORCH_CHECK(sigmas.dtype() == rgbs.dtype(),
                "All floating point tensors must be on the same dtype");
    TORCH_CHECK(sigmas.dtype() == deltaTs.dtype(),
                "All floating point tensors must be on the same dtype");
    TORCH_CHECK(sigmas.dtype() == ts.dtype(),
                "All floating point tensors must be on the same dtype");
    TORCH_CHECK(jOffsets.dtype() == torch::dtype(JOffsetsScalarType).dtype(),
                "jOffsets must be of type torch.int32");

    TORCH_CHECK(sigmas.size(0) == rgbs.size(0),
                "sigmas and rgbs must have the same number of elements");
    TORCH_CHECK(sigmas.size(0) == deltaTs.size(0),
                "sigmas and deltaTs must have the same number of elements");
    TORCH_CHECK(sigmas.size(0) == ts.size(0),
                "sigmas and ts must have the same number of elements");

    torch::Tensor outOpacity = torch::zeros({numRays}, sigmas.options());
    torch::Tensor outDepth   = torch::zeros({numRays}, sigmas.options());
    torch::Tensor outRgb     = torch::zeros({numRays, rgbs.size(1)}, sigmas.options());
    torch::Tensor outWs      = torch::zeros({N}, sigmas.options());
    torch::Tensor outTotalSamples =
        torch::zeros({numRays}, torch::dtype(torch::kLong).device(sigmas.device()));

    FVDB_DISPATCH_KERNEL_DEVICE(sigmas.device(), [&]() {
        dispatchVolumeRender<DeviceTag>(sigmas,
                                        rgbs,
                                        deltaTs,
                                        ts,
                                        jOffsets,
                                        static_cast<float>(tsmtThreshold),
                                        outOpacity,
                                        outDepth,
                                        outRgb,
                                        outWs,
                                        outTotalSamples);
    });

    return {outRgb, outDepth, outOpacity, outWs, outTotalSamples};
}

std::tuple<torch::Tensor, torch::Tensor>
volumeRenderBackward(const torch::Tensor &dLdOpacity,
                     const torch::Tensor &dLdDepth,
                     const torch::Tensor &dLdRgb,
                     const torch::Tensor &dLdWs,
                     const torch::Tensor &sigmas,
                     const torch::Tensor &rgbs,
                     const torch::Tensor &ws,
                     const torch::Tensor &deltas,
                     const torch::Tensor &ts,
                     const torch::Tensor &jOffsets,
                     const torch::Tensor &opacity,
                     const torch::Tensor &depth,
                     const torch::Tensor &rgb,
                     float tsmtThreshold) {
    const int64_t N = sigmas.size(0);

    TORCH_CHECK(rgbs.dim() == 2,
                "rgbs must be a 2D tensor of shape (N, C), but got a ",
                rgbs.dim(),
                "D tensor.");
    TORCH_CHECK(dLdRgb.dim() == 2,
                "dLdRgb must be a 2D tensor of shape (numRays, C), but got a ",
                dLdRgb.dim(),
                "D tensor.");
    TORCH_CHECK(rgb.dim() == 2,
                "rgb must be a 2D tensor of shape (numRays, C), but got a ",
                rgb.dim(),
                "D tensor.");
    TORCH_CHECK(
        rgbs.size(1) > 0, "rgbs must have at least one channel, but got ", rgbs.size(1), ".");
    TORCH_CHECK(rgbs.size(1) <= MAX_VOLUME_RENDER_CHANNELS,
                "Volume rendering backward supports at most ",
                MAX_VOLUME_RENDER_CHANNELS,
                " channels, but got ",
                rgbs.size(1),
                ".");
    TORCH_CHECK(dLdRgb.size(1) == rgbs.size(1),
                "dLdRgb and rgbs must have the same channel dimension, but got ",
                dLdRgb.size(1),
                " and ",
                rgbs.size(1),
                ".");
    TORCH_CHECK(rgb.size(1) == rgbs.size(1),
                "rgb and rgbs must have the same channel dimension, but got ",
                rgb.size(1),
                " and ",
                rgbs.size(1),
                ".");

    torch::Tensor dLdSigmas = torch::zeros({N}, sigmas.options());
    torch::Tensor dLdRgbs   = torch::zeros({N, rgbs.size(1)}, sigmas.options());

    FVDB_DISPATCH_KERNEL_DEVICE(sigmas.device(), [&]() {
        dispatchVolumeRenderBackward<DeviceTag>(dLdOpacity,
                                                dLdDepth,
                                                dLdRgb,
                                                dLdWs,
                                                sigmas,
                                                rgbs,
                                                ws,
                                                deltas,
                                                ts,
                                                jOffsets,
                                                opacity,
                                                depth,
                                                rgb,
                                                tsmtThreshold,
                                                dLdSigmas,
                                                dLdRgbs);
    });

    return {dLdSigmas, dLdRgbs};
}

} // namespace ops
} // namespace detail
} // namespace fvdb
