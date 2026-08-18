// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/gsplat/RasterizeWorldSpaceGaussiansBackward.h>
#include <fvdb/detail/utils/Nvtx.h>
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/cuda/GradientReduction.h>
#include <fvdb/detail/utils/cuda/Prefetch.h>
#include <fvdb/detail/utils/cuda/Utils.cuh>
#include <fvdb/detail/utils/cuda/WarpReduce.cuh>
#include <fvdb/detail/utils/gsplat/GaussianRasterizeFromWorld.cuh>
#include <fvdb/detail/utils/gsplat/GaussianRasterizeOptionalInputs.h>

#include <nanovdb/util/cuda/Util.h>

#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/from_blob.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>

#include <cooperative_groups.h>

#include <cstdint>
#include <vector>

namespace fvdb::detail::ops {
namespace cg = cooperative_groups;

namespace {

template <uint32_t NUM_CHANNELS> struct SharedGaussian {
    int32_t id;                       // flattened id in [0, C*N)
    nanovdb::math::Vec3<float> mean;  // world mean
    nanovdb::math::Vec4<float> quat;  // wxyz
    nanovdb::math::Vec3<float> scale; // exp(log_scales)
    nanovdb::math::Mat3<float> isclR; // S^{-1} R^T
    float opacity;
};

template <uint32_t NUM_CHANNELS, typename Camera> struct RasterizeFromWorldBackwardArgs {
    RasterizeFromWorldCommonArgs commonArgs;
    Camera camera;
    uint32_t blockOffset;
    // Forward outputs
    fvdb::TorchRAcc64<float, 4> renderedAlphas; // [C,H,W,1]
    fvdb::TorchRAcc64<int32_t, 3> lastIds;      // [C,H,W]
    // Grad outputs
    fvdb::TorchRAcc64<float, 4> dLossDRenderedFeatures; // [C,H,W,D]
    fvdb::TorchRAcc64<float, 4> dLossDRenderedAlphas;   // [C,H,W,1]
    // Outputs (grads)
    fvdb::TorchRAcc64<float, 2> dMeans;     // [N,3]
    fvdb::TorchRAcc64<float, 2> dQuats;     // [N,4]
    fvdb::TorchRAcc64<float, 2> dLogScales; // [N,3]
    fvdb::TorchRAcc64<float, 3> dFeatures;  // [C,N,D]
    fvdb::TorchRAcc64<float, 2> dOpacities; // [C,N]
};

template <uint32_t NUM_CHANNELS, typename Camera>
__global__ void
rasterizeFromWorld3DGSBackwardKernel(
    const RasterizeFromWorldBackwardArgs<NUM_CHANNELS, Camera> args) {
    // Fully expanding every channel operation makes the 512/513-channel kernels prohibitively
    // expensive for ptxas to optimize, especially for SM 120. Preserve full unrolling for the
    // smaller kernels while keeping the generated code bounded for larger channel counts.
    constexpr uint32_t CHANNEL_UNROLL = NUM_CHANNELS <= 32 ? 32 : 4;

    auto block               = cg::this_thread_block();
    const uint32_t blockSize = blockDim.x * blockDim.y;
    const auto &common       = args.commonArgs;

    uint32_t camId, tileRow, tileCol, row, col;
    common.denseCoordinates(camId, tileRow, tileCol, row, col, args.blockOffset);
    const bool inside = (row < common.imageHeight && col < common.imageWidth);

    // Parity with classic rasterizer: masked tiles contribute nothing.
    //
    // IMPORTANT: this kernel uses block-level barriers later (`block.sync`). Any early return must
    // be taken by *all* threads in the block, otherwise edge tiles can deadlock when some threads
    // are `!inside`. So we make the return block-wide.
    const bool tileMasked = common.tileMasked(camId, tileRow, tileCol);
    if (tileMasked) {
        return;
    }

    extern __shared__ char smem[];
    Camera cameraLocal = args.camera;
    uintptr_t smemAddr = reinterpret_cast<uintptr_t>(smem);
    smemAddr           = alignUpAddress(smemAddr, alignof(nanovdb::math::Mat3<float>));
    cameraLocal.loadSharedMemory(reinterpret_cast<void *>(smemAddr));
    block.sync();

    const nanovdb::math::Ray<float> ray = cameraLocal.unprojectPixelToRay(camId, row, col);

    // Whether this pixel participates in the backward pass.
    //
    // NOTE: We must *not* early-return for `!inside` because the kernel uses `block.sync` later.
    const bool rayValid = ray.dir().dot(ray.dir()) > 0.0f;
    const bool done     = inside && rayValid;

    // Gaussian range for this tile.
    const auto [rangeStart, rangeEnd] = common.tileGaussianRange(camId, tileRow, tileCol);

    // If the tile has no intersections, there is nothing to do. This must be a block-wide return.
    if (rangeEnd <= rangeStart) {
        return;
    }

    // Forward state for this pixel.
    int32_t lastIntersectionOffset = -1;
    float T_final                  = 1.0f;
    float T                        = 1.0f;

    float v_render_c[NUM_CHANNELS];
#pragma unroll CHANNEL_UNROLL
    for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
        v_render_c[k] = 0.f;
    }
    float v_render_a = 0.f;

    if (done) {
        lastIntersectionOffset = args.lastIds[camId][row][col];

        const float alphaFinal = args.renderedAlphas[camId][row][col][0];
        T_final                = 1.0f - alphaFinal;
        T                      = T_final;

#pragma unroll CHANNEL_UNROLL
        for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
            v_render_c[k] = args.dLossDRenderedFeatures[camId][row][col][k];
        }
        v_render_a = args.dLossDRenderedAlphas[camId][row][col][0];
    }

    float buffer[NUM_CHANNELS];
#pragma unroll CHANNEL_UNROLL
    for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
        buffer[k] = 0.f;
    }

    // Shared memory for gaussian batches (after camera-op shared state).
    uintptr_t gaussAddr = smemAddr + cameraLocal.numSharedMemBytes();
    gaussAddr           = alignUpAddress(gaussAddr, alignof(int32_t));
    int32_t *idBatch    = reinterpret_cast<int32_t *>(gaussAddr);               // [blockSize]
    gaussAddr += static_cast<size_t>(blockSize) * sizeof(int32_t);
    gaussAddr    = alignUpAddress(gaussAddr, alignof(SharedGaussian<NUM_CHANNELS>));
    auto *gBatch = reinterpret_cast<SharedGaussian<NUM_CHANNELS> *>(gaussAddr); // [blockSize]

    const uint32_t threadRank      = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    const int64_t nIsects  = rangeEnd - rangeStart;
    const int64_t nBatches = (nIsects + blockSize - 1) / blockSize;

    // Reduce the last tile-relative intersection offset within the warp for early skipping.
    const int32_t lastIntersectionOffsetInWarp =
        cg::reduce(warp, lastIntersectionOffset, cg::greater<int32_t>());

    for (int64_t b = 0; b < nBatches; ++b) {
        block.sync();
        const int64_t batchEndOffset = nIsects - 1 - blockSize * b;
        const int64_t remaining      = batchEndOffset + 1;
        const uint32_t batchSize =
            static_cast<uint32_t>(remaining < blockSize ? remaining : blockSize);
        const int64_t idx = rangeStart + batchEndOffset - threadRank;

        if (idx >= rangeStart) {
            const int32_t flatId = common.tileGaussianIds[idx];
            idBatch[threadRank]  = flatId;
            const int32_t gid    = flatId % (int32_t)common.means.size(0);
            const int32_t cid    = flatId / (int32_t)common.means.size(0);

            const nanovdb::math::Vec3<float> mean_w(
                common.means[gid][0], common.means[gid][1], common.means[gid][2]);
            const nanovdb::math::Vec4<float> quat_wxyz(common.quats[gid][0],
                                                       common.quats[gid][1],
                                                       common.quats[gid][2],
                                                       common.quats[gid][3]);
            const nanovdb::math::Vec3<float> scale(__expf(common.logScales[gid][0]),
                                                   __expf(common.logScales[gid][1]),
                                                   __expf(common.logScales[gid][2]));
            const nanovdb::math::Mat3<float> isclR = computeIsclRot<float>(quat_wxyz, scale);
            const float op                         = common.opacities[cid][gid];

            gBatch[threadRank].id      = flatId;
            gBatch[threadRank].mean    = mean_w;
            gBatch[threadRank].quat    = quat_wxyz;
            gBatch[threadRank].scale   = scale;
            gBatch[threadRank].isclR   = isclR;
            gBatch[threadRank].opacity = op;
        }

        block.sync();

        // Process gaussians in this batch, from back-to-front.
        const int64_t startT64 = batchEndOffset > lastIntersectionOffsetInWarp
                                     ? batchEndOffset - lastIntersectionOffsetInWarp
                                     : 0;
        const uint32_t startT  = startT64 < batchSize ? static_cast<uint32_t>(startT64) : batchSize;
        for (uint32_t t = startT; t < batchSize; ++t) {
            bool valid = done;
            if (batchEndOffset - t > lastIntersectionOffset) {
                valid = false;
            }

            float alpha = 0.f;
            float opac  = 0.f;
            float vis   = 0.f;

            nanovdb::math::Vec3<float> mean_w(0.f);
            nanovdb::math::Vec4<float> quat_wxyz(1.f, 0.f, 0.f, 0.f);
            nanovdb::math::Vec3<float> scale(1.f);
            nanovdb::math::Mat3<float> Mt;
            nanovdb::math::Vec3<float> o_minus_mu(0.f);
            nanovdb::math::Vec3<float> gro(0.f), grd(0.f), grd_n(0.f), gcrod(0.f);
            float grayDist = 0.f;

            if (valid) {
                const SharedGaussian<NUM_CHANNELS> g = gBatch[t];
                mean_w                               = g.mean;
                quat_wxyz                            = g.quat;
                scale                                = g.scale;
                Mt                                   = g.isclR;
                opac                                 = g.opacity;

                o_minus_mu        = ray.eye() - mean_w;
                gro               = Mt * o_minus_mu;
                grd               = Mt * ray.dir();
                grd_n             = fvdb::detail::ops::normalizeSafe<float>(grd);
                gcrod             = grd_n.cross(gro);
                grayDist          = gcrod.dot(gcrod);
                const float power = -0.5f * grayDist;
                vis               = __expf(power);
                alpha             = min(kAlphaThreshold, opac * vis);
                if (power > 0.f || alpha < 1.f / 255.f) {
                    valid = false;
                }
            }

            if (!warp.any(valid)) {
                continue;
            }

            float v_feat_local[NUM_CHANNELS];
#pragma unroll CHANNEL_UNROLL
            for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
                v_feat_local[k] = 0.f;
            }

            nanovdb::math::Vec3<float> v_mean_local(0.f);
            float v_quat_local[4] = {0.f, 0.f, 0.f, 0.f};
            nanovdb::math::Vec3<float> v_logscale_local(0.f);
            float v_opacity_local = 0.f;

            if (valid) {
                const float ra = 1.0f / (1.0f - alpha);
                T *= ra;

                const float fac = alpha * T;
#pragma unroll CHANNEL_UNROLL
                for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
                    v_feat_local[k] = fac * v_render_c[k];
                }

                // v_alpha accumulation
                float v_alpha        = 0.f;
                const int32_t flatId = idBatch[t];
                const int32_t cid    = flatId / (int32_t)common.means.size(0);
                const int32_t gid    = flatId % (int32_t)common.means.size(0);

#pragma unroll CHANNEL_UNROLL
                for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
                    const float c = common.features[cid][gid][k];
                    v_alpha += (c * T - buffer[k] * ra) * v_render_c[k];
                }

                v_alpha += T_final * ra * v_render_a;

                if (common.backgrounds != nullptr) {
                    float accum = 0.f;
#pragma unroll CHANNEL_UNROLL
                    for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
                        accum += common.backgroundValue(camId, k) * v_render_c[k];
                    }
                    v_alpha += -T_final * ra * accum;
                }

                if (opac * vis <= kAlphaThreshold) {
                    const float v_vis                        = opac * v_alpha;
                    const float v_gradDist                   = -0.5f * vis * v_vis;
                    const nanovdb::math::Vec3<float> v_gcrod = 2.0f * v_gradDist * gcrod;
                    const nanovdb::math::Vec3<float> v_grd_n = -(v_gcrod.cross(gro));
                    const nanovdb::math::Vec3<float> v_gro   = v_gcrod.cross(grd_n);

                    const nanovdb::math::Vec3<float> v_grd = normalizeSafeVJP<float>(grd, v_grd_n);

                    // v_Mt = outer(v_grd, ray.dir) + outer(v_gro, (ray.eye() - mean))
                    const nanovdb::math::Vec3<float> rayDir = ray.dir();
                    nanovdb::math::Mat3<float> v_Mt =
                        nanovdb::math::Mat3<float>(v_grd[0] * rayDir[0],
                                                   v_grd[0] * rayDir[1],
                                                   v_grd[0] * rayDir[2],
                                                   v_grd[1] * rayDir[0],
                                                   v_grd[1] * rayDir[1],
                                                   v_grd[1] * rayDir[2],
                                                   v_grd[2] * rayDir[0],
                                                   v_grd[2] * rayDir[1],
                                                   v_grd[2] * rayDir[2]);
                    v_Mt += nanovdb::math::Mat3<float>(v_gro[0] * o_minus_mu[0],
                                                       v_gro[0] * o_minus_mu[1],
                                                       v_gro[0] * o_minus_mu[2],
                                                       v_gro[1] * o_minus_mu[0],
                                                       v_gro[1] * o_minus_mu[1],
                                                       v_gro[1] * o_minus_mu[2],
                                                       v_gro[2] * o_minus_mu[0],
                                                       v_gro[2] * o_minus_mu[1],
                                                       v_gro[2] * o_minus_mu[2]);

                    const nanovdb::math::Vec3<float> v_o_minus_mu = Mt.transpose() * v_gro;
                    v_mean_local += -v_o_minus_mu;

                    nanovdb::math::Vec4<float> dQuat(0.f);
                    nanovdb::math::Vec3<float> dLogScale(0.f);
                    isclRotVectorJacobianProduct<float>(quat_wxyz, scale, v_Mt, dQuat, dLogScale);
                    v_quat_local[0] += dQuat[0];
                    v_quat_local[1] += dQuat[1];
                    v_quat_local[2] += dQuat[2];
                    v_quat_local[3] += dQuat[3];
                    v_logscale_local += dLogScale;

                    v_opacity_local = vis * v_alpha;
                }

                // buffer update
#pragma unroll CHANNEL_UNROLL
                for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
                    const int32_t flatId = idBatch[t];
                    const int32_t cid    = flatId / (int32_t)common.means.size(0);
                    const int32_t gid    = flatId % (int32_t)common.means.size(0);
                    buffer[k] += common.features[cid][gid][k] * fac;
                }
            }

            // Warp-reduce and atomic add once per gaussian per warp.
            warpSumMut(v_opacity_local, warp);
            warpSumMut(v_mean_local, warp);
            warpSumMut<4>(v_quat_local, warp);
            warpSumMut(v_logscale_local, warp);
            if constexpr (NUM_CHANNELS <= 32) {
                warpSumMut<NUM_CHANNELS>(v_feat_local, warp);
            } else {
                warpSumMut(v_feat_local, NUM_CHANNELS, warp);
            }

            if (warp.thread_rank() == 0) {
                const int32_t flatId = idBatch[t];
                const int32_t cid    = flatId / (int32_t)common.means.size(0);
                const int32_t gid    = flatId % (int32_t)common.means.size(0);

                // Per-camera grads
                float *dFeaturesGaussianPtr = args.dFeatures.data() +
                                              cid * args.dFeatures.stride(0) +
                                              gid * args.dFeatures.stride(1);
#pragma unroll CHANNEL_UNROLL
                for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
                    atomicAdd(dFeaturesGaussianPtr + k * args.dFeatures.stride(2), v_feat_local[k]);
                }
                float *dOpacityGaussianPtr = args.dOpacities.data() +
                                             cid * args.dOpacities.stride(0) +
                                             gid * args.dOpacities.stride(1);
                atomicAdd(dOpacityGaussianPtr, v_opacity_local);

                // Geometry grads (shared across cameras)
                float *dMeansPtr = args.dMeans.data() + gid * args.dMeans.stride(0);
                atomicAdd(dMeansPtr + 0 * args.dMeans.stride(1), v_mean_local[0]);
                atomicAdd(dMeansPtr + 1 * args.dMeans.stride(1), v_mean_local[1]);
                atomicAdd(dMeansPtr + 2 * args.dMeans.stride(1), v_mean_local[2]);

                float *dQuatsPtr = args.dQuats.data() + gid * args.dQuats.stride(0);
                atomicAdd(dQuatsPtr + 0 * args.dQuats.stride(1), v_quat_local[0]);
                atomicAdd(dQuatsPtr + 1 * args.dQuats.stride(1), v_quat_local[1]);
                atomicAdd(dQuatsPtr + 2 * args.dQuats.stride(1), v_quat_local[2]);
                atomicAdd(dQuatsPtr + 3 * args.dQuats.stride(1), v_quat_local[3]);

                float *dLogScalesPtr = args.dLogScales.data() + gid * args.dLogScales.stride(0);
                atomicAdd(dLogScalesPtr + 0 * args.dLogScales.stride(1), v_logscale_local[0]);
                atomicAdd(dLogScalesPtr + 1 * args.dLogScales.stride(1), v_logscale_local[1]);
                atomicAdd(dLogScalesPtr + 2 * args.dLogScales.stride(1), v_logscale_local[2]);
            }
        }
    }
}

template <uint32_t NUM_CHANNELS, typename Camera>
void
launchBackwardKernel(const torch::Tensor &means,
                     const torch::Tensor &quats,
                     const torch::Tensor &logScales,
                     const torch::Tensor &features,
                     const torch::Tensor &opacities,
                     const Camera &camera,
                     const uint32_t imageWidth,
                     const uint32_t imageHeight,
                     const uint32_t imageOriginW,
                     const uint32_t imageOriginH,
                     const uint32_t tileSize,
                     const torch::Tensor &tileOffsets,
                     const torch::Tensor &tileGaussianIds,
                     const torch::Tensor &renderedAlphas,
                     const torch::Tensor &lastIds,
                     const torch::Tensor &dLossDRenderedFeatures,
                     const torch::Tensor &dLossDRenderedAlphas,
                     const at::optional<torch::Tensor> &backgrounds,
                     const at::optional<torch::Tensor> &masks,
                     const torch::Tensor &dMeans,
                     const torch::Tensor &dQuats,
                     const torch::Tensor &dLogScales,
                     const torch::Tensor &dFeatures,
                     const torch::Tensor &dOpacities,
                     const uint32_t blockOffset,
                     const uint32_t blockCount,
                     const cudaStream_t stream) {
    const int64_t C = features.size(0);

    const uint32_t tileExtentW = (imageWidth + tileSize - 1) / tileSize;
    const uint32_t tileExtentH = (imageHeight + tileSize - 1) / tileSize;
    const dim3 blockDim(tileSize, tileSize, 1);
    const dim3 gridDim(blockCount, 1, 1);
    const int64_t totalIntersections = tileGaussianIds.size(0);

    RasterizeFromWorldCommonArgs args{
        imageWidth,
        imageHeight,
        imageOriginW,
        imageOriginH,
        tileSize,
        tileExtentW,
        tileExtentH,
        NUM_CHANNELS,
        totalIntersections,
        tileOffsets.packed_accessor64<int64_t, 3, torch::RestrictPtrTraits>(),
        tileGaussianIds.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
        nullptr,
        nullptr,
        means.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
        quats.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
        logScales.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
        features.packed_accessor64<float, 3, torch::RestrictPtrTraits>(),
        opacities.packed_accessor64<float, 2, torch::RestrictPtrTraits>()};

    const PreparedRasterOptionalInputs opt = prepareRasterOptionalInputs(
        features, C, tileExtentH, tileExtentW, (int64_t)NUM_CHANNELS, backgrounds, masks);
    args.backgrounds = opt.backgrounds;
    args.masks       = opt.masks;

    const size_t blockSize = (size_t)tileSize * (size_t)tileSize;
    size_t sharedMem       = (alignof(nanovdb::math::Mat3<float>) - 1) + camera.numSharedMemBytes();
    sharedMem += (alignof(int32_t) - 1) + blockSize * sizeof(int32_t);
    sharedMem += (alignof(SharedGaussian<NUM_CHANNELS>) - 1) +
                 blockSize * sizeof(SharedGaussian<NUM_CHANNELS>);

    RasterizeFromWorldBackwardArgs<NUM_CHANNELS, Camera> kernelArgs{
        args,
        camera,
        blockOffset,
        renderedAlphas.packed_accessor64<float, 4, torch::RestrictPtrTraits>(),
        lastIds.packed_accessor64<int32_t, 3, torch::RestrictPtrTraits>(),
        dLossDRenderedFeatures.packed_accessor64<float, 4, torch::RestrictPtrTraits>(),
        dLossDRenderedAlphas.packed_accessor64<float, 4, torch::RestrictPtrTraits>(),
        dMeans.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
        dQuats.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
        dLogScales.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
        dFeatures.packed_accessor64<float, 3, torch::RestrictPtrTraits>(),
        dOpacities.packed_accessor64<float, 2, torch::RestrictPtrTraits>()};

    if (cudaFuncSetAttribute(rasterizeFromWorld3DGSBackwardKernel<NUM_CHANNELS, Camera>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             sharedMem) != cudaSuccess) {
        AT_ERROR("Failed to set maximum shared memory size (requested ",
                 sharedMem,
                 " bytes), try lowering tile size or camera batch size.");
    }

    rasterizeFromWorld3DGSBackwardKernel<NUM_CHANNELS, Camera>
        <<<gridDim, blockDim, sharedMem, stream>>>(kernelArgs);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <uint32_t NUM_CHANNELS, typename Camera>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
launchBackwardCUDA(const torch::Tensor &means,
                   const torch::Tensor &quats,
                   const torch::Tensor &logScales,
                   const torch::Tensor &features,
                   const torch::Tensor &opacities,
                   const Camera &camera,
                   const uint32_t imageWidth,
                   const uint32_t imageHeight,
                   const uint32_t imageOriginW,
                   const uint32_t imageOriginH,
                   const uint32_t tileSize,
                   const torch::Tensor &tileOffsets,
                   const torch::Tensor &tileGaussianIds,
                   const torch::Tensor &renderedAlphas,
                   const torch::Tensor &lastIds,
                   const torch::Tensor &dLossDRenderedFeatures,
                   const torch::Tensor &dLossDRenderedAlphas,
                   const at::optional<torch::Tensor> &backgrounds,
                   const at::optional<torch::Tensor> &masks) {
    torch::Tensor dMeans     = torch::zeros_like(means);
    torch::Tensor dQuats     = torch::zeros_like(quats);
    torch::Tensor dLogScales = torch::zeros_like(logScales);
    torch::Tensor dFeatures  = torch::zeros_like(features);
    torch::Tensor dOpacities = torch::zeros_like(opacities);

    const uint32_t tileExtentW = (imageWidth + tileSize - 1) / tileSize;
    const uint32_t tileExtentH = (imageHeight + tileSize - 1) / tileSize;
    const uint32_t blockCount  = features.size(0) * tileExtentH * tileExtentW;
    if (blockCount > 0) {
        auto stream = at::cuda::getCurrentCUDAStream(means.device().index());
        launchBackwardKernel<NUM_CHANNELS, Camera>(means,
                                                   quats,
                                                   logScales,
                                                   features,
                                                   opacities,
                                                   camera,
                                                   imageWidth,
                                                   imageHeight,
                                                   imageOriginW,
                                                   imageOriginH,
                                                   tileSize,
                                                   tileOffsets,
                                                   tileGaussianIds,
                                                   renderedAlphas,
                                                   lastIds,
                                                   dLossDRenderedFeatures,
                                                   dLossDRenderedAlphas,
                                                   backgrounds,
                                                   masks,
                                                   dMeans,
                                                   dQuats,
                                                   dLogScales,
                                                   dFeatures,
                                                   dOpacities,
                                                   0,
                                                   blockCount,
                                                   stream);
    }
    return {dMeans, dQuats, dLogScales, dFeatures, dOpacities};
}

template <uint32_t NUM_CHANNELS, typename Camera>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
launchBackwardPrivateUse1(const torch::Tensor &means,
                          const torch::Tensor &quats,
                          const torch::Tensor &logScales,
                          const torch::Tensor &features,
                          const torch::Tensor &opacities,
                          const Camera &camera,
                          const uint32_t imageWidth,
                          const uint32_t imageHeight,
                          const uint32_t imageOriginW,
                          const uint32_t imageOriginH,
                          const uint32_t tileSize,
                          const torch::Tensor &tileOffsets,
                          const torch::Tensor &tileGaussianIds,
                          const torch::Tensor &renderedAlphas,
                          const torch::Tensor &lastIds,
                          const torch::Tensor &dLossDRenderedFeatures,
                          const torch::Tensor &dLossDRenderedAlphas,
                          const at::optional<torch::Tensor> &backgrounds,
                          const at::optional<torch::Tensor> &masks) {
    TORCH_CHECK(tileSize > 0, "Tile size must be greater than 0");

    const uint32_t tileExtentW = (imageWidth + tileSize - 1) / tileSize;
    const uint32_t tileExtentH = (imageHeight + tileSize - 1) / tileSize;
    const uint32_t tileCount   = features.size(0) * tileExtentH * tileExtentW;
    if (means.numel() == 0 || tileGaussianIds.numel() == 0 || tileCount == 0) {
        return {torch::zeros_like(means),
                torch::zeros_like(quats),
                torch::zeros_like(logScales),
                torch::zeros_like(features),
                torch::zeros_like(opacities)};
    }

    // Each GPU accumulates into device-local buffers. The managed outputs are populated after the
    // local gradients have been reduced across devices.
    torch::Tensor dMeans     = torch::empty(means.sizes(), means.options());
    torch::Tensor dQuats     = torch::empty(quats.sizes(), quats.options());
    torch::Tensor dLogScales = torch::empty(logScales.sizes(), logScales.options());
    torch::Tensor dFeatures  = torch::empty(features.sizes(), features.options());
    torch::Tensor dOpacities = torch::empty(opacities.sizes(), opacities.options());

    const at::optional<torch::Tensor> contiguousBackgrounds =
        backgrounds.has_value() ? std::make_optional(backgrounds.value().contiguous())
                                : std::nullopt;
    const at::optional<torch::Tensor> contiguousMasks =
        masks.has_value() ? std::make_optional(masks.value().contiguous()) : std::nullopt;

    const auto deviceCount = c10::cuda::device_count();
    TORCH_CHECK(deviceCount > 0, "PrivateUse1 rasterization requires at least one CUDA device");
    std::vector<cudaEvent_t> events(deviceCount);
    std::vector<float *> dMeansLocalPtrs(deviceCount, nullptr);
    std::vector<float *> dQuatsLocalPtrs(deviceCount, nullptr);
    std::vector<float *> dLogScalesLocalPtrs(deviceCount, nullptr);
    std::vector<float *> dFeaturesLocalPtrs(deviceCount, nullptr);
    std::vector<float *> dOpacitiesLocalPtrs(deviceCount, nullptr);
    std::vector<torch::Tensor> dMeansLocals(deviceCount);
    std::vector<torch::Tensor> dQuatsLocals(deviceCount);
    std::vector<torch::Tensor> dLogScalesLocals(deviceCount);
    std::vector<torch::Tensor> dFeaturesLocals(deviceCount);
    std::vector<torch::Tensor> dOpacitiesLocals(deviceCount);

    for (const auto deviceId: c10::irange(deviceCount)) {
        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        auto stream = c10::cuda::getCurrentCUDAStream(deviceId);
        C10_CUDA_CHECK(cudaEventCreateWithFlags(&events[deviceId], cudaEventDisableTiming));
        C10_CUDA_CHECK(cudaEventRecord(events[deviceId], stream));
    }

    std::vector<torch::Tensor> tileTensors = {
        renderedAlphas, lastIds, dLossDRenderedFeatures, dLossDRenderedAlphas};
    if (contiguousMasks.has_value()) {
        tileTensors.emplace_back(contiguousMasks.value());
    }

    for (const auto deviceId: c10::irange(deviceCount)) {
        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        auto stream = c10::cuda::getStreamFromPool(false, deviceId);
        C10_CUDA_CHECK(cudaStreamWaitEvent(stream, events[deviceId]));

        const auto [deviceTileOffset, deviceTileCount] = deviceChunk(tileCount, deviceId);
        if (deviceTileCount > 0) {
            std::vector<void *> prefetchPointers;
            std::vector<size_t> prefetchSizes;
            const TilePrefetchRange tileRange{static_cast<uint32_t>(deviceTileOffset),
                                              static_cast<uint32_t>(deviceTileCount),
                                              tileExtentH,
                                              tileExtentW,
                                              imageHeight,
                                              imageWidth,
                                              tileSize};
            appendPerTilePrefetchRanges(prefetchPointers, prefetchSizes, tileTensors, tileRange);
            memPrefetchBatchAsync(prefetchPointers, prefetchSizes, deviceId, stream);
        }
        C10_CUDA_CHECK(cudaEventRecord(events[deviceId], stream));
    }

    for (const auto deviceId: c10::irange(deviceCount)) {
        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        auto stream = c10::cuda::getCurrentCUDAStream(deviceId);
        C10_CUDA_CHECK(cudaStreamWaitEvent(stream, events[deviceId]));
        C10_CUDA_CHECK(cudaEventDestroy(events[deviceId]));

        const auto [deviceTileOffset, deviceTileCount] = deviceChunk(tileCount, deviceId);
        const auto localTensorOptions =
            at::TensorOptions().dtype(means.scalar_type()).device(at::kCUDA, deviceId);

        const auto allocateLocalGradient = [&](const torch::Tensor &like, float *&ptr) {
            const size_t numBytes = like.numel() * like.element_size();
            C10_CUDA_CHECK(cudaMallocAsync(&ptr, numBytes, stream));
            C10_CUDA_CHECK(cudaMemsetAsync(ptr, 0, numBytes, stream));
            return at::from_blob(ptr, like.sizes(), localTensorOptions);
        };

        dMeansLocals[deviceId] = allocateLocalGradient(means, dMeansLocalPtrs[deviceId]);
        dQuatsLocals[deviceId] = allocateLocalGradient(quats, dQuatsLocalPtrs[deviceId]);
        dLogScalesLocals[deviceId] =
            allocateLocalGradient(logScales, dLogScalesLocalPtrs[deviceId]);
        dFeaturesLocals[deviceId] = allocateLocalGradient(features, dFeaturesLocalPtrs[deviceId]);
        dOpacitiesLocals[deviceId] =
            allocateLocalGradient(opacities, dOpacitiesLocalPtrs[deviceId]);

        if (deviceTileCount > 0) {
            launchBackwardKernel<NUM_CHANNELS, Camera>(means,
                                                       quats,
                                                       logScales,
                                                       features,
                                                       opacities,
                                                       camera,
                                                       imageWidth,
                                                       imageHeight,
                                                       imageOriginW,
                                                       imageOriginH,
                                                       tileSize,
                                                       tileOffsets,
                                                       tileGaussianIds,
                                                       renderedAlphas,
                                                       lastIds,
                                                       dLossDRenderedFeatures,
                                                       dLossDRenderedAlphas,
                                                       contiguousBackgrounds,
                                                       contiguousMasks,
                                                       dMeansLocals[deviceId],
                                                       dQuatsLocals[deviceId],
                                                       dLogScalesLocals[deviceId],
                                                       dFeaturesLocals[deviceId],
                                                       dOpacitiesLocals[deviceId],
                                                       static_cast<uint32_t>(deviceTileOffset),
                                                       static_cast<uint32_t>(deviceTileCount),
                                                       stream);
        }
    }

    reduceGradientShards<float>(dMeansLocals, dMeans);
    reduceGradientShards<float>(dQuatsLocals, dQuats);
    reduceGradientShards<float>(dLogScalesLocals, dLogScales);
    reduceGradientShards<float>(dFeaturesLocals, dFeatures);
    reduceGradientShards<float>(dOpacitiesLocals, dOpacities);

    for (const auto deviceId: c10::irange(deviceCount)) {
        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        auto stream = c10::cuda::getCurrentCUDAStream(deviceId);
        C10_CUDA_CHECK(cudaFreeAsync(dMeansLocalPtrs[deviceId], stream));
        C10_CUDA_CHECK(cudaFreeAsync(dQuatsLocalPtrs[deviceId], stream));
        C10_CUDA_CHECK(cudaFreeAsync(dLogScalesLocalPtrs[deviceId], stream));
        C10_CUDA_CHECK(cudaFreeAsync(dFeaturesLocalPtrs[deviceId], stream));
        C10_CUDA_CHECK(cudaFreeAsync(dOpacitiesLocalPtrs[deviceId], stream));
    }

    mergeStreams();
    return {dMeans, dQuats, dLogScales, dFeatures, dOpacities};
}

} // namespace

template <torch::DeviceType>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianRasterizeFromWorld3DGSBackward(const torch::Tensor &means,
                                               const torch::Tensor &quats,
                                               const torch::Tensor &logScales,
                                               const torch::Tensor &features,
                                               const torch::Tensor &opacities,
                                               const torch::Tensor &worldToCamMatricesStart,
                                               const torch::Tensor &worldToCamMatricesEnd,
                                               const torch::Tensor &projectionMatrices,
                                               const torch::Tensor &distortionCoeffs,
                                               RollingShutterType rollingShutterType,
                                               DistortionModel cameraModel,
                                               uint32_t imageWidth,
                                               uint32_t imageHeight,
                                               uint32_t imageOriginW,
                                               uint32_t imageOriginH,
                                               uint32_t tileSize,
                                               const torch::Tensor &tileOffsets,
                                               const torch::Tensor &tileGaussianIds,
                                               const torch::Tensor &renderedAlphas,
                                               const torch::Tensor &lastIds,
                                               const torch::Tensor &dLossDRenderedFeatures,
                                               const torch::Tensor &dLossDRenderedAlphas,
                                               const at::optional<torch::Tensor> &backgrounds,
                                               const at::optional<torch::Tensor> &masks);

template <>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianRasterizeFromWorld3DGSBackward<torch::kCUDA>(
    const torch::Tensor &means,
    const torch::Tensor &quats,
    const torch::Tensor &logScales,
    const torch::Tensor &features,
    const torch::Tensor &opacities,
    const torch::Tensor &worldToCamMatricesStart,
    const torch::Tensor &worldToCamMatricesEnd,
    const torch::Tensor &projectionMatrices,
    const torch::Tensor &distortionCoeffs,
    const RollingShutterType rollingShutterType,
    const DistortionModel cameraModel,
    const uint32_t imageWidth,
    const uint32_t imageHeight,
    const uint32_t imageOriginW,
    const uint32_t imageOriginH,
    const uint32_t tileSize,
    const torch::Tensor &tileOffsets,
    const torch::Tensor &tileGaussianIds,
    const torch::Tensor &renderedAlphas,
    const torch::Tensor &lastIds,
    const torch::Tensor &dLossDRenderedFeatures,
    const torch::Tensor &dLossDRenderedAlphas,
    const at::optional<torch::Tensor> &backgrounds,
    const at::optional<torch::Tensor> &masks) {
    FVDB_FUNC_RANGE();

    const at::cuda::OptionalCUDAGuard device_guard(device_of(means));

    TORCH_CHECK_VALUE(means.is_cuda(), "means must be CUDA");
    TORCH_CHECK_VALUE(features.is_cuda(), "features must be CUDA");
    TORCH_CHECK_VALUE(opacities.is_cuda(), "opacities must be CUDA");
    TORCH_CHECK_VALUE(renderedAlphas.is_cuda(), "renderedAlphas must be CUDA");
    TORCH_CHECK_VALUE(lastIds.is_cuda(), "lastIds must be CUDA");
    TORCH_CHECK_VALUE(tileOffsets.scalar_type() == torch::kInt64,
                      "tileOffsets must have dtype int64");
    TORCH_CHECK_VALUE(lastIds.scalar_type() == torch::kInt32, "lastIds must have dtype int32");

    const auto checkFloat32 = [](const torch::Tensor &tensor, const char *name) {
        TORCH_CHECK_VALUE(
            tensor.scalar_type() == torch::kFloat32, name, " must have dtype float32");
    };
    checkFloat32(means, "means");
    checkFloat32(quats, "quats");
    checkFloat32(logScales, "logScales");
    checkFloat32(features, "features");
    checkFloat32(opacities, "opacities");
    checkFloat32(worldToCamMatricesStart, "worldToCamMatricesStart");
    checkFloat32(worldToCamMatricesEnd, "worldToCamMatricesEnd");
    checkFloat32(projectionMatrices, "projectionMatrices");
    checkFloat32(distortionCoeffs, "distortionCoeffs");
    checkFloat32(renderedAlphas, "renderedAlphas");
    checkFloat32(dLossDRenderedFeatures, "dLossDRenderedFeatures");
    checkFloat32(dLossDRenderedAlphas, "dLossDRenderedAlphas");

    const int64_t C = features.size(0);
    const int64_t N = means.size(0);

    TORCH_CHECK_VALUE(opacities.dim() == 2, "opacities must have shape [C,N]");
    TORCH_CHECK_VALUE(opacities.size(0) == C && opacities.size(1) == N,
                      "opacities must have shape [C,N] matching features and N");

    const uint32_t channels = (uint32_t)features.size(2);

#define CALL_BWD_WITH_OP(NCH, OP_TYPE, OP_VAL)                          \
    case NCH:                                                           \
        return launchBackwardCUDA<NCH, OP_TYPE>(means,                  \
                                                quats,                  \
                                                logScales,              \
                                                features,               \
                                                opacities,              \
                                                OP_VAL,                 \
                                                imageWidth,             \
                                                imageHeight,            \
                                                imageOriginW,           \
                                                imageOriginH,           \
                                                tileSize,               \
                                                tileOffsets,            \
                                                tileGaussianIds,        \
                                                renderedAlphas,         \
                                                lastIds,                \
                                                dLossDRenderedFeatures, \
                                                dLossDRenderedAlphas,   \
                                                backgrounds,            \
                                                masks);

    if (cameraModel == DistortionModel::ORTHOGRAPHIC) {
        const OrthographicWithDistortionCamera<float> camera{worldToCamMatricesStart,
                                                             worldToCamMatricesEnd,
                                                             projectionMatrices,
                                                             static_cast<uint32_t>(C),
                                                             (int32_t)imageWidth,
                                                             (int32_t)imageHeight,
                                                             (int32_t)imageOriginW,
                                                             (int32_t)imageOriginH,
                                                             rollingShutterType};
        switch (channels) {
            CALL_BWD_WITH_OP(1, OrthographicWithDistortionCamera<float>, camera)
            CALL_BWD_WITH_OP(2, OrthographicWithDistortionCamera<float>, camera)
            CALL_BWD_WITH_OP(3, OrthographicWithDistortionCamera<float>, camera)
            CALL_BWD_WITH_OP(4, OrthographicWithDistortionCamera<float>, camera)
            CALL_BWD_WITH_OP(5, OrthographicWithDistortionCamera<float>, camera)
            CALL_BWD_WITH_OP(8, OrthographicWithDistortionCamera<float>, camera)
            CALL_BWD_WITH_OP(9, OrthographicWithDistortionCamera<float>, camera)
            CALL_BWD_WITH_OP(16, OrthographicWithDistortionCamera<float>, camera)
            CALL_BWD_WITH_OP(17, OrthographicWithDistortionCamera<float>, camera)
            CALL_BWD_WITH_OP(32, OrthographicWithDistortionCamera<float>, camera)
            CALL_BWD_WITH_OP(33, OrthographicWithDistortionCamera<float>, camera)
            CALL_BWD_WITH_OP(64, OrthographicWithDistortionCamera<float>, camera)
            CALL_BWD_WITH_OP(65, OrthographicWithDistortionCamera<float>, camera)
            CALL_BWD_WITH_OP(128, OrthographicWithDistortionCamera<float>, camera)
            CALL_BWD_WITH_OP(129, OrthographicWithDistortionCamera<float>, camera)
            CALL_BWD_WITH_OP(192, OrthographicWithDistortionCamera<float>, camera)
            CALL_BWD_WITH_OP(193, OrthographicWithDistortionCamera<float>, camera)
            CALL_BWD_WITH_OP(256, OrthographicWithDistortionCamera<float>, camera)
            CALL_BWD_WITH_OP(257, OrthographicWithDistortionCamera<float>, camera)
            CALL_BWD_WITH_OP(512, OrthographicWithDistortionCamera<float>, camera)
            CALL_BWD_WITH_OP(513, OrthographicWithDistortionCamera<float>, camera)
        default:
            TORCH_CHECK_VALUE(
                false, "Unsupported channels for rasterize-from-world-3dgs backward: ", channels);
        }
    } else {
        const PerspectiveWithDistortionCamera<float> camera{worldToCamMatricesStart,
                                                            worldToCamMatricesEnd,
                                                            projectionMatrices,
                                                            distortionCoeffs,
                                                            static_cast<uint32_t>(C),
                                                            distortionCoeffs.size(1),
                                                            (int32_t)imageWidth,
                                                            (int32_t)imageHeight,
                                                            (int32_t)imageOriginW,
                                                            (int32_t)imageOriginH,
                                                            rollingShutterType,
                                                            cameraModel};
        switch (channels) {
            CALL_BWD_WITH_OP(1, PerspectiveWithDistortionCamera<float>, camera)
            CALL_BWD_WITH_OP(2, PerspectiveWithDistortionCamera<float>, camera)
            CALL_BWD_WITH_OP(3, PerspectiveWithDistortionCamera<float>, camera)
            CALL_BWD_WITH_OP(4, PerspectiveWithDistortionCamera<float>, camera)
            CALL_BWD_WITH_OP(5, PerspectiveWithDistortionCamera<float>, camera)
            CALL_BWD_WITH_OP(8, PerspectiveWithDistortionCamera<float>, camera)
            CALL_BWD_WITH_OP(9, PerspectiveWithDistortionCamera<float>, camera)
            CALL_BWD_WITH_OP(16, PerspectiveWithDistortionCamera<float>, camera)
            CALL_BWD_WITH_OP(17, PerspectiveWithDistortionCamera<float>, camera)
            CALL_BWD_WITH_OP(32, PerspectiveWithDistortionCamera<float>, camera)
            CALL_BWD_WITH_OP(33, PerspectiveWithDistortionCamera<float>, camera)
            CALL_BWD_WITH_OP(64, PerspectiveWithDistortionCamera<float>, camera)
            CALL_BWD_WITH_OP(65, PerspectiveWithDistortionCamera<float>, camera)
            CALL_BWD_WITH_OP(128, PerspectiveWithDistortionCamera<float>, camera)
            CALL_BWD_WITH_OP(129, PerspectiveWithDistortionCamera<float>, camera)
            CALL_BWD_WITH_OP(192, PerspectiveWithDistortionCamera<float>, camera)
            CALL_BWD_WITH_OP(193, PerspectiveWithDistortionCamera<float>, camera)
            CALL_BWD_WITH_OP(256, PerspectiveWithDistortionCamera<float>, camera)
            CALL_BWD_WITH_OP(257, PerspectiveWithDistortionCamera<float>, camera)
            CALL_BWD_WITH_OP(512, PerspectiveWithDistortionCamera<float>, camera)
            CALL_BWD_WITH_OP(513, PerspectiveWithDistortionCamera<float>, camera)
        default:
            TORCH_CHECK_VALUE(
                false, "Unsupported channels for rasterize-from-world-3dgs backward: ", channels);
        }
    }

#undef CALL_BWD_WITH_OP
}

template <>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianRasterizeFromWorld3DGSBackward<torch::kPrivateUse1>(
    const torch::Tensor &means,
    const torch::Tensor &quats,
    const torch::Tensor &logScales,
    const torch::Tensor &features,
    const torch::Tensor &opacities,
    const torch::Tensor &worldToCamMatricesStart,
    const torch::Tensor &worldToCamMatricesEnd,
    const torch::Tensor &projectionMatrices,
    const torch::Tensor &distortionCoeffs,
    const RollingShutterType rollingShutterType,
    const DistortionModel cameraModel,
    const uint32_t imageWidth,
    const uint32_t imageHeight,
    const uint32_t imageOriginW,
    const uint32_t imageOriginH,
    const uint32_t tileSize,
    const torch::Tensor &tileOffsets,
    const torch::Tensor &tileGaussianIds,
    const torch::Tensor &renderedAlphas,
    const torch::Tensor &lastIds,
    const torch::Tensor &dLossDRenderedFeatures,
    const torch::Tensor &dLossDRenderedAlphas,
    const at::optional<torch::Tensor> &backgrounds,
    const at::optional<torch::Tensor> &masks) {
    FVDB_FUNC_RANGE();

    TORCH_CHECK_VALUE(means.is_privateuseone(), "means must be a PrivateUse1 tensor");
    TORCH_CHECK_VALUE(quats.is_privateuseone(), "quats must be a PrivateUse1 tensor");
    TORCH_CHECK_VALUE(logScales.is_privateuseone(), "logScales must be a PrivateUse1 tensor");
    TORCH_CHECK_VALUE(features.is_privateuseone(), "features must be a PrivateUse1 tensor");
    TORCH_CHECK_VALUE(opacities.is_privateuseone(), "opacities must be a PrivateUse1 tensor");
    TORCH_CHECK_VALUE(worldToCamMatricesStart.is_privateuseone(),
                      "worldToCamMatricesStart must be a PrivateUse1 tensor");
    TORCH_CHECK_VALUE(worldToCamMatricesEnd.is_privateuseone(),
                      "worldToCamMatricesEnd must be a PrivateUse1 tensor");
    TORCH_CHECK_VALUE(projectionMatrices.is_privateuseone(),
                      "projectionMatrices must be a PrivateUse1 tensor");
    TORCH_CHECK_VALUE(distortionCoeffs.is_privateuseone(),
                      "distortionCoeffs must be a PrivateUse1 tensor");
    TORCH_CHECK_VALUE(tileOffsets.is_privateuseone(), "tileOffsets must be a PrivateUse1 tensor");
    TORCH_CHECK_VALUE(tileGaussianIds.is_privateuseone(),
                      "tileGaussianIds must be a PrivateUse1 tensor");
    TORCH_CHECK_VALUE(renderedAlphas.is_privateuseone(),
                      "renderedAlphas must be a PrivateUse1 tensor");
    TORCH_CHECK_VALUE(lastIds.is_privateuseone(), "lastIds must be a PrivateUse1 tensor");
    TORCH_CHECK_VALUE(dLossDRenderedFeatures.is_privateuseone(),
                      "dLossDRenderedFeatures must be a PrivateUse1 tensor");
    TORCH_CHECK_VALUE(dLossDRenderedAlphas.is_privateuseone(),
                      "dLossDRenderedAlphas must be a PrivateUse1 tensor");
    TORCH_CHECK_VALUE(tileOffsets.scalar_type() == torch::kInt64,
                      "tileOffsets must have dtype int64");
    TORCH_CHECK_VALUE(tileGaussianIds.scalar_type() == torch::kInt32,
                      "tileGaussianIds must have dtype int32");
    TORCH_CHECK_VALUE(lastIds.scalar_type() == torch::kInt32, "lastIds must have dtype int32");

    const int64_t C = features.size(0);
    const int64_t N = means.size(0);
    TORCH_CHECK_VALUE(opacities.dim() == 2, "opacities must have shape [C,N]");
    TORCH_CHECK_VALUE(opacities.size(0) == C && opacities.size(1) == N,
                      "opacities must have shape [C,N] matching features and N");

    const uint32_t channels = static_cast<uint32_t>(features.size(2));

#define CALL_BWD_PRIVATEUSE1_WITH_OP(NCH, OP_TYPE, OP_VAL)                     \
    case NCH:                                                                  \
        return launchBackwardPrivateUse1<NCH, OP_TYPE>(means,                  \
                                                       quats,                  \
                                                       logScales,              \
                                                       features,               \
                                                       opacities,              \
                                                       OP_VAL,                 \
                                                       imageWidth,             \
                                                       imageHeight,            \
                                                       imageOriginW,           \
                                                       imageOriginH,           \
                                                       tileSize,               \
                                                       tileOffsets,            \
                                                       tileGaussianIds,        \
                                                       renderedAlphas,         \
                                                       lastIds,                \
                                                       dLossDRenderedFeatures, \
                                                       dLossDRenderedAlphas,   \
                                                       backgrounds,            \
                                                       masks);

    if (cameraModel == DistortionModel::ORTHOGRAPHIC) {
        const OrthographicWithDistortionCamera<float> camera{worldToCamMatricesStart,
                                                             worldToCamMatricesEnd,
                                                             projectionMatrices,
                                                             static_cast<uint32_t>(C),
                                                             static_cast<int32_t>(imageWidth),
                                                             static_cast<int32_t>(imageHeight),
                                                             static_cast<int32_t>(imageOriginW),
                                                             static_cast<int32_t>(imageOriginH),
                                                             rollingShutterType};
        switch (channels) {
            CALL_BWD_PRIVATEUSE1_WITH_OP(1, OrthographicWithDistortionCamera<float>, camera)
            CALL_BWD_PRIVATEUSE1_WITH_OP(2, OrthographicWithDistortionCamera<float>, camera)
            CALL_BWD_PRIVATEUSE1_WITH_OP(3, OrthographicWithDistortionCamera<float>, camera)
            CALL_BWD_PRIVATEUSE1_WITH_OP(4, OrthographicWithDistortionCamera<float>, camera)
            CALL_BWD_PRIVATEUSE1_WITH_OP(5, OrthographicWithDistortionCamera<float>, camera)
            CALL_BWD_PRIVATEUSE1_WITH_OP(8, OrthographicWithDistortionCamera<float>, camera)
            CALL_BWD_PRIVATEUSE1_WITH_OP(9, OrthographicWithDistortionCamera<float>, camera)
            CALL_BWD_PRIVATEUSE1_WITH_OP(16, OrthographicWithDistortionCamera<float>, camera)
            CALL_BWD_PRIVATEUSE1_WITH_OP(17, OrthographicWithDistortionCamera<float>, camera)
            CALL_BWD_PRIVATEUSE1_WITH_OP(32, OrthographicWithDistortionCamera<float>, camera)
            CALL_BWD_PRIVATEUSE1_WITH_OP(33, OrthographicWithDistortionCamera<float>, camera)
            CALL_BWD_PRIVATEUSE1_WITH_OP(64, OrthographicWithDistortionCamera<float>, camera)
            CALL_BWD_PRIVATEUSE1_WITH_OP(65, OrthographicWithDistortionCamera<float>, camera)
            CALL_BWD_PRIVATEUSE1_WITH_OP(128, OrthographicWithDistortionCamera<float>, camera)
            CALL_BWD_PRIVATEUSE1_WITH_OP(129, OrthographicWithDistortionCamera<float>, camera)
            CALL_BWD_PRIVATEUSE1_WITH_OP(192, OrthographicWithDistortionCamera<float>, camera)
            CALL_BWD_PRIVATEUSE1_WITH_OP(193, OrthographicWithDistortionCamera<float>, camera)
            CALL_BWD_PRIVATEUSE1_WITH_OP(256, OrthographicWithDistortionCamera<float>, camera)
            CALL_BWD_PRIVATEUSE1_WITH_OP(257, OrthographicWithDistortionCamera<float>, camera)
            CALL_BWD_PRIVATEUSE1_WITH_OP(512, OrthographicWithDistortionCamera<float>, camera)
            CALL_BWD_PRIVATEUSE1_WITH_OP(513, OrthographicWithDistortionCamera<float>, camera)
        default:
            TORCH_CHECK_VALUE(
                false, "Unsupported channels for rasterize-from-world-3dgs backward: ", channels);
        }
    } else {
        const PerspectiveWithDistortionCamera<float> camera{worldToCamMatricesStart,
                                                            worldToCamMatricesEnd,
                                                            projectionMatrices,
                                                            distortionCoeffs,
                                                            static_cast<uint32_t>(C),
                                                            distortionCoeffs.size(1),
                                                            static_cast<int32_t>(imageWidth),
                                                            static_cast<int32_t>(imageHeight),
                                                            static_cast<int32_t>(imageOriginW),
                                                            static_cast<int32_t>(imageOriginH),
                                                            rollingShutterType,
                                                            cameraModel};
        switch (channels) {
            CALL_BWD_PRIVATEUSE1_WITH_OP(1, PerspectiveWithDistortionCamera<float>, camera)
            CALL_BWD_PRIVATEUSE1_WITH_OP(2, PerspectiveWithDistortionCamera<float>, camera)
            CALL_BWD_PRIVATEUSE1_WITH_OP(3, PerspectiveWithDistortionCamera<float>, camera)
            CALL_BWD_PRIVATEUSE1_WITH_OP(4, PerspectiveWithDistortionCamera<float>, camera)
            CALL_BWD_PRIVATEUSE1_WITH_OP(5, PerspectiveWithDistortionCamera<float>, camera)
            CALL_BWD_PRIVATEUSE1_WITH_OP(8, PerspectiveWithDistortionCamera<float>, camera)
            CALL_BWD_PRIVATEUSE1_WITH_OP(9, PerspectiveWithDistortionCamera<float>, camera)
            CALL_BWD_PRIVATEUSE1_WITH_OP(16, PerspectiveWithDistortionCamera<float>, camera)
            CALL_BWD_PRIVATEUSE1_WITH_OP(17, PerspectiveWithDistortionCamera<float>, camera)
            CALL_BWD_PRIVATEUSE1_WITH_OP(32, PerspectiveWithDistortionCamera<float>, camera)
            CALL_BWD_PRIVATEUSE1_WITH_OP(33, PerspectiveWithDistortionCamera<float>, camera)
            CALL_BWD_PRIVATEUSE1_WITH_OP(64, PerspectiveWithDistortionCamera<float>, camera)
            CALL_BWD_PRIVATEUSE1_WITH_OP(65, PerspectiveWithDistortionCamera<float>, camera)
            CALL_BWD_PRIVATEUSE1_WITH_OP(128, PerspectiveWithDistortionCamera<float>, camera)
            CALL_BWD_PRIVATEUSE1_WITH_OP(129, PerspectiveWithDistortionCamera<float>, camera)
            CALL_BWD_PRIVATEUSE1_WITH_OP(192, PerspectiveWithDistortionCamera<float>, camera)
            CALL_BWD_PRIVATEUSE1_WITH_OP(193, PerspectiveWithDistortionCamera<float>, camera)
            CALL_BWD_PRIVATEUSE1_WITH_OP(256, PerspectiveWithDistortionCamera<float>, camera)
            CALL_BWD_PRIVATEUSE1_WITH_OP(257, PerspectiveWithDistortionCamera<float>, camera)
            CALL_BWD_PRIVATEUSE1_WITH_OP(512, PerspectiveWithDistortionCamera<float>, camera)
            CALL_BWD_PRIVATEUSE1_WITH_OP(513, PerspectiveWithDistortionCamera<float>, camera)
        default:
            TORCH_CHECK_VALUE(
                false, "Unsupported channels for rasterize-from-world-3dgs backward: ", channels);
        }
    }

#undef CALL_BWD_PRIVATEUSE1_WITH_OP
}

template <>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianRasterizeFromWorld3DGSBackward<torch::kCPU>(const torch::Tensor &,
                                                            const torch::Tensor &,
                                                            const torch::Tensor &,
                                                            const torch::Tensor &,
                                                            const torch::Tensor &,
                                                            const torch::Tensor &,
                                                            const torch::Tensor &,
                                                            const torch::Tensor &,
                                                            const torch::Tensor &,
                                                            const RollingShutterType,
                                                            const DistortionModel,
                                                            const uint32_t,
                                                            const uint32_t,
                                                            const uint32_t,
                                                            const uint32_t,
                                                            const uint32_t,
                                                            const torch::Tensor &,
                                                            const torch::Tensor &,
                                                            const torch::Tensor &,
                                                            const torch::Tensor &,
                                                            const torch::Tensor &,
                                                            const torch::Tensor &,
                                                            const at::optional<torch::Tensor> &,
                                                            const at::optional<torch::Tensor> &) {
    TORCH_CHECK_VALUE(false, "dispatchGaussianRasterizeFromWorld3DGSBackward does not support CPU");
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
rasterizeWorldSpaceGaussiansBwd(const torch::Tensor &means,
                                const torch::Tensor &quats,
                                const torch::Tensor &logScales,
                                const torch::Tensor &features,
                                const torch::Tensor &opacities,
                                const torch::Tensor &worldToCamMatricesStart,
                                const torch::Tensor &worldToCamMatricesEnd,
                                const torch::Tensor &projectionMatrices,
                                const torch::Tensor &distortionCoeffs,
                                const RollingShutterType rollingShutterType,
                                const DistortionModel cameraModel,
                                const uint32_t imageWidth,
                                const uint32_t imageHeight,
                                const uint32_t imageOriginW,
                                const uint32_t imageOriginH,
                                const uint32_t tileSize,
                                const torch::Tensor &tileOffsets,
                                const torch::Tensor &tileGaussianIds,
                                const torch::Tensor &renderedAlphas,
                                const torch::Tensor &lastIds,
                                const torch::Tensor &dLossDRenderedFeatures,
                                const torch::Tensor &dLossDRenderedAlphas,
                                const at::optional<torch::Tensor> &backgrounds,
                                const at::optional<torch::Tensor> &masks) {
    return FVDB_DISPATCH_KERNEL(means.device(), [&]() {
        return dispatchGaussianRasterizeFromWorld3DGSBackward<DeviceTag>(means,
                                                                         quats,
                                                                         logScales,
                                                                         features,
                                                                         opacities,
                                                                         worldToCamMatricesStart,
                                                                         worldToCamMatricesEnd,
                                                                         projectionMatrices,
                                                                         distortionCoeffs,
                                                                         rollingShutterType,
                                                                         cameraModel,
                                                                         imageWidth,
                                                                         imageHeight,
                                                                         imageOriginW,
                                                                         imageOriginH,
                                                                         tileSize,
                                                                         tileOffsets,
                                                                         tileGaussianIds,
                                                                         renderedAlphas,
                                                                         lastIds,
                                                                         dLossDRenderedFeatures,
                                                                         dLossDRenderedAlphas,
                                                                         backgrounds,
                                                                         masks);
    });
}

} // namespace fvdb::detail::ops
