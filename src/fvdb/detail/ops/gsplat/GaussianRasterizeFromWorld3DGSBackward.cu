// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/gsplat/GaussianRasterizeFromWorld3DGS.cuh>
#include <fvdb/detail/ops/gsplat/GaussianRasterizeFromWorld3DGSBackward.h>
#include <fvdb/detail/ops/gsplat/GaussianWarpUtils.cuh>

#include <fvdb/detail/utils/Nvtx.h>

#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>

#include <cooperative_groups.h>

namespace fvdb::detail::ops {
namespace cg = cooperative_groups;

namespace {

template <uint32_t NUM_CHANNELS>
struct SharedGaussian {
    int32_t id;                       // flattened id in [0, C*N)
    nanovdb::math::Vec3<float> mean;  // world mean
    nanovdb::math::Vec4<float> quat;  // wxyz
    nanovdb::math::Vec3<float> scale; // exp(log_scales)
    nanovdb::math::Mat3<float> isclR; // S^{-1} R^T
    float opacity;
};

template <uint32_t NUM_CHANNELS>
__global__ void
rasterizeFromWorld3DGSBackwardKernel(
    // Gaussians
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> means,     // [N,3]
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> quats,     // [N,4]
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> logScales, // [N,3]
    // Per-camera
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits>
        features, // [C,N,D]
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits>
        opacities, // [C,N]
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits>
        worldToCamStart, // [C,4,4]
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits>
        worldToCamEnd, // [C,4,4]
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits>
        K, // [C,3,3]
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits>
        distortionCoeffs, // [C,K]
    const int64_t numDistCoeffs,
    const RollingShutterType rollingShutterType,
    const CameraModel cameraModel,
    // Settings
    const uint32_t imageWidth,
    const uint32_t imageHeight,
    const uint32_t imageOriginW,
    const uint32_t imageOriginH,
    const uint32_t tileSize,
    const uint32_t tileExtentW,
    const uint32_t tileExtentH,
    // Intersections
    const torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits>
        tileOffsets, // [C, tileExtentH, tileExtentW]
    const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits>
        tileGaussianIds, // [n_isects]
    const int32_t totalIntersections,
    // Forward outputs
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits>
        renderedAlphas, // [C,H,W,1]
    const torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits>
        lastIds, // [C,H,W]
    // Grad outputs
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits>
        dLossDRenderedFeatures, // [C,H,W,D]
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits>
        dLossDRenderedAlphas, // [C,H,W,1]
    // Backgrounds
    const float *__restrict__ backgrounds, // [C,D] or nullptr
    // Optional tile masks
    const bool *__restrict__ masks, // [C, tileExtentH, tileExtentW] or nullptr
    // Outputs (grads)
    float *__restrict__ dMeans,     // [N,3]
    float *__restrict__ dQuats,     // [N,4]
    float *__restrict__ dLogScales, // [N,3]
    float *__restrict__ dFeatures,  // [C,N,D]
    float *__restrict__ dOpacities  // [C,N]
) {
    auto block          = cg::this_thread_block();
    const uint32_t blockSize = blockDim.x * blockDim.y;

    const uint32_t globalLinearBlock = blockIdx.x;
    const uint32_t camId             = globalLinearBlock / (tileExtentH * tileExtentW);
    const uint32_t tileLinear        = globalLinearBlock - camId * (tileExtentH * tileExtentW);
    const uint32_t tileRow           = tileLinear / tileExtentW;
    const uint32_t tileCol           = tileLinear - tileRow * tileExtentW;

    const uint32_t row = tileRow * tileSize + threadIdx.y;
    const uint32_t col = tileCol * tileSize + threadIdx.x;
    const bool inside  = (row < imageHeight && col < imageWidth);

    // Parity with classic rasterizer: masked tiles contribute nothing.
    if (inside && masks != nullptr &&
        !masks[camId * tileExtentH * tileExtentW + tileRow * tileExtentW + tileCol]) {
        return;
    }

    // Camera pose for this camera.
    nanovdb::math::Mat3<float> R_wc_start, R_wc_end;
    nanovdb::math::Vec3<float> t_wc_start, t_wc_end;
    {
        R_wc_start = nanovdb::math::Mat3<float>(worldToCamStart[camId][0][0],
                                                worldToCamStart[camId][0][1],
                                                worldToCamStart[camId][0][2],
                                                worldToCamStart[camId][1][0],
                                                worldToCamStart[camId][1][1],
                                                worldToCamStart[camId][1][2],
                                                worldToCamStart[camId][2][0],
                                                worldToCamStart[camId][2][1],
                                                worldToCamStart[camId][2][2]);
        t_wc_start = nanovdb::math::Vec3<float>(worldToCamStart[camId][0][3],
                                                worldToCamStart[camId][1][3],
                                                worldToCamStart[camId][2][3]);

        R_wc_end   = nanovdb::math::Mat3<float>(worldToCamEnd[camId][0][0],
                                              worldToCamEnd[camId][0][1],
                                              worldToCamEnd[camId][0][2],
                                              worldToCamEnd[camId][1][0],
                                              worldToCamEnd[camId][1][1],
                                              worldToCamEnd[camId][1][2],
                                              worldToCamEnd[camId][2][0],
                                              worldToCamEnd[camId][2][1],
                                              worldToCamEnd[camId][2][2]);
        t_wc_end   = nanovdb::math::Vec3<float>(worldToCamEnd[camId][0][3],
                                              worldToCamEnd[camId][1][3],
                                              worldToCamEnd[camId][2][3]);
    }

    nanovdb::math::Mat3<float> K_cam(K[camId][0][0],
                                     K[camId][0][1],
                                     K[camId][0][2],
                                     K[camId][1][0],
                                     K[camId][1][1],
                                     K[camId][1][2],
                                     K[camId][2][0],
                                     K[camId][2][1],
                                     K[camId][2][2]);
    const float *distPtr =
        (numDistCoeffs > 0) ? &distortionCoeffs[camId][0] : nullptr;

    const WorldRay<float> ray = pixelToWorldRay<float>(row,
                                                       col,
                                                       imageWidth,
                                                       imageHeight,
                                                       imageOriginW,
                                                       imageOriginH,
                                                       R_wc_start,
                                                       t_wc_start,
                                                       R_wc_end,
                                                       t_wc_end,
                                                       K_cam,
                                                       distPtr,
                                                       numDistCoeffs,
                                                       rollingShutterType,
                                                       cameraModel);

    bool done = inside && ray.valid;

    // Gaussian range for this tile.
    const int32_t rangeStart = tileOffsets[camId][tileRow][tileCol];
    int32_t rangeEnd         = 0;
    if ((camId == (uint32_t)(features.size(0) - 1)) &&
        (tileRow == tileExtentH - 1) && (tileCol == tileExtentW - 1)) {
        rangeEnd = totalIntersections;
    } else if (tileCol + 1 < tileExtentW) {
        rangeEnd = tileOffsets[camId][tileRow][tileCol + 1];
    } else {
        if (tileRow + 1 < tileExtentH) {
            rangeEnd = tileOffsets[camId][tileRow + 1][0];
        } else {
            rangeEnd = tileOffsets[camId + 1][0][0];
        }
    }

    if (!done || rangeEnd <= rangeStart) {
        return;
    }

    // Forward state for this pixel.
    const int32_t binFinal = lastIds[camId][row][col];
    const float alphaFinal = renderedAlphas[camId][row][col][0];
    float T_final          = 1.0f - alphaFinal;
    float T                = T_final;

    float v_render_c[NUM_CHANNELS];
#pragma unroll
    for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
        v_render_c[k] = dLossDRenderedFeatures[camId][row][col][k];
    }
    const float v_render_a = dLossDRenderedAlphas[camId][row][col][0];

    float buffer[NUM_CHANNELS];
#pragma unroll
    for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
        buffer[k] = 0.f;
    }

    // Shared memory for gaussian batches.
    extern __shared__ char smem[];
    int32_t *idBatch = reinterpret_cast<int32_t *>(smem); // [blockSize]
    auto *gBatch =
        reinterpret_cast<SharedGaussian<NUM_CHANNELS> *>(&idBatch[blockSize]); // [blockSize]

    const uint32_t threadRank = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    const int32_t nIsects   = rangeEnd - rangeStart;
    const uint32_t nBatches = (nIsects + (int32_t)blockSize - 1) / (int32_t)blockSize;

    // Reduce max binFinal within warp (used to early-skip).
    const int32_t warpBinFinal = cg::reduce(warp, binFinal, cg::greater<int32_t>());

    for (uint32_t b = 0; b < nBatches; ++b) {
        block.sync();
        const int32_t batchEnd  = rangeEnd - 1 - (int32_t)(blockSize * b);
        const int32_t batchSize = min((int32_t)blockSize, batchEnd + 1 - rangeStart);
        const int32_t idx       = batchEnd - (int32_t)threadRank;

        if (idx >= rangeStart) {
            const int32_t flatId = tileGaussianIds[idx];
            idBatch[threadRank]  = flatId;
            const int32_t gid    = flatId % (int32_t)means.size(0);
            const int32_t cid    = flatId / (int32_t)means.size(0);

            const nanovdb::math::Vec3<float> mean_w(means[gid][0], means[gid][1], means[gid][2]);
            const nanovdb::math::Vec4<float> quat_wxyz(
                quats[gid][0], quats[gid][1], quats[gid][2], quats[gid][3]);
            const nanovdb::math::Vec3<float> scale(
                __expf(logScales[gid][0]), __expf(logScales[gid][1]), __expf(logScales[gid][2]));
            const nanovdb::math::Mat3<float> isclR = computeIsclRot<float>(quat_wxyz, scale);
            const float op                         = opacities[cid][gid];

            gBatch[threadRank].id      = flatId;
            gBatch[threadRank].mean    = mean_w;
            gBatch[threadRank].quat    = quat_wxyz;
            gBatch[threadRank].scale   = scale;
            gBatch[threadRank].isclR   = isclR;
            gBatch[threadRank].opacity = op;
        }

        block.sync();

        // Process gaussians in this batch, from back-to-front.
        const int32_t startT = max(0, batchEnd - warpBinFinal);
        for (int32_t t = startT; t < batchSize; ++t) {
            bool valid = done;
            if (batchEnd - t > binFinal) {
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
                mean_w    = g.mean;
                quat_wxyz = g.quat;
                scale     = g.scale;
                Mt        = g.isclR;
                opac      = g.opacity;

                o_minus_mu = ray.origin - mean_w;
                gro        = Mt * o_minus_mu;
                grd        = Mt * ray.dir;
                grd_n      = normalizeSafe<float>(grd);
                gcrod      = grd_n.cross(gro);
                grayDist   = gcrod.dot(gcrod);
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
#pragma unroll
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
#pragma unroll
                for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
                    v_feat_local[k] = fac * v_render_c[k];
                }

                // v_alpha accumulation
                float v_alpha = 0.f;
                const int32_t flatId = idBatch[t];
                const int32_t cid    = flatId / (int32_t)means.size(0);
                const int32_t gid    = flatId % (int32_t)means.size(0);

#pragma unroll
                for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
                    const float c = features[cid][gid][k];
                    v_alpha += (c * T - buffer[k] * ra) * v_render_c[k];
                }

                v_alpha += T_final * ra * v_render_a;

                if (backgrounds != nullptr) {
                    float accum = 0.f;
#pragma unroll
                    for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
                        accum += backgrounds[camId * NUM_CHANNELS + k] * v_render_c[k];
                    }
                    v_alpha += -T_final * ra * accum;
                }

                if (opac * vis <= kAlphaThreshold) {
                    const float v_vis       = opac * v_alpha;
                    const float v_gradDist  = -0.5f * vis * v_vis;
                    const nanovdb::math::Vec3<float> v_gcrod = 2.0f * v_gradDist * gcrod;
                    const nanovdb::math::Vec3<float> v_grd_n = -(v_gcrod.cross(gro));
                    const nanovdb::math::Vec3<float> v_gro   = v_gcrod.cross(grd_n);

                    const nanovdb::math::Vec3<float> v_grd = normalizeSafeVJP<float>(grd, v_grd_n);

                    // v_Mt = outer(v_grd, ray.dir) + outer(v_gro, (ray.origin - mean))
                    nanovdb::math::Mat3<float> v_Mt =
                        nanovdb::math::Mat3<float>(v_grd[0] * ray.dir[0],
                                                   v_grd[0] * ray.dir[1],
                                                   v_grd[0] * ray.dir[2],
                                                   v_grd[1] * ray.dir[0],
                                                   v_grd[1] * ray.dir[1],
                                                   v_grd[1] * ray.dir[2],
                                                   v_grd[2] * ray.dir[0],
                                                   v_grd[2] * ray.dir[1],
                                                   v_grd[2] * ray.dir[2]);
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
                    isclRotVectorJacobianProduct<float>(
                        quat_wxyz, scale, v_Mt, dQuat, dLogScale);
                    v_quat_local[0] += dQuat[0];
                    v_quat_local[1] += dQuat[1];
                    v_quat_local[2] += dQuat[2];
                    v_quat_local[3] += dQuat[3];
                    v_logscale_local += dLogScale;

                    v_opacity_local = vis * v_alpha;
                }

                // buffer update
#pragma unroll
                for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
                    const int32_t flatId = idBatch[t];
                    const int32_t cid    = flatId / (int32_t)means.size(0);
                    const int32_t gid    = flatId % (int32_t)means.size(0);
                    buffer[k] += features[cid][gid][k] * fac;
                }
            }

            // Warp-reduce and atomic add once per gaussian per warp.
            warpSumMut(v_opacity_local, warp);
            warpSumMut(v_mean_local, warp);
            warpSumMut<4>(v_quat_local, warp);
            warpSumMut(v_logscale_local, warp);
            warpSumMut<NUM_CHANNELS>(v_feat_local, warp);

            if (warp.thread_rank() == 0) {
                const int32_t flatId = idBatch[t];
                const int32_t cid    = flatId / (int32_t)means.size(0);
                const int32_t gid    = flatId % (int32_t)means.size(0);

                // Per-camera grads
                float *dFeatPtr = dFeatures + ((cid * (int32_t)means.size(0) + gid) * (int32_t)NUM_CHANNELS);
#pragma unroll
                for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
                    atomicAdd_system(dFeatPtr + k, v_feat_local[k]);
                }
                atomicAdd_system(dOpacities + (cid * (int32_t)means.size(0) + gid), v_opacity_local);

                // Geometry grads (shared across cameras)
                float *dMeanPtr = dMeans + gid * 3;
                atomicAdd_system(dMeanPtr + 0, v_mean_local[0]);
                atomicAdd_system(dMeanPtr + 1, v_mean_local[1]);
                atomicAdd_system(dMeanPtr + 2, v_mean_local[2]);

                float *dQuatPtr = dQuats + gid * 4;
                atomicAdd_system(dQuatPtr + 0, v_quat_local[0]);
                atomicAdd_system(dQuatPtr + 1, v_quat_local[1]);
                atomicAdd_system(dQuatPtr + 2, v_quat_local[2]);
                atomicAdd_system(dQuatPtr + 3, v_quat_local[3]);

                float *dLsPtr = dLogScales + gid * 3;
                atomicAdd_system(dLsPtr + 0, v_logscale_local[0]);
                atomicAdd_system(dLsPtr + 1, v_logscale_local[1]);
                atomicAdd_system(dLsPtr + 2, v_logscale_local[2]);
            }
        }
    }
}

template <uint32_t NUM_CHANNELS>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
launchBackward(const torch::Tensor &means,
               const torch::Tensor &quats,
               const torch::Tensor &logScales,
               const torch::Tensor &features,
               const torch::Tensor &opacities,
               const torch::Tensor &worldToCamStart,
               const torch::Tensor &worldToCamEnd,
               const torch::Tensor &K,
               const torch::Tensor &distortionCoeffs,
               const RollingShutterType rollingShutterType,
               const CameraModel cameraModel,
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
    const int64_t C = features.size(0);
    const int64_t N = means.size(0);

    torch::Tensor dMeans = torch::zeros_like(means);
    torch::Tensor dQuats = torch::zeros_like(quats);
    torch::Tensor dLogScales = torch::zeros_like(logScales);
    torch::Tensor dFeatures = torch::zeros_like(features);
    torch::Tensor dOpacities = torch::zeros_like(opacities);

    const uint32_t tileExtentW = (imageWidth + tileSize - 1) / tileSize;
    const uint32_t tileExtentH = (imageHeight + tileSize - 1) / tileSize;
    const dim3 blockDim(tileSize, tileSize, 1);
    const dim3 gridDim(C * tileExtentH * tileExtentW, 1, 1);
    const int32_t totalIntersections = (int32_t)tileGaussianIds.size(0);
    const int64_t numDistCoeffs      = distortionCoeffs.size(1);

    const float *bgPtr = backgrounds.has_value() ? backgrounds.value().data_ptr<float>() : nullptr;
    const bool *maskPtr = nullptr;
    torch::Tensor masksContig;
    if (masks.has_value()) {
        TORCH_CHECK_VALUE(masks.value().scalar_type() == torch::kBool, "masks must have dtype=bool");
        TORCH_CHECK_VALUE(masks.value().sizes() == torch::IntArrayRef({C, (int64_t)tileExtentH, (int64_t)tileExtentW}),
                          "masks must have shape [C, tileExtentH, tileExtentW]");
        masksContig = masks.value().contiguous();
        maskPtr     = masksContig.data_ptr<bool>();
    }

    const size_t blockSize = (size_t)tileSize * (size_t)tileSize;
    const size_t sharedMem = blockSize * (sizeof(int32_t) + sizeof(SharedGaussian<NUM_CHANNELS>));

    auto stream = at::cuda::getDefaultCUDAStream();
    rasterizeFromWorld3DGSBackwardKernel<NUM_CHANNELS><<<gridDim, blockDim, sharedMem, stream>>>(
        means.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        quats.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        logScales.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        features.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        opacities.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        worldToCamStart.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        worldToCamEnd.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        K.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        distortionCoeffs.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        numDistCoeffs,
        rollingShutterType,
        cameraModel,
        imageWidth,
        imageHeight,
        imageOriginW,
        imageOriginH,
        tileSize,
        tileExtentW,
        tileExtentH,
        tileOffsets.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        tileGaussianIds.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
        totalIntersections,
        renderedAlphas.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        lastIds.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        dLossDRenderedFeatures.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        dLossDRenderedAlphas.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        bgPtr,
        maskPtr,
        dMeans.data_ptr<float>(),
        dQuats.data_ptr<float>(),
        dLogScales.data_ptr<float>(),
        dFeatures.data_ptr<float>(),
        dOpacities.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {dMeans, dQuats, dLogScales, dFeatures, dOpacities};
}

} // namespace

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
    const CameraModel cameraModel,
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

    TORCH_CHECK_VALUE(means.is_cuda(), "means must be CUDA");
    TORCH_CHECK_VALUE(features.is_cuda(), "features must be CUDA");
    TORCH_CHECK_VALUE(renderedAlphas.is_cuda(), "renderedAlphas must be CUDA");
    TORCH_CHECK_VALUE(lastIds.is_cuda(), "lastIds must be CUDA");

    const uint32_t channels = (uint32_t)features.size(2);

#define CALL_BWD(NCH)                                                                       \
    case NCH:                                                                               \
        return launchBackward<NCH>(means,                                                   \
                                   quats,                                                   \
                                   logScales,                                               \
                                   features,                                                \
                                   opacities,                                               \
                                   worldToCamMatricesStart,                                 \
                                   worldToCamMatricesEnd,                                   \
                                   projectionMatrices,                                      \
                                   distortionCoeffs,                                        \
                                   rollingShutterType,                                      \
                                   cameraModel,                                             \
                                   imageWidth,                                              \
                                   imageHeight,                                             \
                                   imageOriginW,                                            \
                                   imageOriginH,                                            \
                                   tileSize,                                                \
                                   tileOffsets,                                             \
                                   tileGaussianIds,                                         \
                                   renderedAlphas,                                          \
                                   lastIds,                                                 \
                                   dLossDRenderedFeatures,                                  \
                                   dLossDRenderedAlphas,                                    \
                                   backgrounds,                                             \
                                   masks);

    switch (channels) {
        CALL_BWD(1)
        CALL_BWD(2)
        CALL_BWD(3)
        CALL_BWD(4)
        CALL_BWD(5)
        CALL_BWD(8)
        CALL_BWD(9)
        CALL_BWD(16)
        CALL_BWD(17)
        CALL_BWD(32)
        CALL_BWD(33)
        CALL_BWD(64)
        CALL_BWD(65)
        CALL_BWD(128)
        CALL_BWD(129)
        CALL_BWD(192)
        CALL_BWD(193)
        CALL_BWD(256)
        CALL_BWD(257)
        CALL_BWD(512)
        CALL_BWD(513)
    default: TORCH_CHECK_VALUE(false, "Unsupported channels for rasterize-from-world-3dgs backward: ", channels);
    }

#undef CALL_BWD
}

template <>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianRasterizeFromWorld3DGSBackward<torch::kCPU>(
    const torch::Tensor &,
    const torch::Tensor &,
    const torch::Tensor &,
    const torch::Tensor &,
    const torch::Tensor &,
    const torch::Tensor &,
    const torch::Tensor &,
    const torch::Tensor &,
    const torch::Tensor &,
    const RollingShutterType,
    const CameraModel,
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
    TORCH_CHECK_VALUE(false, "dispatchGaussianRasterizeFromWorld3DGSBackward is CUDA-only");
}

} // namespace fvdb::detail::ops

