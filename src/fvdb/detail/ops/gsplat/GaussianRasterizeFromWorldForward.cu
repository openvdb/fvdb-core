// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/gsplat/GaussianRasterizeFromWorld.cuh>
#include <fvdb/detail/ops/gsplat/GaussianRasterizeFromWorldForward.h>

#include <fvdb/detail/utils/Nvtx.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>

#include <ATen/cuda/CUDAContext.h>

#include <cooperative_groups.h>

namespace fvdb::detail::ops {
namespace cg = cooperative_groups;

namespace {

template <uint32_t NUM_CHANNELS>
struct SharedGaussian {
    int32_t id;                       // flattened id in [0, C*N)
    nanovdb::math::Vec3<float> mean;  // world mean
    nanovdb::math::Mat3<float> isclR; // S^{-1} R^T
    float opacity;
};

template <uint32_t NUM_CHANNELS>
__global__ void
rasterizeFromWorld3DGSForwardKernel(
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
        worldToCamStart, // [C,4,4] (view as 3D accessor for contiguous)
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
    // Backgrounds
    const float *__restrict__ backgrounds, // [C, D] or nullptr
    // Optional tile masks
    const bool *__restrict__ masks, // [C, tileExtentH, tileExtentW] or nullptr
    // Outputs
    float *__restrict__ outFeatures, // [C,H,W,D]
    float *__restrict__ outAlphas,   // [C,H,W,1] packed as [C*H*W]
    int32_t *__restrict__ outLastIds // [C,H,W]
) {
    const uint32_t blockSize = blockDim.x * blockDim.y;
    auto block               = cg::this_thread_block();

    const uint32_t globalLinearBlock = blockIdx.x;
    const uint32_t camId             = globalLinearBlock / (tileExtentH * tileExtentW);
    const uint32_t tileLinear        = globalLinearBlock - camId * (tileExtentH * tileExtentW);
    const uint32_t tileRow           = tileLinear / tileExtentW;
    const uint32_t tileCol           = tileLinear - tileRow * tileExtentW;

    const uint32_t row = tileRow * tileSize + threadIdx.y;
    const uint32_t col = tileCol * tileSize + threadIdx.x;
    const bool inside  = (row < imageHeight && col < imageWidth);

    // Pixel index within this camera.
    const uint32_t pixId = row * imageWidth + col;
    const uint32_t imgBaseFeat =
        (camId * imageHeight * imageWidth + pixId) * NUM_CHANNELS; // feature base
    const uint32_t imgBasePix = (camId * imageHeight * imageWidth + pixId); // alpha/last base

    // Parity with classic rasterizer: masked tiles write background and exit.
    if (inside && masks != nullptr &&
        !masks[camId * tileExtentH * tileExtentW + tileRow * tileExtentW + tileCol]) {
        outAlphas[imgBasePix]  = 0.0f;
        outLastIds[imgBasePix] = 0;
#pragma unroll
        for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
            outFeatures[imgBaseFeat + k] = (backgrounds != nullptr)
                                               ? backgrounds[camId * NUM_CHANNELS + k]
                                               : 0.0f;
        }
        return;
    }

    // Build camera pose for this camera (start/end).
    nanovdb::math::Mat3<float> R_wc_start, R_wc_end;
    nanovdb::math::Vec3<float> t_wc_start, t_wc_end;
    {
        // Extract 3x3 rotation and translation from [4,4] row-major.
        const float m00s = worldToCamStart[camId][0][0];
        const float m01s = worldToCamStart[camId][0][1];
        const float m02s = worldToCamStart[camId][0][2];
        const float m10s = worldToCamStart[camId][1][0];
        const float m11s = worldToCamStart[camId][1][1];
        const float m12s = worldToCamStart[camId][1][2];
        const float m20s = worldToCamStart[camId][2][0];
        const float m21s = worldToCamStart[camId][2][1];
        const float m22s = worldToCamStart[camId][2][2];
        R_wc_start       = nanovdb::math::Mat3<float>(m00s,
                                                m01s,
                                                m02s,
                                                m10s,
                                                m11s,
                                                m12s,
                                                m20s,
                                                m21s,
                                                m22s);
        t_wc_start       = nanovdb::math::Vec3<float>(worldToCamStart[camId][0][3],
                                                worldToCamStart[camId][1][3],
                                                worldToCamStart[camId][2][3]);

        const float m00e = worldToCamEnd[camId][0][0];
        const float m01e = worldToCamEnd[camId][0][1];
        const float m02e = worldToCamEnd[camId][0][2];
        const float m10e = worldToCamEnd[camId][1][0];
        const float m11e = worldToCamEnd[camId][1][1];
        const float m12e = worldToCamEnd[camId][1][2];
        const float m20e = worldToCamEnd[camId][2][0];
        const float m21e = worldToCamEnd[camId][2][1];
        const float m22e = worldToCamEnd[camId][2][2];
        R_wc_end         = nanovdb::math::Mat3<float>(m00e,
                                              m01e,
                                              m02e,
                                              m10e,
                                              m11e,
                                              m12e,
                                              m20e,
                                              m21e,
                                              m22e);
        t_wc_end         = nanovdb::math::Vec3<float>(worldToCamEnd[camId][0][3],
                                              worldToCamEnd[camId][1][3],
                                              worldToCamEnd[camId][2][3]);
    }

    nanovdb::math::Mat3<float> K_cam;
    K_cam = nanovdb::math::Mat3<float>(K[camId][0][0],
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

    bool done = (!inside) || (!ray.valid);

    // Determine gaussian range for this tile.
    const int32_t rangeStart = tileOffsets[camId][tileRow][tileCol];
    int32_t rangeEnd         = 0;
    if ((camId == (uint32_t)(features.size(0) - 1)) &&
        (tileRow == tileExtentH - 1) && (tileCol == tileExtentW - 1)) {
        rangeEnd = totalIntersections;
    } else if (tileCol + 1 < tileExtentW) {
        rangeEnd = tileOffsets[camId][tileRow][tileCol + 1];
    } else {
        // wrap to next row/camera
        if (tileRow + 1 < tileExtentH) {
            rangeEnd = tileOffsets[camId][tileRow + 1][0];
        } else {
            // next camera
            rangeEnd = tileOffsets[camId + 1][0][0];
        }
    }

    // If no intersections, just write background.
    if (inside && rangeEnd <= rangeStart) {
        // alpha=0, output background if provided else 0.
        outAlphas[imgBasePix]  = 0.0f;
        outLastIds[imgBasePix] = 0;
#pragma unroll
        for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
            outFeatures[imgBaseFeat + k] = (backgrounds != nullptr)
                                               ? backgrounds[camId * NUM_CHANNELS + k]
                                               : 0.0f;
        }
        return;
    }

    // Shared memory for batched gaussians.
    extern __shared__ char smem[];
    int32_t *idBatch = reinterpret_cast<int32_t *>(smem); // [blockSize]
    auto *gaussBatch =
        reinterpret_cast<SharedGaussian<NUM_CHANNELS> *>(&idBatch[blockSize]); // [blockSize]

    float T = 1.0f;
    uint32_t curIdx = 0;
    float pixOut[NUM_CHANNELS];
#pragma unroll
    for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
        pixOut[k] = 0.f;
    }

    const int32_t nIsects    = rangeEnd - rangeStart;
    const uint32_t nBatches  = (nIsects + blockSize - 1) / blockSize;
    const uint32_t threadRank = block.thread_rank();

    for (uint32_t b = 0; b < nBatches; ++b) {
        if (__syncthreads_count(done) >= (int)blockSize) {
            break;
        }

        const int32_t batchStart = rangeStart + (int32_t)(blockSize * b);
        const int32_t idx        = batchStart + (int32_t)threadRank;
        if (idx < rangeEnd) {
            const int32_t flatId = tileGaussianIds[idx];
            idBatch[threadRank]  = flatId;
            const int32_t gid    = flatId % (int32_t)means.size(0);

            const nanovdb::math::Vec3<float> mean_w(
                means[gid][0], means[gid][1], means[gid][2]);
            const nanovdb::math::Vec4<float> quat_wxyz(
                quats[gid][0], quats[gid][1], quats[gid][2], quats[gid][3]);
            const nanovdb::math::Vec3<float> scale(
                __expf(logScales[gid][0]), __expf(logScales[gid][1]), __expf(logScales[gid][2]));
            const nanovdb::math::Mat3<float> isclR = computeIsclRot<float>(quat_wxyz, scale);
            const int32_t cid                      = flatId / (int32_t)means.size(0);
            const float op                         = opacities[cid][gid];

            gaussBatch[threadRank].id      = flatId;
            gaussBatch[threadRank].mean    = mean_w;
            gaussBatch[threadRank].isclR   = isclR;
            gaussBatch[threadRank].opacity = op;
        }

        block.sync();

        const uint32_t batchSize =
            min((uint32_t)blockSize, (uint32_t)max(0, rangeEnd - batchStart));
        for (uint32_t t = 0; (t < batchSize) && !done; ++t) {
            const SharedGaussian<NUM_CHANNELS> g = gaussBatch[t];
            const nanovdb::math::Vec3<float> gro = g.isclR * (ray.origin - g.mean);
            const nanovdb::math::Vec3<float> grd = normalizeSafe<float>(g.isclR * ray.dir);
            const nanovdb::math::Vec3<float> gcrod = grd.cross(gro);
            const float grayDist = gcrod.dot(gcrod);
            const float power    = -0.5f * grayDist;
            const float vis      = __expf(power);
            float alpha          = min(kAlphaThreshold, g.opacity * vis);
            if (power > 0.f || alpha < 1.f / 255.f) {
                continue;
            }
            const float nextT = T * (1.0f - alpha);
            if (nextT <= 1e-4f) {
                done = true;
                break;
            }
            const float contrib = alpha * T;
            const int32_t cid   = g.id / (int32_t)means.size(0);
            const int32_t gid   = g.id % (int32_t)means.size(0);
#pragma unroll
            for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
                pixOut[k] += features[cid][gid][k] * contrib;
            }
            curIdx = (uint32_t)(batchStart + (int32_t)t);
            T      = nextT;
        }
    }

    if (!inside) {
        return;
    }

    outAlphas[imgBasePix]  = 1.0f - T;
    outLastIds[imgBasePix] = (int32_t)curIdx;
#pragma unroll
    for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
        outFeatures[imgBaseFeat + k] =
            (backgrounds != nullptr) ? (pixOut[k] + T * backgrounds[camId * NUM_CHANNELS + k])
                                     : pixOut[k];
    }
}

template <uint32_t NUM_CHANNELS>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
launchForward(const torch::Tensor &means,
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
              const at::optional<torch::Tensor> &backgrounds,
              const at::optional<torch::Tensor> &masks) {
    const int64_t C = features.size(0);

    auto opts = features.options();
    torch::Tensor outFeatures =
        torch::zeros({C, (int64_t)imageHeight, (int64_t)imageWidth, (int64_t)NUM_CHANNELS}, opts);
    torch::Tensor outAlphas =
        torch::zeros({C, (int64_t)imageHeight, (int64_t)imageWidth, 1}, opts);
    torch::Tensor outLastIds =
        torch::zeros({C, (int64_t)imageHeight, (int64_t)imageWidth},
                     torch::TensorOptions().dtype(torch::kInt32).device(features.device()));

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
    const size_t sharedMem =
        blockSize * (sizeof(int32_t) + sizeof(SharedGaussian<NUM_CHANNELS>));

    auto stream = at::cuda::getDefaultCUDAStream();
    rasterizeFromWorld3DGSForwardKernel<NUM_CHANNELS><<<gridDim, blockDim, sharedMem, stream>>>(
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
        bgPtr,
        maskPtr,
        outFeatures.data_ptr<float>(),
        outAlphas.data_ptr<float>(),
        outLastIds.data_ptr<int32_t>());

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {outFeatures, outAlphas, outLastIds};
}

} // namespace

template <>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianRasterizeFromWorld3DGSForward<torch::kCUDA>(
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
    const at::optional<torch::Tensor> &backgrounds,
    const at::optional<torch::Tensor> &masks) {
    FVDB_FUNC_RANGE();

    TORCH_CHECK_VALUE(means.is_cuda(), "means must be CUDA");
    TORCH_CHECK_VALUE(features.is_cuda(), "features must be CUDA");
    TORCH_CHECK_VALUE(tileOffsets.is_cuda(), "tileOffsets must be CUDA");
    TORCH_CHECK_VALUE(tileGaussianIds.is_cuda(), "tileGaussianIds must be CUDA");

    TORCH_CHECK_VALUE(means.dim() == 2 && means.size(1) == 3, "means must have shape [N,3]");
    TORCH_CHECK_VALUE(quats.dim() == 2 && quats.size(1) == 4, "quats must have shape [N,4]");
    TORCH_CHECK_VALUE(logScales.dim() == 2 && logScales.size(1) == 3,
                      "logScales must have shape [N,3]");
    TORCH_CHECK_VALUE(features.dim() == 3, "features must have shape [C,N,D]");
    TORCH_CHECK_VALUE(opacities.dim() == 2, "opacities must have shape [C,N]");

    const int64_t C = features.size(0);
    const int64_t N = means.size(0);
    TORCH_CHECK_VALUE(features.size(1) == N, "features must have shape [C,N,D] matching N");
    TORCH_CHECK_VALUE(opacities.size(0) == C && opacities.size(1) == N,
                      "opacities must have shape [C,N] matching features and N");

    TORCH_CHECK_VALUE(worldToCamMatricesStart.sizes() == torch::IntArrayRef({C, 4, 4}),
                      "worldToCamMatricesStart must have shape [C,4,4]");
    TORCH_CHECK_VALUE(worldToCamMatricesEnd.sizes() == torch::IntArrayRef({C, 4, 4}),
                      "worldToCamMatricesEnd must have shape [C,4,4]");
    TORCH_CHECK_VALUE(projectionMatrices.sizes() == torch::IntArrayRef({C, 3, 3}),
                      "projectionMatrices must have shape [C,3,3]");

    const int64_t numDistCoeffs = distortionCoeffs.size(1);
    TORCH_CHECK_VALUE(distortionCoeffs.dim() == 2 && distortionCoeffs.size(0) == C,
                      "distortionCoeffs must have shape [C,K]");
    if (cameraModel == CameraModel::OPENCV_RADTAN_5 || cameraModel == CameraModel::OPENCV_RATIONAL_8 ||
        cameraModel == CameraModel::OPENCV_RADTAN_THIN_PRISM_9 ||
        cameraModel == CameraModel::OPENCV_THIN_PRISM_12) {
        TORCH_CHECK_VALUE(numDistCoeffs == 12, "For CameraModel::OPENCV_* distortionCoeffs must be [C,12]");
    }

    const uint32_t channels = (uint32_t)features.size(2);

#define CALL_FWD(NCH)                                                                                \
    case NCH:                                                                                       \
        return launchForward<NCH>(means,                                                             \
                                  quats,                                                             \
                                  logScales,                                                         \
                                  features,                                                          \
                                  opacities,                                                         \
                                  worldToCamMatricesStart,                                           \
                                  worldToCamMatricesEnd,                                             \
                                  projectionMatrices,                                                \
                                  distortionCoeffs,                                                  \
                                  rollingShutterType,                                                \
                                  cameraModel,                                                       \
                                  imageWidth,                                                        \
                                  imageHeight,                                                       \
                                  imageOriginW,                                                      \
                                  imageOriginH,                                                      \
                                  tileSize,                                                          \
                                  tileOffsets,                                                       \
                                  tileGaussianIds,                                                   \
                                  backgrounds,                                                       \
                                  masks);

    switch (channels) {
        CALL_FWD(1)
        CALL_FWD(2)
        CALL_FWD(3)
        CALL_FWD(4)
        CALL_FWD(5)
        CALL_FWD(8)
        CALL_FWD(9)
        CALL_FWD(16)
        CALL_FWD(17)
        CALL_FWD(32)
        CALL_FWD(33)
        CALL_FWD(64)
        CALL_FWD(65)
        CALL_FWD(128)
        CALL_FWD(129)
        CALL_FWD(192)
        CALL_FWD(193)
        CALL_FWD(256)
        CALL_FWD(257)
        CALL_FWD(512)
        CALL_FWD(513)
    default: TORCH_CHECK_VALUE(false, "Unsupported channels for rasterize-from-world-3dgs: ", channels);
    }

#undef CALL_FWD
}

template <>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianRasterizeFromWorld3DGSForward<torch::kCPU>(
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
    const at::optional<torch::Tensor> &,
    const at::optional<torch::Tensor> &) {
    TORCH_CHECK_VALUE(false, "dispatchGaussianRasterizeFromWorld3DGSForward is CUDA-only");
}

} // namespace fvdb::detail::ops

