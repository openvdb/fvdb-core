// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/gsplat/GaussianRasterizeFromWorld.cuh>
#include <fvdb/detail/ops/gsplat/GaussianRasterizeFromWorldBackward.h>
#include <fvdb/detail/ops/gsplat/GaussianRasterizeOptionalInputs.h>
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

template <uint32_t NUM_CHANNELS> struct SharedGaussian {
    int32_t id;                       // flattened id in [0, C*N)
    nanovdb::math::Vec3<float> mean;  // world mean
    nanovdb::math::Vec4<float> quat;  // wxyz
    nanovdb::math::Vec3<float> scale; // exp(log_scales)
    nanovdb::math::Mat3<float> isclR; // S^{-1} R^T
    float opacity;
};

template <uint32_t NUM_CHANNELS, typename CameraOp> struct RasterizeFromWorldBackwardArgs {
    RasterizeFromWorldCommonArgs commonArgs;
    CameraOp cameraOp;
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

template <uint32_t NUM_CHANNELS, typename CameraOp>
__global__ void
rasterizeFromWorld3DGSBackwardKernel(
    const RasterizeFromWorldBackwardArgs<NUM_CHANNELS, CameraOp> args) {
    auto block               = cg::this_thread_block();
    const uint32_t blockSize = blockDim.x * blockDim.y;
    const auto &common       = args.commonArgs;

    uint32_t camId, tileRow, tileCol, row, col;
    common.denseCoordinates(camId, tileRow, tileCol, row, col);
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
    CameraOp cameraOpLocal = args.cameraOp;
    cameraOpLocal.loadSharedMemory(smem);
    block.sync();

    const nanovdb::math::Ray<float> ray = cameraOpLocal.projectToRay(camId, row, col);

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
    int32_t binFinal = -1;
    float T_final    = 1.0f;
    float T          = 1.0f;

    float v_render_c[NUM_CHANNELS];
#pragma unroll
    for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
        v_render_c[k] = 0.f;
    }
    float v_render_a = 0.f;

    if (done) {
        binFinal = args.lastIds[camId][row][col];

        const float alphaFinal = args.renderedAlphas[camId][row][col][0];
        T_final                = 1.0f - alphaFinal;
        T                      = T_final;

#pragma unroll
        for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
            v_render_c[k] = args.dLossDRenderedFeatures[camId][row][col][k];
        }
        v_render_a = args.dLossDRenderedAlphas[camId][row][col][0];
    }

    float buffer[NUM_CHANNELS];
#pragma unroll
    for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
        buffer[k] = 0.f;
    }

    // Shared memory for gaussian batches (after camera-op shared state).
    char *gaussSmem  = smem + cameraOpLocal.numSharedMemBytes();
    int32_t *idBatch = reinterpret_cast<int32_t *>(gaussSmem);                 // [blockSize]
    auto *gBatch =
        reinterpret_cast<SharedGaussian<NUM_CHANNELS> *>(&idBatch[blockSize]); // [blockSize]

    const uint32_t threadRank      = block.thread_rank();
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
                float v_alpha        = 0.f;
                const int32_t flatId = idBatch[t];
                const int32_t cid    = flatId / (int32_t)common.means.size(0);
                const int32_t gid    = flatId % (int32_t)common.means.size(0);

#pragma unroll
                for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
                    const float c = common.features[cid][gid][k];
                    v_alpha += (c * T - buffer[k] * ra) * v_render_c[k];
                }

                v_alpha += T_final * ra * v_render_a;

                if (common.backgrounds != nullptr) {
                    float accum = 0.f;
#pragma unroll
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
#pragma unroll
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
            warpSumMut<NUM_CHANNELS>(v_feat_local, warp);

            if (warp.thread_rank() == 0) {
                const int32_t flatId = idBatch[t];
                const int32_t cid    = flatId / (int32_t)common.means.size(0);
                const int32_t gid    = flatId % (int32_t)common.means.size(0);

                // Per-camera grads
                float *dFeaturesGaussianPtr = args.dFeatures.data() +
                                              cid * args.dFeatures.stride(0) +
                                              gid * args.dFeatures.stride(1);
#pragma unroll
                for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
                    atomicAdd_system(dFeaturesGaussianPtr + k * args.dFeatures.stride(2),
                                     v_feat_local[k]);
                }
                float *dOpacityGaussianPtr = args.dOpacities.data() +
                                             cid * args.dOpacities.stride(0) +
                                             gid * args.dOpacities.stride(1);
                atomicAdd_system(dOpacityGaussianPtr, v_opacity_local);

                // Geometry grads (shared across cameras)
                float *dMeansPtr = args.dMeans.data() + gid * args.dMeans.stride(0);
                atomicAdd_system(dMeansPtr + 0 * args.dMeans.stride(1), v_mean_local[0]);
                atomicAdd_system(dMeansPtr + 1 * args.dMeans.stride(1), v_mean_local[1]);
                atomicAdd_system(dMeansPtr + 2 * args.dMeans.stride(1), v_mean_local[2]);

                float *dQuatsPtr = args.dQuats.data() + gid * args.dQuats.stride(0);
                atomicAdd_system(dQuatsPtr + 0 * args.dQuats.stride(1), v_quat_local[0]);
                atomicAdd_system(dQuatsPtr + 1 * args.dQuats.stride(1), v_quat_local[1]);
                atomicAdd_system(dQuatsPtr + 2 * args.dQuats.stride(1), v_quat_local[2]);
                atomicAdd_system(dQuatsPtr + 3 * args.dQuats.stride(1), v_quat_local[3]);

                float *dLogScalesPtr = args.dLogScales.data() + gid * args.dLogScales.stride(0);
                atomicAdd_system(dLogScalesPtr + 0 * args.dLogScales.stride(1),
                                 v_logscale_local[0]);
                atomicAdd_system(dLogScalesPtr + 1 * args.dLogScales.stride(1),
                                 v_logscale_local[1]);
                atomicAdd_system(dLogScalesPtr + 2 * args.dLogScales.stride(1),
                                 v_logscale_local[2]);
            }
        }
    }
}

template <uint32_t NUM_CHANNELS, typename CameraOp>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
launchBackward(const torch::Tensor &means,
               const torch::Tensor &quats,
               const torch::Tensor &logScales,
               const torch::Tensor &features,
               const torch::Tensor &opacities,
               const CameraOp &cameraOp,
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

    torch::Tensor dMeans     = torch::zeros_like(means);
    torch::Tensor dQuats     = torch::zeros_like(quats);
    torch::Tensor dLogScales = torch::zeros_like(logScales);
    torch::Tensor dFeatures  = torch::zeros_like(features);
    torch::Tensor dOpacities = torch::zeros_like(opacities);

    const uint32_t tileExtentW = (imageWidth + tileSize - 1) / tileSize;
    const uint32_t tileExtentH = (imageHeight + tileSize - 1) / tileSize;
    const dim3 blockDim(tileSize, tileSize, 1);
    const dim3 gridDim(C * tileExtentH * tileExtentW, 1, 1);
    const int32_t totalIntersections = static_cast<int32_t>(tileGaussianIds.size(0));

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
        tileOffsets.packed_accessor64<int32_t, 3, torch::RestrictPtrTraits>(),
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
    const size_t sharedMem = cameraOp.numSharedMemBytes() +
                             blockSize * (sizeof(int32_t) + sizeof(SharedGaussian<NUM_CHANNELS>));

    RasterizeFromWorldBackwardArgs<NUM_CHANNELS, CameraOp> kernelArgs{
        args,
        cameraOp,
        renderedAlphas.packed_accessor64<float, 4, torch::RestrictPtrTraits>(),
        lastIds.packed_accessor64<int32_t, 3, torch::RestrictPtrTraits>(),
        dLossDRenderedFeatures.packed_accessor64<float, 4, torch::RestrictPtrTraits>(),
        dLossDRenderedAlphas.packed_accessor64<float, 4, torch::RestrictPtrTraits>(),
        dMeans.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
        dQuats.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
        dLogScales.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
        dFeatures.packed_accessor64<float, 3, torch::RestrictPtrTraits>(),
        dOpacities.packed_accessor64<float, 2, torch::RestrictPtrTraits>()};

    auto stream = at::cuda::getDefaultCUDAStream();
    rasterizeFromWorld3DGSBackwardKernel<NUM_CHANNELS, CameraOp>
        <<<gridDim, blockDim, sharedMem, stream>>>(kernelArgs);
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
    const DistortionModel cameraModel,
    const RenderSettings &settings,
    const torch::Tensor &tileOffsets,
    const torch::Tensor &tileGaussianIds,
    const torch::Tensor &renderedAlphas,
    const torch::Tensor &lastIds,
    const torch::Tensor &dLossDRenderedFeatures,
    const torch::Tensor &dLossDRenderedAlphas,
    const at::optional<torch::Tensor> &backgrounds,
    const at::optional<torch::Tensor> &masks) {
    FVDB_FUNC_RANGE();

    const uint32_t imageWidth   = settings.imageWidth;
    const uint32_t imageHeight  = settings.imageHeight;
    const uint32_t imageOriginW = settings.imageOriginW;
    const uint32_t imageOriginH = settings.imageOriginH;
    const uint32_t tileSize     = settings.tileSize;

    TORCH_CHECK_VALUE(means.is_cuda(), "means must be CUDA");
    TORCH_CHECK_VALUE(features.is_cuda(), "features must be CUDA");
    TORCH_CHECK_VALUE(opacities.is_cuda(), "opacities must be CUDA");
    TORCH_CHECK_VALUE(renderedAlphas.is_cuda(), "renderedAlphas must be CUDA");
    TORCH_CHECK_VALUE(lastIds.is_cuda(), "lastIds must be CUDA");

    const int64_t C = features.size(0);
    const int64_t N = means.size(0);

    // Opacities may be provided either per-camera ([C,N]) or shared across cameras ([N]).
    // TODO(fvdb): Avoid materializing a repeated [C,N] tensor when opacities are shared across
    // cameras (similar to PR #451).
    torch::Tensor opacitiesBatched = opacities;
    if (opacitiesBatched.dim() == 1) {
        TORCH_CHECK_VALUE(opacitiesBatched.size(0) == N,
                          "opacities must have shape [N] or [C,N] matching N");
        opacitiesBatched = opacitiesBatched.unsqueeze(0).repeat({C, 1});
    }
    opacitiesBatched = opacitiesBatched.contiguous();

    const uint32_t channels = (uint32_t)features.size(2);

#define CALL_BWD_WITH_OP(NCH, OP_TYPE, OP_VAL)            \
    case NCH:                                              \
        return launchBackward<NCH, OP_TYPE>(means,         \
                                            quats,         \
                                            logScales,     \
                                            features,      \
                                            opacitiesBatched, \
                                            OP_VAL,                  \
                                            imageWidth,              \
                                            imageHeight,             \
                                            imageOriginW,            \
                                            imageOriginH,            \
                                            tileSize,                \
                                            tileOffsets,             \
                                            tileGaussianIds,         \
                                            renderedAlphas,          \
                                            lastIds,                 \
                                            dLossDRenderedFeatures,  \
                                            dLossDRenderedAlphas,    \
                                            backgrounds,             \
                                            masks);

    if (cameraModel == DistortionModel::ORTHOGRAPHIC) {
        const OrthographicWithDistortionCameraOp<float> cameraOp{
            worldToCamMatricesStart.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            worldToCamMatricesEnd.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            projectionMatrices.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            static_cast<uint32_t>(C),
            (int32_t)imageWidth,
            (int32_t)imageHeight,
            (int32_t)imageOriginW,
            (int32_t)imageOriginH,
            rollingShutterType};
        switch (channels) {
            CALL_BWD_WITH_OP(1, OrthographicWithDistortionCameraOp<float>, cameraOp)
            CALL_BWD_WITH_OP(2, OrthographicWithDistortionCameraOp<float>, cameraOp)
            CALL_BWD_WITH_OP(3, OrthographicWithDistortionCameraOp<float>, cameraOp)
            CALL_BWD_WITH_OP(4, OrthographicWithDistortionCameraOp<float>, cameraOp)
            CALL_BWD_WITH_OP(5, OrthographicWithDistortionCameraOp<float>, cameraOp)
            CALL_BWD_WITH_OP(8, OrthographicWithDistortionCameraOp<float>, cameraOp)
            CALL_BWD_WITH_OP(9, OrthographicWithDistortionCameraOp<float>, cameraOp)
            CALL_BWD_WITH_OP(16, OrthographicWithDistortionCameraOp<float>, cameraOp)
            CALL_BWD_WITH_OP(17, OrthographicWithDistortionCameraOp<float>, cameraOp)
            CALL_BWD_WITH_OP(32, OrthographicWithDistortionCameraOp<float>, cameraOp)
            CALL_BWD_WITH_OP(33, OrthographicWithDistortionCameraOp<float>, cameraOp)
            CALL_BWD_WITH_OP(64, OrthographicWithDistortionCameraOp<float>, cameraOp)
            CALL_BWD_WITH_OP(65, OrthographicWithDistortionCameraOp<float>, cameraOp)
            CALL_BWD_WITH_OP(128, OrthographicWithDistortionCameraOp<float>, cameraOp)
            CALL_BWD_WITH_OP(129, OrthographicWithDistortionCameraOp<float>, cameraOp)
            CALL_BWD_WITH_OP(192, OrthographicWithDistortionCameraOp<float>, cameraOp)
            CALL_BWD_WITH_OP(193, OrthographicWithDistortionCameraOp<float>, cameraOp)
            CALL_BWD_WITH_OP(256, OrthographicWithDistortionCameraOp<float>, cameraOp)
            CALL_BWD_WITH_OP(257, OrthographicWithDistortionCameraOp<float>, cameraOp)
            CALL_BWD_WITH_OP(512, OrthographicWithDistortionCameraOp<float>, cameraOp)
            CALL_BWD_WITH_OP(513, OrthographicWithDistortionCameraOp<float>, cameraOp)
        default:
            TORCH_CHECK_VALUE(false,
                              "Unsupported channels for rasterize-from-world-3dgs backward: ",
                              channels);
        }
    } else {
        const PerspectiveWithDistortionCameraOp<float> cameraOp{
            worldToCamMatricesStart.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            worldToCamMatricesEnd.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            projectionMatrices.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            distortionCoeffs.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            static_cast<uint32_t>(C),
            distortionCoeffs.size(1),
            (int32_t)imageWidth,
            (int32_t)imageHeight,
            (int32_t)imageOriginW,
            (int32_t)imageOriginH,
            rollingShutterType,
            cameraModel};
        switch (channels) {
            CALL_BWD_WITH_OP(1, PerspectiveWithDistortionCameraOp<float>, cameraOp)
            CALL_BWD_WITH_OP(2, PerspectiveWithDistortionCameraOp<float>, cameraOp)
            CALL_BWD_WITH_OP(3, PerspectiveWithDistortionCameraOp<float>, cameraOp)
            CALL_BWD_WITH_OP(4, PerspectiveWithDistortionCameraOp<float>, cameraOp)
            CALL_BWD_WITH_OP(5, PerspectiveWithDistortionCameraOp<float>, cameraOp)
            CALL_BWD_WITH_OP(8, PerspectiveWithDistortionCameraOp<float>, cameraOp)
            CALL_BWD_WITH_OP(9, PerspectiveWithDistortionCameraOp<float>, cameraOp)
            CALL_BWD_WITH_OP(16, PerspectiveWithDistortionCameraOp<float>, cameraOp)
            CALL_BWD_WITH_OP(17, PerspectiveWithDistortionCameraOp<float>, cameraOp)
            CALL_BWD_WITH_OP(32, PerspectiveWithDistortionCameraOp<float>, cameraOp)
            CALL_BWD_WITH_OP(33, PerspectiveWithDistortionCameraOp<float>, cameraOp)
            CALL_BWD_WITH_OP(64, PerspectiveWithDistortionCameraOp<float>, cameraOp)
            CALL_BWD_WITH_OP(65, PerspectiveWithDistortionCameraOp<float>, cameraOp)
            CALL_BWD_WITH_OP(128, PerspectiveWithDistortionCameraOp<float>, cameraOp)
            CALL_BWD_WITH_OP(129, PerspectiveWithDistortionCameraOp<float>, cameraOp)
            CALL_BWD_WITH_OP(192, PerspectiveWithDistortionCameraOp<float>, cameraOp)
            CALL_BWD_WITH_OP(193, PerspectiveWithDistortionCameraOp<float>, cameraOp)
            CALL_BWD_WITH_OP(256, PerspectiveWithDistortionCameraOp<float>, cameraOp)
            CALL_BWD_WITH_OP(257, PerspectiveWithDistortionCameraOp<float>, cameraOp)
            CALL_BWD_WITH_OP(512, PerspectiveWithDistortionCameraOp<float>, cameraOp)
            CALL_BWD_WITH_OP(513, PerspectiveWithDistortionCameraOp<float>, cameraOp)
        default:
            TORCH_CHECK_VALUE(false,
                              "Unsupported channels for rasterize-from-world-3dgs backward: ",
                              channels);
        }
    }

#undef CALL_BWD_WITH_OP
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
                                                            const RenderSettings &,
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
