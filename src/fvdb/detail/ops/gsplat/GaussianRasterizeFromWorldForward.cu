// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/gsplat/GaussianRasterizeFromWorld.cuh>
#include <fvdb/detail/ops/gsplat/GaussianRasterizeFromWorldForward.h>
#include <fvdb/detail/ops/gsplat/GaussianRasterizeOptionalInputs.h>
#include <fvdb/detail/utils/Nvtx.h>

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
    nanovdb::math::Mat3<float> isclR; // S^{-1} R^T
    float opacity;
};

template <uint32_t NUM_CHANNELS, typename CameraOp> struct RasterizeFromWorldForwardArgs {
    RasterizeFromWorldCommonArgs commonArgs;
    CameraOp cameraOp;
    fvdb::TorchRAcc64<float, 4> outFeatures;  // [C,H,W,D]
    fvdb::TorchRAcc64<float, 4> outAlphas;    // [C,H,W,1]
    fvdb::TorchRAcc64<int32_t, 3> outLastIds; // [C,H,W]

    inline __device__ void
    volumeRenderTileForward() const {
        const uint32_t blockSize = blockDim.x * blockDim.y;
        auto block               = cg::this_thread_block();
        const auto &common       = commonArgs;

        uint32_t camId, tileRow, tileCol, row, col;
        common.denseCoordinates(camId, tileRow, tileCol, row, col);
        const bool inside     = (row < common.imageHeight && col < common.imageWidth);
        float *outFeaturesPtr = outFeatures.data() + camId * outFeatures.stride(0) +
                                row * outFeatures.stride(1) + col * outFeatures.stride(2);
        float *outAlphaPtr = outAlphas.data() + camId * outAlphas.stride(0) +
                             row * outAlphas.stride(1) + col * outAlphas.stride(2);
        int32_t *outLastIdPtr = outLastIds.data() + camId * outLastIds.stride(0) +
                                row * outLastIds.stride(1) + col * outLastIds.stride(2);

        // Parity with classic rasterizer: masked tiles write background and exit.
        //
        // IMPORTANT: this kernel uses block-level barriers later (`__syncthreads_count`,
        // `block.sync`). Any early return must be taken by *all* threads in the block, otherwise
        // edge tiles can deadlock when some threads are `!inside`. So we make the return
        // block-wide.
        const bool tileMasked = common.tileMasked(camId, tileRow, tileCol);
        if (tileMasked) {
            if (inside) {
                outAlphaPtr[0]  = 0.0f;
                outLastIdPtr[0] = -1;
#pragma unroll
                for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
                    outFeaturesPtr[k * outFeatures.stride(3)] = common.backgroundValue(camId, k);
                }
            }
            return;
        }

        extern __shared__ char smem[];
        CameraOp cameraOpLocal = cameraOp;
        cameraOpLocal.loadSharedMemory(smem);
        __syncthreads();

        const nanovdb::math::Ray<float> ray = cameraOpLocal.projectToRay(camId, row, col);

        const bool rayValid = ray.dir().dot(ray.dir()) > 0.0f;
        bool done           = (!inside) || (!rayValid);

        // Determine gaussian range for this tile.
        const auto [rangeStart, rangeEnd] = common.tileGaussianRange(camId, tileRow, tileCol);

        // If no intersections, just write background.
        //
        // As above, this must be a block-wide return to avoid deadlocks on edge tiles.
        if (rangeEnd <= rangeStart) {
            if (inside) {
                // alpha=0, output background if provided else 0.
                outAlphaPtr[0]  = 0.0f;
                outLastIdPtr[0] = -1;
#pragma unroll
                for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
                    outFeaturesPtr[k * outFeatures.stride(3)] = common.backgroundValue(camId, k);
                }
            }
            return;
        }

        // Shared memory for batched gaussians (after camera-op shared state).
        char *gaussSmem  = smem + cameraOpLocal.numSharedMemBytes();
        int32_t *idBatch = reinterpret_cast<int32_t *>(gaussSmem);                 // [blockSize]
        auto *gaussBatch =
            reinterpret_cast<SharedGaussian<NUM_CHANNELS> *>(&idBatch[blockSize]); // [blockSize]

        float transmittance = 1.0f;
        int32_t curIdx      = -1;
        float pixOut[NUM_CHANNELS];
#pragma unroll
        for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
            pixOut[k] = 0.f;
        }

        const int32_t nIsects     = rangeEnd - rangeStart;
        const uint32_t nBatches   = (nIsects + blockSize - 1) / blockSize;
        const uint32_t threadRank = block.thread_rank();

        for (uint32_t b = 0; b < nBatches; ++b) {
            if (__syncthreads_count(done) >= (int)blockSize) {
                break;
            }

            const int32_t batchStart = rangeStart + (int32_t)(blockSize * b);
            const int32_t idx        = batchStart + (int32_t)threadRank;
            if (idx < rangeEnd) {
                const int32_t flatId = common.tileGaussianIds[idx];
                idBatch[threadRank]  = flatId;
                const int32_t gid    = flatId % (int32_t)common.means.size(0);

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
                const int32_t cid                      = flatId / (int32_t)common.means.size(0);
                const float op                         = common.opacities[cid][gid];

                gaussBatch[threadRank].id      = flatId;
                gaussBatch[threadRank].mean    = mean_w;
                gaussBatch[threadRank].isclR   = isclR;
                gaussBatch[threadRank].opacity = op;
            }

            __syncthreads();

            const uint32_t batchSize =
                min((uint32_t)blockSize, (uint32_t)max(0, rangeEnd - batchStart));
            for (uint32_t t = 0; (t < batchSize) && !done; ++t) {
                const SharedGaussian<NUM_CHANNELS> g = gaussBatch[t];
                // 3DGS ray-ellipsoid visibility in "whitened" coordinates (see 3D-GUT paper
                // Fig. 11). gro   = S^{-1} R^T (o - μ) grd   = normalize( S^{-1} R^T d ) gcrod =
                // grd × gro  (distance proxy to principal axis in whitened space)
                const nanovdb::math::Vec3<float> gro   = g.isclR * (ray.eye() - g.mean);
                const nanovdb::math::Vec3<float> grd =
                    fvdb::detail::ops::normalizeSafe<float>(g.isclR * ray.dir());
                const nanovdb::math::Vec3<float> gcrod = grd.cross(gro);
                const float grayDist                   = gcrod.dot(gcrod);
                const float power                      = -0.5f * grayDist;
                const float vis                        = __expf(power);
                float alpha                            = min(kAlphaThreshold, g.opacity * vis);
                if (power > 0.f || alpha < 1.f / 255.f) {
                    continue;
                }
                const float nextTransmittance = transmittance * (1.0f - alpha);
                if (nextTransmittance <= 1e-4f) {
                    done = true;
                    break;
                }
                const float contrib = alpha * transmittance;
                const int32_t cid   = g.id / (int32_t)common.means.size(0);
                const int32_t gid   = g.id % (int32_t)common.means.size(0);
#pragma unroll
                for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
                    pixOut[k] += common.features[cid][gid][k] * contrib;
                }
                curIdx        = (uint32_t)(batchStart + (int32_t)t);
                transmittance = nextTransmittance;
            }
        }

        if (!inside) {
            return;
        }

        outAlphaPtr[0]  = 1.0f - transmittance;
        outLastIdPtr[0] = curIdx;
#pragma unroll
        for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
            outFeaturesPtr[k * outFeatures.stride(3)] =
                pixOut[k] + transmittance * common.backgroundValue(camId, k);
        }
    }
};

template <uint32_t NUM_CHANNELS, typename CameraOp>
__global__ void
rasterizeGaussiansFromWorld(const RasterizeFromWorldForwardArgs<NUM_CHANNELS, CameraOp> args) {
    args.volumeRenderTileForward();
}

template <uint32_t NUM_CHANNELS, typename CameraOp>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
launchForward(const torch::Tensor &means,
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
              const at::optional<torch::Tensor> &backgrounds,
              const at::optional<torch::Tensor> &masks) {
    const int64_t C = features.size(0);

    auto opts = features.options();
    torch::Tensor outFeatures =
        torch::zeros({C, (int64_t)imageHeight, (int64_t)imageWidth, (int64_t)NUM_CHANNELS}, opts);
    torch::Tensor outAlphas = torch::zeros({C, (int64_t)imageHeight, (int64_t)imageWidth, 1}, opts);
    torch::Tensor outLastIds =
        torch::zeros({C, (int64_t)imageHeight, (int64_t)imageWidth},
                     torch::TensorOptions().dtype(torch::kInt32).device(features.device()));

    const uint32_t tileExtentW = (imageWidth + tileSize - 1) / tileSize;
    const uint32_t tileExtentH = (imageHeight + tileSize - 1) / tileSize;
    const dim3 blockDim(tileSize, tileSize, 1);
    const dim3 gridDim(C * tileExtentH * tileExtentW, 1, 1);

    const int32_t totalIntersections = (int32_t)tileGaussianIds.size(0);

    RasterizeFromWorldCommonArgs commonArgs{
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
    commonArgs.backgrounds = opt.backgrounds;
    commonArgs.masks       = opt.masks;

    RasterizeFromWorldForwardArgs<NUM_CHANNELS, CameraOp> args{
        commonArgs,
        cameraOp,
        outFeatures.packed_accessor64<float, 4, torch::RestrictPtrTraits>(),
        outAlphas.packed_accessor64<float, 4, torch::RestrictPtrTraits>(),
        outLastIds.packed_accessor64<int32_t, 3, torch::RestrictPtrTraits>()};

    const size_t blockSize = (size_t)tileSize * (size_t)tileSize;
    const size_t sharedMem = cameraOp.numSharedMemBytes() +
                             blockSize * (sizeof(int32_t) + sizeof(SharedGaussian<NUM_CHANNELS>));

    auto stream = at::cuda::getDefaultCUDAStream();
    rasterizeGaussiansFromWorld<NUM_CHANNELS, CameraOp><<<gridDim, blockDim, sharedMem, stream>>>(
        args);

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
    const DistortionModel cameraModel,
    const RenderSettings &settings,
    const torch::Tensor &tileOffsets,
    const torch::Tensor &tileGaussianIds,
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
    TORCH_CHECK_VALUE(tileOffsets.is_cuda(), "tileOffsets must be CUDA");
    TORCH_CHECK_VALUE(tileGaussianIds.is_cuda(), "tileGaussianIds must be CUDA");

    TORCH_CHECK_VALUE(means.dim() == 2 && means.size(1) == 3, "means must have shape [N,3]");
    TORCH_CHECK_VALUE(quats.dim() == 2 && quats.size(1) == 4, "quats must have shape [N,4]");
    TORCH_CHECK_VALUE(logScales.dim() == 2 && logScales.size(1) == 3,
                      "logScales must have shape [N,3]");
    TORCH_CHECK_VALUE(features.dim() == 3, "features must have shape [C,N,D]");

    const int64_t C = features.size(0);
    const int64_t N = means.size(0);
    TORCH_CHECK_VALUE(features.size(1) == N, "features must have shape [C,N,D] matching N");
    TORCH_CHECK_VALUE(opacities.is_cuda(), "opacities must be CUDA");

    // Opacities may be provided either per-camera ([C,N]) or shared across cameras ([N]).
    // TODO(fvdb): Avoid materializing a repeated [C,N] tensor when opacities are shared across
    // cameras. We should be able to pass a view (or otherwise avoid the repeat) similar to the
    // approach used in PR #451 for other rasterization paths.
    torch::Tensor opacitiesBatched = opacities;
    if (opacitiesBatched.dim() == 1) {
        TORCH_CHECK_VALUE(opacitiesBatched.size(0) == N,
                          "opacities must have shape [N] or [C,N] matching N");
        opacitiesBatched = opacitiesBatched.unsqueeze(0).repeat({C, 1});
    }
    TORCH_CHECK_VALUE(opacitiesBatched.dim() == 2, "opacities must have shape [C,N]");
    TORCH_CHECK_VALUE(opacitiesBatched.size(0) == C && opacitiesBatched.size(1) == N,
                      "opacities must have shape [C,N] matching features and N");
    opacitiesBatched = opacitiesBatched.contiguous();

    TORCH_CHECK_VALUE(worldToCamMatricesStart.sizes() == torch::IntArrayRef({C, 4, 4}),
                      "worldToCamMatricesStart must have shape [C,4,4]");
    TORCH_CHECK_VALUE(worldToCamMatricesEnd.sizes() == torch::IntArrayRef({C, 4, 4}),
                      "worldToCamMatricesEnd must have shape [C,4,4]");
    TORCH_CHECK_VALUE(projectionMatrices.sizes() == torch::IntArrayRef({C, 3, 3}),
                      "projectionMatrices must have shape [C,3,3]");

    const int64_t numDistCoeffs = distortionCoeffs.size(1);
    TORCH_CHECK_VALUE(distortionCoeffs.dim() == 2 && distortionCoeffs.size(0) == C,
                      "distortionCoeffs must have shape [C,K]");
    if (cameraModel == DistortionModel::OPENCV_RADTAN_5 ||
        cameraModel == DistortionModel::OPENCV_RATIONAL_8 ||
        cameraModel == DistortionModel::OPENCV_RADTAN_THIN_PRISM_9 ||
        cameraModel == DistortionModel::OPENCV_THIN_PRISM_12) {
        TORCH_CHECK_VALUE(numDistCoeffs == 12,
                          "For DistortionModel::OPENCV_* distortionCoeffs must be [C,12]");
    }

    const uint32_t channels = (uint32_t)features.size(2);

#define CALL_FWD_WITH_OP(NCH, OP_TYPE, OP_VAL)            \
    case NCH:                                              \
        return launchForward<NCH, OP_TYPE>(means,          \
                                           quats,          \
                                           logScales,      \
                                           features,       \
                                           opacitiesBatched, \
                                           OP_VAL,                  \
                                           imageWidth,              \
                                           imageHeight,             \
                                           imageOriginW,            \
                                           imageOriginH,            \
                                           tileSize,                \
                                           tileOffsets,             \
                                           tileGaussianIds,         \
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
            CALL_FWD_WITH_OP(1, OrthographicWithDistortionCameraOp<float>, cameraOp)
            CALL_FWD_WITH_OP(2, OrthographicWithDistortionCameraOp<float>, cameraOp)
            CALL_FWD_WITH_OP(3, OrthographicWithDistortionCameraOp<float>, cameraOp)
            CALL_FWD_WITH_OP(4, OrthographicWithDistortionCameraOp<float>, cameraOp)
            CALL_FWD_WITH_OP(5, OrthographicWithDistortionCameraOp<float>, cameraOp)
            CALL_FWD_WITH_OP(8, OrthographicWithDistortionCameraOp<float>, cameraOp)
            CALL_FWD_WITH_OP(9, OrthographicWithDistortionCameraOp<float>, cameraOp)
            CALL_FWD_WITH_OP(16, OrthographicWithDistortionCameraOp<float>, cameraOp)
            CALL_FWD_WITH_OP(17, OrthographicWithDistortionCameraOp<float>, cameraOp)
            CALL_FWD_WITH_OP(32, OrthographicWithDistortionCameraOp<float>, cameraOp)
            CALL_FWD_WITH_OP(33, OrthographicWithDistortionCameraOp<float>, cameraOp)
            CALL_FWD_WITH_OP(64, OrthographicWithDistortionCameraOp<float>, cameraOp)
            CALL_FWD_WITH_OP(65, OrthographicWithDistortionCameraOp<float>, cameraOp)
            CALL_FWD_WITH_OP(128, OrthographicWithDistortionCameraOp<float>, cameraOp)
            CALL_FWD_WITH_OP(129, OrthographicWithDistortionCameraOp<float>, cameraOp)
            CALL_FWD_WITH_OP(192, OrthographicWithDistortionCameraOp<float>, cameraOp)
            CALL_FWD_WITH_OP(193, OrthographicWithDistortionCameraOp<float>, cameraOp)
            CALL_FWD_WITH_OP(256, OrthographicWithDistortionCameraOp<float>, cameraOp)
            CALL_FWD_WITH_OP(257, OrthographicWithDistortionCameraOp<float>, cameraOp)
            CALL_FWD_WITH_OP(512, OrthographicWithDistortionCameraOp<float>, cameraOp)
            CALL_FWD_WITH_OP(513, OrthographicWithDistortionCameraOp<float>, cameraOp)
        default:
            TORCH_CHECK_VALUE(false,
                              "Unsupported channels for rasterize-from-world-3dgs: ",
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
            CALL_FWD_WITH_OP(1, PerspectiveWithDistortionCameraOp<float>, cameraOp)
            CALL_FWD_WITH_OP(2, PerspectiveWithDistortionCameraOp<float>, cameraOp)
            CALL_FWD_WITH_OP(3, PerspectiveWithDistortionCameraOp<float>, cameraOp)
            CALL_FWD_WITH_OP(4, PerspectiveWithDistortionCameraOp<float>, cameraOp)
            CALL_FWD_WITH_OP(5, PerspectiveWithDistortionCameraOp<float>, cameraOp)
            CALL_FWD_WITH_OP(8, PerspectiveWithDistortionCameraOp<float>, cameraOp)
            CALL_FWD_WITH_OP(9, PerspectiveWithDistortionCameraOp<float>, cameraOp)
            CALL_FWD_WITH_OP(16, PerspectiveWithDistortionCameraOp<float>, cameraOp)
            CALL_FWD_WITH_OP(17, PerspectiveWithDistortionCameraOp<float>, cameraOp)
            CALL_FWD_WITH_OP(32, PerspectiveWithDistortionCameraOp<float>, cameraOp)
            CALL_FWD_WITH_OP(33, PerspectiveWithDistortionCameraOp<float>, cameraOp)
            CALL_FWD_WITH_OP(64, PerspectiveWithDistortionCameraOp<float>, cameraOp)
            CALL_FWD_WITH_OP(65, PerspectiveWithDistortionCameraOp<float>, cameraOp)
            CALL_FWD_WITH_OP(128, PerspectiveWithDistortionCameraOp<float>, cameraOp)
            CALL_FWD_WITH_OP(129, PerspectiveWithDistortionCameraOp<float>, cameraOp)
            CALL_FWD_WITH_OP(192, PerspectiveWithDistortionCameraOp<float>, cameraOp)
            CALL_FWD_WITH_OP(193, PerspectiveWithDistortionCameraOp<float>, cameraOp)
            CALL_FWD_WITH_OP(256, PerspectiveWithDistortionCameraOp<float>, cameraOp)
            CALL_FWD_WITH_OP(257, PerspectiveWithDistortionCameraOp<float>, cameraOp)
            CALL_FWD_WITH_OP(512, PerspectiveWithDistortionCameraOp<float>, cameraOp)
            CALL_FWD_WITH_OP(513, PerspectiveWithDistortionCameraOp<float>, cameraOp)
        default:
            TORCH_CHECK_VALUE(false,
                              "Unsupported channels for rasterize-from-world-3dgs: ",
                              channels);
        }
    }

#undef CALL_FWD_WITH_OP
}

template <>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianRasterizeFromWorld3DGSForward<torch::kCPU>(const torch::Tensor &,
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
                                                           const at::optional<torch::Tensor> &,
                                                           const at::optional<torch::Tensor> &) {
    TORCH_CHECK_VALUE(false, "dispatchGaussianRasterizeFromWorld3DGSForward is CUDA-only");
}

} // namespace fvdb::detail::ops
