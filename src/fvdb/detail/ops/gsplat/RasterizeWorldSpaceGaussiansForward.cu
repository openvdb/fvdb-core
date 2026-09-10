// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/gsplat/RasterizeWorldSpaceGaussiansForward.h>
#include <fvdb/detail/utils/Nvtx.h>
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/cuda/Prefetch.h>
#include <fvdb/detail/utils/cuda/Utils.cuh>
#include <fvdb/detail/utils/gsplat/GaussianRasterizeFromWorld.cuh>
#include <fvdb/detail/utils/gsplat/GaussianRasterizeOptionalInputs.h>

#include <ATen/cuda/CUDAContext.h>
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
    nanovdb::math::Mat3<float> isclR; // S^{-1} R^T
    float opacity;
};

template <uint32_t NUM_CHANNELS, typename Camera> struct RasterizeFromWorldForwardArgs {
    RasterizeFromWorldCommonArgs commonArgs;
    Camera camera;
    uint32_t blockOffset;
    fvdb::TorchRAcc64<float, 4> outFeatures;  // [C,H,W,D]
    fvdb::TorchRAcc64<float, 4> outAlphas;    // [C,H,W,1]
    fvdb::TorchRAcc64<int32_t, 3> outLastIds; // [C,H,W]

    inline __device__ void
    volumeRenderTileForward() const {
        // Fully expanding every channel operation makes the 512/513-channel kernels prohibitively
        // expensive for ptxas to optimize, especially for SM 120. Preserve full unrolling for the
        // smaller kernels while keeping the generated code bounded for larger channel counts.
        constexpr uint32_t CHANNEL_UNROLL = NUM_CHANNELS <= 32 ? 32 : 4;

        const uint32_t blockSize = blockDim.x * blockDim.y;
        auto block               = cg::this_thread_block();
        const auto &common       = commonArgs;

        uint32_t camId, tileRow, tileCol, row, col;
        common.denseCoordinates(camId, tileRow, tileCol, row, col, blockOffset);
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
#pragma unroll CHANNEL_UNROLL
                for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
                    outFeaturesPtr[k * outFeatures.stride(3)] = common.backgroundValue(camId, k);
                }
            }
            return;
        }

        extern __shared__ char smem[];
        Camera cameraLocal = camera;
        uintptr_t smemAddr = reinterpret_cast<uintptr_t>(smem);
        smemAddr           = alignUpAddress(smemAddr, alignof(nanovdb::math::Mat3<float>));
        cameraLocal.loadSharedMemory(reinterpret_cast<void *>(smemAddr));
        __syncthreads();

        const nanovdb::math::Ray<float> ray = cameraLocal.unprojectPixelToRay(camId, row, col);

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
#pragma unroll CHANNEL_UNROLL
                for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
                    outFeaturesPtr[k * outFeatures.stride(3)] = common.backgroundValue(camId, k);
                }
            }
            return;
        }

        // Shared memory for batched gaussians (after camera-op shared state).
        uintptr_t gaussAddr = smemAddr + cameraLocal.numSharedMemBytes();
        gaussAddr           = alignUpAddress(gaussAddr, alignof(int32_t));
        int32_t *idBatch    = reinterpret_cast<int32_t *>(gaussAddr); // [blockSize]
        gaussAddr += static_cast<size_t>(blockSize) * sizeof(int32_t);
        gaussAddr = alignUpAddress(gaussAddr, alignof(SharedGaussian<NUM_CHANNELS>));
        auto *gaussBatch =
            reinterpret_cast<SharedGaussian<NUM_CHANNELS> *>(gaussAddr); // [blockSize]

        float transmittance            = 1.0f;
        int32_t lastIntersectionOffset = -1;
        float pixOut[NUM_CHANNELS];
#pragma unroll CHANNEL_UNROLL
        for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
            pixOut[k] = 0.f;
        }

        const int64_t nIsects     = rangeEnd - rangeStart;
        const int64_t nBatches    = (nIsects + blockSize - 1) / blockSize;
        const uint32_t threadRank = block.thread_rank();

        for (int64_t b = 0; b < nBatches; ++b) {
            if (__syncthreads_count(done) >= (int)blockSize) {
                break;
            }

            const int64_t batchOffset = blockSize * b;
            const int64_t batchStart  = rangeStart + batchOffset;
            const int64_t idx         = batchStart + threadRank;
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

            const int64_t remaining = rangeEnd - batchStart;
            const uint32_t batchSize =
                static_cast<uint32_t>(remaining < blockSize ? remaining : blockSize);
            for (uint32_t t = 0; (t < batchSize) && !done; ++t) {
                const SharedGaussian<NUM_CHANNELS> g = gaussBatch[t];
                // 3DGS ray-ellipsoid visibility in "whitened" coordinates (see 3D-GUT paper
                // Fig. 11). gro   = S^{-1} R^T (o - μ) grd   = normalize( S^{-1} R^T d ) gcrod =
                // grd × gro  (distance proxy to principal axis in whitened space)
                const nanovdb::math::Vec3<float> gro = g.isclR * (ray.eye() - g.mean);
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
                if (nextTransmittance <= kTransmittanceThreshold) {
                    done = true;
                    break;
                }
                const float contrib = alpha * transmittance;
                const int32_t cid   = g.id / (int32_t)common.means.size(0);
                const int32_t gid   = g.id % (int32_t)common.means.size(0);
#pragma unroll CHANNEL_UNROLL
                for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
                    pixOut[k] += common.features[cid][gid][k] * contrib;
                }
                lastIntersectionOffset = static_cast<int32_t>(batchOffset + t);
                transmittance          = nextTransmittance;
            }
        }

        if (!inside) {
            return;
        }

        outAlphaPtr[0]  = 1.0f - transmittance;
        outLastIdPtr[0] = lastIntersectionOffset;
#pragma unroll CHANNEL_UNROLL
        for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
            outFeaturesPtr[k * outFeatures.stride(3)] =
                pixOut[k] + transmittance * common.backgroundValue(camId, k);
        }
    }
};

template <uint32_t NUM_CHANNELS, typename Camera>
__global__ void
rasterizeGaussiansFromWorld(const RasterizeFromWorldForwardArgs<NUM_CHANNELS, Camera> args) {
    args.volumeRenderTileForward();
}

template <uint32_t NUM_CHANNELS, typename Camera>
void
launchForwardKernel(const torch::Tensor &means,
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
                    const at::optional<torch::Tensor> &backgrounds,
                    const at::optional<torch::Tensor> &masks,
                    const torch::Tensor &outFeatures,
                    const torch::Tensor &outAlphas,
                    const torch::Tensor &outLastIds,
                    const uint32_t blockOffset,
                    const uint32_t blockCount,
                    const cudaStream_t stream) {
    const int64_t C = features.size(0);

    const uint32_t tileExtentW = (imageWidth + tileSize - 1) / tileSize;
    const uint32_t tileExtentH = (imageHeight + tileSize - 1) / tileSize;
    const dim3 blockDim(tileSize, tileSize, 1);
    const dim3 gridDim(blockCount, 1, 1);

    const int64_t totalIntersections = tileGaussianIds.size(0);

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
    commonArgs.backgrounds = opt.backgrounds;
    commonArgs.masks       = opt.masks;

    RasterizeFromWorldForwardArgs<NUM_CHANNELS, Camera> args{
        commonArgs,
        camera,
        blockOffset,
        outFeatures.packed_accessor64<float, 4, torch::RestrictPtrTraits>(),
        outAlphas.packed_accessor64<float, 4, torch::RestrictPtrTraits>(),
        outLastIds.packed_accessor64<int32_t, 3, torch::RestrictPtrTraits>()};

    const size_t blockSize = (size_t)tileSize * (size_t)tileSize;
    size_t sharedMem       = (alignof(nanovdb::math::Mat3<float>) - 1) + camera.numSharedMemBytes();
    sharedMem += (alignof(int32_t) - 1) + blockSize * sizeof(int32_t);
    sharedMem += (alignof(SharedGaussian<NUM_CHANNELS>) - 1) +
                 blockSize * sizeof(SharedGaussian<NUM_CHANNELS>);

    if (cudaFuncSetAttribute(rasterizeGaussiansFromWorld<NUM_CHANNELS, Camera>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             sharedMem) != cudaSuccess) {
        AT_ERROR("Failed to set maximum shared memory size (requested ",
                 sharedMem,
                 " bytes), try lowering tile size or camera batch size.");
    }

    rasterizeGaussiansFromWorld<NUM_CHANNELS, Camera>
        <<<gridDim, blockDim, sharedMem, stream>>>(args);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <uint32_t NUM_CHANNELS, typename Camera>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
launchForwardCUDA(const torch::Tensor &means,
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
                  const at::optional<torch::Tensor> &backgrounds,
                  const at::optional<torch::Tensor> &masks) {
    const int64_t C = features.size(0);
    const auto opts = features.options();
    torch::Tensor outFeatures =
        torch::zeros({C, (int64_t)imageHeight, (int64_t)imageWidth, (int64_t)NUM_CHANNELS}, opts);
    torch::Tensor outAlphas = torch::zeros({C, (int64_t)imageHeight, (int64_t)imageWidth, 1}, opts);
    torch::Tensor outLastIds =
        torch::zeros({C, (int64_t)imageHeight, (int64_t)imageWidth},
                     torch::TensorOptions().dtype(torch::kInt32).device(features.device()));

    const uint32_t tileExtentW = (imageWidth + tileSize - 1) / tileSize;
    const uint32_t tileExtentH = (imageHeight + tileSize - 1) / tileSize;
    const uint32_t blockCount  = C * tileExtentH * tileExtentW;
    if (blockCount > 0) {
        auto stream = at::cuda::getCurrentCUDAStream(means.device().index());
        launchForwardKernel<NUM_CHANNELS, Camera>(means,
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
                                                  backgrounds,
                                                  masks,
                                                  outFeatures,
                                                  outAlphas,
                                                  outLastIds,
                                                  0,
                                                  blockCount,
                                                  stream);
    }
    return {outFeatures, outAlphas, outLastIds};
}

template <uint32_t NUM_CHANNELS, typename Camera>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
launchForwardPrivateUse1(const torch::Tensor &means,
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
                         const at::optional<torch::Tensor> &backgrounds,
                         const at::optional<torch::Tensor> &masks) {
    TORCH_CHECK(tileSize > 0, "Tile size must be greater than 0");

    const int64_t C               = features.size(0);
    const uint32_t tileExtentW    = (imageWidth + tileSize - 1) / tileSize;
    const uint32_t tileExtentH    = (imageHeight + tileSize - 1) / tileSize;
    const uint32_t tilesPerCamera = tileExtentH * tileExtentW;
    const uint32_t tileCount      = C * tilesPerCamera;

    // Every in-bounds pixel is written by exactly one tile, including masked tiles and tiles with
    // no intersections, so the managed outputs do not need a separate initialization kernel.
    const auto opts = features.options();
    torch::Tensor outFeatures =
        torch::empty({C, (int64_t)imageHeight, (int64_t)imageWidth, (int64_t)NUM_CHANNELS}, opts);
    torch::Tensor outAlphas = torch::empty({C, (int64_t)imageHeight, (int64_t)imageWidth, 1}, opts);
    torch::Tensor outLastIds =
        torch::empty({C, (int64_t)imageHeight, (int64_t)imageWidth}, opts.dtype(torch::kInt32));

    if (tileCount == 0) {
        return {outFeatures, outAlphas, outLastIds};
    }

    const at::optional<torch::Tensor> contiguousBackgrounds =
        backgrounds.has_value() ? std::make_optional(backgrounds.value().contiguous())
                                : std::nullopt;
    const at::optional<torch::Tensor> contiguousMasks =
        masks.has_value() ? std::make_optional(masks.value().contiguous()) : std::nullopt;
    const torch::Tensor contiguousTileOffsets     = tileOffsets.contiguous();
    const torch::Tensor contiguousTileGaussianIds = tileGaussianIds.contiguous();

    const auto deviceCount = c10::cuda::device_count();
    std::vector<cudaEvent_t> events(deviceCount);
    for (const auto deviceId: c10::irange(deviceCount)) {
        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        auto stream = c10::cuda::getCurrentCUDAStream(deviceId);
        C10_CUDA_CHECK(cudaEventCreateWithFlags(&events[deviceId], cudaEventDisableTiming));
        C10_CUDA_CHECK(cudaEventRecord(events[deviceId], stream));
    }

    std::vector<torch::Tensor> tileTensors = {
        contiguousTileOffsets, outFeatures, outAlphas, outLastIds};
    if (contiguousMasks.has_value()) {
        tileTensors.emplace_back(contiguousMasks.value());
    }
    std::vector<torch::Tensor> cameraTensors = {features, opacities};
    if (contiguousBackgrounds.has_value()) {
        cameraTensors.emplace_back(contiguousBackgrounds.value());
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
            const uint32_t cameraOffset = static_cast<uint32_t>(deviceTileOffset) / tilesPerCamera;
            const uint32_t cameraCount =
                cuda::ceil_div(static_cast<uint32_t>(deviceTileOffset + deviceTileCount),
                               tilesPerCamera) -
                cameraOffset;
            appendPerCameraPrefetchRanges(
                prefetchPointers, prefetchSizes, cameraTensors, cameraOffset, cameraCount);
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
        if (deviceTileCount > 0) {
            launchForwardKernel<NUM_CHANNELS, Camera>(means,
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
                                                      contiguousTileOffsets,
                                                      contiguousTileGaussianIds,
                                                      contiguousBackgrounds,
                                                      contiguousMasks,
                                                      outFeatures,
                                                      outAlphas,
                                                      outLastIds,
                                                      static_cast<uint32_t>(deviceTileOffset),
                                                      static_cast<uint32_t>(deviceTileCount),
                                                      stream);
        }
    }

    mergeStreams();
    return {outFeatures, outAlphas, outLastIds};
}

} // namespace

template <torch::DeviceType>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianRasterizeFromWorld3DGSForward(const torch::Tensor &means,
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
                                              const at::optional<torch::Tensor> &backgrounds,
                                              const at::optional<torch::Tensor> &masks);

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

    const at::cuda::OptionalCUDAGuard deviceGuard(device_of(means));

    TORCH_CHECK_VALUE(means.is_cuda(), "means must be CUDA");
    TORCH_CHECK_VALUE(quats.is_cuda(), "quats must be CUDA");
    TORCH_CHECK_VALUE(logScales.is_cuda(), "logScales must be CUDA");
    TORCH_CHECK_VALUE(features.is_cuda(), "features must be CUDA");
    TORCH_CHECK_VALUE(opacities.is_cuda(), "opacities must be CUDA");
    TORCH_CHECK_VALUE(worldToCamMatricesStart.is_cuda(), "worldToCamMatricesStart must be CUDA");
    TORCH_CHECK_VALUE(worldToCamMatricesEnd.is_cuda(), "worldToCamMatricesEnd must be CUDA");
    TORCH_CHECK_VALUE(projectionMatrices.is_cuda(), "projectionMatrices must be CUDA");
    TORCH_CHECK_VALUE(distortionCoeffs.is_cuda(), "distortionCoeffs must be CUDA");
    TORCH_CHECK_VALUE(tileOffsets.is_cuda(), "tileOffsets must be CUDA");
    TORCH_CHECK_VALUE(tileGaussianIds.is_cuda(), "tileGaussianIds must be CUDA");
    TORCH_CHECK_VALUE(tileOffsets.scalar_type() == torch::kInt64,
                      "tileOffsets must have dtype int64");

    TORCH_CHECK_VALUE(means.dim() == 2 && means.size(1) == 3, "means must have shape [N,3]");
    TORCH_CHECK_VALUE(quats.dim() == 2 && quats.size(1) == 4, "quats must have shape [N,4]");
    TORCH_CHECK_VALUE(logScales.dim() == 2 && logScales.size(1) == 3,
                      "logScales must have shape [N,3]");
    TORCH_CHECK_VALUE(features.dim() == 3, "features must have shape [C,N,D]");

    const int64_t C = features.size(0);
    const int64_t N = means.size(0);
    TORCH_CHECK_VALUE(features.size(1) == N, "features must have shape [C,N,D] matching N");

    TORCH_CHECK_VALUE(opacities.dim() == 2, "opacities must have shape [C,N]");
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
    if (cameraModel == DistortionModel::OPENCV_RADTAN_5 ||
        cameraModel == DistortionModel::OPENCV_RATIONAL_8 ||
        cameraModel == DistortionModel::OPENCV_RADTAN_THIN_PRISM_9 ||
        cameraModel == DistortionModel::OPENCV_THIN_PRISM_12) {
        TORCH_CHECK_VALUE(numDistCoeffs == 12,
                          "For DistortionModel::OPENCV_* distortionCoeffs must be [C,12]");
    }

    const uint32_t channels = (uint32_t)features.size(2);

#define CALL_FWD_WITH_OP(NCH, OP_TYPE, OP_VAL)                  \
    case NCH:                                                   \
        return launchForwardCUDA<NCH, OP_TYPE>(means,           \
                                               quats,           \
                                               logScales,       \
                                               features,        \
                                               opacities,       \
                                               OP_VAL,          \
                                               imageWidth,      \
                                               imageHeight,     \
                                               imageOriginW,    \
                                               imageOriginH,    \
                                               tileSize,        \
                                               tileOffsets,     \
                                               tileGaussianIds, \
                                               backgrounds,     \
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
            CALL_FWD_WITH_OP(1, OrthographicWithDistortionCamera<float>, camera)
            CALL_FWD_WITH_OP(2, OrthographicWithDistortionCamera<float>, camera)
            CALL_FWD_WITH_OP(3, OrthographicWithDistortionCamera<float>, camera)
            CALL_FWD_WITH_OP(4, OrthographicWithDistortionCamera<float>, camera)
            CALL_FWD_WITH_OP(5, OrthographicWithDistortionCamera<float>, camera)
            CALL_FWD_WITH_OP(8, OrthographicWithDistortionCamera<float>, camera)
            CALL_FWD_WITH_OP(9, OrthographicWithDistortionCamera<float>, camera)
            CALL_FWD_WITH_OP(16, OrthographicWithDistortionCamera<float>, camera)
            CALL_FWD_WITH_OP(17, OrthographicWithDistortionCamera<float>, camera)
            CALL_FWD_WITH_OP(32, OrthographicWithDistortionCamera<float>, camera)
            CALL_FWD_WITH_OP(33, OrthographicWithDistortionCamera<float>, camera)
            CALL_FWD_WITH_OP(64, OrthographicWithDistortionCamera<float>, camera)
            CALL_FWD_WITH_OP(65, OrthographicWithDistortionCamera<float>, camera)
            CALL_FWD_WITH_OP(128, OrthographicWithDistortionCamera<float>, camera)
            CALL_FWD_WITH_OP(129, OrthographicWithDistortionCamera<float>, camera)
            CALL_FWD_WITH_OP(192, OrthographicWithDistortionCamera<float>, camera)
            CALL_FWD_WITH_OP(193, OrthographicWithDistortionCamera<float>, camera)
            CALL_FWD_WITH_OP(256, OrthographicWithDistortionCamera<float>, camera)
            CALL_FWD_WITH_OP(257, OrthographicWithDistortionCamera<float>, camera)
            CALL_FWD_WITH_OP(512, OrthographicWithDistortionCamera<float>, camera)
            CALL_FWD_WITH_OP(513, OrthographicWithDistortionCamera<float>, camera)
        default:
            TORCH_CHECK_VALUE(
                false, "Unsupported channels for rasterize-from-world-3dgs: ", channels);
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
            CALL_FWD_WITH_OP(1, PerspectiveWithDistortionCamera<float>, camera)
            CALL_FWD_WITH_OP(2, PerspectiveWithDistortionCamera<float>, camera)
            CALL_FWD_WITH_OP(3, PerspectiveWithDistortionCamera<float>, camera)
            CALL_FWD_WITH_OP(4, PerspectiveWithDistortionCamera<float>, camera)
            CALL_FWD_WITH_OP(5, PerspectiveWithDistortionCamera<float>, camera)
            CALL_FWD_WITH_OP(8, PerspectiveWithDistortionCamera<float>, camera)
            CALL_FWD_WITH_OP(9, PerspectiveWithDistortionCamera<float>, camera)
            CALL_FWD_WITH_OP(16, PerspectiveWithDistortionCamera<float>, camera)
            CALL_FWD_WITH_OP(17, PerspectiveWithDistortionCamera<float>, camera)
            CALL_FWD_WITH_OP(32, PerspectiveWithDistortionCamera<float>, camera)
            CALL_FWD_WITH_OP(33, PerspectiveWithDistortionCamera<float>, camera)
            CALL_FWD_WITH_OP(64, PerspectiveWithDistortionCamera<float>, camera)
            CALL_FWD_WITH_OP(65, PerspectiveWithDistortionCamera<float>, camera)
            CALL_FWD_WITH_OP(128, PerspectiveWithDistortionCamera<float>, camera)
            CALL_FWD_WITH_OP(129, PerspectiveWithDistortionCamera<float>, camera)
            CALL_FWD_WITH_OP(192, PerspectiveWithDistortionCamera<float>, camera)
            CALL_FWD_WITH_OP(193, PerspectiveWithDistortionCamera<float>, camera)
            CALL_FWD_WITH_OP(256, PerspectiveWithDistortionCamera<float>, camera)
            CALL_FWD_WITH_OP(257, PerspectiveWithDistortionCamera<float>, camera)
            CALL_FWD_WITH_OP(512, PerspectiveWithDistortionCamera<float>, camera)
            CALL_FWD_WITH_OP(513, PerspectiveWithDistortionCamera<float>, camera)
        default:
            TORCH_CHECK_VALUE(
                false, "Unsupported channels for rasterize-from-world-3dgs: ", channels);
        }
    }

#undef CALL_FWD_WITH_OP
}

template <>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianRasterizeFromWorld3DGSForward<torch::kPrivateUse1>(
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
    const at::optional<torch::Tensor> &backgrounds,
    const at::optional<torch::Tensor> &masks) {
    FVDB_FUNC_RANGE();

    TORCH_CHECK_VALUE(means.is_privateuseone(), "means must be PrivateUse1");
    TORCH_CHECK_VALUE(quats.is_privateuseone(), "quats must be PrivateUse1");
    TORCH_CHECK_VALUE(logScales.is_privateuseone(), "logScales must be PrivateUse1");
    TORCH_CHECK_VALUE(features.is_privateuseone(), "features must be PrivateUse1");
    TORCH_CHECK_VALUE(opacities.is_privateuseone(), "opacities must be PrivateUse1");
    TORCH_CHECK_VALUE(worldToCamMatricesStart.is_privateuseone(),
                      "worldToCamMatricesStart must be PrivateUse1");
    TORCH_CHECK_VALUE(worldToCamMatricesEnd.is_privateuseone(),
                      "worldToCamMatricesEnd must be PrivateUse1");
    TORCH_CHECK_VALUE(projectionMatrices.is_privateuseone(),
                      "projectionMatrices must be PrivateUse1");
    TORCH_CHECK_VALUE(distortionCoeffs.is_privateuseone(), "distortionCoeffs must be PrivateUse1");
    TORCH_CHECK_VALUE(tileOffsets.is_privateuseone(), "tileOffsets must be PrivateUse1");
    TORCH_CHECK_VALUE(tileGaussianIds.is_privateuseone(), "tileGaussianIds must be PrivateUse1");
    TORCH_CHECK_VALUE(tileOffsets.scalar_type() == torch::kInt64,
                      "tileOffsets must have dtype int64");

    TORCH_CHECK_VALUE(means.dim() == 2 && means.size(1) == 3, "means must have shape [N,3]");
    TORCH_CHECK_VALUE(quats.dim() == 2 && quats.size(1) == 4, "quats must have shape [N,4]");
    TORCH_CHECK_VALUE(logScales.dim() == 2 && logScales.size(1) == 3,
                      "logScales must have shape [N,3]");
    TORCH_CHECK_VALUE(features.dim() == 3, "features must have shape [C,N,D]");

    const int64_t C = features.size(0);
    const int64_t N = means.size(0);
    TORCH_CHECK_VALUE(features.size(1) == N, "features must have shape [C,N,D] matching N");

    TORCH_CHECK_VALUE(opacities.dim() == 2, "opacities must have shape [C,N]");
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
    if (cameraModel == DistortionModel::OPENCV_RADTAN_5 ||
        cameraModel == DistortionModel::OPENCV_RATIONAL_8 ||
        cameraModel == DistortionModel::OPENCV_RADTAN_THIN_PRISM_9 ||
        cameraModel == DistortionModel::OPENCV_THIN_PRISM_12) {
        TORCH_CHECK_VALUE(numDistCoeffs == 12,
                          "For DistortionModel::OPENCV_* distortionCoeffs must be [C,12]");
    }

    const uint32_t channels = (uint32_t)features.size(2);

#define CALL_FWD_PRIVATEUSE1_WITH_OP(NCH, OP_TYPE, OP_VAL)             \
    case NCH:                                                          \
        return launchForwardPrivateUse1<NCH, OP_TYPE>(means,           \
                                                      quats,           \
                                                      logScales,       \
                                                      features,        \
                                                      opacities,       \
                                                      OP_VAL,          \
                                                      imageWidth,      \
                                                      imageHeight,     \
                                                      imageOriginW,    \
                                                      imageOriginH,    \
                                                      tileSize,        \
                                                      tileOffsets,     \
                                                      tileGaussianIds, \
                                                      backgrounds,     \
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
            CALL_FWD_PRIVATEUSE1_WITH_OP(1, OrthographicWithDistortionCamera<float>, camera)
            CALL_FWD_PRIVATEUSE1_WITH_OP(2, OrthographicWithDistortionCamera<float>, camera)
            CALL_FWD_PRIVATEUSE1_WITH_OP(3, OrthographicWithDistortionCamera<float>, camera)
            CALL_FWD_PRIVATEUSE1_WITH_OP(4, OrthographicWithDistortionCamera<float>, camera)
            CALL_FWD_PRIVATEUSE1_WITH_OP(5, OrthographicWithDistortionCamera<float>, camera)
            CALL_FWD_PRIVATEUSE1_WITH_OP(8, OrthographicWithDistortionCamera<float>, camera)
            CALL_FWD_PRIVATEUSE1_WITH_OP(9, OrthographicWithDistortionCamera<float>, camera)
            CALL_FWD_PRIVATEUSE1_WITH_OP(16, OrthographicWithDistortionCamera<float>, camera)
            CALL_FWD_PRIVATEUSE1_WITH_OP(17, OrthographicWithDistortionCamera<float>, camera)
            CALL_FWD_PRIVATEUSE1_WITH_OP(32, OrthographicWithDistortionCamera<float>, camera)
            CALL_FWD_PRIVATEUSE1_WITH_OP(33, OrthographicWithDistortionCamera<float>, camera)
            CALL_FWD_PRIVATEUSE1_WITH_OP(64, OrthographicWithDistortionCamera<float>, camera)
            CALL_FWD_PRIVATEUSE1_WITH_OP(65, OrthographicWithDistortionCamera<float>, camera)
            CALL_FWD_PRIVATEUSE1_WITH_OP(128, OrthographicWithDistortionCamera<float>, camera)
            CALL_FWD_PRIVATEUSE1_WITH_OP(129, OrthographicWithDistortionCamera<float>, camera)
            CALL_FWD_PRIVATEUSE1_WITH_OP(192, OrthographicWithDistortionCamera<float>, camera)
            CALL_FWD_PRIVATEUSE1_WITH_OP(193, OrthographicWithDistortionCamera<float>, camera)
            CALL_FWD_PRIVATEUSE1_WITH_OP(256, OrthographicWithDistortionCamera<float>, camera)
            CALL_FWD_PRIVATEUSE1_WITH_OP(257, OrthographicWithDistortionCamera<float>, camera)
            CALL_FWD_PRIVATEUSE1_WITH_OP(512, OrthographicWithDistortionCamera<float>, camera)
            CALL_FWD_PRIVATEUSE1_WITH_OP(513, OrthographicWithDistortionCamera<float>, camera)
        default:
            TORCH_CHECK_VALUE(
                false, "Unsupported channels for rasterize-from-world-3dgs: ", channels);
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
            CALL_FWD_PRIVATEUSE1_WITH_OP(1, PerspectiveWithDistortionCamera<float>, camera)
            CALL_FWD_PRIVATEUSE1_WITH_OP(2, PerspectiveWithDistortionCamera<float>, camera)
            CALL_FWD_PRIVATEUSE1_WITH_OP(3, PerspectiveWithDistortionCamera<float>, camera)
            CALL_FWD_PRIVATEUSE1_WITH_OP(4, PerspectiveWithDistortionCamera<float>, camera)
            CALL_FWD_PRIVATEUSE1_WITH_OP(5, PerspectiveWithDistortionCamera<float>, camera)
            CALL_FWD_PRIVATEUSE1_WITH_OP(8, PerspectiveWithDistortionCamera<float>, camera)
            CALL_FWD_PRIVATEUSE1_WITH_OP(9, PerspectiveWithDistortionCamera<float>, camera)
            CALL_FWD_PRIVATEUSE1_WITH_OP(16, PerspectiveWithDistortionCamera<float>, camera)
            CALL_FWD_PRIVATEUSE1_WITH_OP(17, PerspectiveWithDistortionCamera<float>, camera)
            CALL_FWD_PRIVATEUSE1_WITH_OP(32, PerspectiveWithDistortionCamera<float>, camera)
            CALL_FWD_PRIVATEUSE1_WITH_OP(33, PerspectiveWithDistortionCamera<float>, camera)
            CALL_FWD_PRIVATEUSE1_WITH_OP(64, PerspectiveWithDistortionCamera<float>, camera)
            CALL_FWD_PRIVATEUSE1_WITH_OP(65, PerspectiveWithDistortionCamera<float>, camera)
            CALL_FWD_PRIVATEUSE1_WITH_OP(128, PerspectiveWithDistortionCamera<float>, camera)
            CALL_FWD_PRIVATEUSE1_WITH_OP(129, PerspectiveWithDistortionCamera<float>, camera)
            CALL_FWD_PRIVATEUSE1_WITH_OP(192, PerspectiveWithDistortionCamera<float>, camera)
            CALL_FWD_PRIVATEUSE1_WITH_OP(193, PerspectiveWithDistortionCamera<float>, camera)
            CALL_FWD_PRIVATEUSE1_WITH_OP(256, PerspectiveWithDistortionCamera<float>, camera)
            CALL_FWD_PRIVATEUSE1_WITH_OP(257, PerspectiveWithDistortionCamera<float>, camera)
            CALL_FWD_PRIVATEUSE1_WITH_OP(512, PerspectiveWithDistortionCamera<float>, camera)
            CALL_FWD_PRIVATEUSE1_WITH_OP(513, PerspectiveWithDistortionCamera<float>, camera)
        default:
            TORCH_CHECK_VALUE(
                false, "Unsupported channels for rasterize-from-world-3dgs: ", channels);
        }
    }

#undef CALL_FWD_PRIVATEUSE1_WITH_OP
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
                                                           const uint32_t,
                                                           const uint32_t,
                                                           const uint32_t,
                                                           const uint32_t,
                                                           const uint32_t,
                                                           const torch::Tensor &,
                                                           const torch::Tensor &,
                                                           const at::optional<torch::Tensor> &,
                                                           const at::optional<torch::Tensor> &) {
    TORCH_CHECK_VALUE(false, "dispatchGaussianRasterizeFromWorld3DGSForward is GPU-only");
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
rasterizeWorldSpaceGaussiansFwd(const torch::Tensor &means,
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
                                const at::optional<torch::Tensor> &backgrounds,
                                const at::optional<torch::Tensor> &masks) {
    return FVDB_DISPATCH_KERNEL(means.device(), [&]() {
        return dispatchGaussianRasterizeFromWorld3DGSForward<DeviceTag>(means,
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
                                                                        backgrounds,
                                                                        masks);
    });
}

} // namespace fvdb::detail::ops
