// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/ops/gsplat/Gaussian2D.cuh>
#include <fvdb/detail/ops/gsplat/GaussianRasterize.cuh>
#include <fvdb/detail/ops/gsplat/GaussianRasterizeTopContributingGaussianIds.h>
#include <fvdb/detail/ops/gsplat/GaussianRenderSettings.h>
#include <fvdb/detail/ops/gsplat/GaussianVectorTypes.cuh>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/Nvtx.h>

#include <c10/cuda/CUDAGuard.h>

#include <optional>

namespace fvdb::detail::ops {
namespace {

/// @brief Sorts three arrays in-place based on depth indices using bubble sort
///
/// This device function performs a bubble sort on three parallel arrays, using the depth indices
/// as the sorting key. The function maintains the correspondence between the three arrays during
/// sorting. This is optimized for small arrays (typically 4-24 elements) where the simplicity and
/// cache locality of bubble sort can be beneficial.
///
/// @param depthIndices Array of depth indices to sort by
/// @param radianceWeights Array of radiance weights corresponding to each depth index
/// @param maxRadianceWeightIndices Array of indices for maximum radiance weights
/// @param numSamples Number of elements in each array to sort
///
/// @note The function modifies all three arrays in-place. The sorting is stable and maintains the
/// relationship between corresponding elements in all three arrays.
///
/// @example
/// uint32_t depths[] = {3, 1, 2};
/// float weights[] = {0.3, 0.1, 0.2};
/// int32_t indices[] = {0, 1, 2};
/// bubbleSortByDepth<float>(depths, weights, indices, 3);
/// // Result:
/// // depths:    {1, 2, 3}
/// // weights:   {0.1, 0.2, 0.3}
/// // indices:   {1, 2, 0}
template <typename ScalarType>
__device__ void
bubbleSortByDepth(uint32_t *depthIndices,
                  ScalarType *radianceWeights,
                  int32_t *maxRadianceWeightIndices,
                  const uint32_t numSamples) {
    for (uint32_t i = 0; i < numSamples - 1; ++i) {
        for (uint32_t j = 0; j < numSamples - 1 - i; ++j) {
            if (depthIndices[j] > depthIndices[j + 1]) {
                // Swap all three arrays to maintain correspondence
                uint32_t tempDepth  = depthIndices[j];
                depthIndices[j]     = depthIndices[j + 1];
                depthIndices[j + 1] = tempDepth;

                ScalarType tempWeight  = radianceWeights[j];
                radianceWeights[j]     = radianceWeights[j + 1];
                radianceWeights[j + 1] = tempWeight;

                int32_t tempId                  = maxRadianceWeightIndices[j];
                maxRadianceWeightIndices[j]     = maxRadianceWeightIndices[j + 1];
                maxRadianceWeightIndices[j + 1] = tempId;
            }
        }
    }
}

// Structure to hold arguments and methods for the rasterize top contributing gaussian ids kernel
template <typename ScalarType, bool IS_PACKED> struct RasterizeTopContributingGaussianIdsArgs {
    using CommonArgs = RasterizeCommonArgs<ScalarType, 1, IS_PACKED>;
    CommonArgs commonArgs;

    uint32_t mNumDepthSamples;

    // In Dense mode, first dimension X = C * imageHeight * imageWidth
    // In Sparse mode, first dimension X = C * nPixels_i (i from 0 to C-1)
    JaggedRAcc64<int32_t, 2> mOutIds;                   // [X, numDepthSamples]
    JaggedRAcc64<ScalarType, 2> mOutWeights;            // [X, numDepthSamples]

    RasterizeTopContributingGaussianIdsArgs(
        const torch::Tensor &means2d,                   // [C, N, 2] or [nnz, 2]
        const torch::Tensor &conics,                    // [C, N, 3] or [nnz, 3]
        const torch::Tensor &opacities,                 // [C, N] or [nnz]
        const at::optional<torch::Tensor> &backgrounds, // [C, 1]
        const at::optional<torch::Tensor> &masks,       // [C, numTilesH, numTilesW]
        const uint32_t imageWidth,
        const uint32_t imageHeight,
        const uint32_t imageOriginW,
        const uint32_t imageOriginH,
        const uint32_t tileSize,
        const uint32_t numDepthSamples,
        const torch::Tensor &tileOffsets,     // [C, numTilesH, numTilesW]
        const torch::Tensor &tileGaussianIds, // [totalIntersections]
        const fvdb::JaggedTensor &outIds,     // [C, imgH, imgW, numDepthSamples]
        const fvdb::JaggedTensor &outWeights, // [C, imgH, imgW, numDepthSamples]
        const std::optional<torch::Tensor> &activeTiles = std::nullopt, // [AT]
        const std::optional<torch::Tensor> &tilePixelMask =
            std::nullopt, // [AT, wordsPerTileBitmask] e.g. [AT, 4]
        const std::optional<torch::Tensor> &tilePixelCumsum = std::nullopt, // [AT]
        const std::optional<torch::Tensor> &pixelMap        = std::nullopt)        // [AP]

        : commonArgs(means2d,
                     conics,
                     opacities,
                     std::nullopt,
                     backgrounds,
                     masks,
                     imageWidth,
                     imageHeight,
                     imageOriginW,
                     imageOriginH,
                     tileSize,
                     0,
                     tileOffsets,
                     tileGaussianIds,
                     activeTiles,
                     tilePixelMask,
                     tilePixelCumsum,
                     pixelMap),
          mNumDepthSamples(numDepthSamples),
          mOutIds(initJaggedAccessor<int32_t, 2>(outIds, "outIds")),
          mOutWeights(initJaggedAccessor<ScalarType, 2>(outWeights, "outWeights")) {}

    /// @brief Write a weight sample for a pixel
    /// @param pixelIndex The index of the pixel
    /// @param depthIndex The depth index of the sample
    /// @param weight The weight to write
    __device__ void
    writeWeight(uint64_t pixelIndex, uint32_t depthIndex, ScalarType weight) {
        mOutWeights.data()[pixelIndex][depthIndex] = weight;
    }

    /// @brief Write an id sample for a pixel
    /// @param pixelIndex The index of the pixel
    /// @param depthIndex The depth index of the sample
    /// @param id The id to write
    __device__ void
    writeId(uint64_t pixelIndex, uint32_t depthIndex, int32_t id) {
        mOutIds.data()[pixelIndex][depthIndex] = id;
    }

    __device__ void
    volumeRenderTileForward(const uint32_t cameraId,
                            const uint32_t row,
                            const uint32_t col,
                            const uint32_t firstGaussianIdInBlock,
                            const uint32_t lastGaussianIdInBlock,
                            const uint32_t blockSize,
                            const bool pixelIsActive,
                            const uint32_t activePixelIndex) {
        alignas(Gaussian2D<ScalarType>) extern __shared__ char s[];
        auto *sharedGaussians = reinterpret_cast<Gaussian2D<ScalarType> *>(s); // [blockSize]

        // Shared memory for the indices and max radiance weights for all pixels in this block
        int32_t *batchMaxRadianceWeightIndices = reinterpret_cast<int32_t *>(
            sharedGaussians + blockSize); // [blockSize * mNumDepthSamples]

        ScalarType *batchRadianceWeights = reinterpret_cast<ScalarType *>(
            batchMaxRadianceWeightIndices +
            blockSize * mNumDepthSamples); // [blockSize *mNumDepthSamples]

        // Shared memory for tracking depth order indices
        uint32_t *batchDepthIndices = reinterpret_cast<uint32_t *>(
            batchRadianceWeights + blockSize * mNumDepthSamples); // [blockSize * mNumDepthSamples]

        const auto tidx = threadIdx.y * blockDim.x + threadIdx.x;

        // Shared memory for the indices and max radiance weights for this pixel in the block
        int32_t *maxRadianceWeightIndices = batchMaxRadianceWeightIndices + tidx * mNumDepthSamples;
        ScalarType *radianceWeights       = batchRadianceWeights + tidx * mNumDepthSamples;
        uint32_t *depthIndices            = batchDepthIndices + tidx * mNumDepthSamples;

        // Initialize the indices and max radiance weights for this pixel in the block
        for (uint32_t w = 0; w < mNumDepthSamples; ++w) {
            radianceWeights[w]          = 0.f;
            maxRadianceWeightIndices[w] = -1;
            depthIndices[w]             = UINT32_MAX; // Large value for uninitialized
        }

        ScalarType accumTransmittance = 1.0f;
        uint32_t depthCounter         = 0; // Counter to track processing order (depth order)

        // We don't return right away if the pixel is not in the image since we want to use
        // this thread to load gaussians into shared memory
        bool done = !pixelIsActive;

        const uint32_t numBatches =
            (lastGaussianIdInBlock - firstGaussianIdInBlock + blockSize - 1) / blockSize;

        // TODO: This condition seems to be met when batch size is >1.  Kept here for debugging.
        // if (numBatches <= 0) {
        //     printf("No batches for this pixel %d %d\n", i, j);
        //     printf("lastGaussianIdInBlock: %d, firstGaussianIdInBlock: %d, blockSize: %d\n",
        //     lastGaussianIdInBlock, firstGaussianIdInBlock, blockSize);
        // }

        // (row, col) coordinates are relative to the specified image origin which may
        // be a crop so we need to add the origin to get the absolute pixel coordinates
        const ScalarType px = col + commonArgs.mImageOriginW + ScalarType{0.5f};
        const ScalarType py = row + commonArgs.mImageOriginH + ScalarType{0.5f};

        // collect and process batches of gaussians
        // each thread loads one gaussian at a time before rasterizing its
        // designated pixel
        for (uint32_t b = 0; b < numBatches; ++b) {
            // Sync threads before we start integrating the next batch
            // If all threads are done, we can break early
            if (__syncthreads_count(done) == blockSize) {
                break;
            }

            // Each thread fetches one gaussian from front to back (tile_gaussian_ids is depth
            // sorted)
            const uint32_t batchStart = firstGaussianIdInBlock + blockSize * b;
            const uint32_t idx        = batchStart + tidx;
            if (idx < lastGaussianIdInBlock) {
                const int32_t g =
                    commonArgs.mTileGaussianIds[idx]; // which gaussian we're rendering
                sharedGaussians[tidx] = commonArgs.getGaussian(g);
            }

            // Sync threads so all gaussians for this batch are loaded in shared memory
            __syncthreads();

            // Volume render Gaussians in this batch
            if (pixelIsActive) { // skip inactive sparse pixels
                const uint32_t batchSize = min(blockSize, lastGaussianIdInBlock - batchStart);
                for (uint32_t t = 0; (t < batchSize) && !done; ++t) {
                    const Gaussian2D<ScalarType> gaussian = sharedGaussians[t];

                    const auto [gaussianIsValid, delta, expMinusSigma, alpha] =
                        commonArgs.evalGaussian(gaussian, px, py);

                    if (!gaussianIsValid) {
                        continue;
                    }

                    const ScalarType nextTransmittance = accumTransmittance * (1.0f - alpha);
                    if (nextTransmittance <= 1e-4f) { // this pixel is done: exclusive
                        done = true;
                        break;
                    }

                    const ScalarType radianceWeight = alpha * accumTransmittance;

                    // Find the index of the smallest radiance weight in our top K samples
                    uint32_t min_idx = 0;
                    for (uint32_t k = 1; k < mNumDepthSamples; ++k) {
                        if (radianceWeights[k] < radianceWeights[min_idx]) {
                            min_idx = k;
                        }
                    }

                    // If this gaussian is more significant than our weakest sample, replace it
                    if (radianceWeight > radianceWeights[min_idx]) {
                        radianceWeights[min_idx]          = radianceWeight;
                        maxRadianceWeightIndices[min_idx] = gaussian.id;
                        depthIndices[min_idx]             = depthCounter;
                    }

                    depthCounter++; // Increment depth order counter

                    accumTransmittance = nextTransmittance;
                }
            }
        }

        if (pixelIsActive) {
            // Sort the samples by depth order before outputting
            bubbleSortByDepth(
                depthIndices, radianceWeights, maxRadianceWeightIndices, mNumDepthSamples);

            const auto pixIdx = commonArgs.pixelIndex(cameraId, row, col, activePixelIndex);

            for (uint32_t k = 0; k < mNumDepthSamples; ++k) {
                writeWeight(pixIdx, k, radianceWeights[k]);
                writeId(pixIdx, k, maxRadianceWeightIndices[k]);
            }
        }
    }
};

/****************************************************************************
 * Rasterization to Pixels Forward Pass
 ****************************************************************************/

template <typename ScalarType, bool IS_PACKED>
__global__ void
rasterizeTopContributingGaussianIdsForward(
    RasterizeTopContributingGaussianIdsArgs<ScalarType, IS_PACKED> args) {
    auto &commonArgs = args.commonArgs;

    // each thread draws one pixel, but also timeshares caching gaussians in a
    // shared tile
    int32_t cameraId;
    int32_t tileRow;
    int32_t tileCol;
    uint32_t row, col;

    cuda::std::tie(cameraId, tileRow, tileCol, row, col) =
        commonArgs.mIsSparse ? commonArgs.sparseCoordinates() : commonArgs.denseCoordinates();

    // NOTE: We keep threads which correspond to pixels outside the image bounds around
    //       to load gaussians from global memory, but they do not contribute to the output.

    // pixelInImage: Whether this pixel is inside the image bounds.
    // activePixelIndex: Index of this pixel in the output for the block if it is active (sparse
    // mode only).
    bool pixelInImage{false};
    uint32_t activePixelIndex{0};
    cuda::std::tie(pixelInImage, activePixelIndex) = commonArgs.activePixelIndex(row, col);

    if (commonArgs.mHasMasks && pixelInImage && !commonArgs.mMasks[cameraId][tileRow][tileCol]) {
        auto pixIdx = commonArgs.pixelIndex(cameraId, row, col, activePixelIndex);

        for (int32_t d = 0; d < args.mNumDepthSamples; ++d) {
            args.writeId(
                pixIdx, d, commonArgs.mHasBackgrounds ? commonArgs.mBackgrounds[cameraId][d] : 0);
        }
        return;
    }

    int32_t firstGaussianIdInBlock;
    int32_t lastGaussianIdInBlock;
    cuda::std::tie(firstGaussianIdInBlock, lastGaussianIdInBlock) =
        commonArgs.tileGaussianRange(cameraId, tileRow, tileCol);

    args.volumeRenderTileForward(cameraId,
                                 row,
                                 col,
                                 firstGaussianIdInBlock,
                                 lastGaussianIdInBlock,
                                 blockDim.x * blockDim.y,
                                 pixelInImage,
                                 activePixelIndex);
}

template <typename ScalarType, bool IS_PACKED>
std::tuple<fvdb::JaggedTensor, fvdb::JaggedTensor>
launchRasterizeTopContributingGaussianIdsForwardKernel(
    // Gaussian parameters
    const torch::Tensor &means2d,                   // [C, N, 2] or [nnz, 2]
    const torch::Tensor &conics,                    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &opacities,                 // [C, N]  or [nnz]
    const at::optional<torch::Tensor> &backgrounds, // [C, channels]
    const at::optional<torch::Tensor> &masks,       // [C, tile_height, tile_width]
    // intersections
    const torch::Tensor &tileOffsets,     // [C, tile_height, tile_width]
    const torch::Tensor &tileGaussianIds, // [n_isects]
    const RenderSettings &settings,       // render settings
    const std::optional<fvdb::JaggedTensor> &pixelsToRender = std::nullopt, // [C, NumPixels, 2]
    const std::optional<torch::Tensor> &activeTiles         = std::nullopt,
    const std::optional<torch::Tensor> &tilePixelMask       = std::nullopt,
    const std::optional<torch::Tensor> &tilePixelCumsum     = std::nullopt,
    const std::optional<torch::Tensor> &pixelMap            = std::nullopt) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(means2d));

    TORCH_CHECK_VALUE(settings.numDepthSamples > 0, "numDepthSamples must be greater than 0");

    // tileOffsets can be 3D (dense) or 1D (sparse)
    const bool tileOffsetsAreSparse = tileOffsets.dim() == 1;
    if (!tileOffsetsAreSparse) {
        TORCH_CHECK_VALUE(tileOffsets.size(2) ==
                              (settings.imageWidth + settings.tileSize - 1) / settings.tileSize,
                          "tileOffsets width must match the number of tiles in image size");
        TORCH_CHECK_VALUE(tileOffsets.size(1) ==
                              (settings.imageHeight + settings.tileSize - 1) / settings.tileSize,
                          "tileOffsets height must match the number of tiles in image size");
    }

    const uint32_t C           = means2d.size(0); // number of cameras
    const uint32_t tileExtentH = (settings.imageHeight + settings.tileSize - 1) / settings.tileSize;
    const uint32_t tileExtentW = (settings.imageWidth + settings.tileSize - 1) / settings.tileSize;

    TORCH_CHECK_VALUE(pixelMap.has_value() == pixelsToRender.has_value(),
                      "pixelMap and pixelsToRender must be provided together");
    if (pixelMap.has_value()) {
        TORCH_CHECK_VALUE(pixelMap.value().size(0) == pixelsToRender.value().numel() / 2,
                          "pixelMap must have the same number of elements as pixelsToRender");
    }

    auto sizes = pixelsToRender.has_value()
                     ? pixelsToRender->lsizes1()
                     : std::vector<int64_t>{C * settings.imageHeight * settings.imageWidth};
    std::vector<torch::Tensor> idsToRenderVec;
    std::vector<torch::Tensor> weightsToRenderVec;

    for (const auto &size: sizes) {
        idsToRenderVec.push_back(
            torch::empty({size, settings.numDepthSamples}, means2d.options().dtype(torch::kInt32)));
        weightsToRenderVec.push_back(
            torch::empty({size, settings.numDepthSamples},
                         means2d.options().dtype(c10::CppTypeToScalarType<ScalarType>::value)));
    }

    auto outIds     = fvdb::JaggedTensor(idsToRenderVec);
    auto outWeights = fvdb::JaggedTensor(weightsToRenderVec);

    const at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    // Each pixel in each tile will cache a gaussian consisting of:
    //   - int32_t  gaussian_id; -- 4 bytes
    //   - vec2t    xy;          -- 8 bytes for float32
    //   - scalar_t opacity;     -- 4 bytes for float32
    //   - vec3t    conic;       -- 12 bytes for float32
    const uint32_t sharedMem =
        settings.tileSize * settings.tileSize *
        (sizeof(Gaussian2D<ScalarType>) +
         (sizeof(int32_t) + sizeof(ScalarType) + sizeof(uint32_t)) * settings.numDepthSamples);

    if (cudaFuncSetAttribute(rasterizeTopContributingGaussianIdsForward<ScalarType, IS_PACKED>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             sharedMem) != cudaSuccess) {
        AT_ERROR("Failed to set maximum shared memory size (requested ",
                 sharedMem,
                 " bytes), try lowering tile_size.");
    }

    const dim3 blockDim = {settings.tileSize, settings.tileSize, 1};
    const dim3 gridDim  = activeTiles.has_value() // sparse mode
                              ? dim3(activeTiles.value().size(0), 1, 1)
                              : dim3(C * tileExtentH * tileExtentW, 1, 1);
    auto args =
        RasterizeTopContributingGaussianIdsArgs<ScalarType, IS_PACKED>(means2d,
                                                                       conics,
                                                                       opacities,
                                                                       backgrounds,
                                                                       masks,
                                                                       settings.imageWidth,
                                                                       settings.imageHeight,
                                                                       settings.imageOriginW,
                                                                       settings.imageOriginH,
                                                                       settings.tileSize,
                                                                       settings.numDepthSamples,
                                                                       tileOffsets,
                                                                       tileGaussianIds,
                                                                       outIds,
                                                                       outWeights,
                                                                       activeTiles,
                                                                       tilePixelMask,
                                                                       tilePixelCumsum,
                                                                       pixelMap);

    rasterizeTopContributingGaussianIdsForward<<<gridDim, blockDim, sharedMem, stream>>>(args);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));

    return std::make_tuple(outIds, outWeights);
}

} // namespace

template <>
std::tuple<torch::Tensor, torch::Tensor>
dispatchGaussianRasterizeTopContributingGaussianIds<torch::kCUDA>(
    // Gaussian parameters
    const torch::Tensor &means2d,         // [C, N, 2]
    const torch::Tensor &conics,          // [C, N, 3]
    const torch::Tensor &opacities,       // [N]
    const torch::Tensor &tileOffsets,     // [C, tile_height, tile_width]
    const torch::Tensor &tileGaussianIds, // [n_isects]
    const RenderSettings &settings        // render settings

) {
    FVDB_FUNC_RANGE();
    const bool isPacked = means2d.dim() == 2;

    const std::optional<torch::Tensor> backgrounds = std::nullopt;
    const std::optional<torch::Tensor> masks       = std::nullopt;

    return AT_DISPATCH_V2(
        opacities.scalar_type(),
        "GaussianRasterizeTopContributingGaussianIds",
        AT_WRAP([&]() {
            auto [ids, weights] =
                isPacked ? launchRasterizeTopContributingGaussianIdsForwardKernel<float, true>(
                               means2d,
                               conics,
                               opacities,
                               backgrounds,
                               masks,
                               tileOffsets,
                               tileGaussianIds,
                               settings)
                         : launchRasterizeTopContributingGaussianIdsForwardKernel<float, false>(
                               means2d,
                               conics,
                               opacities,
                               backgrounds,
                               masks,
                               tileOffsets,
                               tileGaussianIds,
                               settings);
            const auto C = means2d.size(0);
            return std::make_tuple(
                ids.jdata().reshape(
                    {C, settings.imageHeight, settings.imageWidth, settings.numDepthSamples}),
                weights.jdata().reshape(
                    {C, settings.imageHeight, settings.imageWidth, settings.numDepthSamples}));
        }),
        AT_EXPAND(AT_FLOATING_TYPES),
        c10::kHalf);
}

template <>
std::tuple<torch::Tensor, torch::Tensor>
dispatchGaussianRasterizeTopContributingGaussianIds<torch::kCPU>(
    // Gaussian parameters
    const torch::Tensor &means2d,         // [C, N, 2]
    const torch::Tensor &conics,          // [C, N, 3]
    const torch::Tensor &opacities,       // [N]
    const torch::Tensor &tileOffsets,     // [C, tile_height, tile_width]
    const torch::Tensor &tileGaussianIds, // [n_isects]
    const RenderSettings &settings        // render settings
) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "CPU implementation not available");
}

template <>
std::tuple<fvdb::JaggedTensor, fvdb::JaggedTensor>
dispatchGaussianSparseRasterizeTopContributingGaussianIds<torch::kCUDA>(
    const torch::Tensor &means2d,         // [C, N, 2]
    const torch::Tensor &conics,          // [C, N, 3]
    const torch::Tensor &opacities,       // [N]
    const torch::Tensor &tileOffsets,     // [C, tile_height, tile_width]
    const torch::Tensor &tileGaussianIds, // [n_isects]
    const fvdb::JaggedTensor &pixelsToRender,
    const torch::Tensor &activeTiles,
    const torch::Tensor &tilePixelMask,
    const torch::Tensor &tilePixelCumsum,
    const torch::Tensor &pixelMap,
    const RenderSettings &settings // render settings
) {
    FVDB_FUNC_RANGE();
    const bool isPacked = means2d.dim() == 2;

    const std::optional<torch::Tensor> backgrounds = std::nullopt;
    const std::optional<torch::Tensor> masks       = std::nullopt;

    return AT_DISPATCH_V2(
        opacities.scalar_type(),
        "GaussianRasterizeTopContributingGaussianIds",
        AT_WRAP([&]() {
            if (isPacked) {
                return launchRasterizeTopContributingGaussianIdsForwardKernel<float, true>(
                    means2d,
                    conics,
                    opacities,
                    backgrounds,
                    masks,
                    tileOffsets,
                    tileGaussianIds,
                    settings,
                    pixelsToRender,
                    activeTiles,
                    tilePixelMask,
                    tilePixelCumsum,
                    pixelMap);
            } else {
                return launchRasterizeTopContributingGaussianIdsForwardKernel<float, false>(
                    means2d,
                    conics,
                    opacities,
                    backgrounds,
                    masks,
                    tileOffsets,
                    tileGaussianIds,
                    settings,
                    pixelsToRender,
                    activeTiles,
                    tilePixelMask,
                    tilePixelCumsum,
                    pixelMap);
            }
        }),
        AT_EXPAND(AT_FLOATING_TYPES),
        c10::kHalf);
}

template <>
std::tuple<fvdb::JaggedTensor, fvdb::JaggedTensor>
dispatchGaussianSparseRasterizeTopContributingGaussianIds<torch::kCPU>(
    const torch::Tensor &means2d,         // [C, N, 2]
    const torch::Tensor &conics,          // [C, N, 3]
    const torch::Tensor &opacities,       // [N]
    const torch::Tensor &tileOffsets,     // [C, tile_height, tile_width]
    const torch::Tensor &tileGaussianIds, // [n_isects]
    const fvdb::JaggedTensor &pixelsToRender,
    const torch::Tensor &activeTiles,
    const torch::Tensor &tilePixelMask,
    const torch::Tensor &tilePixelCumsum,
    const torch::Tensor &pixelMap,
    const RenderSettings &settings // render settings
) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "CPU implementation not available");
}

} // namespace fvdb::detail::ops
