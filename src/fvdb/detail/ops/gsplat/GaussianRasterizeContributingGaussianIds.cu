// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/ops/gsplat/Gaussian2D.cuh>
#include <fvdb/detail/ops/gsplat/GaussianRasterize.cuh>
#include <fvdb/detail/ops/gsplat/GaussianRasterizeContributingGaussianIds.h>
#include <fvdb/detail/ops/gsplat/GaussianRasterizeTopContributingGaussianIds.h>
#include <fvdb/detail/ops/gsplat/GaussianRenderSettings.h>
#include <fvdb/detail/ops/gsplat/GaussianVectorTypes.cuh>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/Nvtx.h>

#include <c10/cuda/CUDAGuard.h>

#include <optional>

namespace fvdb::detail::ops {
namespace {

/// @brief Copy data from the padded 'Top Contributing Gaussian IDs' JaggedTensor to an unpadded
/// 'Contributing Gaussian IDs' JaggedTensor
///
/// This function copies data from srcJagged (ldim=1 with fixed-size padding) to dstJagged (ldim=2
/// with variable sizes). Each pixel in srcJagged has maxSamplesPerPixel elements, with unused
/// slots padded with -1. We copy only the valid samples (count given by numValidSamples).
///
/// @param srcJagged Source JaggedTensor (ldim=1) with fixed-size padding per pixel
/// @param dstJagged Destination JaggedTensor (ldim=2) with variable sizes per pixel
/// @param numValidSamples JaggedTensor containing count of valid samples per pixel
/// @param maxSamplesPerPixel Maximum samples per pixel in source (including padding)
void
copyPaddedJaggedToJagged(const fvdb::JaggedTensor &srcJagged,
                         fvdb::JaggedTensor &dstJagged,
                         const fvdb::JaggedTensor &numValidSamples,
                         const int32_t maxSamplesPerPixel) {
    const int64_t numPixels = srcJagged.jdata().size(0);
    TORCH_CHECK_VALUE(numPixels == (dstJagged.joffsets().size(0) - 1),
                      "Source and destination must have the same number of pixels");
    TORCH_CHECK_VALUE(numPixels == numValidSamples.numel(),
                      "numValidSamples must have one entry per pixel");

    if (numPixels == 0) {
        return;
    }

    // srcJagged.jdata() has shape [numPixels, maxSamplesPerPixel]
    // We want to extract only the first numValidSamples[i] elements from each row i
    // dstJagged.jdata() has shape [sum(numValidSamples)]

    // Create a mask indicating which elements are valid (not padding)
    // mask[i, j] = true if j < numValidSamples[i]
    torch::Tensor sampleIndices =
        torch::arange(maxSamplesPerPixel,
                      torch::TensorOptions().dtype(torch::kInt32).device(numValidSamples.device()));
    torch::Tensor mask = sampleIndices.unsqueeze(0) < numValidSamples.jdata().unsqueeze(1);

    // Use the mask to extract valid samples and copy to destination
    dstJagged.jdata().copy_(srcJagged.jdata().index({mask}));
}

// Structure to hold arguments and methods for the rasterize top contributing gaussian ids kernel
template <typename ScalarType, bool IS_PACKED> struct RasterizeContributingGaussianIdsArgs {
    using CommonArgs = RasterizeCommonArgs<ScalarType, 1, IS_PACKED>;
    CommonArgs commonArgs;

    uint32_t mNumDepthSamples;

    // In Dense mode, first dimension X = C * imageHeight * imageWidth
    // In Sparse mode, first dimension X = C * nPixels_i (i from 0 to C-1)
    JaggedRAcc64<int32_t, 2> mOutIds;                   // [X, numDepthSamples]
    JaggedRAcc64<ScalarType, 2> mOutWeights;            // [X, numDepthSamples]

    RasterizeContributingGaussianIdsArgs(
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

        const auto tidx = threadIdx.y * blockDim.x + threadIdx.x;

        // Thread-local buffers for accumulating writes
        // Most pixels have < 64 contributing gaussians in typical scenes
        constexpr uint32_t MAX_LOCAL_BUFFER_SIZE = 64;
        int32_t localIds[MAX_LOCAL_BUFFER_SIZE];
        ScalarType localWeights[MAX_LOCAL_BUFFER_SIZE];

        ScalarType accumTransmittance = 1.0f;
        uint32_t writeIndex           = 0;

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

        const auto pixIdx =
            pixelIsActive ? commonArgs.pixelIndex(cameraId, row, col, activePixelIndex) : 0;

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

                    // Write to thread-local buffer instead of global memory
                    if (writeIndex < MAX_LOCAL_BUFFER_SIZE) {
                        localIds[writeIndex]     = gaussian.id;
                        localWeights[writeIndex] = radianceWeight;
                        writeIndex++;
                    } else {
                        // Overflow: buffer is full, fall back to direct write (rare case)
                        writeId(pixIdx, writeIndex, gaussian.id);
                        writeWeight(pixIdx, writeIndex, radianceWeight);
                        writeIndex++;
                    }

                    accumTransmittance = nextTransmittance;
                }
            }
        }

        // Flush thread-local buffers to global memory
        if (pixelIsActive && writeIndex > 0) {
            // Write buffered results from thread-local storage to global memory
            const uint32_t numToWrite = min(writeIndex, MAX_LOCAL_BUFFER_SIZE);
            for (uint32_t i = 0; i < numToWrite; ++i) {
                writeId(pixIdx, i, localIds[i]);
                writeWeight(pixIdx, i, localWeights[i]);
            }
        }
    }
};

/****************************************************************************
 * Rasterization to Pixels Forward Pass
 ****************************************************************************/

template <typename ScalarType, bool IS_PACKED>
__global__ void
rasterizeContributingGaussianIdsForward(
    RasterizeContributingGaussianIdsArgs<ScalarType, IS_PACKED> args) {
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

        for (uint32_t d = 0; d < args.mNumDepthSamples; ++d) {
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
launchRasterizeContributingGaussianIdsForwardKernel(
    // Gaussian parameters
    const torch::Tensor &means2d,                   // [C, N, 2] or [nnz, 2]
    const torch::Tensor &conics,                    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &opacities,                 // [C, N]  or [nnz]
    const at::optional<torch::Tensor> &backgrounds, // [C, channels]
    const at::optional<torch::Tensor> &masks,       // [C, tile_height, tile_width]
    // intersections
    const torch::Tensor &tileOffsets,     // [C, tile_height, tile_width]
    const torch::Tensor &tileGaussianIds, // [n_isects]
    // render settings
    const std::optional<fvdb::JaggedTensor> &maybeNumContributingGaussians, // [C, NumPixels, 1]
    const RenderSettings &settings,                                         // render settings
    // sparse rendering parameters
    const std::optional<fvdb::JaggedTensor> &pixelsToRender = std::nullopt, // [C, NumPixels, 2]
    const std::optional<torch::Tensor> &activeTiles         = std::nullopt,
    const std::optional<torch::Tensor> &tilePixelMask       = std::nullopt,
    const std::optional<torch::Tensor> &tilePixelCumsum     = std::nullopt,
    const std::optional<torch::Tensor> &pixelMap            = std::nullopt) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(means2d));

    // Check if we're in top-k mode (numDepthSamples > 0 indicates top-k)
    // If so, call the top-k kernel and reformat the results
    if (settings.numDepthSamples > 0) {
        // Call the top-k dispatch function
        fvdb::JaggedTensor outIds, outWeights;

        if (pixelsToRender.has_value()) {
            // Sparse mode: call sparse top-k dispatch
            std::tie(outIds, outWeights) =
                dispatchGaussianSparseRasterizeTopContributingGaussianIds<torch::kCUDA>(
                    means2d,
                    conics,
                    opacities,
                    tileOffsets,
                    tileGaussianIds,
                    pixelsToRender.value(),
                    activeTiles.value(),
                    tilePixelMask.value(),
                    tilePixelCumsum.value(),
                    pixelMap.value(),
                    settings);
        } else {
            // Dense mode: call dense top-k dispatch
            torch::Tensor denseIds, denseWeights;
            std::tie(denseIds, denseWeights) =
                dispatchGaussianRasterizeTopContributingGaussianIds<torch::kCUDA>(
                    means2d, conics, opacities, tileOffsets, tileGaussianIds, settings);

            // Convert dense output [C, H, W, K] to JaggedTensor format
            const auto C = denseIds.size(0);
            const auto H = denseIds.size(1);
            const auto W = denseIds.size(2);
            const auto K = denseIds.size(3);

            // Reshape to [C*H*W, K] for easier processing
            denseIds     = denseIds.reshape({C * H * W, K});
            denseWeights = denseWeights.reshape({C * H * W, K});

            // Create JaggedTensor with fixed K samples per pixel
            std::vector<torch::Tensor> idsVec, weightsVec;
            for (int64_t c = 0; c < C; ++c) {
                idsVec.emplace_back(denseIds.slice(0, c * H * W, (c + 1) * H * W));
                weightsVec.emplace_back(denseWeights.slice(0, c * H * W, (c + 1) * H * W));
            }
            outIds     = fvdb::JaggedTensor(idsVec);
            outWeights = fvdb::JaggedTensor(weightsVec);
        }

        // Now convert from fixed-size top-k format to variable-size format
        // Create a numValidSamples tensor by counting non-(-1) entries in outIds
        torch::Tensor numValidSamples = (outIds.jdata() != -1).sum(-1).to(torch::kInt32);

        // Use numValidSamples to create the properly formatted output
        const auto outputNumPixels             = numValidSamples.numel();
        const auto numContributingGaussiansSum = numValidSamples.sum().item<int64_t>();

        // Build JaggedTensor structure similar to below
        // We need to convert numValidSamples into a proper JaggedTensor format
        const auto C = means2d.size(0);
        torch::Tensor offsets =
            torch::arange(
                0,
                C + 1,
                torch::TensorOptions().dtype(fvdb::JOffsetsScalarType).device(means2d.device())) *
            (outputNumPixels / C);
        torch::Tensor listIds = torch::empty(
            {0, 1}, torch::TensorOptions().dtype(fvdb::JLIdxScalarType).device(means2d.device()));

        JaggedTensor numValidSamplesJagged =
            JaggedTensor::from_data_offsets_and_list_ids(numValidSamples, offsets, listIds);

        const auto outputNumCameras = numValidSamplesJagged.num_outer_lists();

        // Populate the linear batch indices (jidx)
        torch::Tensor outIndices = torch::repeat_interleave(
            torch::arange(
                outputNumPixels,
                torch::TensorOptions().dtype(fvdb::JIdxScalarType).device(means2d.device())),
            numValidSamplesJagged.jdata());

        // Populate the LoL indices (ListIdx)
        torch::Tensor pixelCountsPerCamera = numValidSamplesJagged.joffsets().diff();

        torch::Tensor cameraIndices = torch::repeat_interleave(
            torch::arange(
                outputNumCameras,
                torch::TensorOptions().dtype(fvdb::JLIdxScalarType).device(means2d.device())),
            pixelCountsPerCamera);

        torch::Tensor cumsumOffsets = torch::cat({torch::zeros({1}, pixelCountsPerCamera.options()),
                                                  pixelCountsPerCamera.slice(0, 0, -1)})
                                          .cumsum(0);

        torch::Tensor repeatedOffsets =
            torch::repeat_interleave(cumsumOffsets, pixelCountsPerCamera);

        torch::Tensor pixelIndices =
            torch::arange(outputNumPixels, pixelCountsPerCamera.options()) - repeatedOffsets;

        torch::Tensor outListIds =
            torch::stack({cameraIndices, pixelIndices}, 1).to(fvdb::JLIdxScalarType);

        // Allocate output data storage
        torch::Tensor outIdsData =
            torch::empty({numContributingGaussiansSum}, means2d.options().dtype(torch::kInt32));
        torch::Tensor outWeightsData =
            torch::empty({numContributingGaussiansSum},
                         means2d.options().dtype(c10::CppTypeToScalarType<ScalarType>::value));

        auto outIdsJaggedSamples = fvdb::JaggedTensor::from_data_indices_and_list_ids(
            outIdsData, outIndices, outListIds, outputNumPixels);
        auto outWeightsJaggedSamples = fvdb::JaggedTensor::from_data_indices_and_list_ids(
            outWeightsData, outIndices, outListIds, outputNumPixels);

        // Copy valid samples from top-k output
        copyPaddedJaggedToJagged(
            outIds, outIdsJaggedSamples, numValidSamplesJagged, settings.numDepthSamples);
        copyPaddedJaggedToJagged(
            outWeights, outWeightsJaggedSamples, numValidSamplesJagged, settings.numDepthSamples);

        return std::make_tuple(outIdsJaggedSamples, outWeightsJaggedSamples);
    }

    TORCH_CHECK_VALUE(maybeNumContributingGaussians.has_value(),
                      "numContributingGaussians must be provided if not using top-k mode");

    const JaggedTensor &numContributingGaussians = maybeNumContributingGaussians.value();

    const auto C                           = means2d.size(0); // number of cameras
    const auto numContributingGaussiansSum = numContributingGaussians.jdata().sum().item<int64_t>();

    TORCH_CHECK_VALUE(numContributingGaussians.numel() > 0,
                      "numContributingGaussians cannot be empty");
    // Ensure that number of contributing gaussians won't overflow the number of tensors we can
    // express in a JaggedTensor
    TORCH_CHECK_VALUE(
        numContributingGaussiansSum <= std::numeric_limits<fvdb::JIdxType>::max(),
        "Number of contributing gaussians is too large to express in the number of tensors we can create in a JaggedTensor");

    // Ensure numContributingGaussians is appropriately sized
    TORCH_CHECK_VALUE(numContributingGaussians.ldim() == 1,
                      "numContributingGaussians must be a single list dimension JaggedTensor");
    if (pixelsToRender.has_value()) {
        TORCH_CHECK_VALUE(
            numContributingGaussians.lsizes1() == pixelsToRender.value().lsizes1(),
            "numContributingGaussians must have the same number of elements as pixelsToRender");
    } else {
        TORCH_CHECK_VALUE(
            torch::equal(numContributingGaussians.joffsets(),
                         torch::arange(0, C + 1, 1, numContributingGaussians.options()) *
                             settings.imageHeight * settings.imageWidth),
            "numContributingGaussians must have the same number of elements as the number of pixels in the images");
    }

    const auto tileExtentH = tileOffsets.size(1);
    const auto tileExtentW = tileOffsets.size(2);

    TORCH_CHECK_VALUE(tileExtentW ==
                          (settings.imageWidth + settings.tileSize - 1) / settings.tileSize,
                      "tileOffsets width must match the number of tiles in image size");
    TORCH_CHECK_VALUE(tileExtentH ==
                          (settings.imageHeight + settings.tileSize - 1) / settings.tileSize,
                      "tileOffsets height must match the number of tiles in image size");

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

    // maximum possible number of depth samples per pixel
    const auto maxDepthSamplesPerPixel = numContributingGaussians.jdata().max().item<int32_t>();

    for (const auto &size: sizes) {
        idsToRenderVec.emplace_back(
            torch::empty({size, maxDepthSamplesPerPixel}, means2d.options().dtype(torch::kInt32)));
        weightsToRenderVec.emplace_back(
            torch::empty({size, maxDepthSamplesPerPixel},
                         means2d.options().dtype(c10::CppTypeToScalarType<ScalarType>::value)));
    }

    auto outIds     = fvdb::JaggedTensor(idsToRenderVec);
    auto outWeights = fvdb::JaggedTensor(weightsToRenderVec);

    // Each pixel in each tile will cache a gaussian consisting of:
    //   - int32_t  gaussian_id; -- 4 bytes
    //   - vec2t    xy;          -- 8 bytes for float32
    //   - scalar_t opacity;     -- 4 bytes for float32
    //   - vec3t    conic;       -- 12 bytes for float32
    // Note: We use thread-local storage for buffering writes, so only need shared memory for
    // Gaussians. We add 32 bytes for alignment padding.
    const uint32_t sharedMem =
        settings.tileSize * settings.tileSize * sizeof(Gaussian2D<ScalarType>);

    if (cudaFuncSetAttribute(rasterizeContributingGaussianIdsForward<ScalarType, IS_PACKED>,
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
    auto args           = RasterizeContributingGaussianIdsArgs<ScalarType, IS_PACKED>(means2d,
                                                                            conics,
                                                                            opacities,
                                                                            backgrounds,
                                                                            masks,
                                                                            settings.imageWidth,
                                                                            settings.imageHeight,
                                                                            settings.imageOriginW,
                                                                            settings.imageOriginH,
                                                                            settings.tileSize,
                                                                            maxDepthSamplesPerPixel,
                                                                            tileOffsets,
                                                                            tileGaussianIds,
                                                                            outIds,
                                                                            outWeights,
                                                                            activeTiles,
                                                                            tilePixelMask,
                                                                            tilePixelCumsum,
                                                                            pixelMap);

    const at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    rasterizeContributingGaussianIdsForward<<<gridDim, blockDim, sharedMem, stream>>>(args);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // TODO: When refactoring GaussianRasterizeTopContributingGaussianIds to
    //  GaussianRasterizeContributingGaussianIds, I realized there would need to be nontrivial
    //  work to refactor the RasterizeCommonArgs (and additional work to the JaggedAccessor) to
    //  support this list-of-lists style of rendering where we write different numbers of
    //  samples per pixel and different numbers of pixels per camera. To save on the lower
    //  level work, we have instead still kept 'rasterizeContributingGaussianIdsForward' with
    //  similar logic rendering a dense, maximum number of maxDepthSamplesPerPixel per-pixel
    //  and we will now copy the results into an appropriately sparse JaggedTensor. This at least
    //  removes the guesswork from the user over what sample count to use and changes the
    //  public interface, but we leave this inefficient copy step for future work to refactor
    //  writing results directly to a JaggedTensor.

    //  Allocate JaggedTensor output storage
    const auto outputNumPixels  = numContributingGaussians.numel();
    const auto outputNumCameras = numContributingGaussians.num_outer_lists();

    // Populate the linear batch indices (jidx)
    torch::Tensor outIndices =
        torch::repeat_interleave(torch::arange(outputNumPixels,
                                               torch::TensorOptions()
                                                   .dtype(fvdb::JIdxScalarType)
                                                   .device(numContributingGaussians.device())),
                                 numContributingGaussians.jdata());

    // Populate the LoL indices (ListIdx)
    // Extract pixel counts per camera from joffsets
    torch::Tensor pixelCountsPerCamera = numContributingGaussians.joffsets().diff();

    // First column: camera indices (repeat each camera index by its pixel count)
    torch::Tensor cameraIndices =
        torch::repeat_interleave(torch::arange(outputNumCameras,
                                               torch::TensorOptions()
                                                   .dtype(fvdb::JLIdxScalarType)
                                                   .device(numContributingGaussians.device())),
                                 pixelCountsPerCamera);

    // Second column: pixel indices within each camera [0...N0-1, 0...N1-1, ...]
    // Create offsets for each camera and subtract from sequential indices
    torch::Tensor cumsumOffsets = torch::cat({torch::zeros({1}, pixelCountsPerCamera.options()),
                                              pixelCountsPerCamera.slice(0, 0, -1)})
                                      .cumsum(0);

    torch::Tensor repeatedOffsets = torch::repeat_interleave(cumsumOffsets, pixelCountsPerCamera);

    torch::Tensor pixelIndices =
        torch::arange(outputNumPixels, pixelCountsPerCamera.options()) - repeatedOffsets;

    // Stack into [totalPixels, 2]
    torch::Tensor outListIds =
        torch::stack({cameraIndices, pixelIndices}, 1).to(fvdb::JLIdxScalarType);

    // Allocate output data storage
    torch::Tensor outIdsData =
        torch::empty({numContributingGaussiansSum}, means2d.options().dtype(torch::kInt32));
    torch::Tensor outWeightsData =
        torch::empty({numContributingGaussiansSum},
                     means2d.options().dtype(c10::CppTypeToScalarType<ScalarType>::value));

    auto outIdsJaggedSamples = fvdb::JaggedTensor::from_data_indices_and_list_ids(
        outIdsData, outIndices, outListIds, outputNumPixels);
    auto outWeightsJaggedSamples = fvdb::JaggedTensor::from_data_indices_and_list_ids(
        outWeightsData, outIndices, outListIds, outputNumPixels);

    // Copy valid samples
    copyPaddedJaggedToJagged(
        outIds, outIdsJaggedSamples, numContributingGaussians, maxDepthSamplesPerPixel);
    copyPaddedJaggedToJagged(
        outWeights, outWeightsJaggedSamples, numContributingGaussians, maxDepthSamplesPerPixel);

    return std::make_tuple(outIdsJaggedSamples, outWeightsJaggedSamples);
}

} // namespace

template <>
__host__ std::tuple<fvdb::JaggedTensor, fvdb::JaggedTensor>
dispatchGaussianRasterizeContributingGaussianIds<torch::kCUDA>(
    // Gaussian parameters
    const torch::Tensor &means2d,         // [C, N, 2]
    const torch::Tensor &conics,          // [C, N, 3]
    const torch::Tensor &opacities,       // [N]
    const torch::Tensor &tileOffsets,     // [C, tile_height, tile_width]
    const torch::Tensor &tileGaussianIds, // [n_isects]
    const RenderSettings &settings,       // render settings
    const std::optional<torch::Tensor> &maybeNumContributingGaussians // [C, H, W]
) {
    FVDB_FUNC_RANGE();
    const bool isPacked = means2d.dim() == 2;

    const std::optional<torch::Tensor> backgrounds = std::nullopt;
    const std::optional<torch::Tensor> masks       = std::nullopt;

    std::optional<fvdb::JaggedTensor> numContributingGaussiansJagged = std::nullopt;

    if (maybeNumContributingGaussians.has_value()) {
        torch::Tensor numContributingGaussians = maybeNumContributingGaussians.value();
        // NOTE: We are converting the format of numContributingGaussians to a JaggedTensor so that
        // both
        //       the dense and sparse dispatch functions can use the same launch*Kernel functions.
        const auto C          = numContributingGaussians.size(0);
        const auto H          = numContributingGaussians.size(1);
        const auto W          = numContributingGaussians.size(2);
        torch::Tensor offsets = torch::arange(0,
                                              C + 1,
                                              torch::TensorOptions()
                                                  .dtype(fvdb::JOffsetsScalarType)
                                                  .device(numContributingGaussians.device())) *
                                (H * W);
        // Simple list (ldim=1), list_ids is empty with shape [0, 1]
        torch::Tensor listIds = torch::empty({0, 1},
                                             torch::TensorOptions()
                                                 .dtype(fvdb::JLIdxScalarType)
                                                 .device(numContributingGaussians.device()));

        numContributingGaussiansJagged.emplace(JaggedTensor::from_data_offsets_and_list_ids(
            numContributingGaussians.flatten(), offsets, listIds));
    }

    return AT_DISPATCH_V2(
        opacities.scalar_type(),
        "GaussianRasterizeContributingGaussianIds",
        AT_WRAP([&]() {
            auto [ids, weights] =
                isPacked ? launchRasterizeContributingGaussianIdsForwardKernel<float, true>(
                               means2d,
                               conics,
                               opacities,
                               backgrounds,
                               masks,
                               tileOffsets,
                               tileGaussianIds,
                               numContributingGaussiansJagged,
                               settings)
                         : launchRasterizeContributingGaussianIdsForwardKernel<float, false>(
                               means2d,
                               conics,
                               opacities,
                               backgrounds,
                               masks,
                               tileOffsets,
                               tileGaussianIds,
                               numContributingGaussiansJagged,
                               settings);
            return std::make_tuple(ids, weights);
        }),
        AT_EXPAND(AT_FLOATING_TYPES),
        c10::kHalf);
}

template <>
std::tuple<fvdb::JaggedTensor, fvdb::JaggedTensor>
dispatchGaussianRasterizeContributingGaussianIds<torch::kCPU>(
    // Gaussian parameters
    const torch::Tensor &means2d,         // [C, N, 2]
    const torch::Tensor &conics,          // [C, N, 3]
    const torch::Tensor &opacities,       // [N]
    const torch::Tensor &tileOffsets,     // [C, tile_height, tile_width]
    const torch::Tensor &tileGaussianIds, // [n_isects]
    const RenderSettings &settings,       // render settings
    const std::optional<torch::Tensor> &maybeNumContributingGaussians) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "CPU implementation not available");
}

template <>
__host__ std::tuple<fvdb::JaggedTensor, fvdb::JaggedTensor>
dispatchGaussianSparseRasterizeContributingGaussianIds<torch::kCUDA>(
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
    const RenderSettings &settings,
    const std::optional<fvdb::JaggedTensor> &maybeNumContributingGaussians) {
    FVDB_FUNC_RANGE();
    const bool isPacked = means2d.dim() == 2;

    const std::optional<torch::Tensor> backgrounds = std::nullopt;
    const std::optional<torch::Tensor> masks       = std::nullopt;

    return AT_DISPATCH_V2(
        opacities.scalar_type(),
        "GaussianRasterizeContributingGaussianIds",
        AT_WRAP([&]() {
            if (isPacked) {
                return launchRasterizeContributingGaussianIdsForwardKernel<float, true>(
                    means2d,
                    conics,
                    opacities,
                    backgrounds,
                    masks,
                    tileOffsets,
                    tileGaussianIds,
                    maybeNumContributingGaussians,
                    settings,
                    pixelsToRender,
                    activeTiles,
                    tilePixelMask,
                    tilePixelCumsum,
                    pixelMap);
            } else {
                return launchRasterizeContributingGaussianIdsForwardKernel<float, false>(
                    means2d,
                    conics,
                    opacities,
                    backgrounds,
                    masks,
                    tileOffsets,
                    tileGaussianIds,
                    maybeNumContributingGaussians,
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
dispatchGaussianSparseRasterizeContributingGaussianIds<torch::kCPU>(
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
    const RenderSettings &settings, // render settings
    const std::optional<fvdb::JaggedTensor> &maybeNumContributingGaussians) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "CPU implementation not available");
}

} // namespace fvdb::detail::ops
