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

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>

#include <cub/cub.cuh>

#include <optional>

namespace fvdb::detail::ops {
namespace {

// Macro to handle CUB's two-pass temporary storage pattern
#define CUB_WRAPPER(func, ...)                                                \
    do {                                                                      \
        size_t tempStorageBytes = 0;                                          \
        func(nullptr, tempStorageBytes, __VA_ARGS__);                         \
        auto &cachingAllocator = *::c10::cuda::CUDACachingAllocator::get();   \
        auto tempStorage       = cachingAllocator.allocate(tempStorageBytes); \
        func(tempStorage.get(), tempStorageBytes, __VA_ARGS__);               \
    } while (false)

/// @brief Optimized copy from padded 'Top Contributing Gaussian IDs' JaggedTensor to an unpadded
/// 'Contributing Gaussian IDs' JaggedTensor
///
/// This function copies data from srcJagged (ldim=1 with fixed-size padding) to dstJagged (ldim=2
/// with variable sizes). Each pixel in srcJagged has maxSamplesPerPixel elements, with unused
/// slots padded. We copy only the valid samples (count determined by dstOffsets).
///
/// @param srcJagged Source JaggedTensor (ldim=1) with fixed-size padding per pixel
/// @param dstJagged Destination JaggedTensor (ldim=2) with variable sizes per pixel
/// @param dstOffsets Precomputed prefix sum of valid counts [numPixels + 1]
/// @param maxSamplesPerPixel Maximum samples per pixel in source (including padding)
void
copyPaddedJaggedToJagged(const fvdb::JaggedTensor &srcJagged,
                         fvdb::JaggedTensor &dstJagged,
                         const torch::Tensor &dstOffsets,
                         const int32_t maxSamplesPerPixel) {
    const int64_t numPixels  = dstOffsets.size(0) - 1;
    const int64_t totalValid = dstJagged.jdata().numel();

    if (numPixels == 0 || totalValid == 0) {
        return;
    }

    torch::Tensor srcData2D =
        srcJagged.jdata().view({numPixels, maxSamplesPerPixel}); //[numPixels, K]

    // OPTIMIZATION: Use searchsorted to expand pixel indices instead of repeat_interleave
    // We use dstOffsets directly (the authoritative source of valid counts per pixel)
    // instead of re-scanning source data which may have uninitialized padding
    // For each output index i, find which pixel it belongs to
    torch::Tensor outputIndices = torch::arange(
        totalValid, torch::TensorOptions().dtype(torch::kInt64).device(dstOffsets.device()));
    torch::Tensor pixelIndices = torch::searchsorted(
        dstOffsets.slice(0, 1), outputIndices, /*out_int32=*/false, /*right=*/true);

    // Compute local index within each pixel: output_idx - pixel_start_offset
    torch::Tensor pixelStartOffsets = dstOffsets.index({pixelIndices});
    torch::Tensor localIndices      = outputIndices - pixelStartOffsets;

    // Compute source linear indices: pixel_idx * K + local_idx
    torch::Tensor srcLinearIndices = pixelIndices * maxSamplesPerPixel + localIndices;

    // Gather from source and copy directly to destination
    torch::Tensor srcFlat = srcData2D.flatten();
    dstJagged.jdata().copy_(srcFlat.index({srcLinearIndices}));
}

/// @brief Build output list IDs tensor for JaggedTensor construction
///
/// Supports two modes:
/// - Uniform: pixels are evenly distributed across cameras (use division/modulo)
/// - Non-uniform: pixels are distributed according to camera offsets (use searchsorted)
///
/// @param outputNumPixels Total number of output pixels
/// @param pixelsPerCamera Pixels per camera for uniform mode (-1 to use non-uniform mode)
/// @param cameraOffsets Camera offsets [numCameras+1] for non-uniform mode (ignored if uniform)
/// @param device Device for output tensor
/// @return outListIds tensor [outputNumPixels, 2]
torch::Tensor
buildOutListIds(int64_t outputNumPixels,
                int64_t pixelsPerCamera,
                const std::optional<torch::Tensor> &cameraOffsets,
                const torch::Device &device) {
    torch::Tensor pixelIndicesFlat =
        torch::arange(outputNumPixels, torch::TensorOptions().dtype(torch::kInt64).device(device));

    torch::Tensor cameraIndices, pixelIndices;

    if (pixelsPerCamera >= 0) {
        // Uniform mode: use division/modulo
        cameraIndices = (pixelIndicesFlat / pixelsPerCamera).to(fvdb::JLIdxScalarType);
        pixelIndices  = (pixelIndicesFlat % pixelsPerCamera).to(fvdb::JLIdxScalarType);
    } else {
        // Non-uniform mode: use searchsorted with camera offsets
        TORCH_CHECK(cameraOffsets.has_value(), "cameraOffsets required for non-uniform mode");
        torch::Tensor offsets = cameraOffsets.value().to(torch::kInt64);
        cameraIndices = torch::searchsorted(offsets.slice(0, 1), pixelIndicesFlat, false, true)
                            .to(fvdb::JLIdxScalarType);
        torch::Tensor cameraStartOffsets = offsets.index({cameraIndices.to(torch::kInt64)});
        pixelIndices = (pixelIndicesFlat - cameraStartOffsets).to(fvdb::JLIdxScalarType);
    }

    return torch::stack({cameraIndices, pixelIndices}, 1);
}

/// @brief Convert padded fixed-size output to variable-size JaggedTensor
///
/// Takes a counts-per-pixel tensor and converts padded source JaggedTensors
/// to variable-size output JaggedTensors. This encapsulates the common pattern of:
/// 1. Computing sample offsets via CUB inclusive sum
/// 2. Computing output indices via searchsorted
/// 3. Computing list IDs (camera/pixel indices)
/// 4. Allocating output data
/// 5. Creating JaggedTensors
/// 6. Copying data from padded source to variable-size destination
///
/// @tparam ScalarType Scalar type for weights
/// @param srcIds Source padded IDs JaggedTensor [numPixels, maxSamples]
/// @param srcWeights Source padded weights JaggedTensor [numPixels, maxSamples]
/// @param countsPerPixel Valid sample count per pixel [numPixels], int64_t contiguous
/// @param totalCount Sum of countsPerPixel (total output samples)
/// @param numCameras Number of cameras
/// @param maxSamplesPerPixel Max samples per pixel in source (K)
/// @param pixelsPerCamera Pixels per camera for uniform mode, or -1 for non-uniform
/// @param cameraOffsets Camera offsets [C+1] for non-uniform mode, nullopt for uniform
/// @param options Tensor options for output data allocation
/// @return Tuple of (outIdsJagged, outWeightsJagged)
template <typename ScalarType>
std::tuple<fvdb::JaggedTensor, fvdb::JaggedTensor>
convertPaddedToVariableSizeJagged(const fvdb::JaggedTensor &srcIds,
                                  const fvdb::JaggedTensor &srcWeights,
                                  const torch::Tensor &countsPerPixel,
                                  int64_t totalCount,
                                  int64_t numCameras,
                                  int32_t maxSamplesPerPixel,
                                  int64_t pixelsPerCamera,
                                  const std::optional<torch::Tensor> &cameraOffsets,
                                  const torch::TensorOptions &options) {
    const int64_t outputNumPixels = countsPerPixel.numel();
    const torch::Device device    = countsPerPixel.device();

    // Handle empty case
    if (totalCount == 0) {
        torch::Tensor emptyData = torch::empty({0}, options.dtype(torch::kInt32));
        torch::Tensor emptyWeights =
            torch::empty({0}, options.dtype(c10::CppTypeToScalarType<ScalarType>::value));
        torch::Tensor emptyIndices =
            torch::empty({0}, torch::TensorOptions().dtype(fvdb::JIdxScalarType).device(device));
        torch::Tensor emptyListIds = torch::empty(
            {0, 2}, torch::TensorOptions().dtype(fvdb::JLIdxScalarType).device(device));

        auto emptyIdsJagged = fvdb::JaggedTensor::from_data_indices_and_list_ids(
            emptyData, emptyIndices, emptyListIds, outputNumPixels);
        auto emptyWeightsJagged = fvdb::JaggedTensor::from_data_indices_and_list_ids(
            emptyWeights, emptyIndices.clone(), emptyListIds.clone(), outputNumPixels);
        return std::make_tuple(emptyIdsJagged, emptyWeightsJagged);
    }

    // Compute sample offsets via CUB inclusive sum
    torch::Tensor sampleOffsets = torch::zeros(
        {outputNumPixels + 1}, torch::TensorOptions().dtype(torch::kInt64).device(device));
    {
        auto stream = at::cuda::getCurrentCUDAStream(device.index());
        CUB_WRAPPER(cub::DeviceScan::InclusiveSum,
                    countsPerPixel.data_ptr<int64_t>(),
                    sampleOffsets.data_ptr<int64_t>() + 1,
                    static_cast<int>(outputNumPixels),
                    stream);
    }

    // Compute output indices via searchsorted
    torch::Tensor flatIndices =
        torch::arange(totalCount, torch::TensorOptions().dtype(torch::kInt64).device(device));
    torch::Tensor outIndices =
        torch::searchsorted(sampleOffsets.slice(0, 1), flatIndices, false, true)
            .to(fvdb::JIdxScalarType);

    // Build list IDs
    torch::Tensor outListIds =
        buildOutListIds(outputNumPixels, pixelsPerCamera, cameraOffsets, device);

    // Allocate output data
    torch::Tensor outIdsData = torch::empty({totalCount}, options.dtype(torch::kInt32));
    torch::Tensor outWeightsData =
        torch::empty({totalCount}, options.dtype(c10::CppTypeToScalarType<ScalarType>::value));

    // Create JaggedTensors
    auto outIdsJagged = fvdb::JaggedTensor::from_jdata_joffsets_jidx_and_lidx_unsafe(
        outIdsData, sampleOffsets, outIndices, outListIds, numCameras);
    auto outWeightsJagged = fvdb::JaggedTensor::from_jdata_joffsets_jidx_and_lidx_unsafe(
        outWeightsData, sampleOffsets, outIndices.clone(), outListIds.clone(), numCameras);

    // Copy from padded source to variable-size destination
    copyPaddedJaggedToJagged(srcIds, outIdsJagged, sampleOffsets, maxSamplesPerPixel);
    copyPaddedJaggedToJagged(srcWeights, outWeightsJagged, sampleOffsets, maxSamplesPerPixel);

    return std::make_tuple(outIdsJagged, outWeightsJagged);
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

                    // Convert global gaussian ID to per-camera local ID (0 to N-1)
                    // In packed mode, gaussian.id is already the local ID
                    int32_t localId;
                    if constexpr (IS_PACKED) {
                        localId = gaussian.id;
                    } else {
                        localId = gaussian.id % commonArgs.mNumGaussiansPerCamera;
                    }

                    // Write to thread-local buffer instead of global memory
                    if (writeIndex < MAX_LOCAL_BUFFER_SIZE) {
                        localIds[writeIndex]     = localId;
                        localWeights[writeIndex] = radianceWeight;
                        writeIndex++;
                    } else {
                        // Overflow: buffer is full, fall back to direct write (rare case)
                        writeId(pixIdx, writeIndex, localId);
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
        torch::Tensor numValidSamples = (outIds.jdata() != -1).to(torch::kInt32).sum(-1);

        const auto outputNumPixels = numValidSamples.numel();

        // Build JaggedTensor structure
        const bool tileOffsetsAreSparse = tileOffsets.dim() == 1;
        const auto C = tileOffsetsAreSparse ? outIds.num_outer_lists() : tileOffsets.size(0);
        const auto pixelsPerCamera = outputNumPixels / C;

        // Prepare counts and compute total
        torch::Tensor numValidSamplesData = numValidSamples.to(torch::kInt64).contiguous();
        const auto totalCount             = numValidSamplesData.sum().item<int64_t>();

        // Use uniform pixels per camera mode
        return convertPaddedToVariableSizeJagged<ScalarType>(
            outIds,
            outWeights,
            numValidSamplesData,
            totalCount,
            C,
            settings.numDepthSamples,
            pixelsPerCamera, // uniform mode
            std::nullopt,    // no camera offsets needed
            means2d.options());
    }

    TORCH_CHECK_VALUE(maybeNumContributingGaussians.has_value(),
                      "numContributingGaussians must be provided if not using top-k mode");

    const JaggedTensor &numContributingGaussians = maybeNumContributingGaussians.value();

    // Get C from tileOffsets (tileOffsets is [C, TH, TW] in dense mode, or [AT+1] in sparse)
    const bool tileOffsetsAreSparse = tileOffsets.dim() == 1;
    const auto C =
        tileOffsetsAreSparse ? numContributingGaussians.num_outer_lists() : tileOffsets.size(0);
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

    const auto tileExtentH = (settings.imageHeight + settings.tileSize - 1) / settings.tileSize;
    const auto tileExtentW = (settings.imageWidth + settings.tileSize - 1) / settings.tileSize;

    if (tileOffsets.dim() == 3) {
        TORCH_CHECK_VALUE(tileOffsets.size(2) == tileExtentW,
                          "tileOffsets width must match the number of tiles in image size");
        TORCH_CHECK_VALUE(tileOffsets.size(1) == tileExtentH,
                          "tileOffsets height must match the number of tiles in image size");
    }

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

    //  Convert padded output to variable-size JaggedTensor
    const auto outputNumCameras = numContributingGaussians.num_outer_lists();

    // Prepare counts tensor
    torch::Tensor numContributingGaussiansData =
        numContributingGaussians.jdata().to(torch::kInt64).contiguous();

    // Use non-uniform pixels per camera mode
    return convertPaddedToVariableSizeJagged<ScalarType>(
        outIds,
        outWeights,
        numContributingGaussiansData,
        numContributingGaussiansSum,
        outputNumCameras,
        maxDepthSamplesPerPixel,
        -1,                                  // non-uniform mode
        numContributingGaussians.joffsets(), // camera offsets
        means2d.options());
}

} // namespace

template <>
__host__ std::tuple<fvdb::JaggedTensor, fvdb::JaggedTensor>
dispatchGaussianRasterizeContributingGaussianIds<torch::kCUDA>(
    // Gaussian parameters
    const torch::Tensor &means2d,         // [C, N, 2] or [nnz, 2]
    const torch::Tensor &conics,          // [C, N, 3] or [nnz, 3]
    const torch::Tensor &opacities,       // [C, N] or [nnz]
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
    const torch::Tensor &means2d,         // [C, N, 2] or [nnz, 2]
    const torch::Tensor &conics,          // [C, N, 3] or [nnz, 3]
    const torch::Tensor &opacities,       // [C, N] or [nnz]
    const torch::Tensor &tileOffsets,     // [C, tile_height, tile_width]
    const torch::Tensor &tileGaussianIds, // [n_isects]
    const RenderSettings &settings,       // render settings
    const std::optional<torch::Tensor> &maybeNumContributingGaussians) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "CPU implementation not available");
}

template <>
__host__ std::tuple<fvdb::JaggedTensor, fvdb::JaggedTensor>
dispatchGaussianSparseRasterizeContributingGaussianIds<torch::kCUDA>(
    const torch::Tensor &means2d,     // [C, N, 2] or [nnz, 2]
    const torch::Tensor &conics,      // [C, N, 3] or [nnz, 3]
    const torch::Tensor &opacities,   // [C, N] or [nnz]
    const torch::Tensor &tileOffsets, // [C, tile_height, tile_width] (dense) or [AT + 1] (sparse)
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
    const torch::Tensor &means2d,     // [C, N, 2] or [nnz, 2]
    const torch::Tensor &conics,      // [C, N, 3] or [nnz, 3]
    const torch::Tensor &opacities,   // [C, N] or [nnz]
    const torch::Tensor &tileOffsets, // [C, tile_height, tile_width] (dense) or [AT + 1] (sparse)
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
