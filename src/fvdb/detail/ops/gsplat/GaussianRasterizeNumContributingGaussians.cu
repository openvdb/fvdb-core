// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#include <fvdb/detail/ops/gsplat/Gaussian2D.cuh>
#include <fvdb/detail/ops/gsplat/GaussianRasterize.cuh>
#include <fvdb/detail/ops/gsplat/GaussianRasterizeNumContributingGaussians.h>
#include <fvdb/detail/ops/gsplat/GaussianRenderSettings.h>
#include <fvdb/detail/ops/gsplat/GaussianVectorTypes.cuh>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/Nvtx.h>

#include <c10/cuda/CUDAGuard.h>

#include <optional>

namespace fvdb::detail::ops {
namespace {

// Structure to hold arguments and methods for the rasterize num contributing gaussians kernel
template <typename ScalarType, bool IS_PACKED, typename TileIntersectionsT>
struct RasterizeNumContributingGaussiansArgs {
    using CommonArgs = RasterizeCommonArgs<ScalarType, 1, IS_PACKED, TileIntersectionsT>;
    CommonArgs commonArgs;

    // In Dense mode, first dimension X = C * imageHeight * imageWidth
    // In Sparse mode, first dimension X = C * nPixels_i (i from 0 to C-1)
    JaggedRAcc64<int32_t, 1> mOutNumContributingGaussians; // [X]
    JaggedRAcc64<ScalarType, 1> mOutAlphas;                // [X]

    RasterizeNumContributingGaussiansArgs(
        const torch::Tensor &means2d,                      // [C, N, 2] or [nnz, 2]
        const torch::Tensor &conics,                       // [C, N, 3] or [nnz, 3]
        const torch::Tensor &opacities,                    // [C, N] or [nnz]
        const at::optional<torch::Tensor> &backgrounds,    // [C, 1]
        const at::optional<torch::Tensor> &masks,          // [C, numTilesH, numTilesW]
        const TileIntersectionsT &tileIntersections,
        const uint32_t imageWidth,
        const uint32_t imageHeight,
        const uint32_t imageOriginW,
        const uint32_t imageOriginH,
        const uint32_t tileSize,
        const fvdb::JaggedTensor &outNumContributingGaussians,          // [C, imgH, imgW]
        const fvdb::JaggedTensor &outAlphas) // [C, imgH, imgW]

        : commonArgs(
              means2d,
              conics,
              opacities,
              std::nullopt,
              backgrounds,
              masks,
              tileIntersections,
              RenderWindow2D{imageWidth, imageHeight, imageOriginW, imageOriginH}),
          mOutNumContributingGaussians(initJaggedAccessor<int32_t, 1>(
              outNumContributingGaussians, "outNumContributingGaussians")),
          mOutAlphas(initJaggedAccessor<ScalarType, 1>(outAlphas, "outAlphas")) {}

    /// @brief Write an alpha sample for a pixel
    /// @param pixelIndex The index of the pixel
    /// @param alpha The alpha value to write
    __device__ void
    writeAlpha(uint64_t pixelIndex, ScalarType alpha) {
        mOutAlphas.data()[pixelIndex] = alpha;
    }

    /// @brief Write a number of contributing Gaussians for a pixel
    /// @param pixelIndex The index of the pixel
    /// @param numContributingGaussians The number of contributing Gaussians to write
    __device__ void
    writeNumContributingGaussians(uint64_t pixelIndex, int32_t numContributingGaussians) {
        mOutNumContributingGaussians.data()[pixelIndex] = numContributingGaussians;
    }

    __device__ void
    volumeRenderTileForward(const uint32_t cameraId,
                            const uint32_t row,
                            const uint32_t col,
                            const int32_t tileRow,
                            const int32_t tileCol,
                            const uint32_t blockSize,
                            const bool pixelIsActive,
                            const uint32_t activePixelIndex) {
        const auto [firstGaussianIdInBlock, lastGaussianIdInBlock] =
            commonArgs.mTileIntersections.tileGaussianRangeFromBlock(
                blockIdx.x, cameraId, tileRow, tileCol);

        alignas(Gaussian2D<ScalarType>) extern __shared__ char s[];
        auto *sharedGaussians = reinterpret_cast<Gaussian2D<ScalarType> *>(s); // [blockSize]

        const auto tidx = threadIdx.y * blockDim.x + threadIdx.x;

        ScalarType accumTransmittance    = 1.0f;
        int32_t numContributingGaussians = 0;

        // We don't return right away if the pixel is not in the image since we want to use
        // this thread to load gaussians into shared memory
        bool done = !pixelIsActive;

        const uint32_t numBatches =
            commonArgs.numGaussianBatches(firstGaussianIdInBlock, lastGaussianIdInBlock, blockSize);

        // (row, col) coordinates are relative to the specified image origin which may
        // be a crop so we need to add the origin to get the absolute pixel coordinates
        const ScalarType px = col + commonArgs.mRenderWindow.originW + ScalarType{0.5f};
        const ScalarType py = row + commonArgs.mRenderWindow.originH + ScalarType{0.5f};

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
            commonArgs.loadGaussianBatchFrontToBack(
                firstGaussianIdInBlock, lastGaussianIdInBlock, b, blockSize, tidx, sharedGaussians);

            // Sync threads so all gaussians for this batch are loaded in shared memory
            __syncthreads();

            // Volume render Gaussians in this batch
            if (pixelIsActive) { // skip inactive sparse pixels
                const uint32_t batchStart =
                    commonArgs.gaussianBatchStartFrontToBack(firstGaussianIdInBlock, b, blockSize);
                const uint32_t batchSize =
                    commonArgs.gaussianBatchSizeFrontToBack(lastGaussianIdInBlock,
                                                            batchStart,
                                                            blockSize);
                for (uint32_t t = 0; (t < batchSize) && !done; ++t) {
                    const Gaussian2D<ScalarType> &gaussian = sharedGaussians[t];

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

                    numContributingGaussians++;

                    accumTransmittance = nextTransmittance;
                }
            }
        }

        if (pixelIsActive) {
            const auto pixIdx = commonArgs.mTileIntersections.pixelIndexFromBlock(
                cameraId, row, col, activePixelIndex, blockIdx.x);

            writeNumContributingGaussians(pixIdx, numContributingGaussians);
            writeAlpha(pixIdx, 1.0f - accumTransmittance);
        }
    }
};

/****************************************************************************
 * Rasterization to Pixels Forward Pass
 ****************************************************************************/

template <typename ScalarType, bool IS_PACKED, typename TileIntersectionsT>
__global__ void
rasterizeNumContributingGaussiansForward(
    RasterizeNumContributingGaussiansArgs<ScalarType, IS_PACKED, TileIntersectionsT> args) {
    auto &commonArgs = args.commonArgs;

    // each thread draws one pixel, but also timeshares caching gaussians in a
    // shared tile
    const auto [cameraId, tileRow, tileCol, row, col] =
        commonArgs.mTileIntersections.coordinates(blockIdx.x, threadIdx.x, threadIdx.y);

    // NOTE: We keep threads which correspond to pixels outside the image bounds around
    //       to load gaussians from global memory, but they do not contribute to the output.

    // pixelInImage: Whether this pixel is inside the image bounds.
    // activePixelIndex: Index of this pixel in the output for the block if it is active
    // (sparse mode only).
    __shared__ typename TileIntersectionsT::ActivePixelScratch activePixelScratch;
    const auto [pixelInImage, activePixelIndex] = commonArgs.mTileIntersections.activePixelIndexFromBlock(
        blockIdx.x, threadIdx.y, threadIdx.x, row, col, activePixelScratch);

    if (commonArgs.mHasMasks && pixelInImage && !commonArgs.mMasks[cameraId][tileRow][tileCol]) {
        auto pixIdx = commonArgs.mTileIntersections.pixelIndexFromBlock(
            cameraId, row, col, activePixelIndex, blockIdx.x);

        args.writeNumContributingGaussians(
            pixIdx, commonArgs.mHasBackgrounds ? commonArgs.mBackgrounds[cameraId][0] : 0);

        return;
    }

    args.volumeRenderTileForward(cameraId,
                                 row,
                                 col,
                                 tileRow,
                                 tileCol,
                                 blockDim.x * blockDim.y,
                                 pixelInImage,
                                 activePixelIndex);
}

template <typename ScalarType, bool IS_PACKED>
std::tuple<fvdb::JaggedTensor, fvdb::JaggedTensor>
launchRasterizeNumContributingGaussiansForwardKernel(
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

    uint32_t C = 0;
    dispatchTileIntersectionsAccessor(
        tileOffsets,
        tileGaussianIds,
        RenderWindow2D{settings.imageWidth, settings.imageHeight, settings.imageOriginW, settings.imageOriginH},
        settings.tileSize,
        0,
        [&](const auto &tileIntersectionsAccessor) {
            TORCH_CHECK_VALUE(tileIntersectionsAccessor.numTilesW() ==
                                  (settings.imageWidth + settings.tileSize - 1) / settings.tileSize,
                              "tileOffsets width must match the number of tiles in image size");
            TORCH_CHECK_VALUE(tileIntersectionsAccessor.numTilesH() ==
                                  (settings.imageHeight + settings.tileSize - 1) / settings.tileSize,
                              "tileOffsets height must match the number of tiles in image size");
            C = tileIntersectionsAccessor.cameraCount(static_cast<uint32_t>(means2d.size(0)));

        },
        activeTiles,
        tilePixelMask,
        tilePixelCumsum,
        pixelMap);

    // Get C from tileOffsets for dense mode
    // For sparse mode, C is unused, only used for output sizing for dense mode
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
    std::vector<torch::Tensor> numContributingGaussiansToRenderVec;
    std::vector<torch::Tensor> alphasToRenderVec;

    for (const auto &size: sizes) {
        numContributingGaussiansToRenderVec.push_back(
            torch::empty({size}, means2d.options().dtype(torch::kInt32)));
        alphasToRenderVec.push_back(torch::empty(
            {size}, means2d.options().dtype(c10::CppTypeToScalarType<ScalarType>::value)));
    }

    auto outNumContributingGaussians = fvdb::JaggedTensor(numContributingGaussiansToRenderVec);
    auto outAlphas                   = fvdb::JaggedTensor(alphasToRenderVec);

    const at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    // Each pixel in each tile will cache a gaussian consisting of:
    //   - int32_t  gaussian_id; -- 4 bytes
    //   - vec2t    xy;          -- 8 bytes for float32
    //   - scalar_t opacity;     -- 4 bytes for float32
    //   - vec3t    conic;       -- 12 bytes for float32
    const uint32_t sharedMem =
        settings.tileSize * settings.tileSize * sizeof(Gaussian2D<ScalarType>);

    const auto launchWithTileIntersections = [&](auto tileIntersections) {
        using TileIntersectionsT = decltype(tileIntersections);
        if (cudaFuncSetAttribute(
                rasterizeNumContributingGaussiansForward<ScalarType, IS_PACKED, TileIntersectionsT>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                sharedMem) != cudaSuccess) {
            AT_ERROR("Failed to set maximum shared memory size (requested ",
                     sharedMem,
                     " bytes), try lowering tile_size.");
        }

        auto args = RasterizeNumContributingGaussiansArgs<ScalarType, IS_PACKED, TileIntersectionsT>(
            means2d,
            conics,
            opacities,
            backgrounds,
            masks,
            tileIntersections,
            settings.imageWidth,
            settings.imageHeight,
            settings.imageOriginW,
            settings.imageOriginH,
            settings.tileSize,
            outNumContributingGaussians,
            outAlphas);
        const dim3 blockDim = {settings.tileSize, settings.tileSize, 1};
        const dim3 gridDim  = {static_cast<uint32_t>(tileIntersections.numActiveTiles()), 1, 1};

        rasterizeNumContributingGaussiansForward<ScalarType, IS_PACKED, TileIntersectionsT>
            <<<gridDim, blockDim, sharedMem, stream>>>(args);
    };

    dispatchTileIntersectionsAccessor(
        tileOffsets,
        tileGaussianIds,
        RenderWindow2D{settings.imageWidth, settings.imageHeight, settings.imageOriginW, settings.imageOriginH},
        settings.tileSize,
        0,
        [&](const auto &tileIntersectionsAccessor) {
            launchWithTileIntersections(tileIntersectionsAccessor);
        },
        activeTiles,
        tilePixelMask,
        tilePixelCumsum,
        pixelMap);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));

    return std::make_tuple(outNumContributingGaussians, outAlphas);
}

} // namespace

template <>
std::tuple<torch::Tensor, torch::Tensor>
dispatchGaussianRasterizeNumContributingGaussians<torch::kCUDA>(
    // Gaussian parameters
    const torch::Tensor &means2d,         // [C, N, 2] or [nnz, 2]
    const torch::Tensor &conics,          // [C, N, 3] or [nnz, 3]
    const torch::Tensor &opacities,       // [C, N] or [nnz]
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
        "GaussianRasterizeNumContributingGaussians",
        AT_WRAP([&]() {
            auto [numContributingGaussians, alphas] =
                isPacked ? launchRasterizeNumContributingGaussiansForwardKernel<float, true>(
                               means2d,
                               conics,
                               opacities,
                               backgrounds,
                               masks,
                               tileOffsets,
                               tileGaussianIds,
                               settings)
                         : launchRasterizeNumContributingGaussiansForwardKernel<float, false>(
                               means2d,
                               conics,
                               opacities,
                               backgrounds,
                               masks,
                               tileOffsets,
                               tileGaussianIds,
                               settings);
            // Get C from tileOffsets for dense mode
            const auto C = tileOffsets.size(0);
            return std::make_tuple(
                numContributingGaussians.jdata().reshape(
                    {C, settings.imageHeight, settings.imageWidth}),
                alphas.jdata().reshape({C, settings.imageHeight, settings.imageWidth}));
        }),
        AT_EXPAND(AT_FLOATING_TYPES),
        c10::kHalf);
}

template <>
std::tuple<torch::Tensor, torch::Tensor>
dispatchGaussianRasterizeNumContributingGaussians<torch::kCPU>(
    // Gaussian parameters
    const torch::Tensor &means2d,         // [C, N, 2] or [nnz, 2]
    const torch::Tensor &conics,          // [C, N, 3] or [nnz, 3]
    const torch::Tensor &opacities,       // [C, N] or [nnz]
    const torch::Tensor &tileOffsets,     // [C, tile_height, tile_width]
    const torch::Tensor &tileGaussianIds, // [n_isects]
    const RenderSettings &settings        // render settings
) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "CPU implementation not available");
}

template <>
std::tuple<fvdb::JaggedTensor, fvdb::JaggedTensor>
dispatchGaussianSparseRasterizeNumContributingGaussians<torch::kCUDA>(
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
    const RenderSettings &settings) { // render settings
    FVDB_FUNC_RANGE();
    const bool isPacked = means2d.dim() == 2;

    const std::optional<torch::Tensor> backgrounds = std::nullopt;
    const std::optional<torch::Tensor> masks       = std::nullopt;

    return AT_DISPATCH_V2(
        opacities.scalar_type(),
        "GaussianRasterizeNumContributingGaussians",
        AT_WRAP([&]() {
            if (isPacked) {
                return launchRasterizeNumContributingGaussiansForwardKernel<float, true>(
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
                return launchRasterizeNumContributingGaussiansForwardKernel<float, false>(
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
dispatchGaussianSparseRasterizeNumContributingGaussians<torch::kCPU>(
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
    const RenderSettings &settings) { // render settings
    TORCH_CHECK_NOT_IMPLEMENTED(false, "CPU implementation not available");
}

} // namespace fvdb::detail::ops
