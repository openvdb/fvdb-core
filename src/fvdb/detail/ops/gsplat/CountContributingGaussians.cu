// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#include <fvdb/detail/ops/gsplat/CountContributingGaussians.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/Nvtx.h>
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/gsplat/Gaussian2D.cuh>
#include <fvdb/detail/utils/gsplat/GaussianRasterize.cuh>

#include <c10/cuda/CUDAGuard.h>

#include <cstdint>
#include <optional>

namespace fvdb::detail::ops {
namespace {

// Structure to hold arguments and methods for the rasterize num contributing gaussians kernel
template <typename ScalarType, bool IS_PACKED> struct RasterizeNumContributingGaussiansArgs {
    using CommonArgs = RasterizeCommonArgs<ScalarType, 1, IS_PACKED>;
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
        const uint32_t imageWidth,
        const uint32_t imageHeight,
        const uint32_t imageOriginW,
        const uint32_t imageOriginH,
        const uint32_t tileSize,
        const torch::Tensor &tileOffsets,                               // [C, numTilesH, numTilesW]
        const torch::Tensor &tileGaussianIds,                           // [totalIntersections]
        const fvdb::JaggedTensor &outNumContributingGaussians,          // [C, imgH, imgW]
        const fvdb::JaggedTensor &outAlphas,                            // [C, imgH, imgW]
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
                     RenderWindow2D{imageWidth, imageHeight, imageOriginW, imageOriginH},
                     tileSize,
                     0,
                     tileOffsets,
                     tileGaussianIds,
                     activeTiles,
                     tilePixelMask,
                     tilePixelCumsum,
                     pixelMap),
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
                            const uint32_t firstGaussianIdInBlock,
                            const uint32_t lastGaussianIdInBlock,
                            const uint32_t blockSize,
                            const bool pixelIsActive,
                            const uint32_t activePixelIndex) {
        alignas(Gaussian2D<ScalarType>) extern __shared__ char s[];
        auto *sharedGaussians = reinterpret_cast<Gaussian2D<ScalarType> *>(s); // [blockSize]

        const auto tidx = threadIdx.y * blockDim.x + threadIdx.x;

        ScalarType accumTransmittance    = 1.0f;
        int32_t numContributingGaussians = 0;

        // We don't return right away if the pixel is not in the image since we want to use
        // this thread to load gaussians into shared memory
        bool done = !pixelIsActive;

        const uint32_t numBatches =
            (lastGaussianIdInBlock - firstGaussianIdInBlock + blockSize - 1) / blockSize;

        // (row, col) coordinates are relative to the specified image origin which may
        // be a crop so we need to add the origin to get the absolute pixel coordinates
        const ScalarType px = col + commonArgs.renderOriginX() + ScalarType{0.5f};
        const ScalarType py = row + commonArgs.renderOriginY() + ScalarType{0.5f};

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
                    const Gaussian2D<ScalarType> &gaussian = sharedGaussians[t];

                    const auto [gaussianIsValid, delta, expMinusSigma, alpha] =
                        commonArgs.evalGaussian(gaussian, px, py);

                    if (!gaussianIsValid) {
                        continue;
                    }

                    const ScalarType nextTransmittance = accumTransmittance * (1.0f - alpha);
                    if (nextTransmittance <= kTransmittanceThreshold) { // this pixel is done
                        done = true;
                        break;
                    }

                    numContributingGaussians++;

                    accumTransmittance = nextTransmittance;
                }
            }
        }

        if (pixelIsActive) {
            const auto pixIdx = commonArgs.pixelIndex(cameraId, row, col, activePixelIndex);

            writeNumContributingGaussians(pixIdx, numContributingGaussians);
            writeAlpha(pixIdx, 1.0f - accumTransmittance);
        }
    }
};

/****************************************************************************
 * Rasterization to Pixels Forward Pass
 ****************************************************************************/

template <typename ScalarType, bool IS_PACKED>
__global__ void
rasterizeNumContributingGaussiansForward(
    RasterizeNumContributingGaussiansArgs<ScalarType, IS_PACKED> args) {
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
    // activePixelIndex: Index of this pixel in the output for the block if it is active
    // (sparse mode only).
    bool pixelInImage{false};
    uint32_t activePixelIndex{0};
    cuda::std::tie(pixelInImage, activePixelIndex) = commonArgs.activePixelIndex(row, col);

    if (commonArgs.mHasMasks && pixelInImage && !commonArgs.mMasks[cameraId][tileRow][tileCol]) {
        auto pixIdx = commonArgs.pixelIndex(cameraId, row, col, activePixelIndex);

        args.writeNumContributingGaussians(
            pixIdx, commonArgs.mHasBackgrounds ? commonArgs.mBackgrounds[cameraId][0] : 0);

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
    const uint32_t imageWidth,
    const uint32_t imageHeight,
    const uint32_t imageOriginW,
    const uint32_t imageOriginH,
    const uint32_t tileSize,
    const std::optional<fvdb::JaggedTensor> &pixelsToRender = std::nullopt, // [C, NumPixels, 2]
    const std::optional<torch::Tensor> &activeTiles         = std::nullopt,
    const std::optional<torch::Tensor> &tilePixelMask       = std::nullopt,
    const std::optional<torch::Tensor> &tilePixelCumsum     = std::nullopt,
    const std::optional<torch::Tensor> &pixelMap            = std::nullopt) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(means2d));

    // tileOffsets can be 3D (dense) or 1D (sparse)
    const bool tileOffsetsAreSparse = tileOffsets.dim() == 1;
    if (!tileOffsetsAreSparse) {
        TORCH_CHECK_VALUE(tileOffsets.size(2) == (imageWidth + tileSize - 1) / tileSize,
                          "tileOffsets width must match the number of tiles in image size");
        TORCH_CHECK_VALUE(tileOffsets.size(1) == (imageHeight + tileSize - 1) / tileSize,
                          "tileOffsets height must match the number of tiles in image size");
    }

    // Get C from tileOffsets for dense mode
    // For sparse mode, C is unused, only used for output sizing for dense mode
    const uint32_t C           = tileOffsetsAreSparse ? 0 : tileOffsets.size(0);
    const uint32_t tileExtentH = (imageHeight + tileSize - 1) / tileSize;
    const uint32_t tileExtentW = (imageWidth + tileSize - 1) / tileSize;

    TORCH_CHECK_VALUE(pixelMap.has_value() == pixelsToRender.has_value(),
                      "pixelMap and pixelsToRender must be provided together");
    if (pixelMap.has_value()) {
        TORCH_CHECK_VALUE(pixelMap.value().size(0) == pixelsToRender.value().numel() / 2,
                          "pixelMap must have the same number of elements as pixelsToRender");
    }

    auto sizes = pixelsToRender.has_value() ? pixelsToRender->lsizes1()
                                            : std::vector<int64_t>{C * imageHeight * imageWidth};
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
    const uint32_t sharedMem = tileSize * tileSize * sizeof(Gaussian2D<ScalarType>);

    if (cudaFuncSetAttribute(rasterizeNumContributingGaussiansForward<ScalarType, IS_PACKED>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             sharedMem) != cudaSuccess) {
        AT_ERROR("Failed to set maximum shared memory size (requested ",
                 sharedMem,
                 " bytes), try lowering tile_size.");
    }

    const dim3 blockDim = {tileSize, tileSize, 1};
    const dim3 gridDim  = activeTiles.has_value() // sparse mode
                              ? dim3(activeTiles.value().size(0), 1, 1)
                              : dim3(C * tileExtentH * tileExtentW, 1, 1);
    auto args =
        RasterizeNumContributingGaussiansArgs<ScalarType, IS_PACKED>(means2d,
                                                                     conics,
                                                                     opacities,
                                                                     backgrounds,
                                                                     masks,
                                                                     imageWidth,
                                                                     imageHeight,
                                                                     imageOriginW,
                                                                     imageOriginH,
                                                                     tileSize,
                                                                     tileOffsets,
                                                                     tileGaussianIds,
                                                                     outNumContributingGaussians,
                                                                     outAlphas,
                                                                     activeTiles,
                                                                     tilePixelMask,
                                                                     tilePixelCumsum,
                                                                     pixelMap);

    rasterizeNumContributingGaussiansForward<<<gridDim, blockDim, sharedMem, stream>>>(args);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));

    return std::make_tuple(outNumContributingGaussians, outAlphas);
}

} // namespace

template <torch::DeviceType>
std::tuple<torch::Tensor, torch::Tensor>
dispatchCountContributingGaussians(const torch::Tensor &means2d,
                                   const torch::Tensor &conics,
                                   const torch::Tensor &opacities,
                                   const torch::Tensor &tileOffsets,
                                   const torch::Tensor &tileGaussianIds,
                                   uint32_t imageWidth,
                                   uint32_t imageHeight,
                                   uint32_t imageOriginW,
                                   uint32_t imageOriginH,
                                   uint32_t tileSize);

template <torch::DeviceType>
std::tuple<fvdb::JaggedTensor, fvdb::JaggedTensor>
dispatchCountContributingGaussiansSparse(const torch::Tensor &means2d,
                                         const torch::Tensor &conics,
                                         const torch::Tensor &opacities,
                                         const torch::Tensor &tileOffsets,
                                         const torch::Tensor &tileGaussianIds,
                                         const fvdb::JaggedTensor &pixelsToRender,
                                         const torch::Tensor &activeTiles,
                                         const torch::Tensor &tilePixelMask,
                                         const torch::Tensor &tilePixelCumsum,
                                         const torch::Tensor &pixelMap,
                                         uint32_t imageWidth,
                                         uint32_t imageHeight,
                                         uint32_t imageOriginW,
                                         uint32_t imageOriginH,
                                         uint32_t tileSize);

template <>
std::tuple<torch::Tensor, torch::Tensor>
dispatchCountContributingGaussians<torch::kCUDA>(
    // Gaussian parameters
    const torch::Tensor &means2d,         // [C, N, 2] or [nnz, 2]
    const torch::Tensor &conics,          // [C, N, 3] or [nnz, 3]
    const torch::Tensor &opacities,       // [C, N] or [nnz]
    const torch::Tensor &tileOffsets,     // [C, tile_height, tile_width]
    const torch::Tensor &tileGaussianIds, // [n_isects]
    uint32_t imageWidth,
    uint32_t imageHeight,
    uint32_t imageOriginW,
    uint32_t imageOriginH,
    uint32_t tileSize

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
                               imageWidth,
                               imageHeight,
                               imageOriginW,
                               imageOriginH,
                               tileSize)
                         : launchRasterizeNumContributingGaussiansForwardKernel<float, false>(
                               means2d,
                               conics,
                               opacities,
                               backgrounds,
                               masks,
                               tileOffsets,
                               tileGaussianIds,
                               imageWidth,
                               imageHeight,
                               imageOriginW,
                               imageOriginH,
                               tileSize);
            // Get C from tileOffsets for dense mode
            const auto C = tileOffsets.size(0);
            return std::make_tuple(
                numContributingGaussians.jdata().reshape({C, imageHeight, imageWidth}),
                alphas.jdata().reshape({C, imageHeight, imageWidth}));
        }),
        AT_EXPAND(AT_FLOATING_TYPES),
        c10::kHalf);
}

template <>
std::tuple<torch::Tensor, torch::Tensor>
dispatchCountContributingGaussians<torch::kCPU>(
    // Gaussian parameters
    const torch::Tensor &means2d,         // [C, N, 2] or [nnz, 2]
    const torch::Tensor &conics,          // [C, N, 3] or [nnz, 3]
    const torch::Tensor &opacities,       // [C, N] or [nnz]
    const torch::Tensor &tileOffsets,     // [C, tile_height, tile_width]
    const torch::Tensor &tileGaussianIds, // [n_isects]
    uint32_t imageWidth,
    uint32_t imageHeight,
    uint32_t imageOriginW,
    uint32_t imageOriginH,
    uint32_t tileSize) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "CPU implementation not available");
}

template <>
std::tuple<fvdb::JaggedTensor, fvdb::JaggedTensor>
dispatchCountContributingGaussiansSparse<torch::kCUDA>(
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
    uint32_t imageWidth,
    uint32_t imageHeight,
    uint32_t imageOriginW,
    uint32_t imageOriginH,
    uint32_t tileSize) {
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
                    imageWidth,
                    imageHeight,
                    imageOriginW,
                    imageOriginH,
                    tileSize,
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
                    imageWidth,
                    imageHeight,
                    imageOriginW,
                    imageOriginH,
                    tileSize,
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
dispatchCountContributingGaussiansSparse<torch::kCPU>(
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
    uint32_t imageWidth,
    uint32_t imageHeight,
    uint32_t imageOriginW,
    uint32_t imageOriginH,
    uint32_t tileSize) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "CPU implementation not available");
}

std::tuple<torch::Tensor, torch::Tensor>
countContributingGaussians(const torch::Tensor &means2d,
                           const torch::Tensor &conics,
                           const torch::Tensor &opacities,
                           const torch::Tensor &tileOffsets,
                           const torch::Tensor &tileGaussianIds,
                           uint32_t imageWidth,
                           uint32_t imageHeight,
                           uint32_t imageOriginW,
                           uint32_t imageOriginH,
                           uint32_t tileSize) {
    return FVDB_DISPATCH_KERNEL_DEVICE(means2d.device(), [&]() {
        return dispatchCountContributingGaussians<DeviceTag>(means2d,
                                                             conics,
                                                             opacities,
                                                             tileOffsets,
                                                             tileGaussianIds,
                                                             imageWidth,
                                                             imageHeight,
                                                             imageOriginW,
                                                             imageOriginH,
                                                             tileSize);
    });
}

std::tuple<fvdb::JaggedTensor, fvdb::JaggedTensor>
countContributingGaussiansSparse(const torch::Tensor &means2d,
                                 const torch::Tensor &conics,
                                 const torch::Tensor &opacities,
                                 const torch::Tensor &tileOffsets,
                                 const torch::Tensor &tileGaussianIds,
                                 const fvdb::JaggedTensor &pixelsToRender,
                                 const torch::Tensor &activeTiles,
                                 const torch::Tensor &tilePixelMask,
                                 const torch::Tensor &tilePixelCumsum,
                                 const torch::Tensor &pixelMap,
                                 uint32_t imageWidth,
                                 uint32_t imageHeight,
                                 uint32_t imageOriginW,
                                 uint32_t imageOriginH,
                                 uint32_t tileSize) {
    return FVDB_DISPATCH_KERNEL_DEVICE(means2d.device(), [&]() {
        return dispatchCountContributingGaussiansSparse<DeviceTag>(means2d,
                                                                   conics,
                                                                   opacities,
                                                                   tileOffsets,
                                                                   tileGaussianIds,
                                                                   pixelsToRender,
                                                                   activeTiles,
                                                                   tilePixelMask,
                                                                   tilePixelCumsum,
                                                                   pixelMap,
                                                                   imageWidth,
                                                                   imageHeight,
                                                                   imageOriginW,
                                                                   imageOriginH,
                                                                   tileSize);
    });
}

} // namespace fvdb::detail::ops
