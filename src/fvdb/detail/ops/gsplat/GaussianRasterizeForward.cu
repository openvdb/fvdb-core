// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/ops/gsplat/Gaussian2D.cuh>
#include <fvdb/detail/ops/gsplat/GaussianRasterize.cuh>
#include <fvdb/detail/ops/gsplat/GaussianRasterizeForward.h>
#include <fvdb/detail/ops/gsplat/GaussianUtils.cuh>
#include <fvdb/detail/ops/gsplat/GaussianVectorTypes.cuh>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/Nvtx.h>

#include <nanovdb/math/Math.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>

#include <cub/block/block_scan.cuh>
#include <cuda/std/tuple>

#include <cstdint>
#include <optional>

namespace fvdb::detail::ops {
namespace {

// comment test
// Structure to hold arguments and methods for the rasterize forward kernel
template <typename ScalarType, uint32_t NUM_CHANNELS, bool IS_PACKED> struct RasterizeForwardArgs {
    using CommonArgs = RasterizeCommonArgs<ScalarType, NUM_CHANNELS, IS_PACKED>;
    CommonArgs commonArgs;

    JaggedRAcc64<ScalarType, 2> mOutFeatures; // [[nPixels, NUM_CHANNELS]_0..._C]
    JaggedRAcc64<ScalarType, 2> mOutAlphas;   // [[nPixels, 1]_0..._C]
    JaggedRAcc64<int32_t, 1> mOutLastIds;     // [[nPixels]_0..._C]

    RasterizeForwardArgs(
        const torch::Tensor &means2d,         // [C, N, 2] or [nnz, 2]
        const torch::Tensor &conics,          // [C, N, 3] or [nnz, 3]
        const torch::Tensor &opacities,       // [C, N] or [nnz]
        const torch::Tensor &features,        // [C, N, NUM_CHANNELS] or [nnz, NUM_CHANNELS]
        const std::optional<torch::Tensor> &backgrounds, // [C, NUM_CHANNELS]
        const std::optional<torch::Tensor> &masks,       // [C, numTilesH, numTilesW]
        const uint32_t imageWidth,
        const uint32_t imageHeight,
        const uint32_t imageOriginW,
        const uint32_t imageOriginH,
        const uint32_t tileSize,
        const torch::Tensor &tileOffsets,     // [C, numTilesH, numTilesW]
        const torch::Tensor &tileGaussianIds, // [totalIntersections]
        // output JaggedTensors:
        // In Dense mode, first dimension X = C * imageHeight * imageWidth
        // In Sparse mode, first dimension X = C * nPixels_i (i from 0 to C-1)
        const fvdb::JaggedTensor &outFeatures,                          // [X, NUM_CHANNELS]
        const fvdb::JaggedTensor &outAlphas,                            // [X, 1]
        const fvdb::JaggedTensor &outLastIds,                           // [X]
        const std::optional<torch::Tensor> &activeTiles = std::nullopt, // [AT]
        const std::optional<torch::Tensor> &tilePixelMask =
            std::nullopt, // [AT, wordsPerTileBitmask] e.g. [AT, 4]
        const std::optional<torch::Tensor> &tilePixelCumsum = std::nullopt, // [AT]
        const std::optional<torch::Tensor> &pixelMap        = std::nullopt)        // [AP]
        : commonArgs(means2d,
                     conics,
                     opacities,
                     features,
                     backgrounds,
                     masks,
                     imageWidth,
                     imageHeight,
                     imageOriginW,
                     imageOriginH,
                     tileSize,
                     tileOffsets,
                     tileGaussianIds,
                     activeTiles,
                     tilePixelMask,
                     tilePixelCumsum,
                     pixelMap),
          mOutFeatures(initJaggedAccessor<ScalarType, 2>(outFeatures, "outFeatures")),
          mOutAlphas(initJaggedAccessor<ScalarType, 2>(outAlphas, "outAlphas")),
          mOutLastIds(initJaggedAccessor<int32_t, 1>(outLastIds, "outLastIds")) {}

    /// @brief Write the alpha value for a pixel
    /// @param pixelIndex The index of the pixel
    /// @param alpha The alpha value to write
    __device__ void
    writeAlpha(uint64_t pixelIndex, ScalarType alpha) {
        mOutAlphas.data()[pixelIndex][0] = alpha;
    }

    /// @brief Write the last ID for a pixel
    /// @param pixelIndex The index of the pixel
    /// @param lastId The last ID to write
    __device__ void
    writeLastId(uint64_t pixelIndex, int32_t lastId) {
        mOutLastIds.data()[pixelIndex] = lastId;
    }

    /// @brief Write the features for a pixel
    /// @param pixelIndex The index of the pixel
    /// @param f The function to write the features
    template <typename F>
    __device__ void
    writeFeatures(uint64_t pixelIndex, F &&f) {
        PRAGMA_UNROLL
        for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
            mOutFeatures.data()[pixelIndex][k] = f(k);
        }
    }

    /// @brief Volume render a tile of Gaussians
    /// @param cameraId The ID of the camera
    /// @param firstGaussianIdInBlock The first Gaussian ID in the block
    /// @param lastGaussianIdInBlock The last Gaussian ID in the block
    /// @param blockSize The size of the block
    /// @param pixelIsActive Whether the pixel is active
    /// @param activePixelIndex The index of the active pixel
    /// @param row The row of the pixel
    /// @param col The column of the pixel
    __device__ void
    volumeRenderTileForward(const uint32_t cameraId,
                            const uint32_t row,
                            const uint32_t col,
                            const uint32_t firstGaussianIdInBlock,
                            const uint32_t lastGaussianIdInBlock,
                            const uint32_t blockSize,
                            const bool pixelIsActive,
                            const uint32_t activePixelIndex) {
        extern __shared__ int s[];
        auto *sharedGaussians = reinterpret_cast<Gaussian2D<ScalarType> *>(s); // [blockSize]

        // NOTE: The accumulated transmittance is used in the backward pass, and
        // since it's a
        //       sum of many small numbers, we should really use double precision.
        //       However, this makes the backward pass 1.5x slower, so we stick with
        //       float for now and sort of just ignore small impact gaussians
        //       ¯\_(ツ)_/¯.
        ScalarType accumTransmittance = 1.0f;
        // index of most recent gaussian to write to this thread's pixel
        int32_t curIdx = -1;

        // We don't return right away if the pixel is not in the image since we want
        // to use this thread to load gaussians into shared memory
        bool done = !pixelIsActive;

        // Process Gaussians in batches of block size (i.e. one Gaussian per thread in the block)
        const uint32_t tidx = threadIdx.y * blockDim.x + threadIdx.x;
        const uint32_t numBatches =
            (lastGaussianIdInBlock - firstGaussianIdInBlock + blockSize - 1) / blockSize;

        // (row, col) coordinates are relative to the specified image origin which may
        // be a crop so we need to add the origin to get the absolute pixel coordinates
        const ScalarType px = col + commonArgs.mImageOriginW + ScalarType{0.5f};
        const ScalarType py = row + commonArgs.mImageOriginH + ScalarType{0.5f};

        // collect and process batches of gaussians
        // each thread loads one gaussian at a time before rasterizing its
        // designated pixel
        ScalarType pixOut[NUM_CHANNELS] = {0.f};
        for (uint32_t b = 0; b < numBatches; ++b) {
            // Sync threads before we start integrating the next batch
            // If all threads are done, we can break early
            if (__syncthreads_count(done) == blockSize) {
                break;
            }

            // Each thread fetches one gaussian from front to back (mTileGaussianIds is depth
            // sorted)
            const uint32_t batchStart = firstGaussianIdInBlock + blockSize * b;
            const uint32_t idx        = batchStart + tidx;
            if (idx < lastGaussianIdInBlock) {
                const int32_t g =
                    commonArgs.mTileGaussianIds[idx]; // which gaussian we're rendering
                sharedGaussians[tidx] = commonArgs.getGaussian(g);
            }

            // Sync threads so all gaussians for this batch are loaded in shared
            // memory
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

                    const ScalarType vis       = alpha * accumTransmittance;
                    const auto featureAccessor = [&]() {
                        if constexpr (IS_PACKED) {
                            return commonArgs.mFeatures[gaussian.id];
                        } else {
                            const int32_t cid = gaussian.id / commonArgs.mNumGaussiansPerCamera;
                            const int32_t gid = gaussian.id % commonArgs.mNumGaussiansPerCamera;
                            return commonArgs.mFeatures[cid][gid];
                        }
                    }();
                    PRAGMA_UNROLL
                    for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
                        pixOut[k] += featureAccessor[k] * vis;
                    }

                    curIdx             = batchStart + t;
                    accumTransmittance = nextTransmittance;
                }
            }
        }

        if (pixelIsActive) {
            const auto pixIdx = commonArgs.pixelIndex(cameraId, row, col, activePixelIndex);
            writeAlpha(pixIdx, 1.0f - accumTransmittance);
            writeLastId(pixIdx, curIdx);
            writeFeatures(pixIdx, [&](uint32_t k) {
                return commonArgs.mHasBackgrounds
                           ? pixOut[k] + accumTransmittance * commonArgs.mBackgrounds[cameraId][k]
                           : pixOut[k];
            });
        }
    }
};

/// @brief Rasterize Gaussians to pixels
/// @param args The arguments for the rasterization
template <typename ScalarType, uint32_t NUM_CHANNELS, bool IS_PACKED>
__global__ void
rasterizeGaussiansForward(RasterizeForwardArgs<ScalarType, NUM_CHANNELS, IS_PACKED> args) {
    auto &commonArgs = args.commonArgs;

    int32_t cameraId;
    int32_t tileRow;
    int32_t tileCol;
    uint32_t row;
    uint32_t col;

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
        args.writeFeatures(pixIdx, [&](uint32_t k) {
            return commonArgs.mHasBackgrounds ? commonArgs.mBackgrounds[cameraId][k] : 0.0f;
        });
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

/// @brief Get the shared memory requirements for the forward pass kernel
/// @param tileSize The size of the tile
/// @return The shared memory required in bytes
template <typename ScalarType>
size_t
getSharedMemRequirements(const size_t tileSize) {
    return tileSize * tileSize * sizeof(Gaussian2D<ScalarType>);
}

template <typename ScalarType, uint32_t NUM_CHANNELS, bool IS_PACKED>
std::tuple<fvdb::JaggedTensor, fvdb::JaggedTensor, fvdb::JaggedTensor>
launchRasterizeForwardKernel(
    // Gaussian parameters
    const torch::Tensor &means2d,                   // [C, N, 2] or [nnz, 2]
    const torch::Tensor &conics,                    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &features,                  // [C, N, channels] or [nnz, channels]
    const torch::Tensor &opacities,                 // [C, N]  or [nnz]
    const at::optional<torch::Tensor> &backgrounds, // [C, channels]
    const at::optional<torch::Tensor> &masks,       // [C, tile_height, tile_width]
    // image size
    const uint32_t imageWidth,
    const uint32_t imageHeight,
    const uint32_t imageOriginW,
    const uint32_t imageOriginH,
    const uint32_t tileSize,
    // intersections
    const torch::Tensor &tileOffsets,     // [C, tile_height, tile_width]
    const torch::Tensor &tileGaussianIds, // [n_isects]
    const std::optional<fvdb::JaggedTensor> &pixelsToRender = std::nullopt,
    const std::optional<torch::Tensor> &activeTiles         = std::nullopt,
    const std::optional<torch::Tensor> &tilePixelMask       = std::nullopt,
    const std::optional<torch::Tensor> &tilePixelCumsum     = std::nullopt,
    const std::optional<torch::Tensor> &pixelMap            = std::nullopt) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(means2d));

    TORCH_CHECK_VALUE(tileOffsets.size(2) == (imageWidth + tileSize - 1) / tileSize,
                      "tileOffsets width must match the number of tiles in image size");
    TORCH_CHECK_VALUE(tileOffsets.size(1) == (imageHeight + tileSize - 1) / tileSize,
                      "tileOffsets height must match the number of tiles in image size");

    const bool packed = means2d.dim() == 2;

    const uint32_t C        = tileOffsets.size(0);          // number of cameras
    const uint32_t N        = packed ? 0 : means2d.size(1); // number of gaussians
    const uint32_t channels = features.size(-1);

    const uint32_t tileExtentH = tileOffsets.size(1);
    const uint32_t tileExtentW = tileOffsets.size(2);

    TORCH_CHECK_VALUE(pixelMap.has_value() == pixelsToRender.has_value(),
                      "pixelMap and pixelsToRender must be provided together");
    if (pixelMap.has_value()) {
        TORCH_CHECK_VALUE(pixelMap.value().size(0) == pixelsToRender.value().numel() / 2,
                          "pixelMap must have the same number of elements as pixelsToRender");
    }

    const auto sizes = pixelsToRender.has_value()
                           ? pixelsToRender.value().lsizes1()
                           : std::vector<int64_t>{C * imageHeight * imageWidth};
    std::vector<torch::Tensor> featuresToRenderVec;
    std::vector<torch::Tensor> alphasToRenderVec;
    std::vector<torch::Tensor> lastIdsToRenderVec;

    for (const auto &size: sizes) {
        featuresToRenderVec.push_back(
            torch::empty({size, channels}, features.options().dtype(torch::kFloat32)));
        alphasToRenderVec.push_back(
            torch::empty({size, 1}, features.options().dtype(torch::kFloat32)));
        lastIdsToRenderVec.push_back(torch::empty({size}, features.options().dtype(torch::kInt32)));
    }

    auto outFeatures = fvdb::JaggedTensor(featuresToRenderVec);
    auto outAlphas   = fvdb::JaggedTensor(alphasToRenderVec);
    auto outLastIds  = fvdb::JaggedTensor(lastIdsToRenderVec);

    auto args = RasterizeForwardArgs<ScalarType, NUM_CHANNELS, IS_PACKED>(means2d,
                                                                          conics,
                                                                          opacities,
                                                                          features,
                                                                          backgrounds,
                                                                          masks,
                                                                          imageWidth,
                                                                          imageHeight,
                                                                          imageOriginW,
                                                                          imageOriginH,
                                                                          tileSize,
                                                                          tileOffsets,
                                                                          tileGaussianIds,
                                                                          outFeatures,
                                                                          outAlphas,
                                                                          outLastIds,
                                                                          activeTiles,
                                                                          tilePixelMask,
                                                                          tilePixelCumsum,
                                                                          pixelMap);

    const at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    // Thread blocks cooperatively cache a tile of Gaussians in shared memory
    const uint32_t sharedMem = getSharedMemRequirements<ScalarType>(tileSize);

    // TODO: an optimization can be done by passing the actual number of
    // channels into the kernel functions and avoid necessary global memory
    // writes. This requires moving the channel padding from python to C side.
    if (cudaFuncSetAttribute(rasterizeGaussiansForward<ScalarType, NUM_CHANNELS, IS_PACKED>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             sharedMem) != cudaSuccess) {
        AT_ERROR("Failed to set maximum shared memory size (requested ",
                 sharedMem,
                 " bytes), try lowering tile_size.");
    }

    const dim3 blockDim = {tileSize, tileSize, 1};
    const dim3 gridDim  = activeTiles.has_value() // sparse mode
                              ? dim3(activeTiles.value().size(0), 1, 1)
                              : dim3(C, tileExtentH, tileExtentW);

    rasterizeGaussiansForward<<<gridDim, blockDim, sharedMem, stream>>>(args);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // In dense mode, we need to reshape the output tensors to the original image size
    // because they are packed into a single JaggedTensor so that the output code is the same
    // for dense and sparse modes.
    if (!args.commonArgs.mIsSparse) {
        outFeatures =
            fvdb::JaggedTensor(outFeatures.jdata().view({C, imageHeight, imageWidth, channels}));
        outAlphas  = fvdb::JaggedTensor(outAlphas.jdata().view({C, imageHeight, imageWidth, 1}));
        outLastIds = fvdb::JaggedTensor(outLastIds.jdata().view({C, imageHeight, imageWidth}));
    }

    return std::make_tuple(outFeatures, outAlphas, outLastIds);
}

} // namespace

template <>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianRasterizeForward<torch::kCUDA>(
    // Gaussian parameters
    const torch::Tensor &means2d,   // [C, N, 2]
    const torch::Tensor &conics,    // [C, N, 3]
    const torch::Tensor &features,  // [C, N, D]
    const torch::Tensor &opacities, // [N]
    const uint32_t imageWidth,
    const uint32_t imageHeight,
    const uint32_t imageOriginW,
    const uint32_t imageOriginH,
    const uint32_t tileSize,
    const torch::Tensor &tileOffsets,    // [C, tile_height, tile_width]
    const torch::Tensor &tileGaussianIds // [n_isects]
) {
    FVDB_FUNC_RANGE();
    const uint32_t channels = features.size(-1);
    const bool isPacked     = means2d.dim() == 2;

    const std::optional<torch::Tensor> backgrounds = std::nullopt;
    const std::optional<torch::Tensor> masks       = std::nullopt;

#define __CALL_FWD_(N)                                                                          \
    case N: {                                                                                   \
        if (isPacked) {                                                                         \
            auto [outFeatures, outAlphas, outLastIds] =                                         \
                launchRasterizeForwardKernel<float, N, true>(means2d,                           \
                                                             conics,                            \
                                                             features,                          \
                                                             opacities,                         \
                                                             backgrounds,                       \
                                                             masks,                             \
                                                             imageWidth,                        \
                                                             imageHeight,                       \
                                                             imageOriginW,                      \
                                                             imageOriginH,                      \
                                                             tileSize,                          \
                                                             tileOffsets,                       \
                                                             tileGaussianIds);                  \
            return std::make_tuple(outFeatures.jdata(), outAlphas.jdata(), outLastIds.jdata()); \
        } else {                                                                                \
            auto [outFeatures, outAlphas, outLastIds] =                                         \
                launchRasterizeForwardKernel<float, N, false>(means2d,                          \
                                                              conics,                           \
                                                              features,                         \
                                                              opacities,                        \
                                                              backgrounds,                      \
                                                              masks,                            \
                                                              imageWidth,                       \
                                                              imageHeight,                      \
                                                              imageOriginW,                     \
                                                              imageOriginH,                     \
                                                              tileSize,                         \
                                                              tileOffsets,                      \
                                                              tileGaussianIds);                 \
            return std::make_tuple(outFeatures.jdata(), outAlphas.jdata(), outLastIds.jdata()); \
        }                                                                                       \
    }

    // Make channels a compile time constant and do everything in register space
    // but at the expense of making this code ugly. NOTE: We do powers of two and
    // powers of two plus one to handle rendering common feature channel
    // dimensions with an optional additional depth channel
    switch (channels) {
        __CALL_FWD_(1)
        __CALL_FWD_(2)
        __CALL_FWD_(3)
        __CALL_FWD_(4)
        __CALL_FWD_(5)
        __CALL_FWD_(8)
        __CALL_FWD_(9)
        __CALL_FWD_(16)
        __CALL_FWD_(17)
        __CALL_FWD_(32)
        __CALL_FWD_(33)
        __CALL_FWD_(64)
        __CALL_FWD_(65)
        __CALL_FWD_(128)
        __CALL_FWD_(129)
        __CALL_FWD_(192)
        __CALL_FWD_(193)
        __CALL_FWD_(256)
        __CALL_FWD_(257)
        __CALL_FWD_(512)
        __CALL_FWD_(513)
    default: AT_ERROR("Unsupported number of channels: ", channels);
    }
}

template <>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianRasterizeForward<torch::kCPU>(
    // Gaussian parameters
    const torch::Tensor &means2d,   // [C, N, 2]
    const torch::Tensor &conics,    // [C, N, 3]
    const torch::Tensor &features,  // [C, N, D]
    const torch::Tensor &opacities, // [N]
    // image size
    const uint32_t imageWidth,
    const uint32_t imageHeight,
    const uint32_t imageOriginW,
    const uint32_t imageOriginH,
    const uint32_t tileSize,
    // intersections
    const torch::Tensor &tileOffsets,    // [C, tile_height, tile_width]
    const torch::Tensor &tileGaussianIds // [n_isects]
) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "CPU implementation not available");
}

template <>
std::tuple<fvdb::JaggedTensor, fvdb::JaggedTensor, fvdb::JaggedTensor>
dispatchGaussianSparseRasterizeForward<torch::kCUDA>(
    // sparse pixel coordinates
    const fvdb::JaggedTensor &pixelsToRender, // [C, maxPixelsPerCamera, 2]
    // Gaussian parameters
    const torch::Tensor &means2d,   // [C, N, 2]
    const torch::Tensor &conics,    // [C, N, 3]
    const torch::Tensor &features,  // [C, N, D]
    const torch::Tensor &opacities, // [N]
    const uint32_t imageWidth,
    const uint32_t imageHeight,
    const uint32_t imageOriginW,
    const uint32_t imageOriginH,
    const uint32_t tileSize,
    const torch::Tensor &tileOffsets,     // [C, tile_height, tile_width]
    const torch::Tensor &tileGaussianIds, // [n_isects]
    const torch::Tensor &activeTiles,
    const torch::Tensor &tilePixelMask,
    const torch::Tensor &tilePixelCumsum,
    const torch::Tensor &pixelMap) {
    FVDB_FUNC_RANGE();
    const uint32_t channels = features.size(-1);
    const bool isPacked     = means2d.dim() == 2;

    const std::optional<torch::Tensor> backgrounds = std::nullopt;
    const std::optional<torch::Tensor> masks       = std::nullopt;

#define __CALL_FWD_SPARSE_(N)                                                     \
    case N: {                                                                     \
        if (isPacked) {                                                           \
            return launchRasterizeForwardKernel<float, N, true>(means2d,          \
                                                                conics,           \
                                                                features,         \
                                                                opacities,        \
                                                                backgrounds,      \
                                                                masks,            \
                                                                imageWidth,       \
                                                                imageHeight,      \
                                                                imageOriginW,     \
                                                                imageOriginH,     \
                                                                tileSize,         \
                                                                tileOffsets,      \
                                                                tileGaussianIds,  \
                                                                pixelsToRender,   \
                                                                activeTiles,      \
                                                                tilePixelMask,    \
                                                                tilePixelCumsum,  \
                                                                pixelMap);        \
        } else {                                                                  \
            return launchRasterizeForwardKernel<float, N, false>(means2d,         \
                                                                 conics,          \
                                                                 features,        \
                                                                 opacities,       \
                                                                 backgrounds,     \
                                                                 masks,           \
                                                                 imageWidth,      \
                                                                 imageHeight,     \
                                                                 imageOriginW,    \
                                                                 imageOriginH,    \
                                                                 tileSize,        \
                                                                 tileOffsets,     \
                                                                 tileGaussianIds, \
                                                                 pixelsToRender,  \
                                                                 activeTiles,     \
                                                                 tilePixelMask,   \
                                                                 tilePixelCumsum, \
                                                                 pixelMap);       \
        }                                                                         \
    }

    // Make channels a compile time constant and do everything in register space
    // but at the expense of making this code ugly. NOTE: We do powers of two and
    // powers of two plus one to handle rendering common feature channel
    // dimensions with an optional additional depth channel
    switch (channels) {
        __CALL_FWD_SPARSE_(1)
        __CALL_FWD_SPARSE_(2)
        __CALL_FWD_SPARSE_(3)
        __CALL_FWD_SPARSE_(4)
        __CALL_FWD_SPARSE_(5)
        __CALL_FWD_SPARSE_(8)
        __CALL_FWD_SPARSE_(9)
        __CALL_FWD_SPARSE_(16)
        __CALL_FWD_SPARSE_(17)
        __CALL_FWD_SPARSE_(32)
        __CALL_FWD_SPARSE_(33)
        __CALL_FWD_SPARSE_(64)
        __CALL_FWD_SPARSE_(65)
        __CALL_FWD_SPARSE_(128)
        __CALL_FWD_SPARSE_(129)
        __CALL_FWD_SPARSE_(192)
        __CALL_FWD_SPARSE_(193)
        __CALL_FWD_SPARSE_(256)
        __CALL_FWD_SPARSE_(257)
        __CALL_FWD_SPARSE_(512)
        __CALL_FWD_SPARSE_(513)
    default: AT_ERROR("Unsupported number of channels: ", channels);
    }
}

template <>
std::tuple<fvdb::JaggedTensor, fvdb::JaggedTensor, fvdb::JaggedTensor>
dispatchGaussianSparseRasterizeForward<torch::kCPU>(
    // sparse pixel coordinates
    const fvdb::JaggedTensor &pixelsToRender, // [C, maxPixelsPerCamera, 2]
    // Gaussian parameters
    const torch::Tensor &means2d,   // [C, N, 2]
    const torch::Tensor &conics,    // [C, N, 3]
    const torch::Tensor &features,  // [C, N, D]
    const torch::Tensor &opacities, // [N]
    const uint32_t imageWidth,
    const uint32_t imageHeight,
    const uint32_t imageOriginW,
    const uint32_t imageOriginH,
    const uint32_t tileSize,
    const torch::Tensor &tileOffsets,     // [C, tile_height, tile_width]
    const torch::Tensor &tileGaussianIds, // [n_isects]
    const torch::Tensor &activeTiles,
    const torch::Tensor &tilePixelMask,
    const torch::Tensor &tilePixelCumsum,
    const torch::Tensor &pixelMap) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "CPU implementation not available");
}

} // namespace fvdb::detail::ops
