// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/Ops.h>
#include <fvdb/detail/ops/gsplat/Gaussian2D.cuh>
#include <fvdb/detail/ops/gsplat/GaussianVectorTypes.cuh>
#include <fvdb/detail/utils/AccessorHelpers.cuh>

#include <optional>

#define CHECK_CUDA(x)       TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)
#define PRAGMA_UNROLL _Pragma("unroll")

namespace fvdb::detail::ops {
namespace {

/**
 * @brief Sorts three arrays in-place based on depth indices using bubble sort.
 *
 * This device function performs a bubble sort on three parallel arrays, using the depth indices
 * as the sorting key. The function maintains the correspondence between the three arrays
 * during sorting. This is optimized for small arrays (typically 4-24 elements) where the
 * simplicity and cache locality of bubble sort can be beneficial.
 *
 * @param depthIndices Array of depth indices to sort by
 * @param radianceWeights Array of radiance weights corresponding to each depth index
 * @param maxRadianceWeightIndices Array of indices for maximum radiance weights
 * @param numSamples Number of elements in each array to sort
 *
 * @note The function modifies all three arrays in-place. The sorting is stable and
 * maintains the relationship between corresponding elements in all three arrays.
 *
 * @example
 * uint32_t depths[] = {3, 1, 2};
 * float weights[] = {0.3, 0.1, 0.2};
 * int32_t indices[] = {0, 1, 2};
 * bubbleSortByDepth(depths, weights, indices, 3);
 * // Result:
 * // depths:    {1, 2, 3}
 * // weights:   {0.1, 0.2, 0.3}
 * // indices:   {1, 2, 0}
 */
__device__ void
bubbleSortByDepth(uint32_t *depthIndices,
                  float *radianceWeights,
                  int32_t *maxRadianceWeightIndices,
                  const uint32_t numSamples) {
    for (uint32_t i = 0; i < numSamples - 1; ++i) {
        for (uint32_t j = 0; j < numSamples - 1 - i; ++j) {
            if (depthIndices[j] > depthIndices[j + 1]) {
                // Swap all three arrays to maintain correspondence
                uint32_t tempDepth  = depthIndices[j];
                depthIndices[j]     = depthIndices[j + 1];
                depthIndices[j + 1] = tempDepth;

                float tempWeight       = radianceWeights[j];
                radianceWeights[j]     = radianceWeights[j + 1];
                radianceWeights[j + 1] = tempWeight;

                int32_t tempId                  = maxRadianceWeightIndices[j];
                maxRadianceWeightIndices[j]     = maxRadianceWeightIndices[j + 1];
                maxRadianceWeightIndices[j + 1] = tempId;
            }
        }
    }
}

template <bool IS_PACKED> struct DeviceArgs {
    using vec2t = nanovdb::math::Vec2<float>;
    using vec3t = nanovdb::math::Vec3<float>;

    uint32_t mNumCameras;
    uint32_t mTotalIntersections;
    uint32_t mImageWidth;
    uint32_t mImageHeight;
    uint32_t mImageOriginW;
    uint32_t mImageOriginH;
    uint32_t mTileOriginW;
    uint32_t mTileOriginH;
    uint32_t mTileSize;
    uint32_t mNumTilesW;
    uint32_t mNumTilesH;
    uint32_t mNumDepthSamples;
    vec2t *__restrict__ mMeans2d;              // [C, N, 2] or [nnz, 2]
    vec3t *__restrict__ mConics;               // [C, N, 3] or [nnz, 3]
    float *__restrict__ mOpacities;            // [C, N] or [nnz]
    int32_t *__restrict__ mBackgrounds;        // [C, 1] or [nnz, 1]
    bool *__restrict__ mMasks;                 // [C, nTilesH, nTilesW]
    int32_t *__restrict__ mTileOffsets;        // [C, nTilesH, nTilesW]
    int32_t *__restrict__ mTileGaussianIds;    // [totalIntersections]
    int32_t *__restrict__ mOutIds;             // [C, imgH, imgW, numDepthSamples, 1]
    float *__restrict__ mOutWeights;           // [C, imgH, imgW, 1]

    DeviceArgs(const torch::Tensor &means2d,   // [C, N, 2] or [nnz, 2]
               const torch::Tensor &conics,    // [C, N, 3] or [nnz, 3]
               const torch::Tensor &opacities, // [C, N] or [nnz]
               const at::optional<torch::Tensor> &backgrounds, // [C, 1]
               const at::optional<torch::Tensor> &masks,       // [C, numTilesH, numTilesW]
               const uint32_t imageWidth,
               const uint32_t imageHeight,
               const uint32_t imageOriginW,
               const uint32_t imageOriginH,
               const uint32_t tileSize,
               const uint32_t numDepthSamples,
               const torch::Tensor &tileOffsets,    // [C, numTilesH, numTilesW]
               const torch::Tensor &tileGaussianIds // [totalIntersections]
    ) {
        checkInputTensor(means2d, "means2d");
        checkInputTensor(conics, "conics");
        checkInputTensor(opacities, "opacities");
        checkInputTensor(tileOffsets, "tileOffsets");
        checkInputTensor(tileGaussianIds, "tileGaussianIds");
        if (backgrounds.has_value()) {
            checkInputTensor(backgrounds.value(), "backgrounds");
        }
        if (masks.has_value()) {
            checkInputTensor(masks.value(), "masks");
        }

        const int64_t numCameras            = tileOffsets.size(0);
        const int64_t numGaussiansPerCamera = IS_PACKED ? 0 : means2d.size(1);
        const int64_t totalGaussians        = IS_PACKED ? means2d.size(0) : 0;
        const int64_t numTilesW             = tileOffsets.size(2);
        const int64_t numTilesH             = tileOffsets.size(1);

        if constexpr (IS_PACKED) {
            TORCH_CHECK_VALUE(means2d.dim() == 2, "Bad number of dims for means2d");
            TORCH_CHECK_VALUE(totalGaussians == means2d.size(0), "Bad size for means2d");
            TORCH_CHECK_VALUE(2 == means2d.size(1), "Bad size for means2d");

            TORCH_CHECK_VALUE(conics.dim() == 2, "Bad number of dims for conics");
            TORCH_CHECK_VALUE(totalGaussians == conics.size(0), "Bad size for conics");
            TORCH_CHECK_VALUE(3 == conics.size(1), "Bad size for conics");

            TORCH_CHECK_VALUE(opacities.dim() == 1, "Bad number of dims for opacities");
            TORCH_CHECK_VALUE(totalGaussians == opacities.size(0), "Bad size for opacities");
        } else {
            TORCH_CHECK_VALUE(means2d.dim() == 3, "Bad number of dims for means2d");
            TORCH_CHECK_VALUE(numCameras == means2d.size(0), "Bad size for means2d");
            TORCH_CHECK_VALUE(numGaussiansPerCamera == means2d.size(1), "Bad size for means2d");
            TORCH_CHECK_VALUE(2 == means2d.size(2), "Bad size for means2d");

            TORCH_CHECK_VALUE(conics.dim() == 3, "Bad number of dims for conics");
            TORCH_CHECK_VALUE(numCameras == conics.size(0), "Bad size for conics");
            TORCH_CHECK_VALUE(numGaussiansPerCamera == conics.size(1), "Bad size for conics");
            TORCH_CHECK_VALUE(3 == conics.size(2), "Bad size for conics");

            TORCH_CHECK_VALUE(opacities.dim() == 2, "Bad number of dims for opacities");
            TORCH_CHECK_VALUE(numCameras == opacities.size(0), "Bad size for opacities");
            TORCH_CHECK_VALUE(numGaussiansPerCamera == opacities.size(1), "Bad size for opacities");
        }

        if (backgrounds.has_value()) {
            TORCH_CHECK_VALUE(backgrounds.value().dim() == 2, "Bad number of dims for backgrounds");
            TORCH_CHECK_VALUE(numCameras == backgrounds.value().size(0),
                              "Bad size for backgrounds");
        }
        if (masks.has_value()) {
            TORCH_CHECK_VALUE(masks.value().dim() == 3, "Bad number of dims for masks");
            TORCH_CHECK_VALUE(numCameras == masks.value().size(0), "Bad size for masks");
            TORCH_CHECK_VALUE(numTilesH == masks.value().size(1), "Bad size for masks");
            TORCH_CHECK_VALUE(numTilesW == masks.value().size(2), "Bad size for masks");
        }

        TORCH_CHECK_VALUE(tileOffsets.dim() == 3, "Bad number of dims for tileOffsets");
        TORCH_CHECK_VALUE(numCameras == tileOffsets.size(0), "Bad size for tileOffsets");
        TORCH_CHECK_VALUE(numTilesH == tileOffsets.size(1), "Bad size for tileOffsets");
        TORCH_CHECK_VALUE(numTilesW == tileOffsets.size(2), "Bad size for tileOffsets");

        mNumCameras = tileOffsets.size(0);
        // mNumGaussiansPerCamera = IS_PACKED ? 0 : means2d.size(1);
        mTotalIntersections = tileGaussianIds.size(0);
        mImageWidth         = imageWidth;
        mImageHeight        = imageHeight;
        mImageOriginW       = imageOriginW;
        mImageOriginH       = imageOriginH;
        mTileOriginW        = imageOriginW / tileSize;
        mTileOriginH        = imageOriginH / tileSize;
        mTileSize           = tileSize;
        mNumTilesW          = tileOffsets.size(2);
        mNumTilesH          = tileOffsets.size(1);
        mNumDepthSamples    = numDepthSamples;

        mMeans2d     = reinterpret_cast<vec2t *>(means2d.data_ptr<float>());
        mConics      = reinterpret_cast<vec3t *>(conics.data_ptr<float>());
        mOpacities   = opacities.data_ptr<float>();
        mBackgrounds = backgrounds.has_value() ? backgrounds.value().data_ptr<int32_t>() : nullptr;
        mMasks       = masks.has_value() ? masks.value().data_ptr<bool>() : nullptr;
        mTileOffsets = tileOffsets.data_ptr<int32_t>();
        mTileGaussianIds = tileGaussianIds.data_ptr<int32_t>();
    }

    void
    setOutputArguments(const torch::Tensor &outIds, const torch::Tensor &outWeights) {
        mOutIds     = outIds.data_ptr<int32_t>();
        mOutWeights = outWeights.data_ptr<float>();
    }

    inline __device__ void
    advancePointersToCameraPixel(const uint32_t cameraId, const uint32_t i, const uint32_t j) {
        const int32_t pixId = i * mImageWidth + j;

        // Move all the pointers forward to the current camera and pixel
        const std::ptrdiff_t offsetForPixels = cameraId * mImageHeight * mImageWidth + pixId;
        const std::ptrdiff_t offsetForTiles  = cameraId * mNumTilesH * mNumTilesW;

        mTileOffsets += offsetForTiles;
        mOutIds += offsetForPixels * mNumDepthSamples;
        mOutWeights += offsetForPixels * mNumDepthSamples;
        if (mBackgrounds != nullptr) {
            mBackgrounds += cameraId;
        }
        if (mMasks != nullptr) {
            mMasks += offsetForTiles;
        }
    }

    inline static void
    checkInputTensor(const torch::Tensor &x, const std::string &name) {
        TORCH_CHECK(x.is_cuda(), "Input ", name, " must be a CUDA tensor");
        TORCH_CHECK(x.is_contiguous(), "Input ", name, " must be contiguous");
    }

    __device__ void
    volumeRenderTileForward(const uint32_t tileStart,
                            const uint32_t tileEnd,
                            const uint32_t blockSize,
                            const uint32_t tileSize,
                            const bool writePixel,
                            const uint32_t i,
                            const uint32_t j) {
        const uint32_t numBatches = (tileEnd - tileStart + blockSize - 1) / blockSize;

        // Ordinal of this thread in the block
        const uint32_t tidx = threadIdx.x * blockDim.y + threadIdx.y;

        // Shared memory for the gaussians processed in a batch in this block
        extern __shared__ int s[];
        Gaussian2D<float> *sharedGaussians =
            reinterpret_cast<Gaussian2D<float> *>(s); // [blockSize]

        // Shared memory for the indices and max radiance weights for all pixels in this block
        int32_t *batchMaxRadianceWeightIndices = reinterpret_cast<int32_t *>(
            sharedGaussians + blockSize); // [blockSize * mNumDepthSamples]

        float *batchRadianceWeights = reinterpret_cast<float *>(
            batchMaxRadianceWeightIndices +
            blockSize * mNumDepthSamples); // [blockSize *mNumDepthSamples]

        // Shared memory for tracking depth order indices
        uint32_t *batchDepthIndices = reinterpret_cast<uint32_t *>(
            batchRadianceWeights + blockSize * mNumDepthSamples); // [blockSize * mNumDepthSamples]

        // Shared memory for the indices and max radiance weights for this pixel in the block
        int32_t *maxRadianceWeightIndices = batchMaxRadianceWeightIndices + tidx * mNumDepthSamples;
        float *radianceWeights            = batchRadianceWeights + tidx * mNumDepthSamples;
        uint32_t *depthIndices            = batchDepthIndices + tidx * mNumDepthSamples;

        // Initialize the indices and max radiance weights for this pixel in the block
        for (uint32_t w = 0; w < mNumDepthSamples; ++w) {
            radianceWeights[w]          = 0.f;
            maxRadianceWeightIndices[w] = -1;
            depthIndices[w]             = UINT32_MAX; // Large value for uninitialized
        }

        const float px = float(j) + 0.5f;
        const float py = float(i) + 0.5f;

        float accumTransmittance = 1.0f;
        uint32_t depthCounter    = 0; // Counter to track processing order (depth order)

        // TODO: This condition seems to be met when batch size is >1.  Kept here for debugging.
        // if (numBatches <= 0) {
        //     printf("No batches for this pixel %d %d\n", i, j);
        //     printf("tileEnd: %d, tileStart: %d, blockSize: %d\n", tileEnd, tileStart, blockSize);
        // }

        // We don't return right away if the pixel is not in the image since we want to use
        // this thread to load gaussians into shared memory
        bool done = !writePixel;

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
            const uint32_t batchStart = tileStart + blockSize * b;
            const uint32_t idx        = batchStart + tidx;
            if (idx < tileEnd) {
                const int32_t g       = mTileGaussianIds[idx]; // which gaussian we're rendering
                sharedGaussians[tidx] = {g, mMeans2d[g], mOpacities[g], mConics[g]};
            }

            // Sync threads so all gaussians for this batch are loaded in shared memory
            __syncthreads();

            // Volume render Gaussians in this batch
            const uint32_t batchSize = min(blockSize, tileEnd - batchStart);
            for (uint32_t t = 0; t < batchSize; ++t) {
                const Gaussian2D<float> &gaussian = sharedGaussians[t];

                const float sigma = gaussian.sigma(px, py);
                const float alpha = min(0.999f, gaussian.opacity * __expf(-sigma));

                if (sigma < 0.f || alpha < 1.f / 255.f) {
                    continue;
                }

                const float radianceWeight = alpha * accumTransmittance;

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

                accumTransmittance *= (float(1.0) - alpha);

                if (accumTransmittance <= float(1e-4)) { // this pixel is done
                    done = true;
                    break;
                }
            }
        }

        if (writePixel) {
            // Sort the samples by depth order before outputting
            bubbleSortByDepth(
                depthIndices, radianceWeights, maxRadianceWeightIndices, mNumDepthSamples);

            for (uint32_t k = 0; k < mNumDepthSamples; ++k) {
                mOutWeights[k] = radianceWeights[k];
                mOutIds[k]     = maxRadianceWeightIndices[k];
            }
        }
    }
};

/****************************************************************************
 * Rasterization to Pixels Forward Pass
 ****************************************************************************/

template <bool IS_PACKED>
__global__ void
rasterizeTopKGaussiansForward(DeviceArgs<IS_PACKED> args) {
    // each thread draws one pixel, but also timeshares caching gaussians in a
    // shared tile

    const int32_t cameraId = blockIdx.x;

    // blockIdx.yz runs from [0, numTilesH] x [0, numTilesW]
    const int32_t tileId =
        (blockIdx.y + args.mTileOriginH) * args.mNumTilesW + (blockIdx.z + args.mTileOriginW);

    // Pixel coordinates run from [0, height] x [0, width]
    // i.e. they are in the local coordinates of the crop starting from pixel
    //      [image_origin_h, image_origin_w] with size [image_height, image_width]
    const uint32_t i = blockIdx.y * args.mTileSize + threadIdx.y;
    const uint32_t j = blockIdx.z * args.mTileSize + threadIdx.x;

    args.advancePointersToCameraPixel(cameraId, i, j);

    // return if out of bounds
    // keep non-rasterizing threads around for reading data
    const bool pixelInImage = (i < args.mImageHeight && j < args.mImageWidth);
    // when the mask is provided, render the background feature/color and return
    // if this tile is labeled as False
    if (args.mMasks != nullptr && pixelInImage && !args.mMasks[tileId]) {
        for (int32_t d = 0; d < args.mNumDepthSamples; ++d) {
            args.mOutIds[d] = args.mBackgrounds == nullptr ? 0 : args.mBackgrounds[0];
        }
        return;
    }

    // Figure out the first and (one past the) last Gaussian ID in this block/tile
    const int32_t firstGaussianIdInBlock = args.mTileOffsets[tileId];
    const int32_t lastGaussianIdInBlock =
        (cameraId == args.mNumCameras - 1) && (tileId == args.mNumTilesW * args.mNumTilesH - 1)
            ? args.mTotalIntersections
            : args.mTileOffsets[tileId + 1];
    const uint32_t blockSize = blockDim.x * blockDim.y;

    // Pixel coordinates in the global image (not just the local crop)
    const uint32_t globalI = i + args.mImageOriginH;
    const uint32_t globalJ = j + args.mImageOriginW;
    args.volumeRenderTileForward(firstGaussianIdInBlock,
                                 lastGaussianIdInBlock,
                                 blockSize,
                                 args.mTileSize,
                                 pixelInImage,
                                 globalI,
                                 globalJ);
}

template <bool IS_PACKED>
std::tuple<torch::Tensor, torch::Tensor>
launchRasterizeTopKForwardKernel(
    // Gaussian parameters
    const torch::Tensor &means2d,                   // [C, N, 2] or [nnz, 2]
    const torch::Tensor &conics,                    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &opacities,                 // [C, N]  or [nnz]
    const at::optional<torch::Tensor> &backgrounds, // [C, channels]
    const at::optional<torch::Tensor> &masks,       // [C, tile_height, tile_width]
    // intersections
    const torch::Tensor &tileOffsets,     // [C, tile_height, tile_width]
    const torch::Tensor &tileGaussianIds, // [n_isects]
    const RenderSettings &settings        // render settings
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(means2d));

    TORCH_CHECK_VALUE(means2d.dim() == 3 || means2d.dim() == 2,
                      "means2d must have 3 dimensions (C, N, 2) or 2 dimensions (nnz, 2)");
    TORCH_CHECK_VALUE(conics.dim() == 3 || conics.dim() == 2,
                      "conics must have 3 dimensions (C, N, 3) or 2 dimensions (nnz, 3)");

    TORCH_CHECK_VALUE(opacities.dim() == 2 || opacities.dim() == 1,
                      "opacities must have 2 dimensions (C, N) or 1 dimension (nnz)");
    if (backgrounds.has_value()) {
        TORCH_CHECK_VALUE(backgrounds.value().dim() == 2,
                          "backgrounds must have 2 dimensions (C, channels)");
    }
    if (masks.has_value()) {
        TORCH_CHECK_VALUE(masks.value().dim() == 3,
                          "masks must have 3 dimensions (C, tile_height, tile_width)");
    }
    TORCH_CHECK_VALUE(tileOffsets.dim() == 3,
                      "tile_offsets must have 3 dimensions (C, tile_height, tile_width)");
    TORCH_CHECK_VALUE(tileGaussianIds.dim() == 1,
                      "tile_gaussian_ids must have 1 dimension (n_isects)");

    TORCH_CHECK_VALUE(settings.numDepthSamples > 0, "numDepthSamples must be greater than 0");

    CHECK_INPUT(means2d);
    CHECK_INPUT(conics);
    CHECK_INPUT(opacities);
    CHECK_INPUT(tileOffsets);
    CHECK_INPUT(tileGaussianIds);
    if (backgrounds.has_value()) {
        CHECK_INPUT(backgrounds.value());
    }
    if (masks.has_value()) {
        CHECK_INPUT(masks.value());
    }

    const bool packed = means2d.dim() == 2;

    const uint32_t C          = tileOffsets.size(0);          // number of cameras
    const uint32_t N          = packed ? 0 : means2d.size(1); // number of gaussians
    const uint32_t tileHeight = tileOffsets.size(1);
    const uint32_t tileWidth  = tileOffsets.size(2);
    const uint32_t nIsects    = tileGaussianIds.size(0);

    const uint32_t tileExtentW = (settings.imageWidth + settings.tileSize - 1) / settings.tileSize;
    const uint32_t tileExtentH = (settings.imageHeight + settings.tileSize - 1) / settings.tileSize;

    // The rendered images to return (one per camera)
    torch::Tensor outIds =
        torch::empty({C, settings.imageHeight, settings.imageWidth, settings.numDepthSamples},
                     means2d.options().dtype(torch::kInt32));
    torch::Tensor weights =
        torch::empty({C, settings.imageHeight, settings.imageWidth, settings.numDepthSamples},
                     means2d.options().dtype(torch::kFloat32));

    const at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    // Each pixel in each tile will cache a gaussian consisting of:
    //   - int32_t  gaussian_id; -- 4 bytes
    //   - vec2t    xy;          -- 8 bytes for float32
    //   - scalar_t opacity;     -- 4 bytes for float32
    //   - vec3t    conic;       -- 12 bytes for float32
    const uint32_t sharedMem =
        settings.tileSize * settings.tileSize *
        (sizeof(Gaussian2D<float>) +
         (sizeof(int32_t) + sizeof(float) + sizeof(uint32_t)) * settings.numDepthSamples);

    if (cudaFuncSetAttribute(rasterizeTopKGaussiansForward<IS_PACKED>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             sharedMem) != cudaSuccess) {
        AT_ERROR("Failed to set maximum shared memory size (requested ",
                 sharedMem,
                 " bytes), try lowering tile_size.");
    }

    const dim3 blockDim = {settings.tileSize, settings.tileSize, 1};
    const dim3 gridDim  = {C, tileExtentH, tileExtentW};

    auto args = DeviceArgs<IS_PACKED>(means2d,
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
                                      tileGaussianIds);

    args.setOutputArguments(outIds, weights);

    rasterizeTopKGaussiansForward<<<gridDim, blockDim, sharedMem, stream>>>(args);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));

    return std::make_tuple(outIds, weights);
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
    const bool isPacked = means2d.dim() == 2;

    const std::optional<torch::Tensor> backgrounds = std::nullopt;
    const std::optional<torch::Tensor> masks       = std::nullopt;

    if (isPacked) {
        return launchRasterizeTopKForwardKernel<true>(
            means2d, conics, opacities, backgrounds, masks, tileOffsets, tileGaussianIds, settings);
    } else {
        return launchRasterizeTopKForwardKernel<false>(
            means2d, conics, opacities, backgrounds, masks, tileOffsets, tileGaussianIds, settings);
    }
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

} // namespace fvdb::detail::ops
