// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/ops/Ops.h>
#include <fvdb/detail/ops/gsplat/Gaussian2D.cuh>
#include <fvdb/detail/ops/gsplat/GaussianUtils.cuh>
#include <fvdb/detail/ops/gsplat/GaussianVectorTypes.cuh>
#include <fvdb/detail/utils/AccessorHelpers.cuh>

#include <cub/block/block_scan.cuh>
#include <cuda/std/tuple>

#include <cstdint>
#include <optional>

#define PRAGMA_UNROLL _Pragma("unroll")

namespace fvdb::detail::ops {
namespace {

// Structure to hold arguments and methods for the rasterize forward kernel
template <typename ScalarType, uint32_t NUM_CHANNELS, bool IS_PACKED> struct DeviceArgs {
    constexpr static std::size_t NUM_OUTER_DIMS    = IS_PACKED ? 1 : 2;
    using vec2t                                    = typename Vec2Type<ScalarType>::type;
    using vec3t                                    = typename Vec3Type<ScalarType>::type;
    template <typename T, int N> using TorchRAcc64 = fvdb::TorchRAcc64<T, N>;
    using ScalarAccessor                           = TorchRAcc64<ScalarType, NUM_OUTER_DIMS>;
    using VectorAccessor                           = TorchRAcc64<ScalarType, NUM_OUTER_DIMS + 1>;

    uint32_t mNumCameras;
    uint32_t mNumGaussiansPerCamera;
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

    // Default tensors used to initialize optional and output tensors in constructor
    torch::Tensor mDefaultBackground;
    torch::Tensor mDefaultMasks;
    torch::Tensor mDefaultOutDenseFeatures;
    torch::Tensor mDefaultOutDenseAlphas;
    torch::Tensor mDefaultOutDenseLastIds;
    torch::Tensor mDefaultOutSparseFeatures;
    torch::Tensor mDefaultOutSparseAlphas;
    torch::Tensor mDefaultOutSparseLastIds;
    torch::Tensor mDefaultActiveTiles;

    torch::Tensor mDefaultTilePixelMask;
    torch::Tensor mDefaultTilePixelCumsum;
    torch::Tensor mDefaultPixelMap;

    VectorAccessor mFeatures;                       // [C, N, NUM_CHANNELS] or [nnz, NUM_CHANNELS]
    VectorAccessor mMeans2d;                        // [C, N, 2] or [nnz, 2]
    VectorAccessor mConics;                         // [C, N, 3] or [nnz, 3]
    ScalarAccessor mOpacities;                      // [C, N] or [nnz]
    TorchRAcc64<ScalarType, 2> mBackgrounds;        // [C, NUM_CHANNELS]
    bool mHasBackgrounds;
    TorchRAcc64<bool, 3> mMasks;                    // [C, nTilesH, nTilesW]
    bool mHasMasks;
    TorchRAcc64<int32_t, 3> mTileOffsets;           // [C, nTilesH, nTilesW]
    TorchRAcc64<int32_t, 1> mTileGaussianIds;       // [totalIntersections]
    TorchRAcc64<ScalarType, 4> mOutDenseFeatures;   // [C, imgH, imgW, NUM_CHANNELS]
    TorchRAcc64<ScalarType, 4> mOutDenseAlphas;     // [C, imgH, imgW, 1]
    TorchRAcc64<int32_t, 3> mOutDenseLastIds;       // [C, imgH, imgW]
    JaggedRAcc32<ScalarType, 2> mOutSparseFeatures; // [[nPixels, NUM_CHANNELS]_0..._C]
    JaggedRAcc32<ScalarType, 2> mOutSparseAlphas;   // [[nPixels, 1]_0..._C]
    JaggedRAcc32<int32_t, 1> mOutSparseLastIds;     // [[nPixels]_0..._C]
    bool mIsSparse;
    TorchRAcc64<int32_t, 1> mActiveTiles;           // [AT]
    TorchRAcc64<uint64_t, 2> mTilePixelMask;        // [AT, wordsPerTile] e.g. [AT, 4]
    TorchRAcc64<int64_t, 1> mTilePixelCumsum;       // [AT]
    TorchRAcc64<int64_t, 1> mPixelMap;              // [AP]

    // Check that the input tensor is a CUDA tensor and is contiguous
    inline void
    checkInputTensor(const torch::Tensor &x, const std::string &name) {
        TORCH_CHECK(x.is_cuda(), "Input ", name, " must be a CUDA tensor");
        TORCH_CHECK(x.is_contiguous(), "Input ", name, " must be contiguous");
    }

    template <typename T, int N>
    inline auto
    initAccessor(const std::optional<torch::Tensor> &tensor,
                 const torch::Tensor &defaultTensor,
                 const std::string &name) {
        if (tensor.has_value()) {
            checkInputTensor(tensor.value(), name);
            return tensor.value().packed_accessor64<T, N, torch::RestrictPtrTraits>();
        } else {
            checkInputTensor(defaultTensor, name + " (default)");
            return defaultTensor.packed_accessor64<T, N, torch::RestrictPtrTraits>();
        }
    }

    template <typename T, int N>
    inline auto
    initAccessor(const torch::Tensor &tensor, const std::string &name) {
        checkInputTensor(tensor, name);
        return tensor.packed_accessor64<T, N, torch::RestrictPtrTraits>();
    }

    template <typename T, int N>
    inline auto
    initJaggedAccessor(const fvdb::JaggedTensor &tensor) {
        return tensor.packed_accessor32<T, N, torch::RestrictPtrTraits>();
    }

    DeviceArgs(const torch::Tensor &means2d,   // [C, N, 2] or [nnz, 2]
               const torch::Tensor &conics,    // [C, N, 3] or [nnz, 3]
               const torch::Tensor &features,  // [C, N, NUM_CHANNELS] or [nnz, NUM_CHANNELS]
               const torch::Tensor &opacities, // [C, N] or [nnz]
               const std::optional<torch::Tensor> &backgrounds, // [C, NUM_CHANNELS]
               const std::optional<torch::Tensor> &masks,       // [C, numTilesH, numTilesW]
               const uint32_t imageWidth,
               const uint32_t imageHeight,
               const uint32_t imageOriginW,
               const uint32_t imageOriginH,
               const uint32_t tileSize,
               const torch::Tensor &tileOffsets,     // [C, numTilesH, numTilesW]
               const torch::Tensor &tileGaussianIds, // [totalIntersections]
               const std::optional<fvdb::JaggedTensor> &pixelsToRender = std::nullopt,
               const std::optional<torch::Tensor> &activeTiles         = std::nullopt, // [AT]
               const std::optional<torch::Tensor> &tilePixelMask =
                   std::nullopt, // [AT, wordsPerTile] e.g. [AT, 4]
               const std::optional<torch::Tensor> &tilePixelCumsum = std::nullopt, // [AT]
               const std::optional<torch::Tensor> &pixelMap        = std::nullopt)        // [AP]
        : mImageWidth(imageWidth), mImageHeight(imageHeight), mImageOriginW(imageOriginW),
          mImageOriginH(imageOriginH), mTileOriginW(imageOriginW / tileSize),
          mTileOriginH(imageOriginH / tileSize), mTileSize(tileSize),
          mFeatures(initAccessor<ScalarType, NUM_OUTER_DIMS + 1>(features, "features")),
          mMeans2d(initAccessor<ScalarType, NUM_OUTER_DIMS + 1>(means2d, "means2d")),
          mConics(initAccessor<ScalarType, NUM_OUTER_DIMS + 1>(conics, "conics")),
          mOpacities(initAccessor<ScalarType, NUM_OUTER_DIMS>(opacities, "opacities")),
          mDefaultBackground(torch::empty({0, 0}, means2d.options())),
          mDefaultMasks(torch::empty({0, 0, 0}, means2d.options().dtype(torch::kBool))),
          mDefaultOutDenseFeatures(torch::empty({0, 0, 0, 0}, features.options())),
          mDefaultOutDenseAlphas(torch::empty({0, 0, 0, 0}, opacities.options())),
          mDefaultOutDenseLastIds(torch::empty({0, 0, 0}, tileOffsets.options())),
          mDefaultOutSparseFeatures(torch::empty({0, 0}, features.options())),
          mDefaultOutSparseAlphas(torch::empty({0, 0}, opacities.options())),
          mDefaultOutSparseLastIds(torch::empty({0}, tileOffsets.options())),
          mDefaultActiveTiles(torch::empty({0}, tileOffsets.options())),
          mDefaultTilePixelMask(torch::empty({0, 0}, means2d.options().dtype(torch::kUInt64))),
          mDefaultTilePixelCumsum(torch::empty({0}, tileOffsets.options().dtype(torch::kInt64))),
          mDefaultPixelMap(torch::empty({0}, tileOffsets.options().dtype(torch::kInt64))),
          mBackgrounds(initAccessor<ScalarType, 2>(backgrounds, mDefaultBackground, "backgrounds")),
          mHasBackgrounds(backgrounds.has_value()),
          mMasks(initAccessor<bool, 3>(masks, mDefaultMasks, "masks")),
          mHasMasks(masks.has_value()),
          mTileOffsets(initAccessor<int32_t, 3>(tileOffsets, "tileOffsets")),
          mTileGaussianIds(initAccessor<int32_t, 1>(tileGaussianIds, "tileGaussianIds")),
          mOutDenseFeatures(initAccessor<ScalarType, 4>(mDefaultOutDenseFeatures, "outFeatures")),
          mOutDenseAlphas(initAccessor<ScalarType, 4>(mDefaultOutDenseAlphas, "outAlphas")),
          mOutDenseLastIds(initAccessor<int32_t, 3>(mDefaultOutDenseLastIds, "outLastGaussianIds")),
          mOutSparseFeatures(initJaggedAccessor<ScalarType, 2>(mDefaultOutSparseFeatures)),
          mOutSparseAlphas(initJaggedAccessor<ScalarType, 2>(mDefaultOutSparseAlphas)),
          mOutSparseLastIds(initJaggedAccessor<int32_t, 1>(mDefaultOutSparseLastIds)),
          mIsSparse(activeTiles.has_value()),
          mActiveTiles(initAccessor<int32_t, 1>(activeTiles, mDefaultActiveTiles, "activeTiles")),
          mTilePixelMask(
              initAccessor<uint64_t, 2>(tilePixelMask, mDefaultTilePixelMask, "tilePixelMask")),
          mTilePixelCumsum(initAccessor<int64_t, 1>(
              tilePixelCumsum, mDefaultTilePixelCumsum, "tilePixelCumsum")),
          mPixelMap(initAccessor<int64_t, 1>(pixelMap, mDefaultPixelMap, "pixelMap")) {
        static_assert(NUM_OUTER_DIMS == 1 || NUM_OUTER_DIMS == 2, "NUM_OUTER_DIMS must be 1 or 2");
        mNumCameras            = mTileOffsets.size(0);
        mNumGaussiansPerCamera = IS_PACKED ? 0 : mMeans2d.size(1);
        mTotalIntersections    = mTileGaussianIds.size(0);
        mNumTilesW             = mTileOffsets.size(2);
        mNumTilesH             = mTileOffsets.size(1);

        checkInputShapes();

        // have to check this here because we don't store pixelsToRender in the DeviceArgs
        if (mIsSparse) {
            TORCH_CHECK_VALUE(pixelMap.value().size(0) == pixelsToRender.value().numel() / 2,
                              "Bad size for pixelMap");
        }
    }

    void
    checkInputShapes() {
        const int64_t totalGaussians = IS_PACKED ? mMeans2d.size(0) : 0;

        if constexpr (IS_PACKED) {
            TORCH_CHECK_VALUE(totalGaussians == mMeans2d.size(0), "Bad size for means2d");
            TORCH_CHECK_VALUE(2 == mMeans2d.size(1), "Bad size for means2d");
            TORCH_CHECK_VALUE(totalGaussians == mConics.size(0), "Bad size for conics");
            TORCH_CHECK_VALUE(3 == mConics.size(1), "Bad size for conics");
            TORCH_CHECK_VALUE(totalGaussians == mFeatures.size(0), "Bad size for features");
            TORCH_CHECK_VALUE(NUM_CHANNELS == mFeatures.size(1), "Bad size for features");

            TORCH_CHECK_VALUE(totalGaussians == mOpacities.size(0), "Bad size for opacities");
        } else {
            TORCH_CHECK_VALUE(mNumCameras == mMeans2d.size(0), "Bad size for means2d");
            TORCH_CHECK_VALUE(mNumGaussiansPerCamera == mMeans2d.size(1), "Bad size for means2d");
            TORCH_CHECK_VALUE(2 == mMeans2d.size(2), "Bad size for means2d");
            TORCH_CHECK_VALUE(mNumCameras == mConics.size(0), "Bad size for conics");
            TORCH_CHECK_VALUE(mNumGaussiansPerCamera == mConics.size(1), "Bad size for conics");
            TORCH_CHECK_VALUE(3 == mConics.size(2), "Bad size for conics");
            TORCH_CHECK_VALUE(mNumCameras == mFeatures.size(0), "Bad size for features");
            TORCH_CHECK_VALUE(mNumGaussiansPerCamera == mFeatures.size(1), "Bad size for features");
            TORCH_CHECK_VALUE(NUM_CHANNELS == mFeatures.size(2), "Bad size for features");
            TORCH_CHECK_VALUE(mNumCameras == mOpacities.size(0), "Bad size for opacities");
            TORCH_CHECK_VALUE(mNumGaussiansPerCamera == mOpacities.size(1),
                              "Bad size for opacities");
        }

        if (mHasBackgrounds) {
            TORCH_CHECK_VALUE(mNumCameras == mBackgrounds.size(0), "Bad size for backgrounds");
            TORCH_CHECK_VALUE(NUM_CHANNELS == mBackgrounds.size(1), "Bad size for backgrounds");
        }
        if (mHasMasks) {
            TORCH_CHECK_VALUE(mNumCameras == mMasks.size(0), "Bad size for masks");
            TORCH_CHECK_VALUE(mNumTilesH == mMasks.size(1), "Bad size for masks");
            TORCH_CHECK_VALUE(mNumTilesW == mMasks.size(2), "Bad size for masks");
        }

        TORCH_CHECK_VALUE(mNumCameras == mTileOffsets.size(0), "Bad size for tileOffsets");
        TORCH_CHECK_VALUE(mNumTilesH == mTileOffsets.size(1), "Bad size for tileOffsets");
        TORCH_CHECK_VALUE(mNumTilesW == mTileOffsets.size(2), "Bad size for tileOffsets");

        if (mIsSparse) {
            TORCH_CHECK_VALUE(mTilePixelMask.size(0) == mActiveTiles.size(0),
                              "Bad size for tilePixelMask");
            TORCH_CHECK_VALUE(mTilePixelMask.size(1) == numWordsPerTileBitmask(mTileSize),
                              "Bad size for tilePixelMask");
            TORCH_CHECK_VALUE(mTilePixelCumsum.size(0) == mActiveTiles.size(0),
                              "Bad size for tilePixelCumsum");
        }
    }

    // TODO: once we improve JaggedTensorAccessor, we can make the rasterization always operate
    // on JaggedTensors and remove the split in the type of the output arguments. Also switch to
    // 64-bit indexed JaggedAccessor.

    // Set the output arguments for the dense output
    void
    setDenseOutputArguments(const torch::Tensor &outFeatures,
                            const torch::Tensor &outAlphas,
                            const torch::Tensor &outLastGaussianIds) {
        mOutDenseFeatures =
            outFeatures.packed_accessor64<ScalarType, 4, torch::RestrictPtrTraits>();
        mOutDenseAlphas = outAlphas.packed_accessor64<ScalarType, 4, torch::RestrictPtrTraits>();
        mOutDenseLastIds =
            outLastGaussianIds.packed_accessor64<int32_t, 3, torch::RestrictPtrTraits>();
    }

    // Set the output arguments for the sparse output
    void
    setSparseOutputArguments(const fvdb::JaggedTensor &outFeatures,
                             const fvdb::JaggedTensor &outAlphas,
                             const fvdb::JaggedTensor &outLastGaussianIds) {
        mOutSparseFeatures =
            outFeatures.packed_accessor32<ScalarType, 2, torch::RestrictPtrTraits>();
        mOutSparseAlphas = outAlphas.packed_accessor32<ScalarType, 2, torch::RestrictPtrTraits>();
        mOutSparseLastIds =
            outLastGaussianIds.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>();
    }

    // Construct a Gaussian2D object from the input tensors at the given index
    inline __device__ Gaussian2D<ScalarType>
    getGaussian(const uint32_t index) {
        if constexpr (IS_PACKED) {
            return Gaussian2D<ScalarType>(
                index,
                vec2t(mMeans2d[index][0], mMeans2d[index][1]),
                mOpacities[index],
                vec3t(mConics[index][0], mConics[index][1], mConics[index][2]));
        } else {
            auto cid = index / mNumGaussiansPerCamera;
            auto gid = index % mNumGaussiansPerCamera;
            return Gaussian2D<ScalarType>(
                index,
                vec2t(mMeans2d[cid][gid][0], mMeans2d[cid][gid][1]),
                mOpacities[cid][gid],
                vec3t(mConics[cid][gid][0], mConics[cid][gid][1], mConics[cid][gid][2]));
        }
    }

    // Get the pixel index for a sparse pixel in the output tensor
    inline __device__ uint64_t
    sparsePixelIndex(const int32_t tileOrdinal, const uint32_t k) {
        // Suppose we're rendering the k^th active pixel in tile_id = active_tiles[t],
        // we write its rendered value to index pixel_map[tile_pixel_cumsum[tile_id - 1] + k] in
        // the output. The -1 is because the cumsum is inclusive
        const auto tilePixelCumsumValue = tileOrdinal > 0 ? mTilePixelCumsum[tileOrdinal - 1] : 0;
        return mPixelMap[tilePixelCumsumValue + k];
    }

    // Check if the current thread is rendering an active sparse pixel in the current tile
    inline __device__ bool
    tilePixelActive() {
        return fvdb::detail::ops::tilePixelActive(
            mTilePixelMask, mTileSize, blockIdx.x, threadIdx.y, threadIdx.x);
    }

    // Get the camera id and tile id from a sparse tile index. Assumes a 1D grid
    // of blocks, where blockIdx.x is the tile ordinal
    inline __device__ std::pair<int32_t, int32_t>
    sparseCameraTileId() {
        const int32_t globalTile = mActiveTiles[blockIdx.x];
        const int32_t cameraId   = globalTile / (mNumTilesW * mNumTilesH);
        const int32_t tileId     = globalTile % (mNumTilesW * mNumTilesH);
        return {cameraId, tileId};
    }

    // Get the camera id, tile coordinates, and pixel coordinates from a sparse
    // tile index. Assumes a 1D grid of blocks, where blockIdx.x is the tile ordinal
    // @return tuple of camera id, tile row, tile col, pixel i, pixel j
    __device__ cuda::std::tuple<int32_t, int32_t, int32_t, uint32_t, uint32_t>
    sparseCoordinates() {
        const auto [cameraId, tileId] = sparseCameraTileId();

        const int32_t tileRow = tileId / mNumTilesW;
        const int32_t tileCol = tileId % mNumTilesW;
        const uint32_t row    = tileRow * mTileSize + threadIdx.y;
        const uint32_t col    = tileCol * mTileSize + threadIdx.x;
        return {cameraId, tileRow, tileCol, row, col};
    }

    // Get the camera id, tile coordinates, and pixel coordinates from a dense block index.
    // Assumes a 3D grid of blocks, where blockIdx.x is the camera id, blockIdx.y is the tile
    // row, and blockIdx.z is the tile column.
    // @return tuple of camera id, tile row, tile col, pixel i, pixel j
    __device__ cuda::std::tuple<int32_t, int32_t, int32_t, uint32_t, uint32_t>
    denseCoordinates() {
        const int32_t cameraId = blockIdx.x;

        // blockIdx.yz runs from [0, numTilesH] x [0, numTilesW]
        const int32_t tileRow = blockIdx.y + mTileOriginH;
        const int32_t tileCol = blockIdx.z + mTileOriginW;

        // Pixel coordinates run from [0, height] x [0, width]
        // i.e. they are in the local coordinates of the crop starting from pixel
        //      [image_origin_h, image_origin_w] with size [image_height,
        //      image_width]
        const uint32_t row = tileRow * mTileSize + threadIdx.y;
        const uint32_t col = tileCol * mTileSize + threadIdx.x;
        return {cameraId, tileRow, tileCol, row, col};
    }

    __device__ void
    volumeRenderTileForward(const uint32_t cameraId,
                            const uint32_t tileStart,
                            const uint32_t tileEnd,
                            const uint32_t blockSize,
                            const uint32_t tileSize,
                            const bool pixelIsActive,
                            const uint32_t activePixelIndex,
                            const uint32_t row,
                            const uint32_t col) {
        using coord2t = typename Vec2Type<int32_t>::type;

        const uint32_t numBatches = (tileEnd - tileStart + blockSize - 1) / blockSize;

        // Ordinal of this thread in the block
        const uint32_t tidx = threadIdx.y * blockDim.x + threadIdx.x;

        // We don't return right away if the pixel is not in the image since we want
        // to use this thread to load gaussians into shared memory
        bool done = !pixelIsActive;

        extern __shared__ int s[];
        Gaussian2D<ScalarType> *sharedGaussians =
            reinterpret_cast<Gaussian2D<ScalarType> *>(s); // [blockSize]

        // NOTE: The accumulated transmittance is used in the backward pass, and
        // since it's a
        //       sum of many small numbers, we should really use double precision.
        //       However, this makes the backward pass 1.5x slower, so we stick with
        //       float for now and sort of just ignore small impact gaussians
        //       ¯\_(ツ)_/¯.
        ScalarType accumTransmittance = 1.0f;
        // index of most recent gaussian to write to this thread's pixel
        int32_t curIdx = -1;

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
            const uint32_t batchStart = tileStart + blockSize * b;
            const uint32_t idx        = batchStart + tidx;
            if (idx < tileEnd) {
                const int32_t g       = mTileGaussianIds[idx]; // which gaussian we're rendering
                sharedGaussians[tidx] = getGaussian(g);
            }

            // Sync threads so all gaussians for this batch are loaded in shared
            // memory
            __syncthreads();

            // Volume render Gaussians in this batch
            if (pixelIsActive) { // skip inactive sparse pixels
                const uint32_t batchSize = min(blockSize, tileEnd - batchStart);
                for (uint32_t t = 0; (t < batchSize) && !done; ++t) {
                    const Gaussian2D<ScalarType> gaussian = sharedGaussians[t];

                    const ScalarType px = ScalarType(col) + 0.5f;
                    const ScalarType py = ScalarType(row) + 0.5f;

                    const vec2t delta      = gaussian.delta(px, py);
                    const ScalarType sigma = gaussian.sigma(delta);
                    const ScalarType alpha = min(0.999f, gaussian.opacity * __expf(-sigma));

                    // TODO: are we quantizing the alpha too early? They could add up to
                    // significant opacity.
                    if (sigma < 0.f || alpha < 1.f / 255.f) {
                        continue;
                    }

                    const ScalarType nextTransmittance =
                        accumTransmittance * (ScalarType(1.0) - alpha);
                    if (nextTransmittance <= ScalarType(1e-4)) { // this pixel is done: exclusive
                        done = true;
                        break;
                    }

                    const ScalarType vis       = alpha * accumTransmittance;
                    const auto featureAccessor = [&]() {
                        if constexpr (IS_PACKED) {
                            return mFeatures[gaussian.id];
                        } else {
                            const int32_t cid = gaussian.id / mNumGaussiansPerCamera;
                            const int32_t gid = gaussian.id % mNumGaussiansPerCamera;
                            return mFeatures[cid][gid];
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
            if (mIsSparse) {
                auto pixelIndex = sparsePixelIndex(blockIdx.x, activePixelIndex);
                mOutSparseAlphas.data()[pixelIndex][0] = 1.0f - accumTransmittance;
                PRAGMA_UNROLL
                for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
                    mOutSparseFeatures.data()[pixelIndex][k] =
                        mHasBackgrounds ? pixOut[k] + accumTransmittance * mBackgrounds[cameraId][k]
                                        : pixOut[k];
                }
                mOutSparseLastIds.data()[pixelIndex] = curIdx;

            } else {
                // Here T is the transmittance AFTER the last gaussian in this pixel.
                // We (should) store double precision as T would be used in backward
                // pass and it can be very small and causing large diff in gradients
                // with float32. However, double precision makes the backward pass 1.5x
                // slower so we stick with float for now.
                mOutDenseAlphas[cameraId][row][col][0] = 1.0f - accumTransmittance;
                PRAGMA_UNROLL
                for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
                    mOutDenseFeatures[cameraId][row][col][k] =
                        mHasBackgrounds ? pixOut[k] + accumTransmittance * mBackgrounds[cameraId][k]
                                        : pixOut[k];
                }
                // index in bin of last gaussian in this pixel
                mOutDenseLastIds[cameraId][row][col] = curIdx;
            }
        }
    }
};

/****************************************************************************
 * Rasterization to Pixels Forward Pass
 ****************************************************************************/

// takes an index from the activeTiles tensor and returns the corresponding tile
// index in the specified camera image
template <typename ScalarType, uint32_t NUM_CHANNELS, bool IS_PACKED>
__global__ void
rasterizeGaussiansForward(DeviceArgs<ScalarType, NUM_CHANNELS, IS_PACKED> args) {
    // each thread draws one pixel, but also timeshares caching gaussians in a
    // shared tile
    int32_t cameraId;
    int32_t tileRow;
    int32_t tileCol;
    uint32_t row, col;

    cuda::std::tie(cameraId, tileRow, tileCol, row, col) =
        args.mIsSparse ? args.sparseCoordinates() : args.denseCoordinates();

    // return if out of bounds
    // keep not rasterizing threads around for reading data
    bool pixelInImage = (row < args.mImageHeight && col < args.mImageWidth);

    // Index of this pixel in the output for the block if it is active
    // Only used in sparse mode, computed using CUB BlockScan
    uint32_t activePixelIndex = 0;

    if (args.mIsSparse) {
        // Use CUB BlockScan to compute the index of each active pixel in the block
        __shared__
        typename cub::BlockScan<uint32_t, 16, cub::BLOCK_SCAN_RAKING, 16>::TempStorage tempStorage;
        pixelInImage = args.tilePixelActive();
        cub::BlockScan<uint32_t, 16, cub::BLOCK_SCAN_RAKING, 16>(tempStorage)
            .ExclusiveSum(pixelInImage, activePixelIndex);
        __syncthreads();
    }

    // when the mask is provided, render the background feature/color and return
    // if this tile is labeled as False
    if (args.mHasMasks && pixelInImage && !args.mMasks[cameraId][tileRow][tileCol]) {
        if (args.mIsSparse) {
            auto pixelIndex = args.sparsePixelIndex(blockIdx.x, activePixelIndex);
            PRAGMA_UNROLL
            for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
                args.mOutSparseFeatures.data()[pixelIndex][k] =
                    args.mHasBackgrounds ? args.mBackgrounds[cameraId][k] : 0.0f;
            }
        } else {
            PRAGMA_UNROLL
            for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
                args.mOutDenseFeatures[cameraId][row][col][k] =
                    args.mHasBackgrounds ? args.mBackgrounds[cameraId][k] : 0.0f;
            }
        }
        return;
    }

    // Figure out the first and (one past the) last Gaussian ID in this block/tile
    const int32_t firstGaussianIdInBlock = args.mTileOffsets[cameraId][tileRow][tileCol];
    auto [nextTileRow, nextTileCol]      = (tileCol < args.mNumTilesW - 1)
                                               ? std::make_tuple(tileRow, tileCol + 1)
                                               : std::make_tuple(tileRow + 1, 0); // wrap around
    const int32_t lastGaussianIdInBlock =
        ((cameraId == args.mNumCameras - 1) && (nextTileRow == args.mNumTilesH))
            ? args.mTotalIntersections
            : args.mTileOffsets[cameraId][nextTileRow][nextTileCol];
    const uint32_t blockSize = blockDim.x * blockDim.y;

    // Pixel coordinates in the global image (not just the local crop)
    const uint32_t globalRow = row + args.mImageOriginH;
    const uint32_t globalCol = col + args.mImageOriginW;
    args.volumeRenderTileForward(cameraId,
                                 firstGaussianIdInBlock,
                                 lastGaussianIdInBlock,
                                 blockSize,
                                 args.mTileSize,
                                 pixelInImage,
                                 activePixelIndex,
                                 globalRow,
                                 globalCol);
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
    using vec3t = typename Vec3Type<ScalarType>::type;
    using vec2t = typename Vec2Type<ScalarType>::type;

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

    auto args = DeviceArgs<ScalarType, NUM_CHANNELS, IS_PACKED>(means2d,
                                                                conics,
                                                                features,
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
                                                                pixelsToRender,
                                                                activeTiles,
                                                                tilePixelMask,
                                                                tilePixelCumsum,
                                                                pixelMap);

    const auto sizes = args.mIsSparse ? pixelsToRender.value().lsizes1() : std::vector<int64_t>{1};
    std::vector<torch::Tensor> featuresToRenderVec;
    std::vector<torch::Tensor> alphasToRenderVec;
    std::vector<torch::Tensor> lastIdsToRenderVec;

    if (args.mIsSparse) {
        for (const auto &size: sizes) {
            featuresToRenderVec.push_back(
                torch::empty({size, channels}, features.options().dtype(torch::kFloat32)));
            alphasToRenderVec.push_back(
                torch::empty({size, 1}, features.options().dtype(torch::kFloat32)));
            lastIdsToRenderVec.push_back(
                torch::empty({size}, features.options().dtype(torch::kInt32)));
        }
    } else {
        featuresToRenderVec.push_back(torch::empty({C, imageHeight, imageWidth, channels},
                                                   means2d.options().dtype(torch::kFloat32)));
        alphasToRenderVec.push_back(torch::empty({C, imageHeight, imageWidth, 1},
                                                 means2d.options().dtype(torch::kFloat32)));
        lastIdsToRenderVec.push_back(
            torch::empty({C, imageHeight, imageWidth}, means2d.options().dtype(torch::kInt32)));
    }

    auto outFeatures = fvdb::JaggedTensor(featuresToRenderVec);
    auto outAlphas   = fvdb::JaggedTensor(alphasToRenderVec);
    auto outLastIds  = fvdb::JaggedTensor(lastIdsToRenderVec);

    if (pixelsToRender.has_value()) {
        args.setSparseOutputArguments(outFeatures, outAlphas, outLastIds);
    } else {
        args.setDenseOutputArguments(outFeatures.jdata(), outAlphas.jdata(), outLastIds.jdata());
    }

    const at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    // Thread blocks cooperatively cache a tile of Gaussians in shared memory
    const uint32_t sharedMem = tileSize * tileSize * sizeof(Gaussian2D<ScalarType>);

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
    C10_CUDA_CHECK(cudaDeviceSynchronize());

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
