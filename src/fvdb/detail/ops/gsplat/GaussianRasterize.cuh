// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_GAUSSIANRASTERIZE_CUH
#define FVDB_DETAIL_OPS_GSPLAT_GAUSSIANRASTERIZE_CUH

#include <fvdb/detail/ops/gsplat/Gaussian2D.cuh>
#include <fvdb/detail/ops/gsplat/GaussianUtils.cuh>
#include <fvdb/detail/utils/AccessorHelpers.cuh>

#include <nanovdb/math/Math.h>

#include <cub/block/block_scan.cuh>
#include <cuda/std/tuple>

#define PRAGMA_UNROLL _Pragma("unroll")

namespace fvdb::detail::ops {

// Initialize an accessor for a tensor. The tensor must be a CUDA tensor.
template <typename T, int N>
inline auto
initAccessor(const torch::Tensor &tensor, const std::string &name) {
    TORCH_CHECK(tensor.is_cuda() || tensor.is_privateuseone(),
                "Tensor ",
                name,
                " must be a CUDA or PrivateUse1 tensor");
    return tensor.packed_accessor64<T, N, torch::RestrictPtrTraits>();
}

// Initialize an accessor for an optional tensor. The tensor must be a CUDA tensor. If the tensor
// is std::nullopt, return an invalid accessor to a temporary empty tensor. This invalid accessor
// should not be used, it only exists because accessors cannot be default-constructed.
template <typename T, int N>
inline auto
initAccessor(const std::optional<torch::Tensor> &tensor,
             torch::TensorOptions defaultOptions,
             const std::string &name) {
    return initAccessor<T, N>(
        tensor.value_or(torch::empty(std::array<int64_t, N>{}, defaultOptions)), name);
}

// Initialize a jagged accessor for a JaggedTensor. The tensor must be a CUDA tensor.
template <typename T, int N>
inline auto
initJaggedAccessor(const fvdb::JaggedTensor &tensor, const std::string &name) {
    TORCH_CHECK(tensor.is_cuda() || tensor.is_privateuseone(),
                "Tensor ",
                name,
                " must be a CUDA or PrivateUse1 tensor");
    return tensor.packed_accessor64<T, N, torch::RestrictPtrTraits>();
}

/// @brief Common fields and helpers for both forward and backward rasterization kernels
/// @tparam ScalarType The scalar type of the Gaussian
/// @tparam NUM_CHANNELS The number of channels of the Gaussian
/// @tparam IS_PACKED Whether the Gaussian is packed (i.e. linearized across the outer dimensions)
template <typename ScalarType, size_t NUM_CHANNELS, bool IS_PACKED> struct RasterizeCommonArgs {
    constexpr static size_t NUM_OUTER_DIMS         = IS_PACKED ? 1 : 2;
    constexpr static ScalarType ALPHA_THRESHOLD    = ScalarType{0.999};
    using vec2t                                    = nanovdb::math::Vec2<ScalarType>;
    using vec3t                                    = nanovdb::math::Vec3<ScalarType>;
    template <typename T, int N> using TorchRAcc64 = fvdb::TorchRAcc64<T, N>;
    using ScalarAccessor                           = TorchRAcc64<ScalarType, NUM_OUTER_DIMS>;
    using VectorAccessor                           = TorchRAcc64<ScalarType, NUM_OUTER_DIMS + 1>;

    // 0 for the kCUDA/single GPU case. For kPrivateUse1/multi-GPU, we distribute the blocks in the
    // single GPU case amongst multiple GPUs and mBlockOffset translates the per-device block index
    // into the corresponding global block index that is unique across all GPUs.
    uint32_t mBlockOffset;
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

    // Common input tensors
    VectorAccessor mMeans2d;                  // [C, N, 2] or [nnz, 2]
    VectorAccessor mConics;                   // [C, N, 3] or [nnz, 3]
    ScalarAccessor mOpacities;                // [C, N] or [nnz]
    TorchRAcc64<int32_t, 1> mTileGaussianIds; // [totalIntersections]
    // Tile offsets: can be 3D [C, nTilesH, nTilesW] (dense) or 1D [AT + 1] (sparse)
    // We store both accessor types but only one is valid based on mTileOffsetsAreSparse
    bool mTileOffsetsAreSparse;
    TorchRAcc64<int32_t, 3>
        mTileOffsets; // [C, nTilesH, nTilesW] - used when !mTileOffsetsAreSparse
    TorchRAcc64<int32_t, 1> mSparseTileOffsets; // [AT + 1] - used when mTileOffsetsAreSparse
    // Common optional input tensors
    bool mHasFeatures;
    VectorAccessor mFeatures;                // [C, N, NUM_CHANNELS] or [nnz, NUM_CHANNELS]
    TorchRAcc64<ScalarType, 2> mBackgrounds; // [C, NUM_CHANNELS]
    bool mHasBackgrounds;
    TorchRAcc64<bool, 3> mMasks;             // [C, nTilesH, nTilesW]
    bool mHasMasks;

    // Common sparse input tensors
    bool mIsSparse;
    TorchRAcc64<int32_t, 1> mActiveTiles;             // [AT]
    TorchRAcc64<uint64_t, 2> mTilePixelMask;          // [AT, wordsPerTile] e.g. [AT, 4]
    TorchRAcc64<int64_t, 1> mTilePixelCumsum;         // [AT]
    TorchRAcc64<int64_t, 1> mPixelMap;                // [AP]

    RasterizeCommonArgs(
        const torch::Tensor &means2d,                 // [C, N, 2] or [nnz, 2]
        const torch::Tensor &conics,                  // [C, N, 3] or [nnz, 3]
        const torch::Tensor &opacities,               // [C, N] or [nnz]
        const std::optional<torch::Tensor> &features, // [C, N, NUM_CHANNELS] or [nnz, NUM_CHANNELS]
        const std::optional<torch::Tensor> &backgrounds, // [C, NUM_CHANNELS]
        const std::optional<torch::Tensor> &masks,       // [C, numTilesH, numTilesW]
        const uint32_t imageWidth,
        const uint32_t imageHeight,
        const uint32_t imageOriginW,
        const uint32_t imageOriginH,
        const uint32_t tileSize,
        const uint32_t blockOffset,
        const torch::Tensor &tileOffsets, // [C, numTilesH, numTilesW] (dense) or [AT + 1] (sparse)
        const torch::Tensor &tileGaussianIds,                           // [totalIntersections]
        const std::optional<torch::Tensor> &activeTiles = std::nullopt, // [AT]
        const std::optional<torch::Tensor> &tilePixelMask =
            std::nullopt, // [AT, wordsPerTileBitmask] e.g. [AT, 4]
        const std::optional<torch::Tensor> &tilePixelCumsum = std::nullopt, // [AT]
        const std::optional<torch::Tensor> &pixelMap        = std::nullopt)        // [AP]
        : mImageWidth(imageWidth), mImageHeight(imageHeight), mImageOriginW(imageOriginW),
          mImageOriginH(imageOriginH), mTileOriginW(imageOriginW / tileSize),
          mTileOriginH(imageOriginH / tileSize), mTileSize(tileSize),
          mMeans2d(initAccessor<ScalarType, NUM_OUTER_DIMS + 1>(means2d, "means2d")),
          mConics(initAccessor<ScalarType, NUM_OUTER_DIMS + 1>(conics, "conics")),
          mOpacities(initAccessor<ScalarType, NUM_OUTER_DIMS>(opacities, "opacities")),
          mTileGaussianIds(initAccessor<int32_t, 1>(tileGaussianIds, "tileGaussianIds")),
          mTileOffsetsAreSparse(tileOffsets.dim() == 1),
          // Initialize 3D accessor for dense mode, or placeholder for sparse mode
          mTileOffsets(mTileOffsetsAreSparse
                           ? initAccessor<int32_t, 3>(
                                 torch::empty({0, 0, 0}, tileOffsets.options()), "tileOffsets")
                           : initAccessor<int32_t, 3>(tileOffsets, "tileOffsets")),
          // Initialize 1D accessor for sparse mode, or placeholder for dense mode
          mSparseTileOffsets(
              mTileOffsetsAreSparse
                  ? initAccessor<int32_t, 1>(tileOffsets, "sparseTileOffsets")
                  : initAccessor<int32_t, 1>(torch::empty({0}, tileOffsets.options()),
                                             "sparseTileOffsets")),
          mHasFeatures(features.has_value()),
          mFeatures(initAccessor<ScalarType, NUM_OUTER_DIMS + 1>(
              features, opacities.options(), "features")),
          mBackgrounds(initAccessor<ScalarType, 2>(backgrounds, means2d.options(), "backgrounds")),
          mHasBackgrounds(backgrounds.has_value()),
          mMasks(initAccessor<bool, 3>(masks, means2d.options().dtype(torch::kBool), "masks")),
          mHasMasks(masks.has_value()), mBlockOffset(blockOffset),
          mIsSparse(activeTiles.has_value()),
          mActiveTiles(initAccessor<int32_t, 1>(activeTiles, tileOffsets.options(), "activeTiles")),
          mTilePixelMask(initAccessor<uint64_t, 2>(
              tilePixelMask, means2d.options().dtype(torch::kUInt64), "tilePixelMask")),
          mTilePixelCumsum(initAccessor<int64_t, 1>(
              tilePixelCumsum, tileOffsets.options().dtype(torch::kInt64), "tilePixelCumsum")),
          mPixelMap(initAccessor<int64_t, 1>(
              pixelMap, tileOffsets.options().dtype(torch::kInt64), "pixelMap")) {
        static_assert(NUM_OUTER_DIMS == 1 || NUM_OUTER_DIMS == 2, "NUM_OUTER_DIMS must be 1 or 2");

        mNumTilesW = (imageWidth + tileSize - 1) / tileSize;
        mNumTilesH = (imageHeight + tileSize - 1) / tileSize;

        if (mTileOffsetsAreSparse) {
            // Sparse mode: means2d is [C, N, 2] for non-packed, [nnz, 2] for packed
            // (For packed sparse mode, mNumCameras isn't used)
            mNumCameras = IS_PACKED ? 0 : means2d.size(0);
        } else {
            // Dense mode: tileOffsets is [C, TH, TW]
            mNumCameras = tileOffsets.size(0);
        }
        mNumGaussiansPerCamera = IS_PACKED ? 0 : mMeans2d.size(1);
        mTotalIntersections    = mTileGaussianIds.size(0);

        checkInputShapes();
    }

    // Check that the input tensor shapes are valid
    void
    checkInputShapes() {
        const int64_t totalGaussians = IS_PACKED ? mMeans2d.size(0) : 0;

        TORCH_CHECK_VALUE(2 == mMeans2d.size(NUM_OUTER_DIMS), "Bad size for means2d");
        TORCH_CHECK_VALUE(3 == mConics.size(NUM_OUTER_DIMS), "Bad size for conics");

        if constexpr (IS_PACKED) {
            TORCH_CHECK_VALUE(totalGaussians == mMeans2d.size(0), "Bad size for means2d");
            TORCH_CHECK_VALUE(totalGaussians == mConics.size(0), "Bad size for conics");
            TORCH_CHECK_VALUE(totalGaussians == mOpacities.size(0), "Bad size for opacities");
        } else {
            TORCH_CHECK_VALUE(mNumCameras == mMeans2d.size(0), "Bad size for means2d");
            TORCH_CHECK_VALUE(mNumGaussiansPerCamera == mMeans2d.size(1), "Bad size for means2d");
            TORCH_CHECK_VALUE(mMeans2d.size(1) == 0, "Zero size for means2d");
            TORCH_CHECK_VALUE(mNumCameras == mConics.size(0), "Bad size for conics");
            TORCH_CHECK_VALUE(mNumGaussiansPerCamera == mConics.size(1), "Bad size for conics");
            TORCH_CHECK_VALUE(mNumCameras == mOpacities.size(0), "Bad size for opacities");
            TORCH_CHECK_VALUE(mNumGaussiansPerCamera == mOpacities.size(1),
                              "Bad size for opacities");
        }

        if (mHasFeatures) {
            TORCH_CHECK_VALUE(NUM_CHANNELS == mFeatures.size(NUM_OUTER_DIMS),
                              "Bad size for features");
            if constexpr (IS_PACKED) {
                TORCH_CHECK_VALUE(totalGaussians == mFeatures.size(0), "Bad size for features");
            } else {
                TORCH_CHECK_VALUE(mNumCameras == mFeatures.size(0), "Bad size for features");
                TORCH_CHECK_VALUE(mNumGaussiansPerCamera == mFeatures.size(1),
                                  "Bad size for features");
            }
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

        // Validate tileOffsets shape
        if (mTileOffsetsAreSparse) {
            // Sparse mode: tileOffsets is 1D [AT + 1]
            if (mIsSparse) {
                TORCH_CHECK_VALUE(mSparseTileOffsets.size(0) == mActiveTiles.size(0) + 1,
                                  "Sparse tileOffsets size must be activeTiles.size(0) + 1");
            }
        } else {
            // Dense mode: tileOffsets is 3D [C, numTilesH, numTilesW]
            TORCH_CHECK_VALUE(mNumCameras == mTileOffsets.size(0), "Bad size for tileOffsets");
            TORCH_CHECK_VALUE(mNumTilesH == mTileOffsets.size(1), "Bad size for tileOffsets");
            TORCH_CHECK_VALUE(mNumTilesW == mTileOffsets.size(2), "Bad size for tileOffsets");
        }

        if (mIsSparse) {
            TORCH_CHECK_VALUE(mTilePixelMask.size(0) == mActiveTiles.size(0),
                              "Bad size for tilePixelMask");
            TORCH_CHECK_VALUE(mTilePixelMask.size(1) == numWordsPerTileBitmask(mTileSize),
                              "Bad size for tilePixelMask");
            TORCH_CHECK_VALUE(mTilePixelCumsum.size(0) == mActiveTiles.size(0),
                              "Bad size for tilePixelCumsum");
        }
    }

    // Get the index of the current pixel in the current block
    inline __device__ cuda::std::tuple<bool, uint32_t>
    activePixelIndex(uint32_t row, uint32_t col) {
        uint32_t index = 0;
        bool pixelInImage =
            mIsSparse ? tilePixelActive() : (row < mImageHeight && col < mImageWidth);

        if (mIsSparse) {
            // Use CUB BlockScan to compute the index of each active pixel in the block
            __shared__
            typename cub::BlockScan<uint32_t, 16, cub::BLOCK_SCAN_RAKING, 16>::TempStorage
                tempStorage;

            cub::BlockScan<uint32_t, 16, cub::BLOCK_SCAN_RAKING, 16>(tempStorage)
                .ExclusiveSum(pixelInImage, index);
            __syncthreads();
        }
        return {pixelInImage, index};
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

    // Evaluate a Gaussian at a given pixel
    // @return tuple: {gaussianIsValid, delta, exp(-sigma),
    //   alpha = min(ALPHA_THRESHOLD, opacity * exp(-sigma))}
    inline __device__ auto
    evalGaussian(const Gaussian2D<ScalarType> &gaussian,
                 const ScalarType px,
                 const ScalarType py) const {
        const auto delta         = gaussian.delta(px, py);
        const auto sigma         = gaussian.sigma(px, py);
        const auto expMinusSigma = __expf(-sigma);
        const auto alpha         = min(ALPHA_THRESHOLD, gaussian.opacity * expMinusSigma);

        const bool gaussianIsValid = !(sigma < 0 || alpha < 1.f / 255.f);

        return std::make_tuple(gaussianIsValid, delta, expMinusSigma, alpha);
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

    inline __device__ uint64_t
    pixelIndex(const uint64_t cameraId,
               const uint64_t row,
               const uint64_t col,
               const uint32_t activePixelIndex) {
        return mIsSparse ? sparsePixelIndex(blockIdx.x + mBlockOffset, activePixelIndex)
                         : cameraId * this->mImageWidth * this->mImageHeight +
                               row * this->mImageWidth + col;
    }

    // Check if the current thread is rendering an active sparse pixel in the current tile
    inline __device__ bool
    tilePixelActive() {
        return fvdb::detail::ops::tilePixelActive(
            mTilePixelMask, mTileSize, blockIdx.x + mBlockOffset, threadIdx.y, threadIdx.x);
    }

    // Get the camera id and tile id from a sparse tile index. Assumes a 1D grid
    // of blocks, where blockIdx.x + mBlockOffset is the tile ordinal
    inline __device__ std::pair<int32_t, int32_t>
    sparseCameraTileId() {
        const int32_t globalTile = mActiveTiles[blockIdx.x + mBlockOffset];
        const int32_t cameraId   = globalTile / (mNumTilesW * mNumTilesH);
        const int32_t tileId     = globalTile % (mNumTilesW * mNumTilesH);
        return {cameraId, tileId};
    }

    // Get the camera id, tile coordinates, and pixel coordinates from a sparse
    // tile index. Assumes a 1D grid of blocks, where blockIdx.x + mBlockOffset is the tile ordinal
    // @return tuple of camera id, tile row, tile col, pixel i, pixel j
    inline __device__ cuda::std::tuple<int32_t, int32_t, int32_t, uint32_t, uint32_t>
    sparseCoordinates() {
        const auto [cameraId, tileId] = sparseCameraTileId();

        const int32_t tileRow = tileId / mNumTilesW;
        const int32_t tileCol = tileId % mNumTilesW;
        const uint32_t row    = tileRow * mTileSize + threadIdx.y;
        const uint32_t col    = tileCol * mTileSize + threadIdx.x;
        return {cameraId, tileRow, tileCol, row, col};
    }

    // Get the camera id, tile coordinates, and pixel coordinates from a dense block index.
    // Assumes a 1D grid of blocks, where blockIdx.x + mBlockOffset is the tile ordinal
    // @return tuple of camera id, tile row, tile col, pixel i, pixel j
    inline __device__ cuda::std::tuple<int32_t, int32_t, int32_t, uint32_t, uint32_t>
    denseCoordinates() {
        auto globalLinearBlockIdx  = blockIdx.x + mBlockOffset;
        const uint32_t tileExtentW = (mImageWidth + mTileSize - 1) / mTileSize;
        const uint32_t tileExtentH = (mImageHeight + mTileSize - 1) / mTileSize;
        dim3 globalBlockIdx(globalLinearBlockIdx / (tileExtentH * tileExtentW),
                            (globalLinearBlockIdx / tileExtentW) % tileExtentH,
                            globalLinearBlockIdx % tileExtentW);

        const int32_t cameraId = globalBlockIdx.x;

        // blockIdx.yz runs from [0, numTilesH] x [0, numTilesW]
        const int32_t tileRow = globalBlockIdx.y + mTileOriginH;
        const int32_t tileCol = globalBlockIdx.z + mTileOriginW;

        // Pixel coordinates run from [0, height] x [0, width]
        // i.e. they are in the local coordinates of the crop starting from pixel
        //      [image_origin_h, image_origin_w] with size [image_height,
        //      image_width]
        const uint32_t row = tileRow * mTileSize + threadIdx.y;
        const uint32_t col = tileCol * mTileSize + threadIdx.x;
        return {cameraId, tileRow, tileCol, row, col};
    }

    // Get the first and last Gaussian ID in the current tile
    inline __device__ cuda::std::tuple<int32_t, int32_t>
    tileGaussianRange(uint32_t cameraId, uint32_t tileRow, uint32_t tileCol) {
        // In sparse mode with 1D tile offsets, use the sparse offsets directly
        // indexed by the tile ordinal (blockIdx.x + mBlockOffset)
        if (mTileOffsetsAreSparse) {
            const int32_t tileOrdinal = blockIdx.x + mBlockOffset;
            return {mSparseTileOffsets[tileOrdinal], mSparseTileOffsets[tileOrdinal + 1]};
        }

        // Dense mode: look up by (cameraId, tileRow, tileCol)
        const int32_t firstGaussianIdInBlock = mTileOffsets[cameraId][tileRow][tileCol];
        auto [nextTileRow, nextTileCol]      = (tileCol < mNumTilesW - 1)
                                                   ? std::make_tuple(tileRow, tileCol + 1)
                                                   : std::make_tuple(tileRow + 1, 0u); // wrap around
        const int32_t lastGaussianIdInBlock =
            ((cameraId == mNumCameras - 1) && (nextTileRow == mNumTilesH))
                ? mTotalIntersections
                : mTileOffsets[cameraId][nextTileRow][nextTileCol];
        return {firstGaussianIdInBlock, lastGaussianIdInBlock};
    }

    /// @brief Get the block dimensions for the forward/backward pass
    /// @return The block dimensions
    const dim3
    getBlockDim() const {
        return {mTileSize, mTileSize, 1};
    }

    /// @brief Get the grid dimensions for the forward/backward pass
    /// @return The grid dimensions
    const dim3
    getGridDim() const {
        if (mIsSparse) {
            // Sparse mode: only launch blocks for active tiles
            return {static_cast<uint32_t>(mActiveTiles.size(0)), 1, 1};
        } else {
            // Dense mode: launch blocks for all tiles
            const uint32_t tileExtentW = (mImageWidth + mTileSize - 1) / mTileSize;
            const uint32_t tileExtentH = (mImageHeight + mTileSize - 1) / mTileSize;
            return {mNumCameras * tileExtentH * tileExtentW, 1, 1};
        }
    }
};

} // namespace fvdb::detail::ops
#endif // FVDB_DETAIL_OPS_GSPLAT_GAUSSIANRASTERIZE_CUH
