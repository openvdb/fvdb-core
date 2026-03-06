// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_GAUSSIANTILEINTERSECTION_H
#define FVDB_DETAIL_OPS_GSPLAT_GAUSSIANTILEINTERSECTION_H

#include <fvdb/detail/ops/gsplat/GaussianRenderSettings.h>

#include <torch/types.h>

#include <optional>
#include <tuple>
#include <utility>

#if defined(__CUDACC__)
#include <fvdb/detail/utils/AccessorHelpers.cuh>

#include <cub/block/block_scan.cuh>
#include <cuda/std/tuple>
#endif

namespace fvdb {
namespace detail {
namespace ops {

template <typename CallbackT>
inline void dispatchTileIntersectionsAccessor(
    const torch::Tensor &tileOffsets,
    const torch::Tensor &tileGaussianIds,
    const RenderWindow2D &renderWindow,
    uint32_t tileSize,
    uint32_t blockOffset,
    CallbackT &&callback,
    const std::optional<torch::Tensor> &activeTiles     = std::nullopt,
    const std::optional<torch::Tensor> &tilePixelMask   = std::nullopt,
    const std::optional<torch::Tensor> &tilePixelCumsum = std::nullopt,
    const std::optional<torch::Tensor> &pixelMap        = std::nullopt);

/// @brief Tile intersection data for fully dense rasterization (Mode 1).
///
/// All image tiles are processed and every pixel within each tile is rendered. Tile offsets are
/// stored as a 3D tensor [C, TH, TW] indexed by camera, tile row, and tile column. The camera
/// count is known from the tile offsets tensor.
///
/// The nested Accessor provides device-callable helpers consumed by the rasterization kernels.
class DenseTileIntersections {
  public:
    DenseTileIntersections() = default;

    /// @brief Compute dense tile intersections from projected Gaussian parameters.
    DenseTileIntersections(const torch::Tensor &means2d,
                           const torch::Tensor &radii,
                           const torch::Tensor &depths,
                           const at::optional<torch::Tensor> &cameraIds,
                           uint32_t tileSize,
                           uint32_t imageHeight,
                           uint32_t imageWidth);

    /// @brief Construct from precomputed tile offsets and Gaussian IDs.
    DenseTileIntersections(torch::Tensor tileOffsets,
                           torch::Tensor tileGaussianIds,
                           uint32_t tileSize);

    /// @brief 3D tile offset tensor [C, TH, TW].
    const torch::Tensor &tileOffsets() const;

    /// @brief Flattened Gaussian IDs sorted by tile [totalIntersections].
    const torch::Tensor &tileGaussianIds() const;

    /// @brief Side length of a square tile in pixels.
    uint32_t tileSize() const;

    /// @brief Total number of Gaussian-tile intersections.
    int64_t totalIntersections() const;

#if defined(__CUDACC__)
    /// @brief Device-side accessor for dense rasterization kernels.
    ///
    /// Every tile in the [C, TH, TW] grid is launched as a CUDA block. Pixel activity is
    /// determined solely by bounds checking (row < H && col < W). The pixel index is a simple
    /// linear index into the dense image: camera * H * W + row * W + col.
    struct Accessor {
      private:
        Accessor(fvdb::TorchRAcc64<int32_t, 3> tileOffsetsIn,
                 fvdb::TorchRAcc64<int32_t, 1> tileGaussianIdsIn,
                 uint32_t numCamerasIn,
                 const RenderWindow2D &renderWindow,
                 uint32_t tileSizeIn,
                 uint32_t blockOffsetIn,
                 int32_t totalIntersectionsIn)
            : mTileOffsets(tileOffsetsIn), mTileGaussianIds(tileGaussianIdsIn),
              mNumCameras(numCamerasIn), mRenderWidth(renderWindow.width),
              mRenderHeight(renderWindow.height), mTileOriginW(renderWindow.originW / tileSizeIn),
              mTileOriginH(renderWindow.originH / tileSizeIn), mTileSize(tileSizeIn),
              mBlockOffset(blockOffsetIn), mTotalIntersections(totalIntersectionsIn) {
            mNumTilesW = renderWindow.tileExtentW(mTileSize);
            mNumTilesH = renderWindow.tileExtentH(mTileSize);
        }

        friend class DenseTileIntersections;
        template <typename CallbackT>
        friend inline void
        dispatchTileIntersectionsAccessor(const torch::Tensor &tileOffsets,
                                          const torch::Tensor &tileGaussianIds,
                                          const RenderWindow2D &renderWindow,
                                          uint32_t tileSize,
                                          uint32_t blockOffset,
                                          CallbackT &&callback,
                                          const std::optional<torch::Tensor> &activeTiles,
                                          const std::optional<torch::Tensor> &tilePixelMask,
                                          const std::optional<torch::Tensor> &tilePixelCumsum,
                                          const std::optional<torch::Tensor> &pixelMap);

      public:
        /// @brief Empty scratch space (dense mode needs no block scan).
        struct ActivePixelScratch {};

        /// @brief Number of tile columns in the render window.
        __host__ __device__ inline uint32_t
        numTilesW() const {
            return mNumTilesW;
        }

        /// @brief Number of tile rows in the render window.
        __host__ __device__ inline uint32_t
        numTilesH() const {
            return mNumTilesH;
        }

        /// @brief Number of cameras (derived from tile offsets dimension 0).
        __host__ __device__ inline uint32_t
        numCameras() const {
            return mNumCameras;
        }

        /// @brief Return the known camera count, ignoring the fallback.
        __host__ __device__ inline uint32_t
        cameraCount(uint32_t fallback) const {
            (void)fallback;
            return mNumCameras;
        }

        /// @brief Side length of a square tile in pixels.
        __host__ __device__ inline uint32_t
        tileSize() const {
            return mTileSize;
        }

        /// @brief Total tiles across all cameras (C * TH * TW).
        __host__ __device__ inline uint32_t
        numActiveTiles() const {
            return mNumCameras * mNumTilesW * mNumTilesH;
        }

        /// @brief Total pixels across all cameras (C * H * W).
        __host__ __device__ inline uint32_t
        pixelCount() const {
            return mNumCameras * mRenderWidth * mRenderHeight;
        }

        /// @brief Dense mode packs all cameras into a single output tensor.
        __host__ __device__ inline uint32_t
        outputNumTensors(uint32_t numCameras) const {
            (void)numCameras;
            return 1;
        }

        /// @brief Total pixel count for output validation.
        __host__ __device__ inline uint64_t
        outputPixelCount(uint32_t numCameras, uint32_t renderHeight, uint32_t renderWidth) const {
            return static_cast<uint64_t>(numCameras) * renderHeight * renderWidth;
        }

        /// @brief Translate a per-device block ordinal to a global tile ordinal.
        __host__ __device__ inline uint32_t
        globalTileOrdinal(uint32_t blockOrdinal) const {
            return blockOrdinal + mBlockOffset;
        }

        /// @brief Look up the Gaussian index at a given intersection position.
        __device__ inline int32_t
        gaussianIdAt(int32_t intersectionIdx) const {
            return mTileGaussianIds[intersectionIdx];
        }

        /// @brief Get the [first, last) Gaussian intersection range for a tile via 3D lookup.
        __device__ inline cuda::std::tuple<int32_t, int32_t>
        tileGaussianRange(uint32_t cameraId, uint32_t tileRow, uint32_t tileCol) const {
            const int32_t firstGaussianIdInBlock = mTileOffsets[cameraId][tileRow][tileCol];
            auto [nextTileRow, nextTileCol]      = (tileCol < mNumTilesW - 1)
                                                       ? cuda::std::make_tuple(tileRow, tileCol + 1)
                                                       : cuda::std::make_tuple(tileRow + 1, 0u);
            const int32_t lastGaussianIdInBlock =
                ((cameraId == mNumCameras - 1) && (nextTileRow == mNumTilesH))
                    ? mTotalIntersections
                    : mTileOffsets[cameraId][nextTileRow][nextTileCol];
            return {firstGaussianIdInBlock, lastGaussianIdInBlock};
        }

        /// @brief Get the Gaussian range for a tile from a block ordinal (delegates to 3D lookup).
        __device__ inline cuda::std::tuple<int32_t, int32_t>
        tileGaussianRangeFromBlock(uint32_t blockOrdinal,
                                   uint32_t cameraId,
                                   uint32_t tileRow,
                                   uint32_t tileCol) const {
            (void)blockOrdinal;
            return tileGaussianRange(cameraId, tileRow, tileCol);
        }

        /// @brief Decompose a global tile ordinal into (cameraId, tileRow, tileCol).
        __device__ inline cuda::std::tuple<int32_t, int32_t, int32_t>
        cameraTileId(uint32_t tileOrdinal,
                     uint32_t numTilesW,
                     uint32_t numTilesH,
                     uint32_t tileOriginW,
                     uint32_t tileOriginH) const {
            const int32_t cameraId = tileOrdinal / (numTilesH * numTilesW);
            const int32_t localRow = (tileOrdinal / numTilesW) % numTilesH;
            const int32_t localCol = tileOrdinal % numTilesW;
            return {cameraId,
                    localRow + static_cast<int32_t>(tileOriginH),
                    localCol + static_cast<int32_t>(tileOriginW)};
        }

        /// @brief Get (cameraId, tileRow, tileCol, pixelRow, pixelCol) from tile and thread
        /// indices.
        __device__ inline cuda::std::tuple<int32_t, int32_t, int32_t, uint32_t, uint32_t>
        coordinates(uint32_t tileOrdinal,
                    uint32_t threadX,
                    uint32_t threadY,
                    uint32_t numTilesW,
                    uint32_t numTilesH,
                    uint32_t tileOriginW,
                    uint32_t tileOriginH,
                    uint32_t tileSize) const {
            const auto [cameraId, tileRow, tileCol] =
                cameraTileId(tileOrdinal, numTilesW, numTilesH, tileOriginW, tileOriginH);
            const uint32_t row = tileRow * tileSize + threadY;
            const uint32_t col = tileCol * tileSize + threadX;
            return {cameraId, tileRow, tileCol, row, col};
        }

        /// @brief Convenience overload using stored tile layout and the block ordinal.
        __device__ inline cuda::std::tuple<int32_t, int32_t, int32_t, uint32_t, uint32_t>
        coordinates(uint32_t blockOrdinal, uint32_t threadX, uint32_t threadY) const {
            return coordinates(globalTileOrdinal(blockOrdinal),
                               threadX,
                               threadY,
                               mNumTilesW,
                               mNumTilesH,
                               mTileOriginW,
                               mTileOriginH,
                               mTileSize);
        }

        /// @brief A pixel is active if it falls within the render window bounds.
        __device__ inline bool
        tilePixelActive(uint32_t tileSideLength,
                        uint32_t tileOrdinal,
                        uint32_t iInTile,
                        uint32_t jInTile,
                        uint32_t row,
                        uint32_t col,
                        uint32_t renderWidth,
                        uint32_t renderHeight) const {
            (void)tileSideLength;
            (void)tileOrdinal;
            (void)iInTile;
            (void)jInTile;
            return row < renderHeight && col < renderWidth;
        }

        /// @brief Convenience overload using stored render dimensions and block ordinal.
        __device__ inline bool
        tilePixelActiveFromBlock(uint32_t blockOrdinal,
                                 uint32_t iInTile,
                                 uint32_t jInTile,
                                 uint32_t row,
                                 uint32_t col) const {
            return tilePixelActive(mTileSize,
                                   globalTileOrdinal(blockOrdinal),
                                   iInTile,
                                   jInTile,
                                   row,
                                   col,
                                   mRenderWidth,
                                   mRenderHeight);
        }

        /// @brief In dense mode, every active pixel has index 0 (no prefix scan needed).
        __device__ inline cuda::std::tuple<bool, uint32_t>
        activePixelIndexFromBlock(uint32_t blockOrdinal,
                                  uint32_t iInTile,
                                  uint32_t jInTile,
                                  uint32_t row,
                                  uint32_t col,
                                  ActivePixelScratch &scratch) const {
            (void)scratch;
            const bool pixelActive =
                tilePixelActiveFromBlock(blockOrdinal, iInTile, jInTile, row, col);
            return {pixelActive, 0u};
        }

        /// @brief Linear index into the dense image: camera * H * W + row * W + col.
        __device__ inline uint64_t
        pixelIndex(uint64_t cameraId,
                   uint64_t row,
                   uint64_t col,
                   uint32_t activePixelIndex,
                   uint32_t renderWidth,
                   uint32_t renderHeight,
                   uint32_t tileOrdinal) const {
            (void)activePixelIndex;
            (void)renderHeight;
            (void)tileOrdinal;
            return cameraId * renderWidth * renderHeight + row * renderWidth + col;
        }

        /// @brief Convenience overload using stored render dimensions and block ordinal.
        __device__ inline uint64_t
        pixelIndexFromBlock(uint64_t cameraId,
                            uint64_t row,
                            uint64_t col,
                            uint32_t activePixelIndex,
                            uint32_t blockOrdinal) const {
            return pixelIndex(cameraId,
                              row,
                              col,
                              activePixelIndex,
                              mRenderWidth,
                              mRenderHeight,
                              globalTileOrdinal(blockOrdinal));
        }

      private:
        fvdb::TorchRAcc64<int32_t, 3> mTileOffsets;     // [C, TH, TW]
        fvdb::TorchRAcc64<int32_t, 1> mTileGaussianIds; // [totalIntersections]
        uint32_t mNumCameras;
        uint32_t mNumTilesH;
        uint32_t mNumTilesW;
        uint32_t mRenderWidth;
        uint32_t mRenderHeight;
        uint32_t mTileOriginW;
        uint32_t mTileOriginH;
        uint32_t mTileSize;
        uint32_t mBlockOffset;
        int32_t mTotalIntersections;
    };

    /// @brief Create a device accessor from this host-side object.
    Accessor accessor(const RenderWindow2D &renderWindow, uint32_t blockOffset) const;
#endif

  private:
    torch::Tensor mTileOffsets;
    torch::Tensor mTileGaussianIds;
    uint32_t mTileSize = 0;
};

/// @brief Tile intersection data for sparse-pixel rasterization with dense (3D) tile offsets
///        (Mode 2).
///
/// Only a subset of pixels are rendered (determined by a bitmask per tile), but tile offsets are
/// stored as a dense 3D tensor [C, TH, TW] so the camera count is known. Active tiles are
/// identified by an explicit list. Pixel output is indexed through a pixel map built from
/// cumulative active-pixel counts.
///
/// This type is used when the caller provides activeTiles (sparse pixel selection) together with
/// 3D tile offsets (dense tile layout). It avoids the abstraction leak of storing dense state
/// inside SparseTileIntersections.
class SparseDenseTileIntersections {
  public:
    SparseDenseTileIntersections() = default;

    /// @brief Construct from precomputed tensors.
    /// @param tileOffsets     Dense tile offsets [C, TH, TW].
    /// @param tileGaussianIds Sorted Gaussian IDs [totalIntersections].
    /// @param tileSize        Side length of a square tile in pixels.
    /// @param activeTiles     Indices of active tiles in the flattened C*TH*TW grid.
    /// @param tilePixelMask   Bitmask of active pixels per tile [AT, wordsPerTile].
    /// @param tilePixelCumsum Cumulative count of active pixels per tile [AT].
    /// @param pixelMap        Map from cumulative pixel index to output index [AP].
    SparseDenseTileIntersections(torch::Tensor tileOffsets,
                                 torch::Tensor tileGaussianIds,
                                 uint32_t tileSize,
                                 torch::Tensor activeTiles,
                                 torch::Tensor tilePixelMask,
                                 torch::Tensor tilePixelCumsum,
                                 torch::Tensor pixelMap);

    /// @brief 3D tile offset tensor [C, TH, TW].
    const torch::Tensor &tileOffsets() const;

    /// @brief Flattened Gaussian IDs sorted by tile [totalIntersections].
    const torch::Tensor &tileGaussianIds() const;

    /// @brief Side length of a square tile in pixels.
    uint32_t tileSize() const;

    /// @brief Indices of active tiles [AT].
    const torch::Tensor &activeTiles() const;

    /// @brief Per-tile bitmask of active pixels [AT, wordsPerTile].
    const torch::Tensor &tilePixelMask() const;

    /// @brief Cumulative active-pixel counts per tile [AT].
    const torch::Tensor &tilePixelCumsum() const;

    /// @brief Map from cumulative pixel index to output index [AP].
    const torch::Tensor &pixelMap() const;

    /// @brief Total number of Gaussian-tile intersections.
    int64_t totalIntersections() const;

#if defined(__CUDACC__)
    /// @brief Device-side accessor for sparse-pixel rasterization with dense tile offsets.
    ///
    /// Only tiles listed in activeTiles are launched as CUDA blocks. Pixel activity within each
    /// tile is determined by a bitmask. The Gaussian range for each tile is looked up via the
    /// dense 3D tile offsets [C, TH, TW]. Pixel indices are computed through the pixel map.
    struct Accessor {
      private:
        Accessor(fvdb::TorchRAcc64<int32_t, 3> tileOffsetsIn,
                 fvdb::TorchRAcc64<int32_t, 1> tileGaussianIdsIn,
                 uint32_t numCamerasIn,
                 fvdb::TorchRAcc64<int32_t, 1> activeTilesIn,
                 fvdb::TorchRAcc64<uint64_t, 2> tilePixelMaskIn,
                 fvdb::TorchRAcc64<int64_t, 1> tilePixelCumsumIn,
                 fvdb::TorchRAcc64<int64_t, 1> pixelMapIn,
                 const RenderWindow2D &renderWindow,
                 uint32_t tileSizeIn,
                 uint32_t blockOffsetIn,
                 int32_t totalIntersectionsIn)
            : mTileOffsets(tileOffsetsIn), mTileGaussianIds(tileGaussianIdsIn),
              mNumCameras(numCamerasIn), mActiveTiles(activeTilesIn),
              mTilePixelMask(tilePixelMaskIn), mTilePixelCumsum(tilePixelCumsumIn),
              mPixelMap(pixelMapIn), mNumActiveTiles(static_cast<uint32_t>(activeTilesIn.size(0))),
              mRenderWidth(renderWindow.width), mRenderHeight(renderWindow.height),
              mTileOriginW(renderWindow.originW / tileSizeIn),
              mTileOriginH(renderWindow.originH / tileSizeIn), mTileSize(tileSizeIn),
              mBlockOffset(blockOffsetIn), mTotalIntersections(totalIntersectionsIn) {
            mNumTilesW = renderWindow.tileExtentW(mTileSize);
            mNumTilesH = renderWindow.tileExtentH(mTileSize);
        }

        friend class SparseDenseTileIntersections;
        template <typename CallbackT>
        friend inline void
        dispatchTileIntersectionsAccessor(const torch::Tensor &tileOffsets,
                                          const torch::Tensor &tileGaussianIds,
                                          const RenderWindow2D &renderWindow,
                                          uint32_t tileSize,
                                          uint32_t blockOffset,
                                          CallbackT &&callback,
                                          const std::optional<torch::Tensor> &activeTiles,
                                          const std::optional<torch::Tensor> &tilePixelMask,
                                          const std::optional<torch::Tensor> &tilePixelCumsum,
                                          const std::optional<torch::Tensor> &pixelMap);

      public:
        using ActivePixelBlockScanT = cub::BlockScan<uint32_t, 16, cub::BLOCK_SCAN_RAKING, 16>;

        /// @brief Shared memory scratch space for the CUB block scan used to compute
        ///        per-thread active-pixel indices within a tile.
        struct ActivePixelScratch {
            typename ActivePixelBlockScanT::TempStorage tempStorage;
        };

        /// @brief Number of active tiles to launch.
        __host__ __device__ inline uint32_t
        numActiveTiles() const {
            return mNumActiveTiles;
        }

        /// @brief Number of tile columns in the render window.
        __host__ __device__ inline uint32_t
        numTilesW() const {
            return mNumTilesW;
        }

        /// @brief Number of tile rows in the render window.
        __host__ __device__ inline uint32_t
        numTilesH() const {
            return mNumTilesH;
        }

        /// @brief Number of cameras (derived from the dense tile offsets).
        __host__ __device__ inline uint32_t
        numCameras() const {
            return mNumCameras;
        }

        /// @brief Return the known camera count, ignoring the fallback.
        __host__ __device__ inline uint32_t
        cameraCount(uint32_t fallback) const {
            (void)fallback;
            return mNumCameras;
        }

        /// @brief Side length of a square tile in pixels.
        __host__ __device__ inline uint32_t
        tileSize() const {
            return mTileSize;
        }

        /// @brief Total number of active (sparse) pixels across all tiles.
        __host__ __device__ inline uint32_t
        pixelCount() const {
            return mPixelMap.size(0);
        }

        /// @brief Sparse modes produce one output tensor per camera.
        __host__ __device__ inline uint32_t
        outputNumTensors(uint32_t numCameras) const {
            return numCameras;
        }

        /// @brief Total active pixel count for output validation.
        __host__ __device__ inline uint64_t
        outputPixelCount(uint32_t numCameras, uint32_t renderHeight, uint32_t renderWidth) const {
            (void)numCameras;
            (void)renderHeight;
            (void)renderWidth;
            return pixelCount();
        }

        /// @brief Translate a per-device block ordinal to a global tile ordinal.
        __host__ __device__ inline uint32_t
        globalTileOrdinal(uint32_t blockOrdinal) const {
            return blockOrdinal + mBlockOffset;
        }

        /// @brief Look up the Gaussian index at a given intersection position.
        __device__ inline int32_t
        gaussianIdAt(int32_t intersectionIdx) const {
            return mTileGaussianIds[intersectionIdx];
        }

        /// @brief Get the [first, last) Gaussian intersection range for a tile via 3D lookup.
        __device__ inline cuda::std::tuple<int32_t, int32_t>
        tileGaussianRange(uint32_t cameraId, uint32_t tileRow, uint32_t tileCol) const {
            const int32_t firstGaussianIdInBlock = mTileOffsets[cameraId][tileRow][tileCol];
            auto [nextTileRow, nextTileCol]      = (tileCol < mNumTilesW - 1)
                                                       ? cuda::std::make_tuple(tileRow, tileCol + 1)
                                                       : cuda::std::make_tuple(tileRow + 1, 0u);
            const int32_t lastGaussianIdInBlock =
                ((cameraId == mNumCameras - 1) && (nextTileRow == mNumTilesH))
                    ? mTotalIntersections
                    : mTileOffsets[cameraId][nextTileRow][nextTileCol];
            return {firstGaussianIdInBlock, lastGaussianIdInBlock};
        }

        /// @brief Get the Gaussian range for a tile from a block ordinal (delegates to 3D lookup).
        __device__ inline cuda::std::tuple<int32_t, int32_t>
        tileGaussianRangeFromBlock(uint32_t blockOrdinal,
                                   uint32_t cameraId,
                                   uint32_t tileRow,
                                   uint32_t tileCol) const {
            (void)blockOrdinal;
            return tileGaussianRange(cameraId, tileRow, tileCol);
        }

        /// @brief Decompose a global tile ordinal into (cameraId, localTileId) using activeTiles.
        __device__ inline cuda::std::tuple<int32_t, int32_t>
        cameraTileId(uint32_t tileOrdinal, uint32_t numTilesW, uint32_t numTilesH) const {
            const int32_t globalTile = mActiveTiles[tileOrdinal];
            const int32_t tileCount  = numTilesW * numTilesH;
            return {globalTile / tileCount, globalTile % tileCount};
        }

        /// @brief Get (cameraId, tileRow, tileCol, pixelRow, pixelCol) from tile and thread
        /// indices.
        __device__ inline cuda::std::tuple<int32_t, int32_t, int32_t, uint32_t, uint32_t>
        coordinates(uint32_t tileOrdinal,
                    uint32_t threadX,
                    uint32_t threadY,
                    uint32_t numTilesW,
                    uint32_t numTilesH,
                    uint32_t tileOriginW,
                    uint32_t tileOriginH,
                    uint32_t tileSize) const {
            (void)tileOriginW;
            (void)tileOriginH;
            const auto [cameraId, tileId] = cameraTileId(tileOrdinal, numTilesW, numTilesH);
            const int32_t tileRow         = tileId / numTilesW;
            const int32_t tileCol         = tileId % numTilesW;
            const uint32_t row            = tileRow * tileSize + threadY;
            const uint32_t col            = tileCol * tileSize + threadX;
            return {cameraId, tileRow, tileCol, row, col};
        }

        /// @brief Convenience overload using stored tile layout and the block ordinal.
        __device__ inline cuda::std::tuple<int32_t, int32_t, int32_t, uint32_t, uint32_t>
        coordinates(uint32_t blockOrdinal, uint32_t threadX, uint32_t threadY) const {
            return coordinates(globalTileOrdinal(blockOrdinal),
                               threadX,
                               threadY,
                               mNumTilesW,
                               mNumTilesH,
                               mTileOriginW,
                               mTileOriginH,
                               mTileSize);
        }

        /// @brief Check pixel activity via bitmask lookup.
        __device__ inline bool
        tilePixelActive(uint32_t tileSideLength,
                        uint32_t tileOrdinal,
                        uint32_t iInTile,
                        uint32_t jInTile,
                        uint32_t row,
                        uint32_t col,
                        uint32_t renderWidth,
                        uint32_t renderHeight) const {
            (void)row;
            (void)col;
            (void)renderWidth;
            (void)renderHeight;
            const uint32_t bitIndex = iInTile * tileSideLength + jInTile;
            const uint32_t wordIdx  = bitIndex / 64;
            const uint32_t bitIdx   = bitIndex % 64;
            return mTilePixelMask[tileOrdinal][wordIdx] & (1ull << bitIdx);
        }

        /// @brief Convenience overload using stored render dimensions and block ordinal.
        __device__ inline bool
        tilePixelActiveFromBlock(uint32_t blockOrdinal,
                                 uint32_t iInTile,
                                 uint32_t jInTile,
                                 uint32_t row,
                                 uint32_t col) const {
            return tilePixelActive(mTileSize,
                                   globalTileOrdinal(blockOrdinal),
                                   iInTile,
                                   jInTile,
                                   row,
                                   col,
                                   mRenderWidth,
                                   mRenderHeight);
        }

        /// @brief Compute the within-tile active-pixel index via CUB block scan.
        __device__ inline cuda::std::tuple<bool, uint32_t>
        activePixelIndexFromBlock(uint32_t blockOrdinal,
                                  uint32_t iInTile,
                                  uint32_t jInTile,
                                  uint32_t row,
                                  uint32_t col,
                                  ActivePixelScratch &scratch) const {
            uint32_t index = 0;
            const bool pixelActive =
                tilePixelActiveFromBlock(blockOrdinal, iInTile, jInTile, row, col);
            ActivePixelBlockScanT(scratch.tempStorage).ExclusiveSum(pixelActive, index);
            __syncthreads();
            return {pixelActive, index};
        }

        /// @brief Compute sparse pixel index via pixel map.
        __device__ inline uint64_t
        pixelIndex(uint64_t cameraId,
                   uint64_t row,
                   uint64_t col,
                   uint32_t activePixelIndex,
                   uint32_t renderWidth,
                   uint32_t renderHeight,
                   uint32_t tileOrdinal) const {
            (void)cameraId;
            (void)row;
            (void)col;
            (void)renderWidth;
            (void)renderHeight;
            return sparsePixelIndex(tileOrdinal, activePixelIndex);
        }

        /// @brief Convenience overload using stored render dimensions and block ordinal.
        __device__ inline uint64_t
        pixelIndexFromBlock(uint64_t cameraId,
                            uint64_t row,
                            uint64_t col,
                            uint32_t activePixelIndex,
                            uint32_t blockOrdinal) const {
            return pixelIndex(cameraId,
                              row,
                              col,
                              activePixelIndex,
                              mRenderWidth,
                              mRenderHeight,
                              globalTileOrdinal(blockOrdinal));
        }

      private:
        __device__ inline int64_t
        sparsePixelIndex(uint32_t tileOrdinal, uint32_t activePixelIndex) const {
            const auto tilePixelCumsumValue =
                tileOrdinal > 0 ? mTilePixelCumsum[tileOrdinal - 1] : 0;
            return mPixelMap[tilePixelCumsumValue + activePixelIndex];
        }

        fvdb::TorchRAcc64<int32_t, 3> mTileOffsets;     // [C, TH, TW]
        fvdb::TorchRAcc64<int32_t, 1> mTileGaussianIds; // [totalIntersections]
        uint32_t mNumCameras;
        fvdb::TorchRAcc64<int32_t, 1> mActiveTiles;     // [AT]
        fvdb::TorchRAcc64<uint64_t, 2> mTilePixelMask;  // [AT, wordsPerTile]
        fvdb::TorchRAcc64<int64_t, 1> mTilePixelCumsum; // [AT]
        fvdb::TorchRAcc64<int64_t, 1> mPixelMap;        // [AP]
        uint32_t mNumActiveTiles;
        uint32_t mNumTilesH;
        uint32_t mNumTilesW;
        uint32_t mRenderWidth;
        uint32_t mRenderHeight;
        uint32_t mTileOriginW;
        uint32_t mTileOriginH;
        uint32_t mTileSize;
        uint32_t mBlockOffset;
        int32_t mTotalIntersections;
    };

    /// @brief Create a device accessor from this host-side object.
    Accessor accessor(const RenderWindow2D &renderWindow, uint32_t blockOffset) const;
#endif

  private:
    torch::Tensor mTileOffsets;
    torch::Tensor mTileGaussianIds;
    torch::Tensor mActiveTiles;
    torch::Tensor mTilePixelMask;
    torch::Tensor mTilePixelCumsum;
    torch::Tensor mPixelMap;
    uint32_t mTileSize = 0;
};

/// @brief Tile intersection data for sparse-pixel rasterization with sparse (1D) tile offsets
///        (Mode 3).
///
/// Only a subset of pixels are rendered (determined by a bitmask per tile), and tile offsets are
/// stored as a 1D tensor [AT + 1] indexed by active-tile ordinal. The camera count is not known
/// from the tile offsets and must be supplied externally as a fallback. Active tiles are
/// identified by an explicit list. Pixel output is indexed through a pixel map.
///
/// This type is used when both the tile layout and the pixel selection are sparse. The 1D tile
/// offsets are indexed directly by the active-tile ordinal rather than by (camera, row, col).
class SparseTileIntersections {
  public:
    SparseTileIntersections() = default;

    /// @brief Compute sparse tile intersections from projected Gaussian parameters.
    SparseTileIntersections(const torch::Tensor &means2d,
                            const torch::Tensor &radii,
                            const torch::Tensor &depths,
                            const torch::Tensor &tileMask,
                            const torch::Tensor &activeTiles,
                            const at::optional<torch::Tensor> &cameraIds,
                            uint32_t tileSize,
                            uint32_t imageHeight,
                            uint32_t imageWidth);

    /// @brief Construct from precomputed tensors.
    SparseTileIntersections(torch::Tensor tileOffsets,
                            torch::Tensor tileGaussianIds,
                            uint32_t tileSize,
                            const std::optional<torch::Tensor> &activeTiles     = std::nullopt,
                            const std::optional<torch::Tensor> &tilePixelMask   = std::nullopt,
                            const std::optional<torch::Tensor> &tilePixelCumsum = std::nullopt,
                            const std::optional<torch::Tensor> &pixelMap        = std::nullopt);

    /// @brief 1D tile offset tensor [AT + 1].
    const torch::Tensor &tileOffsets() const;

    /// @brief Flattened Gaussian IDs sorted by tile [totalIntersections].
    const torch::Tensor &tileGaussianIds() const;

    /// @brief Side length of a square tile in pixels.
    uint32_t tileSize() const;

    /// @brief Indices of active tiles [AT].
    const torch::Tensor &activeTiles() const;

    /// @brief Per-tile bitmask of active pixels [AT, wordsPerTile].
    const torch::Tensor &tilePixelMask() const;

    /// @brief Cumulative active-pixel counts per tile [AT].
    const torch::Tensor &tilePixelCumsum() const;

    /// @brief Map from cumulative pixel index to output index [AP].
    const torch::Tensor &pixelMap() const;

    /// @brief Total number of Gaussian-tile intersections.
    int64_t totalIntersections() const;

#if defined(__CUDACC__)
    /// @brief Device-side accessor for sparse-pixel rasterization with sparse (1D) tile offsets.
    ///
    /// Only tiles listed in activeTiles are launched as CUDA blocks. Pixel activity within each
    /// tile is determined by a bitmask. The Gaussian range for each tile is looked up via the
    /// 1D tile offsets [AT + 1] indexed by the active-tile ordinal. The camera count is not
    /// available from tile offsets and must be supplied as a fallback. Pixel indices are computed
    /// through the pixel map.
    struct Accessor {
      private:
        Accessor(fvdb::TorchRAcc64<int32_t, 1> tileOffsetsIn,
                 fvdb::TorchRAcc64<int32_t, 1> tileGaussianIdsIn,
                 fvdb::TorchRAcc64<int32_t, 1> activeTilesIn,
                 fvdb::TorchRAcc64<uint64_t, 2> tilePixelMaskIn,
                 fvdb::TorchRAcc64<int64_t, 1> tilePixelCumsumIn,
                 fvdb::TorchRAcc64<int64_t, 1> pixelMapIn,
                 const RenderWindow2D &renderWindow,
                 uint32_t tileSizeIn,
                 uint32_t blockOffsetIn,
                 int32_t totalIntersectionsIn)
            : mTileOffsets(tileOffsetsIn), mTileGaussianIds(tileGaussianIdsIn),
              mActiveTiles(activeTilesIn), mTilePixelMask(tilePixelMaskIn),
              mTilePixelCumsum(tilePixelCumsumIn), mPixelMap(pixelMapIn),
              mNumActiveTiles(static_cast<uint32_t>(activeTilesIn.size(0))),
              mRenderWidth(renderWindow.width), mRenderHeight(renderWindow.height),
              mTileOriginW(renderWindow.originW / tileSizeIn),
              mTileOriginH(renderWindow.originH / tileSizeIn), mTileSize(tileSizeIn),
              mBlockOffset(blockOffsetIn), mTotalIntersections(totalIntersectionsIn) {
            mNumTilesW = renderWindow.tileExtentW(mTileSize);
            mNumTilesH = renderWindow.tileExtentH(mTileSize);
        }

        friend class SparseTileIntersections;
        template <typename CallbackT>
        friend inline void
        dispatchTileIntersectionsAccessor(const torch::Tensor &tileOffsets,
                                          const torch::Tensor &tileGaussianIds,
                                          const RenderWindow2D &renderWindow,
                                          uint32_t tileSize,
                                          uint32_t blockOffset,
                                          CallbackT &&callback,
                                          const std::optional<torch::Tensor> &activeTiles,
                                          const std::optional<torch::Tensor> &tilePixelMask,
                                          const std::optional<torch::Tensor> &tilePixelCumsum,
                                          const std::optional<torch::Tensor> &pixelMap);

      public:
        using ActivePixelBlockScanT = cub::BlockScan<uint32_t, 16, cub::BLOCK_SCAN_RAKING, 16>;

        /// @brief Shared memory scratch space for the CUB block scan used to compute
        ///        per-thread active-pixel indices within a tile.
        struct ActivePixelScratch {
            typename ActivePixelBlockScanT::TempStorage tempStorage;
        };

        /// @brief Number of active tiles to launch.
        __host__ __device__ inline uint32_t
        numActiveTiles() const {
            return mNumActiveTiles;
        }

        /// @brief Number of tile columns in the render window.
        __host__ __device__ inline uint32_t
        numTilesW() const {
            return mNumTilesW;
        }

        /// @brief Number of tile rows in the render window.
        __host__ __device__ inline uint32_t
        numTilesH() const {
            return mNumTilesH;
        }

        /// @brief Camera count is unknown from 1D offsets; always returns 0.
        __host__ __device__ inline uint32_t
        numCameras() const {
            return 0;
        }

        /// @brief Camera count is unknown from 1D offsets; returns the caller-supplied fallback.
        __host__ __device__ inline uint32_t
        cameraCount(uint32_t fallback) const {
            return fallback;
        }

        /// @brief Side length of a square tile in pixels.
        __host__ __device__ inline uint32_t
        tileSize() const {
            return mTileSize;
        }

        /// @brief Total number of active (sparse) pixels across all tiles.
        __host__ __device__ inline uint32_t
        pixelCount() const {
            return mPixelMap.size(0);
        }

        /// @brief Sparse modes produce one output tensor per camera.
        __host__ __device__ inline uint32_t
        outputNumTensors(uint32_t numCameras) const {
            return numCameras;
        }

        /// @brief Total active pixel count for output validation.
        __host__ __device__ inline uint64_t
        outputPixelCount(uint32_t numCameras, uint32_t renderHeight, uint32_t renderWidth) const {
            (void)numCameras;
            (void)renderHeight;
            (void)renderWidth;
            return pixelCount();
        }

        /// @brief Translate a per-device block ordinal to a global tile ordinal.
        __host__ __device__ inline uint32_t
        globalTileOrdinal(uint32_t blockOrdinal) const {
            return blockOrdinal + mBlockOffset;
        }

        /// @brief Look up the Gaussian index at a given intersection position.
        __device__ inline int32_t
        gaussianIdAt(int32_t intersectionIdx) const {
            return mTileGaussianIds[intersectionIdx];
        }

        /// @brief Get the [first, last) Gaussian intersection range via 1D ordinal lookup.
        __device__ inline cuda::std::tuple<int32_t, int32_t>
        tileGaussianRange(uint32_t tileOrdinal) const {
            return {mTileOffsets[tileOrdinal], mTileOffsets[tileOrdinal + 1]};
        }

        /// @brief Get the Gaussian range for a tile from a block ordinal (uses 1D ordinal lookup).
        __device__ inline cuda::std::tuple<int32_t, int32_t>
        tileGaussianRangeFromBlock(uint32_t blockOrdinal,
                                   uint32_t cameraId,
                                   uint32_t tileRow,
                                   uint32_t tileCol) const {
            (void)cameraId;
            (void)tileRow;
            (void)tileCol;
            return tileGaussianRange(globalTileOrdinal(blockOrdinal));
        }

        /// @brief Decompose a global tile ordinal into (cameraId, localTileId) using activeTiles.
        __device__ inline cuda::std::tuple<int32_t, int32_t>
        cameraTileId(uint32_t tileOrdinal, uint32_t numTilesW, uint32_t numTilesH) const {
            const int32_t globalTile = mActiveTiles[tileOrdinal];
            const int32_t tileCount  = numTilesW * numTilesH;
            return {globalTile / tileCount, globalTile % tileCount};
        }

        /// @brief Get (cameraId, tileRow, tileCol, pixelRow, pixelCol) from tile and thread
        /// indices.
        __device__ inline cuda::std::tuple<int32_t, int32_t, int32_t, uint32_t, uint32_t>
        coordinates(uint32_t tileOrdinal,
                    uint32_t threadX,
                    uint32_t threadY,
                    uint32_t numTilesW,
                    uint32_t numTilesH,
                    uint32_t tileOriginW,
                    uint32_t tileOriginH,
                    uint32_t tileSize) const {
            (void)tileOriginW;
            (void)tileOriginH;
            const auto [cameraId, tileId] = cameraTileId(tileOrdinal, numTilesW, numTilesH);
            const int32_t tileRow         = tileId / numTilesW;
            const int32_t tileCol         = tileId % numTilesW;
            const uint32_t row            = tileRow * tileSize + threadY;
            const uint32_t col            = tileCol * tileSize + threadX;
            return {cameraId, tileRow, tileCol, row, col};
        }

        /// @brief Convenience overload using stored tile layout and the block ordinal.
        __device__ inline cuda::std::tuple<int32_t, int32_t, int32_t, uint32_t, uint32_t>
        coordinates(uint32_t blockOrdinal, uint32_t threadX, uint32_t threadY) const {
            return coordinates(globalTileOrdinal(blockOrdinal),
                               threadX,
                               threadY,
                               mNumTilesW,
                               mNumTilesH,
                               mTileOriginW,
                               mTileOriginH,
                               mTileSize);
        }

        /// @brief Check pixel activity via bitmask lookup.
        __device__ inline bool
        tilePixelActive(uint32_t tileSideLength,
                        uint32_t tileOrdinal,
                        uint32_t iInTile,
                        uint32_t jInTile,
                        uint32_t row,
                        uint32_t col,
                        uint32_t renderWidth,
                        uint32_t renderHeight) const {
            (void)row;
            (void)col;
            (void)renderWidth;
            (void)renderHeight;
            const uint32_t bitIndex = iInTile * tileSideLength + jInTile;
            const uint32_t wordIdx  = bitIndex / 64;
            const uint32_t bitIdx   = bitIndex % 64;
            return mTilePixelMask[tileOrdinal][wordIdx] & (1ull << bitIdx);
        }

        /// @brief Convenience overload using stored render dimensions and block ordinal.
        __device__ inline bool
        tilePixelActiveFromBlock(uint32_t blockOrdinal,
                                 uint32_t iInTile,
                                 uint32_t jInTile,
                                 uint32_t row,
                                 uint32_t col) const {
            return tilePixelActive(mTileSize,
                                   globalTileOrdinal(blockOrdinal),
                                   iInTile,
                                   jInTile,
                                   row,
                                   col,
                                   mRenderWidth,
                                   mRenderHeight);
        }

        /// @brief Compute the within-tile active-pixel index via CUB block scan.
        __device__ inline cuda::std::tuple<bool, uint32_t>
        activePixelIndexFromBlock(uint32_t blockOrdinal,
                                  uint32_t iInTile,
                                  uint32_t jInTile,
                                  uint32_t row,
                                  uint32_t col,
                                  ActivePixelScratch &scratch) const {
            uint32_t index = 0;
            const bool pixelActive =
                tilePixelActiveFromBlock(blockOrdinal, iInTile, jInTile, row, col);
            ActivePixelBlockScanT(scratch.tempStorage).ExclusiveSum(pixelActive, index);
            __syncthreads();
            return {pixelActive, index};
        }

        /// @brief Compute sparse pixel index via pixel map.
        __device__ inline uint64_t
        pixelIndex(uint64_t cameraId,
                   uint64_t row,
                   uint64_t col,
                   uint32_t activePixelIndex,
                   uint32_t renderWidth,
                   uint32_t renderHeight,
                   uint32_t tileOrdinal) const {
            (void)cameraId;
            (void)row;
            (void)col;
            (void)renderWidth;
            (void)renderHeight;
            return sparsePixelIndex(tileOrdinal, activePixelIndex);
        }

        /// @brief Convenience overload using stored render dimensions and block ordinal.
        __device__ inline uint64_t
        pixelIndexFromBlock(uint64_t cameraId,
                            uint64_t row,
                            uint64_t col,
                            uint32_t activePixelIndex,
                            uint32_t blockOrdinal) const {
            return pixelIndex(cameraId,
                              row,
                              col,
                              activePixelIndex,
                              mRenderWidth,
                              mRenderHeight,
                              globalTileOrdinal(blockOrdinal));
        }

      private:
        __device__ inline int64_t
        sparsePixelIndex(uint32_t tileOrdinal, uint32_t activePixelIndex) const {
            const auto tilePixelCumsumValue =
                tileOrdinal > 0 ? mTilePixelCumsum[tileOrdinal - 1] : 0;
            return mPixelMap[tilePixelCumsumValue + activePixelIndex];
        }

        fvdb::TorchRAcc64<int32_t, 1> mTileOffsets;     // [AT + 1]
        fvdb::TorchRAcc64<int32_t, 1> mTileGaussianIds; // [totalIntersections]
        fvdb::TorchRAcc64<int32_t, 1> mActiveTiles;     // [AT]
        fvdb::TorchRAcc64<uint64_t, 2> mTilePixelMask;  // [AT, wordsPerTile]
        fvdb::TorchRAcc64<int64_t, 1> mTilePixelCumsum; // [AT]
        fvdb::TorchRAcc64<int64_t, 1> mPixelMap;        // [AP]
        uint32_t mNumActiveTiles;
        uint32_t mNumTilesH;
        uint32_t mNumTilesW;
        uint32_t mRenderWidth;
        uint32_t mRenderHeight;
        uint32_t mTileOriginW;
        uint32_t mTileOriginH;
        uint32_t mTileSize;
        uint32_t mBlockOffset;
        int32_t mTotalIntersections;
    };

    /// @brief Create a device accessor from this host-side object.
    Accessor accessor(const RenderWindow2D &renderWindow, uint32_t blockOffset) const;
#endif

  private:
    torch::Tensor mTileOffsets;
    torch::Tensor mTileGaussianIds;
    torch::Tensor mActiveTiles;
    torch::Tensor mTilePixelMask;
    torch::Tensor mTilePixelCumsum;
    torch::Tensor mPixelMap;
    uint32_t mTileSize = 0;
};

#if defined(__CUDACC__)
/// @brief Dispatch a callback with the appropriate tile intersection accessor type.
///
/// Selects one of three accessor types based on the inputs:
///   - No activeTiles, 3D tileOffsets  -> DenseTileIntersections::Accessor       (Mode 1)
///   - activeTiles present, 3D offsets -> SparseDenseTileIntersections::Accessor  (Mode 2)
///   - activeTiles present, 1D offsets -> SparseTileIntersections::Accessor       (Mode 3)
///
/// The callback receives the accessor by value and is typically a lambda that constructs
/// kernel args and launches the rasterization kernel. The accessor type is resolved at compile
/// time through the auto parameter in the callback.
template <typename CallbackT>
inline void
dispatchTileIntersectionsAccessor(const torch::Tensor &tileOffsets,
                                  const torch::Tensor &tileGaussianIds,
                                  const RenderWindow2D &renderWindow,
                                  uint32_t tileSize,
                                  uint32_t blockOffset,
                                  CallbackT &&callback,
                                  const std::optional<torch::Tensor> &activeTiles,
                                  const std::optional<torch::Tensor> &tilePixelMask,
                                  const std::optional<torch::Tensor> &tilePixelCumsum,
                                  const std::optional<torch::Tensor> &pixelMap) {
    if (activeTiles.has_value()) {
        TORCH_CHECK_VALUE(tilePixelMask.has_value(),
                          "tilePixelMask must be provided for sparse tile intersections");
        TORCH_CHECK_VALUE(tilePixelCumsum.has_value(),
                          "tilePixelCumsum must be provided for sparse tile intersections");
        TORCH_CHECK_VALUE(pixelMap.has_value(),
                          "pixelMap must be provided for sparse tile intersections");
        auto tileGaussianIdsAcc =
            tileGaussianIds.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>();
        auto activeTilesAcc =
            activeTiles.value().packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>();
        auto tilePixelMaskAcc =
            tilePixelMask.value().packed_accessor64<uint64_t, 2, torch::RestrictPtrTraits>();
        auto tilePixelCumsumAcc =
            tilePixelCumsum.value().packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>();
        auto pixelMapAcc =
            pixelMap.value().packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>();
        auto totalIntersections = static_cast<int32_t>(tileGaussianIds.size(0));

        if (tileOffsets.dim() == 3) {
            // Mode 2: sparse pixels, dense tile offsets
            std::forward<CallbackT>(callback)(SparseDenseTileIntersections::Accessor(
                tileOffsets.packed_accessor64<int32_t, 3, torch::RestrictPtrTraits>(),
                tileGaussianIdsAcc,
                static_cast<uint32_t>(tileOffsets.size(0)),
                activeTilesAcc,
                tilePixelMaskAcc,
                tilePixelCumsumAcc,
                pixelMapAcc,
                renderWindow,
                tileSize,
                blockOffset,
                totalIntersections));
        } else {
            // Mode 3: sparse pixels, sparse tile offsets
            TORCH_CHECK_VALUE(
                tileOffsets.dim() == 1,
                "tileOffsets must be 1D [AT+1] for sparse tile intersections, got dim=",
                tileOffsets.dim());
            std::forward<CallbackT>(callback)(SparseTileIntersections::Accessor(
                tileOffsets.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                tileGaussianIdsAcc,
                activeTilesAcc,
                tilePixelMaskAcc,
                tilePixelCumsumAcc,
                pixelMapAcc,
                renderWindow,
                tileSize,
                blockOffset,
                totalIntersections));
        }
        return;
    }
    // Mode 1: dense tiles, dense pixels
    TORCH_CHECK_VALUE(tileOffsets.dim() == 3,
                      "tileOffsets must be 3D [C, TH, TW] for dense tile intersections, got dim=",
                      tileOffsets.dim());
    std::forward<CallbackT>(callback)(DenseTileIntersections::Accessor(
        tileOffsets.packed_accessor64<int32_t, 3, torch::RestrictPtrTraits>(),
        tileGaussianIds.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
        static_cast<uint32_t>(tileOffsets.size(0)),
        renderWindow,
        tileSize,
        blockOffset,
        static_cast<int32_t>(tileGaussianIds.size(0))));
}
#endif

/// @brief Build dense tile intersections for rasterization.
template <torch::DeviceType>
DenseTileIntersections
dispatchDenseTileIntersections(const torch::Tensor &means2d,                 // [C, N, 2] or [M, 2]
                               const torch::Tensor &radii,                   // [C, N] or [M]
                               const torch::Tensor &depths,                  // [C, N] or [M]
                               const at::optional<torch::Tensor> &cameraIds, // NULL or [M]
                               const uint32_t tileSize,
                               const uint32_t imageHeight,
                               const uint32_t imageWidth);

/// @brief Build sparse tile intersections for sparse rasterization.
template <torch::DeviceType>
SparseTileIntersections
dispatchSparseTileIntersections(const torch::Tensor &means2d,                 // [C, N, 2] or [M, 2]
                                const torch::Tensor &radii,                   // [C, N] or [M]
                                const torch::Tensor &depths,                  // [C, N] or [M]
                                const torch::Tensor &tileMask,                // [C, H, W]
                                const torch::Tensor &activeTiles,             // [num_active_tiles]
                                const at::optional<torch::Tensor> &cameraIds, // NULL or [M]
                                const uint32_t tileSize,
                                const uint32_t imageHeight,
                                const uint32_t imageWidth);

/// @brief Compute the intersection of 2D Gaussians with image tiles for efficient rasterization.
/// Legacy tuple API retained for compatibility.
template <torch::DeviceType>
std::tuple<torch::Tensor, torch::Tensor>
dispatchGaussianTileIntersection(const torch::Tensor &means2d, // [C, N, 2] or [M, 2]
                                 const torch::Tensor &radii,   // [C, N] or [M]
                                 const torch::Tensor &depths,  // [C, N] or [M]
                                 const at::optional<torch::Tensor> &cameraIds, // NULL or [M]
                                 const uint32_t tileSize,
                                 const uint32_t imageHeight,
                                 const uint32_t imageWidth);

/// @brief Compute the intersection of 2D Gaussians with image tiles for sparse rendering.
/// Legacy tuple API retained for compatibility.
template <torch::DeviceType>
std::tuple<torch::Tensor, torch::Tensor>
dispatchGaussianSparseTileIntersection(const torch::Tensor &means2d,     // [C, N, 2] or [M, 2]
                                       const torch::Tensor &radii,       // [C, N] or [M]
                                       const torch::Tensor &depths,      // [C, N] or [M]
                                       const torch::Tensor &tileMask,    // [C, H, W]
                                       const torch::Tensor &activeTiles, // [num_active_tiles]
                                       const at::optional<torch::Tensor> &cameraIds, // NULL or [M]
                                       const uint32_t tileSize,
                                       const uint32_t imageHeight,
                                       const uint32_t imageWidth);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_GSPLAT_GAUSSIANTILEINTERSECTION_H
