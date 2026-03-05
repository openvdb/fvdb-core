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

class DenseTileIntersections {
  public:
    DenseTileIntersections() = default;
    DenseTileIntersections(const torch::Tensor &means2d,
                           const torch::Tensor &radii,
                           const torch::Tensor &depths,
                           const at::optional<torch::Tensor> &cameraIds,
                           uint32_t tileSize,
                           uint32_t imageHeight,
                           uint32_t imageWidth);
    DenseTileIntersections(torch::Tensor tileOffsets,
                           torch::Tensor tileGaussianIds,
                           uint32_t tileSize);

    const torch::Tensor &tileOffsets() const;

    const torch::Tensor &tileGaussianIds() const;

    uint32_t tileSize() const;

    int64_t totalIntersections() const;

#if defined(__CUDACC__)
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
        struct ActivePixelScratch {};

        __host__ __device__ inline uint32_t
        numTilesW() const {
            return mNumTilesW;
        }
        __host__ __device__ inline uint32_t
        numTilesH() const {
            return mNumTilesH;
        }
        __host__ __device__ inline uint32_t
        numCameras() const {
            return mNumCameras;
        }
        __host__ __device__ inline uint32_t
        cameraCount(uint32_t fallback) const {
            (void)fallback;
            return mNumCameras;
        }
        __host__ __device__ inline uint32_t
        tileSize() const {
            return mTileSize;
        }
        __host__ __device__ inline uint32_t
        numActiveTiles() const {
            return mNumCameras * mNumTilesW * mNumTilesH;
        }
        __host__ __device__ inline uint32_t
        pixelCount() const {
            return mNumCameras * mRenderWidth * mRenderHeight;
        }
        __host__ __device__ inline uint32_t
        outputNumTensors(uint32_t numCameras) const {
            (void)numCameras;
            return 1;
        }
        __host__ __device__ inline uint64_t
        outputPixelCount(uint32_t numCameras, uint32_t renderHeight, uint32_t renderWidth) const {
            return static_cast<uint64_t>(numCameras) * renderHeight * renderWidth;
        }

        __host__ __device__ inline uint32_t
        globalTileOrdinal(uint32_t blockOrdinal) const {
            return blockOrdinal + mBlockOffset;
        }

        __device__ inline int32_t
        gaussianIdAt(int32_t intersectionIdx) const {
            return mTileGaussianIds[intersectionIdx];
        }

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

        __device__ inline cuda::std::tuple<int32_t, int32_t>
        tileGaussianRangeFromBlock(uint32_t blockOrdinal,
                                   uint32_t cameraId,
                                   uint32_t tileRow,
                                   uint32_t tileCol) const {
            (void)blockOrdinal;
            return tileGaussianRange(cameraId, tileRow, tileCol);
        }

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

    Accessor accessor(const RenderWindow2D &renderWindow, uint32_t blockOffset) const;
#endif

  private:
    torch::Tensor mTileOffsets;
    torch::Tensor mTileGaussianIds;
    uint32_t mTileSize = 0;
};

class SparseTileIntersections {
  public:
    SparseTileIntersections() = default;
    SparseTileIntersections(const torch::Tensor &means2d,
                            const torch::Tensor &radii,
                            const torch::Tensor &depths,
                            const torch::Tensor &tileMask,
                            const torch::Tensor &activeTiles,
                            const at::optional<torch::Tensor> &cameraIds,
                            uint32_t tileSize,
                            uint32_t imageHeight,
                            uint32_t imageWidth);
    SparseTileIntersections(torch::Tensor tileOffsets,
                            torch::Tensor tileGaussianIds,
                            uint32_t tileSize,
                            const std::optional<torch::Tensor> &activeTiles     = std::nullopt,
                            const std::optional<torch::Tensor> &tilePixelMask   = std::nullopt,
                            const std::optional<torch::Tensor> &tilePixelCumsum = std::nullopt,
                            const std::optional<torch::Tensor> &pixelMap        = std::nullopt);

    const torch::Tensor &tileOffsets() const;

    const torch::Tensor &tileGaussianIds() const;

    uint32_t tileSize() const;

    const torch::Tensor &activeTiles() const;

    const torch::Tensor &tilePixelMask() const;

    const torch::Tensor &tilePixelCumsum() const;

    const torch::Tensor &pixelMap() const;

    int64_t totalIntersections() const;

#if defined(__CUDACC__)
    struct Accessor {
      private:
        Accessor(fvdb::TorchRAcc64<int32_t, 1> sparseTileOffsetsIn,
                 fvdb::TorchRAcc64<int32_t, 3> denseTileOffsetsIn,
                 bool useDenseTileOffsetsIn,
                 uint32_t denseNumCamerasIn,
                 fvdb::TorchRAcc64<int32_t, 1> tileGaussianIdsIn,
                 fvdb::TorchRAcc64<int32_t, 1> activeTilesIn,
                 fvdb::TorchRAcc64<uint64_t, 2> tilePixelMaskIn,
                 fvdb::TorchRAcc64<int64_t, 1> tilePixelCumsumIn,
                 fvdb::TorchRAcc64<int64_t, 1> pixelMapIn,
                 const RenderWindow2D &renderWindow,
                 uint32_t tileSizeIn,
                 uint32_t blockOffsetIn,
                 int32_t totalIntersectionsIn)
            : mSparseTileOffsets(sparseTileOffsetsIn), mDenseTileOffsets(denseTileOffsetsIn),
              mUseDenseTileOffsets(useDenseTileOffsetsIn), mDenseNumCameras(denseNumCamerasIn),
              mTileGaussianIds(tileGaussianIdsIn), mActiveTiles(activeTilesIn),
              mTilePixelMask(tilePixelMaskIn), mTilePixelCumsum(tilePixelCumsumIn),
              mPixelMap(pixelMapIn), mNumActiveTiles(static_cast<uint32_t>(activeTilesIn.size(0))),
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
        struct ActivePixelScratch {
            typename ActivePixelBlockScanT::TempStorage tempStorage;
        };

        __host__ __device__ inline uint32_t
        numActiveTiles() const {
            return mNumActiveTiles;
        }
        __host__ __device__ inline uint32_t
        numTilesW() const {
            return mNumTilesW;
        }
        __host__ __device__ inline uint32_t
        numTilesH() const {
            return mNumTilesH;
        }
        __host__ __device__ inline uint32_t
        numCameras() const {
            return 0;
        }
        __host__ __device__ inline uint32_t
        cameraCount(uint32_t fallback) const {
            return fallback;
        }
        __host__ __device__ inline uint32_t
        tileSize() const {
            return mTileSize;
        }
        __host__ __device__ inline uint32_t
        pixelCount() const {
            return mPixelMap.size(0);
        }
        __host__ __device__ inline uint32_t
        outputNumTensors(uint32_t numCameras) const {
            return numCameras;
        }
        __host__ __device__ inline uint64_t
        outputPixelCount(uint32_t numCameras, uint32_t renderHeight, uint32_t renderWidth) const {
            (void)numCameras;
            (void)renderHeight;
            (void)renderWidth;
            return pixelCount();
        }

        __host__ __device__ inline uint32_t
        globalTileOrdinal(uint32_t blockOrdinal) const {
            return blockOrdinal + mBlockOffset;
        }

        __device__ inline int32_t
        gaussianIdAt(int32_t intersectionIdx) const {
            return mTileGaussianIds[intersectionIdx];
        }

        __device__ inline cuda::std::tuple<int32_t, int32_t>
        tileGaussianRange(uint32_t tileOrdinal) const {
            return {mSparseTileOffsets[tileOrdinal], mSparseTileOffsets[tileOrdinal + 1]};
        }

        __device__ inline cuda::std::tuple<int32_t, int32_t>
        tileGaussianRangeFromBlock(uint32_t blockOrdinal,
                                   uint32_t cameraId,
                                   uint32_t tileRow,
                                   uint32_t tileCol) const {
            if (mUseDenseTileOffsets) {
                return denseTileGaussianRange(cameraId, tileRow, tileCol);
            }
            return tileGaussianRange(globalTileOrdinal(blockOrdinal));
        }

        __device__ inline cuda::std::tuple<int32_t, int32_t>
        cameraTileId(uint32_t tileOrdinal, uint32_t numTilesW, uint32_t numTilesH) const {
            const int32_t globalTile = mActiveTiles[tileOrdinal];
            const int32_t tileCount  = numTilesW * numTilesH;
            return {globalTile / tileCount, globalTile % tileCount};
        }

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

        __device__ inline cuda::std::tuple<int32_t, int32_t>
        denseTileGaussianRange(uint32_t cameraId, uint32_t tileRow, uint32_t tileCol) const {
            const int32_t firstGaussianIdInBlock = mDenseTileOffsets[cameraId][tileRow][tileCol];
            auto [nextTileRow, nextTileCol]      = (tileCol < mNumTilesW - 1)
                                                       ? cuda::std::make_tuple(tileRow, tileCol + 1)
                                                       : cuda::std::make_tuple(tileRow + 1, 0u);
            const int32_t lastGaussianIdInBlock =
                ((cameraId == mDenseNumCameras - 1) && (nextTileRow == mNumTilesH))
                    ? mTotalIntersections
                    : mDenseTileOffsets[cameraId][nextTileRow][nextTileCol];
            return {firstGaussianIdInBlock, lastGaussianIdInBlock};
        }

        fvdb::TorchRAcc64<int32_t, 1>
            mSparseTileOffsets; // [AT + 1] (valid when !mUseDenseTileOffsets)
        fvdb::TorchRAcc64<int32_t, 3>
            mDenseTileOffsets;  // [C, TH, TW] (valid when mUseDenseTileOffsets)
        bool mUseDenseTileOffsets;
        uint32_t mDenseNumCameras;
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

        if (tileOffsets.dim() == 1) {
            auto dummyDense = torch::empty({0, 0, 0}, tileOffsets.options());
            std::forward<CallbackT>(callback)(SparseTileIntersections::Accessor(
                tileOffsets.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                dummyDense.packed_accessor64<int32_t, 3, torch::RestrictPtrTraits>(),
                false,
                0,
                tileGaussianIdsAcc,
                activeTilesAcc,
                tilePixelMaskAcc,
                tilePixelCumsumAcc,
                pixelMapAcc,
                renderWindow,
                tileSize,
                blockOffset,
                totalIntersections));
        } else {
            auto dummySparse = torch::empty({0}, tileOffsets.options());
            std::forward<CallbackT>(callback)(SparseTileIntersections::Accessor(
                dummySparse.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                tileOffsets.packed_accessor64<int32_t, 3, torch::RestrictPtrTraits>(),
                true,
                static_cast<uint32_t>(tileOffsets.size(0)),
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
