// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <fvdb/detail/utils/cuda/Prefetch.h>
#include <fvdb/detail/utils/cuda/Utils.cuh>

#include <c10/core/ScalarType.h>
#include <torch/types.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace {

constexpr uint32_t kNumTilesH      = 2;
constexpr uint32_t kNumTilesW      = 3;
constexpr uint32_t kImageH         = 7;
constexpr uint32_t kImageW         = 10;
constexpr uint32_t kTileSize       = 4;
constexpr uint32_t kChannels       = 3;
constexpr uint32_t kTilesPerCamera = kNumTilesH * kNumTilesW;

class PrefetchTest : public ::testing::Test {
  protected:
    void
    checkPerTileRanges(uint32_t batchSize, uint32_t deviceCount) {
        const auto options = torch::TensorOptions().dtype(torch::kFloat32);
        const auto batch   = static_cast<int64_t>(batchSize);

        auto tileTensor                    = torch::empty({batch,
                                                           static_cast<int64_t>(kNumTilesH),
                                                           static_cast<int64_t>(kNumTilesW),
                                                           static_cast<int64_t>(kChannels)},
                                       options);
        auto imageTensor                   = torch::empty({batch,
                                                           static_cast<int64_t>(kImageH),
                                                           static_cast<int64_t>(kImageW),
                                                           static_cast<int64_t>(kChannels)},
                                        options);
        std::vector<torch::Tensor> tensors = {tileTensor, imageTensor};
        const uint64_t totalTiles          = static_cast<uint64_t>(batchSize) * kTilesPerCamera;

        for (uint32_t deviceId = 0; deviceId < deviceCount; ++deviceId) {
            const auto [chunkOffset, chunkCount] = fvdb::detail::deviceAlignedChunk(
                1, batchSize * kTilesPerCamera, deviceId, deviceCount);
            const auto tileOffset = static_cast<uint32_t>(chunkOffset);
            const auto tileCount  = static_cast<uint32_t>(chunkCount);
            ASSERT_GT(tileCount, 0u);
            const uint64_t tileRangeEnd = static_cast<uint64_t>(tileOffset) + tileCount;

            // Chunks must stay inside the batched tile domain.
            EXPECT_LT(tileOffset, tileRangeEnd);
            EXPECT_LE(tileRangeEnd, totalTiles);
            if (deviceId + 1 == deviceCount) {
                EXPECT_EQ(tileRangeEnd, totalTiles);
            }

            std::vector<void *> prefetchPointers;
            std::vector<size_t> prefetchSizes;
            const fvdb::detail::TilePrefetchRange range{
                tileOffset, tileCount, kNumTilesH, kNumTilesW, kImageH, kImageW, kTileSize};
            fvdb::detail::appendPerTilePrefetchRanges(
                prefetchPointers, prefetchSizes, tensors, range);

            auto *const tileBase           = static_cast<uint8_t *>(tileTensor.data_ptr());
            auto *const imageBase          = static_cast<uint8_t *>(imageTensor.data_ptr());
            const size_t scalarSize        = c10::elementSize(tileTensor.scalar_type());
            const size_t tileSizeBytes     = tileTensor.stride(2) * scalarSize;
            const size_t cameraStrideBytes = imageTensor.stride(0) * scalarSize;

            const uint32_t cameraOffset = tileOffset / kTilesPerCamera;
            const uint32_t cameraCount =
                static_cast<uint32_t>((tileRangeEnd + kTilesPerCamera - 1) / kTilesPerCamera) -
                cameraOffset;
            const size_t expectedRangeCount = static_cast<size_t>(cameraCount) + 1;

            // One tile range, plus one image range per touched camera.
            ASSERT_EQ(prefetchPointers.size(), expectedRangeCount);
            ASSERT_EQ(prefetchSizes.size(), expectedRangeCount);

            const auto tileBaseAddress  = reinterpret_cast<std::uintptr_t>(tileBase);
            const auto imageBaseAddress = reinterpret_cast<std::uintptr_t>(imageBase);
            const auto tileTensorEnd =
                tileBaseAddress + static_cast<size_t>(tileTensor.numel()) * scalarSize;
            const auto tileRangeBegin = reinterpret_cast<std::uintptr_t>(prefetchPointers[0]);
            const auto tilePointerEnd = tileRangeBegin + prefetchSizes[0];

            // The first range should exactly match the requested tile-layout span.
            EXPECT_EQ(prefetchPointers[0],
                      static_cast<void *>(tileBase + tileOffset * tileSizeBytes));
            EXPECT_EQ(prefetchSizes[0], tileCount * tileSizeBytes);
            EXPECT_GT(prefetchSizes[0], 0ul);

            // The tile span must be non-empty and within tensor storage.
            EXPECT_LT(tileRangeBegin, tilePointerEnd);
            EXPECT_GE(tileRangeBegin, tileBaseAddress);
            EXPECT_LE(tilePointerEnd, tileTensorEnd);

            for (uint32_t cameraIndex = 0; cameraIndex < cameraCount; ++cameraIndex) {
                const size_t rangeIndex = static_cast<size_t>(cameraIndex) + 1;
                const uint32_t cameraId = cameraOffset + cameraIndex;
                const auto cameraBegin  = imageBaseAddress + cameraId * cameraStrideBytes;
                const auto cameraEnd    = cameraBegin + cameraStrideBytes;
                const auto rangeBegin =
                    reinterpret_cast<std::uintptr_t>(prefetchPointers[rangeIndex]);
                const auto rangeEnd        = rangeBegin + prefetchSizes[rangeIndex];
                const auto cameraTileBegin = static_cast<uint64_t>(cameraId) * kTilesPerCamera;
                const auto cameraTileEnd   = cameraTileBegin + kTilesPerCamera;

                // Image ranges must be non-empty and contained in one camera.
                EXPECT_GT(prefetchSizes[rangeIndex], 0ul);
                EXPECT_LT(rangeBegin, rangeEnd);
                EXPECT_GE(rangeBegin, cameraBegin);
                EXPECT_LE(rangeEnd, cameraEnd);

                // A fully covered camera should prefetch its whole image slice.
                if (static_cast<uint64_t>(tileOffset) <= cameraTileBegin &&
                    cameraTileEnd <= tileRangeEnd) {
                    EXPECT_EQ(rangeBegin, cameraBegin);
                    EXPECT_EQ(prefetchSizes[rangeIndex], cameraStrideBytes);
                }
            }
        }
    }
};

TEST_F(PrefetchTest, PerTileRangesBatch1Device1) {
    checkPerTileRanges(1, 1);
}

TEST_F(PrefetchTest, PerTileRangesBatch1Device2) {
    checkPerTileRanges(1, 2);
}

TEST_F(PrefetchTest, PerTileRangesBatch2Device1) {
    checkPerTileRanges(2, 1);
}

TEST_F(PrefetchTest, PerTileRangesBatch2Device2) {
    checkPerTileRanges(2, 2);
}

} // namespace
