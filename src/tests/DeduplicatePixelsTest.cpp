// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include "utils/Tensor.h"

#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/ops/gsplat/GaussianDeduplicatePixels.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <tuple>
#include <vector>

using fvdb::test::tensorOpts;

static constexpr int64_t kImageWidth  = 64;
static constexpr int64_t kImageHeight = 64;

template <typename CoordType> struct DeduplicatePixelsTest : public ::testing::Test {};

using CoordTypes = ::testing::Types<std::int32_t, std::int64_t>;
TYPED_TEST_SUITE(DeduplicatePixelsTest, CoordTypes);

TYPED_TEST(DeduplicatePixelsTest, Empty) {
    auto opts   = tensorOpts<TypeParam>(torch::kCUDA);
    auto pixels = fvdb::JaggedTensor(torch::empty({0, 2}, opts));

    auto [uniquePixels, inverseIndices, hasDuplicates] =
        fvdb::detail::ops::deduplicatePixels(pixels, kImageWidth, kImageHeight);

    EXPECT_FALSE(hasDuplicates);
    EXPECT_EQ(inverseIndices.size(0), 0);
    EXPECT_EQ(uniquePixels.rsize(0), 0);
}

TYPED_TEST(DeduplicatePixelsTest, SinglePixel) {
    auto opts   = tensorOpts<TypeParam>(torch::kCPU);
    auto coords = torch::tensor({{5, 10}}, opts);
    auto pixels = fvdb::JaggedTensor(std::vector<torch::Tensor>{coords}).to(torch::kCUDA);

    auto [uniquePixels, inverseIndices, hasDuplicates] =
        fvdb::detail::ops::deduplicatePixels(pixels, kImageWidth, kImageHeight);

    EXPECT_FALSE(hasDuplicates);
    EXPECT_EQ(uniquePixels.rsize(0), 1);
}

TYPED_TEST(DeduplicatePixelsTest, AllUnique) {
    auto opts   = tensorOpts<TypeParam>(torch::kCPU);
    auto coords = torch::tensor({{0, 0}, {0, 1}, {1, 0}, {1, 1}, {2, 3}}, opts);
    auto pixels = fvdb::JaggedTensor(std::vector<torch::Tensor>{coords}).to(torch::kCUDA);

    auto [uniquePixels, inverseIndices, hasDuplicates] =
        fvdb::detail::ops::deduplicatePixels(pixels, kImageWidth, kImageHeight);

    EXPECT_FALSE(hasDuplicates);
    EXPECT_EQ(uniquePixels.rsize(0), 5);
    // inverseIndices should be a permutation of [0..4] (identity if no dups)
    EXPECT_EQ(inverseIndices.size(0), 5);
}

TYPED_TEST(DeduplicatePixelsTest, SomeDuplicates) {
    auto opts = tensorOpts<TypeParam>(torch::kCPU);
    // (0,0) appears at index 0 and 2
    auto coords = torch::tensor({{0, 0}, {1, 1}, {0, 0}, {2, 2}}, opts);
    auto pixels = fvdb::JaggedTensor(std::vector<torch::Tensor>{coords}).to(torch::kCUDA);

    auto [uniquePixels, inverseIndices, hasDuplicates] =
        fvdb::detail::ops::deduplicatePixels(pixels, kImageWidth, kImageHeight);

    EXPECT_TRUE(hasDuplicates);
    EXPECT_EQ(uniquePixels.rsize(0), 3);
    EXPECT_EQ(inverseIndices.size(0), 4);

    // Indices 0 and 2 in the original both map to the same unique index
    auto inv = inverseIndices.cpu();
    EXPECT_EQ(inv[0].template item<int64_t>(), inv[2].template item<int64_t>());
    // Indices 1 and 3 map to different unique indices
    EXPECT_NE(inv[1].template item<int64_t>(), inv[3].template item<int64_t>());
}

TYPED_TEST(DeduplicatePixelsTest, AllSamePixel) {
    auto opts   = tensorOpts<TypeParam>(torch::kCPU);
    auto coords = torch::tensor({{5, 5}, {5, 5}, {5, 5}, {5, 5}}, opts);
    auto pixels = fvdb::JaggedTensor(std::vector<torch::Tensor>{coords}).to(torch::kCUDA);

    auto [uniquePixels, inverseIndices, hasDuplicates] =
        fvdb::detail::ops::deduplicatePixels(pixels, kImageWidth, kImageHeight);

    EXPECT_TRUE(hasDuplicates);
    EXPECT_EQ(uniquePixels.rsize(0), 1);
    EXPECT_EQ(inverseIndices.size(0), 4);

    // All inverse indices should be 0
    auto inv = inverseIndices.cpu();
    for (int i = 0; i < 4; i++) {
        EXPECT_EQ(inv[i].template item<int64_t>(), 0);
    }
}

TYPED_TEST(DeduplicatePixelsTest, MultiBatchNoDuplicates) {
    auto opts = tensorOpts<TypeParam>(torch::kCPU);
    // Same (0,0) in different batches should NOT be considered duplicates
    auto batch0 = torch::tensor({{0, 0}, {1, 1}}, opts);
    auto batch1 = torch::tensor({{0, 0}, {2, 2}}, opts);
    auto pixels = fvdb::JaggedTensor(std::vector<torch::Tensor>{batch0, batch1}).to(torch::kCUDA);

    auto [uniquePixels, inverseIndices, hasDuplicates] =
        fvdb::detail::ops::deduplicatePixels(pixels, kImageWidth, kImageHeight);

    EXPECT_FALSE(hasDuplicates);
    EXPECT_EQ(uniquePixels.rsize(0), 4);
    EXPECT_EQ(uniquePixels.num_outer_lists(), 2);
}

TYPED_TEST(DeduplicatePixelsTest, MultiBatchWithDuplicates) {
    auto opts = tensorOpts<TypeParam>(torch::kCPU);
    // Batch 0: (0,0) duplicated; Batch 1: (0,0) alone (not a dup of batch 0's)
    auto batch0 = torch::tensor({{0, 0}, {1, 1}, {0, 0}}, opts);
    auto batch1 = torch::tensor({{0, 0}, {3, 3}}, opts);
    auto pixels = fvdb::JaggedTensor(std::vector<torch::Tensor>{batch0, batch1}).to(torch::kCUDA);

    auto [uniquePixels, inverseIndices, hasDuplicates] =
        fvdb::detail::ops::deduplicatePixels(pixels, kImageWidth, kImageHeight);

    EXPECT_TRUE(hasDuplicates);
    EXPECT_EQ(uniquePixels.num_outer_lists(), 2);
    // Batch 0: 2 unique pixels (0,0) and (1,1); Batch 1: 2 unique pixels (0,0) and (3,3)
    EXPECT_EQ(uniquePixels.rsize(0), 4);
    EXPECT_EQ(inverseIndices.size(0), 5);

    // Original indices 0 and 2 (both batch 0, pixel (0,0)) should map to same unique
    auto inv = inverseIndices.cpu();
    EXPECT_EQ(inv[0].template item<int64_t>(), inv[2].template item<int64_t>());
}

TYPED_TEST(DeduplicatePixelsTest, MultiBatchAllSamePixel) {
    auto opts   = tensorOpts<TypeParam>(torch::kCPU);
    auto batch0 = torch::tensor({{1, 1}, {1, 1}, {1, 1}}, opts);
    auto batch1 = torch::tensor({{2, 2}, {2, 2}}, opts);
    auto pixels = fvdb::JaggedTensor(std::vector<torch::Tensor>{batch0, batch1}).to(torch::kCUDA);

    auto [uniquePixels, inverseIndices, hasDuplicates] =
        fvdb::detail::ops::deduplicatePixels(pixels, kImageWidth, kImageHeight);

    EXPECT_TRUE(hasDuplicates);
    EXPECT_EQ(uniquePixels.num_outer_lists(), 2);
    EXPECT_EQ(uniquePixels.rsize(0), 2);

    auto offsets = uniquePixels.joffsets().cpu();
    EXPECT_EQ(offsets[0].template item<int64_t>(), 0);
    EXPECT_EQ(offsets[1].template item<int64_t>(), 1);
    EXPECT_EQ(offsets[2].template item<int64_t>(), 2);

    auto inv = inverseIndices.cpu();
    EXPECT_EQ(inv[0].template item<int64_t>(), inv[1].template item<int64_t>());
    EXPECT_EQ(inv[0].template item<int64_t>(), inv[2].template item<int64_t>());
    EXPECT_EQ(inv[3].template item<int64_t>(), inv[4].template item<int64_t>());
    EXPECT_NE(inv[0].template item<int64_t>(), inv[3].template item<int64_t>());
}

TYPED_TEST(DeduplicatePixelsTest, RoundTripSomeDuplicates) {
    auto opts   = tensorOpts<TypeParam>(torch::kCPU);
    auto coords = torch::tensor({{3, 7}, {1, 2}, {3, 7}, {5, 5}, {1, 2}, {9, 0}}, opts);
    auto pixels = fvdb::JaggedTensor(std::vector<torch::Tensor>{coords}).to(torch::kCUDA);

    auto [uniquePixels, inverseIndices, hasDuplicates] =
        fvdb::detail::ops::deduplicatePixels(pixels, kImageWidth, kImageHeight);

    EXPECT_TRUE(hasDuplicates);
    EXPECT_EQ(uniquePixels.rsize(0), 4);

    // Round-trip: indexing unique pixels by inverseIndices should reconstruct the original
    auto reconstructed = uniquePixels.jdata().index_select(0, inverseIndices);
    EXPECT_TRUE(torch::equal(reconstructed.cpu(), coords.to(reconstructed.dtype())));
}

TYPED_TEST(DeduplicatePixelsTest, RoundTripMultiBatch) {
    auto opts   = tensorOpts<TypeParam>(torch::kCPU);
    auto batch0 = torch::tensor({{2, 3}, {4, 5}, {2, 3}}, opts);
    auto batch1 = torch::tensor({{6, 7}, {6, 7}, {8, 9}}, opts);
    auto pixels = fvdb::JaggedTensor(std::vector<torch::Tensor>{batch0, batch1}).to(torch::kCUDA);

    auto [uniquePixels, inverseIndices, hasDuplicates] =
        fvdb::detail::ops::deduplicatePixels(pixels, kImageWidth, kImageHeight);

    EXPECT_TRUE(hasDuplicates);

    auto originalJdata = pixels.jdata();
    auto reconstructed = uniquePixels.jdata().index_select(0, inverseIndices);
    EXPECT_TRUE(torch::equal(reconstructed.cpu(), originalJdata.cpu().to(reconstructed.dtype())));
}

TYPED_TEST(DeduplicatePixelsTest, JaggedTensorOffsets) {
    auto opts   = tensorOpts<TypeParam>(torch::kCPU);
    auto batch0 = torch::tensor({{0, 0}, {0, 0}, {1, 1}}, opts);         // 3 pixels, 2 unique
    auto batch1 = torch::tensor({{2, 2}}, opts);                         // 1 pixel, 1 unique
    auto batch2 = torch::tensor({{3, 3}, {4, 4}, {3, 3}, {4, 4}}, opts); // 4 pixels, 2 unique
    auto pixels =
        fvdb::JaggedTensor(std::vector<torch::Tensor>{batch0, batch1, batch2}).to(torch::kCUDA);

    auto [uniquePixels, inverseIndices, hasDuplicates] =
        fvdb::detail::ops::deduplicatePixels(pixels, kImageWidth, kImageHeight);

    EXPECT_TRUE(hasDuplicates);
    EXPECT_EQ(uniquePixels.num_outer_lists(), 3);
    EXPECT_EQ(uniquePixels.rsize(0), 5); // 2 + 1 + 2

    // Verify per-batch counts via offsets
    auto offsets = uniquePixels.joffsets().cpu();
    EXPECT_EQ(offsets[0].template item<int64_t>(), 0);
    EXPECT_EQ(offsets[1].template item<int64_t>(), 2); // batch 0: 2 unique
    EXPECT_EQ(offsets[2].template item<int64_t>(), 3); // batch 1: 1 unique
    EXPECT_EQ(offsets[3].template item<int64_t>(), 5); // batch 2: 2 unique
}
