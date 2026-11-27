// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Argument checks of the batched topology builder (BatchedTopologyBuilder.cuh) that no Python
// entry point can reach, because every op builds valid passes and sources. Each check is driven
// from C++ here, through both pass entry points (runBatchedTopologyPass and buildChunk) where it
// applies, next to a valid Prune pass as the positive control.

#include <fvdb/GridBatchData.h>
#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/ops/BuildGridFromIjk.h>
#include <fvdb/detail/utils/cuda/StreamOrdering.h>
#include <fvdb/detail/utils/nanovdb/BatchedTopologyBuilder.cuh>

#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <torch/cuda.h>
#include <torch/types.h>

#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

namespace fvdb::detail {
namespace {

namespace b = batched;

// Runs `fn` and expects a c10::Error whose message contains `substr`.
template <typename Fn>
void
expectThrowsWith(Fn &&fn, const std::string &substr) {
    try {
        fn();
        ADD_FAILURE() << "expected an error containing \"" << substr << "\"";
    } catch (const c10::Error &e) {
        EXPECT_NE(std::string(e.what()).find(substr), std::string::npos) << e.what();
    }
}

class BatchedTopologyBuilderTest : public ::testing::Test {
  protected:
    void
    SetUp() override {
        if (!torch::cuda::is_available()) {
            GTEST_SKIP() << "CUDA is required for the batched topology builder";
        }
        mGuard.emplace(mDevice);
        // Two grids with leaves in several root tiles, and an empty grid between them.
        const auto opts = torch::TensorOptions().dtype(torch::kInt32);
        std::vector<torch::Tensor> ijks{
            torch::tensor({0, 0, 0, 9, 9, 9, 4100, -4100, 7}, opts).reshape({-1, 3}).to(mDevice),
            torch::empty({0, 3}, opts).to(mDevice),
            torch::tensor({-20, 3, 5, -20, 3, 6, 40, 40, 40}, opts).reshape({-1, 3}).to(mDevice),
        };
        std::vector<nanovdb::Vec3d> sizes(ijks.size(), nanovdb::Vec3d(1.0));
        std::vector<nanovdb::Vec3d> origins(ijks.size(), nanovdb::Vec3d(0.0));
        mBatch  = ops::createNanoGridFromIJK(JaggedTensor(ijks), sizes, origins);
        mStream = storageStream(mDevice);
        mSource = b::sourceFromGridBatch(*mBatch);
        for (int64_t count: mSource.leafCounts) {
            mNumLeaves += count;
        }
        ASSERT_GT(mNumLeaves, 0);
        // All-ones keep masks: a Prune pass with them keeps every source voxel.
        mKeep = torch::full({mNumLeaves * int64_t(b::kLeafMaskWords)},
                            -1,
                            torch::TensorOptions().dtype(torch::kInt64).device(mDevice));
    }

    const b::LeafMaskT *
    keepMasks() const {
        return reinterpret_cast<const b::LeafMaskT *>(mKeep.data_ptr<int64_t>());
    }

    // Root-tile maps for buildChunk, as runBatchedTopologyPass derives them.
    std::vector<b::ProcessedTileMap>
    tilesFor(const b::TopologyPassSpec &pass) const {
        const auto uppers = b::readUpperBBoxes(mSource, mStream);
        std::vector<b::ProcessedTileMap> tiles;
        for (const auto &u: uppers) {
            tiles.push_back(b::processedTiles(u, pass));
        }
        return tiles;
    }

    torch::Device mDevice{torch::kCUDA, 0};
    std::optional<c10::cuda::CUDAGuard> mGuard;
    c10::intrusive_ptr<GridBatchData> mBatch;
    cudaStream_t mStream = nullptr;
    b::BatchedTopologySource mSource;
    int64_t mNumLeaves = 0;
    torch::Tensor mKeep;
};

TEST_F(BatchedTopologyBuilderTest, ValidPrunePassKeepsEverySourceVoxel) {
    const auto pass                 = b::TopologyPassSpec::prune(keepMasks(), mNumLeaves);
    b::BatchedTopologyResult result = b::runBatchedTopologyPass(mSource, pass, mStream);
    ASSERT_EQ(result.grids.size(), mSource.grids.size());
    EXPECT_EQ(result.leafCounts, mSource.leafCounts);
    GridStorage storage = b::storageFromResult(std::move(result), mDevice, mStream);
    EXPECT_EQ(size_t(storage.gridCount()), mSource.grids.size());
}

TEST_F(BatchedTopologyBuilderTest, PruneSpecRejectsInconsistentArguments) {
    expectThrowsWith([] { b::TopologyPassSpec::prune(nullptr, 3); }, "non-null pointer");
    expectThrowsWith([&] { b::TopologyPassSpec::prune(keepMasks(), -1); }, "must not be negative");
    // A zero count with no pointer is valid: a source without leaves needs no keep masks.
    EXPECT_NO_THROW(b::TopologyPassSpec::prune(nullptr, 0));
}

TEST_F(BatchedTopologyBuilderTest, PrunePassRejectsWrongKeepMaskCount) {
    for (const int64_t count: {mNumLeaves - 1, mNumLeaves + 1}) {
        const auto pass       = b::TopologyPassSpec::prune(keepMasks(), count);
        const std::string msg = "Prune pass has " + std::to_string(count) + " keep masks for " +
                                std::to_string(mNumLeaves) + " source leaves";
        expectThrowsWith([&] { b::runBatchedTopologyPass(mSource, pass, mStream); }, msg);
        // buildChunk checks the pass itself, not only through runBatchedTopologyPass.
        const auto tiles = tilesFor(pass);
        b::BatchedTopologyResult result;
        expectThrowsWith(
            [&] {
                b::buildChunk(
                    mSource, pass, 0, int32_t(mSource.grids.size()), tiles, result, mStream);
            },
            msg);
    }
}

TEST_F(BatchedTopologyBuilderTest, BoxDilatePassRejectsBoxOutsideUnitRange) {
    const auto pass       = b::TopologyPassSpec::boxDilate(nanovdb::Coord(-2), nanovdb::Coord(1));
    const std::string msg = "BoxDilate pass components must be in {-1,0} / {0,1}";
    expectThrowsWith([&] { b::runBatchedTopologyPass(mSource, pass, mStream); }, msg);
    std::vector<b::ProcessedTileMap> tiles(mSource.grids.size());
    b::BatchedTopologyResult result;
    expectThrowsWith(
        [&] {
            b::buildChunk(mSource, pass, 0, int32_t(mSource.grids.size()), tiles, result, mStream);
        },
        msg);
}

TEST_F(BatchedTopologyBuilderTest, NullSourceGridIsRejected) {
    b::BatchedTopologySource source = mSource;
    source.grids[1]                 = nullptr;
    const auto pass                 = b::TopologyPassSpec::prune(keepMasks(), mNumLeaves);
    expectThrowsWith([&] { b::runBatchedTopologyPass(source, pass, mStream); }, "null source grid");
    expectThrowsWith([&] { b::sourceLeafTables(source, mStream); }, "null source grid");
}

TEST_F(BatchedTopologyBuilderTest, StorageFromResultRejectsEmptyOrInconsistentResult) {
    const std::string msg = "storageFromResult: empty or inconsistent pass result";
    expectThrowsWith([&] { b::storageFromResult(b::BatchedTopologyResult{}, mDevice, mStream); },
                     msg);
    // A valid result with one grid's byte size dropped.
    b::BatchedTopologyResult result = b::runBatchedTopologyPass(
        mSource, b::TopologyPassSpec::prune(keepMasks(), mNumLeaves), mStream);
    result.gridBytes.pop_back();
    expectThrowsWith([&] { b::storageFromResult(std::move(result), mDevice, mStream); }, msg);
}

TEST_F(BatchedTopologyBuilderTest, SourceLeafTablesComeOnlyFromTheSource) {
    static_assert(!std::is_default_constructible_v<b::SourceLeafTables>,
                  "SourceLeafTables must be built by sourceLeafTables()");
    const b::SourceLeafTables tables = b::sourceLeafTables(mSource, mStream);
    EXPECT_EQ(tables.numGrids(), int32_t(mSource.grids.size()));
    EXPECT_EQ(tables.numLeaves(), mNumLeaves);
    EXPECT_NE(tables.gridPtrs(), nullptr);
    EXPECT_NE(tables.leafStartPtr(), nullptr);
}

} // namespace
} // namespace fvdb::detail
