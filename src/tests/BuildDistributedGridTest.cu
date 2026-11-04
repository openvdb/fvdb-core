// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <fvdb/GridStorage.h>
#include <fvdb/TorchDeviceResource.h>
#include <fvdb/detail/utils/cuda/Utils.cuh>
#include <fvdb/detail/utils/nanovdb/BuildDistributedGrid.cuh>
#include <fvdb/detail/utils/nanovdb/DeviceGridHandleUtils.cuh>

#include <nanovdb/NanoVDB.h>

#include <c10/core/Allocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>
#include <torch/types.h>

#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <vector>

namespace {

// fvdb-core does not register a PrivateUse1 backend, so the test stands one in: managed memory,
// counted so the test can see which grid buffers are still alive.
struct ManagedPrivateUse1Allocator final : c10::Allocator {
    std::atomic<int> live{0};

    c10::DataPtr
    allocate(size_t bytes) override {
        void *p = nullptr;
        if (bytes) {
            C10_CUDA_CHECK(cudaMallocManaged(&p, bytes));
            live.fetch_add(1);
        }
        return {
            p, p, &ManagedPrivateUse1Allocator::deleteCounted, c10::Device(c10::kPrivateUse1, 0)};
    }

    c10::DeleterFnPtr
    raw_deleter() const override {
        return &ManagedPrivateUse1Allocator::deleteCounted;
    }

    void
    copy_data(void *dest, const void *src, std::size_t count) const override {
        std::memcpy(dest, src, count);
    }

    static ManagedPrivateUse1Allocator &
    instance() {
        static ManagedPrivateUse1Allocator a;
        return a;
    }

  private:
    static void
    deleteCounted(void *p) {
        if (p) {
            instance().live.fetch_sub(1);
            C10_CUDA_CHECK(cudaFree(p));
        }
    }
};

} // namespace

#if CCCL_DEVICE_MERGE_SUPPORTED

namespace {

using fvdb::GridStorage;

const torch::Device kPrivateUse1(torch::kPrivateUse1, 0);
const torch::Device kCuda0(torch::kCUDA, 0);

// A 4^3 block plus two voxels in far leaves, so the grid spans several nodes.
std::vector<nanovdb::Coord>
testCoords() {
    std::vector<nanovdb::Coord> coords;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 4; ++k)
                coords.emplace_back(i, j, k);
    coords.emplace_back(100, -50, 7);
    coords.emplace_back(-300, 2, 900);
    return coords;
}

class BuildDistributedGridTest : public ::testing::Test {
  protected:
    void
    SetUp() override {
        c10::SetAllocator(c10::DeviceType::PrivateUse1, &alloc());
        // Initializes torch's CUDA caching allocator, which CUDA grid storage allocates from.
        (void)torch::empty({1}, torch::TensorOptions().device(kCuda0));
        mExpected = testCoords();
        C10_CUDA_CHECK(cudaMallocManaged(&mCoords, mExpected.size() * sizeof(nanovdb::Coord)));
        std::copy(mExpected.begin(), mExpected.end(), mCoords);
    }

    void
    TearDown() override {
        C10_CUDA_CHECK(cudaFree(mCoords));
        EXPECT_EQ(alloc().live.load(), 0) << "a PrivateUse1 grid buffer leaked";
    }

    static ManagedPrivateUse1Allocator &
    alloc() {
        return ManagedPrivateUse1Allocator::instance();
    }

    GridStorage
    build() {
        return GridStorage(
            fvdb::detail::buildDistributedGrid<nanovdb::ValueOnIndex>(mCoords, mExpected.size()),
            kPrivateUse1);
    }

    void
    expectGrid(const nanovdb::NanoGrid<nanovdb::ValueOnIndex> *grid) const {
        ASSERT_NE(grid, nullptr);
        EXPECT_EQ(grid->activeVoxelCount(), mExpected.size());
        const auto acc = grid->getAccessor();
        for (const auto &c: mExpected) {
            EXPECT_TRUE(acc.isActive(c)) << c[0] << "," << c[1] << "," << c[2];
        }
        EXPECT_FALSE(acc.isActive(nanovdb::Coord(5, 5, 5)));
    }

    std::vector<nanovdb::Coord> mExpected;
    nanovdb::Coord *mCoords = nullptr;
};

std::vector<std::byte>
hostCopyOf(const GridStorage &storage) {
    const GridStorage host = storage.to(torch::kCPU);
    const auto *p          = static_cast<const std::byte *>(host.hostBytes());
    return std::vector<std::byte>(p, p + host.bufferSize());
}

} // namespace

TEST_F(BuildDistributedGridTest, HoldsTheBuiltGridAsUnifiedStorage) {
    GridStorage storage = build();

    // The build buffer is the storage: one PrivateUse1 allocation, no copy.
    EXPECT_EQ(alloc().live.load(), 1);
    EXPECT_TRUE(storage.isDevice());
    EXPECT_FALSE(storage.isHost());
    EXPECT_EQ(storage.device(), kPrivateUse1);
    EXPECT_EQ(storage.stream(), cudaStream_t{});
    ASSERT_EQ(storage.gridCount(), 1u);
    EXPECT_NE(storage.deviceBytes(), nullptr);
    EXPECT_EQ(storage.hostBytes(), storage.deviceBytes());
    EXPECT_NO_THROW(storage.unifiedHandle());
    EXPECT_THROW(storage.deviceHandle(), c10::Error);
    EXPECT_THROW(storage.hostHandle(), c10::Error);

    // Unified memory, so the host can read the grid through either accessor.
    expectGrid(storage.hostGridAt<nanovdb::ValueOnIndex>(0));
    expectGrid(storage.deviceGridAt<nanovdb::ValueOnIndex>(0));
}

TEST_F(BuildDistributedGridTest, CopiesOutOfUnifiedStorage) {
    GridStorage storage    = build();
    const auto original    = hostCopyOf(storage);
    const auto *unifiedRaw = static_cast<const std::byte *>(storage.hostBytes());
    ASSERT_EQ(original.size(), storage.bufferSize());
    EXPECT_TRUE(std::equal(original.begin(), original.end(), unifiedRaw));

    GridStorage cuda = storage.to(kCuda0);
    EXPECT_NO_THROW(cuda.deviceHandle());
    EXPECT_EQ(hostCopyOf(cuda), original);

    // A PrivateUse1 destination is an ordinary device handle over the PrivateUse1 allocator.
    GridStorage pu1 = storage.to(kPrivateUse1);
    EXPECT_NO_THROW(pu1.deviceHandle());
    EXPECT_EQ(alloc().live.load(), 2);
    expectGrid(pu1.hostGridAt<nanovdb::ValueOnIndex>(0));
}

TEST_F(BuildDistributedGridTest, MergesUnifiedParts) {
    fvdb::detail::GridStorageParts parts(kPrivateUse1, cudaStream_t{}, 2);
    parts.add(build());
    parts.add(build());
    GridStorage merged = parts.merge();

    EXPECT_NO_THROW(merged.deviceHandle());
    ASSERT_EQ(merged.gridCount(), 2u);
    for (uint32_t i = 0; i < 2; ++i) {
        const auto *grid = merged.hostGridAt<nanovdb::ValueOnIndex>(i);
        expectGrid(grid);
        EXPECT_EQ(grid->gridIndex(), i);
        EXPECT_EQ(grid->gridCount(), 2u);
    }
}

#else

TEST(BuildDistributedGridTest, Skipped) {
    GTEST_SKIP() << "DistributedPointsToGrid requires CUDA 12.8 or later";
}

#endif
