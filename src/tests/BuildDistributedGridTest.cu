// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <fvdb/GridStorage.h>
#include <fvdb/TorchDeviceResource.h>
#include <fvdb/detail/utils/cuda/Utils.cuh>
#include <fvdb/detail/utils/nanovdb/BuildDistributedGrid.cuh>

#include <nanovdb/NanoVDB.h>

#include <c10/core/Allocator.h>
#include <c10/cuda/CUDAException.h>
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

TEST(BuildDistributedGridTest, BuildsIntoPrivateUse1StorageAndFreesTheUnifiedSource) {
    auto &alloc = ManagedPrivateUse1Allocator::instance();
    c10::SetAllocator(c10::DeviceType::PrivateUse1, &alloc);

    // A 4^3 block plus two voxels in far leaves, so the grid spans several nodes.
    std::vector<nanovdb::Coord> expected;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 4; ++k)
                expected.emplace_back(i, j, k);
    expected.emplace_back(100, -50, 7);
    expected.emplace_back(-300, 2, 900);

    nanovdb::Coord *coords = nullptr;
    C10_CUDA_CHECK(cudaMallocManaged(&coords, expected.size() * sizeof(nanovdb::Coord)));
    std::copy(expected.begin(), expected.end(), coords);

    const torch::Device device(torch::kPrivateUse1, 0);
    const cudaStream_t stream = cudaStream_t{};
    {
        const fvdb::DeviceGridBuffer proto = fvdb::GridStorage::deviceProto(device, stream);
        auto handle = fvdb::detail::buildDistributedGrid<nanovdb::ValueOnIndex>(
            coords, expected.size(), proto, stream);

        // Only the copied grid storage is left; the unified build buffer was freed.
        EXPECT_EQ(alloc.live.load(), 1);
        EXPECT_TRUE(handle.buffer().resource().device().is_privateuseone());
        ASSERT_EQ(handle.gridCount(), 1u);

        // PrivateUse1 memory is unified, so the host can read the device grid.
        const auto *grid = handle.deviceGrid<nanovdb::ValueOnIndex>();
        ASSERT_NE(grid, nullptr);
        EXPECT_EQ(grid->activeVoxelCount(), expected.size());
        const auto acc = grid->getAccessor();
        for (const auto &c: expected) {
            EXPECT_TRUE(acc.isActive(c)) << c[0] << "," << c[1] << "," << c[2];
        }
        EXPECT_FALSE(acc.isActive(nanovdb::Coord(5, 5, 5)));
    }
    EXPECT_EQ(alloc.live.load(), 0);

    C10_CUDA_CHECK(cudaFree(coords));
}

#else

TEST(BuildDistributedGridTest, BuildsIntoPrivateUse1StorageAndFreesTheUnifiedSource) {
    GTEST_SKIP() << "DistributedPointsToGrid requires CUDA 12.8 or later";
}

#endif
