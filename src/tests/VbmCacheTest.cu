// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// VbmCacheTest.cu -- Tests for the lazily-built per-grid VoxelBlockManager cache.
//
// The VBM decode contract these tests pin down:
//   - decoding slot s of grid bi yields the leaf/voxel whose sequential value index is s,
//     i.e. leaf.getValue(voxelOffset) == s and leaf.offsetToGlobalCoord(voxelOffset) is the
//     s-th active coordinate in the grid's sequential order (== activeGridCoords row s-1);
//   - cache entries are built once per grid and shared between a GridBatchData and its
//     sliced/indexed views (same underlying buffer);
//   - empty grids yield a zero GridVbm and never touch the device;
//   - every fVDB grid production path emits sequential (breadth-first, fixed-size) grids,
//     which the decode requires (grid->isSequential()).
//
#include <fvdb/GridBatchData.h>
#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/VbmCache.h>
#include <fvdb/detail/ops/ActiveGridCoords.h>
#include <fvdb/detail/ops/BuildGridFromIjk.h>
#include <fvdb/detail/ops/BuildPaddedGrid.h>
#include <fvdb/detail/ops/ConcatenateGrids.h>
#include <fvdb/detail/ops/IndexGrid.h>
#include <fvdb/detail/ops/MakeContiguous.h>

#include <nanovdb/tools/cuda/VoxelBlockManager.cuh>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/types.h>

#include <gtest/gtest.h>

#include <vector>

using namespace fvdb;
using namespace fvdb::detail;

namespace {

bool
cudaIsAvailable() {
    int count = 0;
    auto err  = cudaGetDeviceCount(&count);
    return err == cudaSuccess && count > 0;
}

// Deterministic pseudo-random unique ijk coordinates: a seeded permutation of a dense box,
// truncated to numVoxels rows. Coordinates are unique by construction so the built grid has
// exactly numVoxels active voxels.
torch::Tensor
randomIjk(int64_t numVoxels, int boxDim, uint64_t seed) {
    const int64_t boxVolume = int64_t(boxDim) * boxDim * boxDim;
    TORCH_CHECK(numVoxels <= boxVolume, "box too small for requested voxel count");
    auto gen  = at::detail::createCPUGenerator(seed);
    auto perm = torch::randperm(boxVolume, gen, torch::kInt64).slice(0, 0, numVoxels);
    auto ijk  = torch::empty({numVoxels, 3}, torch::kInt32);
    ijk.select(1, 0).copy_(perm.div(boxDim * boxDim, "floor"));
    ijk.select(1, 1).copy_(perm.div(boxDim, "floor").remainder(boxDim));
    ijk.select(1, 2).copy_(perm.remainder(boxDim));
    return ijk;
}

c10::intrusive_ptr<GridBatchData>
makeBatch(const std::vector<torch::Tensor> &ijkPerGrid, torch::Device device) {
    std::vector<torch::Tensor> onDevice;
    std::vector<nanovdb::Vec3d> voxelSizes, origins;
    for (const auto &ijk: ijkPerGrid) {
        onDevice.push_back(ijk.to(device));
        voxelSizes.push_back({1.0, 1.0, 1.0});
        origins.push_back({0.0, 0.0, 0.0});
    }
    JaggedTensor jt(onDevice);
    return ops::createNanoGridFromIJK(jt, voxelSizes, origins);
}

using VbmT = nanovdb::tools::cuda::VoxelBlockManager<VbmCache::kLog2BlockWidth>;

// One thread per VBM slot: decode and record the coordinate and value index of the decoded
// voxel so the host can compare against the legacy (leaf-scan) ground truth.
__global__ void
decodeAllSlotsKernel(const nanovdb::OnIndexGrid *grid,
                     const uint32_t *firstLeafID,
                     const uint64_t *jumpMap,
                     uint64_t firstOffset,
                     uint64_t lastOffset,
                     int32_t *outIjk,      // [numVoxels, 3]
                     int64_t *outValueIdx) // [numVoxels]
{
    const uint64_t blockFirstOffset = firstOffset + uint64_t(blockIdx.x) * VbmCache::kBlockWidth;
    const uint64_t slot             = blockFirstOffset + threadIdx.x;
    if (slot > lastOffset) {
        return;
    }
    uint32_t leafIndex;
    uint16_t voxelOffset;
    VbmT::decodeInverseMap(grid,
                           firstLeafID[blockIdx.x],
                           jumpMap + uint64_t(blockIdx.x) * VbmCache::kJumpMapWordCount,
                           blockFirstOffset,
                           int(threadIdx.x),
                           leafIndex,
                           voxelOffset);
    if (leafIndex == VbmT::UnusedLeafIndex) {
        return;
    }
    const auto &leaf         = grid->tree().template getFirstNode<0>()[leafIndex];
    const nanovdb::Coord ijk = leaf.offsetToGlobalCoord(voxelOffset);
    const uint64_t row       = slot - 1;
    outIjk[row * 3 + 0]      = ijk[0];
    outIjk[row * 3 + 1]      = ijk[1];
    outIjk[row * 3 + 2]      = ijk[2];
    outValueIdx[row]         = int64_t(leaf.getValue(voxelOffset));
}

// Decode every slot of grid `bi` through the cache and assert coordinate and value-index
// parity against the legacy leaf-scan ground truth (activeGridCoords on a CPU twin).
void
expectDecodeParity(GridBatchData &batch, int64_t bi, const torch::Tensor &expectedIjkCpu) {
    auto vbm = batch.vbmCache().get(batch, bi);
    ASSERT_EQ(int64_t(vbm.lastOffset), batch.numVoxelsAt(bi));
    ASSERT_EQ(vbm.firstOffset, 1u);
    ASSERT_EQ(
        vbm.blockCount,
        uint32_t((batch.numVoxelsAt(bi) + VbmCache::kBlockWidth - 1) >> VbmCache::kLog2BlockWidth));

    const int64_t numVoxels = batch.numVoxelsAt(bi);
    auto opts               = torch::TensorOptions().device(batch.device());
    auto outIjk             = torch::full({numVoxels, 3}, -12345, opts.dtype(torch::kInt32));
    auto outValueIdx        = torch::zeros({numVoxels}, opts.dtype(torch::kInt64));

    decodeAllSlotsKernel<<<vbm.blockCount, VbmCache::kBlockWidth>>>(
        batch.deviceGridPtrAt(bi),
        vbm.firstLeafID,
        vbm.jumpMap,
        vbm.firstOffset,
        vbm.lastOffset,
        outIjk.data_ptr<int32_t>(),
        outValueIdx.data_ptr<int64_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    EXPECT_TRUE(torch::equal(outIjk.cpu(), expectedIjkCpu));
    // Sequential OnIndex invariant: the decoded voxel's value index equals its slot.
    EXPECT_TRUE(torch::equal(outValueIdx.cpu(), torch::arange(1, numVoxels + 1, torch::kInt64)));
}

} // namespace

TEST(VbmCacheTest, DecodeParity) {
    if (!cudaIsAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }
    auto ijk       = randomIjk(50000, 64, /*seed=*/42);
    auto cudaBatch = makeBatch({ijk}, torch::Device(torch::kCUDA, 0));
    auto cpuBatch  = makeBatch({ijk}, torch::Device(torch::kCPU));
    ASSERT_EQ(cudaBatch->numVoxelsAt(0), 50000);

    auto expected = ops::activeGridCoords(*cpuBatch).jdata();
    expectDecodeParity(*cudaBatch, 0, expected);
}

TEST(VbmCacheTest, MultiBatchDecodeParity) {
    if (!cudaIsAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }
    // Different sizes, deliberately including exact multiples of the block width (512) and
    // off-by-one sizes around it.
    std::vector<torch::Tensor> ijks = {randomIjk(100, 16, 1),
                                       randomIjk(512, 32, 2),
                                       randomIjk(513, 32, 3),
                                       randomIjk(9000, 48, 4)};
    auto cudaBatch                  = makeBatch(ijks, torch::Device(torch::kCUDA, 0));
    auto cpuBatch                   = makeBatch(ijks, torch::Device(torch::kCPU));
    auto expected                   = ops::activeGridCoords(*cpuBatch);

    for (int64_t bi = 0; bi < cudaBatch->batchSize(); ++bi) {
        SCOPED_TRACE("grid " + std::to_string(bi));
        expectDecodeParity(*cudaBatch, bi, expected.index(bi).jdata());
    }
}

TEST(VbmCacheTest, ViewSharing) {
    if (!cudaIsAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }
    std::vector<torch::Tensor> ijks = {
        randomIjk(600, 16, 5), randomIjk(700, 16, 6), randomIjk(800, 16, 7)};
    auto parent = makeBatch(ijks, torch::Device(torch::kCUDA, 0));
    auto slice  = ops::indexGrid(*parent, 1, 3, 1);

    // The slice shares the parent's cache object, and the same logical grid resolves to the
    // same cached entry (same device pointers) through either.
    ASSERT_EQ(&slice->vbmCache(), &parent->vbmCache());
    auto fromSlice  = slice->vbmCache().get(*slice, 0);
    auto fromParent = parent->vbmCache().get(*parent, 1);
    EXPECT_EQ(fromSlice.firstLeafID, fromParent.firstLeafID);
    EXPECT_EQ(fromSlice.jumpMap, fromParent.jumpMap);
    EXPECT_EQ(fromSlice.blockCount, fromParent.blockCount);

    // Decode parity through the view.
    auto cpuBatch = makeBatch(ijks, torch::Device(torch::kCPU));
    auto expected = ops::activeGridCoords(*cpuBatch);
    expectDecodeParity(*slice, 0, expected.index(1).jdata());
    expectDecodeParity(*slice, 1, expected.index(2).jdata());
}

TEST(VbmCacheTest, EmptyGrid) {
    if (!cudaIsAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }
    std::vector<torch::Tensor> ijks = {torch::empty({0, 3}, torch::kInt32), randomIjk(100, 16, 8)};
    auto batch                      = makeBatch(ijks, torch::Device(torch::kCUDA, 0));
    ASSERT_EQ(batch->numVoxelsAt(0), 0);

    auto vbm = batch->vbmCache().get(*batch, 0);
    EXPECT_EQ(vbm.blockCount, 0u);
    EXPECT_EQ(vbm.firstLeafID, nullptr);
    EXPECT_EQ(vbm.jumpMap, nullptr);

    auto cpuBatch = makeBatch(ijks, torch::Device(torch::kCPU));
    auto expected = ops::activeGridCoords(*cpuBatch);
    expectDecodeParity(*batch, 1, expected.index(1).jdata());
}

TEST(VbmCacheTest, RepeatedGetIsCached) {
    if (!cudaIsAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }
    auto batch = makeBatch({randomIjk(2000, 32, 9)}, torch::Device(torch::kCUDA, 0));
    auto first = batch->vbmCache().get(*batch, 0);
    auto again = batch->vbmCache().get(*batch, 0);
    EXPECT_EQ(first.firstLeafID, again.firstLeafID);
    EXPECT_EQ(first.jumpMap, again.jumpMap);
}

// A consumer on a different stream than the build must (a) observe the finished build (the
// build event wait) and (b) keep the cached buffers alive until its kernels drain, even if the
// owning GridBatchData is destroyed while they are still in flight (record_stream with the
// caching allocator). Decode parity through the side stream verifies (a); destroying the batch
// immediately after the async launch exercises (b) -- a lifetime bug here surfaces as corrupt
// output or as an invalid access under compute-sanitizer.
TEST(VbmCacheTest, CrossStreamConsumerLifetime) {
    if (!cudaIsAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }
    auto ijk      = randomIjk(50000, 64, 12);
    auto cpuBatch = makeBatch({ijk}, torch::Device(torch::kCPU));
    auto expected = ops::activeGridCoords(*cpuBatch).jdata();

    auto device             = torch::Device(torch::kCUDA, 0);
    auto opts               = torch::TensorOptions().device(device);
    const int64_t numVoxels = 50000;
    auto outIjk             = torch::full({numVoxels, 3}, -12345, opts.dtype(torch::kInt32));
    auto outValueIdx        = torch::zeros({numVoxels}, opts.dtype(torch::kInt64));

    {
        auto batch = makeBatch({ijk}, device);
        ASSERT_EQ(batch->numVoxelsAt(0), numVoxels);
        // Build the cache entry on the default stream.
        (void)batch->vbmCache().get(*batch, 0);

        // Consume it from a side stream, then destroy the batch (and with it the cache) while
        // the side stream's kernel may still be running.
        c10::cuda::CUDAStream sideStream = c10::cuda::getStreamFromPool(false, device.index());
        {
            c10::cuda::CUDAStreamGuard streamGuard(sideStream);
            auto vbm = batch->vbmCache().get(*batch, 0);
            decodeAllSlotsKernel<<<vbm.blockCount, VbmCache::kBlockWidth, 0, sideStream>>>(
                batch->deviceGridPtrAt(0),
                vbm.firstLeafID,
                vbm.jumpMap,
                vbm.firstOffset,
                vbm.lastOffset,
                outIjk.data_ptr<int32_t>(),
                outValueIdx.data_ptr<int64_t>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    } // batch (and VbmCache) destroyed here, side-stream kernel possibly in flight

    C10_CUDA_CHECK(cudaDeviceSynchronize());
    EXPECT_TRUE(torch::equal(outIjk.cpu(), expected));
    EXPECT_TRUE(torch::equal(outValueIdx.cpu(), torch::arange(1, numVoxels + 1, torch::kInt64)));
}

// The register decode requires grid->isSequential() (fixed-size, breadth-first leaves). Pin
// that invariant for every grid production path, on CPU grids where the header is readable.
TEST(VbmCacheTest, ProductionPathsAreSequential) {
    auto cpu = torch::Device(torch::kCPU);

    std::vector<torch::Tensor> ijks = {randomIjk(1000, 24, 10), randomIjk(1500, 24, 11)};
    auto built                      = makeBatch(ijks, cpu);
    auto padded                     = ops::buildPaddedGrid(*built, -1, 1, false, false);
    auto sliced                     = ops::indexGrid(*built, 1, 2, 1);
    auto concatenated               = ops::concatenateGrids({built, padded});
    auto contiguous                 = ops::makeContiguous(sliced);

    for (const auto &[name, batch]:
         std::vector<std::pair<const char *, c10::intrusive_ptr<GridBatchData>>>{
             {"createNanoGridFromIJK", built},
             {"buildPaddedGrid", padded},
             {"indexGrid", sliced},
             {"concatenateGrids", concatenated},
             {"makeContiguous", contiguous}}) {
        for (int64_t bi = 0; bi < batch->batchSize(); ++bi) {
            EXPECT_TRUE(batch->hostGridPtrAt(bi)->isSequential())
                << name << " grid " << bi << " is not sequential";
        }
    }
}
