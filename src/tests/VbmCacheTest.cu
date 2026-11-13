// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// VbmCacheTest.cu -- Tests for the lazily-built, batch-wide VoxelBlockManager cache.
//
// The VBM decode contract these tests pin down:
//   - decoding slot s of grid bi yields the leaf/voxel whose sequential value index is s,
//     i.e. leaf.getValue(voxelOffset) == s and leaf.offsetToGlobalCoord(voxelOffset) is the
//     s-th active coordinate in the grid's sequential order (== activeGridCoords row s-1);
//   - a batch's VBMs are built once, together; every GridBatchData, views included, owns its
//     own, and a view's VBM for a grid matches its parent's for the same grid bitwise;
//   - empty grids yield a zero GridVbm and never touch the device;
//   - every fVDB grid production path emits sequential (breadth-first, fixed-size) grids,
//     which the decode requires (grid->isSequential()).
//
#include <fvdb/GridBatchData.h>
#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/VbmCache.h>
#include <fvdb/detail/ops/ActiveGridCoords.h>
#include <fvdb/detail/ops/BuildCoarseGridFromFine.h>
#include <fvdb/detail/ops/BuildDenseGrid.h>
#include <fvdb/detail/ops/BuildDilatedGrid.h>
#include <fvdb/detail/ops/BuildFineGridFromCoarse.h>
#include <fvdb/detail/ops/BuildGridForConv.h>
#include <fvdb/detail/ops/BuildGridForConvTranspose.h>
#include <fvdb/detail/ops/BuildGridFromIjk.h>
#include <fvdb/detail/ops/BuildGridFromNearestVoxelsToPoints.h>
#include <fvdb/detail/ops/BuildGridFromPoints.h>
#include <fvdb/detail/ops/BuildMergedGrids.h>
#include <fvdb/detail/ops/BuildPaddedGrid.h>
#include <fvdb/detail/ops/BuildPrunedGrid.h>
#include <fvdb/detail/ops/CloneGrid.h>
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
    auto vbm = batch.vbmAt(bi);
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

__global__ void
isSequentialKernel(const nanovdb::OnIndexGrid *grid, bool *out) {
    *out = grid->isSequential();
}

bool
deviceGridIsSequential(const GridBatchData &batch, int64_t bi) {
    auto flag =
        torch::zeros({1}, torch::TensorOptions().device(batch.device()).dtype(torch::kBool));
    isSequentialKernel<<<1, 1>>>(batch.deviceGridPtrAt(bi), flag.data_ptr<bool>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return flag.item<bool>();
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
    // Different sizes, deliberately including exact multiples of the block width and
    // off-by-one sizes around it.
    constexpr int64_t W             = VbmCache::kBlockWidth;
    std::vector<torch::Tensor> ijks = {randomIjk(100, 16, 1),
                                       randomIjk(W, 32, 2),
                                       randomIjk(W + 1, 32, 3),
                                       randomIjk(4 * W - 1, 32, 5),
                                       randomIjk(9000, 48, 4)};
    auto cudaBatch                  = makeBatch(ijks, torch::Device(torch::kCUDA, 0));
    auto cpuBatch                   = makeBatch(ijks, torch::Device(torch::kCPU));
    auto expected                   = ops::activeGridCoords(*cpuBatch);

    for (int64_t bi = 0; bi < cudaBatch->batchSize(); ++bi) {
        SCOPED_TRACE("grid " + std::to_string(bi));
        expectDecodeParity(*cudaBatch, bi, expected.index(bi).jdata());
    }
}

TEST(VbmCacheTest, ViewsBuildTheirOwn) {
    if (!cudaIsAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }
    std::vector<torch::Tensor> ijks = {
        randomIjk(600, 16, 5), randomIjk(700, 16, 6), randomIjk(800, 16, 7)};
    auto parent = makeBatch(ijks, torch::Device(torch::kCUDA, 0));
    auto slice  = ops::indexGrid(*parent, 1, 3, 1);

    // The slice builds its own VBMs, and its grid 0 is the parent's grid 1, byte for byte.
    ASSERT_NE(slice->vbm().firstLeafID, parent->vbm().firstLeafID);
    const auto fromSlice  = slice->vbmAt(0);
    const auto fromParent = parent->vbmAt(1);
    ASSERT_EQ(fromSlice.blockCount, fromParent.blockCount);
    auto opts       = torch::TensorOptions().device(parent->device());
    const int64_t n = fromSlice.blockCount;
    auto asInt32    = [&](const uint32_t *p) {
        return torch::from_blob(const_cast<uint32_t *>(p), {n}, opts.dtype(torch::kInt32));
    };
    auto asInt64 = [&](const uint64_t *p) {
        return torch::from_blob(const_cast<uint64_t *>(p),
                                {n * VbmCache::kJumpMapWordCount},
                                opts.dtype(torch::kInt64));
    };
    EXPECT_TRUE(torch::equal(asInt32(fromSlice.firstLeafID), asInt32(fromParent.firstLeafID)));
    EXPECT_TRUE(torch::equal(asInt64(fromSlice.jumpMap), asInt64(fromParent.jumpMap)));

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

    auto vbm = batch->vbmAt(0);
    EXPECT_EQ(vbm.blockCount, 0u);
    EXPECT_EQ(vbm.firstLeafID, nullptr);
    EXPECT_EQ(vbm.jumpMap, nullptr);

    auto cpuBatch = makeBatch(ijks, torch::Device(torch::kCPU));
    auto expected = ops::activeGridCoords(*cpuBatch);
    expectDecodeParity(*batch, 1, expected.index(1).jdata());
}

TEST(VbmCacheTest, RepeatedAcquireIsCached) {
    if (!cudaIsAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }
    auto batch        = makeBatch({randomIjk(2000, 32, 9)}, torch::Device(torch::kCUDA, 0));
    const auto *first = &batch->vbm();
    const auto *again = &batch->vbm();
    EXPECT_EQ(first, again);
    EXPECT_EQ(batch->vbmAt(0).firstLeafID, first->firstLeafID);
    EXPECT_EQ(batch->vbmAt(0).jumpMap, first->jumpMap);
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
        // Build the VBMs on the default stream.
        (void)batch->vbm();

        // Consume it from a side stream, then destroy the batch (and with it the cache) while
        // the side stream's kernel may still be running.
        c10::cuda::CUDAStream sideStream = c10::cuda::getStreamFromPool(false, device.index());
        {
            c10::cuda::CUDAStreamGuard streamGuard(sideStream);
            auto vbm = batch->vbmAt(0);
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

// The decode trusts the CUDA builders to emit sequential grids whose leaf value indices run
// 1..numVoxels in leaf order. Check every CUDA production path by decoding every slot and
// comparing against the leaf scan on a CPU clone of the same grid bytes.
TEST(VbmCacheTest, CudaProductionPathsDecode) {
    if (!cudaIsAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }
    const auto cuda = torch::Device(torch::kCUDA, 0);
    const std::vector<nanovdb::Vec3d> unit2(2, nanovdb::Vec3d(1.0)), zero2(2, nanovdb::Vec3d(0.0));

    std::vector<torch::Tensor> ijks = {randomIjk(3000, 40, 20), randomIjk(700, 24, 21)};
    auto built                      = makeBatch(ijks, cuda);

    std::vector<torch::Tensor> points;
    for (const auto &ijk: ijks) {
        points.push_back(ijk.to(cuda).to(torch::kFloat32) * 0.97f + 0.3f);
    }
    const JaggedTensor pointsJt(points);

    auto keep = built->jaggedTensor(
        torch::arange(built->totalVoxels(), torch::TensorOptions().device(cuda)).remainder(3) != 0);

    const nanovdb::Coord k3(3), s1(1), s2(2);
    std::vector<std::pair<const char *, c10::intrusive_ptr<GridBatchData>>> batches = {
        {"createNanoGridFromIJK", built},
        {"buildGridFromPoints", ops::buildGridFromPoints(pointsJt, unit2, zero2)},
        {"buildGridFromNearestVoxelsToPoints",
         ops::buildGridFromNearestVoxelsToPoints(pointsJt, unit2, zero2)},
        {"createNanoGridFromDense",
         ops::createNanoGridFromDense(
             2, nanovdb::Coord(-3), nanovdb::Coord(19, 13, 9), cuda, std::nullopt, unit2, zero2)},
        {"buildPaddedGrid", ops::buildPaddedGrid(*built, -1, 1, false, false)},
        {"dilateGrid", ops::dilateGrid(*built, {1, 2})},
        {"buildGridForConv k3 s1", ops::buildGridForConv(*built, k3, s1)},
        {"buildGridForConv k3 s2", ops::buildGridForConv(*built, k3, s2)},
        {"buildGridForConvTranspose k3 s2", ops::buildGridForConvTranspose(*built, k3, s2)},
        {"buildCoarseGridFromFine", ops::buildCoarseGridFromFine(*built, s2)},
        {"buildFineGridFromCoarse", ops::buildFineGridFromCoarse(*built, s2, std::nullopt)},
        {"pruneGrid", ops::pruneGrid(*built, keep)},
        {"mergeGrids", ops::mergeGrids(*built, *ops::buildPaddedGrid(*built, 0, 2, false, false))},
    };

    for (const auto &[name, batch]: batches) {
        ASSERT_TRUE(batch->device().is_cuda()) << name;
        auto expected = ops::activeGridCoords(*ops::cloneGrid(*batch, torch::kCPU));
        for (int64_t bi = 0; bi < batch->batchSize(); ++bi) {
            SCOPED_TRACE(std::string(name) + " grid " + std::to_string(bi));
            ASSERT_GT(batch->numVoxelsAt(bi), 0);
            EXPECT_TRUE(deviceGridIsSequential(*batch, bi));
            expectDecodeParity(*batch, bi, expected.index(bi).jdata());
        }
    }
}

TEST(VbmCacheTest, NegativeGridIndex) {
    if (!cudaIsAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }
    auto batch =
        makeBatch({randomIjk(100, 16, 30), randomIjk(300, 16, 31)}, torch::Device(torch::kCUDA, 0));
    const auto last     = batch->vbmAt(1);
    const auto negative = batch->vbmAt(-1);
    EXPECT_EQ(negative.firstLeafID, last.firstLeafID);
    EXPECT_EQ(negative.jumpMap, last.jumpMap);
    EXPECT_EQ(negative.blockCount, last.blockCount);
    EXPECT_ANY_THROW(batch->vbmAt(2));
}
