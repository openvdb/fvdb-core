// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <fvdb/GridStorage.h>
#include <fvdb/TorchDeviceResource.h>
#include <fvdb/detail/utils/nanovdb/CreateEmptyGridStorage.h>
#include <fvdb/detail/utils/nanovdb/DeviceGridHandleUtils.cuh>
#include <fvdb/detail/utils/nanovdb/GridHeaderUtils.h>

#include <nanovdb/GridHandle.h>
#include <nanovdb/HostBuffer.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/GridBuilder.h>

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/types.h>

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace fvdb;

namespace {

const torch::Device kCuda0(torch::kCUDA, 0);

using HostHandle   = GridStorage::HostHandle;
using DeviceHandle = GridStorage::DeviceHandle;

// A small float grid with `n` voxels along a diagonal, value = coordinate sum + base. Checksums
// are disabled so that host- and device-assembled batches are byte-identical (the header fixups
// on merge disable them anyway).
HostHandle
makeFloatGrid(int n, float base) {
    nanovdb::tools::build::Grid<float> grid(0.0f, "g" + std::to_string(n));
    auto acc = grid.getAccessor();
    for (int i = 0; i < n; ++i) {
        acc.setValue(nanovdb::Coord(i, 2 * i, 3 * i), base + float(6 * i));
    }
    return nanovdb::tools::
        createNanoGrid<nanovdb::tools::build::Grid<float>, float, nanovdb::HostBuffer>(
            grid, nanovdb::tools::StatsMode::Default, nanovdb::CheckMode::Disable);
}

HostHandle
makeBatch(std::vector<int> sizes) {
    std::vector<HostHandle> parts;
    for (size_t i = 0; i < sizes.size(); ++i) {
        parts.push_back(makeFloatGrid(sizes[i], 10.0f * float(i + 1)));
    }
    return nanovdb::mergeGrids(parts);
}

std::vector<uint8_t>
hostBytesOf(const GridStorage &s) {
    if (s.hostBytes() == nullptr) {
        throw std::runtime_error("storage is not host-readable");
    }
    std::vector<uint8_t> out(s.bufferSize());
    std::memcpy(out.data(), s.hostBytes(), s.bufferSize());
    return out;
}

std::vector<uint8_t>
deviceBytesOf(const GridStorage &s) {
    std::vector<uint8_t> out(s.bufferSize());
    if (s.bufferSize()) {
        if (cudaStreamSynchronize(s.stream()) != cudaSuccess ||
            cudaMemcpy(out.data(), s.deviceBytes(), s.bufferSize(), cudaMemcpyDeviceToHost) !=
                cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(cudaGetLastError()));
        }
    }
    return out;
}

__global__ void
spinThenFill(uint8_t *ptr, uint64_t n, uint8_t value, long long cycles) {
    const long long start = clock64();
    while (clock64() - start < cycles) {}
    if (ptr) {
        for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
             i += gridDim.x * blockDim.x) {
            ptr[i] = value;
        }
    }
}
constexpr long long kSpinCycles = 500'000'000; // ~0.25 s

void
initTorchCudaAllocator() {
    (void)torch::empty({1}, torch::TensorOptions().device(kCuda0));
}

// ---------------------------------------------------------------------------------------------

TEST(GridStorageTest, HostAccessorsAnswerFromMetadata) {
    GridStorage s(makeBatch({4, 7}));
    EXPECT_TRUE(s.isHost());
    EXPECT_TRUE(s.device().is_cpu());
    ASSERT_EQ(s.gridCount(), 2u);
    EXPECT_EQ(s.gridOffset(0), 0u);
    EXPECT_EQ(s.gridOffset(1), s.gridSize(0));
    EXPECT_EQ(s.gridSize(0) + s.gridSize(1), s.bufferSize());
    EXPECT_EQ(s.gridType(0), nanovdb::GridType::Float);
    EXPECT_EQ(s.gridType(1), nanovdb::GridType::Float);
    EXPECT_NE(s.hostBytes(), nullptr);
    EXPECT_EQ(s.deviceBytes(), nullptr);
    EXPECT_EQ(s.deviceGridAt<float>(0), nullptr);
    EXPECT_EQ(s.hostGridAt<double>(0), nullptr) << "wrong value type must not match";
    EXPECT_EQ(s.hostGridAt<float>(2), nullptr) << "out of range must not match";

    const auto *g1 = s.hostGridAt<float>(1);
    ASSERT_NE(g1, nullptr);
    EXPECT_EQ(g1->gridIndex(), 1u);
    EXPECT_EQ(g1->gridCount(), 2u);
    EXPECT_FLOAT_EQ(g1->tree().getValue(nanovdb::Coord(3, 6, 9)), 20.0f + 18.0f);
    EXPECT_EQ(s.stream(), cudaStream_t{});
}

TEST(GridStorageTest, EmptyStorageKeepsItsDevice) {
    auto cpu = GridStorage::empty(torch::kCPU);
    EXPECT_TRUE(cpu.isEmpty());
    EXPECT_TRUE(cpu.device().is_cpu());
    EXPECT_EQ(cpu.gridCount(), 0u);

    auto dev = GridStorage::empty(kCuda0);
    EXPECT_TRUE(dev.isEmpty());
    EXPECT_TRUE(dev.isDevice());
    EXPECT_EQ(dev.device(), kCuda0);
    EXPECT_EQ(dev.deviceHandle().buffer().resource().device(), kCuda0)
        << "the exposed buffer must carry the device too, for anyone who re-wraps or protos it";
    EXPECT_EQ(GridStorage(std::move(dev.deviceHandle())).device(), kCuda0);
    EXPECT_EQ(dev.deviceBytes(), nullptr);
    EXPECT_EQ(dev.hostBytes(), nullptr);

    // Moves of empty storage allocate nothing and land on the requested device (cuda::copyTo
    // alone would return a handle on the *current* device).
    auto moved = cpu.to(kCuda0);
    EXPECT_TRUE(moved.isEmpty());
    EXPECT_EQ(moved.device(), kCuda0);
    auto side        = c10::cuda::getStreamFromPool(false, 0);
    auto movedOnSide = cpu.to(kCuda0, side.stream());
    EXPECT_EQ(movedOnSide.stream(), side.stream())
        << "an empty move still retains the requested stream";
    EXPECT_EQ(movedOnSide.deviceHandle().buffer().resource().device(), kCuda0);
    auto back = dev.to(torch::kCPU);
    EXPECT_TRUE(back.isEmpty());
    EXPECT_TRUE(back.device().is_cpu());
}

TEST(GridStorageTest, DeviceProtoCarriesDeviceAndStream) {
    initTorchCudaAllocator();
    auto side  = c10::cuda::getStreamFromPool(false, 0);
    auto proto = GridStorage::deviceProto(kCuda0, side.stream());
    EXPECT_EQ(proto.size_bytes(), 0u);
    EXPECT_EQ(proto.data(), nullptr);
    EXPECT_EQ(proto.stream(), side.stream());
    EXPECT_EQ(proto.resource().device(), kCuda0);

    // What the builders do with it: allocate the output through the proto's resource.
    auto out = nanovdb::cuda::detail::createDeviceStorage<DeviceGridBuffer>(
        4096, &proto, 0, side.stream());
    EXPECT_EQ(out.size_bytes(), 4096u);
    EXPECT_NE(out.data(), nullptr);
    EXPECT_EQ(out.stream(), side.stream());
    EXPECT_EQ(out.resource().device(), kCuda0);
}

TEST(GridStorageTest, RoundTripThroughCudaPreservesBytesAndMetadata) {
    initTorchCudaAllocator();
    GridStorage host(makeBatch({3, 5, 2}));
    const auto original = hostBytesOf(host);
    const auto stream   = c10::cuda::getCurrentCUDAStream(0).stream();

    GridStorage dev = host.to(kCuda0, stream);
    EXPECT_TRUE(dev.isDevice());
    EXPECT_EQ(dev.device(), kCuda0);
    EXPECT_EQ(dev.stream(), stream);
    EXPECT_EQ(dev.bufferSize(), host.bufferSize());
    ASSERT_EQ(dev.gridCount(), 3u);
    for (uint32_t i = 0; i < 3; ++i) {
        EXPECT_EQ(dev.gridSize(i), host.gridSize(i));
        EXPECT_EQ(dev.gridOffset(i), host.gridOffset(i));
        EXPECT_EQ(dev.gridType(i), host.gridType(i));
    }
    EXPECT_EQ(dev.hostBytes(), nullptr);
    EXPECT_NE(dev.deviceBytes(), nullptr);
    EXPECT_NE(dev.deviceGridAt<float>(2), nullptr);
    EXPECT_EQ(dev.hostGridAt<float>(2), nullptr);
    EXPECT_EQ(deviceBytesOf(dev), original);

    GridStorage back = dev.to(torch::kCPU);
    EXPECT_TRUE(back.isHost());
    EXPECT_EQ(hostBytesOf(back), original);

    // Same-device copies are deep copies on both sides.
    GridStorage dev2 = dev.to(kCuda0);
    EXPECT_NE(dev2.deviceBytes(), dev.deviceBytes());
    EXPECT_EQ(deviceBytesOf(dev2), original);
    GridStorage host2 = host.to(torch::kCPU);
    EXPECT_NE(host2.hostBytes(), host.hostBytes());
    EXPECT_EQ(hostBytesOf(host2), original);
}

TEST(GridStorageTest, ExplicitStreamZeroWhileNonDefaultStreamIsCurrent) {
    initTorchCudaAllocator();
    GridStorage host(makeBatch({6}));
    const auto original = hostBytesOf(host);
    auto side           = c10::cuda::getStreamFromPool(false, 0);
    c10::cuda::CUDAStreamGuard streamGuard(side);

    GridStorage dev = host.to(kCuda0, cudaStream_t{});
    EXPECT_EQ(dev.stream(), cudaStream_t{}) << "an explicit legacy stream is honoured as retained";
    EXPECT_EQ(deviceBytesOf(dev), original);

    GridStorage devCurrent = host.to(kCuda0);
    EXPECT_EQ(devCurrent.stream(), side.stream())
        << "the omitted stream is the current torch stream";
    EXPECT_EQ(deviceBytesOf(devCurrent), original);

    GridStorage back = dev.to(torch::kCPU, cudaStream_t{});
    EXPECT_EQ(hostBytesOf(back), original);
}

// Contract test from the #770 design review: after to() returns, destroying the source must be
// safe even though the device-to-device copy may still be in flight on another stream, and even
// when the source's retained stream is not torch's current stream (its block is keyed to the
// retained stream, so that is where torch will hand it out again). The copy stream is held open by
// a spin kernel so the copy is deterministically pending; the source is destroyed; a same-sized
// block is allocated on the source's retained stream (which, with caching, gets the source's bytes
// back) and overwritten there. Without the source-stream-after-copy ordering that overwrite would
// land under the copy.
TEST(GridStorageTest, SourceMayBeDestroyedRightAfterAsyncCopy) {
    initTorchCudaAllocator();
    GridStorage host(makeBatch({64, 64}));
    const auto original = hostBytesOf(host);
    const auto current  = c10::cuda::getCurrentCUDAStream(0);
    auto srcStream      = c10::cuda::getStreamFromPool(false, 0); // retained by the source
    auto copyStream     = c10::cuda::getStreamFromPool(false, 0);
    ASSERT_NE(srcStream.stream(), current.stream());
    ASSERT_NE(srcStream.stream(), copyStream.stream());

    auto src = std::make_unique<GridStorage>(host.to(kCuda0, srcStream.stream()));
    ASSERT_EQ(src->stream(), srcStream.stream());
    ASSERT_EQ(cudaStreamSynchronize(srcStream.stream()), cudaSuccess);
    const void *srcPtr    = src->deviceBytes();
    const uint64_t nbytes = src->bufferSize();

    spinThenFill<<<1, 1, 0, copyStream.stream()>>>(nullptr, 0, 0, kSpinCycles);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    GridStorage dst = src->to(kCuda0, copyStream.stream()); // queued behind the spin
    if (cudaStreamQuery(copyStream.stream()) != cudaErrorNotReady) {
        // The destination allocation synchronized the device (caching disabled: a raw
        // cudaMalloc), or the host was descheduled past the spin; the copy has landed and there
        // is nothing to observe.
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        GTEST_SKIP() << "copy completed before the source could be destroyed under it";
    }

    src.reset(); // frees on srcStream, where torch may reuse the block immediately

    DeviceGridBuffer reuse(
        srcStream.stream(), TorchDeviceResource(kCuda0), nbytes, nanovdb::cuda::noInit);
    if (reuse.data() != srcPtr) {
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        GTEST_SKIP() << "allocator did not hand the source block back; nothing to observe";
    }
    ASSERT_EQ(cudaMemsetAsync(reuse.data(), 0xEE, nbytes, srcStream.stream()), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(srcStream.stream()), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(copyStream.stream()), cudaSuccess);

    EXPECT_EQ(deviceBytesOf(dst), original) << "the copy read bytes the source's successor wrote";
}

// The block is keyed to the buffer's retained stream: a buffer freed on stream B is handed back to
// the next allocation on B, not to one on torch's current stream.
TEST(GridStorageTest, StorageBlockReturnsToItsRetainedStream) {
    if (c10::cuda::CUDACachingAllocator::name() != "native") {
        GTEST_SKIP() << "only the native caching allocator keys free blocks by stream";
    }
    for (const char *name: {"PYTORCH_NO_CUDA_MEMORY_CACHING", "PYTORCH_NO_HIP_MEMORY_CACHING"}) {
        if (const char *env = std::getenv(name); env && *env && *env != '0') {
            GTEST_SKIP()
                << "without caching every allocation is a fresh cudaMalloc; nothing is keyed";
        }
    }
    initTorchCudaAllocator();
    auto side = c10::cuda::getStreamFromPool(false, 0);
    ASSERT_NE(side.stream(), c10::cuda::getCurrentCUDAStream(0).stream());
    constexpr uint64_t nbytes = 1u << 20;
    const void *first         = nullptr;
    {
        DeviceGridBuffer a(
            side.stream(), TorchDeviceResource(kCuda0), nbytes, nanovdb::cuda::noInit);
        first = a.data();
    }
    DeviceGridBuffer onCurrent(c10::cuda::getCurrentCUDAStream(0).stream(),
                               TorchDeviceResource(kCuda0),
                               nbytes,
                               nanovdb::cuda::noInit);
    DeviceGridBuffer onSide(
        side.stream(), TorchDeviceResource(kCuda0), nbytes, nanovdb::cuda::noInit);
    if (onSide.data() != first && onCurrent.data() != first) {
        GTEST_SKIP() << "allocator did not reuse the block at all (caching disabled?)";
    }
    EXPECT_EQ(onSide.data(), first) << "the freed block should be reusable on its retained stream";
    EXPECT_NE(onCurrent.data(), first) << "and not on an unrelated stream";
}

TEST(GridStorageTest, SynchronousResourceApiWorksUnderAnyCurrentDevice) {
    initTorchCudaAllocator();
    TorchDeviceResource r(kCuda0);
    void *p = r.allocate(4096, TorchDeviceResource::DEFAULT_ALIGNMENT);
    ASSERT_NE(p, nullptr);
    ASSERT_EQ(cudaMemset(p, 0, 4096), cudaSuccess);
    r.deallocate(p, 4096, TorchDeviceResource::DEFAULT_ALIGNMENT);
}

TEST(GridStorageTest, MergeGridStoragesMatchesHostMerge) {
    initTorchCudaAllocator();
    const auto stream = c10::cuda::getCurrentCUDAStream(0).stream();
    std::vector<int> sizes{5, 1, 9};
    std::vector<HostHandle> hostParts;
    std::vector<GridStorage> devParts;
    std::vector<GridStorage> cpuParts;
    for (size_t i = 0; i < sizes.size(); ++i) {
        hostParts.push_back(makeFloatGrid(sizes[i], 10.0f * float(i + 1)));
        cpuParts.emplace_back(makeFloatGrid(sizes[i], 10.0f * float(i + 1)));
        devParts.push_back(
            GridStorage(makeFloatGrid(sizes[i], 10.0f * float(i + 1))).to(kCuda0, stream));
    }
    // Also empty storage in the middle, which contributes nothing.
    devParts.insert(devParts.begin() + 1, GridStorage::empty(kCuda0));

    const HostHandle expected = nanovdb::mergeGrids(hostParts);
    std::vector<uint8_t> expectedBytes(expected.bufferSize());
    std::memcpy(expectedBytes.data(), expected.data(), expected.bufferSize());

    GridStorage merged = fvdb::detail::mergeGridStorages(std::move(devParts), kCuda0, stream);
    ASSERT_TRUE(merged.isDevice());
    EXPECT_EQ(merged.device(), kCuda0);
    EXPECT_EQ(merged.stream(), stream);
    ASSERT_EQ(merged.gridCount(), 3u);
    EXPECT_EQ(merged.bufferSize(), expected.bufferSize());
    for (uint32_t i = 0; i < 3; ++i) {
        EXPECT_EQ(merged.gridSize(i), expected.gridSize(i));
        EXPECT_EQ(merged.gridType(i), expected.gridType(i));
    }
    EXPECT_EQ(deviceBytesOf(merged), expectedBytes);

    // The same over host parts onto the host, as the CPU builders could use it.
    GridStorage mergedCpu =
        fvdb::detail::mergeGridStorages(std::move(cpuParts), torch::kCPU, cudaStream_t{});
    ASSERT_TRUE(mergedCpu.isHost());
    ASSERT_EQ(mergedCpu.gridCount(), 3u);
    ASSERT_EQ(mergedCpu.bufferSize(), expected.bufferSize());
    EXPECT_EQ(std::memcmp(mergedCpu.hostBytes(), expectedBytes.data(), expectedBytes.size()), 0);

    // A single part is returned as is, whatever it holds.
    std::vector<GridStorage> one;
    one.emplace_back(GridStorage(makeFloatGrid(3, 1.0f)).to(kCuda0, stream));
    const void *before = one[0].deviceBytes();
    GridStorage same   = fvdb::detail::mergeGridStorages(std::move(one), kCuda0, stream);
    EXPECT_EQ(same.deviceBytes(), before);

    // All-empty input keeps the destination device and stream rather than a default resource.
    std::vector<GridStorage> none;
    none.emplace_back(GridStorage::empty(kCuda0));
    none.emplace_back(GridStorage::empty(kCuda0));
    GridStorage emptyMerge = fvdb::detail::mergeGridStorages(std::move(none), kCuda0, stream);
    EXPECT_TRUE(emptyMerge.isEmpty());
    EXPECT_EQ(emptyMerge.device(), kCuda0);
    EXPECT_EQ(emptyMerge.stream(), stream);

    // And it parses: constructing a handle from the raw bytes validates the whole chain.
    auto bytes = merged.deviceHandle().buffer().copy(stream);
    ASSERT_NO_THROW({
        DeviceHandle reparsed(std::move(bytes));
        EXPECT_EQ(reparsed.gridCount(), 3u);
    });
}

TEST(GridStorageTest, GridStoragePartsSharesEmptyAndRepeats) {
    initTorchCudaAllocator();
    const auto stream = c10::cuda::getCurrentCUDAStream(0).stream();
    // Expected: [empty, A, A, empty, B] assembled on the host by NanoVDB.
    std::vector<HostHandle> hostParts;
    hostParts.push_back(fvdb::detail::createEmptyGridStorage(torch::kCPU).hostHandle().copy());
    hostParts.push_back(makeFloatGrid(4, 1.0f));
    hostParts.push_back(makeFloatGrid(4, 1.0f));
    hostParts.push_back(fvdb::detail::createEmptyGridStorage(torch::kCPU).hostHandle().copy());
    hostParts.push_back(makeFloatGrid(7, 2.0f));
    HostHandle expected = nanovdb::mergeGrids(hostParts);
    // The empty grids carry a checksum from createNanoGrid; assembled storage disables them.
    for (uint32_t i = 0; i < expected.gridCount(); ++i) {
        const_cast<nanovdb::GridData *>(expected.gridData(i))->mChecksum.disable();
    }

    fvdb::detail::GridStorageParts parts(kCuda0, stream, 2);
    parts.addEmpty();
    const size_t a = parts.add(GridStorage(makeFloatGrid(4, 1.0f)).to(kCuda0, stream));
    parts.addRepeat(a);
    parts.addEmpty();
    parts.add(GridStorage(makeFloatGrid(7, 2.0f)).to(kCuda0, stream));
    ASSERT_EQ(parts.size(), 5u);
    GridStorage merged = parts.merge();

    ASSERT_EQ(merged.gridCount(), 5u);
    EXPECT_EQ(merged.bufferSize(), expected.bufferSize());
    for (uint32_t i = 0; i < 5; ++i) {
        EXPECT_EQ(merged.gridSize(i), expected.gridSize(i));
        EXPECT_EQ(merged.gridType(i), expected.gridType(i));
    }
    std::vector<uint8_t> expectedBytes(expected.bufferSize());
    std::memcpy(expectedBytes.data(), expected.data(), expected.bufferSize());
    EXPECT_EQ(deviceBytesOf(merged), expectedBytes);

    // A lone shared empty item is handed out as the batch itself.
    fvdb::detail::GridStorageParts one(kCuda0, stream, 0);
    one.addEmpty();
    GridStorage lone = one.merge();
    EXPECT_EQ(lone.gridCount(), 1u);
    EXPECT_EQ(lone.device(), kCuda0);
}

TEST(GridStorageTest, MakeDeviceHandleFromLayoutChecksExtent) {
    initTorchCudaAllocator();
    const auto stream = c10::cuda::getCurrentCUDAStream(0).stream();
    GridStorage dev   = GridStorage(makeBatch({2, 3})).to(kCuda0, stream);
    auto meta         = dev.metadata();

    auto good =
        fvdb::detail::makeDeviceHandleFromLayout(dev.deviceHandle().buffer().copy(stream), meta);
    EXPECT_EQ(good.gridCount(), 2u);
    EXPECT_EQ(good.gridSize(1), dev.gridSize(1));

    meta[1].size += 64; // runs past the buffer
    EXPECT_THROW(
        fvdb::detail::makeDeviceHandleFromLayout(dev.deviceHandle().buffer().copy(stream), meta),
        c10::Error);

    meta           = dev.metadata();
    meta[1].offset = ~uint64_t{0} - 31; // offset + size wraps to a small number
    EXPECT_THROW(
        fvdb::detail::makeDeviceHandleFromLayout(dev.deviceHandle().buffer().copy(stream), meta),
        c10::Error);

    meta = dev.metadata();
    meta[1].offset += NANOVDB_DATA_ALIGNMENT; // a gap: the chain parse would not find grid 1
    EXPECT_THROW(
        fvdb::detail::makeDeviceHandleFromLayout(dev.deviceHandle().buffer().copy(stream), meta),
        c10::Error);

    auto packed = dev.deviceHandle().buffer().copy(stream);
    EXPECT_THROW(fvdb::detail::makeDeviceHandleFromLayout(std::move(packed), {}), c10::Error)
        << "no metadata over a non-empty buffer";
}

TEST(GridStorageTest, NormalizeStandaloneGridHeaderMakesAGridWrappable) {
    GridStorage batch(makeBatch({3, 4}));
    ASSERT_EQ(batch.gridCount(), 2u);
    const uint64_t size   = batch.gridSize(1);
    const uint64_t offset = batch.gridOffset(1);

    auto copyOut = [&]() {
        nanovdb::HostBuffer buf(size);
        std::memcpy(buf.data(), static_cast<const uint8_t *>(batch.hostBytes()) + offset, size);
        return buf;
    };

    // Straight out of the batch the header still says "grid 1 of 2": GridHandle rejects it.
    EXPECT_THROW(
        {
            HostHandle rejected{copyOut()};
            (void)rejected;
        },
        std::runtime_error);

    nanovdb::HostBuffer buf = copyOut();
    auto *data              = reinterpret_cast<nanovdb::GridData *>(buf.data());
    fvdb::detail::normalizeStandaloneGridHeader(data, size);
    EXPECT_EQ(data->mGridIndex, 0u);
    EXPECT_EQ(data->mGridCount, 1u);
    EXPECT_EQ(data->mGridSize, size);
    EXPECT_EQ(data->mBlindMetadataCount, 0u);
    EXPECT_EQ(data->mBlindMetadataOffset, size);
    EXPECT_TRUE(data->mChecksum.isEmpty());

    HostHandle standalone(std::move(buf));
    ASSERT_EQ(standalone.gridCount(), 1u);
    EXPECT_EQ(standalone.gridSize(0), size);
    GridStorage s(std::move(standalone));
    const auto *g = s.hostGridAt<float>(0);
    ASSERT_NE(g, nullptr);
    EXPECT_FLOAT_EQ(g->tree().getValue(nanovdb::Coord(3, 6, 9)), 20.0f + 18.0f);
}

} // namespace
