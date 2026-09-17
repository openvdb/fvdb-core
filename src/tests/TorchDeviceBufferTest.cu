// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

// This translation unit is the deprecated shim itself (or its test); it is the one place the
// deprecation must not fire.
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#include <fvdb/TorchDeviceBuffer.h>
#include <fvdb/TorchResource.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/types.h>

#include <gtest/gtest.h>

#include <cstring>
#include <vector>

using namespace fvdb;

namespace {

const torch::Device kCuda0(torch::kCUDA, 0);

// Busy-waits for roughly the given number of GPU clock cycles, then fills. Used to hold a stream
// open so that a missing stream dependency shows up deterministically rather than as a race.
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

// ~0.25 s at 2 GHz: far longer than any copy or launch here. The two ordering tests below observe
// stream state while this spin is still running; a host thread descheduled for longer than the
// spin at the wrong moment turns a would-be failure into a pass (never a pass into a failure).
constexpr long long kSpinCycles = 500'000'000;

// The buffer allocates through torch's caching allocator, which torch initializes lazily on the
// first tensor allocation; a standalone gtest binary has made none yet.
void
initTorchCudaAllocator() {
    (void)torch::empty({1}, torch::TensorOptions().device(kCuda0));
}

// A zero-byte buffer owns no allocation, so moving it between devices must not allocate (the
// resources treat a null result from a zero-byte request as failure) and must still record the
// requested device.
TEST(TorchDeviceBufferTest, EmptyBufferMovesToCudaWithoutAllocating) {
    TorchDeviceBuffer buf;
    ASSERT_EQ(buf.size(), 0u);
    ASSERT_TRUE(buf.device().is_cpu());

    ASSERT_NO_THROW(buf.to(kCuda0));
    EXPECT_EQ(buf.size(), 0u);
    EXPECT_EQ(buf.deviceData(), nullptr);
    EXPECT_EQ(buf.device(), kCuda0);

    ASSERT_NO_THROW(buf.to(torch::kCPU));
    EXPECT_EQ(buf.size(), 0u);
    EXPECT_EQ(buf.data(), nullptr);
    EXPECT_TRUE(buf.device().is_cpu());
}

TEST(TorchDeviceBufferTest, EmptyBufferMovesToSameDevice) {
    TorchDeviceBuffer cuda(0, kCuda0);
    ASSERT_NO_THROW(cuda.to(kCuda0));
    EXPECT_EQ(cuda.device(), kCuda0);
    EXPECT_EQ(cuda.size(), 0u);

    TorchDeviceBuffer cpu;
    ASSERT_NO_THROW(cpu.to(torch::kCPU));
    EXPECT_TRUE(cpu.device().is_cpu());
}

// The move must carry the bytes across, and the upload must be complete when to() returns even
// though it runs on the (non-blocking) current torch stream. The side stream is held open by a
// spin kernel before the move, so an upload that was merely enqueued would not have landed when
// the legacy-stream readback below runs.
TEST(TorchDeviceBufferTest, MoveToCudaIsCompleteOnReturn) {
    constexpr uint64_t kBytes = 1u << 20;
    std::vector<uint8_t> pattern(kBytes);
    for (uint64_t i = 0; i < kBytes; ++i) {
        pattern[i] = static_cast<uint8_t>(i * 31u + 7u);
    }
    TorchDeviceBuffer buf(kBytes, torch::kCPU);
    std::memcpy(buf.data(), pattern.data(), kBytes);
    initTorchCudaAllocator();

    auto side = c10::cuda::getStreamFromPool(false, 0);
    {
        c10::cuda::CUDAStreamGuard streamGuard(side);
        spinThenFill<<<1, 1, 0, side.stream()>>>(nullptr, 0, 0, kSpinCycles);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
        buf.to(kCuda0);
    }
    // to() synchronized the side stream, so the spin kernel enqueued ahead of the upload is
    // finished. The byte check below cannot by itself prove that: CUDA permits a pageable-memory
    // upload to complete synchronously, in which case even the spin would be waited out inside
    // the copy call. On the GPUs this was verified on it is not, and removing the
    // synchronization from to() fails this query.
    EXPECT_EQ(cudaStreamQuery(side.stream()), cudaSuccess)
        << "to() returned with work still queued on the stream it copied on";
    EXPECT_EQ(buf.device(), kCuda0);
    ASSERT_EQ(buf.size(), kBytes);
    ASSERT_NE(buf.deviceData(), nullptr);

    std::vector<uint8_t> host(kBytes);
    ASSERT_EQ(cudaMemcpy(host.data(), buf.deviceData(), kBytes, cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(std::memcmp(host.data(), pattern.data(), kBytes), 0);

    buf.to(torch::kCPU);
    ASSERT_TRUE(buf.device().is_cpu());
    ASSERT_NE(buf.data(), nullptr);
    EXPECT_EQ(std::memcmp(buf.data(), pattern.data(), kBytes), 0);
}

// The buffer's stream parameter is the stream the caller will write on; the allocation itself is
// keyed to torch's current stream and the writer must be ordered after it. To observe that, give
// the block a previous tenant whose work is still running on the current stream when the buffer
// is constructed: a kernel that spins and then fills the block. Without the dependency the
// writer's memset would land during the spin and be overwritten by the fill.
TEST(TorchDeviceBufferTest, ConstructWithForeignWriteStreamIsOrderedAfterAllocation) {
    constexpr uint64_t kBytes = 1u << 16;
    initTorchCudaAllocator();
    const auto current = c10::cuda::getCurrentCUDAStream(0);
    auto writer        = c10::cuda::getStreamFromPool(false, 0);
    ASSERT_NE(writer.stream(), current.stream());

    // Previous tenant: allocated on the current stream, still being written when it is freed.
    TorchResource scratch;
    auto *prev = static_cast<uint8_t *>(
        scratch.allocate_async(kBytes, TorchResource::DEFAULT_ALIGNMENT, current.stream()));
    spinThenFill<<<32, 256, 0, current.stream()>>>(prev, kBytes, 0xCD, kSpinCycles);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    scratch.deallocate_async(prev, kBytes, TorchResource::DEFAULT_ALIGNMENT, current.stream());

    TorchDeviceBuffer buf(kBytes, kCuda0, writer.stream());
    ASSERT_NE(buf.deviceData(), nullptr);
    if (buf.deviceData() != prev) {
        // The allocator did not hand the block back (e.g. the cudaMallocAsync backend, whose
        // frees are stream-ordered). Nothing to observe.
        ASSERT_EQ(cudaStreamSynchronize(current.stream()), cudaSuccess);
        GTEST_SKIP() << "allocator did not reuse the previous tenant's block";
    }

    if (cudaStreamQuery(current.stream()) != cudaErrorNotReady) {
        // The previous tenant already finished (this thread was descheduled for longer than the
        // spin), so an unordered memset could not have been overwritten. Nothing to observe.
        ASSERT_EQ(cudaStreamSynchronize(current.stream()), cudaSuccess);
        GTEST_SKIP() << "previous tenant finished before the writer's memset was enqueued";
    }
    ASSERT_EQ(cudaMemsetAsync(buf.deviceData(), 0xAB, kBytes, writer.stream()), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(writer.stream()), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(current.stream()), cudaSuccess);

    std::vector<uint8_t> host(kBytes);
    ASSERT_EQ(cudaMemcpy(host.data(), buf.deviceData(), kBytes, cudaMemcpyDeviceToHost),
              cudaSuccess);
    for (uint64_t i = 0; i < kBytes; i += 4099) {
        ASSERT_EQ(host[i], 0xAB) << "at byte " << i << ": the previous tenant's fill won";
    }
}

} // namespace
