// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#include <fvdb/detail/utils/cuda/Prefetch.h>

#include <nanovdb/util/cuda/Util.h>

#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>

#include <cuda/std/chrono>

namespace fvdb {
namespace detail {

namespace {

__global__ void
prefetchDelayKernel() {
    // The sleep-launch plus prefetch API sequence has a 34-39 us p99 in bonsai_dgx_1 through 4.
    // Start with 50 us of pending work to reduce the idle-stream host submission cliff.
    constexpr int kPrefetchDelayMicroseconds = 50;

    // The device clock measures elapsed time independently of SM clock frequency. A single
    // __nanosleep call can return early, so check elapsed time before allowing prefetch to run.
    const auto start = cuda::std::chrono::system_clock::now();
    do {
        __nanosleep(1000);
    } while (cuda::std::chrono::system_clock::now() - start <
             cuda::std::chrono::microseconds(kPrefetchDelayMicroseconds));
}

} // namespace

void
memDiscardAndPrefetchBatchAsync(std::vector<void *> &prefetchPointers,
                                std::vector<size_t> &prefetchSizes,
                                int deviceId,
                                cudaStream_t stream) {
    TORCH_CHECK(stream, "cudaMemDiscardAndPrefetchBatchAsync does not support the default stream");
    if (prefetchPointers.empty()) {
        return;
    }

#if (CUDART_VERSION < 13000)
    for (size_t i = 0; i < prefetchPointers.size(); ++i) {
        prefetchDelayKernel<<<1, 1, 0, stream>>>();
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        C10_CUDA_CHECK(nanovdb::util::cuda::memPrefetchAsync(
            prefetchPointers[i], prefetchSizes[i], deviceId, stream));
    }
#else
    const cudaMemLocation location                 = {cudaMemLocationTypeDevice, deviceId};
    std::vector<cudaMemLocation> prefetchLocations = {location};
    std::vector<size_t> prefetchLocationIndices    = {0};

    // Prepare all arguments before the delay launch to minimize the host gap before prefetch.
    prefetchDelayKernel<<<1, 1, 0, stream>>>();
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    C10_CUDA_CHECK(cudaMemDiscardAndPrefetchBatchAsync(prefetchPointers.data(),
                                                       prefetchSizes.data(),
                                                       prefetchPointers.size(),
                                                       prefetchLocations.data(),
                                                       prefetchLocationIndices.data(),
                                                       prefetchLocations.size(),
                                                       0,
                                                       stream));
#endif
}

} // namespace detail
} // namespace fvdb
