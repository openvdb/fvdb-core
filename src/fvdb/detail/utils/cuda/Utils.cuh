// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_CUDA_UTILS_CUH
#define FVDB_DETAIL_UTILS_CUDA_UTILS_CUH

#ifndef CCCL_DEVICE_MERGE_SUPPORTED
#define CCCL_DEVICE_MERGE_SUPPORTED (__CUDACC_VER_MAJOR__ >= 12 && __CUDACC_VER_MINOR__ >= 8)
#endif

#include <c10/cuda/CUDAStream.h>

namespace fvdb {

namespace detail {

static constexpr size_t kPageSize = 1u << 21;

inline std::tuple<size_t, size_t>
deviceAlignedChunk(size_t alignment, size_t size, c10::DeviceIndex device) {
    size_t chunkSize = alignment * ((size + alignment * c10::cuda::device_count() - 1) /
                                    (c10::cuda::device_count() * alignment));
    auto chunkOffset = chunkSize * device;
    if (chunkOffset + chunkSize > size) {
        chunkOffset = std::min(chunkOffset, size);
        chunkSize   = std::min(chunkSize, size - chunkOffset);
    }
    return std::make_tuple(chunkOffset, chunkSize);
}

inline std::tuple<size_t, size_t>
deviceChunk(size_t size, c10::DeviceIndex device) {
    return deviceAlignedChunk(1, size, device);
}

inline void
mergeStreams() {
    constexpr int mergeDeviceId = 0;
    cudaEvent_t mergeEvent      = 0;
    std::vector<cudaEvent_t> events(c10::cuda::device_count());

    // Create an event for each device and record it in their respective streams
    for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        auto stream = c10::cuda::getCurrentCUDAStream(deviceId);
        C10_CUDA_CHECK(cudaEventCreate(&events[deviceId], cudaEventDisableTiming));
        C10_CUDA_CHECK(cudaEventRecord(events[deviceId], stream));
    }

    // Create an event on the merge device
    C10_CUDA_CHECK(cudaSetDevice(mergeDeviceId));
    C10_CUDA_CHECK(cudaEventCreate(&mergeEvent, cudaEventDisableTiming));
    auto mergeStream = c10::cuda::getCurrentCUDAStream(mergeDeviceId);
    // On the merge stream, wait until the per-device events have completed
    for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
        C10_CUDA_CHECK(cudaStreamWaitEvent(mergeStream, events[deviceId]));
    }
    // Record an event on the merge stream to signify that the per-device events have been merged
    C10_CUDA_CHECK(cudaEventRecord(mergeEvent, mergeStream));

    // On each per-device stream, wait on the merge event
    for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        auto stream = c10::cuda::getCurrentCUDAStream(deviceId);
        C10_CUDA_CHECK(cudaStreamWaitEvent(stream, mergeEvent));
    }

    // Destroy events
    for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
        C10_CUDA_CHECK(cudaEventDestroy(events[deviceId]));
    }
    C10_CUDA_CHECK(cudaEventDestroy(mergeEvent));
}

/// @brief Get the maximum shared memory per block optin for a given device
///
/// This function memoizes the result for each device to minimize calls to
/// cudaDeviceGetAttribute.
///
/// This function is thread-safe.
///
/// @param device The device to get the maximum shared memory per block optin for
/// @return The maximum shared memory per block optin in bytes
inline int
getMaxSharedMemory(int device) {
    TORCH_CHECK(device >= 0, "Invalid device index: ", device);

    static const int deviceCount = c10::cuda::device_count();
    TORCH_CHECK(device < deviceCount, "Device index out of range: ", device);

    static std::vector<std::atomic<int>> cache(deviceCount);
    static std::vector<std::once_flag> initFlags(deviceCount);

    int cached = cache[device].load(std::memory_order_relaxed);
    if (cached > 0) {
        return cached;
    }

    std::call_once(initFlags[device], [device]() {
        int maxSharedMemory = 0;
        TORCH_CHECK(cudaDeviceGetAttribute(&maxSharedMemory,
                                           cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                           device) == cudaSuccess,
                    "Failed to get max shared memory per block");
        cache[device].store(maxSharedMemory, std::memory_order_relaxed);
    });

    cached = cache[device].load(std::memory_order_relaxed);
    TORCH_CHECK(cached > 0, "Failed to initialize max shared memory for device ", device);
    return cached;
};

} // namespace detail

} // namespace fvdb

#endif // FVDB_DETAIL_UTILS_CUDA_UTILS_CUH
