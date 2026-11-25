// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/utils/cuda/Utils.cuh>

#include <c10/util/irange.h>

#include <vector>

namespace fvdb::detail {

void
mergeStreams() {
    constexpr int mergeDeviceId = 0;
    cudaEvent_t mergeEvent      = 0;
    std::vector<cudaEvent_t> events(c10::cuda::device_count());

    // Create an event for each device and record it in their respective streams
    for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        auto stream = c10::cuda::getCurrentCUDAStream(deviceId);
        C10_CUDA_CHECK(cudaEventCreateWithFlags(&events[deviceId], cudaEventDisableTiming));
        C10_CUDA_CHECK(cudaEventRecord(events[deviceId], stream));
    }

    // Create an event on the merge device
    C10_CUDA_CHECK(cudaSetDevice(mergeDeviceId));
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&mergeEvent, cudaEventDisableTiming));
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

} // namespace fvdb::detail
