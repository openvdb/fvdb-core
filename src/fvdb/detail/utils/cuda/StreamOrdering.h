// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_CUDA_STREAMORDERING_H
#define FVDB_DETAIL_UTILS_CUDA_STREAMORDERING_H

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/types.h>

#include <cuda_runtime_api.h>

namespace fvdb::detail {

/// @brief Waits for everything currently enqueued on @p stream of @p device. A null handle names
///        that device's default stream, so the synchronize runs under its guard; for PrivateUse1,
///        whose null stream has no single device behind it, every device is synchronized.
inline void
synchronizeStream(cudaStream_t stream, const torch::Device &device) {
    if (stream != nullptr) {
        C10_CUDA_CHECK(cudaStreamSynchronize(stream));
        return;
    }
    if (device.is_cuda()) {
        c10::cuda::CUDAGuard deviceGuard(device);
        C10_CUDA_CHECK(cudaStreamSynchronize(stream));
        return;
    }
    for (c10::DeviceIndex i = 0; i < c10::cuda::device_count(); ++i) {
        c10::cuda::CUDAGuard deviceGuard(i);
        C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    }
}

/// @brief Orders @p waiter after everything currently enqueued on @p signaler, where the two
///        streams belong to the given devices. A no-op when they are the same stream on the same
///        device (two null handles on different devices are different streams). The event is
///        created, recorded and destroyed under @p signalerDevice, which CUDA requires for the
///        record, and the wait is enqueued under @p waiterDevice, so that a null handle names the
///        right device's default stream. Raw CUDA calls rather than c10::cuda::CUDAEvent so that
///        the cudaStreamLegacy and cudaStreamPerThread sentinels, which c10's stream conversion
///        rejects, work like any other handle.
///
///        PrivateUse1 storage is unified memory whose stream has no CUDA device to record an event
///        under; when either side is PrivateUse1 the signaler is synchronized instead.
inline void
orderStreamAfter(cudaStream_t waiter,
                 const torch::Device &waiterDevice,
                 cudaStream_t signaler,
                 const torch::Device &signalerDevice) {
    // PrivateUse1 first: two null handles on PrivateUse1 need not be the same GPU's default
    // stream, so the equality shortcut below must not apply to them.
    if (!signalerDevice.is_cuda() || !waiterDevice.is_cuda()) {
        synchronizeStream(signaler, signalerDevice);
        return;
    }
    if (waiter == signaler && waiterDevice == signalerDevice) {
        return;
    }
    cudaEvent_t ev = nullptr;
    cudaError_t err;
    {
        c10::cuda::CUDAGuard deviceGuard(signalerDevice);
        C10_CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
        err = cudaEventRecord(ev, signaler);
    }
    if (err == cudaSuccess) {
        c10::cuda::CUDAGuard deviceGuard(waiterDevice);
        err = cudaStreamWaitEvent(waiter, ev, 0);
    }
    cudaError_t destroyErr;
    {
        c10::cuda::CUDAGuard deviceGuard(signalerDevice);
        destroyErr = cudaEventDestroy(ev);
    }
    C10_CUDA_CHECK(err == cudaSuccess ? destroyErr : err);
}

} // namespace fvdb::detail

#endif // FVDB_DETAIL_UTILS_CUDA_STREAMORDERING_H
