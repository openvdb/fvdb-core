// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_CUDA_UTILS_CUH
#define FVDB_DETAIL_UTILS_CUDA_UTILS_CUH

#ifndef CCCL_DEVICE_MERGE_SUPPORTED
#define CCCL_DEVICE_MERGE_SUPPORTED (__CUDACC_VER_MAJOR__ >= 12 && __CUDACC_VER_MINOR__ >= 8)
#endif

#include <c10/cuda/CUDAFunctions.h>

template <typename index_t>
inline std::tuple<index_t, index_t>
deviceOffsetAndCount(index_t count, c10::DeviceIndex deviceId) {
    auto deviceCount        = (count + c10::cuda::device_count() - 1) / c10::cuda::device_count();
    const auto deviceOffset = deviceCount * deviceId;
    deviceCount             = std::min(deviceCount, count - deviceOffset);
    return std::make_tuple(deviceOffset, deviceCount);
}

#endif // FVDB_DETAIL_UTILS_CUDA_UTILS_CUH
