// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_CUDA_UTILS_CUH
#define FVDB_DETAIL_UTILS_CUDA_UTILS_CUH

#ifndef CCCL_DEVICE_MERGE_SUPPORTED
#define CCCL_DEVICE_MERGE_SUPPORTED (__CUDACC_VER_MAJOR__ >= 12 && __CUDACC_VER_MINOR__ >= 8)
#endif

#include <c10/cuda/CUDAFunctions.h>

namespace fvdb {

namespace detail {

static constexpr size_t kPageSize = 1u << 21;

template <typename index_t>
inline std::tuple<index_t, index_t>
deviceOffsetAndCount(index_t count, c10::DeviceIndex deviceId) {
    auto deviceCount  = (count + c10::cuda::device_count() - 1) / c10::cuda::device_count();
    auto deviceOffset = deviceCount * deviceId;
    if (deviceOffset + deviceCount > count) {
        deviceOffset = std::min(deviceOffset, count);
        deviceCount  = std::min(deviceCount, count - deviceOffset);
    }
    return std::make_tuple(deviceOffset, deviceCount);
}

inline std::tuple<size_t, size_t>
alignedChunk(size_t alignment, size_t size, c10::DeviceIndex device) {
    size_t chunkSize = alignment * ((size + alignment * c10::cuda::device_count() - 1) /
                       (c10::cuda::device_count() * alignment));
    auto chunkOffset = chunkSize * device;
    if (chunkOffset + chunkSize > size) {
        chunkOffset = std::min(chunkOffset, size);
        chunkSize   = std::min(chunkSize, size - chunkOffset);
    }
    return std::make_tuple(chunkOffset, chunkSize);
}

} // namespace detail

} // namespace fvdb

#endif // FVDB_DETAIL_UTILS_CUDA_UTILS_CUH
