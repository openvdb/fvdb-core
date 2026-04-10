// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_CUDA_CUBWRAPPER_CUH
#define FVDB_DETAIL_UTILS_CUDA_CUBWRAPPER_CUH

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>

/// Convenience wrapper for CUB API calls that require a temp-storage probe + allocation
/// pattern. Usage:
///   CUB_WRAPPER(cub::DeviceRadixSort::SortPairs, keys, values, N, 0, sizeof(int64_t) * 8, stream);
#define CUB_WRAPPER(func, ...)                                                                     \
    do {                                                                                           \
        size_t tempStorageBytes = 0;                                                               \
        C10_CUDA_CHECK(func(nullptr, tempStorageBytes, __VA_ARGS__));                              \
        auto &cachingAllocator = *::c10::cuda::CUDACachingAllocator::get();                        \
        auto tempStorage =                                                                         \
            tempStorageBytes > 0 ? cachingAllocator.allocate(tempStorageBytes) : ::c10::DataPtr(); \
        C10_CUDA_CHECK(func(tempStorage.get(), tempStorageBytes, __VA_ARGS__));                    \
    } while (false)

#endif // FVDB_DETAIL_UTILS_CUDA_CUBWRAPPER_CUH
