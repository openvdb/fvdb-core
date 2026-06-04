// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_CUDA_ALIGNMENT_CUH
#define FVDB_DETAIL_UTILS_CUDA_ALIGNMENT_CUH

#include <cstddef>
#include <cstdint>

namespace fvdb {
namespace detail {

/// Round up a byte count to the next multiple of a power-of-two alignment.
#ifdef __CUDACC__
inline __host__ __device__
#else
inline
#endif
constexpr size_t
alignUpBytes(const size_t value, const size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

/// Advance an address (as uintptr_t) to the next multiple of a power-of-two alignment.
#ifdef __CUDACC__
inline __host__ __device__
#else
inline
#endif
constexpr uintptr_t
alignUpAddress(const uintptr_t value, const size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_UTILS_CUDA_ALIGNMENT_CUH
