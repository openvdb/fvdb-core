// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_CUDA_BINSEARCH_CUH
#define FVDB_DETAIL_UTILS_CUDA_BINSEARCH_CUH

#include <cstdint>

namespace fvdb {
namespace detail {

/// @brief Binary search on a sorted array for the insertion point of a value.
///
/// Finds the last index where `arr[i] <= val` in a non-decreasing array.
///
/// @pre `len > 0` and `val >= arr[0]`. All current call sites satisfy both
///      preconditions (tile-offset lookups are always within bounds).
///      Passing `len == 0` or `val < arr[0]` returns 0 as a safe fallback,
///      but callers should not rely on this.
///
/// Time complexity: O(log n).
///
/// @tparam T Element type (must support comparison operators).
/// @param arr Pointer to the sorted array.
/// @param len Length of the array.
/// @param val Value to search for.
/// @return Index of the last element <= val, or 0 if preconditions are violated.
template <class T>
inline __device__ uint32_t
binSearch(const T *arr, const uint32_t len, const T val) {
    if (len == 0) {
        return 0;
    }
    uint32_t low = 0, high = len - 1;
    while (low <= high) {
        const uint32_t mid = (low + high) / 2;
        if (arr[mid] <= val) {
            low = mid + 1;
        } else {
            if (mid == 0) {
                return 0;
            }
            high = mid - 1;
        }
    }
    return low - 1;
}

} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_UTILS_CUDA_BINSEARCH_CUH
