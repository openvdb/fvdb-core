// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_CUDA_CACHING_CUH
#define FVDB_DETAIL_UTILS_CUDA_CACHING_CUH

#include <dispatch/dispatch/macros.h>

#include <type_traits>

// ---------------------------------------------------------------------------
// Cache-hint helpers for write-once outputs and read-only inputs.
//
// These are deliberately tiny and inline. They route stores through the streaming-store path
// (`__stwt`, `.CS` qualifier in SASS) so write-once output tensors don't get promoted into L1
// and evict the voxel-data working set, and route loads through the read-only data cache (`__ldg`,
// `.NC` qualifier in SASS) so read-mostly side-buffer data shares cache capacity instead of
// competing with the active-mask leaf data on L1.
//
// `__stwt` and `__ldg` are only available on device, and only for a fixed set of scalar and
// vector types (see the CUDA C++ Programming Guide, "Load and Store Functions Using Cache
// Hints"). For host compilation and for scalar types without a matching overload (e.g. c10::Half)
// we fall back to a plain assignment / dereference; NVCC will fully inline both branches so the
// CPU path is unaffected.
// ---------------------------------------------------------------------------

namespace fvdb {
namespace detail {
namespace ops {

template <typename T>
__hostdev__ __forceinline__ void
_storeStreaming(T *ptr, const T value) {
#ifdef __CUDA_ARCH__
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, int> ||
                  std::is_same_v<T, unsigned int> || std::is_same_v<T, long long> ||
                  std::is_same_v<T, unsigned long long>) {
        __stwt(ptr, value);
    } else {
        *ptr = value;
    }
#else
    *ptr = value;
#endif
}

template <typename T>
__hostdev__ __forceinline__ void
_storeStreamingPair(T *ptr, const T a, const T b) {
#ifdef __CUDA_ARCH__
    if constexpr (std::is_same_v<T, float>) {
        __stwt(reinterpret_cast<float2 *>(ptr), make_float2(a, b));
    } else if constexpr (std::is_same_v<T, double>) {
        __stwt(reinterpret_cast<double2 *>(ptr), make_double2(a, b));
    } else {
        _storeStreaming(ptr + 0, a);
        _storeStreaming(ptr + 1, b);
    }
#else
    ptr[0] = a;
    ptr[1] = b;
#endif
}

// Read a value through the read-only data cache when the type / device permit. `__ldg` exists for
// the standard scalar widths and pointer types; for c10::Half / at::BFloat16 etc. we fall back to
// a plain dereference.
template <typename T>
__hostdev__ __forceinline__ T
_loadReadOnly(const T *ptr) {
#ifdef __CUDA_ARCH__
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, int> ||
                  std::is_same_v<T, unsigned int> || std::is_same_v<T, long long> ||
                  std::is_same_v<T, unsigned long long>) {
        return __ldg(ptr);
    } else {
        return *ptr;
    }
#else
    return *ptr;
#endif
}

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_UTILS_CUDA_CACHING_CUH
