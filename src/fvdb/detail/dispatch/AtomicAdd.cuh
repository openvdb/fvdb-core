// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// AtomicAdd.cuh — Tag-dispatched atomic add for CPU and GPU.
//
// Provides a single __hostdev__ function that selects the right atomic
// implementation based on the dispatch tag's device coordinate:
//
//   CPU:  std::atomic_ref<T> for float/double (C++20, relaxed ordering)
//         CAS loop with float promotion for half/bfloat16
//   GPU:  CUDA atomicAdd (with half/bfloat16 → native type conversion)
//
// Usage inside a __hostdev__ callback:
//
//     atomic_add(tg, &out_v(row, col), value);
//
#ifndef FVDB_DETAIL_DISPATCH_ATOMIC_ADD_CUH
#define FVDB_DETAIL_DISPATCH_ATOMIC_ADD_CUH

#include "dispatch/macros.h"
#include "dispatch/torch/dispatch.h"

#if !defined(__CUDA_ARCH__)
#include <atomic>
#include <cstring>
#endif

#if defined(__CUDACC__)
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#endif

#include <type_traits>

namespace fvdb {
namespace detail {
namespace dispatch {

#if !defined(__CUDA_ARCH__)
namespace cpu_atomic_detail {

// std::atomic_ref<T>::fetch_add works for float and double (C++20).
// For c10::Half and c10::BFloat16 it doesn't — they're not standard
// arithmetic types. We use a CAS loop: promote to float, add, demote.
template <typename T>
void
atomic_add_cpu(T *dst, T src) {
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
        std::atomic_ref<T>(*dst).fetch_add(src, std::memory_order_relaxed);
    } else {
        // CAS loop for half types: load the 16-bit value as uint16_t,
        // promote to float, add, demote, compare-and-swap.
        static_assert(sizeof(T) == 2, "CAS fallback assumes 16-bit type");
        auto *raw = reinterpret_cast<std::atomic<uint16_t> *>(dst);
        uint16_t old_bits = raw->load(std::memory_order_relaxed);
        uint16_t new_bits;
        do {
            T old_val;
            std::memcpy(&old_val, &old_bits, sizeof(T));
            T new_val = static_cast<T>(static_cast<float>(old_val) + static_cast<float>(src));
            std::memcpy(&new_bits, &new_val, sizeof(T));
        } while (
            !raw->compare_exchange_weak(old_bits, new_bits, std::memory_order_relaxed));
    }
}

} // namespace cpu_atomic_detail
#endif // !__CUDA_ARCH__

/// @brief Device-dispatched atomic add.
///
/// @tparam Tag  A dispatch tag carrying at least a torch::DeviceType.
/// @tparam T    The scalar type (float, double, c10::Half, c10::BFloat16, etc.).
/// @param tg    Tag instance (selects CPU vs GPU path).
/// @param dst   Pointer to the destination value.
/// @param src   Value to add atomically.
template <typename Tag, typename T>
    requires ::dispatch::with_type<Tag, torch::DeviceType>
__hostdev__ void
atomic_add(Tag /*tg*/, T *dst, T src) {
#if defined(__CUDA_ARCH__)
    // Device compilation: CUDA atomicAdd.
    // c10::Half and c10::BFloat16 are layout-compatible with __half and
    // __nv_bfloat16 but are distinct types, so we reinterpret_cast.
    if constexpr (std::is_same_v<T, c10::Half>) {
        atomicAdd(reinterpret_cast<__half *>(dst), static_cast<__half>(src));
    } else if constexpr (std::is_same_v<T, c10::BFloat16>) {
        atomicAdd(reinterpret_cast<__nv_bfloat16 *>(dst), static_cast<__nv_bfloat16>(src));
    } else {
        atomicAdd(dst, src);
    }
#else
    // Host compilation: CPU path.
    if constexpr (::dispatch::cpu_tag<Tag>) {
        cpu_atomic_detail::atomic_add_cpu(dst, src);
    }
#endif
}

} // namespace dispatch
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_DISPATCH_ATOMIC_ADD_CUH
