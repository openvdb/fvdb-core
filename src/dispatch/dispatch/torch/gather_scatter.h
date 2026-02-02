// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// gather_scatter: Device-agnostic gather and scatter_add primitives for sparse ops.
//
// These are building blocks for sparse convolution and similar algorithms that
// need to gather inputs by index and scatter-add outputs back.
//
#ifndef DISPATCH_DISPATCH_TORCH_GATHER_SCATTER_H
#define DISPATCH_DISPATCH_TORCH_GATHER_SCATTER_H

#include "dispatch/macros.h"
#include "dispatch/tag_match.h"
#include "dispatch/torch/for_each.h"
#include "dispatch/torch/types.h"
#include "dispatch/torch/views.h"

#include <array>
#include <atomic>
#include <cstdint>

namespace dispatch {

namespace detail {

//------------------------------------------------------------------------------
// atomic_add_helper - single-element atomic add, specialized per device
//------------------------------------------------------------------------------

template <torch::DeviceType Dev, torch::ScalarType Stype> struct atomic_add_helper;

/// CPU: std::atomic_ref (C++20)
template <torch::ScalarType Stype> struct atomic_add_helper<torch::kCPU, Stype> {
    using value_type = torch_scalar_cpp_type_t<Stype>;
    static void
    apply(value_type *dst, value_type src) {
        std::atomic_ref<value_type>(*dst).fetch_add(src, std::memory_order_relaxed);
    }
};

#if defined(__CUDACC__)

/// GPU: CUDA atomicAdd
template <torch::ScalarType Stype> struct atomic_add_helper<torch::kCUDA, Stype> {
    using value_type = torch_scalar_cpp_type_t<Stype>;
    __device__ static void
    apply(value_type *dst, value_type src) {
        atomicAdd(dst, src);
    }
};

template <torch::ScalarType Stype> struct atomic_add_helper<torch::kPrivateUse1, Stype> {
    using value_type = torch_scalar_cpp_type_t<Stype>;
    __device__ static void
    apply(value_type *dst, value_type src) {
        atomicAdd(dst, src);
    }
};

#endif // __CUDACC__

//------------------------------------------------------------------------------
// gather_helper
//------------------------------------------------------------------------------

template <typename Tag, typename SrcView, typename DstView, typename IdxView> struct gather_helper;

template <typename Tag, torch::DeviceType Dev, torch::ScalarType Stype>
    requires tag_match<Tag, Dev, Stype>
struct gather_helper<Tag,
                     matrix_const_view<Dev, Stype>,
                     matrix_mutable_view<Dev, Stype>,
                     vector_const_view<Dev, torch::kInt32>> {
    static void
    apply(Tag t,
          matrix_const_view<Dev, Stype> src,
          matrix_mutable_view<Dev, Stype> dst,
          vector_const_view<Dev, torch::kInt32> idx) {
        for_each_nd<2>(t, {dst.rows, dst.cols}, [=] __hostdev__(Tag, std::array<int64_t, 2> ij) {
            auto const [i, j]  = ij;
            auto const src_row = idx[i];
            if (src_row >= 0) {
                dst(i, j) = src(src_row, j);
            }
        });
    }
};

//------------------------------------------------------------------------------
// scatter_add_helper
//------------------------------------------------------------------------------

template <typename Tag, typename SrcView, typename DstView, typename IdxView>
struct scatter_add_helper;

template <typename Tag, torch::DeviceType Dev, torch::ScalarType Stype>
    requires tag_match<Tag, Dev, Stype>
struct scatter_add_helper<Tag,
                          matrix_const_view<Dev, Stype>,
                          matrix_mutable_view<Dev, Stype>,
                          vector_const_view<Dev, torch::kInt32>> {
    static void
    apply(Tag t,
          matrix_const_view<Dev, Stype> src,
          matrix_mutable_view<Dev, Stype> dst,
          vector_const_view<Dev, torch::kInt32> idx) {
        for_each_nd<2>(t, {src.rows, src.cols}, [=] __hostdev__(Tag, std::array<int64_t, 2> ij) {
            auto const [i, j]  = ij;
            auto const dst_row = idx[i];
            if (dst_row >= 0) {
                atomic_add_helper<Dev, Stype>::apply(&dst(dst_row, j), src(i, j));
            }
        });
    }
};

} // namespace detail

//------------------------------------------------------------------------------
// Public interface
//------------------------------------------------------------------------------

/// gather: dst[i, :] = src[indices[i], :] for each i in [0, dst.rows)
///
/// If indices[i] < 0, that row is skipped (dst unchanged).
template <typename Tag, typename SrcView, typename DstView, typename IdxView>
void
gather(Tag t, SrcView src, DstView dst, IdxView idx) {
    detail::gather_helper<Tag, SrcView, DstView, IdxView>::apply(t, src, dst, idx);
}

/// scatter_add: dst[indices[i], :] += src[i, :] for each i in [0, src.rows)
///
/// If indices[i] < 0, that row is skipped.
/// Thread-safe: uses atomicAdd on GPU, std::atomic_ref on CPU.
template <typename Tag, typename SrcView, typename DstView, typename IdxView>
void
scatter_add(Tag t, SrcView src, DstView dst, IdxView idx) {
    detail::scatter_add_helper<Tag, SrcView, DstView, IdxView>::apply(t, src, dst, idx);
}

} // namespace dispatch

#endif // DISPATCH_DISPATCH_TORCH_GATHER_SCATTER_H
