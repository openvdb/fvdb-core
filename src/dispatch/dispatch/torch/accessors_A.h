// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "dispatch/detail.h"
#include "dispatch/macros.h"
#include "dispatch/torch/types.h"
#include "dispatch/types.h"

#include <torch/extension.h>

#include <cuda_runtime.h>

#include <array>
#include <bit>
#include <concepts>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace dispatch {

// =========================================================================
// 1. Extended Extents Traits
// =========================================================================

// -------------------------------------------------------------------------
// extent_at - get the extent value at a specific dimension index
// -------------------------------------------------------------------------

template <size_t I, typename T> struct extent_at;

template <size_t I, size_t S0, size_t... Ss> struct extent_at<I, extents<S0, Ss...>> {
    static_assert(I < (1 + sizeof...(Ss)), "extent_at: index out of bounds");
    static consteval size_t
    value() {
        if constexpr (I == 0) {
            return S0;
        } else {
            return extent_at<I - 1, extents<Ss...>>::value();
        }
    }
};

template <size_t I, typename T>
consteval size_t
extent_at_v() {
    return extent_at<I, T>::value();
}

// -------------------------------------------------------------------------
// volume_suffix - product of all extents after dimension I (exclusive)
// Used for computing strides in row-major layout
// -------------------------------------------------------------------------

template <size_t I, typename T> struct volume_suffix;

template <size_t I, size_t S0, size_t... Ss> struct volume_suffix<I, extents<S0, Ss...>> {
    static_assert(I < (1 + sizeof...(Ss)), "volume_suffix: index out of bounds");
    static consteval size_t
    value() {
        if constexpr (I >= sizeof...(Ss)) {
            // Last dimension or beyond - suffix is 1
            return 1;
        } else if constexpr (I == 0) {
            // Suffix after first = volume of tail
            return volume_v<extents<Ss...>>();
        } else {
            return volume_suffix<I - 1, extents<Ss...>>::value();
        }
    }
};

template <size_t I, typename T>
consteval size_t
volume_suffix_v() {
    return volume_suffix<I, T>::value();
}

// =========================================================================
// 2. Core Concepts & Configuration
// =========================================================================

// Concept: Validates that T acts like a shape via the ndim/volume traits
template <typename T>
concept extents_shape = requires {
    { ndim_v<T>() } -> std::convertible_to<size_t>;
    { volume_v<T>() } -> std::convertible_to<size_t>;
};

// Concept: Numeric types valid for processing/vectorization
template <typename T>
concept numeric =
    std::is_arithmetic_v<T> || std::same_as<T, torch::Half> || std::same_as<T, torch::BFloat16>;

// -------------------------------------------------------------------------
// Work Block: An extents with a runtime anchor point
// -------------------------------------------------------------------------

template <extents_like shape_t> struct block {
    // Runtime State: The anchor point of the block
    using coord_storage = std::array<int64_t, ndim_v<shape_t>()>;
    coord_storage start;

    __hostdev__ constexpr int64_t
    coord(size_t d) const {
        return start[d];
    }
};

template <extents_like shape_t>
    requires(ndim_v<shape_t>() == 1)
struct block<shape_t> {
    using coord_storage = int64_t;
    coord_storage start;

    __hostdev__ constexpr int64_t
    coord(size_t d) const {
        return (d == 0) ? start : 0;
    }
};

// -------------------------------------------------------------------------
// Traits for block - delegate to underlying extents
// -------------------------------------------------------------------------

template <extents_like shape_t> struct ndim<block<shape_t>> {
    static consteval size_t
    value() {
        return ndim_v<shape_t>();
    }
};

template <extents_like shape_t> struct volume<block<shape_t>> {
    static consteval size_t
    value() {
        return volume_v<shape_t>();
    }
};

template <size_t I, extents_like shape_t> struct extent_at<I, block<shape_t>> {
    static consteval size_t
    value() {
        return extent_at_v<I, shape_t>();
    }
};

template <size_t I, extents_like shape_t> struct volume_suffix<I, block<shape_t>> {
    static consteval size_t
    value() {
        return volume_suffix_v<I, shape_t>();
    }
};

// Type trait to extract shape from block
template <typename T> struct block_shape;
template <extents_like shape_t> struct block_shape<block<shape_t>> {
    using type = shape_t;
};
template <typename T> using block_shape_t = typename block_shape<T>::type;

template <typename T>
concept worker_block = requires {
    { volume_v<T>() } -> std::convertible_to<size_t>;
    { ndim_v<T>() } -> std::convertible_to<size_t>;
    typename block_shape_t<T>;
};

// =========================================================================
// 3. Register Fragment
// =========================================================================

template <numeric T, size_t N> struct alignas(alignof(T)) fragment {
    T data[N];

    // C++20 Default Spaceship
    auto operator<=>(fragment const &) const = default;

    __hostdev__ constexpr T &
    operator[](size_t i) {
        return data[i];
    }
    __hostdev__ constexpr T const &
    operator[](size_t i) const {
        return data[i];
    }

    // Scalar conversion convenience
    __hostdev__ constexpr
    operator T &()
        requires(N == 1)
    {
        return data[0];
    }
    __hostdev__ constexpr
    operator T const &() const
        requires(N == 1)
    {
        return data[0];
    }
};

// =========================================================================
// 4. Stride Policies (Coordinate Aware & Zero-Byte Optimized)
// =========================================================================

namespace detail {

// Policy: Contiguous (Rank 1 only)
// [[no_unique_address]] ensures 0 bytes storage
struct stride_contiguous {
    __hostdev__ explicit constexpr stride_contiguous(torch::Tensor const &) noexcept {}

    __hostdev__ constexpr int64_t
    get_base_offset(int64_t start) const noexcept {
        return start;
    }

    __hostdev__ constexpr int64_t
    get_step_stride(size_t) const noexcept {
        return 1;
    }
};

// Policy: Rank-1 Strided
struct stride_rank1 {
    int64_t stride;
    __hostdev__ explicit constexpr stride_rank1(torch::Tensor const &t) : stride(t.stride(0)) {}

    __hostdev__ constexpr int64_t
    get_base_offset(int64_t start) const noexcept {
        return start * stride;
    }

    __hostdev__ constexpr int64_t
    get_step_stride(size_t) const noexcept {
        return stride;
    }
};

// Policy: Rank-N Strided
// Required for any Rank > 1 tile, even if contiguous in memory, to handle row jumps.
template <size_t Rank> struct stride_rank_n {
    std::array<int64_t, Rank> strides;

    __hostdev__ explicit stride_rank_n(torch::Tensor const &t) {
        for (size_t i = 0; i < Rank; ++i)
            strides[i] = t.stride(i);
    }

    template <typename array_t>
    __hostdev__ constexpr int64_t
    get_base_offset(array_t const &start_coords) const noexcept {
        int64_t off = 0;
        DISPATCH_UNROLL
        for (size_t i = 0; i < Rank; ++i)
            off += start_coords[i] * strides[i];
        return off;
    }

    __hostdev__ constexpr int64_t
    get_step_stride(size_t dim) const noexcept {
        return strides[dim];
    }
};

// Meta-Selector:
// If Rank > 1, we MUST use stride_rank_n to safely handle tiling offsets (pitch),
// even if the user claims contiguous.
// stride_contiguous is reserved strictly for 1D linear access.
template <contiguity C, size_t Rank>
using stride_policy_t =
    std::conditional_t<(C == contiguity::contiguous && Rank == 1),
                       stride_contiguous,
                       std::conditional_t<Rank == 1, stride_rank1, stride_rank_n<Rank>>>;
} // namespace detail

// =========================================================================
// 5. Vectorization Logic (ConstEval)
// =========================================================================

namespace detail {
template <size_t Bytes> struct vec_type {
    using type = void;
};
template <> struct vec_type<16> {
    using type = int4;
};
template <> struct vec_type<8> {
    using type = int2;
};
template <> struct vec_type<4> {
    using type = int;
};

// Check if the innermost dimension (row) is vectorizable
template <size_t row_bytes, contiguity C, typename Policy>
consteval bool
can_vectorize_row() {
    if constexpr (std::is_void_v<typename vec_type<row_bytes>::type>)
        return false;
    // If we have a vector type, we must also ensure layout allows it.
    // We will do a runtime check for last_stride == 1 inside the accessor.
    return true;
}
} // namespace detail

// =========================================================================
// 6. Accessor Interface
// =========================================================================

template <torch::DeviceType D,
          torch::ScalarType S,
          contiguity C,
          size_t index_rank,
          extents_shape element_shape_t>
struct element_accessor;

// -------------------------------------------------------------------------
// Specialization: CUDA
// -------------------------------------------------------------------------

template <torch::ScalarType S, contiguity C, size_t index_rank, extents_shape element_shape_t>
struct element_accessor<torch::kCUDA, S, C, index_rank, element_shape_t> {
    using scalar_type = torch_scalar_cpp_type_t<S>;

    // Consteval helper (no storage symbols)
    static consteval size_t
    element_numel() {
        return volume_v<element_shape_t>();
    }

    scalar_type *__restrict__ data_ptr_;

    // C++20: This member takes 0 bytes if stride_contiguous is selected.
    [[no_unique_address]] detail::stride_policy_t<C, index_rank> indexer_;

    explicit element_accessor(torch::Tensor const &t) : indexer_(t) {
        TORCH_CHECK(t.device().type() == torch::kCUDA, "Tensor must be CUDA");
        TORCH_CHECK(t.scalar_type() == S, "Scalar type mismatch");
        if constexpr (C == contiguity::contiguous)
            TORCH_CHECK(t.is_contiguous());
        // Rank check: Tensor dims >= index_rank (iteration space) + shape_rank (element space)
        TORCH_CHECK(t.dim() >= static_cast<int64_t>(index_rank + ndim_v<element_shape_t>()),
                    "Rank mismatch");
        data_ptr_ = t.data_ptr<scalar_type>();
    }

    // --- Load ---

    template <worker_block B>
    __device__ __forceinline__ auto
    load(B const &blk) const -> fragment<scalar_type, volume_v<B>() * element_numel()> {
        fragment<scalar_type, volume_v<B>() * element_numel()> frag;

        // Calculate the starting memory offset based on block's anchor point
        int64_t base_offset = indexer_.get_base_offset(blk.start);

        // Recursively load dimensions (starting at depth 0)
        recursive_load_impl<B, 0>(blk, frag, 0, base_offset);

        return frag;
    }

    // --- Recursive Loader (Depth-Based) ---

    template <typename B, size_t Depth>
    __device__ __forceinline__ void
    recursive_load_impl(B const &blk,
                        fragment<scalar_type, volume_v<B>() * element_numel()> &frag,
                        size_t frag_offset,
                        int64_t mem_offset) const {
        // BASE CASE: Innermost Dimension (The Row)
        if constexpr (Depth == ndim_v<B>() - 1) {
            constexpr size_t inner_size = extent_at_v<Depth, B>();
            constexpr size_t row_bytes  = inner_size * element_numel() * sizeof(scalar_type);

            // 1. Vectorization Eligibility Check
            constexpr bool eligible = detail::can_vectorize_row<row_bytes, C, decltype(indexer_)>();
            bool perform_vector_load = false;

            if constexpr (eligible) {
                // Runtime Stride Check (Compiler optimizes out for stride_contiguous)
                if (indexer_.get_step_stride(Depth) == 1) {
                    perform_vector_load = true;
                }
            }

            // 2. Vector Load Path (int4/int2)
            if (perform_vector_load) {
                using vec_t    = typename detail::vec_type<row_bytes>::type;
                uintptr_t addr = reinterpret_cast<uintptr_t>(data_ptr_ + mem_offset);

                // Alignment Check
                if (addr % alignof(vec_t) == 0) {
                    vec_t const *vec_src = reinterpret_cast<vec_t const *>(data_ptr_ + mem_offset);
                    // __ldg uses texture cache (read-only), bypassing L1
                    vec_t loaded = __ldg(vec_src);

                    // Interpret cast into the fragment array (standard CUDA idiom)
                    *reinterpret_cast<vec_t *>(&frag.data[frag_offset]) = loaded;
                    return;
                }
            }

            // 3. Scalar Loop Path
            int64_t stride = indexer_.get_step_stride(Depth);
            DISPATCH_UNROLL
            for (size_t i = 0; i < inner_size; ++i) {
                scalar_type const *src = data_ptr_ + mem_offset + (i * stride);
                DISPATCH_UNROLL
                for (size_t e = 0; e < element_numel(); ++e) {
                    frag.data[frag_offset + i * element_numel() + e] = src[e];
                }
            }
        }
        // RECURSIVE STEP: Outer Dimensions
        else {
            constexpr size_t dim_size    = extent_at_v<Depth, B>();
            constexpr size_t frag_stride = volume_suffix_v<Depth, B>() * element_numel();

            int64_t stride = indexer_.get_step_stride(Depth);

            DISPATCH_UNROLL
            for (size_t i = 0; i < dim_size; ++i) {
                recursive_load_impl<B, Depth + 1>(
                    blk, frag, frag_offset + i * frag_stride, mem_offset + i * stride);
            }
        }
    }

    // --- Prepare (No-Op) ---
    template <worker_block B>
    __device__ __forceinline__ auto
    prepare(B const &) const -> fragment<scalar_type, volume_v<B>() * element_numel()> {
        return {};
    }

    // --- Store ---

    template <worker_block B>
    __device__ __forceinline__ void
    store(B const &blk, fragment<scalar_type, volume_v<B>() * element_numel()> const &frag) const {
        int64_t base_offset = indexer_.get_base_offset(blk.start);
        recursive_store_impl<B, 0>(blk, frag, 0, base_offset);
    }

    template <typename B, size_t Depth>
    __device__ __forceinline__ void
    recursive_store_impl(B const &blk,
                         fragment<scalar_type, volume_v<B>() * element_numel()> const &frag,
                         size_t frag_offset,
                         int64_t mem_offset) const {
        if constexpr (Depth == ndim_v<B>() - 1) {
            constexpr size_t inner_size = extent_at_v<Depth, B>();
            constexpr size_t row_bytes  = inner_size * element_numel() * sizeof(scalar_type);

            constexpr bool eligible = detail::can_vectorize_row<row_bytes, C, decltype(indexer_)>();
            bool perform_vector_store = false;

            if constexpr (eligible) {
                if (indexer_.get_step_stride(Depth) == 1)
                    perform_vector_store = true;
            }

            if (perform_vector_store) {
                using vec_t    = typename detail::vec_type<row_bytes>::type;
                uintptr_t addr = reinterpret_cast<uintptr_t>(data_ptr_ + mem_offset);

                if (addr % alignof(vec_t) == 0) {
                    vec_t *vec_dst = reinterpret_cast<vec_t *>(data_ptr_ + mem_offset);
                    *vec_dst       = *reinterpret_cast<vec_t const *>(&frag.data[frag_offset]);
                    return;
                }
            }

            int64_t stride = indexer_.get_step_stride(Depth);
            DISPATCH_UNROLL
            for (size_t i = 0; i < inner_size; ++i) {
                scalar_type *dst = data_ptr_ + mem_offset + (i * stride);
                DISPATCH_UNROLL
                for (size_t e = 0; e < element_numel(); ++e) {
                    dst[e] = frag.data[frag_offset + i * element_numel() + e];
                }
            }
        } else {
            constexpr size_t dim_size    = extent_at_v<Depth, B>();
            constexpr size_t frag_stride = volume_suffix_v<Depth, B>() * element_numel();
            int64_t stride               = indexer_.get_step_stride(Depth);

            DISPATCH_UNROLL
            for (size_t i = 0; i < dim_size; ++i) {
                recursive_store_impl<B, Depth + 1>(
                    blk, frag, frag_offset + i * frag_stride, mem_offset + i * stride);
            }
        }
    }
};

// -------------------------------------------------------------------------
// Specialization: CPU & Universal (Host Fallback)
// -------------------------------------------------------------------------

template <torch::DeviceType D,
          torch::ScalarType S,
          contiguity C,
          size_t index_rank,
          extents_shape element_shape_t>
    requires(D == torch::kCPU || D == torch::kPrivateUse1)
struct element_accessor<D, S, C, index_rank, element_shape_t> {
    using scalar_type = torch_scalar_cpp_type_t<S>;
    static consteval size_t
    element_numel() {
        return volume_v<element_shape_t>();
    }

    scalar_type *__restrict__ data_ptr_;
    [[no_unique_address]] detail::stride_policy_t<C, index_rank> indexer_;

    explicit element_accessor(torch::Tensor const &t) : indexer_(t) {
        TORCH_CHECK(t.device().type() == D);
        TORCH_CHECK(t.scalar_type() == S);
        if constexpr (C == contiguity::contiguous)
            TORCH_CHECK(t.is_contiguous());
        data_ptr_ = t.data_ptr<scalar_type>();
    }

    template <worker_block B>
    __hostdev__ auto
    load(B const &blk) const -> fragment<scalar_type, volume_v<B>() * element_numel()> {
        fragment<scalar_type, volume_v<B>() * element_numel()> frag;
        int64_t base_offset = indexer_.get_base_offset(blk.start);
        recursive_load_host<B, 0>(blk, frag, 0, base_offset);
        return frag;
    }

    template <typename B, size_t Depth>
    __hostdev__ void
    recursive_load_host(B const &blk,
                        fragment<scalar_type, volume_v<B>() * element_numel()> &frag,
                        size_t frag_offset,
                        int64_t mem_offset) const {
        if constexpr (Depth == ndim_v<B>() - 1) {
            constexpr size_t inner_size = extent_at_v<Depth, B>();
            int64_t stride              = indexer_.get_step_stride(Depth);
            for (size_t i = 0; i < inner_size; ++i) {
                scalar_type const *src = data_ptr_ + mem_offset + (i * stride);
                for (size_t e = 0; e < element_numel(); ++e)
                    frag.data[frag_offset + i * element_numel() + e] = src[e];
            }
        } else {
            constexpr size_t dim_size    = extent_at_v<Depth, B>();
            constexpr size_t frag_stride = volume_suffix_v<Depth, B>() * element_numel();
            int64_t stride               = indexer_.get_step_stride(Depth);
            for (size_t i = 0; i < dim_size; ++i) {
                recursive_load_host<B, Depth + 1>(
                    blk, frag, frag_offset + i * frag_stride, mem_offset + i * stride);
            }
        }
    }

    template <worker_block B>
    __hostdev__ auto
    prepare(B const &) const -> fragment<scalar_type, volume_v<B>() * element_numel()> {
        return {};
    }

    template <worker_block B>
    __hostdev__ void
    store(B const &blk, fragment<scalar_type, volume_v<B>() * element_numel()> const &frag) const {
        int64_t base_offset = indexer_.get_base_offset(blk.start);
        recursive_store_host<B, 0>(blk, frag, 0, base_offset);
    }

    template <typename B, size_t Depth>
    __hostdev__ void
    recursive_store_host(B const &blk,
                         fragment<scalar_type, volume_v<B>() * element_numel()> const &frag,
                         size_t frag_offset,
                         int64_t mem_offset) const {
        if constexpr (Depth == ndim_v<B>() - 1) {
            constexpr size_t inner_size = extent_at_v<Depth, B>();
            int64_t stride              = indexer_.get_step_stride(Depth);
            for (size_t i = 0; i < inner_size; ++i) {
                scalar_type *dst = data_ptr_ + mem_offset + (i * stride);
                for (size_t e = 0; e < element_numel(); ++e)
                    dst[e] = frag.data[frag_offset + i * element_numel() + e];
            }
        } else {
            constexpr size_t dim_size    = extent_at_v<Depth, B>();
            constexpr size_t frag_stride = volume_suffix_v<Depth, B>() * element_numel();
            int64_t stride               = indexer_.get_step_stride(Depth);
            for (size_t i = 0; i < dim_size; ++i) {
                recursive_store_host<B, Depth + 1>(
                    blk, frag, frag_offset + i * frag_stride, mem_offset + i * stride);
            }
        }
    }
};

} // namespace dispatch
