// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Block-based element accessor framework for efficient tensor iteration.
//
// This header provides:
//   - Block types: linear_block, tile_block
//   - Block storage: block_storage for register-resident data
//   - Element accessor: element_accessor for load/store from tensors
//   - Scheduling: for_each_blocks for block-based iteration
//
#ifndef DISPATCH_DISPATCH_TORCH_ACCESSORS_C_H
#define DISPATCH_DISPATCH_TORCH_ACCESSORS_C_H

#include "dispatch/detail.h"
#include "dispatch/macros.h"
#include "dispatch/torch/types.h"
#include "dispatch/types.h"

#include <torch/extension.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace dispatch {

// =========================================================================
// 1. Block Types
// =========================================================================
// Blocks represent a chunk of work with a compile-time shape and runtime anchor.

// -------------------------------------------------------------------------
// linear_block - 1D vectorized iteration block
// -------------------------------------------------------------------------
// Used for linear iteration over elements. The block has Vec elements
// starting at base index.

template <size_t Vec> struct linear_block {
    static_assert(Vec > 0, "linear_block Vec must be > 0");

    int64_t base;

    static consteval size_t
    size() {
        return Vec;
    }
};

// Traits for linear_block
template <size_t Vec> struct ndim<linear_block<Vec>> {
    static consteval size_t
    value() {
        return 1;
    }
};

template <size_t Vec> struct volume<linear_block<Vec>> {
    static consteval size_t
    value() {
        return Vec;
    }
};

template <size_t I, size_t Vec> struct extent_at<I, linear_block<Vec>> {
    static_assert(I == 0, "linear_block has only one dimension");
    static consteval size_t
    value() {
        return Vec;
    }
};

template <size_t I, size_t Vec> struct volume_suffix<I, linear_block<Vec>> {
    static_assert(I == 0, "linear_block has only one dimension");
    static consteval size_t
    value() {
        return 1;
    }
};

// -------------------------------------------------------------------------
// tile_block - N-dimensional tiled iteration block
// -------------------------------------------------------------------------
// Used for tiled iteration. The block has shape Shape starting at origin.

template <extents_like shape_t> struct tile_block {
    static_assert(volume_v<shape_t>() > 0, "tile_block shape must be non-empty");

    std::array<int64_t, ndim_v<shape_t>()> origin;

    static consteval size_t
    size() {
        return volume_v<shape_t>();
    }
};

// Traits for tile_block - delegate to underlying shape
template <extents_like shape_t> struct ndim<tile_block<shape_t>> {
    static consteval size_t
    value() {
        return ndim_v<shape_t>();
    }
};

template <extents_like shape_t> struct volume<tile_block<shape_t>> {
    static consteval size_t
    value() {
        return volume_v<shape_t>();
    }
};

template <size_t I, extents_like shape_t> struct extent_at<I, tile_block<shape_t>> {
    static consteval size_t
    value() {
        return extent_at_v<I, shape_t>();
    }
};

template <size_t I, extents_like shape_t> struct volume_suffix<I, tile_block<shape_t>> {
    static consteval size_t
    value() {
        return volume_suffix_v<I, shape_t>();
    }
};

// Type trait to extract shape from tile_block
template <typename T> struct tile_block_shape;
template <extents_like shape_t> struct tile_block_shape<tile_block<shape_t>> {
    using type = shape_t;
};
template <typename T> using tile_block_shape_t = typename tile_block_shape<T>::type;

// -------------------------------------------------------------------------
// Block concepts
// -------------------------------------------------------------------------

template <typename T>
concept block_like = requires {
    { volume_v<T>() } -> std::convertible_to<size_t>;
    { ndim_v<T>() } -> std::convertible_to<size_t>;
    { T::size() } -> std::convertible_to<size_t>;
};

// =========================================================================
// 2. Block Storage (Register Fragment)
// =========================================================================
// Storage for block data in registers/local memory.

namespace detail {

template <typename V>
consteval size_t
block_alignment() {
    return (alignof(V) > 16) ? alignof(V) : 16;
}

} // namespace detail

template <typename V, size_t N>
struct alignas(detail::block_alignment<V>()) block_storage {
    static_assert(N > 0, "block_storage N must be > 0");

    V data[N];

    // C++20 Default Spaceship for testing
    auto operator<=>(block_storage const &) const = default;

    __hostdev__ constexpr V &
    operator[](size_t i) {
        return data[i];
    }
    __hostdev__ constexpr V const &
    operator[](size_t i) const {
        return data[i];
    }

    __hostdev__ constexpr V *
    ptr() {
        return data;
    }
    __hostdev__ constexpr V const *
    ptr() const {
        return data;
    }

    static consteval size_t
    size() {
        return N;
    }
};

// -------------------------------------------------------------------------
// Packed element for multi-scalar elements (e.g., vec3, mat3x3)
// -------------------------------------------------------------------------

template <typename T, size_t N> struct packed_element {
    static_assert(N > 0, "packed_element N must be > 0");

    T v[N];

    __hostdev__ constexpr T &
    operator[](size_t i) {
        return v[i];
    }
    __hostdev__ constexpr T const &
    operator[](size_t i) const {
        return v[i];
    }
};

// Value type selection: scalar for extents<>, packed_element otherwise
template <typename T, typename Shape> struct element_value_type;

template <typename T> struct element_value_type<T, extents<>> {
    using type = T;
};

template <typename T, size_t... Es>
    requires(sizeof...(Es) > 0)
struct element_value_type<T, extents<Es...>> {
    using type = packed_element<T, (Es * ... * size_t{1})>;
};

template <typename T, typename Shape>
using element_value_t = typename element_value_type<T, Shape>::type;

// =========================================================================
// 3. Stride Policies
// =========================================================================
// Zero-cost abstractions for different stride patterns.

namespace detail {

// Policy: Contiguous (Rank 1 only) - zero storage via [[no_unique_address]]
struct stride_contiguous {
    __hostdev__ explicit constexpr stride_contiguous(torch::Tensor const &) noexcept {}

    __hostdev__ constexpr int64_t
    get_offset(int64_t base) const noexcept {
        return base;
    }

    __hostdev__ constexpr int64_t
    get_stride(size_t) const noexcept {
        return 1;
    }
};

// Policy: Rank-1 Strided
struct stride_rank1 {
    int64_t stride_;

    __hostdev__ explicit constexpr stride_rank1(torch::Tensor const &t)
        : stride_(t.stride(0)) {}

    __hostdev__ constexpr int64_t
    get_offset(int64_t base) const noexcept {
        return base * stride_;
    }

    __hostdev__ constexpr int64_t
    get_stride(size_t) const noexcept {
        return stride_;
    }
};

// Policy: Rank-N Strided
template <size_t Rank> struct stride_rank_n {
    std::array<int64_t, Rank> strides_;

    __hostdev__ explicit stride_rank_n(torch::Tensor const &t) {
        for (size_t i = 0; i < Rank; ++i)
            strides_[i] = t.stride(i);
    }

    template <typename array_t>
    __hostdev__ constexpr int64_t
    get_offset(array_t const &coords) const noexcept {
        int64_t off = 0;
DISPATCH_UNROLL
        for (size_t i = 0; i < Rank; ++i)
            off += coords[i] * strides_[i];
        return off;
    }

    __hostdev__ constexpr int64_t
    get_stride(size_t dim) const noexcept {
        return strides_[dim];
    }
};

// Meta-selector for stride policy
template <contiguity C, size_t Rank>
using stride_policy_t =
    std::conditional_t<(C == contiguity::contiguous && Rank == 1),
                       stride_contiguous,
                       std::conditional_t<Rank == 1, stride_rank1, stride_rank_n<Rank>>>;

// Index type selection: int32_t for CUDA (like PackedTensorAccessor32), int64_t for CPU
template <torch::DeviceType D>
using index_t = std::conditional_t<(D == torch::kCUDA || D == torch::kPrivateUse1),
                                   int32_t,
                                   int64_t>;

} // namespace detail

// =========================================================================
// 4. Element Accessor
// =========================================================================
// Provides load/store for blocks of elements from tensors.

template <torch::DeviceType D,
          torch::ScalarType S,
          contiguity C,
          size_t index_rank,
          extents_like element_shape_t>
struct element_accessor {
    using scalar_type = torch_scalar_cpp_type_t<S>;
    using value_type  = element_value_t<scalar_type, element_shape_t>;
    using index_t     = detail::index_t<D>;

    static consteval size_t
    elem_numel() {
        return volume_v<element_shape_t>();
    }

    static consteval size_t
    elem_rank() {
        return ndim_v<element_shape_t>();
    }

    static consteval bool
    is_scalar_element() {
        return elem_rank() == 0;
    }

    scalar_type *data_;
    [[no_unique_address]] detail::stride_policy_t<C, index_rank> indexer_;

    // For strided access to element dimensions (use max(1, elem_rank()) to avoid zero-size array)
    static consteval size_t
    elem_strides_size() {
        return elem_rank() > 0 ? elem_rank() : 1;
    }
    std::array<int64_t, elem_strides_size()> elem_strides_;
    bool elem_is_contiguous_;

    element_accessor() = default;

    static element_accessor
    from_tensor(torch::Tensor const &t) {
        TORCH_CHECK(t.defined(), "element_accessor: tensor is undefined");
        TORCH_CHECK(t.device().type() == D, "element_accessor: device mismatch");
        TORCH_CHECK(t.scalar_type() == S, "element_accessor: scalar type mismatch");

        constexpr int64_t expected_dim = static_cast<int64_t>(index_rank + elem_rank());
        TORCH_CHECK(t.dim() == expected_dim,
                    "element_accessor: expected ",
                    expected_dim,
                    " dims, got ",
                    t.dim());

        if constexpr (C == contiguity::contiguous) {
            TORCH_CHECK(t.is_contiguous(), "element_accessor: tensor must be contiguous");
        }

        element_accessor acc;
        acc.data_ = t.data_ptr<scalar_type>();
        acc.indexer_ = detail::stride_policy_t<C, index_rank>(t);

        // Element strides
        if constexpr (elem_rank() > 0) {
            acc.elem_is_contiguous_ = true;
            int64_t expected_stride = 1;
            for (int i = static_cast<int>(elem_rank()) - 1; i >= 0; --i) {
                size_t const dim_idx = index_rank + static_cast<size_t>(i);
                acc.elem_strides_[static_cast<size_t>(i)] = t.stride(static_cast<int64_t>(dim_idx));
                if (acc.elem_strides_[static_cast<size_t>(i)] != expected_stride) {
                    acc.elem_is_contiguous_ = false;
                }
                expected_stride *= t.size(static_cast<int64_t>(dim_idx));
            }
        } else {
            acc.elem_is_contiguous_ = true;
        }

        return acc;
    }

    // --- Load for linear_block ---

    template <size_t Vec>
    __hostdev__ auto
    load(linear_block<Vec> const &blk) const -> block_storage<value_type, Vec> {
        block_storage<value_type, Vec> out;
        constexpr size_t EN = elem_numel();

        if constexpr (C == contiguity::contiguous || is_scalar_element()) {
            // Fast path: contiguous data
            scalar_type const *p = data_ + blk.base * static_cast<int64_t>(EN);

            if constexpr (is_scalar_element()) {
DISPATCH_UNROLL
                for (size_t i = 0; i < Vec; ++i) {
                    out[i] = p[i];
                }
            } else {
DISPATCH_UNROLL
                for (size_t i = 0; i < Vec; ++i) {
                    load_packed_contiguous(p + i * EN, out[i]);
                }
            }
        } else {
            // Strided path
            int64_t offset = indexer_.get_offset(blk.base);
DISPATCH_UNROLL
            for (size_t i = 0; i < Vec; ++i) {
                if constexpr (is_scalar_element()) {
                    out[i] = data_[offset];
                } else {
                    load_element(data_ + offset, out[i]);
                }
                offset += indexer_.get_stride(0);
            }
        }

        return out;
    }

    // --- Load for tile_block ---

    template <extents_like shape_t>
    __hostdev__ auto
    load(tile_block<shape_t> const &blk) const -> block_storage<value_type, volume_v<shape_t>()> {
        static_assert(ndim_v<shape_t>() == index_rank, "tile_block rank must match index_rank");

        block_storage<value_type, volume_v<shape_t>()> out;
        load_tile_recursive<shape_t, 0>(blk.origin, out, 0, compute_base_offset(blk.origin));
        return out;
    }

    // --- Prepare (returns uninitialized storage) ---

    template <block_like Block>
    __hostdev__ auto
    prepare(Block const &) const -> block_storage<value_type, volume_v<Block>()> {
        return {};
    }

    // --- Store for linear_block ---

    template <size_t Vec>
    __hostdev__ void
    store(linear_block<Vec> const &blk, block_storage<value_type, Vec> const &buf) const {
        constexpr size_t EN = elem_numel();

        if constexpr (C == contiguity::contiguous || is_scalar_element()) {
            scalar_type *p = data_ + blk.base * static_cast<int64_t>(EN);

            if constexpr (is_scalar_element()) {
DISPATCH_UNROLL
                for (size_t i = 0; i < Vec; ++i) {
                    p[i] = buf[i];
                }
            } else {
DISPATCH_UNROLL
                for (size_t i = 0; i < Vec; ++i) {
                    store_packed_contiguous(p + i * EN, buf[i]);
                }
            }
        } else {
            int64_t offset = indexer_.get_offset(blk.base);
DISPATCH_UNROLL
            for (size_t i = 0; i < Vec; ++i) {
                if constexpr (is_scalar_element()) {
                    data_[offset] = buf[i];
                } else {
                    store_element(data_ + offset, buf[i]);
                }
                offset += indexer_.get_stride(0);
            }
        }
    }

    // --- Store for tile_block ---

    template <extents_like shape_t>
    __hostdev__ void
    store(tile_block<shape_t> const &blk,
          block_storage<value_type, volume_v<shape_t>()> const &buf) const {
        static_assert(ndim_v<shape_t>() == index_rank, "tile_block rank must match index_rank");

        store_tile_recursive<shape_t, 0>(blk.origin, buf, 0, compute_base_offset(blk.origin));
    }

  private:
    // Helper: compute base offset for tile origin
    template <typename array_t>
    __hostdev__ int64_t
    compute_base_offset(array_t const &origin) const {
        if constexpr (index_rank == 1) {
            return indexer_.get_offset(origin[0]);
        } else {
            return indexer_.get_offset(origin);
        }
    }

    // Helper: load packed element contiguously
    template <size_t N>
    __hostdev__ static void
    load_packed_contiguous(scalar_type const *src, packed_element<scalar_type, N> &dst) {
DISPATCH_UNROLL
        for (size_t i = 0; i < N; ++i) {
            dst[i] = src[i];
        }
    }

    // Helper: store packed element contiguously
    template <size_t N>
    __hostdev__ static void
    store_packed_contiguous(scalar_type *dst, packed_element<scalar_type, N> const &src) {
DISPATCH_UNROLL
        for (size_t i = 0; i < N; ++i) {
            dst[i] = src[i];
        }
    }

    // Helper: load element with strides
    template <size_t N>
    __hostdev__ void
    load_element(scalar_type const *base, packed_element<scalar_type, N> &dst) const {
        if (elem_is_contiguous_) {
            load_packed_contiguous(base, dst);
        } else {
            load_element_strided<0, N>(base, dst, 0);
        }
    }

    // Returns the next linear index after loading
    template <size_t Dim, size_t N>
    __hostdev__ size_t
    load_element_strided(scalar_type const *base,
                         packed_element<scalar_type, N> &dst,
                         size_t linear) const {
        if constexpr (Dim == elem_rank()) {
            dst[linear] = *base;
            return linear + 1;
        } else {
            constexpr size_t dim_size = extent_at_v<Dim, element_shape_t>();
DISPATCH_UNROLL
            for (size_t i = 0; i < dim_size; ++i) {
                linear = load_element_strided<Dim + 1, N>(
                    base + static_cast<ptrdiff_t>(i) * elem_strides_[Dim], dst, linear);
            }
            return linear;
        }
    }

    // Helper: store element with strides
    template <size_t N>
    __hostdev__ void
    store_element(scalar_type *base, packed_element<scalar_type, N> const &src) const {
        if (elem_is_contiguous_) {
            store_packed_contiguous(base, src);
        } else {
            store_element_strided<0, N>(base, src, 0);
        }
    }

    // Returns the next linear index after storing
    template <size_t Dim, size_t N>
    __hostdev__ size_t
    store_element_strided(scalar_type *base,
                          packed_element<scalar_type, N> const &src,
                          size_t linear) const {
        if constexpr (Dim == elem_rank()) {
            *base = src[linear];
            return linear + 1;
        } else {
            constexpr size_t dim_size = extent_at_v<Dim, element_shape_t>();
DISPATCH_UNROLL
            for (size_t i = 0; i < dim_size; ++i) {
                linear = store_element_strided<Dim + 1, N>(
                    base + static_cast<ptrdiff_t>(i) * elem_strides_[Dim], src, linear);
            }
            return linear;
        }
    }

    // Recursive tile load
    template <typename shape_t, size_t Depth, typename array_t>
    __hostdev__ void
    load_tile_recursive(array_t const &origin,
                        block_storage<value_type, volume_v<shape_t>()> &out,
                        size_t frag_offset,
                        int64_t mem_offset) const {
        if constexpr (Depth == ndim_v<shape_t>() - 1) {
            // Innermost dimension
            constexpr size_t inner_size = extent_at_v<Depth, shape_t>();
            constexpr size_t EN         = elem_numel();
            int64_t stride              = indexer_.get_stride(Depth);

DISPATCH_UNROLL
            for (size_t i = 0; i < inner_size; ++i) {
                if constexpr (is_scalar_element()) {
                    out[frag_offset + i] = data_[mem_offset + i * stride];
                } else {
                    load_element(data_ + mem_offset + i * stride, out[frag_offset + i]);
                }
            }
        } else {
            constexpr size_t dim_size    = extent_at_v<Depth, shape_t>();
            constexpr size_t frag_stride = volume_suffix_v<Depth, shape_t>();
            int64_t stride               = indexer_.get_stride(Depth);

DISPATCH_UNROLL
            for (size_t i = 0; i < dim_size; ++i) {
                load_tile_recursive<shape_t, Depth + 1>(
                    origin, out, frag_offset + i * frag_stride, mem_offset + i * stride);
            }
        }
    }

    // Recursive tile store
    template <typename shape_t, size_t Depth, typename array_t>
    __hostdev__ void
    store_tile_recursive(array_t const &origin,
                         block_storage<value_type, volume_v<shape_t>()> const &buf,
                         size_t frag_offset,
                         int64_t mem_offset) const {
        if constexpr (Depth == ndim_v<shape_t>() - 1) {
            constexpr size_t inner_size = extent_at_v<Depth, shape_t>();
            int64_t stride              = indexer_.get_stride(Depth);

DISPATCH_UNROLL
            for (size_t i = 0; i < inner_size; ++i) {
                if constexpr (is_scalar_element()) {
                    data_[mem_offset + i * stride] = buf[frag_offset + i];
                } else {
                    store_element(data_ + mem_offset + i * stride, buf[frag_offset + i]);
                }
            }
        } else {
            constexpr size_t dim_size    = extent_at_v<Depth, shape_t>();
            constexpr size_t frag_stride = volume_suffix_v<Depth, shape_t>();
            int64_t stride               = indexer_.get_stride(Depth);

DISPATCH_UNROLL
            for (size_t i = 0; i < dim_size; ++i) {
                store_tile_recursive<shape_t, Depth + 1>(
                    origin, buf, frag_offset + i * frag_stride, mem_offset + i * stride);
            }
        }
    }
};

// =========================================================================
// 5. Scheduling - for_each_blocks
// =========================================================================
// Provides block-based iteration with automatic tail handling.

// Schedule policy types
template <size_t Vec> struct linear_schedule {
    static consteval size_t
    vec_size() {
        return Vec;
    }
};

// Forward declarations for for_each (from for_each.h)
// These will be defined when for_each.h is included
template <int64_t GrainSize, int BlockDim, typename Tag, typename Func>
void for_each(Tag t, int64_t count, Func &&func);

// for_each_blocks: block-based iteration with tail handling
template <typename Schedule, typename Tag, typename Func>
void
for_each_blocks(Tag t, int64_t count, Func &&func) {
    constexpr size_t Vec = Schedule::vec_size();

    int64_t const full_blocks = count / static_cast<int64_t>(Vec);
    int64_t const tail        = count % static_cast<int64_t>(Vec);

    // Process full blocks
    if (full_blocks > 0) {
        for_each<1, 256>(t, full_blocks, [=] __hostdev__(Tag tt, int64_t bi) mutable {
            linear_block<Vec> blk;
            blk.base = bi * static_cast<int64_t>(Vec);
            func(tt, blk);
        });
    }

    // Process tail elements one at a time
    if (tail > 0) {
        for_each<1, 256>(t, tail, [=] __hostdev__(Tag tt, int64_t i) mutable {
            linear_block<1> blk;
            blk.base = full_blocks * static_cast<int64_t>(Vec) + i;
            func(tt, blk);
        });
    }
}

} // namespace dispatch

#endif // DISPATCH_DISPATCH_TORCH_ACCESSORS_C_H
