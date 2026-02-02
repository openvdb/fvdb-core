// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// views: Lightweight device- and type-aware tensor wrappers.
//
// These provide type-safe access to tensor data by carrying device and scalar type
// as template parameters, enabling efficient, low-overhead operations across devices.
//
#ifndef DISPATCH_DISPATCH_TORCH_VIEWS_H
#define DISPATCH_DISPATCH_TORCH_VIEWS_H

#include "dispatch/macros.h"
#include "dispatch/tag_match.h"
#include "dispatch/torch/for_each.h"
#include "dispatch/torch/types.h"

#include <array>
#include <atomic>
#include <cstdint>

namespace dispatch {

template <torch::DeviceType dev, torch::ScalarType stype> struct device_scalar_pair {
    using value_type = torch_scalar_cpp_type_t<stype>;
    static consteval torch::DeviceType
    device() {
        return dev;
    }
    static consteval torch::ScalarType
    scalar_type() {
        return stype;
    }
};

//------------------------------------------------------------------------------
// View types - lightweight wrappers for pointer + dimensions
//------------------------------------------------------------------------------
// These carry device and scalar type as template parameters, enabling
// type-safe dispatch through the helper struct specialization pattern.

/// Same as PyTorch's max tensor rank
constexpr int64_t view_max_rank = 8;

/// Flat read-only view: any-rank tensor accessed as linear elements
/// Handles both contiguous and strided tensors efficiently.
template <torch::DeviceType Dev, torch::ScalarType Stype>
struct flat_const_view : device_scalar_pair<Dev, Stype> {
    using typename device_scalar_pair<Dev, Stype>::value_type;

    value_type const *data;
    int64_t sizes[view_max_rank];
    int64_t strides[view_max_rank];
    int64_t rank;
    int64_t numel;
    bool is_contiguous;

    __hostdev__
    flat_const_view(value_type const *d,
                    int64_t const *sz,
                    int64_t const *st,
                    int64_t r,
                    int64_t n,
                    bool contig)
        : data(d), rank(r), numel(n), is_contiguous(contig) {
        for (int64_t i = 0; i < r; ++i) {
            sizes[i]   = sz[i];
            strides[i] = st[i];
        }
    }

    explicit flat_const_view(torch::Tensor const &t)
        : data(t.data_ptr<value_type>()), rank(t.dim()), numel(t.numel()),
          is_contiguous(t.is_contiguous()) {
        for (int64_t i = 0; i < rank; ++i) {
            sizes[i]   = t.size(i);
            strides[i] = t.stride(i);
        }
    }

    __hostdev__ int64_t
    offset(int64_t i) const {
        if (is_contiguous)
            return i;
        int64_t off = 0;
        for (int64_t d = rank - 1; d >= 0; --d) {
            off += (i % sizes[d]) * strides[d];
            i /= sizes[d];
        }
        return off;
    }

    __hostdev__ value_type
    operator[](int64_t i) const {
        return data[offset(i)];
    }
};

/// Flat mutable view: any-rank tensor accessed as linear elements
template <torch::DeviceType Dev, torch::ScalarType Stype>
struct flat_mutable_view : device_scalar_pair<Dev, Stype> {
    using typename device_scalar_pair<Dev, Stype>::value_type;

    value_type *data;
    int64_t sizes[view_max_rank];
    int64_t strides[view_max_rank];
    int64_t rank;
    int64_t numel;
    bool is_contiguous;

    __hostdev__
    flat_mutable_view(
        value_type *d, int64_t const *sz, int64_t const *st, int64_t r, int64_t n, bool contig)
        : data(d), rank(r), numel(n), is_contiguous(contig) {
        for (int64_t i = 0; i < r; ++i) {
            sizes[i]   = sz[i];
            strides[i] = st[i];
        }
    }

    explicit flat_mutable_view(torch::Tensor &t)
        : data(t.data_ptr<value_type>()), rank(t.dim()), numel(t.numel()),
          is_contiguous(t.is_contiguous()) {
        for (int64_t i = 0; i < rank; ++i) {
            sizes[i]   = t.size(i);
            strides[i] = t.stride(i);
        }
    }

    __hostdev__ int64_t
    offset(int64_t i) const {
        if (is_contiguous)
            return i;
        int64_t off = 0;
        for (int64_t d = rank - 1; d >= 0; --d) {
            off += (i % sizes[d]) * strides[d];
            i /= sizes[d];
        }
        return off;
    }

    __hostdev__ value_type &
    operator[](int64_t i) const {
        return data[offset(i)];
    }
};

/// 2D read-only view: (rows x cols) matrix
template <torch::DeviceType Dev, torch::ScalarType Stype>
struct matrix_const_view : device_scalar_pair<Dev, Stype> {
    using typename device_scalar_pair<Dev, Stype>::value_type;

    value_type const *data;
    int64_t rows;
    int64_t cols;

    __hostdev__
    matrix_const_view(value_type const *d, int64_t r, int64_t c)
        : data(d), rows(r), cols(c) {}

    explicit matrix_const_view(torch::Tensor const &t)
        : data(t.data_ptr<value_type>()), rows(t.size(0)), cols(t.size(1)) {}

    __hostdev__ value_type
    operator()(int64_t row, int64_t col) const {
        return data[row * cols + col];
    }
};

/// 2D mutable view: (rows x cols) matrix
template <torch::DeviceType Dev, torch::ScalarType Stype>
struct matrix_mutable_view : device_scalar_pair<Dev, Stype> {
    using typename device_scalar_pair<Dev, Stype>::value_type;

    value_type *data;
    int64_t rows;
    int64_t cols;

    __hostdev__
    matrix_mutable_view(value_type *d, int64_t r, int64_t c)
        : data(d), rows(r), cols(c) {}

    explicit matrix_mutable_view(torch::Tensor &t)
        : data(t.data_ptr<value_type>()), rows(t.size(0)), cols(t.size(1)) {}

    __hostdev__ value_type &
    operator()(int64_t row, int64_t col) const {
        return data[row * cols + col];
    }
};

/// 1D read-only view with stride: for index arrays
template <torch::DeviceType Dev, torch::ScalarType Stype>
struct vector_const_view : device_scalar_pair<Dev, Stype> {
    using typename device_scalar_pair<Dev, Stype>::value_type;

    value_type const *data;
    int64_t count;
    int64_t stride;

    __hostdev__
    vector_const_view(value_type const *d, int64_t c, int64_t s)
        : data(d), count(c), stride(s) {}

    explicit vector_const_view(torch::Tensor const &t)
        : data(t.data_ptr<value_type>()), count(t.numel()), stride(t.stride(0)) {}

    __hostdev__ value_type
    operator[](int64_t i) const {
        return data[i * stride];
    }
};

/// 1D mutable view with stride
template <torch::DeviceType Dev, torch::ScalarType Stype>
struct vector_mutable_view : device_scalar_pair<Dev, Stype> {
    using typename device_scalar_pair<Dev, Stype>::value_type;

    value_type *data;
    int64_t count;
    int64_t stride;

    __hostdev__
    vector_mutable_view(value_type *d, int64_t c, int64_t s)
        : data(d), count(c), stride(s) {}

    explicit vector_mutable_view(torch::Tensor &t)
        : data(t.data_ptr<value_type>()), count(t.numel()), stride(t.stride(0)) {}

    __hostdev__ value_type &
    operator[](int64_t i) const {
        return data[i * stride];
    }
};

//------------------------------------------------------------------------------
// Fragment views - vectorized access for flat element-wise operations
//------------------------------------------------------------------------------
// Fragment views load/store multiple elements at once for improved memory
// bandwidth and instruction-level parallelism.
//
// Template parameters:
//   - Dev: torch::DeviceType (kCPU, kCUDA, kPrivateUse1)
//   - Stype: torch::ScalarType (kFloat, kHalf, etc.)
//   - Contig: contiguity (contiguous or strided)
//
// Key concepts:
//   - optimal_load_width: Target bytes per memory transaction (16 bytes = 128 bits)
//   - elements_per_fragment: Elements loaded at once (vectorized for contiguous, 1 for strided)
//   - fragment: Small fixed-size array of elements
//
// Contiguity specialization:
//   - contiguous: Lightweight (data + numel), vectorized fragment loads
//   - strided: Composes flat_*_view for offset computation, elements_per_frag = 1
//
// Usage pattern:
//   1. Dispatch on contiguity to select specialization
//   2. Process data in fragment-sized chunks via for_each on num_fragments()
//   3. For contiguous with elements_per_frag > 1, handle tail with scalar access
//   4. For strided (elements_per_frag = 1), tail_count() is always 0

/// Fragment configuration - specialized per device
template <torch::DeviceType Dev> struct fragment_config {
    /// Optimal memory transaction size in bytes (128-bit loads on GPU)
    static consteval int64_t
    optimal_load_width() {
        return 16;
    }
};

/// CPU doesn't benefit from fragment loads - use element-at-a-time
template <> struct fragment_config<torch::kCPU> {
    static consteval int64_t
    optimal_load_width() {
        return 1;
    }
};

/// Compute elements per fragment for contiguous access
template <torch::DeviceType Dev, torch::ScalarType Stype>
consteval int64_t
elements_per_fragment_contiguous() {
    constexpr int64_t width     = fragment_config<Dev>::optimal_load_width();
    constexpr int64_t elem_size = sizeof(torch_scalar_cpp_type_t<Stype>);
    return (width / elem_size) > 0 ? (width / elem_size) : 1;
}

/// Fragment type: small array of elements for vectorized access
template <typename T, int64_t N> struct fragment {
    T data[N];

    __hostdev__ T &
    operator[](int64_t i) {
        return data[i];
    }
    __hostdev__ T const &
    operator[](int64_t i) const {
        return data[i];
    }

    static consteval int64_t
    size() {
        return N;
    }
};

//------------------------------------------------------------------------------
// flat_fragment_const_view - primary template declaration
//------------------------------------------------------------------------------
template <torch::DeviceType Dev, torch::ScalarType Stype, contiguity Contig>
struct flat_fragment_const_view;

//------------------------------------------------------------------------------
// flat_fragment_const_view - contiguous specialization (lightweight, vectorized)
//------------------------------------------------------------------------------
template <torch::DeviceType Dev, torch::ScalarType Stype>
struct flat_fragment_const_view<Dev, Stype, contiguity::contiguous>
    : device_scalar_pair<Dev, Stype> {
    using typename device_scalar_pair<Dev, Stype>::value_type;

    static consteval int64_t
    elements_per_frag() {
        return elements_per_fragment_contiguous<Dev, Stype>();
    }
    using fragment_type = fragment<value_type, elements_per_frag()>;

    value_type const *data;
    int64_t numel;

    __hostdev__
    flat_fragment_const_view(value_type const *d, int64_t n)
        : data(d), numel(n) {}

    explicit flat_fragment_const_view(torch::Tensor const &t)
        : data(t.data_ptr<value_type>()), numel(t.numel()) {}

    __hostdev__ int64_t
    num_fragments() const {
        return numel / elements_per_frag();
    }

    __hostdev__ int64_t
    tail_count() const {
        return numel % elements_per_frag();
    }

    __hostdev__ int64_t
    tail_offset() const {
        return num_fragments() * elements_per_frag();
    }

    __hostdev__ fragment_type
    load_fragment(int64_t frag_idx) const {
        fragment_type frag;
        value_type const *src = data + frag_idx * elements_per_frag();
        DISPATCH_UNROLL
        for (int64_t i = 0; i < elements_per_frag(); ++i) {
            frag[i] = src[i];
        }
        return frag;
    }

    __hostdev__ value_type
    operator[](int64_t i) const {
        return data[i];
    }
};

//------------------------------------------------------------------------------
// flat_fragment_const_view - strided specialization (composes flat_const_view)
//------------------------------------------------------------------------------
template <torch::DeviceType Dev, torch::ScalarType Stype>
struct flat_fragment_const_view<Dev, Stype, contiguity::strided> : device_scalar_pair<Dev, Stype> {
    using typename device_scalar_pair<Dev, Stype>::value_type;

    static consteval int64_t
    elements_per_frag() {
        return 1; // No vectorization for strided
    }
    using fragment_type = fragment<value_type, 1>;

    flat_const_view<Dev, Stype> inner;

    __hostdev__
    flat_fragment_const_view(flat_const_view<Dev, Stype> const &v)
        : inner(v) {}

    explicit flat_fragment_const_view(torch::Tensor const &t) : inner(t) {}

    __hostdev__ int64_t
    num_fragments() const {
        return inner.numel; // Each element is its own "fragment"
    }

    __hostdev__ int64_t
    tail_count() const {
        return 0; // No tail when elements_per_frag = 1
    }

    __hostdev__ int64_t
    tail_offset() const {
        return inner.numel;
    }

    __hostdev__ fragment_type
    load_fragment(int64_t frag_idx) const {
        return {inner[frag_idx]}; // Uses inner.offset() for strided access
    }

    __hostdev__ value_type
    operator[](int64_t i) const {
        return inner[i];
    }
};

//------------------------------------------------------------------------------
// flat_fragment_mutable_view - primary template declaration
//------------------------------------------------------------------------------
template <torch::DeviceType Dev, torch::ScalarType Stype, contiguity Contig>
struct flat_fragment_mutable_view;

//------------------------------------------------------------------------------
// flat_fragment_mutable_view - contiguous specialization
//------------------------------------------------------------------------------
template <torch::DeviceType Dev, torch::ScalarType Stype>
struct flat_fragment_mutable_view<Dev, Stype, contiguity::contiguous>
    : device_scalar_pair<Dev, Stype> {
    using typename device_scalar_pair<Dev, Stype>::value_type;

    static consteval int64_t
    elements_per_frag() {
        return elements_per_fragment_contiguous<Dev, Stype>();
    }
    using fragment_type = fragment<value_type, elements_per_frag()>;

    value_type *data;
    int64_t numel;

    __hostdev__
    flat_fragment_mutable_view(value_type *d, int64_t n)
        : data(d), numel(n) {}

    explicit flat_fragment_mutable_view(torch::Tensor &t)
        : data(t.data_ptr<value_type>()), numel(t.numel()) {}

    __hostdev__ int64_t
    num_fragments() const {
        return numel / elements_per_frag();
    }

    __hostdev__ int64_t
    tail_count() const {
        return numel % elements_per_frag();
    }

    __hostdev__ int64_t
    tail_offset() const {
        return num_fragments() * elements_per_frag();
    }

    __hostdev__ fragment_type
    load_fragment(int64_t frag_idx) const {
        fragment_type frag;
        value_type const *src = data + frag_idx * elements_per_frag();
        DISPATCH_UNROLL
        for (int64_t i = 0; i < elements_per_frag(); ++i) {
            frag[i] = src[i];
        }
        return frag;
    }

    __hostdev__ void
    store_fragment(int64_t frag_idx, fragment_type const &frag) const {
        value_type *dst = data + frag_idx * elements_per_frag();
        DISPATCH_UNROLL
        for (int64_t i = 0; i < elements_per_frag(); ++i) {
            dst[i] = frag[i];
        }
    }

    __hostdev__ value_type &
    operator[](int64_t i) const {
        return data[i];
    }
};

//------------------------------------------------------------------------------
// flat_fragment_mutable_view - strided specialization
//------------------------------------------------------------------------------
template <torch::DeviceType Dev, torch::ScalarType Stype>
struct flat_fragment_mutable_view<Dev, Stype, contiguity::strided>
    : device_scalar_pair<Dev, Stype> {
    using typename device_scalar_pair<Dev, Stype>::value_type;

    static consteval int64_t
    elements_per_frag() {
        return 1;
    }
    using fragment_type = fragment<value_type, 1>;

    flat_mutable_view<Dev, Stype> inner;

    __hostdev__
    flat_fragment_mutable_view(flat_mutable_view<Dev, Stype> const &v)
        : inner(v) {}

    explicit flat_fragment_mutable_view(torch::Tensor &t) : inner(t) {}

    __hostdev__ int64_t
    num_fragments() const {
        return inner.numel;
    }

    __hostdev__ int64_t
    tail_count() const {
        return 0;
    }

    __hostdev__ int64_t
    tail_offset() const {
        return inner.numel;
    }

    __hostdev__ fragment_type
    load_fragment(int64_t frag_idx) const {
        return {inner[frag_idx]};
    }

    __hostdev__ void
    store_fragment(int64_t frag_idx, fragment_type const &frag) const {
        inner[frag_idx] = frag[0];
    }

    __hostdev__ value_type &
    operator[](int64_t i) const {
        return inner[i];
    }
};

} // namespace dispatch

#endif // DISPATCH_DISPATCH_TORCH_VIEWS_H
