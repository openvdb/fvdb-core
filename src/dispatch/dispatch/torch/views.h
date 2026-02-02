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

} // namespace dispatch

#endif // DISPATCH_DISPATCH_TORCH_VIEWS_H
