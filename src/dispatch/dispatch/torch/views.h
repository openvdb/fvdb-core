// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// views.h — Stride-correct tensor access for CPU and GPU.
//
// Two view families:
//
//   flat_in / flat_out   — Rank-free flat access via operator[](int64_t).
//                          Designed for for_each elementwise patterns where the
//                          tensor's shape is irrelevant and we just need every
//                          element visited once. Handles arbitrary rank internally.
//
//   tensor_in / tensor_out — Multi-index access via operator()(i, j, ...).
//                            Rank is a compile-time template parameter. For
//                            structured access patterns (gather-scatter, morton
//                            encoding, channel ops with explicit indices).
//
// Both families are specialized on contiguity:
//   contiguous — row-major offset, no stride multiply
//   strided    — full stride computation (non-contiguous, transposed, broadcast)
//
// Contiguity is resolved at dispatch time (it's a dispatch axis), not per-element.
// The op author writes one kernel; the dispatch table instantiates both specializations.
//
// All views are trivially copyable PODs after construction, safe for CUDA kernel capture.
// Constructors are host-only (call torch API); access methods are __hostdev__.
//
#ifndef DISPATCH_DISPATCH_TORCH_VIEWS_H
#define DISPATCH_DISPATCH_TORCH_VIEWS_H

#include "dispatch/detail/core_types.h"
#include "dispatch/macros.h"
#include "dispatch/torch/types.h"

#include <torch/types.h>

#include <cassert>
#include <cstdint>

namespace dispatch {

// =============================================================================
// tensor_in — read-only tensor access
// =============================================================================

template <torch::DeviceType Dev,
          torch::ScalarType Stype,
          int64_t Rank,
          contiguity Contig = contiguity::strided>
struct tensor_in;

// -----------------------------------------------------------------------------
// tensor_in — strided specialization
// -----------------------------------------------------------------------------

template <torch::DeviceType Dev, torch::ScalarType Stype, int64_t Rank>
struct tensor_in<Dev, Stype, Rank, contiguity::strided> {
    static_assert(Rank > 0, "Rank must be positive");

    using value_type = torch_scalar_cpp_type_t<Stype>;

    value_type const *data;
    int64_t sizes[Rank];
    int64_t strides[Rank];

    explicit tensor_in(torch::Tensor const &t) : data(t.data_ptr<value_type>()) {
        assert(t.dim() >= Rank && "Tensor rank must be >= view Rank");
        for (int64_t d = 0; d < Rank; ++d) {
            sizes[d]   = t.size(d);
            strides[d] = t.stride(d);
        }
    }

    template <typename... Idx>
    __hostdev__ value_type
    operator()(Idx... idx) const {
        static_assert(sizeof...(Idx) == Rank, "Number of indices must equal Rank");
        int64_t const indices[] = {static_cast<int64_t>(idx)...};
        int64_t offset          = 0;
        for (int64_t d = 0; d < Rank; ++d)
            offset += indices[d] * strides[d];
        return data[offset];
    }

    __hostdev__ int64_t
    size(int64_t d) const {
        return sizes[d];
    }

    __hostdev__ int64_t
    stride(int64_t d) const {
        return strides[d];
    }

    __hostdev__ int64_t
    numel() const {
        int64_t n = 1;
        for (int64_t d = 0; d < Rank; ++d)
            n *= sizes[d];
        return n;
    }
};

// -----------------------------------------------------------------------------
// tensor_in — contiguous specialization
// -----------------------------------------------------------------------------

template <torch::DeviceType Dev, torch::ScalarType Stype, int64_t Rank>
struct tensor_in<Dev, Stype, Rank, contiguity::contiguous> {
    static_assert(Rank > 0, "Rank must be positive");

    using value_type = torch_scalar_cpp_type_t<Stype>;

    value_type const *data;
    int64_t sizes[Rank];

    explicit tensor_in(torch::Tensor const &t) : data(t.data_ptr<value_type>()) {
        assert(t.dim() >= Rank && "Tensor rank must be >= view Rank");
        assert(t.is_contiguous() && "Contiguous view requires contiguous tensor");
        for (int64_t d = 0; d < Rank; ++d) {
            sizes[d] = t.size(d);
        }
    }

    template <typename... Idx>
    __hostdev__ value_type
    operator()(Idx... idx) const {
        static_assert(sizeof...(Idx) == Rank, "Number of indices must equal Rank");
        int64_t const indices[] = {static_cast<int64_t>(idx)...};
        int64_t offset          = indices[0];
        for (int64_t d = 1; d < Rank; ++d)
            offset = offset * sizes[d] + indices[d];
        return data[offset];
    }

    __hostdev__ int64_t
    size(int64_t d) const {
        return sizes[d];
    }

    __hostdev__ int64_t
    stride(int64_t d) const {
        // Compute row-major stride on the fly
        int64_t s = 1;
        for (int64_t i = Rank - 1; i > d; --i)
            s *= sizes[i];
        return s;
    }

    __hostdev__ int64_t
    numel() const {
        int64_t n = 1;
        for (int64_t d = 0; d < Rank; ++d)
            n *= sizes[d];
        return n;
    }
};

// =============================================================================
// tensor_out — writable tensor access
// =============================================================================

template <torch::DeviceType Dev,
          torch::ScalarType Stype,
          int64_t Rank,
          contiguity Contig = contiguity::strided>
struct tensor_out;

// -----------------------------------------------------------------------------
// tensor_out — strided specialization
// -----------------------------------------------------------------------------

template <torch::DeviceType Dev, torch::ScalarType Stype, int64_t Rank>
struct tensor_out<Dev, Stype, Rank, contiguity::strided> {
    static_assert(Rank > 0, "Rank must be positive");

    using value_type = torch_scalar_cpp_type_t<Stype>;

    value_type *data;
    int64_t sizes[Rank];
    int64_t strides[Rank];

    explicit tensor_out(torch::Tensor &t) : data(t.data_ptr<value_type>()) {
        assert(t.dim() >= Rank && "Tensor rank must be >= view Rank");
        for (int64_t d = 0; d < Rank; ++d) {
            sizes[d]   = t.size(d);
            strides[d] = t.stride(d);
        }
    }

    template <typename... Idx>
    __hostdev__ value_type &
    operator()(Idx... idx) const {
        static_assert(sizeof...(Idx) == Rank, "Number of indices must equal Rank");
        int64_t const indices[] = {static_cast<int64_t>(idx)...};
        int64_t offset          = 0;
        for (int64_t d = 0; d < Rank; ++d)
            offset += indices[d] * strides[d];
        return data[offset];
    }

    __hostdev__ int64_t
    size(int64_t d) const {
        return sizes[d];
    }

    __hostdev__ int64_t
    stride(int64_t d) const {
        return strides[d];
    }

    __hostdev__ int64_t
    numel() const {
        int64_t n = 1;
        for (int64_t d = 0; d < Rank; ++d)
            n *= sizes[d];
        return n;
    }
};

// -----------------------------------------------------------------------------
// tensor_out — contiguous specialization
// -----------------------------------------------------------------------------

template <torch::DeviceType Dev, torch::ScalarType Stype, int64_t Rank>
struct tensor_out<Dev, Stype, Rank, contiguity::contiguous> {
    static_assert(Rank > 0, "Rank must be positive");

    using value_type = torch_scalar_cpp_type_t<Stype>;

    value_type *data;
    int64_t sizes[Rank];

    explicit tensor_out(torch::Tensor &t) : data(t.data_ptr<value_type>()) {
        assert(t.dim() >= Rank && "Tensor rank must be >= view Rank");
        assert(t.is_contiguous() && "Contiguous view requires contiguous tensor");
        for (int64_t d = 0; d < Rank; ++d) {
            sizes[d] = t.size(d);
        }
    }

    template <typename... Idx>
    __hostdev__ value_type &
    operator()(Idx... idx) const {
        static_assert(sizeof...(Idx) == Rank, "Number of indices must equal Rank");
        int64_t const indices[] = {static_cast<int64_t>(idx)...};
        int64_t offset          = indices[0];
        for (int64_t d = 1; d < Rank; ++d)
            offset = offset * sizes[d] + indices[d];
        return data[offset];
    }

    __hostdev__ int64_t
    size(int64_t d) const {
        return sizes[d];
    }

    __hostdev__ int64_t
    stride(int64_t d) const {
        int64_t s = 1;
        for (int64_t i = Rank - 1; i > d; --i)
            s *= sizes[i];
        return s;
    }

    __hostdev__ int64_t
    numel() const {
        int64_t n = 1;
        for (int64_t d = 0; d < Rank; ++d)
            n *= sizes[d];
        return n;
    }
};

// =============================================================================
// flat_in — rank-free read-only flat access via operator[]
// =============================================================================
//
// For elementwise ops where for_each iterates [0, numel) and the tensor's shape
// is irrelevant. The view maps a flat linear index to the correct memory location
// regardless of the tensor's actual rank.
//
//   auto v = flat_in<dev, stype, contig>(tensor);
//   val = v[flat_idx];
//

template <torch::DeviceType Dev, torch::ScalarType Stype, contiguity Contig = contiguity::strided>
struct flat_in;

// -----------------------------------------------------------------------------
// flat_in — strided specialization
// -----------------------------------------------------------------------------
// Stores runtime ndim + sizes/strides. operator[] unravels the flat index into
// multi-dimensional indices via div/mod, then computes the physical offset via
// strides. Broadcast dimensions (stride 0) work naturally.

template <torch::DeviceType Dev, torch::ScalarType Stype>
struct flat_in<Dev, Stype, contiguity::strided> {
    static constexpr int kMaxRank = 12;

    using value_type = torch_scalar_cpp_type_t<Stype>;

    value_type const *data;
    int ndim;
    int64_t sizes[kMaxRank];
    int64_t strides[kMaxRank];

    explicit flat_in(torch::Tensor const &t) : data(t.data_ptr<value_type>()), ndim(t.dim()) {
        assert(ndim > 0 && "Tensor must have at least one dimension");
        assert(ndim <= kMaxRank && "Tensor rank exceeds flat_in::kMaxRank");
        for (int d = 0; d < ndim; ++d) {
            sizes[d]   = t.size(d);
            strides[d] = t.stride(d);
        }
    }

    __hostdev__ value_type
    operator[](int64_t flat_idx) const {
        int64_t offset = 0;
        for (int d = ndim - 1; d >= 0; --d) {
            offset += (flat_idx % sizes[d]) * strides[d];
            flat_idx /= sizes[d];
        }
        return data[offset];
    }
};

// -----------------------------------------------------------------------------
// flat_in — contiguous specialization
// -----------------------------------------------------------------------------
// Just a pointer. operator[] is data[flat_idx]. Zero overhead for any rank.

template <torch::DeviceType Dev, torch::ScalarType Stype>
struct flat_in<Dev, Stype, contiguity::contiguous> {
    using value_type = torch_scalar_cpp_type_t<Stype>;

    value_type const *data;

    explicit flat_in(torch::Tensor const &t) : data(t.data_ptr<value_type>()) {
        assert(t.is_contiguous() && "Contiguous flat_in requires contiguous tensor");
    }

    __hostdev__ value_type
    operator[](int64_t flat_idx) const {
        return data[flat_idx];
    }
};

// =============================================================================
// flat_out — rank-free writable flat access via operator[]
// =============================================================================

template <torch::DeviceType Dev, torch::ScalarType Stype, contiguity Contig = contiguity::strided>
struct flat_out;

// -----------------------------------------------------------------------------
// flat_out — strided specialization
// -----------------------------------------------------------------------------

template <torch::DeviceType Dev, torch::ScalarType Stype>
struct flat_out<Dev, Stype, contiguity::strided> {
    static constexpr int kMaxRank = 12;

    using value_type = torch_scalar_cpp_type_t<Stype>;

    value_type *data;
    int ndim;
    int64_t sizes[kMaxRank];
    int64_t strides[kMaxRank];

    explicit flat_out(torch::Tensor &t) : data(t.data_ptr<value_type>()), ndim(t.dim()) {
        assert(ndim > 0 && "Tensor must have at least one dimension");
        assert(ndim <= kMaxRank && "Tensor rank exceeds flat_out::kMaxRank");
        for (int d = 0; d < ndim; ++d) {
            sizes[d]   = t.size(d);
            strides[d] = t.stride(d);
        }
    }

    __hostdev__ value_type &
    operator[](int64_t flat_idx) const {
        int64_t offset = 0;
        for (int d = ndim - 1; d >= 0; --d) {
            offset += (flat_idx % sizes[d]) * strides[d];
            flat_idx /= sizes[d];
        }
        return data[offset];
    }
};

// -----------------------------------------------------------------------------
// flat_out — contiguous specialization
// -----------------------------------------------------------------------------

template <torch::DeviceType Dev, torch::ScalarType Stype>
struct flat_out<Dev, Stype, contiguity::contiguous> {
    using value_type = torch_scalar_cpp_type_t<Stype>;

    value_type *data;

    explicit flat_out(torch::Tensor &t) : data(t.data_ptr<value_type>()) {
        assert(t.is_contiguous() && "Contiguous flat_out requires contiguous tensor");
    }

    __hostdev__ value_type &
    operator[](int64_t flat_idx) const {
        return data[flat_idx];
    }
};

} // namespace dispatch

#endif // DISPATCH_DISPATCH_TORCH_VIEWS_H
