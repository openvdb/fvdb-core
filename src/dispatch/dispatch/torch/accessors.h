// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Unified tensor accessor with contiguity-aware specializations.
//
// Provides two accessor specializations based on contiguity:
//   - contiguous: Direct data[idx] access (no stride math)
//   - strided: N-D indexing with stride computation
//
// Unlike PyTorch's TensorAccessor/PackedTensorAccessor which always compute
// stride offsets, our contiguous specialization eliminates N multiplications
// per element access for contiguous tensors.
//
// The accessor takes torch::ScalarType as a template parameter and derives
// the C++ value_type internally, so callers work with torch types directly.
//
#ifndef DISPATCH_DISPATCH_TORCH_ACCESSORS_H
#define DISPATCH_DISPATCH_TORCH_ACCESSORS_H

#include "dispatch/macros.h"
#include "dispatch/torch/types.h"
#include "dispatch/types.h"

#include <torch/torch.h>

#include <array>
#include <cstdint>

namespace dispatch {

//------------------------------------------------------------------------------
// accessor: Unified tensor accessor with contiguity specializations
//------------------------------------------------------------------------------
// Template parameters:
//   Stype  - torch::ScalarType (e.g., torch::kFloat32)
//   Contig - contiguity enum (contiguous or strided)
//   Rank   - Number of dimensions (default 1)
//
// The accessor derives value_type from torch_scalar_cpp_type_t<Stype> internally,
// so callers pass torch types directly without manual type conversion.
//
// Usage:
//   auto acc = accessor<torch::kFloat32, contiguity::contiguous, 1>::from_tensor(t);
//   acc[idx] = 42.0f;  // contiguous: direct linear access
//
//   auto acc2 = accessor<torch::kFloat32, contiguity::strided, 2>::from_tensor(t);
//   acc2(i, j) = 42.0f;  // strided: N-D access with stride computation

// Primary template declaration
template <torch::ScalarType Stype, contiguity Contig, int64_t Rank = 1> struct accessor;

//------------------------------------------------------------------------------
// Contiguous specialization: direct linear indexing
//------------------------------------------------------------------------------
// For contiguous tensors, we can access elements directly via data[idx]
// without any stride computation. This eliminates N multiplications per access.

template <torch::ScalarType Stype, int64_t Rank>
struct accessor<Stype, contiguity::contiguous, Rank> {
    using value_type = torch_scalar_cpp_type_t<Stype>;

    value_type *data;

    //--------------------------------------------------------------------------
    // Construction
    //--------------------------------------------------------------------------

    __hostdev__
    accessor()
        : data(nullptr) {}

    __hostdev__ explicit accessor(value_type *ptr) : data(ptr) {}

    //--------------------------------------------------------------------------
    // Factory from torch::Tensor
    //--------------------------------------------------------------------------
    // Host-only; resulting accessor can be used on device.

    static accessor
    from_tensor(torch::Tensor t) {
        TORCH_CHECK_VALUE(
            t.dim() == Rank, "accessor: tensor rank mismatch, expected ", Rank, ", got ", t.dim());
        return accessor{t.data_ptr<value_type>()};
    }

    //--------------------------------------------------------------------------
    // Linear indexing: acc[idx]
    //--------------------------------------------------------------------------
    // Direct access without stride computation.

    __hostdev__ value_type &
    operator[](int64_t idx) const {
        return data[idx];
    }
};

//------------------------------------------------------------------------------
// Strided specialization: N-D indexing with stride computation
//------------------------------------------------------------------------------
// For non-contiguous tensors (views, transposes, etc.), we compute
// offset = sum(indices[d] * strides[d]) to find the element.

template <torch::ScalarType Stype, int64_t Rank> struct accessor<Stype, contiguity::strided, Rank> {
    using value_type = torch_scalar_cpp_type_t<Stype>;

    value_type *data;
    int64_t strides[Rank]; // in elements, not bytes

    //--------------------------------------------------------------------------
    // Construction
    //--------------------------------------------------------------------------

    __hostdev__
    accessor()
        : data(nullptr), strides{} {}

    __hostdev__
    accessor(value_type *data_ptr, int64_t const *strides_ptr)
        : data(data_ptr) {
        for (int64_t d = 0; d < Rank; ++d) {
            strides[d] = strides_ptr[d];
        }
    }

    //--------------------------------------------------------------------------
    // Factory from torch::Tensor
    //--------------------------------------------------------------------------
    // Host-only; resulting accessor can be used on device.

    static accessor
    from_tensor(torch::Tensor t) {
        TORCH_CHECK_VALUE(
            t.dim() == Rank, "accessor: tensor rank mismatch, expected ", Rank, ", got ", t.dim());

        accessor acc;
        acc.data = t.data_ptr<value_type>();
        for (int64_t d = 0; d < Rank; ++d) {
            acc.strides[d] = t.stride(d);
        }
        return acc;
    }

    //--------------------------------------------------------------------------
    // N-D indexing via pointer to indices
    //--------------------------------------------------------------------------
    // offset = sum(indices[d] * strides[d] for d in range(Rank))

    __hostdev__ value_type &
    operator()(int64_t const *indices) const {
        int64_t offset = 0;
        DISPATCH_UNROLL
        for (int64_t d = 0; d < Rank; ++d) {
            offset += indices[d] * strides[d];
        }
        return data[offset];
    }

    //--------------------------------------------------------------------------
    // N-D indexing via std::array
    //--------------------------------------------------------------------------

    __hostdev__ value_type &
    operator()(std::array<int64_t, Rank> const &indices) const {
        return (*this)(indices.data());
    }

    //--------------------------------------------------------------------------
    // Variadic indexing: acc(i, j, k)
    //--------------------------------------------------------------------------

    template <typename... Idx>
        requires(sizeof...(Idx) == Rank && (std::is_integral_v<Idx> && ...))
    __hostdev__ value_type &
    operator()(Idx... indices) const {
        int64_t idx_array[Rank] = {static_cast<int64_t>(indices)...};
        return (*this)(idx_array);
    }

    //--------------------------------------------------------------------------
    // Stride accessor (for debugging)
    //--------------------------------------------------------------------------

    __hostdev__ int64_t
    stride(int64_t d) const {
        return strides[d];
    }
};

} // namespace dispatch

#endif // DISPATCH_DISPATCH_TORCH_ACCESSORS_H
