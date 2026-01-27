// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Stride-aware N-dimensional accessor for PyTorch tensors.
// Computes memory offsets from N-D indices using strides, supporting
// both contiguous and non-contiguous tensors without copying.
//
#ifndef DISPATCH_DISPATCH_TORCH_ND_ACCESSOR_H
#define DISPATCH_DISPATCH_TORCH_ND_ACCESSOR_H

#include "dispatch/macros.h"
#include "dispatch/torch/types.h"

#include <torch/torch.h>

#include <array>
#include <cstdint>

namespace dispatch {

//------------------------------------------------------------------------------
// nd_accessor: Stride-aware N-dimensional tensor accessor
//------------------------------------------------------------------------------
// Provides direct N-D indexing into tensor data using strides to compute
// memory offsets. Works with both contiguous and non-contiguous tensors.
//
// Unlike PyTorch's TensorAccessor which uses chained [] operators,
// this accessor takes all indices at once and computes the offset in
// a single operation, making it suitable for use with linear_to_nd.
//
// Template parameters:
//   T    - Scalar type (float, double, etc.)
//   Rank - Number of dimensions
//
// Usage:
//   auto acc = nd_accessor<float, 2>::from_tensor(tensor);
//   float val = acc({i, j});           // Using std::array
//   acc({i, j}) = 42.0f;               // Assignment
//   float val2 = acc(i, j);            // Variadic form

template <typename T, int64_t Rank> struct nd_accessor {
    T *data;
    int64_t strides[Rank]; // in elements, not bytes

    //--------------------------------------------------------------------------
    // Construction from torch::Tensor
    //--------------------------------------------------------------------------

    __hostdev__
    nd_accessor()
        : data(nullptr), strides{} {}

    __hostdev__
    nd_accessor(T *data_ptr, int64_t const *strides_ptr)
        : data(data_ptr) {
        for (int64_t d = 0; d < Rank; ++d) {
            strides[d] = strides_ptr[d];
        }
    }

    // Static factory from torch::Tensor
    // Note: This is a host-only function; the resulting accessor can be used on device
    static nd_accessor
    from_tensor(torch::Tensor t) {
        TORCH_CHECK_VALUE(t.dim() == Rank,
                          "nd_accessor: tensor rank mismatch, expected ",
                          Rank,
                          ", got ",
                          t.dim());

        nd_accessor acc;
        acc.data = t.data_ptr<T>();
        for (int64_t d = 0; d < Rank; ++d) {
            acc.strides[d] = t.stride(d);
        }
        return acc;
    }

    //--------------------------------------------------------------------------
    // N-D indexing via pointer to indices
    //--------------------------------------------------------------------------
    // offset = sum(indices[d] * strides[d] for d in range(Rank))

    __hostdev__ T &
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

    __hostdev__ T &
    operator()(std::array<int64_t, Rank> const &indices) const {
        return (*this)(indices.data());
    }

    //--------------------------------------------------------------------------
    // Variadic indexing: acc(i, j, k)
    //--------------------------------------------------------------------------

    template <typename... Idx>
        requires(sizeof...(Idx) == Rank && (std::is_integral_v<Idx> && ...))
    __hostdev__ T &
    operator()(Idx... indices) const {
        int64_t idx_array[Rank] = {static_cast<int64_t>(indices)...};
        return (*this)(idx_array);
    }

    //--------------------------------------------------------------------------
    // Single linear index access (for Rank=1 tensors)
    //--------------------------------------------------------------------------
    // This provides [] operator for linear indexing when Rank=1

    __hostdev__ T &
    operator[](int64_t idx) const
        requires(Rank == 1)
    {
        return data[idx * strides[0]];
    }

    //--------------------------------------------------------------------------
    // Accessors for shape info (not stored, but useful for debugging)
    //--------------------------------------------------------------------------

    __hostdev__ int64_t
    stride(int64_t d) const {
        return strides[d];
    }
};

//------------------------------------------------------------------------------
// Factory function for creating nd_accessor from torch_concrete_tensor
//------------------------------------------------------------------------------

template <torch::DeviceType Device, torch::ScalarType Stype, size_t Rank>
nd_accessor<torch_scalar_cpp_type_t<Stype>, static_cast<int64_t>(Rank)>
make_nd_accessor(torch_concrete_tensor<Device, Stype, Rank> ct) {
    using T = torch_scalar_cpp_type_t<Stype>;
    return nd_accessor<T, static_cast<int64_t>(Rank)>::from_tensor(ct.tensor);
}

// Convenience overload that takes just a tensor (for when type is known at call site)
template <typename T, int64_t Rank>
nd_accessor<T, Rank>
make_nd_accessor(torch::Tensor t) {
    return nd_accessor<T, Rank>::from_tensor(t);
}

} // namespace dispatch

#endif // DISPATCH_DISPATCH_TORCH_ND_ACCESSOR_H
