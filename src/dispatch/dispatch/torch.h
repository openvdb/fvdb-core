// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef DISPATCH_DISPATCH_TORCH_H
#define DISPATCH_DISPATCH_TORCH_H

#include "dispatch/torch_types.h"

#include <ATen/core/TensorAccessor.h>

#include <cstdint>
#include <functional>
#include <string>
#include <tuple>
#include <type_traits>

namespace dispatch {

//------------------------------------------------------------------------------
// Dispatch coordinate stringification
//------------------------------------------------------------------------------
// Overloads of torch_coord_to_string for each dispatch coordinate type.
// Each overload produces a labeled string like "device=CUDA" or "stype=Float".
// Add new overloads here when introducing new dispatch coordinate types.

inline std::string
torch_coord_to_string(c10::DeviceType dev) {
    return std::string("device=") + c10::DeviceTypeName(dev);
}

inline std::string
torch_coord_to_string(c10::ScalarType stype) {
    return std::string("stype=") + c10::toString(stype);
}

inline std::string
torch_coord_to_string(placement p) {
    return std::string("placement=") + to_string(p);
}

inline std::string
torch_coord_to_string(determinism d) {
    return std::string("determinism=") + to_string(d);
}

inline std::string
torch_coord_to_string(contiguity c) {
    return std::string("contiguity=") + to_string(c);
}

// Fallback for arithmetic types (integers, floats) - no label, just the value
template <typename T>
    requires std::is_arithmetic_v<T>
std::string
torch_coord_to_string(T value) {
    return std::to_string(value);
}

//------------------------------------------------------------------------------
// Contiguity helper
//------------------------------------------------------------------------------

inline contiguity
torch_get_contiguity(torch::Tensor tensor) {
    return tensor.is_contiguous() ? contiguity::contiguous : contiguity::strided;
}

//------------------------------------------------------------------------------
// torch_format_dispatch_coords - format a tuple of dispatch coordinates
//------------------------------------------------------------------------------
// Produces a comma-separated list like:
//   "device=CUDA, stype=Float, contiguity=contiguous"

template <typename... CoordTypes>
std::string
torch_format_dispatch_coords(std::tuple<CoordTypes...> const &coords) {
    return std::apply(
        [](auto const &...values) {
            std::string result;
            bool first = true;
            ((result += (first ? "" : ", ") + torch_coord_to_string(values), first = false), ...);
            return result;
        },
        coords);
}

//------------------------------------------------------------------------------
// torch_dispatch - invoke a dispatcher with Torch-friendly error handling
//------------------------------------------------------------------------------
// Wraps dispatcher invocation, catching any exception from failed dispatch
// lookups and converting them to a user-friendly TORCH_CHECK_VALUE error.

template <typename Dispatcher, typename... CoordTypes, typename... Args>
auto
torch_dispatch(std::string_view function_name,
               Dispatcher &&dispatcher,
               std::tuple<CoordTypes...> const &dispatch_coord,
               Args &&...args) {
    try {
        return std::invoke(dispatcher, dispatch_coord, std::forward<Args>(args)...);
    } catch (...) {
        TORCH_CHECK_VALUE(false,
                          function_name,
                          ": unsupported dispatch combination - ",
                          torch_format_dispatch_coords(dispatch_coord));
    }
}

//------------------------------------------------------------------------------
// torch_concrete_tensor CPU accessor
//------------------------------------------------------------------------------
// Extracts the C++ scalar type from the ScalarType enum to call torch's accessor.
// Matches any torch_concrete_tensor with CPU device.

template <torch::ScalarType Stype, size_t Rank>
auto
torch_accessor(torch_concrete_tensor<torch::kCPU, Stype, Rank> ct) {
    using scalar_t = torch_scalar_cpp_type_t<Stype>;
    // Note: Host TensorAccessor only takes <T, N>, no index type parameter
    // (unlike packed_accessor64/32 for CUDA)
    return ct.tensor.template accessor<scalar_t, Rank>();
}

#ifdef __CUDACC__
//------------------------------------------------------------------------------
// torch_concrete_tensor CUDA/PrivateUse1 accessors (nvcc only)
//------------------------------------------------------------------------------
// Extracts the C++ scalar type from the ScalarType enum to call torch's packed_accessor.
// Matches any torch_concrete_tensor with CUDA or PrivateUse1 device.
// These use RestrictPtrTraits which requires nvcc compilation.

template <torch::ScalarType Stype, size_t Rank, typename IndexT = int64_t>
auto
torch_accessor(torch_concrete_tensor<torch::kCUDA, Stype, Rank> ct) {
    using scalar_t = torch_scalar_cpp_type_t<Stype>;
    if constexpr (std::is_same_v<IndexT, int64_t>) {
        return ct.tensor.template packed_accessor64<scalar_t, Rank, at::RestrictPtrTraits>();
    } else {
        return ct.tensor.template packed_accessor32<scalar_t, Rank, at::RestrictPtrTraits>();
    }
}

template <torch::ScalarType Stype, size_t Rank, typename IndexT = int64_t>
auto
torch_accessor(torch_concrete_tensor<torch::kPrivateUse1, Stype, Rank> ct) {
    using scalar_t = torch_scalar_cpp_type_t<Stype>;
    if constexpr (std::is_same_v<IndexT, int64_t>) {
        return ct.tensor.template packed_accessor64<scalar_t, Rank, at::RestrictPtrTraits>();
    } else {
        return ct.tensor.template packed_accessor32<scalar_t, Rank, at::RestrictPtrTraits>();
    }
}
#endif // __CUDACC__

} // namespace dispatch

#endif // DISPATCH_DISPATCH_TORCH_H
