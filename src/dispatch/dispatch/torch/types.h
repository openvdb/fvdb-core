// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// PyTorch-specific type mappings and axis typedefs. Lightweight header suitable
// for forward declarations in user headers. Provides torch_scalar_cpp_type_t
// and pre-defined device/scalar-type axes.
//
#ifndef DISPATCH_DISPATCH_TORCH_TYPES_H
#define DISPATCH_DISPATCH_TORCH_TYPES_H

#include "dispatch/types.h"

#include <torch/types.h>

#include <cstddef>
#include <type_traits>

namespace dispatch {

//------------------------------------------------------------------------------
// Scalar C++ type mapping
//------------------------------------------------------------------------------

template <torch::ScalarType S>
using torch_scalar_cpp_type_t = typename c10::impl::ScalarTypeToCPPType<S>::type;

//------------------------------------------------------------------------------
// Device axes
//------------------------------------------------------------------------------

using torch_cpu_cuda_device_axis = axis<torch::kCPU, torch::kCUDA>;
using torch_full_device_axis     = axis<torch::kCPU, torch::kCUDA, torch::kPrivateUse1>;

//------------------------------------------------------------------------------
// Device type classification - compile-time
//------------------------------------------------------------------------------

template <torch::DeviceType D> constexpr bool torch_is_cpu_device_v = (D == torch::kCPU);

template <torch::DeviceType D>
concept torch_cpu_device = torch_is_cpu_device_v<D>;

template <torch::DeviceType D> constexpr bool torch_is_cuda_device_v = (D == torch::kCUDA);

template <torch::DeviceType D>
concept torch_cuda_device = torch_is_cuda_device_v<D>;

template <torch::DeviceType D>
constexpr bool torch_is_gpu_device_v = (D == torch::kCUDA || D == torch::kPrivateUse1);

template <torch::DeviceType D>
concept torch_gpu_device = torch_is_gpu_device_v<D>;

//------------------------------------------------------------------------------
// Device type classification - runtime
//------------------------------------------------------------------------------

inline bool
is_torch_cpu_device(torch::DeviceType dev) {
    return dev == torch::kCPU;
}

inline bool
is_torch_cuda_device(torch::DeviceType dev) {
    return dev == torch::kCUDA;
}

inline bool
is_torch_gpu_device(torch::DeviceType dev) {
    return dev == torch::kCUDA || dev == torch::kPrivateUse1;
}

//------------------------------------------------------------------------------
// Scalar type axes
//------------------------------------------------------------------------------

// All float scalar types in torch (no complex)
using torch_full_float_stype_axis =
    axis<torch::kBFloat16, torch::kFloat16, torch::kFloat32, torch::kFloat64>;

// Just the builtin floats
using torch_builtin_float_stype_axis = axis<torch::kFloat32, torch::kFloat64>;

// All signed integer scalar types in torch
using torch_full_signed_int_stype_axis =
    axis<torch::kInt8, torch::kInt16, torch::kInt32, torch::kInt64>;

// All numeric scalar types in torch (no complex)
using torch_full_numeric_stype_axis = axis<torch::kInt8,
                                           torch::kInt16,
                                           torch::kInt32,
                                           torch::kInt64,
                                           torch::kBFloat16,
                                           torch::kFloat16,
                                           torch::kFloat32,
                                           torch::kFloat64>;

//------------------------------------------------------------------------------
// Scalar type classification - compile-time
//------------------------------------------------------------------------------

template <torch::ScalarType T>
constexpr bool torch_is_integer_stype_v =
    T == torch::kByte || T == torch::kChar || T == torch::kShort || T == torch::kInt ||
    T == torch::kLong || T == torch::kBool;

template <torch::ScalarType T>
concept torch_integer_stype = torch_is_integer_stype_v<T>;

template <torch::ScalarType T>
constexpr bool torch_is_float_stype_v =
    T == torch::kFloat || T == torch::kDouble || T == torch::kHalf || T == torch::kBFloat16;

template <torch::ScalarType T>
concept torch_float_stype = torch_is_float_stype_v<T>;

//------------------------------------------------------------------------------
// Scalar type classification - runtime
//------------------------------------------------------------------------------

inline bool
is_torch_integer_stype(torch::ScalarType stype) {
    return stype == torch::kByte || stype == torch::kChar || stype == torch::kShort ||
           stype == torch::kInt || stype == torch::kLong || stype == torch::kBool;
}

inline bool
is_torch_float_stype(torch::ScalarType stype) {
    return stype == torch::kFloat || stype == torch::kDouble || stype == torch::kHalf ||
           stype == torch::kBFloat16;
}

} // namespace dispatch

#endif // DISPATCH_DISPATCH_TORCH_TYPES_H
