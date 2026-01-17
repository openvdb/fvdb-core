// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef FVDB_DETAIL_DISPATCH_TORCHDISPATCH_H
#define FVDB_DETAIL_DISPATCH_TORCHDISPATCH_H

#include "fvdb/detail/dispatch/TypesFwd.h"

#include <torch/types.h>

#include <concepts>
#include <cstdint>
#include <string>
#include <type_traits>

namespace fvdb {
namespace dispatch {

using CpuCudaDeviceAxis = Values<torch::kCPU, torch::kCUDA>;
using FullDeviceAxis    = Values<torch::kCPU, torch::kCUDA, torch::kPrivateUse1>;

// Use all the float dtypes in torch. Have one that's just regular built-in float types,
// then add the weird float types. Don't use complex.
using FullFloatDtypeAxis =
    Values<torch::kBFloat16, torch::kFloat16, torch::kFloat32, torch::kFloat64>;

// Just the builtin floats
using BuiltinFloatDtypeAxis = Values<torch::kFloat32, torch::kFloat64>;

// All the numeric integer dtypes in torch.
using FullSignedIntDtypeAxis = Values<torch::kInt8, torch::kInt16, torch::kInt32, torch::kInt64>;

// Add all the numeric dtypes in torch, using their bit-width as the value. Only the scalar though,
// no complex.
using FullNumericDtypeAxis = Values<torch::kInt8,
                                    torch::kInt16,
                                    torch::kInt32,
                                    torch::kInt64,
                                    torch::kBFloat16,
                                    torch::kFloat16,
                                    torch::kFloat32,
                                    torch::kFloat64>;

// Concept to match integer torch::ScalarTypes (compile-time)
template <torch::ScalarType T>
constexpr bool is_integer_scalar_type_v =
    T == torch::kByte || T == torch::kChar || T == torch::kShort || T == torch::kInt ||
    T == torch::kLong || T == torch::kBool;

template <torch::ScalarType T>
concept IntegerTorchScalarType = is_integer_scalar_type_v<T>;

// Runtime check for integer scalar types
inline bool
isIntegerScalarType(torch::ScalarType stype) {
    return stype == torch::kByte || stype == torch::kChar || stype == torch::kShort ||
           stype == torch::kInt || stype == torch::kLong || stype == torch::kBool;
}

// Concept to match floating-point torch::ScalarTypes (compile-time)
template <torch::ScalarType T>
constexpr bool is_float_scalar_type_v =
    T == torch::kFloat || T == torch::kDouble || T == torch::kHalf || T == torch::kBFloat16;

template <torch::ScalarType T>
concept FloatTorchScalarType = is_float_scalar_type_v<T>;

// Runtime check for floating-point scalar types
inline bool
isFloatScalarType(torch::ScalarType stype) {
    return stype == torch::kFloat || stype == torch::kDouble || stype == torch::kHalf ||
           stype == torch::kBFloat16;
}

// =============================================================================
// Dispatch Coordinate Stringification
// =============================================================================
//
// Overloads of coordToString for each dispatch coordinate type.
// Each overload produces a labeled string like "device=CUDA" or "dtype=Float".
// Add new overloads here when introducing new dispatch coordinate types.
//

inline std::string
coordToString(c10::DeviceType dev) {
    return std::string("device=") + c10::DeviceTypeName(dev);
}

inline std::string
coordToString(c10::ScalarType dtype) {
    return std::string("dtype=") + c10::toString(dtype);
}

inline std::string
coordToString(Placement p) {
    return std::string("placement=") + toString(p);
}

inline std::string
coordToString(Determinism d) {
    return std::string("determinism=") + toString(d);
}

inline std::string
coordToString(Contiguity c) {
    return std::string("contiguity=") + toString(c);
}

// Fallback for arithmetic types (integers, floats) - no label, just the value
template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
std::string
coordToString(T value) {
    return std::to_string(value);
}

// -----

inline Contiguity
getContiguity(torch::Tensor tensor) {
    return tensor.is_contiguous() ? Contiguity::Contiguous : Contiguity::Strided;
}

// -----------------------------------------------------------------------------
// formatDispatchCoords - format a tuple of dispatch coordinates as a string
// -----------------------------------------------------------------------------
// Produces a comma-separated list like:
//   "device=CUDA, dtype=Float, contiguity=Contiguous"

template <typename... CoordTypes>
std::string
formatDispatchCoords(std::tuple<CoordTypes...> const &coords) {
    return std::apply(
        [](auto const &...values) {
            std::string result;
            bool first = true;
            ((result += (first ? "" : ", ") + coordToString(values), first = false), ...);
            return result;
        },
        coords);
}

// =============================================================================
// torchDispatch - invoke a dispatcher with Torch-friendly error handling
// =============================================================================
//
// Wraps dispatcher invocation, catching any exception from failed dispatch
// lookups and converting them to a user-friendly TORCH_CHECK_VALUE error.
//

template <typename Dispatcher, typename... CoordTypes, typename... Args>
auto
torchDispatch(std::string_view function_name,
              Dispatcher &&dispatcher,
              std::tuple<CoordTypes...> const &dispatch_coord,
              Args &&...args) {
    try {
        return std::invoke(dispatcher, dispatch_coord, std::forward<Args>(args)...);
    } catch (...) {
        TORCH_CHECK_VALUE(false,
                          function_name,
                          ": unsupported dispatch combination - ",
                          formatDispatchCoords(dispatch_coord));
    }
}

} // namespace dispatch
} // namespace fvdb

#endif // FVDB_DETAIL_DISPATCH_TORCHDISPATCH_H
