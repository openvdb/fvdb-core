// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// PyTorch dispatch utilities: coordinate stringification and dispatch wrapper
// with Torch-friendly error handling.
//
#ifndef DISPATCH_DISPATCH_TORCH_DISPATCH_H
#define DISPATCH_DISPATCH_TORCH_DISPATCH_H

#include "dispatch/dispatch_table.h"
#include "dispatch/torch/types.h"

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
// Contiguity helpers
//------------------------------------------------------------------------------

inline contiguity
torch_get_contiguity(torch::Tensor tensor) {
    return tensor.is_contiguous() ? contiguity::contiguous : contiguity::strided;
}

// Returns contiguous ONLY if ALL tensors are contiguous.
// Prevents combinatorial explosion in binary/ternary ops by making contiguity
// a single boolean decision rather than 2^N dispatch combinations.
//
// Usage:
//   auto contig = combined_contiguity(input, output);
//   auto contig = combined_contiguity(lhs, rhs, output);
template <typename... Tensors>
    requires(std::is_same_v<std::remove_cvref_t<Tensors>, torch::Tensor> && ...)
contiguity
combined_contiguity(Tensors const &...tensors) {
    bool const all_contiguous = (tensors.is_contiguous() && ...);
    return all_contiguous ? contiguity::contiguous : contiguity::strided;
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
    } catch (dispatch_lookup_error const &) {
        // Only catch dispatch lookup failures - other exceptions propagate unchanged
        TORCH_CHECK_VALUE(false,
                          function_name,
                          ": unsupported dispatch combination - ",
                          torch_format_dispatch_coords(dispatch_coord));
    }
}

} // namespace dispatch

#endif // DISPATCH_DISPATCH_TORCH_DISPATCH_H
