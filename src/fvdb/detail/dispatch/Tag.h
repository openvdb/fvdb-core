// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_DISPATCH_TAG_H
#define FVDB_DETAIL_DISPATCH_TAG_H

#include <type_traits>

namespace fvdb {
namespace dispatch {

// that wrap the device enum values. The Tag template creates a unique type for
// each enum value.

template <auto V> struct Tag {
    static constexpr auto value = V;
};

// -----------------------------------------------------------------------------
// IntDispatchAxis: DispatchAxis from a list of integer values
// -----------------------------------------------------------------------------
// For dispatch over compile-time integer values (e.g., supported channel counts,
// kernel sizes, etc.). Each integer becomes a distinct type via IntegralTag.

// A tag type that wraps an integral value, inheriting from std::integral_constant
// for standard library compatibility (implicit conversion, ::value, etc.)
template <auto V> struct IntegralTag : std::integral_constant<decltype(V), V> {};

} // namespace dispatch
} // namespace fvdb

#endif // FVDB_DETAIL_DISPATCH_TAG_H
