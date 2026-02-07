// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// axis<V...>: a collection of values that are all the same type and all unique.
// Represents a single dimension of a dispatch space.
//
#ifndef DISPATCH_DISPATCH_AXIS_H
#define DISPATCH_DISPATCH_AXIS_H

#include "dispatch/consteval_types.h"

#include <cstddef>
#include <type_traits>

namespace dispatch {

//------------------------------------------------------------------------------
// axis
//------------------------------------------------------------------------------

template <auto... V> struct axis {
    static_assert(sizeof...(V) > 0, "axis must have at least one value");
};

template <auto V0> struct axis<V0> {
    using value_type = decltype(V0);
};

namespace detail {

template <typename T, typename... Rest>
consteval bool
unique_values(T head, Rest... tail) {
    if constexpr (sizeof...(tail) == 0) {
        return true;
    } else {
        return ((head != tail) && ...) && unique_values(tail...);
    }
}

} // namespace detail

template <auto V0, auto... V> struct axis<V0, V...> {
    using value_type = decltype(V0);
    static_assert((std::is_same_v<value_type, decltype(V)> && ... && true),
                  "axis values must be the same type");
    static_assert(detail::unique_values<value_type>(V0, V...), "axis values must be unique");
};

//------------------------------------------------------------------------------
// is_axis trait
//------------------------------------------------------------------------------

template <typename T> struct is_axis : consteval_false_type {};

template <auto... V> struct is_axis<axis<V...>> : consteval_true_type {};

template <typename T>
consteval bool
is_axis_v() {
    return is_axis<T>::value();
}

template <typename T>
concept axis_like = is_axis_v<T>();

//------------------------------------------------------------------------------
// axis_value_type trait
//------------------------------------------------------------------------------

template <typename T> struct axis_value_type {
    static_assert(axis_like<T>, "axis_value_type requires an axis type");
};

template <axis_like Axis> struct axis_value_type<Axis> {
    using type = typename Axis::value_type;
};

template <typename T> using axis_value_type_t = typename axis_value_type<T>::type;

//------------------------------------------------------------------------------
// extent: number of values in an axis
//------------------------------------------------------------------------------

template <typename T> struct extent;

template <auto... V> struct extent<axis<V...>> {
    static consteval size_t
    value() {
        return sizeof...(V);
    }
};

template <typename T>
consteval size_t
extent_v() {
    return extent<T>::value();
}

} // namespace dispatch

#endif // DISPATCH_DISPATCH_AXIS_H
