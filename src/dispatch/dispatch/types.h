// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Foundation types for the dispatch system: axis, axes, tag, extents, indices,
// and enums (placement, determinism, contiguity). These provide the compile-time
// type system for defining dispatch spaces and coordinates.
//
#ifndef DISPATCH_DISPATCH_TYPES_H
#define DISPATCH_DISPATCH_TYPES_H

#include <cstddef>
#include <type_traits>

namespace dispatch {

//------------------------------------------------------------------------------
// By avoiding the creation of static constexpr variables in favor of
// consteval functions, we guarantee that those values exist only at
// compile-time and are never instantiated just because they might be passed
// as a reference to some function. It's a little bit more verbose, but it
// addresses a real issue with deeply nested templates and nvcc.
//------------------------------------------------------------------------------

template <bool B> struct consteval_bool_type {
    static consteval bool
    value() {
        return B;
    }
};

using consteval_true_type  = consteval_bool_type<true>;
using consteval_false_type = consteval_bool_type<false>;

//------------------------------------------------------------------------------
// EXTENTS
//------------------------------------------------------------------------------

template <size_t... S> struct extents {};
template <typename T> struct is_extents : consteval_false_type {};
template <size_t... S> struct is_extents<extents<S...>> : consteval_true_type {};
template <typename T>
consteval bool
is_extents_v() {
    return is_extents<T>::value();
}

//------------------------------------------------------------------------------
// INDICES
//------------------------------------------------------------------------------
template <size_t... I> struct indices {};
template <typename T> struct is_indices : consteval_false_type {};
template <size_t... I> struct is_indices<indices<I...>> : consteval_true_type {};
template <typename T>
consteval bool
is_indices_v() {
    return is_indices<T>::value();
}

//------------------------------------------------------------------------------
// TYPES
//------------------------------------------------------------------------------
template <typename... T> struct types {};
template <typename T> struct is_types : consteval_false_type {};
template <typename... T> struct is_types<types<T...>> : consteval_true_type {};
template <typename T>
consteval bool
is_types_v() {
    return is_types<T>::value();
}

//------------------------------------------------------------------------------
// TAG
//------------------------------------------------------------------------------
template <auto... V> struct tag {};
template <typename T> struct is_tag : consteval_false_type {};
template <auto... V> struct is_tag<tag<V...>> : consteval_true_type {};
template <typename T>
consteval bool
is_tag_v() {
    return is_tag<T>::value();
}

//------------------------------------------------------------------------------
// AXIS
//------------------------------------------------------------------------------
// An axis is a collection of values that are all the same type, and are
// all unique.
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

template <typename T> struct is_axis : consteval_false_type {};
template <auto... V> struct is_axis<axis<V...>> : consteval_true_type {};
template <typename T>
consteval bool
is_axis_v() {
    return is_axis<T>::value();
}

//------------------------------------------------------------------------------
// AXES - type
//------------------------------------------------------------------------------
template <typename... Axes> struct axes {
    static_assert(sizeof...(Axes) > 0, "axes must have at least one axis");
    static_assert((is_axis_v<Axes>() && ... && true), "All template parameters must be axis types");
};

template <typename T> struct is_axes : consteval_false_type {};
template <typename... Axes> struct is_axes<axes<Axes...>> : consteval_true_type {};
template <typename T>
consteval bool
is_axes_v() {
    return is_axes<T>::value();
}

//------------------------------------------------------------------------------
// ENUMS
//------------------------------------------------------------------------------
enum class placement { in_place, out_of_place };
enum class determinism { not_required, required };
enum class contiguity { strided, contiguous };

using full_placement_axis   = axis<placement::in_place, placement::out_of_place>;
using full_determinism_axis = axis<determinism::not_required, determinism::required>;
using full_contiguity_axis  = axis<contiguity::strided, contiguity::contiguous>;

// Stringification helpers for enum values
inline char const *
to_string(placement p) {
    switch (p) {
    case placement::in_place: return "in_place";
    case placement::out_of_place: return "out_of_place";
    }
    return "unknown";
}

inline char const *
to_string(determinism d) {
    switch (d) {
    case determinism::not_required: return "not_required";
    case determinism::required: return "required";
    }
    return "unknown";
}

inline char const *
to_string(contiguity c) {
    switch (c) {
    case contiguity::strided: return "strided";
    case contiguity::contiguous: return "contiguous";
    }
    return "unknown";
}

} // namespace dispatch

#endif // DISPATCH_DISPATCH_TYPES_H
