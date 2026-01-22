// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef DISPATCH_TRAITS_H
#define DISPATCH_TRAITS_H

#include "dispatch/types.h"

#include <array>
#include <cstddef>
#include <tuple>
#include <utility>

namespace dispatch {

// -----------------------------------------------------------------------------
// By avoiding the creation of static constexpr variables in favor of
// consteval functions, we guarantee that those values exist only at
// compile-time and are never instantiated just because they might be passed
// as a reference to some function. It's a little bit more verbose, but it
// addresses a real issue with deeply nested templates and nvcc.

template <bool B> struct consteval_bool_type {
    static consteval bool
    value() {
        return B;
    }
};

using consteval_true_type  = consteval_bool_type<true>;
using consteval_false_type = consteval_bool_type<false>;

//------------------------------------------------------------------------------
// SIZES - trait
//------------------------------------------------------------------------------

// actual sizes type comes from types.h
// template <size_t... S> struct sizes {};

template <typename T> struct is_sizes : consteval_false_type {};
template <size_t... S> struct is_sizes<sizes<S...>> : consteval_true_type {};

template <typename T>
consteval bool
is_sizes_v() {
    return is_sizes<T>::value();
}

//------------------------------------------------------------------------------
// INDICES - trait
//------------------------------------------------------------------------------

// actual indices type comes from types.h
// template <size_t... I> struct indices {};

template <typename T> struct is_indices : consteval_false_type {};
template <size_t... I> struct is_indices<indices<I...>> : consteval_true_type {};

template <typename T>
consteval bool
is_indices_v() {
    return is_indices<T>::value();
}

//------------------------------------------------------------------------------
// TYPES - trait
//------------------------------------------------------------------------------

// actual types type comes from types.h
// template <typename... T> struct types {};

template <typename T> struct is_types : consteval_false_type {};
template <typename... T> struct is_types<types<T...>> : consteval_true_type {};

template <typename T>
consteval bool
is_types_v() {
    return is_types<T>::value();
}

//------------------------------------------------------------------------------
// AXIS - trait
//------------------------------------------------------------------------------

// actual axis type comes from types.h
// template <auto... V> struct axis {};

template <typename T> struct is_axis : consteval_false_type {};
template <auto... V> struct is_axis<axis<V...>> : consteval_true_type {};

template <typename T>
consteval bool
is_axis_v() {
    return is_axis<T>::value();
}

//------------------------------------------------------------------------------
// TAG - trait
//------------------------------------------------------------------------------

// actual tag type comes from types.h
// template <auto... V> struct tag {};

template <typename T> struct is_tag : consteval_false_type {};
template <auto... V> struct is_tag<tag<V...>> : consteval_true_type {};

template <typename T>
consteval bool
is_tag_v() {
    return is_tag<T>::value();
}

//------------------------------------------------------------------------------
// AXIS_TYPES - trait
//------------------------------------------------------------------------------

// types from axis
template <typename Axis> struct axis_types {
    static_assert(is_axis_v<Axis>(), "axis_types requires an axis type");
};

template <auto... V> struct axis_types<axis<V...>> {
    using type = types<decltype(V)...>;
};

template <typename Axis> using axis_types_t = typename axis_types<Axis>::type;

//------------------------------------------------------------------------------
// AXES - trait
//------------------------------------------------------------------------------

template <typename T> struct is_axes : consteval_false_type {};
template <axis... Axes> struct is_axes<axes<Axes...>> : consteval_true_type {};

template <typename T>
consteval bool
is_axes_v() {
    return is_axes<T>::value();
}

//------------------------------------------------------------------------------
// MONOTYPED - trait and concept
//------------------------------------------------------------------------------

template <typename T> struct is_monotyped : consteval_false_type {};

template <> struct is_monotyped<types<>> : consteval_true_type {};
template <typename T> struct is_monotyped<types<T>> : consteval_true_type {};
template <typename T, typename... Ts> struct is_monotyped<types<T, Ts...>> {
    static consteval bool
    value() {
        return (std::is_same_v<T, Ts> && ... && true);
    }
};

template <auto... V> struct is_monotyped<axis<V...>> : is_monotyped<axis_types_t<axis<V...>>> {};

template <typename T>
consteval bool
is_monotyped_v() {
    return is_monotyped<T>::value();
}

template <typename T>
concept monotyped = is_monotyped_v<T>();

//------------------------------------------------------------------------------
// SPACE - concept and trait
//------------------------------------------------------------------------------

template <typename T> struct is_space : consteval_false_type {};

template <typename T>
consteval bool
is_space_v() {
    return is_space<T>::value();
}

template <typename T>
concept space = is_space_v<T>();

//------------------------------------------------------------------------------
// POINT - concept and trait
//------------------------------------------------------------------------------
template <typename T> struct is_point : consteval_false_type {};

template <typename T>
consteval bool
is_point_v() {
    return is_point<T>::value();
}

template <typename T>
concept point = is_point_v<T>();

//------------------------------------------------------------------------------
// DIMENSIONAL - concept and trait
//------------------------------------------------------------------------------
template <typename T> struct is_dimensional : consteval_false_type {};

template <space T> struct is_dimensional<T> : consteval_true_type {};

template <point T> struct is_dimensional<T> : consteval_true_type {};

template <typename T>
consteval bool
is_dimensional_v() {
    return is_dimensional<T>::value();
}

template <typename T>
concept dimensional = is_dimensional_v<T>();

//------------------------------------------------------------------------------
// Ndim - number of dimensions in a type
//------------------------------------------------------------------------------
template <typename T> struct ndim {
    static_assert(dimensional<T>, "ndim requires a dimensional type");
}

template <typename T>
consteval size_t
ndim_v() {
    return ndim<T>::value();
}

//------------------------------------------------------------------------------
// Equidimensional - check if two types have the same number of dimensions
//------------------------------------------------------------------------------
template <typename T1, typename T2> struct is_equidimensional_with {
    static_assert(dimensional<T1> && dimensional<T2>,
                  "is_equidimensional_with requires two dimensional types");
    static consteval bool
    value() {
        return ndim_v<T1>() == ndim_v<T2>();
    }
};

template <typename T1, typename T2>
consteval bool
is_equidimensional_with_v() {
    return is_equidimensional_with<T1, T2>::value();
}

template <typename T1, typename T2>
concept equidimensional_with =
    dimensional<T1> && dimensional<T2> && is_equidimensional_with_v<T1, T2>();

//------------------------------------------------------------------------------
// Volume - number of elements for the space. Only defined for spaces, not
// for points.
//------------------------------------------------------------------------------
template <typename T> struct volume {
    static_assert(space<T>, "volume requires a space type");
};

template <typename T>
consteval size_t
volume_v() {
    return volume<T>::value();
}

// -----------------------------------------------------------------------------
// NON_EMPTY - concept and trait
// -----------------------------------------------------------------------------
template <typename T> struct is_non_empty {
    static_assert(space<T>, "is_non_empty requires a space type");
    static consteval bool
    value() {
        return volume_v<T>() > 0;
    }
};

template <typename T>
consteval bool
is_non_empty_v() {
    return volume_v<T>() > 0;
}

template <typename T>
concept non_empty = space<T> && is_non_empty_v<T>();

// -----------------------------------------------------------------------------
// WITHIN (contained by)
// concept and trait for checking if a point is within a space
// or a space is a subspace of another space
// -----------------------------------------------------------------------------

template <typename Sub, typename Full> struct is_within : consteval_false_type {
    static_assert(dimensional<Sub> && space<Full>,
                  "is_within requires a dimensional type and a space type");
    static_assert(equidimensional_with<Sub, Full>,
                  "is_within requires the sub and full types to be equidimensional");
};

template <typename Sub, typename Full>
consteval bool
is_within_v() {
    return is_within<Sub, Full>::value();
}

template <typename Sub, typename Full>
concept within =
    dimensional<Sub> && space<Full> && equidimensional_with<Sub, Full> && is_within_v<Sub, Full>();

// -----------------------------------------------------------------------------
// Prepend value - trait for adding a new value to a value sequence
// -----------------------------------------------------------------------------

template <auto Value, typename Sequence> struct prepend_value;

template <size_t S, size_t... Ss> struct prepend_value<S, sizes<Ss...>> {
    using type = sizes<S, Ss...>;
};

template <size_t I, size_t... Is> struct prepend_value<I, indices<Is...>> {
    using type = indices<I, Is...>;
};

template <auto V, auto... Vs> struct prepend_value<V, axis<Vs...>> {
    using type = axis<V, Vs...>;
};

template <auto V, auto... Vs> struct prepend_value<V, tag<Vs...>> {
    using type = tag<V, Vs...>;
};

template <auto Value, typename Sequence>
using prepend_value_t = typename prepend_value<Value, Sequence>::type;

// -----------------------------------------------------------------------------
// Prepend type - trait for adding a new type to a type sequence
// -----------------------------------------------------------------------------

template <typename Type, typename Sequence> struct prepend_type;

template <typename Type, typename... Types> struct prepend_type<Type, types<Types...>> {
    using type = types<Type, Types...>;
};

template <typename Type, typename... Types> struct prepend_type<Type, axis<Types...>> {
    using type = axis<Type, Types...>;
};

template <typename Type, typename Sequence>
using prepend_type_t = typename prepend_type<Type, Sequence>::type;

// -----------------------------------------------------------------------------
// array_from_indices - convert index_sequence to std::array at compile time
// -----------------------------------------------------------------------------

template <typename T> struct array_from;

template <size_t... I> struct array_from<std::index_sequence<I...>> {
    static constexpr std::array<size_t, sizeof...(I)> value = {I...};
};

template <size_t... S> struct array_from<sizes<S...>> {
    static constexpr std::array<size_t, sizeof...(S)> value = {S...};
};

template <size_t... I> struct array_from<indices<I...>> {
    static constexpr std::array<size_t, sizeof...(I)> value = {I...};
};

// =============================================================================
// Tuple Utilities
// =============================================================================
//
// Compile-time and constexpr utilities for tuple manipulation.
//
// =============================================================================

// -----------------------------------------------------------------------------
// tuple_head - extract the first element of a tuple
// -----------------------------------------------------------------------------

template <typename Tuple>
constexpr auto
tuple_head(Tuple const &t) {
    return std::get<0>(t);
}

// -----------------------------------------------------------------------------
// tuple_tail - extract all elements except the first
// -----------------------------------------------------------------------------
// Returns a new tuple containing elements [1, N) from the input tuple.
// For a tuple of size 1, returns an empty tuple.

template <typename Tuple>
constexpr auto
tuple_tail(Tuple const &t) {
    return std::apply([](auto, auto... tail) { return std::make_tuple(tail...); }, t);
}

// -----------------------------------------------------------------------------
// TupleTail_t - compile-time type of tuple_tail result
// -----------------------------------------------------------------------------

template <typename Tuple> struct TupleTail;

template <typename Head, typename... Tail> struct TupleTail<std::tuple<Head, Tail...>> {
    using type = std::tuple<Tail...>;
};

template <typename Tuple> using TupleTail_t = typename TupleTail<Tuple>::type;

// -----------------------------------------------------------------------------
// TupleHead_t - compile-time type of the first element
// -----------------------------------------------------------------------------

template <typename Tuple> struct TupleHead;

template <typename Head, typename... Tail> struct TupleHead<std::tuple<Head, Tail...>> {
    using type = Head;
};

template <typename Tuple> using TupleHead_t = typename TupleHead<Tuple>::type;

} // namespace dispatch

#endif // DISPATCH_TRAITS_H
