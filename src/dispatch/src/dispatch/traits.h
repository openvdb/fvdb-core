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

template <typename T>
concept extents_like = is_extents_v<T>();
template <typename T>
concept indices_like = is_indices_v<T>();
template <typename T>
concept types_like = is_types_v<T>();
template <typename T>
concept tag_like = is_tag_v<T>();
template <typename T>
concept axis_like = is_axis_v<T>();
template <typename T>
concept axes_like = is_axes_v<T>();

template <typename T> struct axis_value_type {
    static_assert(axis_like<T>, "axis_value_type requires a axis type");
};

template <axis_like Axis> struct axis_value_type<Axis> {
    using type = typename Axis::value_type;
};

template <typename T> using axis_value_type_t = typename axis_value_type<T>::type;

//------------------------------------------------------------------------------
// EXTENT - trait
//------------------------------------------------------------------------------
// extent - this is basically "size", though avoiding the use of the term
// which is overloaded. We keep ndim, volume, and extent distinct.
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

//------------------------------------------------------------------------------
// Ndim - number of dimensions in a type
//------------------------------------------------------------------------------
template <typename T> struct ndim;

template <size_t... S> struct ndim<extents<S...>> {
    static consteval size_t
    value() {
        return sizeof...(S);
    }
};

template <size_t... I> struct ndim<indices<I...>> {
    static consteval size_t
    value() {
        return sizeof...(I);
    }
};

template <auto... V> struct ndim<tag<V...>> {
    static consteval size_t
    value() {
        return sizeof...(V);
    }
};

// An axis is always 1-dimensional.
template <auto... V> struct ndim<axis<V...>> {
    static consteval size_t
    value() {
        return 1;
    }
};

template <typename... Axes> struct ndim<axes<Axes...>> {
    static consteval size_t
    value() {
        return sizeof...(Axes);
    }
};

template <typename T>
consteval size_t
ndim_v() {
    return ndim<T>::value();
}

//------------------------------------------------------------------------------
// Equidimensional - check if two types have the same number of dimensions
//------------------------------------------------------------------------------
template <typename T1, typename T2> struct is_equidimensional_with {
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
concept equidimensional_with = is_equidimensional_with_v<T1, T2>();

//------------------------------------------------------------------------------
// Volume - number of elements for the space. Only defined for spaces, not
// for points.
//------------------------------------------------------------------------------
template <typename T> struct volume {
    static consteval size_t
    value() {
        return 0;
    }
};

template <size_t... S> struct volume<extents<S...>> {
    static consteval size_t
    value() {
        return (S * ... * 1);
    }
};

template <typename... Axes> struct volume<axes<Axes...>> {
    static consteval size_t
    value() {
        return (extent_v<Axes>() * ... * 1);
    }
};

template <typename T>
consteval size_t
volume_v() {
    return volume<T>::value();
}

// -----------------------------------------------------------------------------
// NON_EMPTY - concept and trait
// -----------------------------------------------------------------------------
template <typename T> struct is_non_empty : consteval_bool_type<(volume_v<T>() > 0)> {};

template <typename T>
consteval bool
is_non_empty_v() {
    return is_non_empty<T>::value();
}

template <typename T>
concept non_empty = is_non_empty_v<T>();

// -----------------------------------------------------------------------------
// WITHIN (contained by)
// concept and trait for checking if a point is within a space
// or a space is a subspace of another space
// -----------------------------------------------------------------------------

template <typename Sub, typename Full> struct is_within : consteval_false_type {
    static_assert(equidimensional_with<Sub, Full>,
                  "is_within requires the sub and full types to be equidimensional");
};

template <typename Sub, typename Full>
consteval bool
is_within_v() {
    return is_within<Sub, Full>::value();
}

template <typename Sub, typename Full>
concept within = is_within_v<Sub, Full>();

// index point in index space
template <size_t... Sub, size_t... Full> struct is_within<indices<Sub...>, extents<Full...>> {
    using sub_type  = indices<Sub...>;
    using full_type = extents<Full...>;
    static_assert(equidimensional_with<sub_type, full_type>,
                  "is_within requires the sub and full to be equidimensional");
    static_assert(non_empty<full_type>, "is_within requires the full to be non-empty");
    static consteval bool
    value() {
        return (Sub < Full && ... && true);
    }
};

// index space in index space
template <size_t... Sub, size_t... Full> struct is_within<extents<Sub...>, extents<Full...>> {
    using sub_type  = extents<Sub...>;
    using full_type = extents<Full...>;
    static_assert(equidimensional_with<sub_type, full_type>,
                  "is_within requires the sub and full to be equidimensional");
    static_assert(non_empty<sub_type> && non_empty<full_type>,
                  "is_within requires the sub and full to be non-empty");
    static consteval bool
    value() {
        return (Sub <= Full && ... && true);
    }
};

// Within for single value and axes
template <auto SubV, auto... FullV> struct is_within<axis<SubV>, axis<FullV...>> {
    using sub_type        = axis<SubV>;
    using sub_value_type  = axis_value_type_t<sub_type>;
    using full_type       = axis<FullV...>;
    using full_value_type = axis_value_type_t<full_type>;
    static_assert(std::is_same_v<sub_value_type, full_value_type>,
                  "is_within requires the sub and full to have the same value type");

    // We know that the value types are unique in the full, because it's impossible
    // to create an axis with non-unique values.
    static consteval bool
    value() {
        return (SubV == FullV || ... || false);
    }
};

// Within for a sub-axis and a full-axis
template <auto... SubV, auto... FullV> struct is_within<axis<SubV...>, axis<FullV...>> {
    using sub_type        = axis<SubV...>;
    using sub_value_type  = axis_value_type_t<sub_type>;
    using full_type       = axis<FullV...>;
    using full_value_type = axis_value_type_t<full_type>;
    static_assert(std::is_same_v<sub_value_type, full_value_type>,
                  "is_within requires the sub and full to have the same value type");

    // Each sub value must be within the full values.
    static consteval bool
    value() {
        return (within<axis<SubV>, full_type> && ... && true);
    }
};

// Within for a single value tag and a single axis axes
template <auto SubV, typename AxisType> struct is_within<tag<SubV>, axes<AxisType>> {
    static consteval bool
    value() {
        return within<axis<SubV>, AxisType>;
    }
};

// Within for multi-value tag and multi-axis axes
template <auto Sub0, auto... SubV, typename AxisType0, typename... AxisTypes>
struct is_within<tag<Sub0, SubV...>, axes<AxisType0, AxisTypes...>> {
    static consteval bool
    value() {
        return within<axis<Sub0>, AxisType0> && within<tag<SubV...>, axes<AxisTypes...>>;
    }
};

// Within for a single axis axes and another single axis axes
template <typename SubAxis, typename FullAxis> struct is_within<axes<SubAxis>, axes<FullAxis>> {
    static consteval bool
    value() {
        return within<SubAxis, FullAxis>;
    }
};

// Within for a multi-axis axes and another multi-axis axes
template <typename SubAxis0,
          typename... SubAxisTypes,
          typename FullAxis0,
          typename... FullAxisTypes>
struct is_within<axes<SubAxis0, SubAxisTypes...>, axes<FullAxis0, FullAxisTypes...>> {
    static consteval bool
    value() {
        return within<SubAxis0, FullAxis0> && within<axes<SubAxisTypes...>, axes<FullAxisTypes...>>;
    }
};

//------------------------------------------------------------------------------
// at_index - trait for getting the element at an index
//------------------------------------------------------------------------------
template <size_t I, typename T> struct at_index {
    static_assert(axis_like<T>, "at_index requires a axis type");
};

template <size_t I, typename T>
static consteval axis_value_type_t<T>
at_index_v() {
    return at_index<I, T>::value();
}

template <size_t I, auto V> struct at_index<I, axis<V>> {
    static_assert(I == 0, "index out of bounds");
    static consteval axis_value_type_t<axis<V>>
    value() {
        return V;
    }
};

template <size_t I, auto V0, auto... Vs> struct at_index<I, axis<V0, Vs...>> {
    static_assert(I < (1 + sizeof...(Vs)), "index out of bounds");
    static consteval axis_value_type_t<axis<V0, Vs...>>
    value() {
        if constexpr (I == 0) {
            return V0;
        } else {
            return at_index_v<I - 1, Vs...>();
        }
    }
};

// -----------------------------------------------------------------------------
// Prepend value - trait for adding a new value to a value sequence
// -----------------------------------------------------------------------------

template <auto Value, typename Sequence> struct prepend_value;

template <size_t S, size_t... Ss> struct prepend_value<S, extents<Ss...>> {
    using type = extents<S, Ss...>;
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

//------------------------------------------------------------------------------
// at_indices = for getting a tag from an indices into axes
//------------------------------------------------------------------------------
template <typename Indices, typename Axes> struct at_indices {
    static_assert(indices_like<Indices>, "at_indices requires an indices type");
    static_assert(axes_like<Axes>, "at_indices requires an axes type");
    static_assert(equidimensional_with<Indices, Axes>,
                  "at_indices requires the indices and axes to be equidimensional");
    static_assert(within<Indices, Axes>, "at_indices requires the indices to be within the axes");
};

template <typename Indices, typename Axes>
using at_indices_t = typename at_indices<Indices, Axes>::type;

template <size_t I, typename AxisType> struct at_indices<indices<I>, axes<AxisType>> {
    static_assert(axis_like<AxisType>, "at_indices requires an axis type");
    using type = tag<at_index_v<I, AxisType>()>;
};

template <size_t I0, size_t... Is, typename AxisType0, typename... AxisTypes>
struct at_indices<indices<I0, Is...>, axes<AxisType0, AxisTypes...>> {
    using indices_type = indices<I0, Is...>;
    using axes_type    = axes<AxisType0, AxisTypes...>;
    static_assert(equidimensional_with<indices_type, axes_type>,
                  "at_indices requires the indices and axes to be equidimensional");
    static_assert(within<indices_type, axes_type>,
                  "at_indices requires the indices to be within the axes");

    using tail_type = at_indices_t<indices<Is...>, axes<AxisTypes...>>;
    using type      = prepend_value_t<at_index_v<I0, AxisType0>(), tail_type>;
};

//------------------------------------------------------------------------------
// index_of - trait for getting the index of a value in an axis
//------------------------------------------------------------------------------
template <auto Value, typename Axis> struct index_of {
    static_assert(axis_like<Axis>, "index_of requires an axis type");
    static_assert(within<axis<Value>, Axis>, "index_of requires the value to be within the axis");
};

template <auto Value, typename Axis>
consteval size_t
index_of_v() {
    return index_of<Value, Axis>::value();
};

template <auto Value, auto AxisValue> struct index_of<Value, axis<AxisValue>> {
    static_assert(within<axis<Value>, axis<AxisValue>>,
                  "index_of requires the value to be within the axis");
    static consteval size_t
    value() {
        return 0;
    }
};

template <auto Value, auto AxisValue0, auto... AxisValues>
struct index_of<Value, axis<AxisValue0, AxisValues...>> {
    static_assert(within<axis<Value>, axis<AxisValue0, AxisValues...>>,
                  "index_of requires the value to be within the axis");
    static consteval size_t
    value() {
        if constexpr (Value == AxisValue0) {
            return 0;
        } else {
            return 1 + index_of_v<Value, AxisValues...>();
        }
    }
};

//------------------------------------------------------------------------------
// indices of - trait for getting the indices of a tag in an axes
//------------------------------------------------------------------------------
template <typename Tag, typename Axes> struct indices_of {
    static_assert(tag_like<Tag>, "indices_of requires a tag type");
    static_assert(axes_like<Axes>, "indices_of requires an axes type");
    static_assert(within<Tag, Axes>, "indices_of requires the tag to be within the axes");
};

template <typename Tag, typename Axes> using indices_of_t = typename indices_of<Tag, Axes>::type;

template <auto Value, typename AxisType> struct indices_of<tag<Value>, axes<AxisType>> {
    using tag_type  = tag<Value>;
    using axes_type = axes<AxisType>;
    static_assert(within<tag_type, axes_type>, "indices_of requires the tag to be within the axes");

    using type = indices<index_of_v<Value, AxisType>()>;
};

template <auto Value0, auto... Values, typename AxisType0, typename... AxisTypes>
struct indices_of<tag<Value0, Values...>, axes<AxisType0, AxisTypes...>> {
    using tag_type  = tag<Value0, Values...>;
    using axes_type = axes<AxisType0, AxisTypes...>;
    static_assert(within<tag_type, axes_type>, "indices_of requires the tag to be within the axes");

    using tail_type = indices_of_t<tag<Values...>, axes<AxisTypes...>>;
    using type      = prepend_value_t<index_of_v<Value0, AxisType0>(), tail_type>;
};

//------------------------------------------------------------------------------
// linear index
//------------------------------------------------------------------------------
template <size_t I, typename Space struct is_linear_index_within : consteval_false_type {
    static_assert(extents_like<Space>, "is_linear_index_within requires an extents type");
    static_assert(non_empty<Space>, "is_linear_index_within requires a non-empty space");
};

template <size_t I, extents_like Space>
struct is_linear_index_within<I, Space> : consteval_bool_type<(I < volume_v<Space>())> {};

template <size_t I, typename Space>
consteval bool
is_linear_index_within_v() {
    return is_linear_index_within<I, Space>::value();
}

template <size_t I, typename Space>
concept linear_index_within = is_linear_index_within_v<I, Space>();

// -----------------------------------------------------------------------------
// indices_from_linear_index
// -----------------------------------------------------------------------------

template <typename Space, size_t linearIndex> struct indices_from_linear_index {
    static_assert(extents_like<Space>, "indices_from_linear_index requires an extents type");
    static_assert(
        linear_index_within<linearIndex, Space>,
        "space must be a non-empty index space and linear index must be less than the volume of the space");
};

template <typename Space, size_t linearIndex>
using indices_from_linear_index_t = typename indices_from_linear_index<Space, linearIndex>::type;

template <size_t S, size_t linearIndex> struct indices_from_linear_index<extents<S>, linearIndex> {
    static_assert(linearIndex < S, "linear index out of bounds");
    using type = indices<linearIndex>;
};

template <size_t S0, size_t... S, size_t linearIndex>
struct indices_from_linear_index<extents<S0, S...>, linearIndex> {
    static_assert(linear_index_within<linearIndex, extents<S0, S...>>,
                  "linear index out of bounds");
    static consteval size_t
    stride() {
        return volume_v<extents<S...>>();
    }

    using type = prepend_value_t<linearIndex / stride(),
                                 indices_from_linear_index_t<extents<S...>, linearIndex % stride()>>;
};

// -----------------------------------------------------------------------------
// linear_index_from_indices
// -----------------------------------------------------------------------------

template <typename Extents, typename Indices> struct linear_index_from_indices {
    static_assert(extents_like<Extents>, "linear_index_from_indices requires an extents type");
    static_assert(indices_like<Indices>, "linear_index_from_indices requires an indices type");
    static_assert(
        equidimensional_with<Extents, Indices>,
        "linear_index_from_indices requires the extents and indices to be equidimensional");
    static_assert(within<Indices, Extents>,
                  "linear_index_from_indices requires the indices to be within the extents");
};

template <typename Extents, typename Indices>
consteval size_t
linear_index_from_indices_v() {
    return linear_index_from_indices<Extents, Indices>::value();
}

template <size_t S, size_t I> struct linear_index_from_indices<extents<S>, indices<I>> {
    static_assert(within<indices<I>, extents<S>>, "indices out of bounds");
    static consteval size_t
    value() {
        return I;
    }
};

template <size_t S, size_t... Ss, size_t I, size_t... Is>
struct linear_index_from_indices<extents<S, Ss...>, indices<I, Is...>> {
    static_assert(within<indices<I, Is...>, extents<S, Ss...>>, "indices out of bounds");
    static consteval size_t
    value() {
        return I * volume_v<extents<Ss...>>() +
               linear_index_from_indices_v<extents<Ss...>, indices<Is...>>();
    }
};

//------------------------------------------------------------------------------
//

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

template <size_t... S> struct array_from<extents<S...>> {
    static constexpr std::array<size_t, sizeof...(S)> value = {S...};
};

template <size_t... I> struct array_from<indices<I...>> {
    static constexpr std::array<size_t, sizeof...(I)> value = {I...};
};

//------------------------------------------------------------------------------
// index-based trait/concept
//------------------------------------------------------------------------------
template <typename T> struct is_index_based : consteval_false_type {};

template <typename T>
consteval bool
is_index_based_v() {
    return is_index_based<T>::value();
}

template <typename T>
concept index_based = is_index_based_v<T>();

//------------------------------------------------------------------------------
// value-based trait/concept
//------------------------------------------------------------------------------
template <typename T> struct is_value_based : consteval_false_type {};

template <typename T>
consteval bool
is_value_based_v() {
    return is_value_based<T>::value();
}

template <typename T>
concept value_based = is_value_based_v<T>();

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
