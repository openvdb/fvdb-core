// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef DISPATCH_INDEX_SPACE_H
#define DISPATCH_INDEX_SPACE_H

#include "dispatch/traits.h"
#include "dispatch/types.h"

#include <array>
#include <cstddef>
#include <functional>
#include <utility>

namespace dispatch {

//------------------------------------------------------------------------------
// Point specializations and trait specializations
//------------------------------------------------------------------------------

template <size_t... I> struct is_index_based<indices<I...>> : consteval_true_type {};

template <size_t... I> struct is_point<indices<I...>> : consteval_true_type {};

template <size_t... I> struct ndim<indices<I...>> {
    static consteval size_t
    value() {
        return sizeof...(I);
    }
};

//------------------------------------------------------------------------------
// Extents specializations and trait specializations
//------------------------------------------------------------------------------

template <size_t... S> struct is_index_based<extents<S...>> : consteval_true_type {};

template <size_t... S> struct is_space<extents<S...>> : consteval_true_type {};

template <size_t... S> struct ndim<extents<S...>> {
    static consteval size_t
    value() {
        return sizeof...(S);
    }
};

template <> struct volume<extents<>> {
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

//------------------------------------------------------------------------------
// index_point concept
//------------------------------------------------------------------------------
template <typename T>
concept index_point = index_based<T> && point<T>;

// -----------------------------------------------------------------------------
// index_space concept
// -----------------------------------------------------------------------------

template <typename T>
concept index_space = index_based<T> && space<T>;

//------------------------------------------------------------------------------
// non_empty_index_space concept
//------------------------------------------------------------------------------
template <typename T>
concept non_empty_index_space = index_space<T> && non_empty<T>;

// -----------------------------------------------------------------------------
// index_point within index_space
// -----------------------------------------------------------------------------

template <size_t I, size_t S>
    requires non_empty<extents<S>>
struct is_within<indices<I>, extents<S>> {
    static consteval bool
    value() {
        return I < S;
    }
};

template <size_t I, size_t... Is, size_t S, size_t... Ss>
    requires non_empty<extents<S, Ss...>> &&
             equidimensional_with<indices<I, Is...>, extents<S, Ss...>>
struct is_within<indices<I, Is...>, extents<S, Ss...>> {
    static consteval bool
    value() {
        return I < S && is_within_v<indices<Is...>, extents<Ss...>>();
    }
};

//------------------------------------------------------------------------------
// index_space within index_space
//------------------------------------------------------------------------------
template <size_t Sub, size_t S>
    requires non_empty<extents<Sub>> && non_empty<extents<S>>
struct is_within<extents<Sub>, extents<S>> {
    static consteval bool
    value() {
        return Sub <= S;
    }
};

template <size_t Sub, size_t... Subs, size_t S, size_t... Ss>
    requires non_empty<extents<Sub, Subs...>> && non_empty<extents<S, Ss...>> &&
             equidimensional_with<extents<Sub, Subs...>, extents<S, Ss...>>
struct is_within<extents<Sub, Subs...>, extents<S, Ss...>> {
    static consteval bool
    value() {
        return Sub <= S && is_within_v<extents<Subs...>, extents<Ss...>>();
    }
};

//------------------------------------------------------------------------------
// index_point_within_space concept
//------------------------------------------------------------------------------
template <typename P, typename S>
concept index_point_within_space = index_point<P> && index_space<S> && within<P, S>;

//------------------------------------------------------------------------------
// index_space_within_space concept
//------------------------------------------------------------------------------
template <typename Sub, typename S>
concept index_space_within_space = index_space<Sub> && index_space<S> && within<Sub, S>;

// -----------------------------------------------------------------------------
// linearly_indexes concept
//------------------------------------------------------------------------------
template <size_t linearIndex, typename S>
concept linearly_indexes = non_empty_index_space<S> && (linearIndex < volume_v<S>());

// -----------------------------------------------------------------------------
// point_from_linear_index
// -----------------------------------------------------------------------------

template <typename Space, size_t linearIndex> struct point_from_linear_index {
    static_assert(
        linearly_indexes<linearIndex, Space>,
        "space must be a non-empty index space and linear index must be less than the volume of the space");
};

template <typename Space, size_t linearIndex>
using point_from_linear_index_t = typename point_from_linear_index<Space, linearIndex>::type;

template <size_t S, size_t linearIndex>
    requires linearly_indexes<linearIndex, extents<S>>
struct point_from_linear_index<extents<S>, linearIndex> {
    using type = indices<linearIndex>;
};

template <size_t S0, size_t... S, size_t linearIndex>
    requires linearly_indexes<linearIndex, extents<S0, S...>>
struct point_from_linear_index<extents<S0, S...>, linearIndex> {
    static consteval size_t
    stride() {
        return volume_v<extents<S...>>();
    }

    using type = prepend_value_t<linearIndex / stride(),
                                 point_from_linear_index_t<extents<S...>, linearIndex % stride()>>;
};

// -----------------------------------------------------------------------------
// linear_index_from_point
// -----------------------------------------------------------------------------

template <typename Space, typename Pt> struct linear_index_from_point {
    static_assert(index_point_within_space<Pt, Space>,
                  "Pt must be an index point within non-empty index space Space");
};

template <typename Space, typename Pt>
consteval size_t
linear_index_from_point_v() {
    return linear_index_from_point<Space, Pt>::value();
}

template <size_t S, size_t I>
    requires index_point_within_space<indices<I>, extents<S>>
struct linear_index_from_point<extents<S>, indices<I>> {
    static consteval size_t
    value() {
        return I;
    }
};

template <size_t S, size_t... Ss, size_t I, size_t... Is>
    requires index_point_within_space<indices<I, Is...>, extents<S, Ss...>>
struct linear_index_from_point<extents<S, Ss...>, indices<I, Is...>> {
    static consteval size_t
    value() {
        return I * volume_v<extents<Ss...>>() +
               linear_index_from_point_v<extents<Ss...>, indices<Is...>>();
    }
};

// -----------------------------------------------------------------------------
// Visitation Utilities for Index Spaces
// -----------------------------------------------------------------------------
//
// Index spaces are traversed in row-major order: the last dimension is
// the fastest-changing. You can visit all coordinates in an index space
// by providing a visitor (callable) to visit_index_space.
//
// Example:
//     using MySpace = Extents<2, 3>; // 2x3 space
//     visit_index_space([](auto coord) {
//         // coord is a Point<I0, I1> for each valid coordinate pair
//     }, MySpace{});
//
// You can also visit multiple spaces in sequence using visit_index_spaces.
// Each space is visited in order, calling the visitor for every coordinate
// in that space.
//
// Example:
//     visit_index_spaces(visitor, MySpace1{}, MySpace2{});
//
// Note: For each coordinate, the visitor is invoked with a default-constructed
// Point<...> object with constexpr values for that coordinate.
// -----------------------------------------------------------------------------

namespace detail {

template <typename Space, typename LinearPointSeq> struct index_space_visit_helper {
    static_assert(non_empty_index_space<Space>, "Space must be a non-empty index space");
    static_assert(index_sequence_like<LinearPointSeq>,
                  "LinearPointSeq must be an index sequence like");
};

template <typename Space, size_t... linearIndices>
    requires non_empty_index_space<Space> &&
             index_sequence_like<std::index_sequence<linearIndices...>>
struct index_space_visit_helper<Space, std::index_sequence<linearIndices...>> {
    template <typename Visitor>
    static void
    visit(Visitor visitor) {
        // Note: don't forward visitor in fold - it's invoked multiple times
        (std::invoke(visitor, point_from_linear_index_t<Space, linearIndices>{}), ...);
    }
};

} // namespace detail

template <typename Visitor, typename Space>
void
visit_index_space(Visitor visitor, Space) {
    static_assert(non_empty_index_space<Space>, "Space must be a non-empty index space");

    // Don't bother with forward - it's invoked multiple times
    detail::index_space_visit_helper<Space, std::make_index_sequence<volume_v<Space>()>>::visit(
        visitor);
}

template <typename Visitor, typename... Spaces>
void
visit_index_spaces(Visitor visitor, Spaces... spaces) {
    static_assert((non_empty_index_space<Spaces> && ...), "Spaces must be non-empty index spaces");

    // Don't bother with forward - it's invoked multiple times
    (visit_index_space(visitor, spaces), ...);
}

} // namespace dispatch

#endif // DISPATCH_INDEX_SPACE_H
