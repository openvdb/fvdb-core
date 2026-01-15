// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_DISPATCH_VALUESPACE_H
#define FVDB_DETAIL_DISPATCH_VALUESPACE_H

#include "fvdb/detail/dispatch/IndexSpace.h"
#include "fvdb/detail/dispatch/Traits.h"
#include "fvdb/detail/dispatch/Values.h"

#include <array>
#include <concepts>
#include <optional>
#include <tuple>
#include <type_traits>

namespace fvdb {
namespace dispatch {

// -----------------------------------------------------------------------------
// ValueAxis concept
// -----------------------------------------------------------------------------
// A ValueAxis is a ValuePack that is same type, unique, and non-empty.

template <typename T>
concept ValueAxis = SameTypeValuePack<T> && UniqueValuePack<T> && NonEmptyValuePack<T>;

template <typename T>
inline consteval bool
is_value_axis() {
    return ValueAxis<T>;
}

// -----------------------------------------------------------------------------
// ValueAxis Traits (delegate to Pack* utilities with ValueAxis constraints)
// -----------------------------------------------------------------------------

// AxisSize: number of values in the axis
template <ValueAxis Axis> using AxisSize = PackSize<Axis>;
template <ValueAxis Axis>
consteval size_t
AxisSize_v() {
    return AxisSize<Axis>::value();
}

// AxisValueType: the common type of all values in the axis
template <ValueAxis Axis> using AxisValueType   = PackValueType<Axis>;
template <ValueAxis Axis> using AxisValueType_t = typename AxisValueType<Axis>::type;

// AxisElement: get value at index I
template <ValueAxis Axis, size_t I> using AxisElement   = PackElement<Axis, I>;
template <ValueAxis Axis, size_t I> using AxisElement_t = typename AxisElement<Axis, I>::type;
template <ValueAxis Axis, size_t I>
consteval auto
AxisElement_v() {
    return AxisElement<Axis, I>::value();
}

// AxisContains: check if axis contains a value
template <ValueAxis Axis, auto V> using AxisContains = PackContains<Axis, V>;
template <ValueAxis Axis, auto V>
consteval bool
AxisContains_v() {
    return AxisContains<Axis, V>::value();
}

// AxisIndex: get definite index of value V (valid since axis is unique)
template <ValueAxis Axis, auto V> using AxisIndex = PackDefiniteIndex<Axis, V>;
template <ValueAxis Axis, auto V>
consteval size_t
AxisIndex_v() {
    return AxisIndex<Axis, V>::value();
}

// AxisSubsetOf: check if SubAxis values are all in SuperAxis
template <typename SubAxis, typename SuperAxis>
concept AxisSubsetOf =
    ValueAxis<SubAxis> && ValueAxis<SuperAxis> && PackIsSubset_v<SuperAxis, SubAxis>();

// -----------------------------------------------------------------------------
// ValueAxis Runtime Functions (constexpr, delegate to pack* functions)
// -----------------------------------------------------------------------------

template <ValueAxis Axis>
constexpr size_t
axisSize(Axis) {
    return AxisSize_v<Axis>();
}

template <ValueAxis Axis>
constexpr auto
axisElement(Axis axis, size_t index) {
    return packElement(axis, index);
}

template <ValueAxis Axis>
constexpr bool
axisContains(Axis axis, auto value) {
    return packContains(axis, value);
}

template <ValueAxis Axis>
constexpr std::optional<size_t>
axisIndex(Axis axis, auto value) {
    return packIndex(axis, value);
}

template <ValueAxis Axis>
constexpr size_t
axisDefiniteIndex(Axis axis, auto value) {
    return packDefiniteIndex(axis, value);
}

//------------------------------------------------------------------------------
// ValueSpace concept
//------------------------------------------------------------------------------
// A Value Space is an ordered set of ValueAxes. It must have at least one
// axis.

// Value Pack Concept (The Category)
template <typename T>
concept ValueSpace = requires { typename T::value_space_tag; };

// ValueAxes Struct (The Concrete Type)
template <ValueAxis... Axes>
    requires(sizeof...(Axes) > 0)
struct ValueAxes {
    using value_space_tag = void;
};

template <typename T>
consteval bool
is_value_space() {
    return ValueSpace<T>;
}

template <ValueAxis... axes> struct Rank<ValueAxes<axes...>> {
    static consteval size_t
    value() {
        return sizeof...(axes);
    }
};

template <ValueAxis... axes> struct Numel<ValueAxes<axes...>> {
    static consteval size_t
    value() {
        return (AxisSize_v<axes>() * ... * 1);
    }
};

// -----------------------------------------------------------------------------
// CoordTypesMatch - checks if a ValuePack has the correct types for a ValueSpace
// -----------------------------------------------------------------------------
// This is useful for runtime queries where the coord may not be in the space
// but has the right types (e.g., testing <kPrivateUse1, kBool, 17> against
// a space that doesn't contain those values but has matching axis types).

template <typename Space, typename Coord> struct coord_types_match {
    static consteval bool
    value() {
        return false;
    }
};

// Base case: single axis, single value
template <ValueAxis Axis0, auto V0>
struct coord_types_match<ValueAxes<Axis0>, Values<V0>> {
    static consteval bool
    value() {
        return std::is_same_v<AxisValueType_t<Axis0>, decltype(V0)>;
    }
};

// Recursive case: multiple axes
template <ValueAxis Axis0, ValueAxis Axis1, ValueAxis... TailAxes, auto V0, auto V1, auto... TailVs>
struct coord_types_match<ValueAxes<Axis0, Axis1, TailAxes...>, Values<V0, V1, TailVs...>> {
    static consteval bool
    value() {
        return std::is_same_v<AxisValueType_t<Axis0>, decltype(V0)> &&
               coord_types_match<ValueAxes<Axis1, TailAxes...>, Values<V1, TailVs...>>::value();
    }
};

template <typename Space, typename Coord>
consteval bool
coord_types_match_v() {
    return coord_types_match<Space, Coord>::value();
}

template <typename Space, typename Coord>
concept CoordTypesMatch =
    ValueSpace<Space> && ValuePack<Coord> && coord_types_match_v<Space, Coord>();

// -----------------------------------------------------------------------------
// SpaceContains - checks if a ValuePack is contained within a ValueSpace
// -----------------------------------------------------------------------------
// Each value in the coord must exist in the corresponding axis.

template <typename Space, typename Coord> struct space_contains {
    static consteval bool
    value() {
        return false;
    }
};

// Base case: single axis, single value
template <ValueAxis Axis0, auto V0>
struct space_contains<ValueAxes<Axis0>, Values<V0>> {
    static consteval bool
    value() {
        return AxisContains_v<Axis0, V0>();
    }
};

// Recursive case: multiple axes
template <ValueAxis Axis0, ValueAxis Axis1, ValueAxis... TailAxes, auto V0, auto V1, auto... TailVs>
struct space_contains<ValueAxes<Axis0, Axis1, TailAxes...>, Values<V0, V1, TailVs...>> {
    static consteval bool
    value() {
        return AxisContains_v<Axis0, V0>() &&
               space_contains<ValueAxes<Axis1, TailAxes...>, Values<V1, TailVs...>>::value();
    }
};

template <typename Space, typename Coord>
consteval bool
space_contains_v() {
    return space_contains<Space, Coord>::value();
}

template <typename Space, typename Coord>
concept SpaceContains = CoordTypesMatch<Space, Coord> && space_contains_v<Space, Coord>();

// -----------------------------------------------------------------------------
// IndexSpaceOf - the IndexSpace associated with a ValueSpace
// -----------------------------------------------------------------------------

template <typename Space> struct IndexSpaceOf;

template <ValueAxis... Axes> struct IndexSpaceOf<ValueAxes<Axes...>> {
    using type = Sizes<AxisSize_v<Axes>()...>;
};

template <ValueSpace Space> using IndexSpaceOf_t = typename IndexSpaceOf<Space>::type;

// -----------------------------------------------------------------------------
// CoordFromPoint - convert an IndexPoint to a ValueCoord within a ValueSpace
// -----------------------------------------------------------------------------

template <ValueSpace Space, IndexPoint Pt> struct CoordFromPoint;

template <ValueAxis Axis, size_t I>
    requires PointInBounds<Sizes<AxisSize_v<Axis>()>, Point<I>>
struct CoordFromPoint<ValueAxes<Axis>, Point<I>> {
    using type = Values<AxisElement_v<Axis, I>()>;
};

template <ValueAxis Axis0, ValueAxis... TailAxes, size_t I0, size_t... TailIs>
    requires PointInBounds<IndexSpaceOf_t<ValueAxes<Axis0, TailAxes...>>, Point<I0, TailIs...>>
struct CoordFromPoint<ValueAxes<Axis0, TailAxes...>, Point<I0, TailIs...>> {
    using type =
        PackPrepend_t<typename CoordFromPoint<ValueAxes<TailAxes...>, Point<TailIs...>>::type,
                      AxisElement_v<Axis0, I0>()>;
};

template <ValueSpace Space, IndexPoint Pt>
    requires PointInBounds<IndexSpaceOf_t<Space>, Pt>
using CoordFromPoint_t = typename CoordFromPoint<Space, Pt>::type;

} // namespace dispatch
} // namespace fvdb

#endif // FVDB_DETAIL_DISPATCH_VALUESPACE_H
