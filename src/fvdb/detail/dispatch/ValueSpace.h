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

// ValueSpace Concept (The Category)
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
template <ValueAxis Axis0, auto V0> struct coord_types_match<ValueAxes<Axis0>, Values<V0>> {
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

template <ValueSpace Space, ValuePack Coord>
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
template <ValueAxis Axis0, auto V0> struct space_contains<ValueAxes<Axis0>, Values<V0>> {
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

template <ValueSpace Space, ValuePack Coord>
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

// -----------------------------------------------------------------------------
// CoordFromLinearIndex - convert a linear index to a ValueCoord within a ValueSpace
// -----------------------------------------------------------------------------
// This chains CoordFromPoint <- PointFromLinearIndex around the same space.

template <ValueSpace Space, size_t linearIndex>
    requires(linearIndex < Numel_v<Space>())
using CoordFromLinearIndex_t =
    CoordFromPoint_t<Space, PointFromLinearIndex_t<IndexSpaceOf_t<Space>, linearIndex>>;

// -----------------------------------------------------------------------------
// PointFromCoord - convert a ValueCoord to an IndexPoint within a ValueSpace
// -----------------------------------------------------------------------------

template <ValueSpace Space, ValuePack Coord> struct PointFromCoord;

// Base case: single axis, single value
template <ValueAxis Axis, auto V>
    requires SpaceContains<ValueAxes<Axis>, Values<V>>
struct PointFromCoord<ValueAxes<Axis>, Values<V>> {
    using type = Point<AxisIndex_v<Axis, V>()>;
};

// Recursive case: multiple axes
template <ValueAxis Axis0, ValueAxis... TailAxes, auto V0, auto... TailVs>
    requires SpaceContains<ValueAxes<Axis0, TailAxes...>, Values<V0, TailVs...>>
struct PointFromCoord<ValueAxes<Axis0, TailAxes...>, Values<V0, TailVs...>> {
    using type = Prepend_t<typename PointFromCoord<ValueAxes<TailAxes...>, Values<TailVs...>>::type,
                           AxisIndex_v<Axis0, V0>()>;
};

template <ValueSpace Space, ValuePack Coord>
    requires SpaceContains<Space, Coord>
using PointFromCoord_t = typename PointFromCoord<Space, Coord>::type;

// -----------------------------------------------------------------------------
// LinearIndexFromCoord - convert a ValueCoord to a linear index within a ValueSpace
// -----------------------------------------------------------------------------
// This chains LinearIndexFromPoint <- PointFromCoord around the same space.

template <ValueSpace Space, ValuePack Coord>
    requires SpaceContains<Space, Coord>
consteval size_t
LinearIndexFromCoord_v() {
    return LinearIndexFromPoint_v<IndexSpaceOf_t<Space>, PointFromCoord_t<Space, Coord>>();
}

// =============================================================================
// ValueCoord <-> Tuple Conversion Utilities
// =============================================================================
//
// These utilities provide type-safe conversions between Values<...> coords
// and std::tuple<...> for runtime operations on value spaces.
//
// Key types:
//   - SpaceTupleType_t<Space>: the std::tuple type for coords in a space
//   - TupleTypesMatchSpace<Tuple, Space>: concept checking tuple/space compatibility
//
// Key functions:
//   - coordToTuple(coord): convert Values<...> to std::tuple (constexpr)
//   - spaceLinearIndex(space, tuple): runtime lookup returning std::optional<size_t>
//
// =============================================================================

// -----------------------------------------------------------------------------
// SpaceTupleType - the std::tuple type for coordinates in a ValueSpace
// -----------------------------------------------------------------------------

template <typename Space> struct SpaceTupleType;

template <ValueAxis... Axes> struct SpaceTupleType<ValueAxes<Axes...>> {
    using type = std::tuple<AxisValueType_t<Axes>...>;
};

template <ValueSpace Space> using SpaceTupleType_t = typename SpaceTupleType<Space>::type;

// -----------------------------------------------------------------------------
// TupleTypesMatchSpace - check if a tuple type matches a space's coordinate type
// -----------------------------------------------------------------------------

template <typename Tuple, typename Space> struct tuple_types_match_space {
    static consteval bool
    value() {
        return false;
    }
};

template <ValueAxis... Axes>
struct tuple_types_match_space<std::tuple<AxisValueType_t<Axes>...>, ValueAxes<Axes...>> {
    static consteval bool
    value() {
        return true;
    }
};

template <typename Tuple, typename Space>
consteval bool
tuple_types_match_space_v() {
    return tuple_types_match_space<std::decay_t<Tuple>, Space>::value();
}

template <typename Tuple, typename Space>
concept TupleTypesMatchSpace = ValueSpace<Space> && tuple_types_match_space_v<Tuple, Space>();

template <typename Tuple, typename Space>
consteval bool
is_tuple_types_match_space() {
    return TupleTypesMatchSpace<Tuple, Space>;
}

// -----------------------------------------------------------------------------
// coordToTuple - convert a Values coord to a std::tuple
// -----------------------------------------------------------------------------
// This is constexpr since the coord values are known at compile time.

template <auto... Vs>
constexpr auto
coordToTuple(Values<Vs...>) {
    return std::make_tuple(Vs...);
}

// Overload with explicit Space for type clarity
template <ValueSpace Space, typename Coord>
    requires CoordTypesMatch<Space, Coord>
constexpr SpaceTupleType_t<Space>
coordToTuple(Space, Coord) {
    return coordToTuple(Coord{});
}

// -----------------------------------------------------------------------------
// spaceLinearIndex - runtime linear index lookup from tuple coordinate
// -----------------------------------------------------------------------------
// Returns std::optional<size_t>: the linear index if coord is in space, nullopt otherwise.
// This is the runtime analog of LinearIndexFromCoord_v for queries where
// the coordinate values aren't known at compile time.
//
// The tuple must have the correct types (TupleTypesMatchSpace), but the values
// may or may not be in the space's axes.

// Base case: single axis
template <ValueAxis Axis, typename T>
    requires std::is_same_v<std::decay_t<T>, AxisValueType_t<Axis>>
constexpr std::optional<size_t>
spaceLinearIndex(ValueAxes<Axis>, std::tuple<T> const &coord) {
    return axisIndex(Axis{}, std::get<0>(coord));
}

// Recursive case: multiple axes
template <ValueAxis Axis0,
          ValueAxis Axis1,
          ValueAxis... TailAxes,
          typename T0,
          typename T1,
          typename... TailTs>
    requires TupleTypesMatchSpace<std::tuple<T0, T1, TailTs...>,
                                  ValueAxes<Axis0, Axis1, TailAxes...>>
constexpr std::optional<size_t>
spaceLinearIndex(ValueAxes<Axis0, Axis1, TailAxes...>, std::tuple<T0, T1, TailTs...> const &coord) {
    // Look up index in first axis
    auto idx0 = axisIndex(Axis0{}, std::get<0>(coord));
    if (!idx0) {
        return std::nullopt;
    }

    // Extract tail tuple and recurse
    auto tailCoord = tuple_tail(coord);

    auto tailLinear = spaceLinearIndex(ValueAxes<Axis1, TailAxes...>{}, tailCoord);
    if (!tailLinear) {
        return std::nullopt;
    }

    // Combine: linear = idx0 * tailNumel + tailLinear (row-major order)
    constexpr size_t tailNumel = Numel_v<ValueAxes<Axis1, TailAxes...>>();
    return *idx0 * tailNumel + *tailLinear;
}

// -----------------------------------------------------------------------------
// Visitation Utilities for Value Spaces
// -----------------------------------------------------------------------------
//
// Value spaces are traversed in row-major order (matching the underlying index
// space). The visitor receives a default-constructed Values<...> coord object
// for each coordinate in the space.
//
// Example:
//     using MySpace = ValueAxes<DeviceAxis, DTypeAxis>;
//     visit_value_space([](auto coord) {
//         // coord is a Values<Device, DType> for each valid combination
//     }, MySpace{});
//
// -----------------------------------------------------------------------------

// PointToCoordVisitor: Adapts a coord visitor to work with index points.
// Named struct for cleaner stack traces and explicit instantiation.
template <ValueSpace Space, typename CoordVisitor> struct PointToCoordVisitor {
    CoordVisitor &visitor;

    template <IndexPoint Pt>
    void
    operator()(Pt) const {
        using Coord = CoordFromPoint_t<Space, Pt>;
        std::invoke(visitor, Coord{});
    }
};

template <typename Visitor, ValueSpace Space>
void
visit_value_space(Visitor &&visitor, Space) {
    using IdxSpace = IndexSpaceOf_t<Space>;
    PointToCoordVisitor<Space, std::remove_reference_t<Visitor>> wrapper{visitor};
    visit_index_space(wrapper, IdxSpace{});
}

template <typename Visitor, ValueSpace... Spaces>
void
visit_value_spaces(Visitor &&visitor, Spaces... spaces) {
    // Note: don't forward visitor in fold - it's used for each space
    (visit_value_space(visitor, spaces), ...);
}

} // namespace dispatch
} // namespace fvdb

#endif // FVDB_DETAIL_DISPATCH_VALUESPACE_H
