// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_DISPATCH_AXISOUTERPRODUCT_H
#define FVDB_DETAIL_DISPATCH_AXISOUTERPRODUCT_H

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
// is_dimensional_axis trait
// -----------------------------------------------------------------------------
// A DimensionalAxis is a SameTypeUniqueValuePack - a set of unique values of
// the same type that can be indexed bidirectionally.

template <typename T> struct is_dimensional_axis : std::false_type {};

template <typename T> inline constexpr bool is_dimensional_axis_v = is_dimensional_axis<T>::value;

template <auto... values> struct DimensionalAxis : SameTypeUniqueValuePack<values...> {
    static_assert(sizeof...(values) > 0, "DimensionalAxis must have a least one value");
};

template <auto... values>
struct is_dimensional_axis<DimensionalAxis<values...>> : std::true_type {};

// Specialization of is_subset_of for DimensionalAxis (delegates to SameTypeUniqueValuePack)
template <auto... SubsetValues, auto... Values>
struct is_subset_of<DimensionalAxis<SubsetValues...>, DimensionalAxis<Values...>>
    : is_subset_of<SameTypeUniqueValuePack<SubsetValues...>, SameTypeUniqueValuePack<Values...>> {};

// -----------------------------------------------------------------------------
// AxisOuterProduct
// -----------------------------------------------------------------------------

template <typename... Axes> struct AxisOuterProduct {
    static_assert(sizeof...(Axes) > 0, "AxisOuterProduct must have at least one axis");
    static_assert((is_dimensional_axis_v<Axes> && ... && true),
                  "All axes must be dimensional axes");

    using index_space_type = IndexSpace<PackSize<Axes>::value()...>;

    static consteval size_t
    rank() {
        return sizeof...(Axes);
    }
    static consteval size_t
    numel() {
        return (PackSize<Axes>::value() * ... * 1);
    }
    using shape = std::index_sequence<PackSize<Axes>::value()...>;
};

template <typename ValueSpace, typename Index> struct CoordFromIndices;

template <typename ValueSpace, typename Index>
using CoordFromIndices_t = typename CoordFromIndices<ValueSpace, Index>::type;

template <> struct CoordFromIndices<AxisOuterProduct<>, std::index_sequence<>> {
    using type = AnyTypeValuePack<>;
};

template <typename Axis, size_t valueIndex>
struct CoordFromIndices<AxisOuterProduct<Axis>, std::index_sequence<valueIndex>> {
    static_assert(valueIndex < PackSize<Axis>::value(), "Index out of bounds");
    using type = AnyTypeValuePack<PackElement_t<Axis, valueIndex>>;
};

template <typename Axis0, typename... TailAxes, size_t valueIndex0, size_t... tailValueIndices>
struct CoordFromIndices<AxisOuterProduct<Axis0, TailAxes...>,
                        std::index_sequence<valueIndex0, tailValueIndices...>> {
    static_assert(sizeof...(tailValueIndices) == sizeof...(TailAxes),
                  "Number of value indices must match number of axes");
    static_assert(valueIndex0 < PackSize<Axis0>::value(), "Index out of bounds");
    using type = PackPrepend_t<
        CoordFromIndices_t<AxisOuterProduct<TailAxes...>, std::index_sequence<tailValueIndices...>>,
        PackElement_t<Axis0, valueIndex0>>;
};

template <typename ValueSpace, typename Coord> struct IndicesFromCoord;

template <typename ValueSpace, typename Coord>
using IndicesFromCoord_t = typename IndicesFromCoord<ValueSpace, Coord>::type;

template <> struct IndicesFromCoord<AxisOuterProduct<>, AnyTypeValuePack<>> {
    using type = std::index_sequence<>;
};

template <typename Axis, auto coordValue>
struct IndicesFromCoord<AxisOuterProduct<Axis>, AnyTypeValuePack<coordValue>> {
    static_assert(PackContains<Axis, coordValue>::value(), "Coordinate value not found in axis");
    using type = std::index_sequence<PackDefiniteFirstIndex<Axis, coordValue>::value()>;
};

template <typename Axis0, typename... TailAxes, auto coordValue0, auto... tailCoordValues>
struct IndicesFromCoord<AxisOuterProduct<Axis0, TailAxes...>,
                        AnyTypeValuePack<coordValue0, tailCoordValues...>> {
    static_assert(sizeof...(tailCoordValues) == sizeof...(TailAxes),
                  "Number of coordinate values must match number of axes");
    static_assert(PackContains<Axis0, coordValue0>::value(), "Coordinate value not found in axis");
    using type = IndexSequencePrepend_t<
        IndicesFromCoord_t<AxisOuterProduct<TailAxes...>, AnyTypeValuePack<tailCoordValues...>>,
        PackDefiniteFirstIndex<Axis0, coordValue0>::value()>;
};

} // namespace dispatch
} // namespace fvdb

#endif // FVDB_DETAIL_DISPATCH_AXISOUTERPRODUCT_H
