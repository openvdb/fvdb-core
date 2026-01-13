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
// Forms the cartesian product of multiple DimensionalAxis types, providing
// linear indexing into the combined space from runtime values.
//
// Given axes A, B, C with sizes |A|, |B|, |C|, the linear index for values
// (a, b, c) is computed as: a_idx * (|B| * |C|) + b_idx * |C| + c_idx
// (row-major order: last axis varies fastest)
//
// Returns std::nullopt if any value is not found in its corresponding axis.

template <typename... Axes> struct AxisOuterProduct {
    static_assert((is_dimensional_axis_v<Axes> && ...),
                  "AxisOuterProduct requires all Axes to be DimensionalAxis types "
                  "(i.e., SameTypeUniqueValuePack instantiations)");
    static_assert(((Axes::size > 0) && ... && true),
                  "AxisOuterProduct does not support empty axes (size must be > 0)");

    using index_space_type = IndexSpace<Axes::size...>;

    static constexpr size_t rank  = index_space_type::rank;
    static constexpr size_t numel = index_space_type::numel;

    using axes_tuple_type = std::tuple<Axes...>;

    template <size_t AxisIndex> struct axis_at_index {
        using type                   = std::tuple_element_t<AxisIndex, axes_tuple_type>;
        static constexpr size_t size = type::size;
    };

    template <typename T> struct CoordFromIndicesTypeHelper;

    template <size_t... ValueIndicesPerAxis>
    struct CoordFromIndicesTypeHelper<std::index_sequence<ValueIndicesPerAxis...>> {
        static_assert(sizeof...(ValueIndicesPerAxis) == rank,
                      "Number of indices must match the number of axes (rank)");

        template <size_t... AxisIndices>
        static auto _make_type(std::index_sequence<AxisIndices...>)
            -> AnyTypeValuePack<axis_at_index<AxisIndices>::type::template value_at<
                std::get<AxisIndices>(std::array<size_t, sizeof...(ValueIndicesPerAxis)>{
                    ValueIndicesPerAxis...})>...>;

        using type = decltype(_make_type(std::make_index_sequence<rank>{}));
    };

    // Convenience alias: index_sequence -> AnyTypeValuePack
    template <typename T>
    using coord_from_indices_type = typename CoordFromIndicesTypeHelper<T>::type;

    // -------------------------------------------------------------------------
    // Reverse conversion: AnyTypeValuePack -> index_sequence
    // -------------------------------------------------------------------------
    // Given coordinate values, produce the index_sequence that would map back
    // to those values. Compile-time errors if:
    //   - Wrong number of values (must equal rank)
    //   - Value type doesn't match axis value_type
    //   - Value not found in corresponding axis

    template <typename T> struct IndicesFromCoordTypeHelper;

    template <auto... CoordValues>
    struct IndicesFromCoordTypeHelper<AnyTypeValuePack<CoordValues...>> {
        static_assert(sizeof...(CoordValues) == rank,
                      "Number of coordinate values must match the number of axes (rank)");

        static constexpr auto coord_tuple = std::make_tuple(CoordValues...);

        template <size_t AxisIdx>
        static constexpr size_t
        _index_for_axis() {
            constexpr auto value = std::get<AxisIdx>(coord_tuple);
            using AxisType       = typename axis_at_index<AxisIdx>::type;
            using ValueType      = std::decay_t<decltype(value)>;
            using AxisValueType  = typename AxisType::value_type;

            static_assert(std::is_same_v<ValueType, AxisValueType>,
                          "Coordinate value type does not match axis value type");

            constexpr auto maybe_idx = AxisType::index_of_value(value);
            static_assert(maybe_idx.has_value(),
                          "Coordinate value not found in corresponding axis");
            return *maybe_idx;
        }

        template <size_t... AxisIndices>
        static auto _make_type(std::index_sequence<AxisIndices...>)
            -> std::index_sequence<_index_for_axis<AxisIndices>()...>;

        using type = decltype(_make_type(std::make_index_sequence<rank>{}));
    };

    // Convenience alias: AnyTypeValuePack -> index_sequence
    template <typename T>
    using indices_from_coord_type = typename IndicesFromCoordTypeHelper<T>::type;

    // -------------------------------------------------------------------------
    // Visitor pattern
    // -------------------------------------------------------------------------
    // Visits every point in the value space, calling the visitor with an
    // AnyTypeValuePack tag type for each coordinate. This is analogous to
    // IndexSpace::visit but operates in value space instead of index space.

    template <typename ValueVisitor>
    static void
    visit(ValueVisitor &&visitor) {
        index_space_type::visit([&visitor](auto index_coord) {
            using index_coord_t = decltype(index_coord);
            using value_coord_t = coord_from_indices_type<index_coord_t>;
            std::invoke(std::forward<ValueVisitor>(visitor), value_coord_t{});
        });
    }
};

// -----------------------------------------------------------------------------
// is_axis_outer_product trait
// -----------------------------------------------------------------------------
// Checks if a type is an AxisOuterProduct instantiation.

template <typename T> struct is_axis_outer_product : std::false_type {};

template <typename... Axes>
struct is_axis_outer_product<AxisOuterProduct<Axes...>> : std::true_type {};

template <typename T>
inline constexpr bool is_axis_outer_product_v = is_axis_outer_product<T>::value;

template <typename T>
concept AxisOuterProductConcept = is_axis_outer_product_v<T>;

// -----------------------------------------------------------------------------
// is_subspace_of trait
// -----------------------------------------------------------------------------
// Checks if one AxisOuterProduct is a subspace of another. A subspace has the
// same number of axes, and each axis is a subset of the corresponding axis in
// the full space.

template <typename SubSpace, typename FullSpace> struct is_subspace_of : std::false_type {};

// Specialization for same-length axis packs only (uses requires to prevent
// instantiation of fold expression when pack lengths differ)
template <typename... SubAxes, typename... FullAxes>
    requires(sizeof...(SubAxes) == sizeof...(FullAxes))
struct is_subspace_of<AxisOuterProduct<SubAxes...>, AxisOuterProduct<FullAxes...>>
    : std::bool_constant<(is_subset_of_v<SubAxes, FullAxes> && ...)> {};

template <typename Sub, typename Full>
inline constexpr bool is_subspace_of_v = is_subspace_of<Sub, Full>::value;

} // namespace dispatch
} // namespace fvdb

#endif // FVDB_DETAIL_DISPATCH_AXISOUTERPRODUCT_H
