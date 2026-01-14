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

template <typename... Axes> struct AxisOuterProductHelper;

template <> struct AxisOuterProductHelper<> {
    static constexpr size_t rank  = 0;
    static constexpr size_t numel = 0;

    template <typename T> struct CoordFromIndicesTypeHelper {
        using type = std::index_sequence<>;
    };
};

template <typename Axis> struct AxisOuterProductHelper<Axis> {
    static_assert(is_dimensional_axis_v<Axis>,
                  "AxisOuterProduct requires the Axis to be a DimensionalAxis type ");
    static_assert(Axis::size > 0,
                  "AxisOuterProduct does not support empty axis (size must be > 0)");

    static constexpr size_t rank  = 1;
    static constexpr size_t numel = Axis::size;

    template <typename T> struct CoordFromIndicesTypeHelper;

    template <size_t ValueIndex>
    struct CoordFromIndicesTypeHelper<std::index_sequence<ValueIndex>> {
        using type = AnyTypeValuePack<Axis::template value_at<ValueIndex>>;
    };
};

template <typename Axis0, typename... TailAxes> struct AxisOuterProductHelper<Axis0, TailAxes...> {
    static_assert(is_dimensional_axis_v<Axis0>,
                  "AxisOuterProduct requires the Axis0 to be a DimensionalAxis type ");
    static_assert(Axis0::size > 0,
                  "AxisOuterProduct does not support empty axis (size must be > 0)");

    using _tail_helper_type = AxisOuterProductHelper<TailAxes...>;

    static constexpr size_t rank  = 1 + _tail_helper_type::rank;
    static constexpr size_t numel = Axis0::size * _tail_helper_type::numel;

    template <typename T> struct CoordFromIndicesTypeHelper;

    template <size_t ValueIndex0, size_t... TailValueIndices>
    struct CoordFromIndicesTypeHelper<std::index_sequence<ValueIndex0, TailValueIndices...>> {
        using _tail_indices_type = std::index_sequence<TailValueIndices...>;
        using _tail_coord_type   = typename _tail_helper_type::template CoordFromIndicesTypeHelper<
              _tail_indices_type>::type;
        using type = typename PrependAnyTypeValue<Axis0::template value_at<ValueIndex0>,
                                                  _tail_coord_type>::type;
    };
};

template <typename... Axes> struct AxisOuterProduct {
    static_assert((is_dimensional_axis_v<Axes> && ...),
                  "AxisOuterProduct requires all Axes to be DimensionalAxis types "
                  "(i.e., SameTypeUniqueValuePack instantiations)");
    static_assert(((Axes::size > 0) && ... && true),
                  "AxisOuterProduct does not support empty axes (size must be > 0)");

    using index_space_type = IndexSpace<Axes::size...>;

    static constexpr size_t rank  = index_space_type::rank;
    static constexpr size_t numel = index_space_type::numel;

    template <typename T>
    using coord_from_indices_type =
        typename AxisOuterProductHelper<Axes...>::template CoordFromIndicesTypeHelper<T>::type;

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

template <typename T> struct CoordFromIndices;

template <typename Axis> struct CoordFromIndices<AxisOuterProduct<Axis>> {
    template <typename... Axes> struct AxisOuterProductHelper;

    template <> struct AxisOuterProductHelper<> {
        static constexpr size_t rank  = 0;
        static constexpr size_t numel = 0;

        template <typename T> struct CoordFromIndicesTypeHelper {
            using type = std::index_sequence<>;
        };
    };

    template <typename Axis> struct AxisOuterProductHelper<Axis> {
        static_assert(is_dimensional_axis_v<Axis>,
                      "AxisOuterProduct requires the Axis to be a DimensionalAxis type ");
        static_assert(Axis::size > 0,
                      "AxisOuterProduct does not support empty axis (size must be > 0)");

        static constexpr size_t rank  = 1;
        static constexpr size_t numel = Axis::size;

        template <typename T> struct CoordFromIndicesTypeHelper;

        template <size_t ValueIndex>
        struct CoordFromIndicesTypeHelper<std::index_sequence<ValueIndex>> {
            using type = AnyTypeValuePack<Axis::template value_at<ValueIndex>>;
        };
    };

    template <typename Axis0, typename... TailAxes>
    struct AxisOuterProductHelper<Axis0, TailAxes...> {
        static_assert(is_dimensional_axis_v<Axis0>,
                      "AxisOuterProduct requires the Axis0 to be a DimensionalAxis type ");
        static_assert(Axis0::size > 0,
                      "AxisOuterProduct does not support empty axis (size must be > 0)");

        using _tail_helper_type = AxisOuterProductHelper<TailAxes...>;

        static constexpr size_t rank  = 1 + _tail_helper_type::rank;
        static constexpr size_t numel = Axis0::size * _tail_helper_type::numel;

        template <typename T> struct CoordFromIndicesTypeHelper;

        template <size_t ValueIndex0, size_t... TailValueIndices>
        struct CoordFromIndicesTypeHelper<std::index_sequence<ValueIndex0, TailValueIndices...>> {
            using _tail_indices_type = std::index_sequence<TailValueIndices...>;
            using _tail_coord_type =
                typename _tail_helper_type::template CoordFromIndicesTypeHelper<
                    _tail_indices_type>::type;
            using type = typename PrependAnyTypeValue<Axis0::template value_at<ValueIndex0>,
                                                      _tail_coord_type>::type;
        };
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
