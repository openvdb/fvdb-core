// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_DISPATCH_PERMUTATIONMAP_H
#define FVDB_DETAIL_DISPATCH_PERMUTATIONMAP_H

#include "fvdb/detail/dispatch/AxisOuterProduct.h"

#include <array>
#include <concepts>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace fvdb {
namespace dispatch {

// =============================================================================
// MAP TYPES
// =============================================================================

// -----------------------------------------------------------------------------
// PermutationArrayMap
// -----------------------------------------------------------------------------
// A fixed-size array-backed map from axis value tuples to T.
// Efficient for small axis products where the full array is acceptable.

template <typename AxesT, typename T, T emptyValue = T(0)> struct PermutationArrayMap {
    static_assert(is_axis_outer_product_v<AxesT>,
                  "PermutationArrayMap AxesT must be an AxisOuterProduct");

    using axes_type        = AxesT;
    using value_type       = T;
    using index_tuple_type = typename axes_type::value_types_tuple_type;

    static constexpr T empty_value = emptyValue;

    std::array<value_type, axes_type::size> storage_;

    constexpr PermutationArrayMap() { storage_.fill(empty_value); }

    // -------------------------------------------------------------------------
    // Linear index accessors (used by generators)
    // -------------------------------------------------------------------------

    constexpr T
    get(size_t linear_idx) const {
        return storage_[linear_idx];
    }

    constexpr void
    set(size_t linear_idx, value_type val) {
        storage_[linear_idx] = val;
    }

    // -------------------------------------------------------------------------
    // Tuple-based accessors (user-facing API)
    // -------------------------------------------------------------------------

    constexpr T
    get(index_tuple_type index) const {
        auto idx = std::apply(
            [](auto... values) { return axes_type::index_of_values(values...); }, index);
        if (!idx.has_value()) {
            return empty_value;
        }
        return storage_[*idx];
    }

    constexpr void
    set(index_tuple_type index, value_type val) {
        auto idx = std::apply(
            [](auto... values) { return axes_type::index_of_values(values...); }, index);
        if (idx.has_value()) {
            storage_[*idx] = val;
        }
    }

    // Convenience operator[] for tuple-based access
    constexpr T
    operator[](index_tuple_type index) const {
        return get(index);
    }
};

// -----------------------------------------------------------------------------
// PermutationUnorderedMap
// -----------------------------------------------------------------------------
// A hash-map-backed map from axis value tuples to T.
// Efficient for large axis products where a sparse representation is preferred.

template <typename AxesT, typename T, T emptyValue = T(0)> struct PermutationUnorderedMap {
    static_assert(is_axis_outer_product_v<AxesT>,
                  "PermutationUnorderedMap AxesT must be an AxisOuterProduct");

    using axes_type        = AxesT;
    using value_type       = T;
    using index_tuple_type = typename axes_type::value_types_tuple_type;

    static constexpr T empty_value = emptyValue;

    std::unordered_map<size_t, value_type> storage_;

    PermutationUnorderedMap() = default;

    // -------------------------------------------------------------------------
    // Linear index accessors (used by generators)
    // -------------------------------------------------------------------------

    T
    get(size_t linear_idx) const {
        auto const it = storage_.find(linear_idx);
        return it != storage_.end() ? it->second : empty_value;
    }

    void
    set(size_t linear_idx, value_type val) {
        storage_[linear_idx] = val;
    }

    // -------------------------------------------------------------------------
    // Tuple-based accessors (user-facing API)
    // -------------------------------------------------------------------------

    T
    get(index_tuple_type index) const {
        auto idx = std::apply(
            [](auto... values) { return axes_type::index_of_values(values...); }, index);
        if (!idx.has_value()) {
            return empty_value;
        }
        auto const it = storage_.find(*idx);
        return it != storage_.end() ? it->second : empty_value;
    }

    void
    set(index_tuple_type index, value_type val) {
        auto idx = std::apply(
            [](auto... values) { return axes_type::index_of_values(values...); }, index);
        if (idx.has_value()) {
            storage_[*idx] = val;
        }
    }
};

// -----------------------------------------------------------------------------
// PermutationMapSelector
// -----------------------------------------------------------------------------
// A trait that selects between PermutationArrayMap and PermutationUnorderedMap
// based on whether AxesT::size exceeds the given Threshold.
//
// If AxesT::size <= Threshold, uses PermutationArrayMap (dense array storage).
// If AxesT::size >  Threshold, uses PermutationUnorderedMap (sparse hash storage).

template <typename AxesT, typename T, T emptyValue, size_t Threshold>
struct PermutationMapSelector {
    static_assert(is_axis_outer_product_v<AxesT>,
                  "PermutationMapSelector AxesT must be an AxisOuterProduct");

    using type = std::conditional_t<(AxesT::size <= Threshold),
                                    PermutationArrayMap<AxesT, T, emptyValue>,
                                    PermutationUnorderedMap<AxesT, T, emptyValue>>;
};

template <typename AxesT, typename T, T emptyValue, size_t Threshold>
using PermutationMapSelector_t =
    typename PermutationMapSelector<AxesT, T, emptyValue, Threshold>::type;

// -----------------------------------------------------------------------------
// PermutationMap
// -----------------------------------------------------------------------------
// Type alias that automatically selects the appropriate map implementation
// based on the axis product size and threshold.
//
// Template parameters:
//   AxesT      - An AxisOuterProduct type defining the key space
//   T          - The value type stored in the map
//   emptyValue - The sentinel value returned for missing entries (default: T(0))
//   Threshold  - Size threshold; uses array if AxesT::size <= Threshold (default: 1024)
//
// Default threshold rationale (for GPU kernel dispatch tables):
//   - Dispatch table lookup occurs on CPU before kernel launch
//   - Kernel launch overhead (10-50+ μs) dwarfs any lookup cost difference
//   - Typical dispatch spaces: 50-500 combinations (dtypes × devices × layouts)
//   - Memory overhead is minimal: 1024 entries × 8 bytes = 8 KB (fits in L1 cache)
//   - Array storage provides O(1) direct indexing with no hash computation
//   - Hash map only benefits truly massive combinatorial spaces (>1024 entries)

template <typename AxesT, typename T, T emptyValue = T(0), size_t Threshold = 1024>
using PermutationMap = PermutationMapSelector_t<AxesT, T, emptyValue, Threshold>;

// =============================================================================
// CONCEPTS
// =============================================================================

// -----------------------------------------------------------------------------
// PermutationMapConcept: Checks if a type is a valid permutation map
// -----------------------------------------------------------------------------
// A PermutationMap must have:
//   - axes_type: An AxisOuterProduct defining the key space
//   - value_type: The type of values stored
//   - get(size_t) -> value_type: Retrieve value at linear index
//   - set(size_t, value_type) -> void: Store value at linear index

template <typename M>
concept PermutationMapConcept =
    requires(M &m, M const &cm, std::size_t idx, typename M::value_type val) {
        typename M::axes_type;
        typename M::value_type;
        { cm.get(idx) } -> std::convertible_to<typename M::value_type>;
        { m.set(idx, val) } -> std::same_as<void>;
    } && AxisOuterProductConcept<typename M::axes_type>;

// -----------------------------------------------------------------------------
// GeneratorForConcept: Checks if a type is a valid generator for a map
// -----------------------------------------------------------------------------

template <typename Gen, typename Map>
concept GeneratorForConcept = PermutationMapConcept<Map> && requires(Map &m) {
    { Gen::apply(m) } -> std::same_as<void>;
};

// =============================================================================
// GENERATOR TYPES
// =============================================================================
// Generators describe how to populate a PermutationMap declaratively.
// They have a static apply() method that populates the map.
//
// Two kinds of generators:
//   1. Value-dependent: Use an Instantiator<Values...>::get() to produce T
//   2. Constant: Produce the same T regardless of Values...

// -----------------------------------------------------------------------------
// PointGenerator: Generates a single value at a specific point
// -----------------------------------------------------------------------------
// Instantiator must be a template with signature:
//   template <auto... Values> struct Inst { static T get(); };

template <template <auto...> typename Instantiator, auto... Values> struct PointGenerator {
    template <typename Map>
        requires PermutationMapConcept<Map>
    static void
    apply(Map &map) {
        // Check that the instantiator returns the correct type
        static_assert(
            std::is_same_v<decltype(Instantiator<Values...>::get()), typename Map::value_type>,
            "Instantiator::get() must return Map::value_type");

        // Check that all values are valid for the map's axes
        static_assert(Map::axes_type::index_of_values(Values...).has_value(),
                      "PointGenerator values must be valid members of the map's axes");

        constexpr auto idx = Map::axes_type::index_of_values(Values...).value();
        map.set(idx, Instantiator<Values...>::get());
    }
};

// -----------------------------------------------------------------------------
// SubspaceGenerator: Generates values over all points in a subspace
// -----------------------------------------------------------------------------
// Subspace must be an AxisOuterProduct whose axes are subsets of the map's axes.
// This is checked at apply() time to prevent accidental full-space instantiation.

template <template <auto...> typename Instantiator, typename Subspace> struct SubspaceGenerator {
    static_assert(is_axis_outer_product_v<Subspace>,
                  "SubspaceGenerator requires Subspace to be an AxisOuterProduct");

  private:
    // Action for for_each_permutation: stores each instantiated value
    template <typename InstType, auto... Vals> struct StoreAction {
        template <typename Map>
        static void
        apply(Map &map) {
            // These are already validated by the outer apply(), but we include
            // them here for defense-in-depth and clearer error messages.
            static_assert(Map::axes_type::index_of_values(Vals...).has_value(),
                          "Subspace values must be valid members of the map's axes");

            constexpr auto idx = Map::axes_type::index_of_values(Vals...).value();
            map.set(idx, InstType::get());
        }
    };

  public:
    template <typename Map>
        requires PermutationMapConcept<Map>
    static void
    apply(Map &map) {
        // Verify the subspace is a subspace of the map's axes.
        // A space is a subspace of itself, so using the full space is valid.
        // This check ensures all subspace values are valid for the map.
        static_assert(is_subspace_of_v<Subspace, typename Map::axes_type>,
                      "SubspaceGenerator: Subspace must be a subspace of the map's axes. "
                      "Each axis of the Subspace must be a subset of (or equal to) the "
                      "corresponding axis in the map.");

        for_each_permutation<Subspace, Instantiator, StoreAction>::apply(map);
    }
};

// -----------------------------------------------------------------------------
// FillGenerator: Fills all points in a subspace with a constant value
// -----------------------------------------------------------------------------
// No instantiation per permutation - just stores the same value everywhere.
// Useful for filling unsupported combinations with a default or error handler.

template <typename Subspace, auto Value> struct FillGenerator {
    static_assert(is_axis_outer_product_v<Subspace>,
                  "FillGenerator requires Subspace to be an AxisOuterProduct");

  private:
    template <auto... Vals> struct FillAction {
        template <typename Map>
        static void
        apply(Map &map) {
            static_assert(Map::axes_type::index_of_values(Vals...).has_value(),
                          "Subspace values must be valid members of the map's axes");

            constexpr auto idx = Map::axes_type::index_of_values(Vals...).value();
            map.set(idx, Value);
        }
    };

  public:
    template <typename Map>
        requires PermutationMapConcept<Map>
    static void
    apply(Map &map) {
        // Verify the subspace is a subspace of the map's axes.
        // A space is a subspace of itself, so using the full space is valid.
        static_assert(is_subspace_of_v<Subspace, typename Map::axes_type>,
                      "FillGenerator: Subspace must be a subspace of the map's axes. "
                      "Each axis of the Subspace must be a subset of (or equal to) the "
                      "corresponding axis in the map.");

        // Verify the value type matches (or is convertible)
        static_assert(std::is_convertible_v<decltype(Value), typename Map::value_type>,
                      "FillGenerator: Value must be convertible to Map::value_type");

        for_each_values<Subspace, FillAction>::apply(map);
    }
};

// -----------------------------------------------------------------------------
// GeneratorList: Aggregates multiple generators
// -----------------------------------------------------------------------------

template <typename... Generators> struct GeneratorList {
    template <typename Map>
        requires PermutationMapConcept<Map>
    static void
    apply(Map &map) {
        (Generators::apply(map), ...);
    }
};

// =============================================================================
// MAP CONSTRUCTION FUNCTIONS
// =============================================================================
// These functions create immutable maps from generators. The pattern:
//   1. build_permutation_map<Map, Generators>() - creates a fresh map from generators
//   2. overlay_permutation_map<Generators>(original) - copies and overlays new values
//
// For truly immutable usage, store the result as const:
//   static const auto map = build_permutation_map<MyMap, MyGenerators>();
//
// For layering additional values on top:
//   auto extended = overlay_permutation_map<MoreGenerators>(original);

// -----------------------------------------------------------------------------
// build_permutation_map: Construct a map from a generator specification
// -----------------------------------------------------------------------------

template <typename Map, typename Generators>
    requires PermutationMapConcept<Map> && GeneratorForConcept<Generators, Map>
Map
build_permutation_map() {
    Map map{};
    Generators::apply(map);
    return map;
}

// -----------------------------------------------------------------------------
// overlay_permutation_map: Derive a new map by overlaying generators onto an existing one
// -----------------------------------------------------------------------------
// Creates a copy of the original map and applies generators on top.
// Values from generators overwrite existing values at the same positions.

template <typename Generators, typename Map>
    requires PermutationMapConcept<Map> && GeneratorForConcept<Generators, Map>
Map
overlay_permutation_map(Map const &original) {
    Map result = original;
    Generators::apply(result);
    return result;
}

} // namespace dispatch
} // namespace fvdb

#endif // FVDB_DETAIL_DISPATCH_PERMUTATIONMAP_H
