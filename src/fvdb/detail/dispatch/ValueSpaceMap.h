// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_DISPATCH_VALUESPACEMAP_H
#define FVDB_DETAIL_DISPATCH_VALUESPACEMAP_H

#include "fvdb/detail/dispatch/ValueSpace.h"

#include <functional>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace fvdb {
namespace dispatch {

// =============================================================================
// RuntimeKey: the erased key type for a ValueSpace
// =============================================================================
// Maps ValueAxes<Axis0, Axis1, ...> -> std::tuple<AxisValueType_t<Axis0>, ...>

template <ValueSpace Space> struct RuntimeKeyType;

template <ValueAxis... Axes> struct RuntimeKeyType<ValueAxes<Axes...>> {
    using type = std::tuple<AxisValueType_t<Axes>...>;
};

template <ValueSpace Space> using RuntimeKey_t = typename RuntimeKeyType<Space>::type;

// =============================================================================
// coordToTuple: convert compile-time Values<...> to runtime std::tuple
// =============================================================================

namespace detail {

template <typename Coord, typename IndexSeq> struct CoordToTupleImpl;

template <typename Coord, size_t... Is> struct CoordToTupleImpl<Coord, std::index_sequence<Is...>> {
    static constexpr auto
    apply() {
        return std::make_tuple(get<Is>(Coord{})...);
    }
};

} // namespace detail

template <ValuePack Coord>
constexpr auto
coordToTuple(Coord) {
    using Impl = detail::CoordToTupleImpl<Coord, std::make_index_sequence<PackSize_v<Coord>()>>;
    return Impl::apply();
}

// Overload accepting values directly (convenience)
template <auto... Vs>
constexpr auto
coordToTuple() {
    return coordToTuple(Values<Vs...>{});
}

// =============================================================================
// TupleHash: hash combiner for std::tuple of hashable types
// =============================================================================

template <typename Tuple> struct TupleHash {
    size_t
    operator()(const Tuple &t) const {
        return hashImpl(t, std::make_index_sequence<std::tuple_size_v<Tuple>>{});
    }

  private:
    template <size_t... Is>
    size_t
    hashImpl(const Tuple &t, std::index_sequence<Is...>) const {
        size_t seed = 0;
        // boost::hash_combine style
        ((seed ^= std::hash<std::tuple_element_t<Is, Tuple>>{}(std::get<Is>(t)) + 0x9e3779b9 +
                  (seed << 6) + (seed >> 2)),
         ...);
        return seed;
    }
};

// =============================================================================
// ValueSpaceMap type selector
// =============================================================================

// Convenience alias for the map type itself
template <ValueSpace Space, typename T> struct ValueSpaceMap {
    using space_type = Space;
    using key_type   = RuntimeKey_t<Space>;
    using value_type = T;
    using type       = std::unordered_map<key_type, value_type, TupleHash<key_type>>;

    template <ValueSpace Space, typename T>
    using ValueSpaceMap_t = typename ValueSpaceMap<Space, T>::type;

} // namespace dispatch
} // namespace fvdb

#endif // FVDB_DETAIL_DISPATCH_ValueSpaceMAP_H
