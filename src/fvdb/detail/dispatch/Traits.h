// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_DISPATCH_TRAITS_H
#define FVDB_DETAIL_DISPATCH_TRAITS_H

#include "fvdb/detail/dispatch/TypesFwd.h"

#include <array>
#include <cstddef>
#include <tuple>
#include <utility>

namespace fvdb {
namespace dispatch {

// =============================================================================
// Index Sequence Utilities
// =============================================================================
//
// Compile-time utilities for working with std::index_sequence.
//
// =============================================================================

// -----------------------------------------------------------------------------
// is_index_sequence - type trait to detect std::index_sequence
// -----------------------------------------------------------------------------

template <typename T> struct is_index_sequence : std::false_type {};

template <size_t... Is> struct is_index_sequence<std::index_sequence<Is...>> : std::true_type {};

// -----------------------------------------------------------------------------
// array_from_indices - convert index_sequence to std::array at compile time
// -----------------------------------------------------------------------------

template <typename T> struct array_from_indices;

template <size_t... Is> struct array_from_indices<std::index_sequence<Is...>> {
    static constexpr std::array<size_t, sizeof...(Is)> value = {Is...};
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
} // namespace fvdb

#endif // FVDB_DETAIL_DISPATCH_TRAITS_H
