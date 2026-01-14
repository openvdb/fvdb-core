// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_DISPATCH_INDEXSPACE_H
#define FVDB_DETAIL_DISPATCH_INDEXSPACE_H

#include "fvdb/detail/dispatch/Traits.h"

#include <nanovdb/util/Util.h>

#include <array>
#include <cstddef>
#include <functional>
#include <utility>

namespace fvdb {
namespace dispatch {

template <size_t... Is> using Sizes = std::index_sequence<Is...>;

template <size_t... Is> using Indices = std::index_sequence<Is...>;

template <typename T> struct is_index_sequence : std::false_type {};

template <size_t... Is> struct is_index_sequence<std::index_sequence<Is...>> : std::true_type {};

template <typename T>
concept IndexSpace = is_index_sequence<T>::value;

template <typename T>
concept IndexPoint = is_index_sequence<T>::value;

template <typename T> struct Rank;

template <size_t... Is> struct Rank<Sizes<Is...>> {
    static consteval size_t
    value() {
        return sizeof...(Is);
    }
};

template <typename T> struct Numel;

template <size_t... Is> struct Numel<Sizes<Is...>> {
    static consteval size_t
    value() {
        return (Is * ... * 1);
    }
};

template <typename T>
concept TensorIndexSpace = IndexSpace<T> && Rank<T>::value() > 0;
template <typename T>
concept ScalarIndexSpace = IndexSpace<T> && Rank<T>::value() == 0;

template <typename T>
concept NonEmptyIndexSpace = IndexSpace<T> && Numel<T>::value() > 0;

template <typename T>
concept TensorIndexPoint = IndexPoint<T> && Rank<T>::value() > 0;
template <typename T>
concept ScalarIndexPoint = IndexPoint<T> && Rank<T>::value() == 0;

template <typename S, typename T>
concept SameRank = Rank<S>::value() == Rank<T>::value();

template <typename T> struct Shape;

template <IndexSpace Space> struct Shape<Space> {
    using type = Space;
};

template <typename T, size_t V> struct Prepend;

template <size_t... Is, size_t V> struct Prepend<Sizes<Is...>, V> {
    using type = Sizes<V, Is...>;
};

template <NonEmptyIndexSpace Space, size_t linearIndex> struct IndicesFromLinearIndex;

template <size_t I, size_t linearIndex>
    requires NonEmptyIndexSpace<Sizes<I>>
struct IndicesFromLinearIndex<Sizes<I>, linearIndex> {
    // happily will use indices out of bounds.
    using type = Indices<linearIndex>;
};

template <size_t I0, size_t... TailIs, size_t linearIndex>
    requires NonEmptyIndexSpace<Sizes<I0, TailIs...>> && NonEmptyIndexSpace<Sizes<TailIs...>>
struct IndicesFromLinearIndex<Sizes<I0, TailIs...>, linearIndex> {
    static consteval size_t
    tailNumel() {
        return (TailIs * ... * 1);
    }

    using type = typename Prepend<
        typename IndicesFromLinearIndex<Sizes<TailIs...>, linearIndex % tailNumel()>::type,
        linearIndex / tailNumel()>::type;
};

template <NonEmptyIndexSpace Space, IndexPoint Index> struct LinearIndexFromIndices;

template <size_t S, size_t I>
    requires NonEmptyIndexSpace<Sizes<S>>
struct LinearIndexFromIndices<Sizes<S>, Indices<I>> {
    static consteval size_t
    value() {
        return I;
    }
};

template <size_t S0, size_t... TailSs, size_t I0, size_t... TailIs>
    requires NonEmptyIndexSpace<Sizes<S0, TailSs...>> && NonEmptyIndexSpace<Sizes<TailSs...>>
struct LinearIndexFromIndices<Sizes<S0, TailSs...>, Indices<I0, TailIs...>> {
    static consteval size_t
    tailNumel() {
        return (TailSs * ... * 1);
    }

    static consteval size_t
    value() {
        return I0 * tailNumel() +
               LinearIndexFromIndices<Sizes<TailSs...>, Indices<TailIs...>>::value();
    }
};

template <NonEmptyIndexSpace Space, typename LinearIndicesSeq> struct IndexSpaceVisitHelper;

template <NonEmptyIndexSpace Space, size_t... linearIndices>
struct IndexSpaceVisitHelper<Space, std::index_sequence<linearIndices...>> {
    template <typename Visitor>
    static void
    visit(Visitor &&visitor) {
        // Note: don't forward visitor in fold - it's invoked multiple times
        (std::invoke(visitor, typename IndicesFromLinearIndex<Space, linearIndices>::type{}), ...);
    }
};

template <typename Visitor, NonEmptyIndexSpace Space>
void
visit_index_space(Visitor &&visitor, Space) {
    IndexSpaceVisitHelper<Space, std::make_index_sequence<Numel<Space>::value()>>::visit(
        std::forward<Visitor>(visitor));
}

template <typename Visitor, NonEmptyIndexSpace... Spaces>
void
visit_index_spaces(Visitor &&visitor, Spaces... spaces) {
    // Note: don't forward visitor in fold - it's used for each space
    (visit_index_space(visitor, spaces), ...);
}

} // namespace dispatch
} // namespace fvdb

#endif // FVDB_DETAIL_DISPATCH_INDEXSPACE_H
