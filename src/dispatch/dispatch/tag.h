// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Self-normalizing tag: an unordered, uniquely-typed value set.
//
// tag<A, B> and tag<B, A> resolve to the same type (tag_storage<...> in
// canonical order), so concrete function signatures like
//
//     void foo(tag<torch::kCPU, contiguity::strided>, args...)
//
// work regardless of the order the user writes the values.
//
// Canonical order is determined by type_label<decltype(V)> for each value V.
//
#ifndef DISPATCH_DISPATCH_TAG_H
#define DISPATCH_DISPATCH_TAG_H

#include "dispatch/consteval_types.h"
#include "dispatch/label.h"
#include "dispatch/label_sorted.h"

#include <cstddef>
#include <type_traits>

namespace dispatch {

//------------------------------------------------------------------------------
// tag_storage: the concrete ordered struct that tag<...> resolves to
//------------------------------------------------------------------------------

template <auto... V> struct tag_storage {};

//------------------------------------------------------------------------------
// is_tag trait (recognizes tag_storage)
//------------------------------------------------------------------------------

template <typename T> struct is_tag : consteval_false_type {};

template <auto... V> struct is_tag<tag_storage<V...>> : consteval_true_type {};

template <typename T>
consteval bool
is_tag_v() {
    return is_tag<T>::value();
}

template <typename T>
concept tag_like = is_tag_v<T>();

//------------------------------------------------------------------------------
// tag: self-normalizing alias
//------------------------------------------------------------------------------
// tag<V...> sorts V... by type_label<decltype(V)> into a canonical
// tag_storage<...>. tag<A, B> and tag<B, A> are the same type.

namespace detail {

// Check that all values have distinct types
template <typename T, typename... Rest>
consteval bool
unique_value_types(T, Rest...) {
    if constexpr (sizeof...(Rest) == 0) {
        return true;
    } else {
        return ((!std::is_same_v<T, Rest>) && ...) && unique_value_types(Rest{}...);
    }
}

// Convert sorted_values to tag_storage
template <typename Sorted> struct sorted_values_to_tag_storage;

template <auto... V> struct sorted_values_to_tag_storage<sorted_values<V...>> {
    using type = tag_storage<V...>;
};

// make_tag: validates and sorts
template <auto... V> struct make_tag {
    static_assert(sizeof...(V) > 0, "tag must have at least one value");
    static_assert(unique_value_types(V...), "tag values must have unique types");

    using sorted = value_sort<V...>;
    using type   = typename sorted_values_to_tag_storage<sorted>::type;
};

} // namespace detail

template <auto... V> using tag = typename detail::make_tag<V...>::type;

} // namespace dispatch

#endif // DISPATCH_DISPATCH_TAG_H
