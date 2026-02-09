// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Order-independent tag matching and value extraction:
//
//   with_type<Tag, T>  — tag contains any value whose decltype is T.
//   with_value<Tag, V> — tag contains this specific value V.
//   tag_get<T>(tag)    — extract the value of type T from the tag.
//   tag_get<T1, T2, ...>(tag) — extract multiple values as a std::tuple.
//
// with_value is defined in terms of with_type so that C++20 concept
// subsumption works correctly: a constraint using with_value<Tag, V>
// is more specific than one using with_type<Tag, decltype(V)>.
//
// Usage:
//
//     template <typename Tag>
//         requires with_value<Tag, torch::kCUDA>
//               && with_type<Tag, torch::ScalarType>
//     void foo(Tag, ...) {
//         constexpr auto stype = tag_get<torch::ScalarType>(Tag{});
//     }
//
#ifndef DISPATCH_DISPATCH_WITH_VALUE_H
#define DISPATCH_DISPATCH_WITH_VALUE_H

#include "dispatch/detail/core_types.h"

#include <tuple>
#include <type_traits>

namespace dispatch {

//------------------------------------------------------------------------------
// is_with_type: trait checking if a tag contains a value of type T
//------------------------------------------------------------------------------

template <typename Tag, typename T> struct is_with_type : consteval_false_type {
    static_assert(tag_like<Tag>, "is_with_type requires a tag type");
};

// Base case: empty tag — type not found
template <typename T> struct is_with_type<tag_storage<>, T> : consteval_false_type {};

// Head matches type T -> found
template <auto Head, auto... Tail, typename T>
    requires(std::is_same_v<decltype(Head), T>)
struct is_with_type<tag_storage<Head, Tail...>, T> : consteval_true_type {};

// Head does not match type T -> recurse
template <auto Head, auto... Tail, typename T>
    requires(!std::is_same_v<decltype(Head), T>)
struct is_with_type<tag_storage<Head, Tail...>, T> : is_with_type<tag_storage<Tail...>, T> {};

template <typename Tag, typename T>
consteval bool
is_with_type_v() {
    return is_with_type<Tag, T>::value();
}

//------------------------------------------------------------------------------
// with_type concept
//------------------------------------------------------------------------------

template <typename Tag, typename T>
concept with_type = is_with_type_v<Tag, T>();

//------------------------------------------------------------------------------
// is_with_value: trait checking if a tag contains specific value V
//------------------------------------------------------------------------------
// Checks both that the type is present AND that the value matches.

template <typename Tag, auto V> struct is_with_value : consteval_false_type {
    static_assert(tag_like<Tag>, "is_with_value requires a tag type");
};

// Base case: empty tag — value not found
template <auto V> struct is_with_value<tag_storage<>, V> : consteval_false_type {};

// Head matches type and value -> found
template <auto Head, auto... Tail, auto V>
    requires(std::is_same_v<decltype(Head), decltype(V)> && Head == V)
struct is_with_value<tag_storage<Head, Tail...>, V> : consteval_true_type {};

// Head matches type but not value -> not found (blocking: same type, wrong value)
template <auto Head, auto... Tail, auto V>
    requires(std::is_same_v<decltype(Head), decltype(V)> && Head != V)
struct is_with_value<tag_storage<Head, Tail...>, V> : consteval_false_type {};

// Head does not match type -> recurse
template <auto Head, auto... Tail, auto V>
    requires(!std::is_same_v<decltype(Head), decltype(V)>)
struct is_with_value<tag_storage<Head, Tail...>, V> : is_with_value<tag_storage<Tail...>, V> {};

template <typename Tag, auto V>
consteval bool
is_with_value_v() {
    return is_with_value<Tag, V>::value();
}

//------------------------------------------------------------------------------
// with_value concept — defined in terms of with_type for subsumption
//------------------------------------------------------------------------------
// Because with_value includes with_type as a conjunct, the compiler knows
// that with_value<Tag, V> subsumes with_type<Tag, decltype(V)>.

template <typename Tag, auto V>
concept with_value = with_type<Tag, decltype(V)> && is_with_value_v<Tag, V>();

//------------------------------------------------------------------------------
// tag_get: extract the value of a given type from a tag
//------------------------------------------------------------------------------
// Single type: tag_get<T>(tag) returns the value of type T.
// Multiple types: tag_get<T1, T2, ...>(tag) returns std::tuple<T1, T2, ...>.
//
// Requires that the tag contains a value of each requested type (enforced
// via with_type concept).

namespace detail {

template <typename T, typename Tag> struct tag_get_helper;

// Head matches type T -> return it
template <typename T, auto Head, auto... Tail>
    requires(std::is_same_v<decltype(Head), T>)
struct tag_get_helper<T, tag_storage<Head, Tail...>> {
    static constexpr T
    value() {
        return Head;
    }
};

// Head does not match type T -> recurse
template <typename T, auto Head, auto... Tail>
    requires(!std::is_same_v<decltype(Head), T>)
struct tag_get_helper<T, tag_storage<Head, Tail...>> {
    static constexpr T
    value() {
        return tag_get_helper<T, tag_storage<Tail...>>::value();
    }
};

} // namespace detail

// Single type: returns the value directly
template <typename T, typename Tag>
    requires tag_like<Tag> && with_type<Tag, T>
constexpr T
tag_get(Tag = {}) {
    return detail::tag_get_helper<T, Tag>::value();
}

// Multiple types: returns a tuple
template <typename T1, typename T2, typename... Ts, typename Tag>
    requires(with_type<Tag, T1> && with_type<Tag, T2> && (with_type<Tag, Ts> && ...))
constexpr auto
tag_get(Tag = {}) {
    return std::tuple{detail::tag_get_helper<T1, Tag>::value(),
                      detail::tag_get_helper<T2, Tag>::value(),
                      detail::tag_get_helper<Ts, Tag>::value()...};
}

} // namespace dispatch

#endif // DISPATCH_DISPATCH_WITH_VALUE_H
