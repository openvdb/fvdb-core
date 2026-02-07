// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Order-independent tag matching via concepts:
//
//   with_type<Tag, T>  — tag contains any value whose decltype is T.
//   with_value<Tag, V> — tag contains this specific value V.
//
// with_value is defined in terms of with_type so that C++20 concept
// subsumption works correctly: a constraint using with_value<Tag, V>
// is more specific than one using with_type<Tag, decltype(V)>.
//
// Usage:
//
//     template <typename Tag>
//         requires with_value<Tag, torch::kCUDA>
//               && with_value<Tag, contiguity::contiguous>
//     void foo(Tag, ...);
//
#ifndef DISPATCH_DISPATCH_WITH_VALUE_H
#define DISPATCH_DISPATCH_WITH_VALUE_H

#include "dispatch/consteval_types.h"
#include "dispatch/tag.h"

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

} // namespace dispatch

#endif // DISPATCH_DISPATCH_WITH_VALUE_H
