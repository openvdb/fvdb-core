// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// tag_match: Type matching utilities for tag types.
// Provides is_tag_match to check if a tag contains specific values,
// using blocking removal logic (first axis of matching type must match value).
//
#ifndef DISPATCH_DISPATCH_TAG_MATCH_H
#define DISPATCH_DISPATCH_TAG_MATCH_H

#include "dispatch/detail.h"
#include "dispatch/types.h"

namespace dispatch {

namespace detail {

//------------------------------------------------------------------------------
// remove_axis
//------------------------------------------------------------------------------
// Removes a value V from a tag, with "blocking" semantics:
// - Find the first axis whose TYPE matches V's type
// - If that axis's VALUE equals V: success, return residual tag
// - If that axis's VALUE differs from V: hard failure (no backtracking)
//------------------------------------------------------------------------------

// Result type for successful removal - carries the residual tag
template <typename Residual> struct remove_axis_success : consteval_true_type {
    using residual = Residual;
};

// Forward declaration
template <typename Tag, auto V> struct remove_axis_impl;

// Base Case: Tag is empty -> Value not found
template <auto V> struct remove_axis_impl<tag<>, V> : consteval_false_type {};

// Recursive Case: Type mismatch, tail fails -> propagate failure
template <auto Head, auto... Tail, auto V>
    requires(!std::is_same_v<decltype(Head), decltype(V)> &&
             !remove_axis_impl<tag<Tail...>, V>::value())
struct remove_axis_impl<tag<Head, Tail...>, V> : consteval_false_type {};

// Recursive Case: Type mismatch, tail succeeds -> prepend Head to residual
template <auto Head, auto... Tail, auto V>
    requires(!std::is_same_v<decltype(Head), decltype(V)> &&
             remove_axis_impl<tag<Tail...>, V>::value())
struct remove_axis_impl<tag<Head, Tail...>, V> : consteval_true_type {
    using tail_result = remove_axis_impl<tag<Tail...>, V>;
    using residual    = typename prepend_value<Head, typename tail_result::residual>::type;
};

// Recursive Case: Type match + value match -> success, residual is Tail
template <auto Head, auto... Tail, auto V>
    requires(std::is_same_v<decltype(Head), decltype(V)> && Head == V)
struct remove_axis_impl<tag<Head, Tail...>, V> : remove_axis_success<tag<Tail...>> {};

// Recursive Case: Type match + value mismatch -> hard failure (blocking)
template <auto Head, auto... Tail, auto V>
    requires(std::is_same_v<decltype(Head), decltype(V)> && Head != V)
struct remove_axis_impl<tag<Head, Tail...>, V> : consteval_false_type {};

// Public alias
template <typename Tag, auto V> using remove_axis = remove_axis_impl<Tag, V>;

} // namespace detail

//------------------------------------------------------------------------------
// is_tag_match
//------------------------------------------------------------------------------
// Checks if a tag matches a sequence of test values.
// Each value is removed in order using blocking removal logic.
//------------------------------------------------------------------------------

// Forward declaration
template <typename T, auto... TestValues> struct is_tag_match;

namespace detail {

// Chain helper: continues matching if previous step succeeded
template <typename StepResult, auto... RemainingValues>
struct match_chain : consteval_false_type {};

template <typename StepResult, auto... RemainingValues>
    requires(StepResult::value())
struct match_chain<StepResult, RemainingValues...>
    : is_tag_match<typename StepResult::residual, RemainingValues...> {};

} // namespace detail

// Primary Template: T is not a tag -> static_assert failure
template <typename T, auto... TestValues> struct is_tag_match : consteval_false_type {
    static_assert(tag_like<T>, "first parameter to is_tag_match must be tag");
};

// Base Case: No TestValues left -> match succeeded
template <tag_like T> struct is_tag_match<T> : consteval_true_type {};

// Recursive Step: Try to remove V, then continue with remaining values
template <tag_like T, auto V, auto... Vs>
struct is_tag_match<T, V, Vs...> : detail::match_chain<detail::remove_axis<T, V>, Vs...> {};

// Convenience consteval function (matches style in types.h)
template <typename T, auto... Vs>
consteval bool
is_tag_match_v() {
    return is_tag_match<T, Vs...>::value();
}

// Concept
template <typename T, auto... Vs>
concept tag_match = is_tag_match_v<T, Vs...>();

//------------------------------------------------------------------------------
// A trait that can take the place of a specific device axis enum type
// so we can talk about gpu/cpu/pvt1 without referencing torch::DeviceType
// explicitly, it can be specialized in torch_types.h

} // namespace dispatch

#endif // DISPATCH_DISPATCH_TAG_MATCH_H
