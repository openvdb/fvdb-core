// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// extents<S...>: compile-time extent (size) sequence for index spaces.
//
#ifndef DISPATCH_DISPATCH_EXTENTS_H
#define DISPATCH_DISPATCH_EXTENTS_H

#include "dispatch/consteval_types.h"

#include <cstddef>

namespace dispatch {

template <size_t... S> struct extents {};

template <typename T> struct is_extents : consteval_false_type {};

template <size_t... S> struct is_extents<extents<S...>> : consteval_true_type {};

template <typename T>
consteval bool
is_extents_v() {
    return is_extents<T>::value();
}

template <typename T>
concept extents_like = is_extents_v<T>();

} // namespace dispatch

#endif // DISPATCH_DISPATCH_EXTENTS_H
