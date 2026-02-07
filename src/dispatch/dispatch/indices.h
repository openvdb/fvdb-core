// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// indices<I...>: compile-time index sequence for coordinates in an index space.
//
#ifndef DISPATCH_DISPATCH_INDICES_H
#define DISPATCH_DISPATCH_INDICES_H

#include "dispatch/consteval_types.h"

#include <cstddef>

namespace dispatch {

template <size_t... I> struct indices {};

template <typename T> struct is_indices : consteval_false_type {};

template <size_t... I> struct is_indices<indices<I...>> : consteval_true_type {};

template <typename T>
consteval bool
is_indices_v() {
    return is_indices<T>::value();
}

template <typename T>
concept indices_like = is_indices_v<T>();

} // namespace dispatch

#endif // DISPATCH_DISPATCH_INDICES_H
