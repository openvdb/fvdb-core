// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef DISPATCH_TYPES_H
#define DISPATCH_TYPES_H

#include <cstddef>
#include <cstdint>

namespace dispatch {

template <size_t... S> struct sizes {};

template <size_t... I> struct indices {};

template <typename... T> struct types {};

template <auto... V> struct axis {};

template <auto... V> struct tag {};

template <typename... Axes> struct axes {};

enum class placement { in_place, out_of_place };
enum class determinism { not_required, required };
enum class contiguity { strided, contiguous };

using full_placement_axis   = axis<placement::in_place, placement::out_of_place>;
using full_determinism_axis = axis<determinism::not_required, determinism::required>;
using full_contiguity_axis  = axis<contiguity::strided, contiguity::contiguous>;

// Stringification helpers for enum values
inline char const *
to_string(placement p) {
    switch (p) {
    case placement::in_place: return "in_place";
    case placement::out_of_place: return "out_of_place";
    }
    return "unknown";
}

inline char const *
to_string(determinism d) {
    switch (d) {
    case determinism::not_required: return "not_required";
    case determinism::required: return "required";
    }
    return "unknown";
}

inline char const *
to_string(contiguity c) {
    switch (c) {
    case contiguity::strided: return "strided";
    case contiguity::contiguous: return "contiguous";
    }
    return "unknown";
}

} // namespace dispatch

#endif // DISPATCH_TYPES_H
