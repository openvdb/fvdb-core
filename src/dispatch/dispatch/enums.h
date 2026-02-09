// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Dispatch enums: placement, determinism, contiguity.
// Each enum has a type_label specialization co-located with its definition,
// as well as convenience axis aliases for the full value set.
//
#ifndef DISPATCH_DISPATCH_ENUMS_H
#define DISPATCH_DISPATCH_ENUMS_H

#include "dispatch/axis.h"
#include "dispatch/label.h"

namespace dispatch {

//------------------------------------------------------------------------------
// placement
//------------------------------------------------------------------------------

enum class placement { in_place, out_of_place };

inline char const *
to_string(placement p) {
    switch (p) {
    case placement::in_place: return "in_place";
    case placement::out_of_place: return "out_of_place";
    }
    return "unknown";
}

template <> struct type_label<placement> {
    static consteval auto
    value() {
        return fixed_label("dispatch.placement");
    }
};

using full_placement_axis = axis<placement::in_place, placement::out_of_place>;

//------------------------------------------------------------------------------
// determinism
//------------------------------------------------------------------------------

enum class determinism { not_required, required };

inline char const *
to_string(determinism d) {
    switch (d) {
    case determinism::not_required: return "not_required";
    case determinism::required: return "required";
    }
    return "unknown";
}

template <> struct type_label<determinism> {
    static consteval auto
    value() {
        return fixed_label("dispatch.determinism");
    }
};

using full_determinism_axis = axis<determinism::not_required, determinism::required>;

//------------------------------------------------------------------------------
// contiguity
//------------------------------------------------------------------------------

enum class contiguity { strided, contiguous };

inline char const *
to_string(contiguity c) {
    switch (c) {
    case contiguity::strided: return "strided";
    case contiguity::contiguous: return "contiguous";
    }
    return "unknown";
}

template <> struct type_label<contiguity> {
    static consteval auto
    value() {
        return fixed_label("dispatch.contiguity");
    }
};

using full_contiguity_axis = axis<contiguity::strided, contiguity::contiguous>;

//------------------------------------------------------------------------------
// scheduling
//------------------------------------------------------------------------------
// Selects the CPU thread pool scheduling strategy:
//   uniform  - Static partitioning (broadcast-style). Optimal when work per
//              element is even. Lowest dispatch overhead.
//   adaptive - Work-stealing (Chase-Lev deques). Optimal when work per element
//              is variable or unpredictable.

enum class scheduling { uniform, adaptive };

inline char const *
to_string(scheduling s) {
    switch (s) {
    case scheduling::uniform: return "uniform";
    case scheduling::adaptive: return "adaptive";
    }
    return "unknown";
}

template <> struct type_label<scheduling> {
    static consteval auto
    value() {
        return fixed_label("dispatch.scheduling");
    }
};

using full_scheduling_axis = axis<scheduling::uniform, scheduling::adaptive>;

} // namespace dispatch

#endif // DISPATCH_DISPATCH_ENUMS_H
