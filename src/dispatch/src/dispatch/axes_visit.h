// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef DISPATCH_VALUE_SPACE_H
#define DISPATCH_VALUE_SPACE_H

#include "dispatch/index_space.h"
#include "dispatch/traits.h"
#include "dispatch/types.h"

#include <array>
#include <concepts>
#include <functional>
#include <optional>
#include <tuple>
#include <type_traits>

namespace dispatch {

//------------------------------------------------------------------------------
// Visiting axes with tags per visitation
//------------------------------------------------------------------------------

namespace detail {

template <typename Axes, typename LinearPointSeq> struct axes_visit_helper {
    static_assert(axes_like<Space>, "Space must be an axes type");
    static_assert(index_sequence_like<LinearPointSeq>,
                  "LinearPointSeq must be an index sequence like");
};

template <axes_like Axes, size_t... linearIndices>
struct axes_visit_helper<Axes, std::index_sequence<linearIndices...>> {
    template <typename Visitor>
    static void
    visit(Visitor &visitor) {
        // Note: don't forward visitor in fold - it's invoked multiple times
        (std::invoke(visitor, tag_from_linear_index_t<Axes, linearIndices>{}), ...);
    }
};

} // namespace detail

template <typename Visitor, typename Axes>
void
visit_axes_space(Visitor &visitor, Axes) {
    static_assert(axes_like<Axes>, "Axes must be an axes type");
    static_assert(non_empty<Axes>, "Axes must be non-empty space");

    // Don't bother with forward - it's invoked multiple times
    detail::axes_visit_helper<Axes, std::make_index_sequence<volume_v<Axes>()>>::visit(visitor);
}

template <typename Visitor, typename... Axes>
void
visit_axes_spaces(Visitor &visitor, Axes... axes) {
    static_assert((axes_like<Axes> && ...), "Axes must be an axes types");
    static_assert((non_empty<Axes> && ...), "Axes must be non-empty spaces");

    // Don't bother with forward - it's invoked multiple times
    (visit_axes_space(visitor, axes), ...);
}

} // namespace dispatch

#endif // DISPATCH_VALUE_SPACE_H
