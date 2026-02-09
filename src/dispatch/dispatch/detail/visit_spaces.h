// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Compile-time visitation over axes/extents spaces. Instantiates the visitor
// separately for each coordinate in the space, enabling per-coordinate template
// instantiation (used internally by dispatch_table construction).
//
#ifndef DISPATCH_DISPATCH_DETAIL_VISIT_SPACES_H
#define DISPATCH_DISPATCH_DETAIL_VISIT_SPACES_H

#include "dispatch/detail/index_math.h"

#include <functional>
#include <utility>

namespace dispatch {

//------------------------------------------------------------------------------
// Visiting extents with indices per visitation
//------------------------------------------------------------------------------

namespace detail {

template <typename Extents, typename LinearPointSeq> struct extents_visit_helper {
    static_assert(extents_like<Extents>, "Extents must be an extents type");
    static_assert(index_sequence_like<LinearPointSeq>,
                  "LinearPointSeq must be an index sequence type");
};

template <extents_like Extents, size_t... linearIndices>
struct extents_visit_helper<Extents, std::index_sequence<linearIndices...>> {
    template <typename Visitor>
    static void
    visit(Visitor &visitor) {
        // Note: don't forward visitor in fold - it's invoked multiple times
        (std::invoke(visitor, indices_from_linear_index_t<Extents, linearIndices>{}), ...);
    }
};

} // namespace detail

// Compile-time visit all index coordinates in an extents space.
// Instantiates visitor separately for each indices<...> coordinate.
template <typename Visitor, typename Extents>
void
visit_extents_space(Visitor &visitor, Extents) {
    static_assert(extents_like<Extents>, "Extents must be an extents type");
    static_assert(non_empty<Extents>, "Extents must be non-empty space");

    // Don't bother with forward - it's invoked multiple times
    detail::extents_visit_helper<Extents, std::make_index_sequence<volume_v<Extents>()>>::visit(
        visitor);
}

template <typename Visitor, typename... Extents>
void
visit_extents_spaces(Visitor &visitor, Extents... extents) {
    static_assert((extents_like<Extents> && ...), "Extents must be an extents types");
    static_assert((non_empty<Extents> && ...), "Extents must be non-empty spaces");

    // Don't bother with forward - it's invoked multiple times
    (visit_extents_space(visitor, extents), ...);
}

//------------------------------------------------------------------------------
// Visiting axes with tags per visitation
//------------------------------------------------------------------------------

namespace detail {

template <typename Axes, typename LinearPointSeq> struct axes_visit_helper {
    static_assert(axes_like<Axes>, "Axes must be an axes type");
    static_assert(index_sequence_like<LinearPointSeq>,
                  "LinearPointSeq must be an index sequence type");
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

// Compile-time visit all tag coordinates in an axes space.
// Instantiates visitor separately for each tag<...> coordinate.
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

#endif // DISPATCH_DISPATCH_DETAIL_VISIT_SPACES_H
