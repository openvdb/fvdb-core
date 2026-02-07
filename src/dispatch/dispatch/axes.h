// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Self-normalizing axes: an unordered set of uniquely-typed axis dimensions.
//
// axes<Axis0, Axis1> and axes<Axis1, Axis0> resolve to the same type
// (axes_storage<...> in canonical order), so the user never needs to worry
// about the order they declare their dispatch dimensions.
//
// Canonical order is determined by type_label<axis_value_type_t<Axis>>.
//
#ifndef DISPATCH_DISPATCH_AXES_H
#define DISPATCH_DISPATCH_AXES_H

#include "dispatch/axis.h"
#include "dispatch/consteval_types.h"
#include "dispatch/label.h"
#include "dispatch/label_sorted.h"

#include <type_traits>

namespace dispatch {

//------------------------------------------------------------------------------
// axes_storage: the concrete ordered struct that axes<...> resolves to
//------------------------------------------------------------------------------

template <typename... Axes> struct axes_storage {
    static_assert(sizeof...(Axes) > 0, "axes must have at least one axis");
    static_assert((is_axis_v<Axes>() && ... && true), "All template parameters must be axis types");
};

//------------------------------------------------------------------------------
// is_axes trait (recognizes axes_storage)
//------------------------------------------------------------------------------

template <typename T> struct is_axes : consteval_false_type {};

template <typename... Axes> struct is_axes<axes_storage<Axes...>> : consteval_true_type {};

template <typename T>
consteval bool
is_axes_v() {
    return is_axes<T>::value();
}

template <typename T>
concept axes_like = is_axes_v<T>();

//------------------------------------------------------------------------------
// axes: self-normalizing alias
//------------------------------------------------------------------------------
// Sorts Axes... by type_label<axis_value_type_t<Axis>> into canonical
// axes_storage<...>. axes<A, B> and axes<B, A> are the same type.

namespace detail {

// Label extractor for axes: extracts type_label from axis_value_type_t
struct axes_label_extractor {
    template <typename Axis>
    static consteval auto
    label() {
        return type_label<axis_value_type_t<Axis>>::value();
    }
};

// Check that all axes have unique value types
template <typename... Axes>
consteval bool
unique_axis_value_types() {
    if constexpr (sizeof...(Axes) <= 1) {
        return true;
    } else {
        // Check by comparing labels — if any two axes have the same label,
        // they have the same value type
        return unique_axis_labels_impl<Axes...>();
    }
}

template <typename Head, typename... Tail>
consteval bool
unique_axis_labels_impl() {
    if constexpr (sizeof...(Tail) == 0) {
        return true;
    } else {
        auto const head_label = type_label<axis_value_type_t<Head>>::value();
        bool const head_unique =
            ((compare_fixed_labels(head_label, type_label<axis_value_type_t<Tail>>::value()) !=
              0) &&
             ...);
        return head_unique && unique_axis_labels_impl<Tail...>();
    }
}

// Convert sorted_types to axes_storage
template <typename Sorted> struct sorted_types_to_axes_storage;

template <typename... Axes> struct sorted_types_to_axes_storage<sorted_types<Axes...>> {
    using type = axes_storage<Axes...>;
};

// make_axes: validates and sorts
template <typename... Axes> struct make_axes {
    static_assert(sizeof...(Axes) > 0, "axes must have at least one axis");
    static_assert((is_axis_v<Axes>() && ... && true), "All template parameters must be axis types");
    static_assert(unique_axis_value_types<Axes...>(), "All axes must have unique value types");

    using sorted = type_sort<axes_label_extractor, Axes...>;
    using type   = typename sorted_types_to_axes_storage<sorted>::type;
};

} // namespace detail

template <typename... Axes> using axes = typename detail::make_axes<Axes...>::type;

} // namespace dispatch

#endif // DISPATCH_DISPATCH_AXES_H
