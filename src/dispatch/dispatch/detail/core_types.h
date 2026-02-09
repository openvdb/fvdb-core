// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Core dispatch type vocabulary. Defines the fundamental building blocks of the
// dispatch system: compile-time boolean wrappers, axes, labels, tags, extents,
// indices, and the concrete dispatch enums.
//
// This file is an implementation detail of the dispatch library. Users should
// include the top-level headers (dispatch_table.h, with_value.h, etc.) instead.
//
#ifndef DISPATCH_DISPATCH_DETAIL_CORE_TYPES_H
#define DISPATCH_DISPATCH_DETAIL_CORE_TYPES_H

#include <cstddef>
#include <type_traits>

namespace dispatch {

// =============================================================================
// consteval_types: compile-time boolean type wrappers
// =============================================================================
// By using consteval member functions instead of static constexpr variables,
// we guarantee values exist only at compile-time and are never instantiated as
// addressable objects. This addresses a real issue with deeply nested templates
// and nvcc.

template <bool B> struct consteval_bool_type {
    static consteval bool
    value() {
        return B;
    }
};

using consteval_true_type  = consteval_bool_type<true>;
using consteval_false_type = consteval_bool_type<false>;

// =============================================================================
// axis: a collection of values that are all the same type and all unique
// =============================================================================
// Represents a single dimension of a dispatch space.

template <auto... V> struct axis {
    static_assert(sizeof...(V) > 0, "axis must have at least one value");
};

template <auto V0> struct axis<V0> {
    using value_type = decltype(V0);
};

namespace detail {

template <typename T, typename... Rest>
consteval bool
unique_values(T head, Rest... tail) {
    if constexpr (sizeof...(tail) == 0) {
        return true;
    } else {
        return ((head != tail) && ...) && unique_values(tail...);
    }
}

} // namespace detail

template <auto V0, auto... V> struct axis<V0, V...> {
    using value_type = decltype(V0);
    static_assert((std::is_same_v<value_type, decltype(V)> && ... && true),
                  "axis values must be the same type");
    static_assert(detail::unique_values<value_type>(V0, V...), "axis values must be unique");
};

//------------------------------------------------------------------------------
// is_axis trait
//------------------------------------------------------------------------------

template <typename T> struct is_axis : consteval_false_type {};

template <auto... V> struct is_axis<axis<V...>> : consteval_true_type {};

template <typename T>
consteval bool
is_axis_v() {
    return is_axis<T>::value();
}

template <typename T>
concept axis_like = is_axis_v<T>();

//------------------------------------------------------------------------------
// axis_value_type trait
//------------------------------------------------------------------------------

template <typename T> struct axis_value_type {
    static_assert(axis_like<T>, "axis_value_type requires an axis type");
};

template <axis_like Axis> struct axis_value_type<Axis> {
    using type = typename Axis::value_type;
};

template <typename T> using axis_value_type_t = typename axis_value_type<T>::type;

//------------------------------------------------------------------------------
// extent: number of values in an axis
//------------------------------------------------------------------------------

template <typename T> struct extent;

template <auto... V> struct extent<axis<V...>> {
    static consteval size_t
    value() {
        return sizeof...(V);
    }
};

template <typename T>
consteval size_t
extent_v() {
    return extent<T>::value();
}

// =============================================================================
// label: fixed_label, type_label, named
// =============================================================================
//
// fixed_label<N>: compile-time string usable as NTTP, with lexicographic
// comparison for canonical ordering.
//
// type_label<T>: trait associating a fixed_label with each axis value type.
// Must be specialized for every type used in tags/axes.
//
// named<Label, T>: structural wrapper for disambiguating multiple axes that
// share the same underlying value type.

//------------------------------------------------------------------------------
// fixed_label: compile-time string for use as NTTP and for ordering
//------------------------------------------------------------------------------

template <std::size_t N> struct fixed_label {
    char data[N]{};

    consteval fixed_label() = default;

    consteval fixed_label(char const (&str)[N]) {
        for (std::size_t i = 0; i < N; ++i) {
            data[i] = str[i];
        }
    }

    static consteval std::size_t
    size() {
        // N includes the null terminator
        return N - 1;
    }

    consteval char
    operator[](std::size_t i) const {
        return data[i];
    }

    consteval bool operator==(fixed_label const &) const = default;
};

// CTAD
template <std::size_t N> fixed_label(char const (&)[N]) -> fixed_label<N>;

//------------------------------------------------------------------------------
// compare_fixed_labels: cross-length lexicographic comparison
//------------------------------------------------------------------------------
// Returns negative if a < b, 0 if equal, positive if a > b.

template <std::size_t N, std::size_t M>
consteval int
compare_fixed_labels(fixed_label<N> const &a, fixed_label<M> const &b) {
    auto const a_len   = N - 1; // exclude null terminator
    auto const b_len   = M - 1;
    auto const min_len = a_len < b_len ? a_len : b_len;
    for (std::size_t i = 0; i < min_len; ++i) {
        if (a.data[i] < b.data[i])
            return -1;
        if (a.data[i] > b.data[i])
            return 1;
    }
    if (a_len < b_len)
        return -1;
    if (a_len > b_len)
        return 1;
    return 0;
}

//------------------------------------------------------------------------------
// type_label: trait associating a fixed_label with each axis value type
//------------------------------------------------------------------------------
// Must be specialized for every type used as a tag value's decltype or as
// the value_type of an axis. The label should be globally unique.

template <typename T> struct type_label {
    static_assert(sizeof(T) == 0,
                  "type_label must be specialized for each axis value type. "
                  "Add: template <> struct type_label<YourType> { "
                  "static consteval auto value() { return fixed_label(\"your.label\"); } };");
};

//------------------------------------------------------------------------------
// named: structural wrapper for same-underlying-type disambiguation
//------------------------------------------------------------------------------

template <fixed_label Label, typename T> struct named {
    using value_type = T;

    T value;

    consteval bool operator==(named const &) const = default;
};

//------------------------------------------------------------------------------
// is_named trait
//------------------------------------------------------------------------------

template <typename T> struct is_named : consteval_false_type {};

template <fixed_label Label, typename T> struct is_named<named<Label, T>> : consteval_true_type {};

template <typename T>
consteval bool
is_named_v() {
    return is_named<T>::value();
}

//------------------------------------------------------------------------------
// type_label specialization for named<Label, T> — self-registering
//------------------------------------------------------------------------------

template <fixed_label Label, typename T> struct type_label<named<Label, T>> {
    static consteval auto
    value() {
        return Label;
    }
};

// =============================================================================
// label_sorted: compile-time insertion sort by type_label
// =============================================================================
//
// value_sort<V...>: sorts NTTP values by type_label<decltype(V)>,
// producing sorted_values<V...> in canonical order.
//
// type_sort<T...>: sorts types by a caller-provided label extraction,
// producing sorted_types<T...> in canonical order.

//------------------------------------------------------------------------------
// Generic sorted output containers
//------------------------------------------------------------------------------

template <auto... V> struct sorted_values {};

template <typename... T> struct sorted_types {};

//------------------------------------------------------------------------------
// Value sort: sort NTTP values by type_label<decltype(V)>
//------------------------------------------------------------------------------

namespace detail {

// Compare two NTTP values by their type_label
template <auto A, auto B>
consteval bool
value_label_less() {
    return compare_fixed_labels(type_label<decltype(A)>::value(),
                                type_label<decltype(B)>::value()) < 0;
}

// ---- insert a value into an already-sorted sorted_values ----

template <auto V, typename Sorted> struct insert_value_into;

// Insert into empty
template <auto V> struct insert_value_into<V, sorted_values<>> {
    using type = sorted_values<V>;
};

// V sorts before Head -> V goes first
template <auto V, auto Head, auto... Tail>
    requires(value_label_less<V, Head>())
struct insert_value_into<V, sorted_values<Head, Tail...>> {
    using type = sorted_values<V, Head, Tail...>;
};

// V does not sort before Head -> Head stays, recurse into Tail
template <auto V, auto Head, auto... Tail>
    requires(!value_label_less<V, Head>())
struct insert_value_into<V, sorted_values<Head, Tail...>> {
  private:
    using tail_inserted = typename insert_value_into<V, sorted_values<Tail...>>::type;

    template <typename S> struct prepend_head;

    template <auto... Vs> struct prepend_head<sorted_values<Vs...>> {
        using type = sorted_values<Head, Vs...>;
    };

  public:
    using type = typename prepend_head<tail_inserted>::type;
};

// ---- insertion sort on value pack ----

template <auto... V> struct value_sort_impl {
    using type = sorted_values<>;
};

template <auto V> struct value_sort_impl<V> {
    using type = sorted_values<V>;
};

template <auto Head, auto... Tail> struct value_sort_impl<Head, Tail...> {
  private:
    using sorted_tail = typename value_sort_impl<Tail...>::type;

  public:
    using type = typename insert_value_into<Head, sorted_tail>::type;
};

} // namespace detail

// Public interface
template <auto... V> using value_sort = typename detail::value_sort_impl<V...>::type;

//------------------------------------------------------------------------------
// Type sort: sort types by type_label applied via a label extractor
//------------------------------------------------------------------------------

namespace detail {

// Compare two types by their labels
template <typename LabelExtractor, typename A, typename B>
consteval bool
type_label_less() {
    return compare_fixed_labels(LabelExtractor::template label<A>(),
                                LabelExtractor::template label<B>()) < 0;
}

// ---- insert a type into an already-sorted sorted_types ----

template <typename LabelExtractor, typename T, typename Sorted> struct insert_type_into;

// Insert into empty
template <typename LabelExtractor, typename T>
struct insert_type_into<LabelExtractor, T, sorted_types<>> {
    using type = sorted_types<T>;
};

// T sorts before Head -> T goes first
template <typename LabelExtractor, typename T, typename Head, typename... Tail>
    requires(type_label_less<LabelExtractor, T, Head>())
struct insert_type_into<LabelExtractor, T, sorted_types<Head, Tail...>> {
    using type = sorted_types<T, Head, Tail...>;
};

// T does not sort before Head -> Head stays, recurse into Tail
template <typename LabelExtractor, typename T, typename Head, typename... Tail>
    requires(!type_label_less<LabelExtractor, T, Head>())
struct insert_type_into<LabelExtractor, T, sorted_types<Head, Tail...>> {
  private:
    using tail_inserted = typename insert_type_into<LabelExtractor, T, sorted_types<Tail...>>::type;

    template <typename S> struct prepend_head;

    template <typename... Ts> struct prepend_head<sorted_types<Ts...>> {
        using type = sorted_types<Head, Ts...>;
    };

  public:
    using type = typename prepend_head<tail_inserted>::type;
};

// ---- insertion sort on type pack ----

template <typename LabelExtractor, typename... T> struct type_sort_impl {
    using type = sorted_types<>;
};

template <typename LabelExtractor, typename T> struct type_sort_impl<LabelExtractor, T> {
    using type = sorted_types<T>;
};

template <typename LabelExtractor, typename Head, typename... Tail>
struct type_sort_impl<LabelExtractor, Head, Tail...> {
  private:
    using sorted_tail = typename type_sort_impl<LabelExtractor, Tail...>::type;

  public:
    using type = typename insert_type_into<LabelExtractor, Head, sorted_tail>::type;
};

} // namespace detail

// Public interface
template <typename LabelExtractor, typename... T>
using type_sort = typename detail::type_sort_impl<LabelExtractor, T...>::type;

// =============================================================================
// extents: compile-time extent (size) sequence for index spaces
// =============================================================================

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

// =============================================================================
// indices: compile-time index sequence for coordinates in an index space
// =============================================================================

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

// =============================================================================
// tag: self-normalizing tag — an unordered, uniquely-typed value set
// =============================================================================
//
// tag<A, B> and tag<B, A> resolve to the same type (tag_storage<...> in
// canonical order). Canonical order is determined by type_label<decltype(V)>.

//------------------------------------------------------------------------------
// tag_storage: the concrete ordered struct that tag<...> resolves to
//------------------------------------------------------------------------------

template <auto... V> struct tag_storage {};

//------------------------------------------------------------------------------
// is_tag trait (recognizes tag_storage)
//------------------------------------------------------------------------------

template <typename T> struct is_tag : consteval_false_type {};

template <auto... V> struct is_tag<tag_storage<V...>> : consteval_true_type {};

template <typename T>
consteval bool
is_tag_v() {
    return is_tag<T>::value();
}

template <typename T>
concept tag_like = is_tag_v<T>();

//------------------------------------------------------------------------------
// tag: self-normalizing alias
//------------------------------------------------------------------------------

namespace detail {

// Check that all values have distinct types
template <typename T, typename... Rest>
consteval bool
unique_value_types(T, Rest...) {
    if constexpr (sizeof...(Rest) == 0) {
        return true;
    } else {
        return ((!std::is_same_v<T, Rest>) && ...) && unique_value_types(Rest{}...);
    }
}

// Convert sorted_values to tag_storage
template <typename Sorted> struct sorted_values_to_tag_storage;

template <auto... V> struct sorted_values_to_tag_storage<sorted_values<V...>> {
    using type = tag_storage<V...>;
};

// make_tag: validates and sorts
template <auto... V> struct make_tag {
    static_assert(sizeof...(V) > 0, "tag must have at least one value");
    static_assert(unique_value_types(V...), "tag values must have unique types");

    using sorted = value_sort<V...>;
    using type   = typename sorted_values_to_tag_storage<sorted>::type;
};

} // namespace detail

template <auto... V> using tag = typename detail::make_tag<V...>::type;

// =============================================================================
// axes: self-normalizing axes — an unordered set of uniquely-typed axis dims
// =============================================================================
//
// axes<Axis0, Axis1> and axes<Axis1, Axis0> resolve to the same type
// (axes_storage<...> in canonical order).

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

namespace detail {

// Label extractor for axes: extracts type_label from axis_value_type_t
struct axes_label_extractor {
    template <typename Axis>
    static consteval auto
    label() {
        return type_label<axis_value_type_t<Axis>>::value();
    }
};

// Check that all axes have unique value types by comparing their labels.

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

template <typename... Axes>
consteval bool
unique_axis_value_types() {
    if constexpr (sizeof...(Axes) <= 1) {
        return true;
    } else {
        return unique_axis_labels_impl<Axes...>();
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

// =============================================================================
// enums: placement, determinism, contiguity, scheduling
// =============================================================================
// Each enum has a type_label specialization co-located with its definition,
// as well as convenience axis aliases for the full value set.

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

#endif // DISPATCH_DISPATCH_DETAIL_CORE_TYPES_H
