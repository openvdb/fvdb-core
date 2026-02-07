// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Compile-time insertion sort by type_label. Provides two sort operations:
//
//   - value_sort<V...>: sorts NTTP values by type_label<decltype(V)>,
//     producing sorted_values<V...> in canonical order.
//
//   - type_sort<T...>: sorts types by a caller-provided label extraction,
//     producing sorted_types<T...> in canonical order.
//
// These are generic containers — tag.h unwraps sorted_values into tag_storage,
// and axes.h unwraps sorted_types into axes_storage.
//
#ifndef DISPATCH_DISPATCH_LABEL_SORTED_H
#define DISPATCH_DISPATCH_LABEL_SORTED_H

#include "dispatch/label.h"

namespace dispatch {

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
// The label extractor is a struct template with a static consteval value()
// member that returns the fixed_label for a given type.

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

} // namespace dispatch

#endif // DISPATCH_DISPATCH_LABEL_SORTED_H
