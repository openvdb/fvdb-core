// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Self-normalizing tag: an unordered, uniquely-typed value set.
//
// tag<A, B> and tag<B, A> resolve to the same type (tag_storage<...> in
// canonical order), so concrete function signatures like
//
//     void foo(tag<torch::kCPU, contiguity::strided>, args...)
//
// work regardless of the order the user writes the values.
//
// Canonical order is determined by a per-type "tag_type_label" trait, which
// associates a fixed_label with each axis value type. The tag alias
// sorts its values lexicographically by tag_type_label to produce a unique
// tag_storage instantiation.
//
// To register a new axis value type for use in tags:
//
//     template <> struct tag_type_label<my_enum> {
//         static consteval auto value() { return fixed_label("my_namespace.my_enum"); }
//     };
//
#ifndef DISPATCH_DISPATCH_TAG_H
#define DISPATCH_DISPATCH_TAG_H

#include <compare>
#include <cstddef>
#include <type_traits>

namespace dispatch {

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

    consteval bool
    operator==(fixed_label const &) const = default;
};

// CTAD
template <std::size_t N> fixed_label(char const (&)[N]) -> fixed_label<N>;

// Cross-length lexicographic comparison.
// Returns negative if a < b, 0 if equal, positive if a > b.
template <std::size_t N, std::size_t M>
consteval int
compare_fixed_labels(fixed_label<N> const &a, fixed_label<M> const &b) {
    constexpr auto a_len = N - 1; // exclude null terminator
    constexpr auto b_len = M - 1;
    auto const min_len   = a_len < b_len ? a_len : b_len;
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
// tag_type_label: trait associating a fixed_label with each axis value type
//------------------------------------------------------------------------------
// Must be specialized for every type used as a tag value's decltype.
// The label should be globally unique (use dotted namespace convention).

template <typename T> struct tag_type_label {
    static_assert(sizeof(T) == 0,
                  "tag_type_label must be specialized for each axis value type. "
                  "Add: template <> struct tag_type_label<YourType> { "
                  "static consteval auto value() { return fixed_label(\"your.type.name\"); } };");
};

//------------------------------------------------------------------------------
// tag_type_label specializations for built-in dispatch enums
//------------------------------------------------------------------------------
// These are co-located here because the enums are defined in types.h within
// the dispatch namespace. When enums move to their own header, these
// specializations should follow.

// Forward declarations — the actual enums live in types.h for now.
// When they move, these can be removed.
enum class placement : int;
enum class determinism : int;
enum class contiguity : int;

template <> struct tag_type_label<placement> {
    static consteval auto
    value() {
        return fixed_label("dispatch.placement");
    }
};

template <> struct tag_type_label<determinism> {
    static consteval auto
    value() {
        return fixed_label("dispatch.determinism");
    }
};

template <> struct tag_type_label<contiguity> {
    static consteval auto
    value() {
        return fixed_label("dispatch.contiguity");
    }
};

//------------------------------------------------------------------------------
// tag_storage: the concrete ordered struct that tag<...> resolves to
//------------------------------------------------------------------------------

template <auto... V> struct tag_storage {};

//------------------------------------------------------------------------------
// is_tag_storage trait
//------------------------------------------------------------------------------

template <bool B> struct consteval_bool {
    static consteval bool
    value() {
        return B;
    }
};

template <typename T> struct is_tag_storage : consteval_bool<false> {};
template <auto... V> struct is_tag_storage<tag_storage<V...>> : consteval_bool<true> {};

template <typename T>
consteval bool
is_tag_storage_v() {
    return is_tag_storage<T>::value();
}

//------------------------------------------------------------------------------
// Compile-time sort of NTTP values by tag_type_label
//------------------------------------------------------------------------------
// Uses insertion sort on the parameter pack. The sort key for a value V
// is tag_type_label<decltype(V)>::value(), compared via compare_fixed_labels.

namespace detail {

// ---- value_less: compare two NTTP values by their tag_type_label ----

template <auto A, auto B>
consteval bool
value_label_less() {
    return compare_fixed_labels(tag_type_label<decltype(A)>::value(),
                                tag_type_label<decltype(B)>::value()) < 0;
}

// ---- insert: insert value V into an already-sorted tag_storage ----

// Base case: insert into empty -> single element
template <auto V> struct insert {
    using type = tag_storage<V>;
};

// Recursive case: insert V into tag_storage<Head, Tail...>
template <auto V, auto... Sorted> struct insert_into;

// Insert into empty (shouldn't normally be reached via the main path,
// but needed as a base case for the recursion)
template <auto V> struct insert_into<V> {
    using type = tag_storage<V>;
};

// V sorts before Head -> V goes first
template <auto V, auto Head, auto... Tail>
    requires(value_label_less<V, Head>())
struct insert_into<V, Head, Tail...> {
    using type = tag_storage<V, Head, Tail...>;
};

// V does not sort before Head -> Head stays, recurse into Tail
template <auto V, auto Head, auto... Tail>
    requires(!value_label_less<V, Head>())
struct insert_into<V, Head, Tail...> {
  private:
    using tail_inserted = typename insert_into<V, Tail...>::type;

    // Prepend Head to the result
    template <typename T> struct prepend_head;
    template <auto... Vs> struct prepend_head<tag_storage<Vs...>> {
        using type = tag_storage<Head, Vs...>;
    };

  public:
    using type = typename prepend_head<tail_inserted>::type;
};

// ---- sort: insertion sort on the full pack ----

// Base case: empty pack
template <auto... V> struct sort_values {
    using type = tag_storage<>;
};

// Base case: single element
template <auto V> struct sort_values<V> {
    using type = tag_storage<V>;
};

// Recursive case: sort the tail, then insert the head
template <auto Head, auto... Tail> struct sort_values<Head, Tail...> {
    using sorted_tail = typename sort_values<Tail...>::type;

    template <typename T> struct do_insert;
    template <auto... Sorted> struct do_insert<tag_storage<Sorted...>> {
        using type = typename insert_into<Head, Sorted...>::type;
    };

    using type = typename do_insert<sorted_tail>::type;
};

// ---- unique_value_types: check that all values have distinct types ----

template <typename T, typename... Rest>
consteval bool
unique_value_types(T, Rest...) {
    if constexpr (sizeof...(Rest) == 0) {
        return true;
    } else {
        return ((!std::is_same_v<T, Rest>) && ...) && unique_value_types(Rest{}...);
    }
}

} // namespace detail

//------------------------------------------------------------------------------
// tag: self-normalizing alias
//------------------------------------------------------------------------------
// tag<V...> sorts V... by tag_type_label into a canonical tag_storage<...>.
// tag<A, B> and tag<B, A> are the same type.

namespace detail {
template <auto... V> struct make_tag {
    static_assert(sizeof...(V) > 0, "tag must have at least one value");
    static_assert(unique_value_types(V...), "tag values must have unique types");
    using type = typename sort_values<V...>::type;
};
} // namespace detail

template <auto... V>
using tag = typename detail::make_tag<V...>::type;

} // namespace dispatch

#endif // DISPATCH_DISPATCH_TAG_H
