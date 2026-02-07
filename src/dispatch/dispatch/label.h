// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Label system for dispatch types. Provides:
//
//   - fixed_label<N>: compile-time string usable as NTTP, with lexicographic
//     comparison for canonical ordering.
//
//   - type_label<T>: trait associating a fixed_label with each axis value type.
//     Must be specialized for every type used in tags/axes. The label should be
//     globally unique (use dotted namespace convention, e.g. "dispatch.placement").
//
//   - named<Label, T>: structural wrapper for disambiguating multiple axes that
//     share the same underlying value type. The type_label specialization is
//     self-registering — it returns the Label template parameter directly.
//
// To register a new axis value type:
//
//     template <>
//     struct type_label<my_enum> {
//         static consteval auto value() { return fixed_label("my_ns.my_enum"); }
//     };
//
// To create a named axis wrapper:
//
//     consteval auto input_stype(auto v) {
//         return named<fixed_label("input_scalar_type"), decltype(v)>{v};
//     }
//
#ifndef DISPATCH_DISPATCH_LABEL_H
#define DISPATCH_DISPATCH_LABEL_H

#include "dispatch/consteval_types.h"

#include <cstddef>

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
// When multiple axes share the same underlying value type (e.g. two axes
// both using ScalarType), wrap the values with named<Label, T> to give
// each axis a distinct type. The Label is a fixed_label that serves as
// both the type discriminator and the type_label (self-registering).

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
// The label IS the template parameter, so no manual registration is needed.

template <fixed_label Label, typename T> struct type_label<named<Label, T>> {
    static consteval auto
    value() {
        return Label;
    }
};

} // namespace dispatch

#endif // DISPATCH_DISPATCH_LABEL_H
