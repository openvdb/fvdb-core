// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_DISPATCH_VALUES_H
#define FVDB_DETAIL_DISPATCH_VALUES_H

#include <array>
#include <optional>
#include <tuple>
#include <type_traits>

namespace fvdb {
namespace dispatch {

namespace {

// Helper trait: compares two values, returning true only if their decayed
// types match AND the values are equal.
template <typename T, typename U> struct type_and_value_equal_impl {
    static constexpr bool
    compare(T const &, U const &) {
        return false;
    }
};

template <typename T> struct type_and_value_equal_impl<T, T> {
    static constexpr bool
    compare(T const &a, T const &b) {
        return a == b;
    }
};

// Wrapper that decays types before comparing
template <typename T, typename U>
struct type_and_value_equal : type_and_value_equal_impl<std::decay_t<T>, std::decay_t<U>> {};

} // namespace

template <auto... values> struct AnyTypeValuePack {
    static constexpr auto value_tuple = std::make_tuple(values...);
    static constexpr auto size        = sizeof...(values);

    // -------------------------------------------------------------------------
    // MEMBERSHIP
    // -------------------------------------------------------------------------
    // Check if value v is in the set of values spanned by this axis (constexpr)
    static constexpr bool
    contains_value(auto v) {
        return (type_and_value_equal<decltype(values), decltype(v)>::compare(values, v) || ...);
    }

    // Returns the index of the first value matching v (considering only values
    // of the same decayed type). Returns std::nullopt if not found.
    static constexpr std::optional<size_t>
    first_index_of_value(auto v) {
        std::optional<size_t> result = std::nullopt;
        auto check_at_index          = [&]<size_t I>() {
            if (!result) {
                auto const &elem = std::get<I>(value_tuple);
                if (type_and_value_equal<decltype(elem), decltype(v)>::compare(elem, v)) {
                    result = I;
                }
            }
        };
        [&]<size_t... Is>(std::index_sequence<Is...>) {
            (check_at_index.template operator()<Is>(), ...);
        }(std::make_index_sequence<size>{});
        return result;
    }

    // Return the value at index, keeping in mind they may be separate types.
    static constexpr auto
    value_at_index(size_t idx) {
        return std::get<idx>(value_tuple);
    }
};

// SameTypeValuePack: like AnyTypeValuePack, but enforces that all values share
// the same type. Values do NOT need to be unique.
template <auto... values> struct SameTypeValuePack;

template <> struct SameTypeValuePack<> : AnyTypeValuePack<> {
    using value_type = void;
};

template <auto... values> struct SameTypeValuePack<values...> : AnyTypeValuePack<values...> {
    using base       = AnyTypeValuePack<values...>;
    using value_type = std::common_type_t<std::decay_t<decltype(values)>...>;

    // Enforce uniform typing via static_assert
    static_assert(sizeof...(values) > 0 &&
                      (std::is_same_v<value_type, std::decay_t<decltype(values)>> && ...),
                  "All values in SameTypeValuePack must have the same type");

    // Returns the index of the first value matching v (considering only values
    // of the same decayed type). Returns std::nullopt if not found.
    static constexpr std::optional<size_t>
    first_index_of_value(value_type v) {
        return base::first_index_of_value(v);
    }

    static value_type
    value_at_index(size_t idx) {
        return base::value_at_index(idx);
    }
};

// Helper to check uniqueness at compile time
namespace {

template <auto... values>
constexpr bool
all_values_unique() {
    if constexpr (sizeof...(values) <= 1) {
        return true;
    } else {
        constexpr auto arr = std::array{values...};
        for (size_t i = 0; i < arr.size(); ++i) {
            for (size_t j = i + 1; j < arr.size(); ++j) {
                if (arr[i] == arr[j]) {
                    return false;
                }
            }
        }
        return true;
    }
}

} // namespace

// SameTypeUniqueValuePack: like SameTypeValuePack, but also enforces that all
// values are unique. Provides index_of_value instead of first_index_of_value.
template <auto... values> struct SameTypeUniqueValuePack : SameTypeValuePack<values...> {
    using base       = SameTypeValuePack<values...>;
    using value_type = typename base::value_type;

    static_assert(all_values_unique<values...>(),
                  "All values in SameTypeUniqueValuePack must be unique");

    // Since values are unique, there's exactly one index per value.
    static constexpr std::optional<size_t>
    index_of_value(value_type v) {
        return base::first_index_of_value(v);
    }
};

// UniqueIntegerPack: like SameTypeUniqueValuePack, but for integer values.
template <int... Values> using UniqueIntegerPack = SameTypeUniqueValuePack<Values...>;

} // namespace dispatch
} // namespace fvdb

#endif // FVDB_DETAIL_DISPATCH_VALUES_H
