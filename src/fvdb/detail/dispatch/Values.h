// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_DISPATCH_VALUES_H
#define FVDB_DETAIL_DISPATCH_VALUES_H

#include <cassert>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <type_traits>

namespace fvdb {
namespace dispatch {

// -----------------------------------------------------------------------------
// ValuesElement
// -----------------------------------------------------------------------------

template <size_t I, auto... Values> struct ValuesElement;

template <size_t I> struct ValuesElement<I> {
    static_assert(I != I, "Can't get an element from an empty value pack");
};

template <auto HeadValue, auto... TailValues> struct ValuesElement<0, HeadValue, TailValues...> {
    using type = decltype(HeadValue);
    static consteval type
    value() {
        return HeadValue;
    }
};

template <size_t I, auto HeadValue, auto... TailValues>
struct ValuesElement<I, HeadValue, TailValues...> {
    static_assert(I > 0, "Specialization for 0 failed to occur");
    static_assert(I <= sizeof...(TailValues), "Index out of bounds");
    using type = typename ValuesElement<I - 1, TailValues...>::type;
    static consteval type
    value() {
        return ValuesElement<I - 1, TailValues...>::value();
    }
};

template <size_t I, auto... values>
using ValuesElement_t = typename ValuesElement<I, values...>::type;

// -----------------------------------------------------------------------------
// ValuesContain
// -----------------------------------------------------------------------------
template <auto testValue, auto... values> struct ValuesContain {
    static consteval bool
    value() {
        return false;
    }
};

template <auto testValue, auto headValue, auto... tailValues>
struct ValuesContain<testValue, headValue, tailValues...> {
    static consteval bool
    value() {
        if constexpr (std::is_same_v<decltype(testValue), decltype(headValue)>) {
            if constexpr (testValue == headValue) {
                return true;
            }
        }
        return ValuesContain<testValue, tailValues...>::value();
    }
};

// -----------------------------------------------------------------------------
// ValuesDefiniteFirstIndex - for when we know that testValue is in the pack,
// always returns the first index that matches both type and value.
// -----------------------------------------------------------------------------
template <auto> inline constexpr bool always_false_v = false;

template <auto testValue, auto... values> struct ValuesDefiniteFirstIndex {
    // This is a dummy test to force the static assert, it always fails.
    static_assert(always_false_v<testValue>, "Value not found in pack");
    static consteval size_t
    value() {
        return 0;
    }
};

template <auto testValue, auto headValue, auto... tailValues>
struct ValuesDefiniteFirstIndex<testValue, headValue, tailValues...> {
    static consteval size_t
    value() {
        if constexpr (std::is_same_v<decltype(testValue), decltype(headValue)>) {
            if constexpr (testValue == headValue) {
                return 0;
            } else {
                return 1 + ValuesDefiniteFirstIndex<testValue, tailValues...>::value();
            }
        } else {
            return 1 + ValuesDefiniteFirstIndex<testValue, tailValues...>::value();
        }
    }
};

// -----------------------------------------------------------------------------
// Values Head
// -----------------------------------------------------------------------------
template <auto... values> struct ValuesHead;

template <auto headValue, auto... tailValues> struct ValuesHead<headValue, tailValues...> {
    using type = decltype(headValue);
    static consteval type
    value() {
        return headValue;
    }
};

// -----------------------------------------------------------------------------
// Values Unique
// -----------------------------------------------------------------------------
template <auto... Values> struct ValuesUnique;

template <> struct ValuesUnique<> {
    static consteval bool
    value() {
        return true;
    }
};

template <auto headValue> struct ValuesUnique<headValue> {
    static consteval bool
    value() {
        return true;
    }
};

template <auto headValue, auto... tailValues> struct ValuesUnique<headValue, tailValues...> {
    static consteval bool
    value() {
        // Unique if types differ OR values differ (must check type first to avoid implicit
        // conversion)
        return ((!std::is_same_v<decltype(headValue), decltype(tailValues)> ||
                 headValue != tailValues) &&
                ...) &&
               ValuesUnique<tailValues...>::value();
    }
};

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// PACKS - have an inheritance chain that allows for increased constraints
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// AnyTypeValuePack
// -----------------------------------------------------------------------------
template <auto... values> struct AnyTypeValuePack {};

// -----------------------------------------------------------------------------
// SameTypeValuePack
// -----------------------------------------------------------------------------

// SameTypeValuePack: like AnyTypeValuePack, but enforces that all values share
// the same type. Values do NOT need to be unique.
template <auto... values> struct SameTypeValuePack;

template <> struct SameTypeValuePack<> : AnyTypeValuePack<> {
    // No value_type defined for empty pack - there's no meaningful type.
    // Attempting to access ::value_type will produce a clear compile error.
};

template <auto value> struct SameTypeValuePack<value> : AnyTypeValuePack<value> {
    using value_type = decltype(value);
};

template <auto headValue, auto... tailValues>
struct SameTypeValuePack<headValue, tailValues...> : AnyTypeValuePack<headValue, tailValues...> {
    using value_type = decltype(headValue);
    static_assert((std::is_same_v<value_type, decltype(tailValues)> && ...),
                  "All values must have the same type");
};

// -----------------------------------------------------------------------------
// SameTypeUniqueValuePack
// -----------------------------------------------------------------------------

// SameTypeUniqueValuePack: like SameTypeValuePack, all values must have the
// same type and be unique.
template <auto... values> struct SameTypeUniqueValuePack;

template <> struct SameTypeUniqueValuePack<> : SameTypeValuePack<> {
    // No value_type defined for empty pack - there's no meaningful type.
    // Attempting to access ::value_type will produce a clear compile error.
};

template <auto value> struct SameTypeUniqueValuePack<value> : SameTypeValuePack<value> {
    using value_type = decltype(value);
};

template <auto headValue, auto... tailValues>
struct SameTypeUniqueValuePack<headValue, tailValues...>
    : SameTypeValuePack<headValue, tailValues...> {
    using value_type = decltype(headValue);
    static_assert(ValuesUnique<headValue, tailValues...>::value(), "All values must be unique");
};

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// PACK DEDUCTIONS
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// PackSize
// -----------------------------------------------------------------------------

template <typename Pack> struct PackSize;
template <auto... values> struct PackSize<AnyTypeValuePack<values...>> {
    static consteval size_t
    value() {
        return sizeof...(values);
    }
};

template <auto... values>
constexpr size_t
packSize(AnyTypeValuePack<values...>) {
    return sizeof...(values);
}

// -----------------------------------------------------------------------------
// PackElement
// -----------------------------------------------------------------------------

template <typename Pack, size_t I> struct PackElement;
template <auto... values, size_t I> struct PackElement<AnyTypeValuePack<values...>, I> {
    using type = ValuesElement_t<I, values...>;
    static consteval auto
    value() {
        return ValuesElement<I, values...>::value();
    }
};

template <typename Pack, size_t I> using PackElement_t = typename PackElement<Pack, I>::type;

// for same type packs, we can have a runtime version. This would never be
// a compile-time version, we have other tools for that.
[[noreturn]] inline void
packElement(AnyTypeValuePack<>, size_t index) {
    throw std::logic_error("Cannot get element from an empty pack");
}

template <auto headValue, auto... tailValues>
constexpr auto
packElement(SameTypeValuePack<headValue, tailValues...>, size_t index) -> decltype(headValue) {
    if (index == 0) {
        return headValue;
    }
    if constexpr (sizeof...(tailValues) > 0) {
        assert(index > 0);
        return packElement(SameTypeValuePack<tailValues...>{}, index - 1);
    }
    throw std::logic_error("Index out of bounds");
}

// -----------------------------------------------------------------------------
// PackContains
// -----------------------------------------------------------------------------

template <typename Pack, auto testValue> struct PackContains;

template <auto testValue> struct PackContains<AnyTypeValuePack<>, testValue> {
    static consteval bool
    value() {
        return false;
    }
};

template <auto... values, auto testValue>
struct PackContains<AnyTypeValuePack<values...>, testValue> {
    static consteval bool
    value() {
        return ValuesContain<testValue, values...>::value();
    }
};

inline constexpr bool
packContains(AnyTypeValuePack<>, auto testValue) {
    return false;
}

template <auto headValue, auto... tailValues>
constexpr bool
packContains(AnyTypeValuePack<headValue, tailValues...>, auto testValue) {
    using testType = std::decay_t<decltype(testValue)>;
    using headType = decltype(headValue);
    if constexpr (std::is_same_v<testType, headType>) {
        if (testValue == headValue) {
            return true;
        }
    }
    return packContains(AnyTypeValuePack<tailValues...>{}, testValue);
}

// -----------------------------------------------------------------------------
// PackDefiniteFirstIndex
// -----------------------------------------------------------------------------

template <typename Pack, auto testValue> struct PackDefiniteFirstIndex;
template <auto... values, auto testValue>
struct PackDefiniteFirstIndex<AnyTypeValuePack<values...>, testValue> {
    static consteval size_t
    value() {
        return ValuesDefiniteFirstIndex<testValue, values...>::value();
    }
};

// Runtime version.
[[noreturn]] inline void
packDefiniteFirstIndex(AnyTypeValuePack<>, auto value) {
    throw std::logic_error("Cannot get definite first index from an empty pack");
}

template <auto headValue, auto... tailValues>
constexpr size_t
packDefiniteFirstIndex(AnyTypeValuePack<headValue, tailValues...>, auto testValue) {
    using testType = std::decay_t<decltype(testValue)>;
    using headType = decltype(headValue);
    if constexpr (std::is_same_v<testType, headType>) {
        if (testValue == headValue) {
            return 0;
        }
    }
    if constexpr (sizeof...(tailValues) > 0) {
        assert(index > 0);
        return packDefiniteFirstIndex(AnyTypeValuePack<tailValues...>{}, testValue);
    }
    throw std::logic_error("Index out of bounds");
}

// -----------------------------------------------------------------------------
// PackDefiniteIndex
// Definite index only defined for the SameTypeUniqueValuePack.
// -----------------------------------------------------------------------------

template <typename Pack, auto testValue> struct PackDefiniteIndex;
template <auto... values, auto testValue>
struct PackDefiniteIndex<SameTypeUniqueValuePack<values...>, testValue> {
    static consteval size_t
    value() {
        return ValuesDefiniteFirstIndex<testValue, values...>::value();
    }
};

[[noreturn]] inline void
packDefiniteIndex(SameTypeUniqueValuePack<>, auto testValue) {
    throw std::logic_error("Cannot get definite index from an empty pack");
}

template <auto headValue, auto... tailValues>
constexpr size_t
packDefiniteIndex(SameTypeUniqueValuePack<headValue, tailValues...>, auto testValue) {
    using testType = std::decay_t<decltype(testValue)>;
    using headType = decltype(headValue);
    static_assert(std::is_same_v<testType, headType>, "Value type must match pack type");
    if (testValue == headValue) {
        return 0;
    }
    if constexpr (sizeof...(tailValues) > 0) {
        return 1 + packDefiniteIndex(SameTypeUniqueValuePack<tailValues...>{}, testValue);
    }
    throw std::logic_error("Value not found in pack");
}

// -----------------------------------------------------------------------------
// Pack First Index, Index (optional)
// -----------------------------------------------------------------------------

// Runtime version.
inline constexpr std::optional<size_t>
packFirstIndex(AnyTypeValuePack<>, auto value) {
    return std::nullopt;
}

template <auto headValue, auto... tailValues>
constexpr std::optional<size_t>
packFirstIndex(AnyTypeValuePack<headValue, tailValues...>, auto testValue) {
    using testType = std::decay_t<decltype(testValue)>;
    using headType = decltype(headValue);
    if constexpr (std::is_same_v<testType, headType>) {
        if (testValue == headValue) {
            return 0;
        }
    }
    if constexpr (sizeof...(tailValues) > 0) {
        auto result = packFirstIndex(AnyTypeValuePack<tailValues...>{}, testValue);
        if (result.has_value()) {
            return result.value() + 1;
        }
    }
    return std::nullopt;
}

inline constexpr std::optional<size_t>
packIndex(SameTypeUniqueValuePack<>, auto testValue) {
    return std::nullopt;
}

template <auto headValue, auto... tailValues>
constexpr std::optional<size_t>
packIndex(SameTypeUniqueValuePack<headValue, tailValues...>, auto testValue) {
    using testType = std::decay_t<decltype(testValue)>;
    using headType = decltype(headValue);
    static_assert(std::is_same_v<testType, headType>, "Value type must match pack type");
    if (testValue == headValue) {
        return 0;
    }
    if constexpr (sizeof...(tailValues) > 0) {
        auto result = packIndex(SameTypeUniqueValuePack<tailValues...>{}, testValue);
        if (result.has_value()) {
            return result.value() + 1;
        }
    }
    return std::nullopt;
}

// -----------------------------------------------------------------------------
// SUBSET TRAITS
// -----------------------------------------------------------------------------

template <typename Pack, typename SubsetPack> struct PackIsSubset;

// Empty subset packs are a subset of an empty value pack.
template <> struct PackIsSubset<AnyTypeValuePack<>, AnyTypeValuePack<>> {
    static consteval bool
    value() {
        return true;
    }
};

// If the pack is empty, a non-empty pack is not a subset of it. The empty/empty case is handled
// above.
template <auto... SubsetValues>
struct PackIsSubset<AnyTypeValuePack<>, AnyTypeValuePack<SubsetValues...>> {
    static consteval bool
    value() {
        return false;
    }
};

// If the subset pack is empty, it is a subset of any pack.
template <auto... Values> struct PackIsSubset<AnyTypeValuePack<Values...>, AnyTypeValuePack<>> {
    static consteval bool
    value() {
        return true;
    }
};

// Pack is a subset of itself.
template <auto... Values>
struct PackIsSubset<AnyTypeValuePack<Values...>, AnyTypeValuePack<Values...>> {
    static consteval bool
    value() {
        return true;
    }
};

// Subset values don't need to be in the same order.
template <auto... Values, auto... SubsetValues>
struct PackIsSubset<AnyTypeValuePack<Values...>, AnyTypeValuePack<SubsetValues...>> {
    static consteval bool
    value() {
        using PrimaryPack = AnyTypeValuePack<Values...>;
        return (PackContains<PrimaryPack, SubsetValues>::value() && ... && true);
    }
};

inline constexpr bool
packIsSubset(AnyTypeValuePack<>, AnyTypeValuePack<>) {
    return true;
}

template <auto... Values>
constexpr bool
packIsSubset(AnyTypeValuePack<Values...>, AnyTypeValuePack<>) {
    return true;
}

template <auto... SubsetValues>
constexpr bool
packIsSubset(AnyTypeValuePack<>, AnyTypeValuePack<SubsetValues...>) {
    return false;
}

template <auto... Values>
constexpr bool
packIsSubset(AnyTypeValuePack<Values...>, AnyTypeValuePack<Values...>) {
    return true;
}

template <auto... Values, auto... SubsetValues>
constexpr bool
packIsSubset(AnyTypeValuePack<Values...> pack, AnyTypeValuePack<SubsetValues...>) {
    return (packContains(pack, SubsetValues) && ... && true);
}

} // namespace dispatch
} // namespace fvdb

#endif // FVDB_DETAIL_DISPATCH_VALUES_H
