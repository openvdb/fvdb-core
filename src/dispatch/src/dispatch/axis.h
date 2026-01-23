// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef DISPATCH_AXIS_H
#define DISPATCH_AXIS_H

#include "dispatch/traits.h"
#include "dispatch/types.h"

#include <cassert>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <type_traits>

namespace dispatch {

template <auto... values> struct is_value_based<axis<values...>> : consteval_true_type {};

template <auto... values> struct extent<axis<values...>> {
    static consteval size_t
    value() {
        return sizeof...(values);
    }
};

template <auto... values>
struct is_non_empty<axis<values...>> : consteval_bool_type<(sizeof...(values) > 0)> {};

// -----------------------------------------------------------------------------
// ValuesElement
// -----------------------------------------------------------------------------

// std::tuple_element<I, Tuple>::type retrieves the type of the I-th element in the tuple Tuple.
//     - I: The zero-based index of the element in the tuple.
//     - Tuple: The tuple type (e.g., std::tuple<int, float, double>) from which to select an
//     element type.

template <size_t I, typename T> struct value_element {
    static_assert(is_value_based_v<T>(), "value_element requires a value based type");
};

template <size_t I, auto V> struct axis_element<I, axis<V>> {
    using type = decltype(V);
    template <size_t I, auto... values> struct axis_element<I, axis<values...>> {
        template <size_t I> struct ValuesElement<I> {
            static_assert(I != I, "Can't get an element from an empty value pack");
        };

        template <auto HeadValue, auto... TailValues>
        struct ValuesElement<0, HeadValue, TailValues...> {
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

        template <size_t I, auto... values>
        consteval auto
        ValuesElement_v() {
            return ValuesElement<I, values...>::value();
        }

        // Curried version for use with pack_apply
        template <size_t I> struct ValuesElementAt {
            template <auto... values> struct apply : ValuesElement<I, values...> {};
        };

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

        template <auto testValue, auto... values>
        consteval bool
        ValuesContain_v() {
            return ValuesContain<testValue, values...>::value();
        }

        // Curried version for use with pack_apply
        template <auto testValue> struct ValuesContainValue {
            template <auto... values> struct apply : ValuesContain<testValue, values...> {};
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

        template <auto testValue, auto... values>
        consteval size_t
        ValuesDefiniteFirstIndex_v() {
            return ValuesDefiniteFirstIndex<testValue, values...>::value();
        }

        // Curried version for use with pack_apply
        template <auto testValue> struct ValuesDefiniteFirstIndexOf {
            template <auto... values>
            struct apply : ValuesDefiniteFirstIndex<testValue, values...> {};
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

        template <auto... values> using ValuesHead_t = typename ValuesHead<values...>::type;

        template <auto... values>
        consteval auto
        ValuesHead_v() {
            return ValuesHead<values...>::value();
        }

        // -----------------------------------------------------------------------------
        // Values Tail - returns Values<tail...> from a non-empty pack
        // -----------------------------------------------------------------------------
        template <auto... values> struct ValuesTail;

        template <auto headValue, auto... tailValues> struct ValuesTail<headValue, tailValues...> {
            using type = Values<tailValues...>;
            static consteval auto
            value() {
                return Values<tailValues...>{};
            }
        };

        template <auto... values> using ValuesTail_t = typename ValuesTail<values...>::type;

        template <auto... values>
        consteval auto
        ValuesTail_v() {
            return ValuesTail<values...>::value();
        }

        // -----------------------------------------------------------------------------
        // Values Same Type
        // -----------------------------------------------------------------------------
        template <auto... values> struct ValuesSameType;
        template <> struct ValuesSameType<> {
            static consteval bool
            value() {
                return true;
            }
        };

        template <auto headValue> struct ValuesSameType<headValue> {
            static consteval bool
            value() {
                return true;
            }
        };

        template <auto headValue, auto... tailValues>
        struct ValuesSameType<headValue, tailValues...> {
            static consteval bool
            value() {
                return (std::is_same_v<decltype(headValue), decltype(tailValues)> && ...);
            }
        };

        template <auto... values>
        consteval bool
        ValuesSameType_v() {
            return ValuesSameType<values...>::value();
        }

        // -----------------------------------------------------------------------------
        // Values Unique
        // -----------------------------------------------------------------------------

        // Helper to check if two values are different (type-safe, avoids bool-compare warning)
        template <auto A, auto B> struct ValuesDiffer {
            static consteval bool
            value() {
                if constexpr (!std::is_same_v<decltype(A), decltype(B)>) {
                    // Different types are always considered different
                    return true;
                } else {
                    return A != B;
                }
            }
        };

        template <auto A, auto B>
        consteval bool
        ValuesDiffer_v() {
            return ValuesDiffer<A, B>::value();
        }

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

        template <auto headValue, auto... tailValues>
        struct ValuesUnique<headValue, tailValues...> {
            static consteval bool
            value() {
                // Unique if head differs from all tail values, and tail is itself unique
                return (ValuesDiffer_v<headValue, tailValues>() && ...) &&
                       ValuesUnique<tailValues...>::value();
            }
        };

        template <auto... values>
        consteval bool
        ValuesUnique_v() {
            return ValuesUnique<values...>::value();
        }

        // -----------------------------------------------------------------------------
        // -----------------------------------------------------------------------------
        // PACKS - have an inheritance chain that allows for increased constraints
        // -----------------------------------------------------------------------------
        // -----------------------------------------------------------------------------

        // =============================================================================
        // is_value_pack trait - detects value pack types via partial specialization
        // =============================================================================

        template <typename T> struct is_value_pack_trait : std::false_type {};

        template <auto... Vs> struct is_value_pack_trait<Values<Vs...>> : std::true_type {};

        template <typename T, T... Vs>
        struct is_value_pack_trait<std::integer_sequence<T, Vs...>> : std::true_type {};

        // Value Pack Concept (The Category)
        template <typename T>
        concept ValuePack = is_value_pack_trait<T>::value;

        // =============================================================================
        // pack_apply - THE central mechanism for unpacking ValuePacks
        // =============================================================================
        // This is the ONLY place with pack-type-specific specializations.
        // All other Pack utilities should use pack_apply_t or pack_apply_v.

        template <typename Pack, template <auto...> typename Template> struct pack_apply_impl;

        template <auto... Vs, template <auto...> typename Template>
        struct pack_apply_impl<Values<Vs...>, Template> {
            using type = Template<Vs...>;
        };

        template <typename T, T... Vs, template <auto...> typename Template>
        struct pack_apply_impl<std::integer_sequence<T, Vs...>, Template> {
            using type = Template<Vs...>;
        };

        // Alias template for applying a template to a pack's values
        template <typename Pack, template <auto...> typename Template>
        using pack_apply_t = typename pack_apply_impl<Pack, Template>::type;

        // Consteval function for getting the value() from an applied template
        template <typename Pack, template <auto...> typename Template>
        consteval auto
        pack_apply_v() {
            return pack_apply_t<Pack, Template>::value();
        }

        // Legacy alias for backward compatibility
        template <ValuePack Pack, template <auto...> typename InspectionType>
        using PackInspection = pack_apply_impl<Pack, InspectionType>;

        template <ValuePack Pack> using PackSize = pack_apply_t<Pack, ValuesSize>;

        template <ValuePack Pack>
        consteval size_t
        PackSize_v() {
            return pack_apply_v<Pack, ValuesSize>();
        }

        template <ValuePack Pack>
        constexpr size_t
        packSize(Pack) {
            return PackSize_v<Pack>();
        }

        template <ValuePack Pack> using PackSameType = pack_apply_t<Pack, ValuesSameType>;

        template <ValuePack Pack>
        consteval bool
        PackSameType_v() {
            return pack_apply_v<Pack, ValuesSameType>();
        }

        template <ValuePack Pack> using PackUnique = pack_apply_t<Pack, ValuesUnique>;

        template <ValuePack Pack>
        consteval bool
        PackUnique_v() {
            return pack_apply_v<Pack, ValuesUnique>();
        }

        template <typename T>
        concept NonEmptyValuePack = ValuePack<T> && PackSize_v<T>() > 0;
        template <typename T>
        concept EmptyValuePack = ValuePack<T> && PackSize_v<T>() == 0;

        template <typename T>
        concept SameTypeValuePack = ValuePack<T> && PackSameType_v<T>();
        template <typename T>
        concept UniqueValuePack = ValuePack<T> && PackUnique_v<T>();

        template <typename T>
        concept SameTypeNonEmptyValuePack = SameTypeValuePack<T> && NonEmptyValuePack<T>;
        template <typename T>
        concept UniqueNonEmptyValuePack = UniqueValuePack<T> && NonEmptyValuePack<T>;

        // =============================================================================
        // Concept Test Helpers (consteval functions to avoid nvcc instantiation issues)
        // =============================================================================

        template <typename T>
        consteval bool
        is_value_pack() {
            return ValuePack<T>;
        }

        template <typename T>
        consteval bool
        is_non_empty_value_pack() {
            return NonEmptyValuePack<T>;
        }

        template <typename T>
        consteval bool
        is_empty_value_pack() {
            return EmptyValuePack<T>;
        }

        template <typename T>
        consteval bool
        is_same_type_value_pack() {
            return SameTypeValuePack<T>;
        }

        template <typename T>
        consteval bool
        is_unique_value_pack() {
            return UniqueValuePack<T>;
        }

        template <typename T>
        consteval bool
        is_same_type_non_empty_value_pack() {
            return SameTypeNonEmptyValuePack<T>;
        }

        template <typename T>
        consteval bool
        is_unique_non_empty_value_pack() {
            return UniqueNonEmptyValuePack<T>;
        }

        // Note: value_pack_contains is defined after ValuePackContains concept below

        // -----------------------------------------------------------------------------
        // -----------------------------------------------------------------------------
        // PACK DEDUCTIONS
        // -----------------------------------------------------------------------------
        // -----------------------------------------------------------------------------

        template <NonEmptyValuePack Pack> using PackHeadValue = pack_apply_t<Pack, ValuesHead>;

        template <NonEmptyValuePack Pack>
        using PackHeadValue_t = typename PackHeadValue<Pack>::type;

        template <NonEmptyValuePack Pack>
        consteval auto
        PackHeadValue_v() {
            return pack_apply_v<Pack, ValuesHead>();
        }

        template <NonEmptyValuePack Pack>
        constexpr auto
        packHeadValue(Pack) {
            return PackHeadValue_v<Pack>();
        }

        // PackTail always returns Values<...> regardless of input pack type
        template <NonEmptyValuePack Pack> using PackTail = pack_apply_t<Pack, ValuesTail>;

        template <NonEmptyValuePack Pack> using PackTail_t = typename PackTail<Pack>::type;

        template <NonEmptyValuePack Pack>
        constexpr auto
        packTail(Pack) {
            return pack_apply_v<Pack, ValuesTail>();
        }

        // -----------------------------------------------------------------------------
        // For value packs which have the same type, return the type.
        // -----------------------------------------------------------------------------
        template <SameTypeNonEmptyValuePack Pack> struct PackValueType {
            using type = typename PackHeadValue<Pack>::type;
        };

        template <SameTypeNonEmptyValuePack Pack>
        using PackValueType_t = typename PackValueType<Pack>::type;

        // -----------------------------------------------------------------------------
        // index_sequence overload
        // -----------------------------------------------------------------------------
        template <SameTypeNonEmptyValuePack Pack>
            requires(std::is_same_v<typename PackValueType<Pack>::type, size_t>)
        struct is_index_sequence<Pack> : std::true_type {};

        // -----------------------------------------------------------------------------
        // PackElement
        // -----------------------------------------------------------------------------

        template <NonEmptyValuePack Pack, size_t I>
        using PackElement = pack_apply_t<Pack, ValuesElementAt<I>::template apply>;

        template <NonEmptyValuePack Pack, size_t I>
        using PackElement_t = typename PackElement<Pack, I>::type;

        template <NonEmptyValuePack Pack, size_t I>
        consteval auto
        PackElement_v() {
            return PackElement<Pack, I>::value();
        }

        // For the function to work, it needs to be same type.
        template <SameTypeValuePack Pack>
        constexpr auto
        packElement(Pack pack, size_t index) {
            if constexpr (NonEmptyValuePack<Pack>) {
                if (index == 0) {
                    return packHeadValue(pack);
                }
                if constexpr (NonEmptyValuePack<typename PackTail<Pack>::type>) {
                    assert(index > 0);
                    return packElement(packTail(pack), index - 1);
                }
            }
            throw std::logic_error("Index out of bounds");
        }

        // get<I>: compile-time indexed access to pack values (tuple-like interface)
        template <size_t I, NonEmptyValuePack Pack>
            requires(I < PackSize_v<Pack>())
        constexpr auto
        get(Pack) {
            return PackElement_v<Pack, I>();
        }

        // -----------------------------------------------------------------------------
        // PackContains
        // -----------------------------------------------------------------------------

        template <ValuePack Pack, auto testValue>
        using PackContains = pack_apply_t<Pack, ValuesContainValue<testValue>::template apply>;

        template <ValuePack Pack, auto testValue>
        consteval bool
        PackContains_v() {
            return PackContains<Pack, testValue>::value();
        }

        inline constexpr auto
        packContains(Values<>, auto testValue) {
            return false;
        }

        template <auto headValue, auto... tailValues>
        constexpr bool
        packContains(Values<headValue, tailValues...>, auto testValue) {
            using testType = std::decay_t<decltype(testValue)>;
            using headType = decltype(headValue);
            if constexpr (std::is_same_v<testType, headType>) {
                if (testValue == headValue) {
                    return true;
                }
            }
            return packContains(Values<tailValues...>{}, testValue);
        }

        template <typename Pack, auto testValue>
        concept ValuePackContains = ValuePack<Pack> && PackContains_v<Pack, testValue>();

        template <typename Pack, auto testValue>
        consteval bool
        value_pack_contains() {
            return ValuePackContains<Pack, testValue>;
        }

        // -----------------------------------------------------------------------------
        // PackDefiniteFirstIndex
        // -----------------------------------------------------------------------------

        template <typename Pack, auto testValue>
            requires ValuePackContains<Pack, testValue>
        using PackDefiniteFirstIndex =
            pack_apply_t<Pack, ValuesDefiniteFirstIndexOf<testValue>::template apply>;

        template <typename Pack, auto testValue>
            requires ValuePackContains<Pack, testValue>
        consteval size_t
        PackDefiniteFirstIndex_v() {
            return PackDefiniteFirstIndex<Pack, testValue>::value();
        }

        template <auto headValue, auto... tailValues>
            requires NonEmptyValuePack<Values<headValue, tailValues...>>
        constexpr size_t
        packDefiniteFirstIndex(Values<headValue, tailValues...>, auto testValue) {
            using testType = std::decay_t<decltype(testValue)>;
            using headType = decltype(headValue);
            if constexpr (std::is_same_v<testType, headType>) {
                if (testValue == headValue) {
                    return 0;
                }
            }
            if constexpr (sizeof...(tailValues) > 0) {
                return 1 + packDefiniteFirstIndex(Values<tailValues...>{}, testValue);
            }
            throw std::logic_error("Value not found in pack");
        }

        // -----------------------------------------------------------------------------
        // PackDefiniteIndex
        // -----------------------------------------------------------------------------

        template <typename Pack, auto testValue>
            requires UniqueNonEmptyValuePack<Pack> && ValuePackContains<Pack, testValue>
        using PackDefiniteIndex =
            pack_apply_t<Pack, ValuesDefiniteFirstIndexOf<testValue>::template apply>;

        template <typename Pack, auto testValue>
            requires UniqueNonEmptyValuePack<Pack> && ValuePackContains<Pack, testValue>
        consteval size_t
        PackDefiniteIndex_v() {
            return PackDefiniteIndex<Pack, testValue>::value();
        }

        template <auto headValue, auto... tailValues>
            requires UniqueNonEmptyValuePack<Values<headValue, tailValues...>>
        constexpr size_t
        packDefiniteIndex(Values<headValue, tailValues...> pack, auto testValue) {
            return packDefiniteFirstIndex(pack, testValue);
        }

        // -----------------------------------------------------------------------------
        // Pack First Index, Index (optional)
        // -----------------------------------------------------------------------------

        // Runtime version.
        inline constexpr std::optional<size_t>
        packFirstIndex(Values<>, auto value) {
            return std::nullopt;
        }

        template <auto headValue, auto... tailValues>
        constexpr std::optional<size_t>
        packFirstIndex(Values<headValue, tailValues...>, auto testValue) {
            using testType = std::decay_t<decltype(testValue)>;
            using headType = decltype(headValue);
            if constexpr (std::is_same_v<testType, headType>) {
                if (testValue == headValue) {
                    return 0;
                }
            }
            auto tailResult = packFirstIndex(Values<tailValues...>{}, testValue);
            if (tailResult) {
                return 1 + *tailResult;
            }
            return std::nullopt;
        }

        inline constexpr std::optional<size_t>
        packIndex(Values<>, auto testValue) {
            return std::nullopt;
        }

        template <auto headValue, auto... tailValues>
            requires UniqueNonEmptyValuePack<Values<headValue, tailValues...>>
        constexpr std::optional<size_t>
        packIndex(Values<headValue, tailValues...> pack, auto testValue) {
            return packFirstIndex(pack, testValue);
        }

        // -----------------------------------------------------------------------------
        // SUBSET TRAITS
        // -----------------------------------------------------------------------------

        template <typename Pack, typename SubsetPack> struct PackIsSubset;

        // Empty subset packs are a subset of an empty value pack.
        template <> struct PackIsSubset<Values<>, Values<>> {
            static consteval bool
            value() {
                return true;
            }
        };

        // If the pack is empty, a non-empty pack is not a subset of it.
        template <auto... SubsetValues>
            requires NonEmptyValuePack<Values<SubsetValues...>>
        struct PackIsSubset<Values<>, Values<SubsetValues...>> {
            static consteval bool
            value() {
                return false;
            }
        };

        // If the subset pack is empty, it is a subset of any pack.
        template <auto... Vs>
            requires NonEmptyValuePack<Values<Vs...>>
        struct PackIsSubset<Values<Vs...>, Values<>> {
            static consteval bool
            value() {
                return true;
            }
        };

        // Subset values don't need to be in the same order.
        template <auto... Vs, auto... SubsetValues>
            requires NonEmptyValuePack<Values<Vs...>> && NonEmptyValuePack<Values<SubsetValues...>>
        struct PackIsSubset<Values<Vs...>, Values<SubsetValues...>> {
            static consteval bool
            value() {
                using PrimaryPack = Values<Vs...>;
                return (PackContains_v<PrimaryPack, SubsetValues>() && ... && true);
            }
        };

        template <typename Pack, typename SubsetPack>
        consteval bool
        PackIsSubset_v() {
            return PackIsSubset<Pack, SubsetPack>::value();
        }

        inline constexpr bool
        packIsSubset(Values<>, Values<>) {
            return true;
        }

        template <auto... Vs>
            requires NonEmptyValuePack<Values<Vs...>>
        constexpr bool
        packIsSubset(Values<Vs...>, Values<>) {
            return true;
        }

        template <auto... SubsetValues>
            requires NonEmptyValuePack<Values<SubsetValues...>>
        constexpr bool
        packIsSubset(Values<>, Values<SubsetValues...>) {
            return false;
        }

        template <auto... Vs, auto... SubsetValues>
            requires NonEmptyValuePack<Values<Vs...>> && NonEmptyValuePack<Values<SubsetValues...>>
        constexpr bool
        packIsSubset(Values<Vs...> pack, Values<SubsetValues...>) {
            return (packContains(pack, SubsetValues) && ... && true);
        }

        // -----------------------------------------------------------------------------
        // Pack Prepend
        // -----------------------------------------------------------------------------

        template <typename Pack, auto headValue> struct PackPrepend;

        template <auto headValue> struct PackPrepend<Values<>, headValue> {
            using type = Values<headValue>;
        };

        template <auto... Vs, auto headValue>
            requires NonEmptyValuePack<Values<Vs...>>
        struct PackPrepend<Values<Vs...>, headValue> {
            using type = Values<headValue, Vs...>;
        };

        template <typename Pack, auto headValue>
        using PackPrepend_t = typename PackPrepend<Pack, headValue>::type;

        template <auto headValue, auto... Vs>
        constexpr auto
        packPrepend(Values<Vs...>) {
            return Values<headValue, Vs...>{};
        }

        // -----------------------------------------------------------------------------
        // Index Sequence Prepend
        // -----------------------------------------------------------------------------

        template <typename Seq, size_t headIndex> struct IndexSequencePrepend;

        template <size_t headIndex> struct IndexSequencePrepend<std::index_sequence<>, headIndex> {
            using type = std::index_sequence<headIndex>;
        };

        template <size_t... Indices, size_t headIndex>
        struct IndexSequencePrepend<std::index_sequence<Indices...>, headIndex> {
            using type = std::index_sequence<headIndex, Indices...>;
        };

        template <typename Seq, size_t headIndex>
        using IndexSequencePrepend_t = typename IndexSequencePrepend<Seq, headIndex>::type;

        template <size_t headIndex, size_t... Indices>
        constexpr auto
        indexSequencePrepend(std::index_sequence<Indices...>) {
            return std::index_sequence<headIndex, Indices...>{};
        }

    } // namespace dispatch
} // namespace fvdb

#endif // FVDB_DETAIL_DISPATCH_VALUES_H
