// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <fvdb/detail/dispatch/Values.h>

#include <gtest/gtest.h>

#include <cstddef>

namespace fvdb {
namespace dispatch {

// =============================================================================
// ValuesElement Tests
// =============================================================================

TEST(ValuesElement, ReturnsCorrectValueAtEachIndex) {
    EXPECT_EQ((ValuesElement_v<0, 10, 20, 30>()), 10);
    EXPECT_EQ((ValuesElement_v<1, 10, 20, 30>()), 20);
    EXPECT_EQ((ValuesElement_v<2, 10, 20, 30>()), 30);
    EXPECT_EQ((ValuesElement_v<1, -10, -20, -30>()), -20);
}

TEST(ValuesElement, WorksWithMixedTypesAndReturnsCorrectType) {
    // Values can have different types
    EXPECT_EQ((ValuesElement_v<0, 42, 'x', true>()), 42);
    EXPECT_EQ((ValuesElement_v<1, 42, 'x', true>()), 'x');
    EXPECT_EQ((ValuesElement_v<2, 42, 'x', true>()), true);

    // Type alias works correctly
    static_assert(std::is_same_v<ValuesElement_t<0, 42, 'x', true>, int>);
    static_assert(std::is_same_v<ValuesElement_t<1, 42, 'x', true>, char>);
    static_assert(std::is_same_v<ValuesElement_t<2, 42, 'x', true>, bool>);
}

TEST(ValuesElement, WorksWithEnumValues) {
    enum class Color { Red, Green, Blue };
    EXPECT_EQ((ValuesElement_v<1, Color::Red, Color::Green, Color::Blue>()), Color::Green);
}

// =============================================================================
// ValuesContain Tests
// =============================================================================

TEST(ValuesContain, FindsValuesInPack) {
    EXPECT_TRUE((ValuesContain_v<10, 10, 20, 30>()));
    EXPECT_TRUE((ValuesContain_v<20, 10, 20, 30>()));
    EXPECT_TRUE((ValuesContain_v<30, 10, 20, 30>()));
}

TEST(ValuesContain, ReturnsFalseForMissingValues) {
    EXPECT_FALSE((ValuesContain_v<99, 10, 20, 30>()));
    EXPECT_FALSE((ValuesContain_v<0>())); // Empty pack
}

TEST(ValuesContain, DistinguishesByTypeAndValue) {
    // Same numeric value but different types should not match
    EXPECT_TRUE((ValuesContain_v<'A', 'A', 'B', 'C'>()));
    EXPECT_FALSE((ValuesContain_v<65, 'A', 'B', 'C'>())); // 65 == 'A' but int != char
}

// =============================================================================
// ValuesDefiniteFirstIndex Tests
// =============================================================================

TEST(ValuesDefiniteFirstIndex, ReturnsCorrectIndexForEachPosition) {
    EXPECT_EQ((ValuesDefiniteFirstIndex_v<10, 10, 20, 30>()), 0u);
    EXPECT_EQ((ValuesDefiniteFirstIndex_v<20, 10, 20, 30>()), 1u);
    EXPECT_EQ((ValuesDefiniteFirstIndex_v<30, 10, 20, 30>()), 2u);
}

TEST(ValuesDefiniteFirstIndex, ReturnsFirstIndexForDuplicates) {
    EXPECT_EQ((ValuesDefiniteFirstIndex_v<20, 10, 20, 30, 20>()), 1u);
    EXPECT_EQ((ValuesDefiniteFirstIndex_v<'x', 'x', 'x', 'x'>()), 0u);
}

TEST(ValuesDefiniteFirstIndex, MatchesByTypeAndValue) {
    // Must match both type and value
    EXPECT_EQ((ValuesDefiniteFirstIndex_v<'B', 'A', 'B', 'C'>()), 1u);
}

// =============================================================================
// ValuesHead Tests
// =============================================================================

TEST(ValuesHead, ReturnsFirstValueAndCorrectType) {
    EXPECT_EQ((ValuesHead_v<10, 20, 30>()), 10);
    EXPECT_EQ((ValuesHead_v<'a', 'b', 'c'>()), 'a');
    EXPECT_EQ((ValuesHead_v<42>()), 42); // Single element

    static_assert(std::is_same_v<ValuesHead_t<42, 'x', true>, int>);
    static_assert(std::is_same_v<ValuesHead_t<'x', 42, true>, char>);
}

// =============================================================================
// ValuesUnique Tests
// =============================================================================

TEST(ValuesUnique, ReturnsTrueForUniqueValues) {
    EXPECT_TRUE((ValuesUnique_v<>()));           // Empty pack
    EXPECT_TRUE((ValuesUnique_v<42>()));         // Single element
    EXPECT_TRUE((ValuesUnique_v<10, 20, 30>())); // All unique
    EXPECT_TRUE((ValuesUnique_v<'a', 'b', 'c'>()));
}

TEST(ValuesUnique, ReturnsFalseForDuplicates) {
    EXPECT_FALSE((ValuesUnique_v<10, 10>()));
    EXPECT_FALSE((ValuesUnique_v<10, 20, 10>()));     // Duplicate at start and end
    EXPECT_FALSE((ValuesUnique_v<10, 20, 20, 30>())); // Adjacent duplicates
}

TEST(ValuesUnique, DifferentTypesAreUnique) {
    // 65 and 'A' have the same numeric value but different types
    EXPECT_TRUE((ValuesUnique_v<65, 'A'>()));
}

// =============================================================================
// PackSize Tests
// =============================================================================

TEST(PackSize, ReturnsCorrectSize) {
    EXPECT_EQ((PackSize_v<Values<>>()), 0u);
    EXPECT_EQ((PackSize_v<Values<42>>()), 1u);
    EXPECT_EQ((PackSize_v<Values<1, 2, 3>>()), 3u);
    EXPECT_EQ((PackSize_v<Values<1, 'x', true>>()), 3u); // Mixed types

    EXPECT_EQ(packSize(Values<>{}), 0u);
    EXPECT_EQ(packSize(Values<1, 2, 3>{}), 3u);
}

TEST(PackSize, WorksWithIntegerSequence) {
    EXPECT_EQ((PackSize_v<std::index_sequence<>>()), 0u);
    EXPECT_EQ((PackSize_v<std::index_sequence<42>>()), 1u);
    EXPECT_EQ((PackSize_v<std::index_sequence<1, 2, 3>>()), 3u);
    EXPECT_EQ((PackSize_v<std::integer_sequence<char, 'a', 'b'>>()), 2u);

    EXPECT_EQ(packSize(std::index_sequence<>{}), 0u);
    EXPECT_EQ(packSize(std::index_sequence<1, 2, 3>{}), 3u);
}

// =============================================================================
// PackElement Tests
// =============================================================================

TEST(PackElement, ReturnsValueAndTypeAtIndex) {
    using Pack = Values<10, 'x', true>;

    EXPECT_EQ((PackElement_v<Pack, 0>()), 10);
    EXPECT_EQ((PackElement_v<Pack, 1>()), 'x');
    EXPECT_EQ((PackElement_v<Pack, 2>()), true);

    static_assert(std::is_same_v<PackElement_t<Pack, 0>, int>);
    static_assert(std::is_same_v<PackElement_t<Pack, 1>, char>);
    static_assert(std::is_same_v<PackElement_t<Pack, 2>, bool>);
}

TEST(PackElement, WorksWithSameTypePack) {
    using Pack = Values<10, 20, 30>;

    EXPECT_EQ((PackElement_v<Pack, 0>()), 10);
    EXPECT_EQ((PackElement_v<Pack, 2>()), 30);
}

TEST(PackElement, WorksWithIntegerSequence) {
    using Seq = std::index_sequence<10, 20, 30>;

    EXPECT_EQ((PackElement_v<Seq, 0>()), 10u);
    EXPECT_EQ((PackElement_v<Seq, 1>()), 20u);
    EXPECT_EQ((PackElement_v<Seq, 2>()), 30u);

    static_assert(std::is_same_v<PackElement_t<Seq, 0>, size_t>);
}

TEST(PackElement, RuntimePackElementFunction) {
    using Pack = Values<10, 20, 30>;

    EXPECT_EQ(packElement(Pack{}, 0), 10);
    EXPECT_EQ(packElement(Pack{}, 1), 20);
    EXPECT_EQ(packElement(Pack{}, 2), 30);
}

// =============================================================================
// PackHeadValue Tests
// =============================================================================

TEST(PackHeadValue, ReturnsFirstValue) {
    EXPECT_EQ((PackHeadValue_v<Values<10, 20, 30>>()), 10);
    EXPECT_EQ((PackHeadValue_v<Values<'a', 'b', 'c'>>()), 'a');
    EXPECT_EQ((PackHeadValue_v<Values<42>>()), 42);

    static_assert(std::is_same_v<PackHeadValue_t<Values<42, 'x'>>, int>);
    static_assert(std::is_same_v<PackHeadValue_t<Values<'x', 42>>, char>);
}

TEST(PackHeadValue, WorksWithIntegerSequence) {
    EXPECT_EQ((PackHeadValue_v<std::index_sequence<10, 20, 30>>()), 10u);
    EXPECT_EQ((PackHeadValue_v<std::integer_sequence<char, 'a', 'b', 'c'>>()), 'a');

    static_assert(std::is_same_v<PackHeadValue_t<std::index_sequence<1, 2>>, size_t>);
    static_assert(std::is_same_v<PackHeadValue_t<std::integer_sequence<char, 'x'>>, char>);
}

// =============================================================================
// PackTail Tests
// =============================================================================

TEST(PackTail, ReturnsTailAsValues) {
    // PackTail always returns Values<...> regardless of input type
    static_assert(std::is_same_v<PackTail_t<Values<10, 20, 30>>, Values<20, 30>>);
    static_assert(std::is_same_v<PackTail_t<Values<42>>, Values<>>);

    auto tail = packTail(Values<10, 20, 30>{});
    static_assert(std::is_same_v<decltype(tail), Values<20, 30>>);
}

TEST(PackTail, IntegerSequenceReturnsValues) {
    // PackTail of integer_sequence also returns Values<...>
    static_assert(std::is_same_v<PackTail_t<std::index_sequence<10, 20, 30>>,
                                 Values<size_t{20}, size_t{30}>>);

    auto tail = packTail(std::index_sequence<10, 20, 30>{});
    static_assert(std::is_same_v<decltype(tail), Values<size_t{20}, size_t{30}>>);
}

// =============================================================================
// PackContains Tests
// =============================================================================

TEST(PackContains, CompileTimeContainsCheck) {
    using Pack = Values<10, 20, 30>;

    EXPECT_TRUE((PackContains_v<Pack, 10>()));
    EXPECT_TRUE((PackContains_v<Pack, 20>()));
    EXPECT_TRUE((PackContains_v<Pack, 30>()));
    EXPECT_FALSE((PackContains_v<Pack, 99>()));
}

TEST(PackContains, RuntimePackContainsFunction) {
    using Pack = Values<10, 20, 30>;

    EXPECT_TRUE(packContains(Pack{}, 10));
    EXPECT_TRUE(packContains(Pack{}, 30));
    EXPECT_FALSE(packContains(Pack{}, 99));
}

TEST(PackContains, DistinguishesByType) {
    using Pack = Values<'A', 'B', 'C'>;

    EXPECT_TRUE((PackContains_v<Pack, 'A'>()));
    EXPECT_FALSE((PackContains_v<Pack, 65>())); // 65 == 'A' but different type
}

TEST(PackContains, WorksWithIntegerSequence) {
    using Seq = std::index_sequence<10, 20, 30>;

    EXPECT_TRUE((PackContains_v<Seq, size_t{10}>()));
    EXPECT_TRUE((PackContains_v<Seq, size_t{20}>()));
    EXPECT_TRUE((PackContains_v<Seq, size_t{30}>()));
    EXPECT_FALSE((PackContains_v<Seq, size_t{99}>()));
    // Type must match - index_sequence uses size_t
    EXPECT_FALSE((PackContains_v<Seq, 10>())); // int != size_t
}

// =============================================================================
// PackDefiniteFirstIndex Tests
// =============================================================================

TEST(PackDefiniteFirstIndex, CompileTimeIndexLookup) {
    using Pack = Values<10, 20, 30>;

    EXPECT_EQ((PackDefiniteFirstIndex_v<Pack, 10>()), 0u);
    EXPECT_EQ((PackDefiniteFirstIndex_v<Pack, 20>()), 1u);
    EXPECT_EQ((PackDefiniteFirstIndex_v<Pack, 30>()), 2u);
}

TEST(PackDefiniteFirstIndex, ReturnsFirstIndexForDuplicates) {
    using Pack = Values<10, 20, 10, 30>;

    EXPECT_EQ((PackDefiniteFirstIndex_v<Pack, 10>()), 0u);
}

TEST(PackDefiniteFirstIndex, RuntimeDefiniteFirstIndex) {
    using Pack = Values<10, 20, 30>;

    EXPECT_EQ(packDefiniteFirstIndex(Pack{}, 10), 0u);
    EXPECT_EQ(packDefiniteFirstIndex(Pack{}, 20), 1u);
    EXPECT_EQ(packDefiniteFirstIndex(Pack{}, 30), 2u);
}

// =============================================================================
// PackDefiniteIndex Tests (UniqueValuePack only)
// =============================================================================

TEST(PackDefiniteIndex, CompileTimeIndexLookup) {
    using Pack = Values<10, 20, 30>;

    EXPECT_EQ((PackDefiniteIndex_v<Pack, 10>()), 0u);
    EXPECT_EQ((PackDefiniteIndex_v<Pack, 20>()), 1u);
    EXPECT_EQ((PackDefiniteIndex_v<Pack, 30>()), 2u);
}

TEST(PackDefiniteIndex, RuntimeIndexLookup) {
    using Pack = Values<10, 20, 30>;

    EXPECT_EQ(packDefiniteIndex(Pack{}, 10), 0u);
    EXPECT_EQ(packDefiniteIndex(Pack{}, 20), 1u);
    EXPECT_EQ(packDefiniteIndex(Pack{}, 30), 2u);
}

// =============================================================================
// packFirstIndex / packIndex Tests (optional return)
// =============================================================================

TEST(PackFirstIndex, ReturnsOptionalIndex) {
    using Pack = Values<10, 20, 30>;

    EXPECT_EQ(packFirstIndex(Pack{}, 10), std::optional<size_t>{0});
    EXPECT_EQ(packFirstIndex(Pack{}, 20), std::optional<size_t>{1});
    EXPECT_EQ(packFirstIndex(Pack{}, 99), std::nullopt);
}

TEST(PackFirstIndex, ReturnsFirstForDuplicates) {
    using Pack = Values<10, 20, 10, 30>;

    EXPECT_EQ(packFirstIndex(Pack{}, 10), std::optional<size_t>{0});
}

TEST(PackIndex, ReturnsOptionalIndexForUniquePack) {
    using Pack = Values<10, 20, 30>;

    EXPECT_EQ(packIndex(Pack{}, 10), std::optional<size_t>{0});
    EXPECT_EQ(packIndex(Pack{}, 20), std::optional<size_t>{1});
    EXPECT_EQ(packIndex(Pack{}, 99), std::nullopt);
}

// =============================================================================
// PackIsSubset Tests
// =============================================================================

TEST(PackIsSubset, EmptyPackCases) {
    using Empty    = Values<>;
    using NonEmpty = Values<1, 2, 3>;

    EXPECT_TRUE((PackIsSubset_v<Empty, Empty>()));
    EXPECT_TRUE((PackIsSubset_v<NonEmpty, Empty>()));  // Empty is subset of anything
    EXPECT_FALSE((PackIsSubset_v<Empty, NonEmpty>())); // Non-empty not subset of empty
}

TEST(PackIsSubset, SubsetChecks) {
    using Full      = Values<1, 2, 3, 4, 5>;
    using Subset    = Values<2, 4>;
    using NotSubset = Values<2, 99>;

    EXPECT_TRUE((PackIsSubset_v<Full, Full>()));       // Pack is subset of itself
    EXPECT_TRUE((PackIsSubset_v<Full, Subset>()));     // Proper subset
    EXPECT_FALSE((PackIsSubset_v<Full, NotSubset>())); // 99 not in Full
}

TEST(PackIsSubset, OrderDoesNotMatter) {
    using Pack     = Values<1, 2, 3>;
    using Reversed = Values<3, 2, 1>;

    EXPECT_TRUE((PackIsSubset_v<Pack, Reversed>()));
}

TEST(PackIsSubset, RuntimeFunction) {
    using Full      = Values<1, 2, 3, 4, 5>;
    using Subset    = Values<2, 4>;
    using NotSubset = Values<2, 99>;

    EXPECT_TRUE(packIsSubset(Full{}, Subset{}));
    EXPECT_FALSE(packIsSubset(Full{}, NotSubset{}));
}

// =============================================================================
// PackPrepend Tests
// =============================================================================

TEST(PackPrepend, PrependsToValuePack) {
    using Original  = Values<20, 30>;
    using Prepended = PackPrepend_t<Original, 10>;

    static_assert(std::is_same_v<Prepended, Values<10, 20, 30>>);
    EXPECT_EQ((PackSize_v<Prepended>()), 3u);
    EXPECT_EQ((PackElement_v<Prepended, 0>()), 10);
}

TEST(PackPrepend, PrependsToSameTypePack) {
    using Original  = Values<20, 30>;
    using Prepended = PackPrepend_t<Original, 10>;

    static_assert(std::is_same_v<Prepended, Values<10, 20, 30>>);
    static_assert(std::is_same_v<PackValueType_t<Prepended>, int>);
}

TEST(PackPrepend, WorksWithMixedTypes) {
    using Original  = Values<'x', true>;
    using Prepended = PackPrepend_t<Original, 42>;

    static_assert(std::is_same_v<Prepended, Values<42, 'x', true>>);
}

TEST(PackPrepend, PrependsToEmptyPack) {
    static_assert(std::is_same_v<PackPrepend_t<Values<>, 10>, Values<10>>);
}

TEST(PackPrepend, RuntimePackPrependFunction) {
    auto result1 = packPrepend<10>(Values<20, 30>{});
    static_assert(std::is_same_v<decltype(result1), Values<10, 20, 30>>);

    auto result2 = packPrepend<42>(Values<'x', true>{});
    static_assert(std::is_same_v<decltype(result2), Values<42, 'x', true>>);

    // Prepend to empty
    auto result3 = packPrepend<10>(Values<>{});
    static_assert(std::is_same_v<decltype(result3), Values<10>>);
}

// =============================================================================
// IndexSequencePrepend Tests
// =============================================================================

TEST(IndexSequencePrepend, PrependsToIndexSequence) {
    using Original  = std::index_sequence<1, 2, 3>;
    using Prepended = IndexSequencePrepend_t<Original, 0>;

    static_assert(std::is_same_v<Prepended, std::index_sequence<0, 1, 2, 3>>);
}

TEST(IndexSequencePrepend, PrependsToEmptySequence) {
    static_assert(
        std::is_same_v<IndexSequencePrepend_t<std::index_sequence<>, 42>, std::index_sequence<42>>);
}

TEST(IndexSequencePrepend, RuntimeFunction) {
    auto result1 = indexSequencePrepend<0>(std::index_sequence<1, 2, 3>{});
    static_assert(std::is_same_v<decltype(result1), std::index_sequence<0, 1, 2, 3>>);

    auto result2 = indexSequencePrepend<42>(std::index_sequence<>{});
    static_assert(std::is_same_v<decltype(result2), std::index_sequence<42>>);
}

// Non-pack types for negative testing
struct NotAPack {};

// =============================================================================
// ValuePack Concept Tests
// =============================================================================

TEST(ValuePackConcept, ValuesTypesSatisfyConcept) {
    // All Values<...> types should satisfy ValuePack
    static_assert(is_value_pack<Values<>>());
    static_assert(is_value_pack<Values<1>>());
    static_assert(is_value_pack<Values<1, 2, 3>>());
    static_assert(is_value_pack<Values<'a', 'b', 'c'>>());
    static_assert(is_value_pack<Values<1, 'x', true>>()); // Mixed types
}

TEST(ValuePackConcept, IntegerSequenceTypesSatisfyConcept) {
    // std::integer_sequence and std::index_sequence also satisfy ValuePack
    static_assert(is_value_pack<std::index_sequence<>>());
    static_assert(is_value_pack<std::index_sequence<0>>());
    static_assert(is_value_pack<std::index_sequence<0, 1, 2>>());
    static_assert(is_value_pack<std::integer_sequence<int, 1, 2, 3>>());
    static_assert(is_value_pack<std::integer_sequence<char, 'a', 'b', 'c'>>());
}

TEST(ValuePackConcept, NonPackTypesDoNotSatisfyConcept) {
    // Regular types should not satisfy ValuePack
    static_assert(!is_value_pack<int>());
    static_assert(!is_value_pack<double>());
    static_assert(!is_value_pack<NotAPack>());
    static_assert(!is_value_pack<std::tuple<int, char>>());
}

// =============================================================================
// NonEmptyValuePack / EmptyValuePack Concept Tests
// =============================================================================

TEST(EmptyValuePackConcept, EmptyPackSatisfiesEmptyNotNonEmpty) {
    using Empty = Values<>;

    static_assert(is_value_pack<Empty>());
    static_assert(is_empty_value_pack<Empty>());
    static_assert(!is_non_empty_value_pack<Empty>());
}

TEST(NonEmptyValuePackConcept, NonEmptyPackSatisfiesNonEmptyNotEmpty) {
    using Single   = Values<42>;
    using Multiple = Values<1, 2, 3>;
    using Mixed    = Values<1, 'x', true>;

    // Single element
    static_assert(is_value_pack<Single>());
    static_assert(is_non_empty_value_pack<Single>());
    static_assert(!is_empty_value_pack<Single>());

    // Multiple elements
    static_assert(is_value_pack<Multiple>());
    static_assert(is_non_empty_value_pack<Multiple>());
    static_assert(!is_empty_value_pack<Multiple>());

    // Mixed types
    static_assert(is_value_pack<Mixed>());
    static_assert(is_non_empty_value_pack<Mixed>());
    static_assert(!is_empty_value_pack<Mixed>());
}

TEST(EmptyNonEmptyValuePackConcept, NonPackTypesDoNotSatisfyEither) {
    static_assert(!is_empty_value_pack<int>());
    static_assert(!is_non_empty_value_pack<int>());
    static_assert(!is_empty_value_pack<NotAPack>());
    static_assert(!is_non_empty_value_pack<NotAPack>());
}

TEST(EmptyNonEmptyValuePackConcept, IntegerSequenceWorks) {
    static_assert(is_empty_value_pack<std::index_sequence<>>());
    static_assert(!is_non_empty_value_pack<std::index_sequence<>>());
    static_assert(!is_empty_value_pack<std::index_sequence<1, 2, 3>>());
    static_assert(is_non_empty_value_pack<std::index_sequence<1, 2, 3>>());
}

// =============================================================================
// SameTypeValuePack Concept Tests
// =============================================================================

TEST(SameTypeValuePackConcept, EmptyPackSatisfiesSameType) {
    // Empty pack vacuously has same type
    static_assert(is_same_type_value_pack<Values<>>());
}

TEST(SameTypeValuePackConcept, SingleElementSatisfiesSameType) {
    static_assert(is_same_type_value_pack<Values<42>>());
    static_assert(is_same_type_value_pack<Values<'x'>>());
    static_assert(is_same_type_value_pack<Values<true>>());
}

TEST(SameTypeValuePackConcept, UniformTypesSatisfySameType) {
    static_assert(is_same_type_value_pack<Values<1, 2, 3>>());
    static_assert(is_same_type_value_pack<Values<'a', 'b', 'c'>>());
    static_assert(is_same_type_value_pack<Values<true, false, true>>());
}

TEST(SameTypeValuePackConcept, MixedTypesDoNotSatisfySameType) {
    static_assert(!is_same_type_value_pack<Values<1, 'x'>>());
    static_assert(!is_same_type_value_pack<Values<1, 2, 'c'>>());
    static_assert(!is_same_type_value_pack<Values<true, 1>>());
    static_assert(!is_same_type_value_pack<Values<42, 'x', true>>());
}

TEST(SameTypeValuePackConcept, NonPackTypesDoNotSatisfy) {
    static_assert(!is_same_type_value_pack<int>());
    static_assert(!is_same_type_value_pack<NotAPack>());
}

TEST(SameTypeValuePackConcept, IntegerSequenceAlwaysSameType) {
    // std::integer_sequence always has same type by definition
    static_assert(is_same_type_value_pack<std::index_sequence<>>());
    static_assert(is_same_type_value_pack<std::index_sequence<1, 2, 3>>());
    static_assert(is_same_type_value_pack<std::integer_sequence<char, 'a', 'b', 'c'>>());
}

// =============================================================================
// UniqueValuePack Concept Tests
// =============================================================================

TEST(UniqueValuePackConcept, EmptyPackSatisfiesUnique) {
    // Empty pack vacuously has unique values
    static_assert(is_unique_value_pack<Values<>>());
}

TEST(UniqueValuePackConcept, SingleElementSatisfiesUnique) {
    static_assert(is_unique_value_pack<Values<42>>());
    static_assert(is_unique_value_pack<Values<'x'>>());
}

TEST(UniqueValuePackConcept, DistinctValuesSatisfyUnique) {
    static_assert(is_unique_value_pack<Values<1, 2, 3>>());
    static_assert(is_unique_value_pack<Values<'a', 'b', 'c'>>());
    static_assert(is_unique_value_pack<Values<1, 'x', true>>()); // Mixed types are unique
}

TEST(UniqueValuePackConcept, DuplicateValuesDoNotSatisfyUnique) {
    static_assert(!is_unique_value_pack<Values<1, 1>>());
    static_assert(!is_unique_value_pack<Values<1, 2, 1>>());
    static_assert(!is_unique_value_pack<Values<'a', 'b', 'a'>>());
    static_assert(!is_unique_value_pack<Values<1, 2, 3, 2>>());
}

TEST(UniqueValuePackConcept, SameNumericValueDifferentTypesAreUnique) {
    // 65 (int) and 'A' (char) have same numeric value but different types
    static_assert(is_unique_value_pack<Values<65, 'A'>>());
}

TEST(UniqueValuePackConcept, NonPackTypesDoNotSatisfy) {
    static_assert(!is_unique_value_pack<int>());
    static_assert(!is_unique_value_pack<NotAPack>());
}

TEST(UniqueValuePackConcept, IntegerSequenceUniqueness) {
    static_assert(is_unique_value_pack<std::index_sequence<>>());
    static_assert(is_unique_value_pack<std::index_sequence<1, 2, 3>>());
    static_assert(!is_unique_value_pack<std::index_sequence<1, 2, 1>>());
}

// =============================================================================
// SameTypeNonEmptyValuePack Concept Tests
// =============================================================================

TEST(SameTypeNonEmptyValuePackConcept, RequiresBothSameTypeAndNonEmpty) {
    // Empty pack: same type but not non-empty
    static_assert(!is_same_type_non_empty_value_pack<Values<>>());

    // Single element: both same type and non-empty
    static_assert(is_same_type_non_empty_value_pack<Values<42>>());

    // Multiple same type: satisfies both
    static_assert(is_same_type_non_empty_value_pack<Values<1, 2, 3>>());

    // Multiple mixed type: non-empty but not same type
    static_assert(!is_same_type_non_empty_value_pack<Values<1, 'x'>>());
}

TEST(SameTypeNonEmptyValuePackConcept, NonPackTypesDoNotSatisfy) {
    static_assert(!is_same_type_non_empty_value_pack<int>());
    static_assert(!is_same_type_non_empty_value_pack<NotAPack>());
}

// =============================================================================
// UniqueNonEmptyValuePack Concept Tests
// =============================================================================

TEST(UniqueNonEmptyValuePackConcept, RequiresBothUniqueAndNonEmpty) {
    // Empty pack: unique but not non-empty
    static_assert(!is_unique_non_empty_value_pack<Values<>>());

    // Single element: both unique and non-empty
    static_assert(is_unique_non_empty_value_pack<Values<42>>());

    // Multiple unique: satisfies both
    static_assert(is_unique_non_empty_value_pack<Values<1, 2, 3>>());

    // Multiple with duplicates: non-empty but not unique
    static_assert(!is_unique_non_empty_value_pack<Values<1, 1>>());
    static_assert(!is_unique_non_empty_value_pack<Values<1, 2, 1>>());
}

TEST(UniqueNonEmptyValuePackConcept, NonPackTypesDoNotSatisfy) {
    static_assert(!is_unique_non_empty_value_pack<int>());
    static_assert(!is_unique_non_empty_value_pack<NotAPack>());
}

// =============================================================================
// ValuePackContains Concept Tests
// =============================================================================

TEST(ValuePackContainsConcept, ContainedValuesSatisfy) {
    using Pack = Values<10, 20, 30>;

    static_assert(value_pack_contains<Pack, 10>());
    static_assert(value_pack_contains<Pack, 20>());
    static_assert(value_pack_contains<Pack, 30>());
}

TEST(ValuePackContainsConcept, MissingValuesDoNotSatisfy) {
    using Pack = Values<10, 20, 30>;

    static_assert(!value_pack_contains<Pack, 99>());
    static_assert(!value_pack_contains<Pack, 0>());
    static_assert(!value_pack_contains<Pack, -10>());
}

TEST(ValuePackContainsConcept, EmptyPackContainsNothing) {
    using Empty = Values<>;

    static_assert(!value_pack_contains<Empty, 0>());
    static_assert(!value_pack_contains<Empty, 42>());
}

TEST(ValuePackContainsConcept, TypeMustMatchExactly) {
    using CharPack = Values<'A', 'B', 'C'>;

    // 'A' is in the pack
    static_assert(value_pack_contains<CharPack, 'A'>());

    // 65 has the same numeric value as 'A', but different type (int vs char)
    static_assert(!value_pack_contains<CharPack, 65>());
}

TEST(ValuePackContainsConcept, WorksWithMixedTypePacks) {
    using Mixed = Values<42, 'x', true>;

    static_assert(value_pack_contains<Mixed, 42>());
    static_assert(value_pack_contains<Mixed, 'x'>());
    static_assert(value_pack_contains<Mixed, true>());

    static_assert(!value_pack_contains<Mixed, 43>());
    static_assert(!value_pack_contains<Mixed, 'y'>());
    static_assert(!value_pack_contains<Mixed, false>());
}

// =============================================================================
// Combined Concept Relationships Tests
// =============================================================================

TEST(ConceptRelationships, SameTypeUniqueNonEmptyPackSatisfiesAll) {
    // A pack that is same-type, unique, and non-empty should satisfy all relevant concepts
    using Pack = Values<1, 2, 3>;

    static_assert(is_value_pack<Pack>());
    static_assert(is_non_empty_value_pack<Pack>());
    static_assert(!is_empty_value_pack<Pack>());
    static_assert(is_same_type_value_pack<Pack>());
    static_assert(is_unique_value_pack<Pack>());
    static_assert(is_same_type_non_empty_value_pack<Pack>());
    static_assert(is_unique_non_empty_value_pack<Pack>());
}

TEST(ConceptRelationships, SameTypeWithDuplicatesNotUnique) {
    // Same type but with duplicates
    using Pack = Values<1, 2, 1>;

    static_assert(is_value_pack<Pack>());
    static_assert(is_non_empty_value_pack<Pack>());
    static_assert(is_same_type_value_pack<Pack>());
    static_assert(!is_unique_value_pack<Pack>());
    static_assert(is_same_type_non_empty_value_pack<Pack>());
    static_assert(!is_unique_non_empty_value_pack<Pack>());
}

TEST(ConceptRelationships, MixedTypeUniqueNotSameType) {
    // Mixed types but all unique
    using Pack = Values<42, 'x', true>;

    static_assert(is_value_pack<Pack>());
    static_assert(is_non_empty_value_pack<Pack>());
    static_assert(!is_same_type_value_pack<Pack>());
    static_assert(is_unique_value_pack<Pack>());
    static_assert(!is_same_type_non_empty_value_pack<Pack>());
    static_assert(is_unique_non_empty_value_pack<Pack>());
}

TEST(ConceptRelationships, EmptyPackEdgeCases) {
    using Empty = Values<>;

    static_assert(is_value_pack<Empty>());
    static_assert(!is_non_empty_value_pack<Empty>());
    static_assert(is_empty_value_pack<Empty>());
    // Empty pack vacuously satisfies same-type and unique
    static_assert(is_same_type_value_pack<Empty>());
    static_assert(is_unique_value_pack<Empty>());
    // But NOT the non-empty variants
    static_assert(!is_same_type_non_empty_value_pack<Empty>());
    static_assert(!is_unique_non_empty_value_pack<Empty>());
}

} // namespace dispatch
} // namespace fvdb
