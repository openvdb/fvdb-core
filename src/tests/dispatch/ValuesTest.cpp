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
    EXPECT_EQ((ValuesElement<0, 10, 20, 30>::value()), 10);
    EXPECT_EQ((ValuesElement<1, 10, 20, 30>::value()), 20);
    EXPECT_EQ((ValuesElement<2, 10, 20, 30>::value()), 30);
    EXPECT_EQ((ValuesElement<1, -10, -20, -30>::value()), -20);
}

TEST(ValuesElement, WorksWithMixedTypesAndReturnsCorrectType) {
    // Values can have different types
    EXPECT_EQ((ValuesElement<0, 42, 'x', true>::value()), 42);
    EXPECT_EQ((ValuesElement<1, 42, 'x', true>::value()), 'x');
    EXPECT_EQ((ValuesElement<2, 42, 'x', true>::value()), true);

    // Type alias works correctly
    static_assert(std::is_same_v<ValuesElement_t<0, 42, 'x', true>, int>);
    static_assert(std::is_same_v<ValuesElement_t<1, 42, 'x', true>, char>);
    static_assert(std::is_same_v<ValuesElement_t<2, 42, 'x', true>, bool>);
}

TEST(ValuesElement, WorksWithEnumValues) {
    enum class Color { Red, Green, Blue };
    EXPECT_EQ((ValuesElement<1, Color::Red, Color::Green, Color::Blue>::value()), Color::Green);
}

// =============================================================================
// ValuesContain Tests
// =============================================================================

TEST(ValuesContain, FindsValuesInPack) {
    EXPECT_TRUE((ValuesContain<10, 10, 20, 30>::value()));
    EXPECT_TRUE((ValuesContain<20, 10, 20, 30>::value()));
    EXPECT_TRUE((ValuesContain<30, 10, 20, 30>::value()));
}

TEST(ValuesContain, ReturnsFalseForMissingValues) {
    EXPECT_FALSE((ValuesContain<99, 10, 20, 30>::value()));
    EXPECT_FALSE((ValuesContain<0>::value())); // Empty pack
}

TEST(ValuesContain, DistinguishesByTypeAndValue) {
    // Same numeric value but different types should not match
    EXPECT_TRUE((ValuesContain<'A', 'A', 'B', 'C'>::value()));
    EXPECT_FALSE((ValuesContain<65, 'A', 'B', 'C'>::value())); // 65 == 'A' but int != char
}

// =============================================================================
// ValuesDefiniteFirstIndex Tests
// =============================================================================

TEST(ValuesDefiniteFirstIndex, ReturnsCorrectIndexForEachPosition) {
    EXPECT_EQ((ValuesDefiniteFirstIndex<10, 10, 20, 30>::value()), 0u);
    EXPECT_EQ((ValuesDefiniteFirstIndex<20, 10, 20, 30>::value()), 1u);
    EXPECT_EQ((ValuesDefiniteFirstIndex<30, 10, 20, 30>::value()), 2u);
}

TEST(ValuesDefiniteFirstIndex, ReturnsFirstIndexForDuplicates) {
    EXPECT_EQ((ValuesDefiniteFirstIndex<20, 10, 20, 30, 20>::value()), 1u);
    EXPECT_EQ((ValuesDefiniteFirstIndex<'x', 'x', 'x', 'x'>::value()), 0u);
}

TEST(ValuesDefiniteFirstIndex, MatchesByTypeAndValue) {
    // Must match both type and value
    EXPECT_EQ((ValuesDefiniteFirstIndex<'B', 'A', 'B', 'C'>::value()), 1u);
}

// =============================================================================
// ValuesHead Tests
// =============================================================================

TEST(ValuesHead, ReturnsFirstValueAndCorrectType) {
    EXPECT_EQ((ValuesHead<10, 20, 30>::value()), 10);
    EXPECT_EQ((ValuesHead<'a', 'b', 'c'>::value()), 'a');
    EXPECT_EQ((ValuesHead<42>::value()), 42); // Single element

    static_assert(std::is_same_v<typename ValuesHead<42, 'x', true>::type, int>);
    static_assert(std::is_same_v<typename ValuesHead<'x', 42, true>::type, char>);
}

// =============================================================================
// ValuesUnique Tests
// =============================================================================

TEST(ValuesUnique, ReturnsTrueForUniqueValues) {
    EXPECT_TRUE((ValuesUnique<>::value()));           // Empty pack
    EXPECT_TRUE((ValuesUnique<42>::value()));         // Single element
    EXPECT_TRUE((ValuesUnique<10, 20, 30>::value())); // All unique
    EXPECT_TRUE((ValuesUnique<'a', 'b', 'c'>::value()));
}

TEST(ValuesUnique, ReturnsFalseForDuplicates) {
    EXPECT_FALSE((ValuesUnique<10, 10>::value()));
    EXPECT_FALSE((ValuesUnique<10, 20, 10>::value()));     // Duplicate at start and end
    EXPECT_FALSE((ValuesUnique<10, 20, 20, 30>::value())); // Adjacent duplicates
}

TEST(ValuesUnique, DifferentTypesAreUnique) {
    // 65 and 'A' have the same numeric value but different types
    EXPECT_TRUE((ValuesUnique<65, 'A'>::value()));
}

// =============================================================================
// AnyTypeValuePack Tests
// =============================================================================

TEST(AnyTypeValuePack, CanHoldMixedTypes) {
    // AnyTypeValuePack allows values of different types
    using Pack = Values<42, 'x', true>;
    EXPECT_EQ(PackSize<Pack>::value(), 3u);
    EXPECT_EQ(packSize(Pack{}), 3u);
}

TEST(AnyTypeValuePack, EmptyPackHasSizeZero) {
    using EmptyPack = Values<>;
    EXPECT_EQ(PackSize<EmptyPack>::value(), 0u);
    EXPECT_EQ(packSize(EmptyPack{}), 0u);
}

// =============================================================================
// SameTypeValuePack Tests
// =============================================================================

TEST(SameTypeValuePack, EnforcesUniformTypeAndProvidesValueType) {
    using IntPack  = Values<10, 20, 30>;
    using CharPack = Values<'a', 'b', 'c'>;

    static_assert(std::is_same_v<PackValueType<IntPack>::type, int>);
    static_assert(std::is_same_v<PackValueType<CharPack>::type, char>);

    EXPECT_EQ(PackSize<IntPack>::value(), 3u);
}

TEST(SameTypeValuePack, AllowsDuplicateValues) {
    using Pack = Values<10, 10, 20, 10>;
    EXPECT_EQ(PackSize<Pack>::value(), 4u);
}

// =============================================================================
// SameTypeUniqueValuePack Tests
// =============================================================================

TEST(SameTypeUniqueValuePack, EnforcesUniformTypeAndUniqueness) {
    using Pack = Values<10, 20, 30>;

    static_assert(std::is_same_v<PackValueType<Pack>::type, int>);
    EXPECT_EQ(PackSize<Pack>::value(), 3u);
}

// =============================================================================
// PackElement Tests
// =============================================================================

TEST(PackElement, ReturnsValueAndTypeAtIndex) {
    using Pack = Values<10, 'x', true>;

    EXPECT_EQ((PackElement<Pack, 0>::value()), 10);
    EXPECT_EQ((PackElement<Pack, 1>::value()), 'x');
    EXPECT_EQ((PackElement<Pack, 2>::value()), true);

    static_assert(std::is_same_v<typename PackElement<Pack, 0>::type, int>);
    static_assert(std::is_same_v<typename PackElement<Pack, 1>::type, char>);
    static_assert(std::is_same_v<typename PackElement<Pack, 2>::type, bool>);
}

TEST(PackElement, WorksWithDerivedPackTypes) {
    using Pack = Values<10, 20, 30>;

    EXPECT_EQ((PackElement<Pack, 0>::value()), 10);
    EXPECT_EQ((PackElement<Pack, 2>::value()), 30);
}

TEST(PackElement, RuntimePackElementFunction) {
    using Pack = Values<10, 20, 30>;

    EXPECT_EQ(packElement(Pack{}, 0), 10);
    EXPECT_EQ(packElement(Pack{}, 1), 20);
    EXPECT_EQ(packElement(Pack{}, 2), 30);
}

// =============================================================================
// PackContains Tests
// =============================================================================

TEST(PackContains, CompileTimeContainsCheck) {
    using Pack = Values<10, 20, 30>;

    EXPECT_TRUE((PackContains<Pack, 10>::value()));
    EXPECT_TRUE((PackContains<Pack, 20>::value()));
    EXPECT_TRUE((PackContains<Pack, 30>::value()));
    EXPECT_FALSE((PackContains<Pack, 99>::value()));
}

TEST(PackContains, WorksWithDerivedPackTypes) {
    using Pack = Values<10, 20, 30>;

    EXPECT_TRUE((PackContains<Pack, 20>::value()));
    EXPECT_FALSE((PackContains<Pack, 99>::value()));
}

TEST(PackContains, RuntimePackContainsFunction) {
    using Pack = Values<10, 20, 30>;

    EXPECT_TRUE(packContains(Pack{}, 10));
    EXPECT_TRUE(packContains(Pack{}, 30));
    EXPECT_FALSE(packContains(Pack{}, 99));
}

TEST(PackContains, DistinguishesByType) {
    using Pack = Values<'A', 'B', 'C'>;

    EXPECT_TRUE((PackContains<Pack, 'A'>::value()));
    EXPECT_FALSE((PackContains<Pack, 65>::value())); // 65 == 'A' but different type
}

// =============================================================================
// PackDefiniteFirstIndex Tests
// =============================================================================

TEST(PackDefiniteFirstIndex, CompileTimeIndexLookup) {
    using Pack = Values<10, 20, 30>;

    EXPECT_EQ((PackDefiniteFirstIndex<Pack, 10>::value()), 0u);
    EXPECT_EQ((PackDefiniteFirstIndex<Pack, 20>::value()), 1u);
    EXPECT_EQ((PackDefiniteFirstIndex<Pack, 30>::value()), 2u);
}

TEST(PackDefiniteFirstIndex, ReturnsFirstIndexForDuplicates) {
    using Pack = Values<10, 20, 10, 30>;

    EXPECT_EQ((PackDefiniteFirstIndex<Pack, 10>::value()), 0u);
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

    EXPECT_EQ((PackDefiniteIndex<Pack, 10>::value()), 0u);
    EXPECT_EQ((PackDefiniteIndex<Pack, 20>::value()), 1u);
    EXPECT_EQ((PackDefiniteIndex<Pack, 30>::value()), 2u);
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

    EXPECT_TRUE((PackIsSubset<Empty, Empty>::value()));
    EXPECT_TRUE((PackIsSubset<NonEmpty, Empty>::value()));  // Empty is subset of anything
    EXPECT_FALSE((PackIsSubset<Empty, NonEmpty>::value())); // Non-empty not subset of empty
}

TEST(PackIsSubset, SubsetChecks) {
    using Full      = Values<1, 2, 3, 4, 5>;
    using Subset    = Values<2, 4>;
    using NotSubset = Values<2, 99>;

    EXPECT_TRUE((PackIsSubset<Full, Full>::value()));       // Pack is subset of itself
    EXPECT_TRUE((PackIsSubset<Full, Subset>::value()));     // Proper subset
    EXPECT_FALSE((PackIsSubset<Full, NotSubset>::value())); // 99 not in Full
}

TEST(PackIsSubset, OrderDoesNotMatter) {
    using Pack     = Values<1, 2, 3>;
    using Reversed = Values<3, 2, 1>;

    EXPECT_TRUE((PackIsSubset<Pack, Reversed>::value()));
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
    EXPECT_EQ(PackSize<Prepended>::value(), 3u);
    EXPECT_EQ((PackElement<Prepended, 0>::value()), 10);
}

TEST(PackPrepend, PrependsToSameTypePack) {
    using Original  = Values<20, 30>;
    using Prepended = PackPrepend_t<Original, 10>;

    static_assert(std::is_same_v<Prepended, Values<10, 20, 30>>);
    static_assert(std::is_same_v<PackValueType<Prepended>::type, int>);
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
struct FakePackNoTag {
    int value;
};

// =============================================================================
// ValuePack Concept Tests
// =============================================================================

TEST(ValuePackConcept, ValuesTypesSatisfyConcept) {
    // All Values<...> types should satisfy ValuePack
    static_assert(is_value_pack<Values<>>);
    static_assert(is_value_pack<Values<1>>);
    static_assert(is_value_pack<Values<1, 2, 3>>);
    static_assert(is_value_pack<Values<'a', 'b', 'c'>>);
    static_assert(is_value_pack<Values<1, 'x', true>>); // Mixed types
}

TEST(ValuePackConcept, NonPackTypesDoNotSatisfyConcept) {
    // Types without value_pack_tag should not satisfy ValuePack
    static_assert(!is_value_pack<int>);
    static_assert(!is_value_pack<double>);
    static_assert(!is_value_pack<NotAPack>);
    static_assert(!is_value_pack<FakePackNoTag>);
    static_assert(!is_value_pack<std::index_sequence<1, 2, 3>>);
}

// =============================================================================
// NonEmptyValuePack / EmptyValuePack Concept Tests
// =============================================================================

TEST(EmptyValuePackConcept, EmptyPackSatisfiesEmptyNotNonEmpty) {
    using Empty = Values<>;

    static_assert(is_value_pack<Empty>);
    static_assert(is_empty_value_pack<Empty>);
    static_assert(!is_non_empty_value_pack<Empty>);
}

TEST(NonEmptyValuePackConcept, NonEmptyPackSatisfiesNonEmptyNotEmpty) {
    using Single   = Values<42>;
    using Multiple = Values<1, 2, 3>;
    using Mixed    = Values<1, 'x', true>;

    // Single element
    static_assert(is_value_pack<Single>);
    static_assert(is_non_empty_value_pack<Single>);
    static_assert(!is_empty_value_pack<Single>);

    // Multiple elements
    static_assert(is_value_pack<Multiple>);
    static_assert(is_non_empty_value_pack<Multiple>);
    static_assert(!is_empty_value_pack<Multiple>);

    // Mixed types
    static_assert(is_value_pack<Mixed>);
    static_assert(is_non_empty_value_pack<Mixed>);
    static_assert(!is_empty_value_pack<Mixed>);
}

TEST(EmptyNonEmptyValuePackConcept, NonPackTypesDoNotSatisfyEither) {
    static_assert(!is_empty_value_pack<int>);
    static_assert(!is_non_empty_value_pack<int>);
    static_assert(!is_empty_value_pack<NotAPack>);
    static_assert(!is_non_empty_value_pack<NotAPack>);
}

// =============================================================================
// SameTypeValuePack Concept Tests
// =============================================================================

TEST(SameTypeValuePackConcept, EmptyPackSatisfiesSameType) {
    // Empty pack vacuously has same type
    static_assert(is_same_type_value_pack<Values<>>);
}

TEST(SameTypeValuePackConcept, SingleElementSatisfiesSameType) {
    static_assert(is_same_type_value_pack<Values<42>>);
    static_assert(is_same_type_value_pack<Values<'x'>>);
    static_assert(is_same_type_value_pack<Values<true>>);
}

TEST(SameTypeValuePackConcept, UniformTypesSatisfySameType) {
    static_assert(is_same_type_value_pack<Values<1, 2, 3>>);
    static_assert(is_same_type_value_pack<Values<'a', 'b', 'c'>>);
    static_assert(is_same_type_value_pack<Values<true, false, true>>);
}

TEST(SameTypeValuePackConcept, MixedTypesDoNotSatisfySameType) {
    static_assert(!is_same_type_value_pack<Values<1, 'x'>>);
    static_assert(!is_same_type_value_pack<Values<1, 2, 'c'>>);
    static_assert(!is_same_type_value_pack<Values<true, 1>>);
    static_assert(!is_same_type_value_pack<Values<42, 'x', true>>);
}

TEST(SameTypeValuePackConcept, NonPackTypesDoNotSatisfy) {
    static_assert(!is_same_type_value_pack<int>);
    static_assert(!is_same_type_value_pack<NotAPack>);
}

// =============================================================================
// UniqueValuePack Concept Tests
// =============================================================================

TEST(UniqueValuePackConcept, EmptyPackSatisfiesUnique) {
    // Empty pack vacuously has unique values
    static_assert(is_unique_value_pack<Values<>>);
}

TEST(UniqueValuePackConcept, SingleElementSatisfiesUnique) {
    static_assert(is_unique_value_pack<Values<42>>);
    static_assert(is_unique_value_pack<Values<'x'>>);
}

TEST(UniqueValuePackConcept, DistinctValuesSatisfyUnique) {
    static_assert(is_unique_value_pack<Values<1, 2, 3>>);
    static_assert(is_unique_value_pack<Values<'a', 'b', 'c'>>);
    static_assert(is_unique_value_pack<Values<1, 'x', true>>); // Mixed types are unique
}

TEST(UniqueValuePackConcept, DuplicateValuesDoNotSatisfyUnique) {
    static_assert(!is_unique_value_pack<Values<1, 1>>);
    static_assert(!is_unique_value_pack<Values<1, 2, 1>>);
    static_assert(!is_unique_value_pack<Values<'a', 'b', 'a'>>);
    static_assert(!is_unique_value_pack<Values<1, 2, 3, 2>>);
}

TEST(UniqueValuePackConcept, SameNumericValueDifferentTypesAreUnique) {
    // 65 (int) and 'A' (char) have same numeric value but different types
    static_assert(is_unique_value_pack<Values<65, 'A'>>);
}

TEST(UniqueValuePackConcept, NonPackTypesDoNotSatisfy) {
    static_assert(!is_unique_value_pack<int>);
    static_assert(!is_unique_value_pack<NotAPack>);
}

// =============================================================================
// SameTypeNonEmptyValuePack Concept Tests
// =============================================================================

TEST(SameTypeNonEmptyValuePackConcept, RequiresBothSameTypeAndNonEmpty) {
    // Empty pack: same type but not non-empty
    static_assert(!is_same_type_non_empty_value_pack<Values<>>);

    // Single element: both same type and non-empty
    static_assert(is_same_type_non_empty_value_pack<Values<42>>);

    // Multiple same type: satisfies both
    static_assert(is_same_type_non_empty_value_pack<Values<1, 2, 3>>);

    // Multiple mixed type: non-empty but not same type
    static_assert(!is_same_type_non_empty_value_pack<Values<1, 'x'>>);
}

TEST(SameTypeNonEmptyValuePackConcept, NonPackTypesDoNotSatisfy) {
    static_assert(!is_same_type_non_empty_value_pack<int>);
    static_assert(!is_same_type_non_empty_value_pack<NotAPack>);
}

// =============================================================================
// UniqueNonEmptyValuePack Concept Tests
// =============================================================================

TEST(UniqueNonEmptyValuePackConcept, RequiresBothUniqueAndNonEmpty) {
    // Empty pack: unique but not non-empty
    static_assert(!is_unique_non_empty_value_pack<Values<>>);

    // Single element: both unique and non-empty
    static_assert(is_unique_non_empty_value_pack<Values<42>>);

    // Multiple unique: satisfies both
    static_assert(is_unique_non_empty_value_pack<Values<1, 2, 3>>);

    // Multiple with duplicates: non-empty but not unique
    static_assert(!is_unique_non_empty_value_pack<Values<1, 1>>);
    static_assert(!is_unique_non_empty_value_pack<Values<1, 2, 1>>);
}

TEST(UniqueNonEmptyValuePackConcept, NonPackTypesDoNotSatisfy) {
    static_assert(!is_unique_non_empty_value_pack<int>);
    static_assert(!is_unique_non_empty_value_pack<NotAPack>);
}

// =============================================================================
// ValuePackContains Concept Tests
// =============================================================================

TEST(ValuePackContainsConcept, ContainedValuesSatisfy) {
    using Pack = Values<10, 20, 30>;

    static_assert(value_pack_contains<Pack, 10>);
    static_assert(value_pack_contains<Pack, 20>);
    static_assert(value_pack_contains<Pack, 30>);
}

TEST(ValuePackContainsConcept, MissingValuesDoNotSatisfy) {
    using Pack = Values<10, 20, 30>;

    static_assert(!value_pack_contains<Pack, 99>);
    static_assert(!value_pack_contains<Pack, 0>);
    static_assert(!value_pack_contains<Pack, -10>);
}

TEST(ValuePackContainsConcept, EmptyPackContainsNothing) {
    using Empty = Values<>;

    static_assert(!value_pack_contains<Empty, 0>);
    static_assert(!value_pack_contains<Empty, 42>);
}

TEST(ValuePackContainsConcept, TypeMustMatchExactly) {
    using CharPack = Values<'A', 'B', 'C'>;

    // 'A' is in the pack
    static_assert(value_pack_contains<CharPack, 'A'>);

    // 65 has the same numeric value as 'A', but different type (int vs char)
    static_assert(!value_pack_contains<CharPack, 65>);
}

TEST(ValuePackContainsConcept, WorksWithMixedTypePacks) {
    using Mixed = Values<42, 'x', true>;

    static_assert(value_pack_contains<Mixed, 42>);
    static_assert(value_pack_contains<Mixed, 'x'>);
    static_assert(value_pack_contains<Mixed, true>);

    static_assert(!value_pack_contains<Mixed, 43>);
    static_assert(!value_pack_contains<Mixed, 'y'>);
    static_assert(!value_pack_contains<Mixed, false>);
}

// =============================================================================
// Combined Concept Relationships Tests
// =============================================================================

TEST(ConceptRelationships, SameTypeUniqueNonEmptyPackSatisfiesAll) {
    // A pack that is same-type, unique, and non-empty should satisfy all relevant concepts
    using Pack = Values<1, 2, 3>;

    static_assert(is_value_pack<Pack>);
    static_assert(is_non_empty_value_pack<Pack>);
    static_assert(!is_empty_value_pack<Pack>);
    static_assert(is_same_type_value_pack<Pack>);
    static_assert(is_unique_value_pack<Pack>);
    static_assert(is_same_type_non_empty_value_pack<Pack>);
    static_assert(is_unique_non_empty_value_pack<Pack>);
}

TEST(ConceptRelationships, SameTypeWithDuplicatesNotUnique) {
    // Same type but with duplicates
    using Pack = Values<1, 2, 1>;

    static_assert(is_value_pack<Pack>);
    static_assert(is_non_empty_value_pack<Pack>);
    static_assert(is_same_type_value_pack<Pack>);
    static_assert(!is_unique_value_pack<Pack>);
    static_assert(is_same_type_non_empty_value_pack<Pack>);
    static_assert(!is_unique_non_empty_value_pack<Pack>);
}

TEST(ConceptRelationships, MixedTypeUniqueNotSameType) {
    // Mixed types but all unique
    using Pack = Values<42, 'x', true>;

    static_assert(is_value_pack<Pack>);
    static_assert(is_non_empty_value_pack<Pack>);
    static_assert(!is_same_type_value_pack<Pack>);
    static_assert(is_unique_value_pack<Pack>);
    static_assert(!is_same_type_non_empty_value_pack<Pack>);
    static_assert(is_unique_non_empty_value_pack<Pack>);
}

TEST(ConceptRelationships, EmptyPackEdgeCases) {
    using Empty = Values<>;

    static_assert(is_value_pack<Empty>);
    static_assert(!is_non_empty_value_pack<Empty>);
    static_assert(is_empty_value_pack<Empty>);
    // Empty pack vacuously satisfies same-type and unique
    static_assert(is_same_type_value_pack<Empty>);
    static_assert(is_unique_value_pack<Empty>);
    // But NOT the non-empty variants
    static_assert(!is_same_type_non_empty_value_pack<Empty>);
    static_assert(!is_unique_non_empty_value_pack<Empty>);
}

} // namespace dispatch
} // namespace fvdb
