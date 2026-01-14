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
    using Pack = AnyTypeValuePack<42, 'x', true>;
    EXPECT_EQ(PackSize<Pack>::value(), 3u);
    EXPECT_EQ(packSize(Pack{}), 3u);
}

TEST(AnyTypeValuePack, EmptyPackHasSizeZero) {
    using EmptyPack = AnyTypeValuePack<>;
    EXPECT_EQ(PackSize<EmptyPack>::value(), 0u);
    EXPECT_EQ(packSize(EmptyPack{}), 0u);
}

// =============================================================================
// SameTypeValuePack Tests
// =============================================================================

TEST(SameTypeValuePack, EnforcesUniformTypeAndProvidesValueType) {
    using IntPack  = SameTypeValuePack<10, 20, 30>;
    using CharPack = SameTypeValuePack<'a', 'b', 'c'>;

    static_assert(std::is_same_v<IntPack::value_type, int>);
    static_assert(std::is_same_v<CharPack::value_type, char>);

    EXPECT_EQ(PackSize<IntPack>::value(), 3u);
}

TEST(SameTypeValuePack, AllowsDuplicateValues) {
    using Pack = SameTypeValuePack<10, 10, 20, 10>;
    EXPECT_EQ(PackSize<Pack>::value(), 4u);
}

TEST(SameTypeValuePack, InheritsFromAnyTypeValuePack) {
    static_assert(std::is_base_of_v<AnyTypeValuePack<1, 2, 3>, SameTypeValuePack<1, 2, 3>>);
}

// =============================================================================
// SameTypeUniqueValuePack Tests
// =============================================================================

TEST(SameTypeUniqueValuePack, EnforcesUniformTypeAndUniqueness) {
    using Pack = SameTypeUniqueValuePack<10, 20, 30>;

    static_assert(std::is_same_v<Pack::value_type, int>);
    EXPECT_EQ(PackSize<Pack>::value(), 3u);
}

TEST(SameTypeUniqueValuePack, InheritsFromSameTypeValuePack) {
    static_assert(std::is_base_of_v<SameTypeValuePack<1, 2, 3>, SameTypeUniqueValuePack<1, 2, 3>>);
    static_assert(std::is_base_of_v<AnyTypeValuePack<1, 2, 3>, SameTypeUniqueValuePack<1, 2, 3>>);
}

// =============================================================================
// PackElement Tests
// =============================================================================

TEST(PackElement, ReturnsValueAndTypeAtIndex) {
    using Pack = AnyTypeValuePack<10, 'x', true>;

    EXPECT_EQ((PackElement<Pack, 0>::value()), 10);
    EXPECT_EQ((PackElement<Pack, 1>::value()), 'x');
    EXPECT_EQ((PackElement<Pack, 2>::value()), true);

    static_assert(std::is_same_v<PackElement_t<Pack, 0>, int>);
    static_assert(std::is_same_v<PackElement_t<Pack, 1>, char>);
    static_assert(std::is_same_v<PackElement_t<Pack, 2>, bool>);
}

TEST(PackElement, WorksWithDerivedPackTypes) {
    using Pack = SameTypeValuePack<10, 20, 30>;

    EXPECT_EQ((PackElement<Pack, 0>::value()), 10);
    EXPECT_EQ((PackElement<Pack, 2>::value()), 30);
}

TEST(PackElement, RuntimePackElementFunction) {
    using Pack = SameTypeValuePack<10, 20, 30>;

    EXPECT_EQ(packElement(Pack{}, 0), 10);
    EXPECT_EQ(packElement(Pack{}, 1), 20);
    EXPECT_EQ(packElement(Pack{}, 2), 30);
}

// =============================================================================
// PackContains Tests
// =============================================================================

TEST(PackContains, CompileTimeContainsCheck) {
    using Pack = AnyTypeValuePack<10, 20, 30>;

    EXPECT_TRUE((PackContains<Pack, 10>::value()));
    EXPECT_TRUE((PackContains<Pack, 20>::value()));
    EXPECT_TRUE((PackContains<Pack, 30>::value()));
    EXPECT_FALSE((PackContains<Pack, 99>::value()));
}

TEST(PackContains, WorksWithDerivedPackTypes) {
    using Pack = SameTypeUniqueValuePack<10, 20, 30>;

    EXPECT_TRUE((PackContains<Pack, 20>::value()));
    EXPECT_FALSE((PackContains<Pack, 99>::value()));
}

TEST(PackContains, RuntimePackContainsFunction) {
    using Pack = AnyTypeValuePack<10, 20, 30>;

    EXPECT_TRUE(packContains(Pack{}, 10));
    EXPECT_TRUE(packContains(Pack{}, 30));
    EXPECT_FALSE(packContains(Pack{}, 99));
}

TEST(PackContains, DistinguishesByType) {
    using Pack = AnyTypeValuePack<'A', 'B', 'C'>;

    EXPECT_TRUE((PackContains<Pack, 'A'>::value()));
    EXPECT_FALSE((PackContains<Pack, 65>::value())); // 65 == 'A' but different type
}

// =============================================================================
// PackDefiniteFirstIndex Tests
// =============================================================================

TEST(PackDefiniteFirstIndex, CompileTimeIndexLookup) {
    using Pack = AnyTypeValuePack<10, 20, 30>;

    EXPECT_EQ((PackDefiniteFirstIndex<Pack, 10>::value()), 0u);
    EXPECT_EQ((PackDefiniteFirstIndex<Pack, 20>::value()), 1u);
    EXPECT_EQ((PackDefiniteFirstIndex<Pack, 30>::value()), 2u);
}

TEST(PackDefiniteFirstIndex, ReturnsFirstIndexForDuplicates) {
    using Pack = AnyTypeValuePack<10, 20, 10, 30>;

    EXPECT_EQ((PackDefiniteFirstIndex<Pack, 10>::value()), 0u);
}

TEST(PackDefiniteFirstIndex, WorksWithDerivedPackTypes) {
    using Pack = SameTypeValuePack<10, 20, 30>;

    EXPECT_EQ((PackDefiniteFirstIndex<Pack, 20>::value()), 1u);
}

// =============================================================================
// PackDefiniteIndex Tests (SameTypeUniqueValuePack only)
// =============================================================================

TEST(PackDefiniteIndex, CompileTimeIndexLookup) {
    using Pack = SameTypeUniqueValuePack<10, 20, 30>;

    EXPECT_EQ((PackDefiniteIndex<Pack, 10>::value()), 0u);
    EXPECT_EQ((PackDefiniteIndex<Pack, 20>::value()), 1u);
    EXPECT_EQ((PackDefiniteIndex<Pack, 30>::value()), 2u);
}

TEST(PackDefiniteIndex, RuntimeIndexLookup) {
    using Pack = SameTypeUniqueValuePack<10, 20, 30>;

    EXPECT_EQ(packDefiniteIndex(Pack{}, 10), 0u);
    EXPECT_EQ(packDefiniteIndex(Pack{}, 20), 1u);
    EXPECT_EQ(packDefiniteIndex(Pack{}, 30), 2u);
}

// =============================================================================
// packFirstIndex / packIndex Tests (optional return)
// =============================================================================

TEST(PackFirstIndex, ReturnsOptionalIndex) {
    using Pack = AnyTypeValuePack<10, 20, 30>;

    EXPECT_EQ(packFirstIndex(Pack{}, 10), std::optional<size_t>{0});
    EXPECT_EQ(packFirstIndex(Pack{}, 20), std::optional<size_t>{1});
    EXPECT_EQ(packFirstIndex(Pack{}, 99), std::nullopt);
}

TEST(PackFirstIndex, ReturnsFirstForDuplicates) {
    using Pack = AnyTypeValuePack<10, 20, 10, 30>;

    EXPECT_EQ(packFirstIndex(Pack{}, 10), std::optional<size_t>{0});
}

TEST(PackIndex, ReturnsOptionalIndexForUniquePack) {
    using Pack = SameTypeUniqueValuePack<10, 20, 30>;

    EXPECT_EQ(packIndex(Pack{}, 10), std::optional<size_t>{0});
    EXPECT_EQ(packIndex(Pack{}, 20), std::optional<size_t>{1});
    EXPECT_EQ(packIndex(Pack{}, 99), std::nullopt);
}

// =============================================================================
// PackIsSubset Tests
// =============================================================================

TEST(PackIsSubset, EmptyPackCases) {
    using Empty    = AnyTypeValuePack<>;
    using NonEmpty = AnyTypeValuePack<1, 2, 3>;

    EXPECT_TRUE((PackIsSubset<Empty, Empty>::value()));
    EXPECT_TRUE((PackIsSubset<NonEmpty, Empty>::value()));  // Empty is subset of anything
    EXPECT_FALSE((PackIsSubset<Empty, NonEmpty>::value())); // Non-empty not subset of empty
}

TEST(PackIsSubset, SubsetChecks) {
    using Full      = AnyTypeValuePack<1, 2, 3, 4, 5>;
    using Subset    = AnyTypeValuePack<2, 4>;
    using NotSubset = AnyTypeValuePack<2, 99>;

    EXPECT_TRUE((PackIsSubset<Full, Full>::value()));       // Pack is subset of itself
    EXPECT_TRUE((PackIsSubset<Full, Subset>::value()));     // Proper subset
    EXPECT_FALSE((PackIsSubset<Full, NotSubset>::value())); // 99 not in Full
}

TEST(PackIsSubset, OrderDoesNotMatter) {
    using Pack     = AnyTypeValuePack<1, 2, 3>;
    using Reversed = AnyTypeValuePack<3, 2, 1>;

    EXPECT_TRUE((PackIsSubset<Pack, Reversed>::value()));
}

TEST(PackIsSubset, WorksWithDerivedPackTypes) {
    using Full   = SameTypeValuePack<1, 2, 3, 4, 5>;
    using Subset = SameTypeUniqueValuePack<2, 4>;

    EXPECT_TRUE((PackIsSubset<Full, Subset>::value()));
}

TEST(PackIsSubset, RuntimeFunction) {
    using Full      = AnyTypeValuePack<1, 2, 3, 4, 5>;
    using Subset    = AnyTypeValuePack<2, 4>;
    using NotSubset = AnyTypeValuePack<2, 99>;

    EXPECT_TRUE(packIsSubset(Full{}, Subset{}));
    EXPECT_FALSE(packIsSubset(Full{}, NotSubset{}));
}

// =============================================================================
// PackPrepend Tests
// =============================================================================

TEST(PackPrepend, PrependsToAnyTypeValuePack) {
    using Original  = AnyTypeValuePack<20, 30>;
    using Prepended = PackPrepend_t<Original, 10>;

    static_assert(std::is_same_v<Prepended, AnyTypeValuePack<10, 20, 30>>);
    EXPECT_EQ(PackSize<Prepended>::value(), 3u);
    EXPECT_EQ((PackElement<Prepended, 0>::value()), 10);
}

TEST(PackPrepend, PrependsToSameTypeValuePack) {
    using Original  = SameTypeValuePack<20, 30>;
    using Prepended = PackPrepend_t<Original, 10>;

    static_assert(std::is_same_v<Prepended, SameTypeValuePack<10, 20, 30>>);
    static_assert(std::is_same_v<Prepended::value_type, int>);
}

TEST(PackPrepend, PrependsToSameTypeUniqueValuePack) {
    using Original  = SameTypeUniqueValuePack<20, 30>;
    using Prepended = PackPrepend_t<Original, 10>;

    static_assert(std::is_same_v<Prepended, SameTypeUniqueValuePack<10, 20, 30>>);
    static_assert(std::is_same_v<Prepended::value_type, int>);
}

TEST(PackPrepend, WorksWithMixedTypes) {
    using Original  = AnyTypeValuePack<'x', true>;
    using Prepended = PackPrepend_t<Original, 42>;

    static_assert(std::is_same_v<Prepended, AnyTypeValuePack<42, 'x', true>>);
}

TEST(PackPrepend, PrependsToEmptyPacks) {
    static_assert(std::is_same_v<PackPrepend_t<AnyTypeValuePack<>, 10>, AnyTypeValuePack<10>>);
    static_assert(std::is_same_v<PackPrepend_t<SameTypeValuePack<>, 10>, SameTypeValuePack<10>>);
    static_assert(
        std::is_same_v<PackPrepend_t<SameTypeUniqueValuePack<>, 10>, SameTypeUniqueValuePack<10>>);
}

TEST(PackPrepend, RuntimePackPrependFunction) {
    auto result1 = packPrepend<10>(SameTypeValuePack<20, 30>{});
    static_assert(std::is_same_v<decltype(result1), SameTypeValuePack<10, 20, 30>>);

    auto result2 = packPrepend<10>(SameTypeUniqueValuePack<20, 30>{});
    static_assert(std::is_same_v<decltype(result2), SameTypeUniqueValuePack<10, 20, 30>>);

    auto result3 = packPrepend<42>(AnyTypeValuePack<'x', true>{});
    static_assert(std::is_same_v<decltype(result3), AnyTypeValuePack<42, 'x', true>>);

    // Prepend to empty
    auto result4 = packPrepend<10>(SameTypeValuePack<>{});
    static_assert(std::is_same_v<decltype(result4), SameTypeValuePack<10>>);
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

} // namespace dispatch
} // namespace fvdb
