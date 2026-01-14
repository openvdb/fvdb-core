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

} // namespace dispatch
} // namespace fvdb
