// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <fvdb/detail/dispatch/Values.h>

#include <gtest/gtest.h>

#include <cstddef>

namespace fvdb {
namespace dispatch {

// =============================================================================
// AnyTypeValuePack Tests
// =============================================================================

TEST(AnyTypeValuePack, EmptyPackHasSizeZero) {
    using EmptyPack = AnyTypeValuePack<>;
    EXPECT_EQ(EmptyPack::size, std::size_t{0});
}

TEST(AnyTypeValuePack, SingleValuePackHasSizeOne) {
    using SinglePack = AnyTypeValuePack<42>;
    EXPECT_EQ(SinglePack::size, std::size_t{1});
}

TEST(AnyTypeValuePack, MultipleValuesHaveCorrectSize) {
    using Pack3 = AnyTypeValuePack<1, 2, 3>;
    EXPECT_EQ(Pack3::size, std::size_t{3});

    using Pack5 = AnyTypeValuePack<10, 20, 30, 40, 50>;
    EXPECT_EQ(Pack5::size, std::size_t{5});
}

TEST(AnyTypeValuePack, MixedTypeValuesHaveCorrectSize) {
    // Different types: int, char, bool
    using MixedPack = AnyTypeValuePack<42, 'a', true>;
    EXPECT_EQ(MixedPack::size, std::size_t{3});
}

TEST(AnyTypeValuePack, ContainsValueReturnsTrueForPresentValue) {
    using Pack = AnyTypeValuePack<1, 2, 3, 4, 5>;
    EXPECT_TRUE(Pack::contains_value(1));
    EXPECT_TRUE(Pack::contains_value(3));
    EXPECT_TRUE(Pack::contains_value(5));
}

TEST(AnyTypeValuePack, ContainsValueReturnsFalseForAbsentValue) {
    using Pack = AnyTypeValuePack<1, 2, 3>;
    EXPECT_FALSE(Pack::contains_value(0));
    EXPECT_FALSE(Pack::contains_value(4));
    EXPECT_FALSE(Pack::contains_value(100));
}

TEST(AnyTypeValuePack, ContainsValueReturnsFalseForEmptyPack) {
    using EmptyPack = AnyTypeValuePack<>;
    EXPECT_FALSE(EmptyPack::contains_value(0));
    EXPECT_FALSE(EmptyPack::contains_value(42));
}

TEST(AnyTypeValuePack, ContainsValueRequiresTypeMatch) {
    // Pack contains int 65, not char 'A' (ASCII 65)
    using IntPack = AnyTypeValuePack<65>;
    EXPECT_TRUE(IntPack::contains_value(65));
    EXPECT_FALSE(IntPack::contains_value('A')); // Different type, should be false

    // Pack contains char 'A', not int 65
    using CharPack = AnyTypeValuePack<'A'>;
    EXPECT_TRUE(CharPack::contains_value('A'));
    EXPECT_FALSE(CharPack::contains_value(65)); // Different type, should be false
}

TEST(AnyTypeValuePack, ContainsValueWithMixedTypes) {
    using MixedPack = AnyTypeValuePack<42, 'x', true>;
    EXPECT_TRUE(MixedPack::contains_value(42));
    EXPECT_TRUE(MixedPack::contains_value('x'));
    EXPECT_TRUE(MixedPack::contains_value(true));

    EXPECT_FALSE(MixedPack::contains_value(43));
    EXPECT_FALSE(MixedPack::contains_value('y'));
    EXPECT_FALSE(MixedPack::contains_value(false));
}

TEST(AnyTypeValuePack, FirstIndexOfValueReturnsCorrectIndex) {
    using Pack = AnyTypeValuePack<10, 20, 30, 40>;
    EXPECT_EQ(Pack::first_index_of_value(10), std::size_t{0});
    EXPECT_EQ(Pack::first_index_of_value(20), std::size_t{1});
    EXPECT_EQ(Pack::first_index_of_value(30), std::size_t{2});
    EXPECT_EQ(Pack::first_index_of_value(40), std::size_t{3});
}

TEST(AnyTypeValuePack, FirstIndexOfValueReturnsNulloptForAbsentValue) {
    using Pack = AnyTypeValuePack<1, 2, 3>;
    EXPECT_FALSE(Pack::first_index_of_value(0).has_value());
    EXPECT_FALSE(Pack::first_index_of_value(4).has_value());
    EXPECT_FALSE(Pack::first_index_of_value(100).has_value());
}

TEST(AnyTypeValuePack, FirstIndexOfValueReturnsNulloptForEmptyPack) {
    using EmptyPack = AnyTypeValuePack<>;
    EXPECT_FALSE(EmptyPack::first_index_of_value(0).has_value());
    EXPECT_FALSE(EmptyPack::first_index_of_value(42).has_value());
}

TEST(AnyTypeValuePack, FirstIndexOfValueReturnsFirstOccurrence) {
    // Duplicate values - should return the first index
    using PackWithDupes = AnyTypeValuePack<1, 2, 1, 3, 1>;
    EXPECT_EQ(PackWithDupes::first_index_of_value(1),
              std::size_t{0}); // First occurrence at index 0
    EXPECT_EQ(PackWithDupes::first_index_of_value(2), std::size_t{1});
    EXPECT_EQ(PackWithDupes::first_index_of_value(3), std::size_t{3});
}

TEST(AnyTypeValuePack, FirstIndexOfValueRequiresTypeMatch) {
    using IntPack = AnyTypeValuePack<65>;
    EXPECT_TRUE(IntPack::first_index_of_value(65).has_value());
    EXPECT_FALSE(IntPack::first_index_of_value('A').has_value()); // Type mismatch
}

TEST(AnyTypeValuePack, FirstIndexOfValueWithMixedTypes) {
    using MixedPack = AnyTypeValuePack<42, 'x', true>;
    EXPECT_EQ(MixedPack::first_index_of_value(42), std::size_t{0});
    EXPECT_EQ(MixedPack::first_index_of_value('x'), std::size_t{1});
    EXPECT_EQ(MixedPack::first_index_of_value(true), std::size_t{2});
}

TEST(AnyTypeValuePack, ValueTupleContainsAllValues) {
    using Pack           = AnyTypeValuePack<10, 20, 30>;
    constexpr auto tuple = Pack::value_tuple;
    EXPECT_EQ(std::get<0>(tuple), 10);
    EXPECT_EQ(std::get<1>(tuple), 20);
    EXPECT_EQ(std::get<2>(tuple), 30);
}

// =============================================================================
// SameTypeValuePack Tests
// =============================================================================

TEST(SameTypeValuePack, EmptyPackHasSizeZero) {
    using EmptyPack = SameTypeValuePack<>;
    // Note: Empty packs intentionally do not define value_type - there's no
    // meaningful type when there are no values. Accessing ::value_type would
    // produce a compile error.
    EXPECT_EQ(EmptyPack::size, std::size_t{0});
}

TEST(SameTypeValuePack, IntPackHasIntValueType) {
    using IntPack = SameTypeValuePack<1, 2, 3>;
    static_assert(std::is_same_v<IntPack::value_type, int>);
}

TEST(SameTypeValuePack, CharPackHasCharValueType) {
    using CharPack = SameTypeValuePack<'a', 'b', 'c'>;
    static_assert(std::is_same_v<CharPack::value_type, char>);
}

TEST(SameTypeValuePack, BoolPackHasBoolValueType) {
    using BoolPack = SameTypeValuePack<true, false>;
    static_assert(std::is_same_v<BoolPack::value_type, bool>);
}

TEST(SameTypeValuePack, ContainsValueWorks) {
    using Pack = SameTypeValuePack<10, 20, 30>;
    EXPECT_TRUE(Pack::contains_value(10));
    EXPECT_TRUE(Pack::contains_value(20));
    EXPECT_TRUE(Pack::contains_value(30));
    EXPECT_FALSE(Pack::contains_value(15));
    EXPECT_FALSE(Pack::contains_value(0));
}

TEST(SameTypeValuePack, FirstIndexOfValueWorks) {
    using Pack = SameTypeValuePack<100, 200, 300>;
    EXPECT_EQ(Pack::first_index_of_value(100), std::size_t{0});
    EXPECT_EQ(Pack::first_index_of_value(200), std::size_t{1});
    EXPECT_EQ(Pack::first_index_of_value(300), std::size_t{2});
    EXPECT_FALSE(Pack::first_index_of_value(400).has_value());
}

TEST(SameTypeValuePack, AllowsDuplicateValues) {
    using PackWithDupes = SameTypeValuePack<5, 5, 5, 10>;
    EXPECT_EQ(PackWithDupes::size, std::size_t{4});
    EXPECT_TRUE(PackWithDupes::contains_value(5));
    EXPECT_TRUE(PackWithDupes::contains_value(10));
    EXPECT_EQ(PackWithDupes::first_index_of_value(5), std::size_t{0}); // Returns first occurrence
}

TEST(SameTypeValuePack, SingleValueWorks) {
    using SinglePack = SameTypeValuePack<42>;
    static_assert(std::is_same_v<SinglePack::value_type, int>);
    EXPECT_EQ(SinglePack::size, std::size_t{1});
    EXPECT_TRUE(SinglePack::contains_value(42));
    EXPECT_FALSE(SinglePack::contains_value(0));
}

// =============================================================================
// SameTypeUniqueValuePack Tests
// =============================================================================

TEST(SameTypeUniqueValuePack, HasCorrectValueType) {
    using IntPack = SameTypeUniqueValuePack<1, 2, 3>;
    static_assert(std::is_same_v<IntPack::value_type, int>);

    using CharPack = SameTypeUniqueValuePack<'a', 'b', 'c'>;
    static_assert(std::is_same_v<CharPack::value_type, char>);
}

TEST(SameTypeUniqueValuePack, IndexOfValueReturnsCorrectIndex) {
    using Pack = SameTypeUniqueValuePack<10, 20, 30, 40, 50>;
    EXPECT_EQ(Pack::index_of_value(10), std::size_t{0});
    EXPECT_EQ(Pack::index_of_value(20), std::size_t{1});
    EXPECT_EQ(Pack::index_of_value(30), std::size_t{2});
    EXPECT_EQ(Pack::index_of_value(40), std::size_t{3});
    EXPECT_EQ(Pack::index_of_value(50), std::size_t{4});
}

TEST(SameTypeUniqueValuePack, IndexOfValueReturnsNulloptForAbsentValue) {
    using Pack = SameTypeUniqueValuePack<1, 2, 3>;
    EXPECT_FALSE(Pack::index_of_value(0).has_value());
    EXPECT_FALSE(Pack::index_of_value(4).has_value());
    EXPECT_FALSE(Pack::index_of_value(-1).has_value());
}

TEST(SameTypeUniqueValuePack, ContainsValueWorks) {
    using Pack = SameTypeUniqueValuePack<5, 10, 15>;
    EXPECT_TRUE(Pack::contains_value(5));
    EXPECT_TRUE(Pack::contains_value(10));
    EXPECT_TRUE(Pack::contains_value(15));
    EXPECT_FALSE(Pack::contains_value(0));
    EXPECT_FALSE(Pack::contains_value(20));
}

TEST(SameTypeUniqueValuePack, SingleValueWorks) {
    using SinglePack = SameTypeUniqueValuePack<99>;
    EXPECT_EQ(SinglePack::size, std::size_t{1});
    EXPECT_EQ(SinglePack::index_of_value(99), std::size_t{0});
    EXPECT_FALSE(SinglePack::index_of_value(0).has_value());
}

TEST(SameTypeUniqueValuePack, NegativeValuesWork) {
    using Pack = SameTypeUniqueValuePack<-10, -5, 0, 5, 10>;
    EXPECT_EQ(Pack::index_of_value(-10), std::size_t{0});
    EXPECT_EQ(Pack::index_of_value(-5), std::size_t{1});
    EXPECT_EQ(Pack::index_of_value(0), std::size_t{2});
    EXPECT_EQ(Pack::index_of_value(5), std::size_t{3});
    EXPECT_EQ(Pack::index_of_value(10), std::size_t{4});
}

// =============================================================================
// UniqueIntegerPack Tests (Alias)
// =============================================================================

TEST(UniqueIntegerPack, IsAliasForSameTypeUniqueValuePack) {
    using IntPack1 = UniqueIntegerPack<1, 2, 3>;
    using IntPack2 = SameTypeUniqueValuePack<1, 2, 3>;
    static_assert(std::is_same_v<IntPack1, IntPack2>);
}

TEST(UniqueIntegerPack, BasicFunctionality) {
    using Pack = UniqueIntegerPack<100, 200, 300>;
    static_assert(std::is_same_v<Pack::value_type, int>);
    EXPECT_EQ(Pack::size, std::size_t{3});
    EXPECT_EQ(Pack::index_of_value(100), std::size_t{0});
    EXPECT_EQ(Pack::index_of_value(200), std::size_t{1});
    EXPECT_EQ(Pack::index_of_value(300), std::size_t{2});
}

// =============================================================================
// is_subset_of Tests
// =============================================================================

TEST(IsSubsetOf, EmptyPackIsSubsetOfEmptyPack) {
    using Empty = SameTypeUniqueValuePack<>;
    static_assert(is_subset_of_v<Empty, Empty>);
    EXPECT_TRUE((is_subset_of_v<Empty, Empty>));
}

TEST(IsSubsetOf, EmptyPackIsSubsetOfAnyPack) {
    using Empty    = SameTypeUniqueValuePack<>;
    using NonEmpty = SameTypeUniqueValuePack<1, 2, 3>;
    static_assert(is_subset_of_v<Empty, NonEmpty>);
    EXPECT_TRUE((is_subset_of_v<Empty, NonEmpty>));
}

TEST(IsSubsetOf, NonEmptyPackIsNotSubsetOfEmptyPack) {
    using Empty    = SameTypeUniqueValuePack<>;
    using NonEmpty = SameTypeUniqueValuePack<1>;
    static_assert(!is_subset_of_v<NonEmpty, Empty>);
    EXPECT_FALSE((is_subset_of_v<NonEmpty, Empty>));
}

TEST(IsSubsetOf, PackIsSubsetOfItself) {
    using Pack = SameTypeUniqueValuePack<1, 2, 3>;
    static_assert(is_subset_of_v<Pack, Pack>);
    EXPECT_TRUE((is_subset_of_v<Pack, Pack>));
}

TEST(IsSubsetOf, ProperSubsetIsRecognized) {
    using Subset   = SameTypeUniqueValuePack<1, 2>;
    using Superset = SameTypeUniqueValuePack<1, 2, 3, 4>;
    static_assert(is_subset_of_v<Subset, Superset>);
    EXPECT_TRUE((is_subset_of_v<Subset, Superset>));
}

TEST(IsSubsetOf, SingleElementSubset) {
    using Subset   = SameTypeUniqueValuePack<2>;
    using Superset = SameTypeUniqueValuePack<1, 2, 3>;
    static_assert(is_subset_of_v<Subset, Superset>);
    EXPECT_TRUE((is_subset_of_v<Subset, Superset>));
}

TEST(IsSubsetOf, SubsetWithDifferentOrder) {
    // Subset elements in different order than superset
    using Subset   = SameTypeUniqueValuePack<3, 1>;
    using Superset = SameTypeUniqueValuePack<1, 2, 3>;
    static_assert(is_subset_of_v<Subset, Superset>);
    EXPECT_TRUE((is_subset_of_v<Subset, Superset>));
}

TEST(IsSubsetOf, NonSubsetIsRecognized) {
    using NotSubset = SameTypeUniqueValuePack<1, 4>; // 4 is not in superset
    using Superset  = SameTypeUniqueValuePack<1, 2, 3>;
    static_assert(!is_subset_of_v<NotSubset, Superset>);
    EXPECT_FALSE((is_subset_of_v<NotSubset, Superset>));
}

TEST(IsSubsetOf, CompletelyDisjointPacksAreNotSubsets) {
    using Pack1 = SameTypeUniqueValuePack<1, 2, 3>;
    using Pack2 = SameTypeUniqueValuePack<4, 5, 6>;
    static_assert(!is_subset_of_v<Pack1, Pack2>);
    static_assert(!is_subset_of_v<Pack2, Pack1>);
    EXPECT_FALSE((is_subset_of_v<Pack1, Pack2>));
    EXPECT_FALSE((is_subset_of_v<Pack2, Pack1>));
}

TEST(IsSubsetOf, LargerPackIsNotSubsetOfSmaller) {
    using Larger  = SameTypeUniqueValuePack<1, 2, 3, 4>;
    using Smaller = SameTypeUniqueValuePack<1, 2>;
    static_assert(!is_subset_of_v<Larger, Smaller>);
    EXPECT_FALSE((is_subset_of_v<Larger, Smaller>));
}

TEST(IsSubsetOf, SingleElementPacks) {
    using Same1     = SameTypeUniqueValuePack<42>;
    using Same2     = SameTypeUniqueValuePack<42>;
    using Different = SameTypeUniqueValuePack<99>;

    static_assert(is_subset_of_v<Same1, Same2>);
    static_assert(!is_subset_of_v<Same1, Different>);
    EXPECT_TRUE((is_subset_of_v<Same1, Same2>));
    EXPECT_FALSE((is_subset_of_v<Same1, Different>));
}

// =============================================================================
// Compile-Time Constexpr Tests
// =============================================================================

TEST(CompileTimeTests, ContainsValueIsConstexpr) {
    using Pack = AnyTypeValuePack<1, 2, 3>;
    static_assert(Pack::contains_value(2));
    static_assert(!Pack::contains_value(4));
}

TEST(CompileTimeTests, FirstIndexOfValueIsConstexpr) {
    using Pack = AnyTypeValuePack<10, 20, 30>;
    static_assert(Pack::first_index_of_value(10) == 0);
    static_assert(Pack::first_index_of_value(20) == 1);
    static_assert(Pack::first_index_of_value(30) == 2);
    static_assert(!Pack::first_index_of_value(40).has_value());
}

TEST(CompileTimeTests, SizeIsConstexpr) {
    using Pack = AnyTypeValuePack<1, 2, 3, 4, 5>;
    static_assert(Pack::size == 5);
}

TEST(CompileTimeTests, ValueTupleIsConstexpr) {
    using Pack           = AnyTypeValuePack<100, 200>;
    constexpr auto tuple = Pack::value_tuple;
    static_assert(std::get<0>(tuple) == 100);
    static_assert(std::get<1>(tuple) == 200);
}

// =============================================================================
// Edge Case Tests
// =============================================================================

TEST(EdgeCases, LargePackSize) {
    using LargePack = AnyTypeValuePack<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15>;
    EXPECT_EQ(LargePack::size, std::size_t{16});
    EXPECT_TRUE(LargePack::contains_value(0));
    EXPECT_TRUE(LargePack::contains_value(15));
    EXPECT_EQ(LargePack::first_index_of_value(15), std::size_t{15});
}

TEST(EdgeCases, NegativeIntegers) {
    using Pack = SameTypeUniqueValuePack<-100, -50, 0, 50, 100>;
    EXPECT_TRUE(Pack::contains_value(-100));
    EXPECT_TRUE(Pack::contains_value(-50));
    EXPECT_TRUE(Pack::contains_value(0));
    EXPECT_EQ(Pack::index_of_value(-100), std::size_t{0});
    EXPECT_EQ(Pack::index_of_value(-50), std::size_t{1});
    EXPECT_EQ(Pack::index_of_value(0), std::size_t{2});
}

TEST(EdgeCases, CharValues) {
    using Pack = SameTypeUniqueValuePack<'a', 'b', 'z'>;
    static_assert(std::is_same_v<Pack::value_type, char>);
    EXPECT_TRUE(Pack::contains_value('a'));
    EXPECT_TRUE(Pack::contains_value('z'));
    EXPECT_FALSE(Pack::contains_value('c'));
    EXPECT_EQ(Pack::index_of_value('a'), std::size_t{0});
    EXPECT_EQ(Pack::index_of_value('z'), std::size_t{2});
}

TEST(EdgeCases, BoolValues) {
    using Pack = SameTypeValuePack<true, false>;
    static_assert(std::is_same_v<Pack::value_type, bool>);
    EXPECT_EQ(Pack::size, std::size_t{2});
    EXPECT_TRUE(Pack::contains_value(true));
    EXPECT_TRUE(Pack::contains_value(false));
}

} // namespace dispatch
} // namespace fvdb
