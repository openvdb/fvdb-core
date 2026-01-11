// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <fvdb/detail/dispatch/PermutationMap.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <string>
#include <type_traits>

namespace fvdb {
namespace dispatch {

// =============================================================================
// Test Axes Definitions
// =============================================================================

using SingleAxis = SameTypeUniqueValuePack<10, 20, 30>;
using CharAxis   = SameTypeUniqueValuePack<'a', 'b'>;
using IntAxis    = SameTypeUniqueValuePack<1, 2, 3>;
using BoolAxis   = SameTypeUniqueValuePack<true, false>;

using SingleAxisSpace = AxisOuterProduct<SingleAxis>;
using TwoAxisSpace    = AxisOuterProduct<IntAxis, CharAxis>;
using ThreeAxisSpace  = AxisOuterProduct<IntAxis, CharAxis, BoolAxis>;

// =============================================================================
// PermutationArrayMap Constructor Tests
// =============================================================================

TEST(PermutationArrayMap, DefaultConstructorInitializesWithEmptyValue) {
    PermutationArrayMap<SingleAxisSpace, int> map;

    // All values should be initialized to empty_value (0)
    for (size_t i = 0; i < SingleAxisSpace::size; ++i) {
        EXPECT_EQ(map.get(i), 0);
    }
}

TEST(PermutationArrayMap, CustomEmptyValueIsUsed) {
    constexpr int sentinel = -999;
    PermutationArrayMap<SingleAxisSpace, int, sentinel> map;

    EXPECT_EQ(map.empty_value, sentinel);
    for (size_t i = 0; i < SingleAxisSpace::size; ++i) {
        EXPECT_EQ(map.get(i), sentinel);
    }
}

TEST(PermutationArrayMap, HasCorrectStorageSize) {
    PermutationArrayMap<TwoAxisSpace, int> map;
    EXPECT_EQ(map.storage_.size(), TwoAxisSpace::size);
    EXPECT_EQ(map.storage_.size(), 6u); // 3 * 2
}

// =============================================================================
// PermutationArrayMap Linear Index Accessor Tests
// =============================================================================

TEST(PermutationArrayMap, SetAndGetWithLinearIndex) {
    PermutationArrayMap<SingleAxisSpace, int> map;

    map.set(0, 100);
    map.set(1, 200);
    map.set(2, 300);

    EXPECT_EQ(map.get(0), 100);
    EXPECT_EQ(map.get(1), 200);
    EXPECT_EQ(map.get(2), 300);
}

TEST(PermutationArrayMap, SetOverwritesPreviousValue) {
    PermutationArrayMap<SingleAxisSpace, int> map;

    map.set(0, 100);
    EXPECT_EQ(map.get(0), 100);

    map.set(0, 999);
    EXPECT_EQ(map.get(0), 999);
}

// =============================================================================
// PermutationArrayMap Tuple Index Accessor Tests
// =============================================================================

TEST(PermutationArrayMap, SetAndGetWithTupleIndexSingleAxis) {
    PermutationArrayMap<SingleAxisSpace, int> map;

    map.set(std::make_tuple(10), 100);
    map.set(std::make_tuple(20), 200);
    map.set(std::make_tuple(30), 300);

    EXPECT_EQ(map.get(std::make_tuple(10)), 100);
    EXPECT_EQ(map.get(std::make_tuple(20)), 200);
    EXPECT_EQ(map.get(std::make_tuple(30)), 300);
}

TEST(PermutationArrayMap, SetAndGetWithTupleIndexTwoAxes) {
    PermutationArrayMap<TwoAxisSpace, int> map;

    // Set values at various positions
    map.set(std::make_tuple(1, 'a'), 100);
    map.set(std::make_tuple(2, 'b'), 200);
    map.set(std::make_tuple(3, 'a'), 300);

    EXPECT_EQ(map.get(std::make_tuple(1, 'a')), 100);
    EXPECT_EQ(map.get(std::make_tuple(2, 'b')), 200);
    EXPECT_EQ(map.get(std::make_tuple(3, 'a')), 300);

    // Unset positions should return empty_value
    EXPECT_EQ(map.get(std::make_tuple(1, 'b')), 0);
    EXPECT_EQ(map.get(std::make_tuple(2, 'a')), 0);
}

TEST(PermutationArrayMap, GetWithInvalidTupleIndexReturnsEmptyValue) {
    PermutationArrayMap<SingleAxisSpace, int, -1> map;
    map.set(std::make_tuple(10), 100);

    // Invalid value in axis should return empty_value
    EXPECT_EQ(map.get(std::make_tuple(15)), -1); // 15 not in axis
    EXPECT_EQ(map.get(std::make_tuple(0)), -1);  // 0 not in axis
}

TEST(PermutationArrayMap, SetWithInvalidTupleIndexIsIgnored) {
    PermutationArrayMap<SingleAxisSpace, int> map;

    // Setting with invalid index should be silently ignored
    map.set(std::make_tuple(15), 999);

    // Map should remain unchanged (all zeros)
    for (size_t i = 0; i < SingleAxisSpace::size; ++i) {
        EXPECT_EQ(map.get(i), 0);
    }
}

TEST(PermutationArrayMap, OperatorBracketWorksLikeTupleGet) {
    PermutationArrayMap<TwoAxisSpace, int> map;

    map.set(std::make_tuple(1, 'a'), 42);
    map.set(std::make_tuple(2, 'b'), 84);

    EXPECT_EQ(map[std::make_tuple(1, 'a')], 42);
    EXPECT_EQ(map[std::make_tuple(2, 'b')], 84);
    EXPECT_EQ(map[std::make_tuple(3, 'a')], 0); // Unset
}

// =============================================================================
// PermutationArrayMap Type Aliases
// =============================================================================

TEST(PermutationArrayMap, TypeAliasesAreCorrect) {
    using Map = PermutationArrayMap<TwoAxisSpace, double>;

    static_assert(std::is_same_v<Map::axes_type, TwoAxisSpace>);
    static_assert(std::is_same_v<Map::value_type, double>);
    static_assert(std::is_same_v<Map::index_tuple_type, std::tuple<int, char>>);
    static_assert(Map::empty_value == 0.0);
}

// =============================================================================
// PermutationUnorderedMap Constructor Tests
// =============================================================================

TEST(PermutationUnorderedMap, DefaultConstructorCreatesEmptyStorage) {
    PermutationUnorderedMap<SingleAxisSpace, int> map;

    // Storage should be empty initially
    EXPECT_TRUE(map.storage_.empty());

    // All lookups should return empty_value
    for (size_t i = 0; i < SingleAxisSpace::size; ++i) {
        EXPECT_EQ(map.get(i), 0);
    }
}

TEST(PermutationUnorderedMap, CustomEmptyValueIsUsed) {
    constexpr int sentinel = -999;
    PermutationUnorderedMap<SingleAxisSpace, int, sentinel> map;

    EXPECT_EQ(map.empty_value, sentinel);
    EXPECT_EQ(map.get(0), sentinel);
}

// =============================================================================
// PermutationUnorderedMap Linear Index Accessor Tests
// =============================================================================

TEST(PermutationUnorderedMap, SetAndGetWithLinearIndex) {
    PermutationUnorderedMap<SingleAxisSpace, int> map;

    map.set(0, 100);
    map.set(2, 300);

    EXPECT_EQ(map.get(0), 100);
    EXPECT_EQ(map.get(1), 0); // Not set
    EXPECT_EQ(map.get(2), 300);
}

TEST(PermutationUnorderedMap, SetOverwritesPreviousValue) {
    PermutationUnorderedMap<SingleAxisSpace, int> map;

    map.set(0, 100);
    EXPECT_EQ(map.get(0), 100);

    map.set(0, 999);
    EXPECT_EQ(map.get(0), 999);
}

TEST(PermutationUnorderedMap, OnlySetValuesExistInStorage) {
    PermutationUnorderedMap<TwoAxisSpace, int> map;

    map.set(0, 100);
    map.set(3, 400);

    EXPECT_EQ(map.storage_.size(), 2u);
}

// =============================================================================
// PermutationUnorderedMap Tuple Index Accessor Tests
// =============================================================================

TEST(PermutationUnorderedMap, SetAndGetWithTupleIndex) {
    PermutationUnorderedMap<TwoAxisSpace, int> map;

    map.set(std::make_tuple(1, 'a'), 100);
    map.set(std::make_tuple(2, 'b'), 200);

    EXPECT_EQ(map.get(std::make_tuple(1, 'a')), 100);
    EXPECT_EQ(map.get(std::make_tuple(2, 'b')), 200);
    EXPECT_EQ(map.get(std::make_tuple(1, 'b')), 0); // Not set
}

TEST(PermutationUnorderedMap, GetWithInvalidTupleIndexReturnsEmptyValue) {
    PermutationUnorderedMap<SingleAxisSpace, int, -1> map;
    map.set(std::make_tuple(10), 100);

    EXPECT_EQ(map.get(std::make_tuple(15)), -1); // Invalid value
}

TEST(PermutationUnorderedMap, SetWithInvalidTupleIndexIsIgnored) {
    PermutationUnorderedMap<SingleAxisSpace, int> map;

    map.set(std::make_tuple(15), 999);

    EXPECT_TRUE(map.storage_.empty());
}

// =============================================================================
// PermutationMapSelector Tests
// =============================================================================

TEST(PermutationMapSelector, SelectsArrayForSmallSpace) {
    using Axes = AxisOuterProduct<SameTypeUniqueValuePack<1, 2, 3>>; // size = 3
    using Map  = PermutationMapSelector_t<Axes, int, 0, 10>;         // threshold = 10

    static_assert(std::is_same_v<Map, PermutationArrayMap<Axes, int, 0>>);
}

TEST(PermutationMapSelector, SelectsUnorderedMapForLargeSpace) {
    using Axes = AxisOuterProduct<SameTypeUniqueValuePack<1, 2, 3>>; // size = 3
    using Map  = PermutationMapSelector_t<Axes, int, 0, 2>;          // threshold = 2

    static_assert(std::is_same_v<Map, PermutationUnorderedMap<Axes, int, 0>>);
}

TEST(PermutationMapSelector, SelectsArrayAtExactThreshold) {
    using Axes = AxisOuterProduct<SameTypeUniqueValuePack<1, 2, 3>>; // size = 3
    using Map  = PermutationMapSelector_t<Axes, int, 0, 3>;          // threshold = size

    static_assert(std::is_same_v<Map, PermutationArrayMap<Axes, int, 0>>);
}

// =============================================================================
// PermutationMap Alias Tests
// =============================================================================

TEST(PermutationMap, UsesDefaultThresholdOf1024) {
    // Small space should use array
    using SmallAxes = AxisOuterProduct<SameTypeUniqueValuePack<1, 2, 3>>;
    using SmallMap  = PermutationMap<SmallAxes, int>;

    static_assert(std::is_same_v<SmallMap, PermutationArrayMap<SmallAxes, int, 0>>);
}

TEST(PermutationMap, SupportsCustomEmptyValue) {
    using Axes = AxisOuterProduct<SameTypeUniqueValuePack<1, 2>>;
    using Map  = PermutationMap<Axes, int, -1>;

    Map map;
    EXPECT_EQ(map.get(0), -1);
}

// =============================================================================
// PermutationMapConcept Tests
// =============================================================================

TEST(PermutationMapConcept, ArrayMapSatisfiesConcept) {
    static_assert(PermutationMapConcept<PermutationArrayMap<SingleAxisSpace, int>>);
}

TEST(PermutationMapConcept, UnorderedMapSatisfiesConcept) {
    static_assert(PermutationMapConcept<PermutationUnorderedMap<SingleAxisSpace, int>>);
}

TEST(PermutationMapConcept, RegularTypesDoNotSatisfyConcept) {
    static_assert(!PermutationMapConcept<int>);
    static_assert(!PermutationMapConcept<std::string>);
    static_assert(!PermutationMapConcept<std::unordered_map<int, int>>);
}

// =============================================================================
// PointGenerator Tests
// =============================================================================

namespace {
// Simple instantiator that returns a constant derived from compile-time values
template <auto V> struct PointInstantiator {
    static int
    get() {
        return static_cast<int>(V) * 10;
    }
};

template <auto V1, auto V2> struct TwoAxisPointInstantiator {
    static int
    get() {
        return static_cast<int>(V1) * 100 + static_cast<int>(V2);
    }
};
} // namespace

TEST(PointGenerator, GeneratesSingleValueAtSpecificPoint) {
    PermutationArrayMap<SingleAxisSpace, int> map;

    PointGenerator<PointInstantiator, 20>::apply(map);

    // Only position for value 20 (index 1) should be set
    EXPECT_EQ(map.get(std::make_tuple(10)), 0);
    EXPECT_EQ(map.get(std::make_tuple(20)), 200); // 20 * 10
    EXPECT_EQ(map.get(std::make_tuple(30)), 0);
}

TEST(PointGenerator, WorksWithTwoAxisSpace) {
    PermutationArrayMap<TwoAxisSpace, int> map;

    PointGenerator<TwoAxisPointInstantiator, 2, 'b'>::apply(map);

    // Only (2, 'b') should be set
    EXPECT_EQ(map.get(std::make_tuple(1, 'a')), 0);
    EXPECT_EQ(map.get(std::make_tuple(1, 'b')), 0);
    EXPECT_EQ(map.get(std::make_tuple(2, 'a')), 0);
    EXPECT_EQ(map.get(std::make_tuple(2, 'b')), 298); // 2*100 + 'b'(98)
    EXPECT_EQ(map.get(std::make_tuple(3, 'a')), 0);
    EXPECT_EQ(map.get(std::make_tuple(3, 'b')), 0);
}

TEST(PointGenerator, WorksWithUnorderedMap) {
    PermutationUnorderedMap<SingleAxisSpace, int> map;

    PointGenerator<PointInstantiator, 30>::apply(map);

    EXPECT_EQ(map.get(std::make_tuple(30)), 300);
    EXPECT_EQ(map.storage_.size(), 1u); // Only one entry stored
}

// =============================================================================
// SubspaceGenerator Tests
// =============================================================================

namespace {
template <auto V> struct SubspaceInstantiator {
    static int
    get() {
        return static_cast<int>(V) * 10;
    }
};

template <auto V1, auto V2> struct TwoAxisSubspaceInstantiator {
    static int
    get() {
        return static_cast<int>(V1) * 1000 + static_cast<int>(V2);
    }
};
} // namespace

TEST(SubspaceGenerator, GeneratesValuesForEntireSubspace) {
    PermutationArrayMap<SingleAxisSpace, int> map;

    // Generate for all values in the axis
    SubspaceGenerator<SubspaceInstantiator, SingleAxisSpace>::apply(map);

    EXPECT_EQ(map.get(std::make_tuple(10)), 100);
    EXPECT_EQ(map.get(std::make_tuple(20)), 200);
    EXPECT_EQ(map.get(std::make_tuple(30)), 300);
}

TEST(SubspaceGenerator, GeneratesValuesForTwoAxisSubspace) {
    PermutationArrayMap<TwoAxisSpace, int> map;

    SubspaceGenerator<TwoAxisSubspaceInstantiator, TwoAxisSpace>::apply(map);

    // Check all combinations
    EXPECT_EQ(map.get(std::make_tuple(1, 'a')), 1097); // 1*1000 + 'a'(97)
    EXPECT_EQ(map.get(std::make_tuple(1, 'b')), 1098);
    EXPECT_EQ(map.get(std::make_tuple(2, 'a')), 2097);
    EXPECT_EQ(map.get(std::make_tuple(2, 'b')), 2098);
    EXPECT_EQ(map.get(std::make_tuple(3, 'a')), 3097);
    EXPECT_EQ(map.get(std::make_tuple(3, 'b')), 3098);
}

TEST(SubspaceGenerator, GeneratesForProperSubspace) {
    // Create a map with a larger axis space
    using LargerAxis  = SameTypeUniqueValuePack<10, 20, 30, 40, 50>;
    using LargerSpace = AxisOuterProduct<LargerAxis>;
    using SubAxis     = SameTypeUniqueValuePack<20, 30>;
    using Subspace    = AxisOuterProduct<SubAxis>;

    PermutationArrayMap<LargerSpace, int> map;

    SubspaceGenerator<SubspaceInstantiator, Subspace>::apply(map);

    // Only 20 and 30 should be set
    EXPECT_EQ(map.get(std::make_tuple(10)), 0);
    EXPECT_EQ(map.get(std::make_tuple(20)), 200);
    EXPECT_EQ(map.get(std::make_tuple(30)), 300);
    EXPECT_EQ(map.get(std::make_tuple(40)), 0);
    EXPECT_EQ(map.get(std::make_tuple(50)), 0);
}

TEST(SubspaceGenerator, WorksWithUnorderedMap) {
    PermutationUnorderedMap<SingleAxisSpace, int> map;

    SubspaceGenerator<SubspaceInstantiator, SingleAxisSpace>::apply(map);

    EXPECT_EQ(map.get(std::make_tuple(10)), 100);
    EXPECT_EQ(map.get(std::make_tuple(20)), 200);
    EXPECT_EQ(map.get(std::make_tuple(30)), 300);
    EXPECT_EQ(map.storage_.size(), 3u);
}

// =============================================================================
// FillGenerator Tests
// =============================================================================

TEST(FillGenerator, FillsSubspaceWithConstantValue) {
    PermutationArrayMap<SingleAxisSpace, int> map;

    FillGenerator<SingleAxisSpace, 42>::apply(map);

    EXPECT_EQ(map.get(std::make_tuple(10)), 42);
    EXPECT_EQ(map.get(std::make_tuple(20)), 42);
    EXPECT_EQ(map.get(std::make_tuple(30)), 42);
}

TEST(FillGenerator, FillsProperSubspaceOnly) {
    using LargerAxis  = SameTypeUniqueValuePack<10, 20, 30, 40>;
    using LargerSpace = AxisOuterProduct<LargerAxis>;
    using SubAxis     = SameTypeUniqueValuePack<20, 30>;
    using Subspace    = AxisOuterProduct<SubAxis>;

    PermutationArrayMap<LargerSpace, int, -1> map;

    FillGenerator<Subspace, 99>::apply(map);

    EXPECT_EQ(map.get(std::make_tuple(10)), -1); // Not in subspace
    EXPECT_EQ(map.get(std::make_tuple(20)), 99); // In subspace
    EXPECT_EQ(map.get(std::make_tuple(30)), 99); // In subspace
    EXPECT_EQ(map.get(std::make_tuple(40)), -1); // Not in subspace
}

TEST(FillGenerator, FillsTwoAxisSubspace) {
    PermutationArrayMap<TwoAxisSpace, int> map;

    FillGenerator<TwoAxisSpace, 123>::apply(map);

    // All positions should have the same value
    for (int i: {1, 2, 3}) {
        for (char c: {'a', 'b'}) {
            EXPECT_EQ(map.get(std::make_tuple(i, c)), 123);
        }
    }
}

TEST(FillGenerator, WorksWithUnorderedMap) {
    PermutationUnorderedMap<SingleAxisSpace, int> map;

    FillGenerator<SingleAxisSpace, 42>::apply(map);

    EXPECT_EQ(map.get(std::make_tuple(10)), 42);
    EXPECT_EQ(map.storage_.size(), 3u);
}

// =============================================================================
// GeneratorList Tests
// =============================================================================

TEST(GeneratorList, AppliesMultipleGeneratorsInOrder) {
    PermutationArrayMap<SingleAxisSpace, int> map;

    using Generators =
        GeneratorList<FillGenerator<SingleAxisSpace, 10>, PointGenerator<PointInstantiator, 20>>;

    Generators::apply(map);

    // FillGenerator sets all to 10, then PointGenerator overwrites 20's entry
    EXPECT_EQ(map.get(std::make_tuple(10)), 10);
    EXPECT_EQ(map.get(std::make_tuple(20)), 200); // Overwritten
    EXPECT_EQ(map.get(std::make_tuple(30)), 10);
}

TEST(GeneratorList, LaterGeneratorsOverwriteEarlierOnes) {
    PermutationArrayMap<SingleAxisSpace, int> map;

    using Generators =
        GeneratorList<FillGenerator<SingleAxisSpace, 100>, FillGenerator<SingleAxisSpace, 200>>;

    Generators::apply(map);

    // All should be 200 (the later value)
    EXPECT_EQ(map.get(std::make_tuple(10)), 200);
    EXPECT_EQ(map.get(std::make_tuple(20)), 200);
    EXPECT_EQ(map.get(std::make_tuple(30)), 200);
}

TEST(GeneratorList, EmptyGeneratorListDoesNothing) {
    PermutationArrayMap<SingleAxisSpace, int, -1> map;

    using Generators = GeneratorList<>;
    Generators::apply(map);

    // All values should still be empty_value
    EXPECT_EQ(map.get(std::make_tuple(10)), -1);
    EXPECT_EQ(map.get(std::make_tuple(20)), -1);
    EXPECT_EQ(map.get(std::make_tuple(30)), -1);
}

TEST(GeneratorList, MultipleSubspaceGenerators) {
    using Axis1 = SameTypeUniqueValuePack<1, 2>;
    using Axis2 = SameTypeUniqueValuePack<10, 20>;
    using Space = AxisOuterProduct<Axis1, Axis2>;
    using SubA  = AxisOuterProduct<SameTypeUniqueValuePack<1>, SameTypeUniqueValuePack<10, 20>>;
    using SubB  = AxisOuterProduct<SameTypeUniqueValuePack<2>, SameTypeUniqueValuePack<10, 20>>;

    PermutationArrayMap<Space, int> map;

    using Generators = GeneratorList<FillGenerator<SubA, 100>, FillGenerator<SubB, 200>>;

    Generators::apply(map);

    EXPECT_EQ(map.get(std::make_tuple(1, 10)), 100);
    EXPECT_EQ(map.get(std::make_tuple(1, 20)), 100);
    EXPECT_EQ(map.get(std::make_tuple(2, 10)), 200);
    EXPECT_EQ(map.get(std::make_tuple(2, 20)), 200);
}

// =============================================================================
// GeneratorForConcept Tests
// =============================================================================

TEST(GeneratorForConcept, ValidGeneratorsSatisfyConcept) {
    using Map = PermutationArrayMap<SingleAxisSpace, int>;

    static_assert(GeneratorForConcept<FillGenerator<SingleAxisSpace, 42>, Map>);
    static_assert(GeneratorForConcept<PointGenerator<PointInstantiator, 10>, Map>);
    static_assert(
        GeneratorForConcept<SubspaceGenerator<SubspaceInstantiator, SingleAxisSpace>, Map>);
    static_assert(GeneratorForConcept<GeneratorList<>, Map>);
}

// =============================================================================
// build_permutation_map Tests
// =============================================================================

TEST(BuildPermutationMap, CreatesMapFromGenerators) {
    using Map        = PermutationArrayMap<SingleAxisSpace, int>;
    using Generators = FillGenerator<SingleAxisSpace, 42>;

    auto map = build_permutation_map<Map, Generators>();

    EXPECT_EQ(map.get(std::make_tuple(10)), 42);
    EXPECT_EQ(map.get(std::make_tuple(20)), 42);
    EXPECT_EQ(map.get(std::make_tuple(30)), 42);
}

TEST(BuildPermutationMap, CreatesMapFromGeneratorList) {
    using Map = PermutationArrayMap<SingleAxisSpace, int>;
    using Generators =
        GeneratorList<FillGenerator<SingleAxisSpace, 10>, PointGenerator<PointInstantiator, 20>>;

    auto map = build_permutation_map<Map, Generators>();

    EXPECT_EQ(map.get(std::make_tuple(10)), 10);
    EXPECT_EQ(map.get(std::make_tuple(20)), 200); // Overwritten by PointGenerator
    EXPECT_EQ(map.get(std::make_tuple(30)), 10);
}

TEST(BuildPermutationMap, WorksWithUnorderedMap) {
    using Map        = PermutationUnorderedMap<SingleAxisSpace, int>;
    using Generators = FillGenerator<SingleAxisSpace, 42>;

    auto map = build_permutation_map<Map, Generators>();

    EXPECT_EQ(map.get(std::make_tuple(10)), 42);
    EXPECT_EQ(map.storage_.size(), 3u);
}

TEST(BuildPermutationMap, CanBeStoredAsConst) {
    using Map        = PermutationArrayMap<SingleAxisSpace, int>;
    using Generators = FillGenerator<SingleAxisSpace, 42>;

    static const auto map = build_permutation_map<Map, Generators>();

    EXPECT_EQ(map.get(std::make_tuple(10)), 42);
}

// =============================================================================
// overlay_permutation_map Tests
// =============================================================================

TEST(OverlayPermutationMap, OverlaysNewValuesOnExistingMap) {
    using Map        = PermutationArrayMap<SingleAxisSpace, int>;
    using BaseGen    = FillGenerator<SingleAxisSpace, 10>;
    using OverlayGen = PointGenerator<PointInstantiator, 20>;

    auto base    = build_permutation_map<Map, BaseGen>();
    auto overlay = overlay_permutation_map<OverlayGen>(base);

    // Base map should be unchanged
    EXPECT_EQ(base.get(std::make_tuple(10)), 10);
    EXPECT_EQ(base.get(std::make_tuple(20)), 10);
    EXPECT_EQ(base.get(std::make_tuple(30)), 10);

    // Overlay map should have the new value at position 20
    EXPECT_EQ(overlay.get(std::make_tuple(10)), 10);
    EXPECT_EQ(overlay.get(std::make_tuple(20)), 200); // Overwritten
    EXPECT_EQ(overlay.get(std::make_tuple(30)), 10);
}

TEST(OverlayPermutationMap, PreservesNonOverlaidValues) {
    using Map = PermutationArrayMap<SingleAxisSpace, int>;
    using BaseGen =
        GeneratorList<PointGenerator<PointInstantiator, 10>, PointGenerator<PointInstantiator, 30>>;
    using OverlayGen = PointGenerator<PointInstantiator, 20>;

    auto base    = build_permutation_map<Map, BaseGen>();
    auto overlay = overlay_permutation_map<OverlayGen>(base);

    EXPECT_EQ(overlay.get(std::make_tuple(10)), 100); // Preserved
    EXPECT_EQ(overlay.get(std::make_tuple(20)), 200); // New
    EXPECT_EQ(overlay.get(std::make_tuple(30)), 300); // Preserved
}

TEST(OverlayPermutationMap, WorksWithUnorderedMap) {
    using Map        = PermutationUnorderedMap<SingleAxisSpace, int>;
    using BaseGen    = FillGenerator<SingleAxisSpace, 10>;
    using OverlayGen = PointGenerator<PointInstantiator, 20>;

    auto base    = build_permutation_map<Map, BaseGen>();
    auto overlay = overlay_permutation_map<OverlayGen>(base);

    EXPECT_EQ(overlay.get(std::make_tuple(20)), 200);
    EXPECT_EQ(overlay.get(std::make_tuple(10)), 10);
}

// =============================================================================
// Edge Cases and Complex Scenarios
// =============================================================================

TEST(PermutationMap, ThreeAxisSpaceWorks) {
    PermutationArrayMap<ThreeAxisSpace, int> map;

    // IntAxis: 1, 2, 3; CharAxis: 'a', 'b'; BoolAxis: true, false
    // Size: 3 * 2 * 2 = 12

    EXPECT_EQ(map.storage_.size(), 12u);

    map.set(std::make_tuple(1, 'a', true), 100);
    map.set(std::make_tuple(3, 'b', false), 200);

    EXPECT_EQ(map.get(std::make_tuple(1, 'a', true)), 100);
    EXPECT_EQ(map.get(std::make_tuple(3, 'b', false)), 200);
    EXPECT_EQ(map.get(std::make_tuple(2, 'a', true)), 0);
}

TEST(PermutationMap, LinearAndTupleAccessAreConsistent) {
    PermutationArrayMap<TwoAxisSpace, int> map;

    // Set via linear index
    auto linearIdx = TwoAxisSpace::index_of_values(2, 'a').value();
    map.set(linearIdx, 999);

    // Read via tuple
    EXPECT_EQ(map.get(std::make_tuple(2, 'a')), 999);

    // Set via tuple
    map.set(std::make_tuple(3, 'b'), 888);

    // Read via linear index
    auto linearIdx2 = TwoAxisSpace::index_of_values(3, 'b').value();
    EXPECT_EQ(map.get(linearIdx2), 888);
}

TEST(PermutationMap, PointerValuesWork) {
    using Map = PermutationArrayMap<SingleAxisSpace, void *, nullptr>;

    int a = 1, b = 2;
    Map map;

    map.set(std::make_tuple(10), &a);
    map.set(std::make_tuple(20), &b);

    EXPECT_EQ(map.get(std::make_tuple(10)), &a);
    EXPECT_EQ(map.get(std::make_tuple(20)), &b);
    EXPECT_EQ(map.get(std::make_tuple(30)), nullptr);
}

// =============================================================================
// Constexpr Tests
// =============================================================================

TEST(PermutationArrayMap, ConstructorAndAccessorsAreConstexpr) {
    // Verify the entire PermutationArrayMap can be used in constexpr context
    constexpr auto testConstexpr = []() {
        PermutationArrayMap<SingleAxisSpace, int> map;
        map.set(0, 42);
        map.set(1, 100);
        return map.get(0) + map.get(1);
    }();

    static_assert(testConstexpr == 142);
    EXPECT_EQ(testConstexpr, 142);
}

TEST(PermutationArrayMap, DefaultInitializationIsConstexpr) {
    // Verify default-initialized values are correct in constexpr context
    constexpr auto testEmptyValue = []() {
        PermutationArrayMap<SingleAxisSpace, int, -1> map;
        return map.get(0);
    }();

    static_assert(testEmptyValue == -1);
    EXPECT_EQ(testEmptyValue, -1);
}

} // namespace dispatch
} // namespace fvdb
