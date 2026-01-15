// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <fvdb/detail/dispatch/ValueSpace.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <optional>

namespace fvdb {
namespace dispatch {

// Test enum for axis testing
enum class Device { CPU, CUDA, Metal };
enum class DType { Float32, Float64, Int32 };

// Axis type aliases for convenience
using DeviceAxis = Values<Device::CPU, Device::CUDA, Device::Metal>;
using DTypeAxis  = Values<DType::Float32, DType::Float64, DType::Int32>;
using IntAxis    = Values<1, 2, 4, 8>;
using CharAxis   = Values<'a', 'b', 'c'>;

// =============================================================================
// ValueAxis Concept Tests
// =============================================================================

TEST(ValueAxis, ConceptRequirements) {
    // ValueAxis requires: SameTypeValuePack && UniqueValuePack && NonEmptyValuePack
    static_assert(is_value_axis<DeviceAxis>());
    static_assert(is_value_axis<DTypeAxis>());
    static_assert(is_value_axis<IntAxis>());
    static_assert(is_value_axis<CharAxis>());
    static_assert(is_value_axis<Values<42>>()); // Single element

    // Fails: empty pack
    static_assert(!is_value_axis<Values<>>());

    // Fails: not unique
    static_assert(!is_value_axis<Values<1, 2, 1>>());

    // Fails: not same type
    static_assert(!is_value_axis<Values<1, 'x', true>>());

    // Fails: not a value pack at all
    static_assert(!is_value_axis<int>());
    static_assert(!is_value_axis<std::index_sequence<1, 2, 3>>());
}

// =============================================================================
// ValueAxis Traits Tests
// =============================================================================

TEST(ValueAxisTraits, AxisSizeReturnsCorrectCount) {
    static_assert(AxisSize_v<DeviceAxis>() == 3);
    static_assert(AxisSize_v<IntAxis>() == 4);
    static_assert(AxisSize_v<Values<42>>() == 1);

    EXPECT_EQ(axisSize(DeviceAxis{}), 3u);
    EXPECT_EQ(axisSize(IntAxis{}), 4u);
}

TEST(ValueAxisTraits, AxisValueTypeExtractsCommonType) {
    static_assert(std::is_same_v<AxisValueType_t<DeviceAxis>, Device>);
    static_assert(std::is_same_v<AxisValueType_t<DTypeAxis>, DType>);
    static_assert(std::is_same_v<AxisValueType_t<IntAxis>, int>);
    static_assert(std::is_same_v<AxisValueType_t<CharAxis>, char>);
}

TEST(ValueAxisTraits, AxisElementAccessByIndex) {
    // Compile-time access
    static_assert(AxisElement_v<DeviceAxis, 0>() == Device::CPU);
    static_assert(AxisElement_v<DeviceAxis, 1>() == Device::CUDA);
    static_assert(AxisElement_v<DeviceAxis, 2>() == Device::Metal);
    static_assert(AxisElement_v<IntAxis, 3>() == 8);

    // Type extraction
    static_assert(std::is_same_v<AxisElement_t<DeviceAxis, 0>, Device>);
    static_assert(std::is_same_v<AxisElement_t<IntAxis, 0>, int>);

    // Runtime access
    EXPECT_EQ(axisElement(IntAxis{}, 0), 1);
    EXPECT_EQ(axisElement(IntAxis{}, 3), 8);
    EXPECT_EQ(axisElement(DeviceAxis{}, 1), Device::CUDA);
}

TEST(ValueAxisTraits, AxisContainsChecksPresence) {
    // Compile-time checks
    static_assert(AxisContains_v<DeviceAxis, Device::CPU>());
    static_assert(AxisContains_v<DeviceAxis, Device::CUDA>());
    static_assert(!AxisContains_v<IntAxis, 3>()); // 3 not in {1, 2, 4, 8}
    static_assert(AxisContains_v<IntAxis, 4>());

    // Runtime checks
    EXPECT_TRUE(axisContains(IntAxis{}, 1));
    EXPECT_TRUE(axisContains(IntAxis{}, 8));
    EXPECT_FALSE(axisContains(IntAxis{}, 3));
    EXPECT_FALSE(axisContains(IntAxis{}, 16));
}

TEST(ValueAxisTraits, AxisIndexFindsPosition) {
    // Compile-time definite index (guaranteed to exist)
    static_assert(AxisIndex_v<DeviceAxis, Device::CPU>() == 0);
    static_assert(AxisIndex_v<DeviceAxis, Device::CUDA>() == 1);
    static_assert(AxisIndex_v<DeviceAxis, Device::Metal>() == 2);
    static_assert(AxisIndex_v<IntAxis, 4>() == 2);

    // Runtime optional index
    EXPECT_EQ(axisIndex(IntAxis{}, 1), std::optional<size_t>{0});
    EXPECT_EQ(axisIndex(IntAxis{}, 8), std::optional<size_t>{3});
    EXPECT_EQ(axisIndex(IntAxis{}, 99), std::nullopt);

    // Runtime definite index
    EXPECT_EQ(axisDefiniteIndex(IntAxis{}, 1), 0u);
    EXPECT_EQ(axisDefiniteIndex(IntAxis{}, 8), 3u);
}

TEST(ValueAxisTraits, AxisSubsetOfRelationship) {
    using Full   = Values<1, 2, 3, 4, 5>;
    using Sub    = Values<2, 4>;
    using NotSub = Values<2, 99>;

    static_assert(AxisSubsetOf<Sub, Full>);
    static_assert(AxisSubsetOf<Full, Full>); // Self-subset
    static_assert(!AxisSubsetOf<NotSub, Full>);
    static_assert(!AxisSubsetOf<Full, Sub>); // Super not subset of sub
}

// =============================================================================
// ValueSpace Concept Tests
// =============================================================================

TEST(ValueSpace, ConceptAndStructure) {
    using Space1D = ValueAxes<DeviceAxis>;
    using Space2D = ValueAxes<DeviceAxis, DTypeAxis>;
    using Space3D = ValueAxes<DeviceAxis, DTypeAxis, IntAxis>;

    static_assert(is_value_space<Space1D>());
    static_assert(is_value_space<Space2D>());
    static_assert(is_value_space<Space3D>());

    // Non-space types
    static_assert(!is_value_space<int>());
    static_assert(!is_value_space<DeviceAxis>()); // Axis, not space
    static_assert(!is_value_space<Values<1, 2, 3>>());
}

TEST(ValueSpace, RankAndNumel) {
    using Space1D = ValueAxes<DeviceAxis>;
    using Space2D = ValueAxes<DeviceAxis, DTypeAxis>;
    using Space3D = ValueAxes<DeviceAxis, DTypeAxis, IntAxis>;

    // Rank
    static_assert(Rank_v<Space1D>() == 1);
    static_assert(Rank_v<Space2D>() == 2);
    static_assert(Rank_v<Space3D>() == 3);

    // Numel = product of axis sizes
    static_assert(Numel_v<Space1D>() == 3);  // 3
    static_assert(Numel_v<Space2D>() == 9);  // 3 * 3
    static_assert(Numel_v<Space3D>() == 36); // 3 * 3 * 4

    EXPECT_EQ(Rank_v<Space2D>(), 2u);
    EXPECT_EQ(Numel_v<Space2D>(), 9u);
}

// =============================================================================
// CoordTypesMatch Tests
// =============================================================================

TEST(CoordTypesMatch, MatchingTypesSatisfy) {
    using Space = ValueAxes<DeviceAxis, DTypeAxis>;

    // Correct types match (even if values aren't in the space, types matter)
    static_assert(coord_types_match_v<Space, Values<Device::CPU, DType::Float32>>());
    static_assert(coord_types_match_v<Space, Values<Device::Metal, DType::Int32>>());

    // CoordTypesMatch concept
    static_assert(CoordTypesMatch<Space, Values<Device::CPU, DType::Float32>>);
    static_assert(CoordTypesMatch<Space, Values<Device::CUDA, DType::Float64>>);
}

TEST(CoordTypesMatch, MismatchedTypesOrRankFail) {
    using Space = ValueAxes<DeviceAxis, DTypeAxis>;

    // Wrong types
    static_assert(!coord_types_match_v<Space, Values<1, 2>>());
    static_assert(!coord_types_match_v<Space, Values<Device::CPU, 42>>());

    // Wrong rank
    static_assert(!coord_types_match_v<Space, Values<Device::CPU>>());
    static_assert(!coord_types_match_v<Space, Values<Device::CPU, DType::Float32, 1>>());
}

TEST(CoordTypesMatch, SingleAxisCase) {
    // Single axis space with single value coord
    static_assert(coord_types_match_v<ValueAxes<Values<42>>, Values<42>>());
}

// =============================================================================
// SpaceContains Tests
// =============================================================================

TEST(SpaceContains, ContainedCoordinatesSatisfy) {
    using Space = ValueAxes<DeviceAxis, DTypeAxis>;

    // All combinations should be contained
    static_assert(space_contains_v<Space, Values<Device::CPU, DType::Float32>>());
    static_assert(space_contains_v<Space, Values<Device::CUDA, DType::Float64>>());
    static_assert(space_contains_v<Space, Values<Device::Metal, DType::Int32>>());

    // SpaceContains concept
    static_assert(SpaceContains<Space, Values<Device::CPU, DType::Float32>>);
}

TEST(SpaceContains, MissingValuesFail) {
    using Space = ValueAxes<IntAxis>; // {1, 2, 4, 8}

    static_assert(space_contains_v<Space, Values<1>>());
    static_assert(space_contains_v<Space, Values<8>>());
    static_assert(!space_contains_v<Space, Values<3>>());  // 3 not in axis
    static_assert(!space_contains_v<Space, Values<16>>()); // 16 not in axis
}

TEST(SpaceContains, MultiAxisValidation) {
    using Space = ValueAxes<DeviceAxis, IntAxis>;

    // Valid: both values in respective axes
    static_assert(space_contains_v<Space, Values<Device::CPU, 1>>());
    static_assert(space_contains_v<Space, Values<Device::Metal, 8>>());

    // Invalid: first value not in first axis (type mismatch)
    static_assert(!space_contains_v<Space, Values<1, 1>>());

    // Invalid: second value not in second axis
    static_assert(!space_contains_v<Space, Values<Device::CPU, 3>>());
}

// =============================================================================
// IndexSpaceOf Tests
// =============================================================================

TEST(IndexSpaceOf, MapsToCorrectSizes) {
    using Space1D = ValueAxes<DeviceAxis>;          // 3 elements
    using Space2D = ValueAxes<DeviceAxis, IntAxis>; // 3, 4 elements
    using Space3D = ValueAxes<DeviceAxis, DTypeAxis, IntAxis>;

    static_assert(std::is_same_v<IndexSpaceOf_t<Space1D>, Sizes<3>>);
    static_assert(std::is_same_v<IndexSpaceOf_t<Space2D>, Sizes<3, 4>>);
    static_assert(std::is_same_v<IndexSpaceOf_t<Space3D>, Sizes<3, 3, 4>>);

    // Verify numel matches
    static_assert(Numel_v<IndexSpaceOf_t<Space2D>>() == Numel_v<Space2D>());
}

// =============================================================================
// CoordFromPoint Tests
// =============================================================================

TEST(CoordFromPoint, ConvertsIndexToValue) {
    using Space = ValueAxes<DeviceAxis, DTypeAxis>;
    // DeviceAxis = {CPU, CUDA, Metal}, DTypeAxis = {Float32, Float64, Int32}

    // Point<0, 0> -> Values<CPU, Float32>
    static_assert(
        std::is_same_v<CoordFromPoint_t<Space, Point<0, 0>>, Values<Device::CPU, DType::Float32>>);

    // Point<1, 2> -> Values<CUDA, Int32>
    static_assert(
        std::is_same_v<CoordFromPoint_t<Space, Point<1, 2>>, Values<Device::CUDA, DType::Int32>>);

    // Point<2, 1> -> Values<Metal, Float64>
    static_assert(std::is_same_v<CoordFromPoint_t<Space, Point<2, 1>>,
                                 Values<Device::Metal, DType::Float64>>);
}

TEST(CoordFromPoint, SingleAxisConversion) {
    using Space = ValueAxes<IntAxis>; // {1, 2, 4, 8}

    static_assert(std::is_same_v<CoordFromPoint_t<Space, Point<0>>, Values<1>>);
    static_assert(std::is_same_v<CoordFromPoint_t<Space, Point<1>>, Values<2>>);
    static_assert(std::is_same_v<CoordFromPoint_t<Space, Point<2>>, Values<4>>);
    static_assert(std::is_same_v<CoordFromPoint_t<Space, Point<3>>, Values<8>>);
}

TEST(CoordFromPoint, ThreeAxisConversion) {
    using Space = ValueAxes<Values<'x', 'y'>, Values<10, 20>, Values<true, false>>;

    // Verify a few points
    static_assert(std::is_same_v<CoordFromPoint_t<Space, Point<0, 0, 0>>, Values<'x', 10, true>>);
    static_assert(std::is_same_v<CoordFromPoint_t<Space, Point<1, 1, 1>>, Values<'y', 20, false>>);
    static_assert(std::is_same_v<CoordFromPoint_t<Space, Point<0, 1, 0>>, Values<'x', 20, true>>);
}

// =============================================================================
// Integration Tests
// =============================================================================

TEST(ValueSpaceIntegration, FullRoundTrip) {
    using Space = ValueAxes<DeviceAxis, IntAxis>;
    // DeviceAxis = {CPU, CUDA, Metal} (3), IntAxis = {1, 2, 4, 8} (4)
    // Total: 12 combinations

    static_assert(Numel_v<Space>() == 12);
    static_assert(std::is_same_v<IndexSpaceOf_t<Space>, Sizes<3, 4>>);

    // Verify coord at specific index
    using Coord = CoordFromPoint_t<Space, Point<1, 2>>; // CUDA, 4
    static_assert(std::is_same_v<Coord, Values<Device::CUDA, 4>>);
    static_assert(SpaceContains<Space, Coord>);

    // Verify index lookup for that coord
    static_assert(AxisIndex_v<DeviceAxis, Device::CUDA>() == 1);
    static_assert(AxisIndex_v<IntAxis, 4>() == 2);
}

TEST(ValueSpaceIntegration, SpaceContainsImpliesCoordTypesMatch) {
    using Space      = ValueAxes<DeviceAxis, DTypeAxis>;
    using ValidCoord = Values<Device::CPU, DType::Float32>;

    // If SpaceContains, then CoordTypesMatch must also be true
    static_assert(SpaceContains<Space, ValidCoord>);
    static_assert(CoordTypesMatch<Space, ValidCoord>);

    // CoordTypesMatch can be true without SpaceContains (hypothetically invalid values)
    // This is tested implicitly - any valid coord that passes SpaceContains also passes types
}

} // namespace dispatch
} // namespace fvdb
