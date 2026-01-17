// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <fvdb/detail/dispatch/ValueSpace.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

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

    // Succeeds! Index sequences are value packs.
    static_assert(is_value_axis<std::index_sequence<1, 2, 3>>());
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
// SpaceHasSubspace Tests
// =============================================================================

TEST(SpaceHasSubspace, IdenticalSpaceIsSubspace) {
    using Space = ValueAxes<DeviceAxis, DTypeAxis>;

    // A space is a subspace of itself
    static_assert(space_has_subspace_v<Space, Space>());
    static_assert(SpaceHasSubspace<Space, Space>);
}

TEST(SpaceHasSubspace, SingleAxisSubspace) {
    using FullAxis = Values<1, 2, 3, 4, 5>;
    using SubAxis  = Values<2, 4>;
    using Space    = ValueAxes<FullAxis>;
    using Subspace = ValueAxes<SubAxis>;

    static_assert(space_has_subspace_v<Space, Subspace>());
    static_assert(SpaceHasSubspace<Space, Subspace>);

    // Reverse is not true
    static_assert(!space_has_subspace_v<Subspace, Space>());
    static_assert(!SpaceHasSubspace<Subspace, Space>);
}

TEST(SpaceHasSubspace, TwoAxisSubspace) {
    using Space    = ValueAxes<DeviceAxis, DTypeAxis>;
    using Subspace = ValueAxes<Values<Device::CPU, Device::CUDA>, Values<DType::Float32>>;

    // Subspace has subset of each axis
    static_assert(space_has_subspace_v<Space, Subspace>());
    static_assert(SpaceHasSubspace<Space, Subspace>);

    // Single device, all dtypes
    using Subspace2 = ValueAxes<Values<Device::CPU>, DTypeAxis>;
    static_assert(space_has_subspace_v<Space, Subspace2>());
    static_assert(SpaceHasSubspace<Space, Subspace2>);
}

TEST(SpaceHasSubspace, ThreeAxisSubspace) {
    using Space = ValueAxes<DeviceAxis, DTypeAxis, IntAxis>;
    using Subspace =
        ValueAxes<Values<Device::CPU>, Values<DType::Float32, DType::Float64>, Values<1, 2>>;

    static_assert(space_has_subspace_v<Space, Subspace>());
    static_assert(SpaceHasSubspace<Space, Subspace>);
}

TEST(SpaceHasSubspace, FailsWhenAxisNotSubset) {
    using Space  = ValueAxes<DeviceAxis, IntAxis>;
    using NotSub = ValueAxes<DeviceAxis, Values<1, 2, 3>>; // 3 not in IntAxis

    static_assert(!space_has_subspace_v<Space, NotSub>());
    static_assert(!SpaceHasSubspace<Space, NotSub>);
}

TEST(SpaceHasSubspace, FailsWhenRankMismatch) {
    using Space1D = ValueAxes<DeviceAxis>;
    using Space2D = ValueAxes<DeviceAxis, DTypeAxis>;

    // Different ranks can't be subspaces
    static_assert(!space_has_subspace_v<Space1D, Space2D>());
    static_assert(!space_has_subspace_v<Space2D, Space1D>());
}

TEST(SpaceHasSubspace, SingleElementSubspace) {
    using Space    = ValueAxes<DeviceAxis, DTypeAxis>;
    using Subspace = ValueAxes<Values<Device::CUDA>, Values<DType::Float64>>;

    // Single element in each axis is valid subspace
    static_assert(space_has_subspace_v<Space, Subspace>());
    static_assert(SpaceHasSubspace<Space, Subspace>);
}

// =============================================================================
// SpaceCovers Tests
// =============================================================================

TEST(SpaceCovers, CoversContainedCoord) {
    using Space = ValueAxes<DeviceAxis, DTypeAxis>;
    using Coord = Values<Device::CPU, DType::Float32>;

    static_assert(space_covers_v<Space, Coord>());
    static_assert(SpaceCovers<Space, Coord>);
}

TEST(SpaceCovers, CoversSubspace) {
    using Space    = ValueAxes<DeviceAxis, DTypeAxis>;
    using Subspace = ValueAxes<Values<Device::CPU>, DTypeAxis>;

    static_assert(space_covers_v<Space, Subspace>());
    static_assert(SpaceCovers<Space, Subspace>);
}

TEST(SpaceCovers, CoversIdenticalSpace) {
    using Space = ValueAxes<IntAxis>;

    static_assert(space_covers_v<Space, Space>());
    static_assert(SpaceCovers<Space, Space>);
}

TEST(SpaceCovers, DoesNotCoverNonContainedCoord) {
    using Space = ValueAxes<IntAxis>; // {1, 2, 4, 8}
    using Coord = Values<3>;          // 3 not in axis

    static_assert(!space_covers_v<Space, Coord>());
    static_assert(!SpaceCovers<Space, Coord>);
}

TEST(SpaceCovers, DoesNotCoverNonSubspace) {
    using Space  = ValueAxes<IntAxis>;         // {1, 2, 4, 8}
    using NotSub = ValueAxes<Values<1, 2, 3>>; // 3 not in axis

    static_assert(!space_covers_v<Space, NotSub>());
    static_assert(!SpaceCovers<Space, NotSub>);
}

TEST(SpaceCovers, DoesNotCoverWrongRankSpace) {
    using Space1D = ValueAxes<DeviceAxis>;
    using Space2D = ValueAxes<DeviceAxis, DTypeAxis>;

    static_assert(!space_covers_v<Space1D, Space2D>());
    static_assert(!SpaceCovers<Space1D, Space2D>);
}

TEST(SpaceCovers, DoesNotCoverNonSpaceNonPack) {
    using Space = ValueAxes<IntAxis>;

    // Neither ValuePack nor ValueSpace
    static_assert(!space_covers_v<Space, int>());
    static_assert(!SpaceCovers<Space, int>);
}

TEST(SpaceCovers, MultiAxisCoord) {
    using Space = ValueAxes<DeviceAxis, IntAxis>;
    using Coord = Values<Device::CUDA, 4>;

    static_assert(space_covers_v<Space, Coord>());
    static_assert(SpaceCovers<Space, Coord>);
}

TEST(SpaceCovers, MultiAxisSubspace) {
    using Space    = ValueAxes<DeviceAxis, IntAxis>;
    using Subspace = ValueAxes<Values<Device::CPU, Device::CUDA>, Values<1, 4>>;

    static_assert(space_covers_v<Space, Subspace>());
    static_assert(SpaceCovers<Space, Subspace>);
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
// CoordFromLinearIndex Tests
// =============================================================================

TEST(CoordFromLinearIndex, ConvertsLinearIndexToValue) {
    using Space = ValueAxes<DeviceAxis, DTypeAxis>;
    // DeviceAxis = {CPU, CUDA, Metal}, DTypeAxis = {Float32, Float64, Int32}
    // Row-major order: DType varies fastest
    // Linear 0 -> (0,0) -> Values<CPU, Float32>
    // Linear 1 -> (0,1) -> Values<CPU, Float64>
    // Linear 2 -> (0,2) -> Values<CPU, Int32>
    // Linear 3 -> (1,0) -> Values<CUDA, Float32>
    // Linear 8 -> (2,2) -> Values<Metal, Int32>

    static_assert(
        std::is_same_v<CoordFromLinearIndex_t<Space, 0>, Values<Device::CPU, DType::Float32>>);
    static_assert(
        std::is_same_v<CoordFromLinearIndex_t<Space, 1>, Values<Device::CPU, DType::Float64>>);
    static_assert(
        std::is_same_v<CoordFromLinearIndex_t<Space, 2>, Values<Device::CPU, DType::Int32>>);
    static_assert(
        std::is_same_v<CoordFromLinearIndex_t<Space, 3>, Values<Device::CUDA, DType::Float32>>);
    static_assert(
        std::is_same_v<CoordFromLinearIndex_t<Space, 5>, Values<Device::CUDA, DType::Int32>>);
    static_assert(
        std::is_same_v<CoordFromLinearIndex_t<Space, 8>, Values<Device::Metal, DType::Int32>>);
}

TEST(CoordFromLinearIndex, SingleAxisConversion) {
    using Space = ValueAxes<IntAxis>; // {1, 2, 4, 8}

    static_assert(std::is_same_v<CoordFromLinearIndex_t<Space, 0>, Values<1>>);
    static_assert(std::is_same_v<CoordFromLinearIndex_t<Space, 1>, Values<2>>);
    static_assert(std::is_same_v<CoordFromLinearIndex_t<Space, 2>, Values<4>>);
    static_assert(std::is_same_v<CoordFromLinearIndex_t<Space, 3>, Values<8>>);
}

TEST(CoordFromLinearIndex, ThreeAxisConversion) {
    using Space = ValueAxes<Values<'x', 'y'>, Values<10, 20>, Values<true, false>>;
    // 2 x 2 x 2 = 8 combinations in row-major order:
    // 0 -> (0,0,0) -> <'x', 10, true>
    // 1 -> (0,0,1) -> <'x', 10, false>
    // 2 -> (0,1,0) -> <'x', 20, true>
    // 3 -> (0,1,1) -> <'x', 20, false>
    // 4 -> (1,0,0) -> <'y', 10, true>
    // 5 -> (1,0,1) -> <'y', 10, false>
    // 6 -> (1,1,0) -> <'y', 20, true>
    // 7 -> (1,1,1) -> <'y', 20, false>

    static_assert(std::is_same_v<CoordFromLinearIndex_t<Space, 0>, Values<'x', 10, true>>);
    static_assert(std::is_same_v<CoordFromLinearIndex_t<Space, 1>, Values<'x', 10, false>>);
    static_assert(std::is_same_v<CoordFromLinearIndex_t<Space, 3>, Values<'x', 20, false>>);
    static_assert(std::is_same_v<CoordFromLinearIndex_t<Space, 4>, Values<'y', 10, true>>);
    static_assert(std::is_same_v<CoordFromLinearIndex_t<Space, 7>, Values<'y', 20, false>>);
}

TEST(CoordFromLinearIndex, MatchesCoordFromPoint) {
    using Space    = ValueAxes<DeviceAxis, IntAxis>;
    using IdxSpace = IndexSpaceOf_t<Space>;

    // Verify CoordFromLinearIndex == CoordFromPoint(PointFromLinearIndex)
    static_assert(std::is_same_v<CoordFromLinearIndex_t<Space, 0>,
                                 CoordFromPoint_t<Space, PointFromLinearIndex_t<IdxSpace, 0>>>);
    static_assert(std::is_same_v<CoordFromLinearIndex_t<Space, 5>,
                                 CoordFromPoint_t<Space, PointFromLinearIndex_t<IdxSpace, 5>>>);
    static_assert(std::is_same_v<CoordFromLinearIndex_t<Space, 11>,
                                 CoordFromPoint_t<Space, PointFromLinearIndex_t<IdxSpace, 11>>>);
}

// =============================================================================
// PointFromCoord Tests
// =============================================================================

TEST(PointFromCoord, ConvertsValueToIndex) {
    using Space = ValueAxes<DeviceAxis, DTypeAxis>;
    // DeviceAxis = {CPU, CUDA, Metal}, DTypeAxis = {Float32, Float64, Int32}

    // Values<CPU, Float32> -> Point<0, 0>
    static_assert(
        std::is_same_v<PointFromCoord_t<Space, Values<Device::CPU, DType::Float32>>, Point<0, 0>>);

    // Values<CUDA, Int32> -> Point<1, 2>
    static_assert(
        std::is_same_v<PointFromCoord_t<Space, Values<Device::CUDA, DType::Int32>>, Point<1, 2>>);

    // Values<Metal, Float64> -> Point<2, 1>
    static_assert(std::is_same_v<PointFromCoord_t<Space, Values<Device::Metal, DType::Float64>>,
                                 Point<2, 1>>);
}

TEST(PointFromCoord, SingleAxisConversion) {
    using Space = ValueAxes<IntAxis>; // {1, 2, 4, 8}

    static_assert(std::is_same_v<PointFromCoord_t<Space, Values<1>>, Point<0>>);
    static_assert(std::is_same_v<PointFromCoord_t<Space, Values<2>>, Point<1>>);
    static_assert(std::is_same_v<PointFromCoord_t<Space, Values<4>>, Point<2>>);
    static_assert(std::is_same_v<PointFromCoord_t<Space, Values<8>>, Point<3>>);
}

TEST(PointFromCoord, ThreeAxisConversion) {
    using Space = ValueAxes<Values<'x', 'y'>, Values<10, 20>, Values<true, false>>;

    // Verify a few coords
    static_assert(std::is_same_v<PointFromCoord_t<Space, Values<'x', 10, true>>, Point<0, 0, 0>>);
    static_assert(std::is_same_v<PointFromCoord_t<Space, Values<'y', 20, false>>, Point<1, 1, 1>>);
    static_assert(std::is_same_v<PointFromCoord_t<Space, Values<'x', 20, true>>, Point<0, 1, 0>>);
}

// =============================================================================
// LinearIndexFromCoord Tests
// =============================================================================

TEST(LinearIndexFromCoord, ConvertsValueToLinearIndex) {
    using Space = ValueAxes<DeviceAxis, DTypeAxis>;
    // DeviceAxis = {CPU, CUDA, Metal}, DTypeAxis = {Float32, Float64, Int32}
    // Row-major order: DType varies fastest
    // Values<CPU, Float32> -> (0,0) -> Linear 0
    // Values<CPU, Float64> -> (0,1) -> Linear 1
    // Values<CPU, Int32> -> (0,2) -> Linear 2
    // Values<CUDA, Float32> -> (1,0) -> Linear 3
    // Values<Metal, Int32> -> (2,2) -> Linear 8

    static_assert(LinearIndexFromCoord_v<Space, Values<Device::CPU, DType::Float32>>() == 0);
    static_assert(LinearIndexFromCoord_v<Space, Values<Device::CPU, DType::Float64>>() == 1);
    static_assert(LinearIndexFromCoord_v<Space, Values<Device::CPU, DType::Int32>>() == 2);
    static_assert(LinearIndexFromCoord_v<Space, Values<Device::CUDA, DType::Float32>>() == 3);
    static_assert(LinearIndexFromCoord_v<Space, Values<Device::CUDA, DType::Int32>>() == 5);
    static_assert(LinearIndexFromCoord_v<Space, Values<Device::Metal, DType::Int32>>() == 8);
}

TEST(LinearIndexFromCoord, SingleAxisConversion) {
    using Space = ValueAxes<IntAxis>; // {1, 2, 4, 8}

    static_assert(LinearIndexFromCoord_v<Space, Values<1>>() == 0);
    static_assert(LinearIndexFromCoord_v<Space, Values<2>>() == 1);
    static_assert(LinearIndexFromCoord_v<Space, Values<4>>() == 2);
    static_assert(LinearIndexFromCoord_v<Space, Values<8>>() == 3);
}

TEST(LinearIndexFromCoord, ThreeAxisConversion) {
    using Space = ValueAxes<Values<'x', 'y'>, Values<10, 20>, Values<true, false>>;
    // 2 x 2 x 2 = 8 combinations in row-major order:
    // <'x', 10, true> -> (0,0,0) -> 0
    // <'x', 10, false> -> (0,0,1) -> 1
    // <'x', 20, false> -> (0,1,1) -> 3
    // <'y', 10, true> -> (1,0,0) -> 4
    // <'y', 20, false> -> (1,1,1) -> 7

    static_assert(LinearIndexFromCoord_v<Space, Values<'x', 10, true>>() == 0);
    static_assert(LinearIndexFromCoord_v<Space, Values<'x', 10, false>>() == 1);
    static_assert(LinearIndexFromCoord_v<Space, Values<'x', 20, false>>() == 3);
    static_assert(LinearIndexFromCoord_v<Space, Values<'y', 10, true>>() == 4);
    static_assert(LinearIndexFromCoord_v<Space, Values<'y', 20, false>>() == 7);
}

TEST(LinearIndexFromCoord, MatchesLinearIndexFromPoint) {
    using Space    = ValueAxes<DeviceAxis, IntAxis>;
    using IdxSpace = IndexSpaceOf_t<Space>;

    // Verify LinearIndexFromCoord == LinearIndexFromPoint(PointFromCoord)
    static_assert(
        LinearIndexFromCoord_v<Space, Values<Device::CPU, 1>>() ==
        LinearIndexFromPoint_v<IdxSpace, PointFromCoord_t<Space, Values<Device::CPU, 1>>>());
    static_assert(
        LinearIndexFromCoord_v<Space, Values<Device::CUDA, 4>>() ==
        LinearIndexFromPoint_v<IdxSpace, PointFromCoord_t<Space, Values<Device::CUDA, 4>>>());
    static_assert(
        LinearIndexFromCoord_v<Space, Values<Device::Metal, 8>>() ==
        LinearIndexFromPoint_v<IdxSpace, PointFromCoord_t<Space, Values<Device::Metal, 8>>>());
}

TEST(LinearIndexFromCoord, RoundTripWithCoordFromLinearIndex) {
    using Space = ValueAxes<DeviceAxis, DTypeAxis>;

    // Verify LinearIndexFromCoord(CoordFromLinearIndex(i)) == i
    static_assert(LinearIndexFromCoord_v<Space, CoordFromLinearIndex_t<Space, 0>>() == 0);
    static_assert(LinearIndexFromCoord_v<Space, CoordFromLinearIndex_t<Space, 4>>() == 4);
    static_assert(LinearIndexFromCoord_v<Space, CoordFromLinearIndex_t<Space, 8>>() == 8);

    // Verify CoordFromLinearIndex(LinearIndexFromCoord(c)) == c
    using Coord1 = Values<Device::CPU, DType::Float32>;
    using Coord2 = Values<Device::CUDA, DType::Int32>;
    using Coord3 = Values<Device::Metal, DType::Float64>;
    static_assert(
        std::is_same_v<CoordFromLinearIndex_t<Space, LinearIndexFromCoord_v<Space, Coord1>()>,
                       Coord1>);
    static_assert(
        std::is_same_v<CoordFromLinearIndex_t<Space, LinearIndexFromCoord_v<Space, Coord2>()>,
                       Coord2>);
    static_assert(
        std::is_same_v<CoordFromLinearIndex_t<Space, LinearIndexFromCoord_v<Space, Coord3>()>,
                       Coord3>);
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

TEST(ValueSpaceIntegration, CoordPointRoundTrip) {
    using Space = ValueAxes<DeviceAxis, IntAxis>;
    // DeviceAxis = {CPU, CUDA, Metal} (3), IntAxis = {1, 2, 4, 8} (4)

    // Point -> Coord -> Point round trip
    using Pt1    = Point<1, 2>;
    using Coord1 = CoordFromPoint_t<Space, Pt1>;
    static_assert(std::is_same_v<Coord1, Values<Device::CUDA, 4>>);
    static_assert(std::is_same_v<PointFromCoord_t<Space, Coord1>, Pt1>);

    // Coord -> Point -> Coord round trip
    using Coord2 = Values<Device::Metal, 8>;
    using Pt2    = PointFromCoord_t<Space, Coord2>;
    static_assert(std::is_same_v<Pt2, Point<2, 3>>);
    static_assert(std::is_same_v<CoordFromPoint_t<Space, Pt2>, Coord2>);

    // Test all corners
    static_assert(
        std::is_same_v<PointFromCoord_t<Space, CoordFromPoint_t<Space, Point<0, 0>>>, Point<0, 0>>);
    static_assert(
        std::is_same_v<PointFromCoord_t<Space, CoordFromPoint_t<Space, Point<2, 3>>>, Point<2, 3>>);
}

// =============================================================================
// visit_value_space Tests
// =============================================================================

TEST(VisitValueSpace, SingleAxisVisitsAllCoords) {
    using Space = ValueAxes<IntAxis>; // {1, 2, 4, 8}

    size_t count = 0;
    std::vector<int> visited;
    visit_value_space(
        [&](auto coord) {
            ++count;
            visited.push_back(get<0>(coord));
        },
        Space{});

    EXPECT_EQ(count, 4u);
    EXPECT_EQ(visited, (std::vector<int>{1, 2, 4, 8}));
}

TEST(VisitValueSpace, TwoAxisVisitsAllCombinations) {
    using Space = ValueAxes<DeviceAxis, DTypeAxis>;
    // 3 devices x 3 dtypes = 9 combinations

    size_t count = 0;
    std::vector<std::pair<Device, DType>> visited;
    visit_value_space(
        [&](auto coord) {
            ++count;
            visited.emplace_back(get<0>(coord), get<1>(coord));
        },
        Space{});

    EXPECT_EQ(count, 9u);
    // Row-major order: last axis varies fastest
    ASSERT_EQ(visited.size(), 9u);
    EXPECT_EQ(visited[0], std::make_pair(Device::CPU, DType::Float32));
    EXPECT_EQ(visited[1], std::make_pair(Device::CPU, DType::Float64));
    EXPECT_EQ(visited[2], std::make_pair(Device::CPU, DType::Int32));
    EXPECT_EQ(visited[3], std::make_pair(Device::CUDA, DType::Float32));
    EXPECT_EQ(visited[8], std::make_pair(Device::Metal, DType::Int32));
}

TEST(VisitValueSpace, ThreeAxisVisitsAll) {
    using Space = ValueAxes<Values<'a', 'b'>, Values<1, 2>, Values<true, false>>;
    // 2 x 2 x 2 = 8 combinations

    size_t count = 0;
    visit_value_space([&](auto) { ++count; }, Space{});

    EXPECT_EQ(count, 8u);
}

TEST(VisitValueSpace, OrderMatchesIndexSpace) {
    using Space    = ValueAxes<DeviceAxis, IntAxis>;
    using IdxSpace = IndexSpaceOf_t<Space>;

    // Collect coords via value space visitation
    std::vector<std::pair<Device, int>> valueOrder;
    visit_value_space([&](auto coord) { valueOrder.emplace_back(get<0>(coord), get<1>(coord)); },
                      Space{});

    // Collect coords via index space visitation with manual conversion
    std::vector<std::pair<Device, int>> indexOrder;
    visit_index_space(
        [&](auto pt) {
            using Coord = CoordFromPoint_t<Space, decltype(pt)>;
            indexOrder.emplace_back(get<0>(Coord{}), get<1>(Coord{}));
        },
        IdxSpace{});

    EXPECT_EQ(valueOrder, indexOrder);
}

TEST(VisitValueSpaces, VisitsMultipleSpaces) {
    using Space1 = ValueAxes<Values<1, 2>>;
    using Space2 = ValueAxes<Values<'x', 'y', 'z'>>;

    size_t count = 0;
    visit_value_spaces([&](auto) { ++count; }, Space1{}, Space2{});

    EXPECT_EQ(count, 5u); // 2 + 3
}

TEST(VisitValueSpaces, MaintainsOrderAcrossSpaces) {
    using Space1 = ValueAxes<Values<1, 2>>;
    using Space2 = ValueAxes<Values<10, 20>>;

    std::vector<int> visited;
    visit_value_spaces([&](auto coord) { visited.push_back(get<0>(coord)); }, Space1{}, Space2{});

    EXPECT_EQ(visited, (std::vector<int>{1, 2, 10, 20}));
}

// =============================================================================
// SpaceTupleType Tests
// =============================================================================

TEST(SpaceTupleType, MapsSpaceToCorrectTupleType) {
    using Space1D = ValueAxes<DeviceAxis>;
    using Space2D = ValueAxes<DeviceAxis, DTypeAxis>;
    using Space3D = ValueAxes<DeviceAxis, DTypeAxis, IntAxis>;

    static_assert(std::is_same_v<SpaceTupleType_t<Space1D>, std::tuple<Device>>);
    static_assert(std::is_same_v<SpaceTupleType_t<Space2D>, std::tuple<Device, DType>>);
    static_assert(std::is_same_v<SpaceTupleType_t<Space3D>, std::tuple<Device, DType, int>>);
}

TEST(SpaceTupleType, MixedAxisTypes) {
    using Space = ValueAxes<IntAxis, CharAxis>;

    static_assert(std::is_same_v<SpaceTupleType_t<Space>, std::tuple<int, char>>);
}

TEST(SpaceTupleType, SingleElementAxes) {
    using Space = ValueAxes<Values<42>, Values<true>>;

    static_assert(std::is_same_v<SpaceTupleType_t<Space>, std::tuple<int, bool>>);
}

// =============================================================================
// TupleTypesMatchSpace Tests
// =============================================================================

TEST(TupleTypesMatchSpace, MatchingTupleTypesSatisfy) {
    using Space = ValueAxes<DeviceAxis, DTypeAxis>;

    // Correct tuple types match
    static_assert(tuple_types_match_space_v<std::tuple<Device, DType>, Space>());
    static_assert(TupleTypesMatchSpace<std::tuple<Device, DType>, Space>);
    static_assert(is_tuple_types_match_space<std::tuple<Device, DType>, Space>());
}

TEST(TupleTypesMatchSpace, ConstRefTupleTypesMatch) {
    using Space = ValueAxes<IntAxis, CharAxis>;

    // decay_t handles const/ref qualifiers
    static_assert(tuple_types_match_space_v<const std::tuple<int, char> &, Space>());
    static_assert(TupleTypesMatchSpace<const std::tuple<int, char>, Space>);
}

TEST(TupleTypesMatchSpace, WrongTypesFail) {
    using Space = ValueAxes<DeviceAxis, DTypeAxis>;

    // Wrong element types
    static_assert(!tuple_types_match_space_v<std::tuple<int, int>, Space>());
    static_assert(!tuple_types_match_space_v<std::tuple<Device, int>, Space>());
    static_assert(!tuple_types_match_space_v<std::tuple<int, DType>, Space>());
    static_assert(!is_tuple_types_match_space<std::tuple<int, int>, Space>());
}

TEST(TupleTypesMatchSpace, WrongRankFails) {
    using Space = ValueAxes<DeviceAxis, DTypeAxis>;

    // Wrong arity
    static_assert(!tuple_types_match_space_v<std::tuple<Device>, Space>());
    static_assert(!tuple_types_match_space_v<std::tuple<Device, DType, int>, Space>());
}

TEST(TupleTypesMatchSpace, SingleAxisSpace) {
    using Space = ValueAxes<IntAxis>;

    static_assert(tuple_types_match_space_v<std::tuple<int>, Space>());
    static_assert(!tuple_types_match_space_v<std::tuple<char>, Space>());
    static_assert(!tuple_types_match_space_v<std::tuple<int, int>, Space>());
}

// =============================================================================
// coordToTuple Tests
// =============================================================================

TEST(CoordToTuple, ConvertsValuesToTuple) {
    // Single value
    constexpr auto t1 = coordToTuple(Values<42>{});
    static_assert(std::is_same_v<decltype(t1), const std::tuple<int>>);
    static_assert(std::get<0>(t1) == 42);

    // Multiple values
    constexpr auto t2 = coordToTuple(Values<Device::CUDA, DType::Float64>{});
    static_assert(std::is_same_v<decltype(t2), const std::tuple<Device, DType>>);
    static_assert(std::get<0>(t2) == Device::CUDA);
    static_assert(std::get<1>(t2) == DType::Float64);

    // Three values
    constexpr auto t3 = coordToTuple(Values<'x', 10, true>{});
    static_assert(std::is_same_v<decltype(t3), const std::tuple<char, int, bool>>);
    static_assert(std::get<0>(t3) == 'x');
    static_assert(std::get<1>(t3) == 10);
    static_assert(std::get<2>(t3) == true);
}

TEST(CoordToTuple, SpaceOverloadReturnsCorrectType) {
    using Space = ValueAxes<DeviceAxis, DTypeAxis>;
    using Coord = Values<Device::Metal, DType::Int32>;

    constexpr auto t = coordToTuple(Space{}, Coord{});
    static_assert(std::is_same_v<decltype(t), const SpaceTupleType_t<Space>>);
    static_assert(std::get<0>(t) == Device::Metal);
    static_assert(std::get<1>(t) == DType::Int32);
}

TEST(CoordToTuple, RuntimeUsage) {
    auto t = coordToTuple(Values<1, 2, 4>{});
    EXPECT_EQ(std::get<0>(t), 1);
    EXPECT_EQ(std::get<1>(t), 2);
    EXPECT_EQ(std::get<2>(t), 4);
}

// =============================================================================
// spaceLinearIndex Tests
// =============================================================================

TEST(SpaceLinearIndex, SingleAxisReturnsAxisIndex) {
    using Space = ValueAxes<IntAxis>; // {1, 2, 4, 8}

    // Values in axis
    EXPECT_EQ(spaceLinearIndex(Space{}, std::make_tuple(1)), std::optional<size_t>{0});
    EXPECT_EQ(spaceLinearIndex(Space{}, std::make_tuple(2)), std::optional<size_t>{1});
    EXPECT_EQ(spaceLinearIndex(Space{}, std::make_tuple(4)), std::optional<size_t>{2});
    EXPECT_EQ(spaceLinearIndex(Space{}, std::make_tuple(8)), std::optional<size_t>{3});

    // Values not in axis
    EXPECT_EQ(spaceLinearIndex(Space{}, std::make_tuple(3)), std::nullopt);
    EXPECT_EQ(spaceLinearIndex(Space{}, std::make_tuple(16)), std::nullopt);
    EXPECT_EQ(spaceLinearIndex(Space{}, std::make_tuple(0)), std::nullopt);
}

TEST(SpaceLinearIndex, TwoAxisReturnsLinearIndex) {
    using Space = ValueAxes<DeviceAxis, DTypeAxis>;
    // DeviceAxis = {CPU, CUDA, Metal}, DTypeAxis = {Float32, Float64, Int32}
    // Row-major order: DType varies fastest
    // (CPU, Float32) -> 0, (CPU, Float64) -> 1, (CPU, Int32) -> 2
    // (CUDA, Float32) -> 3, (CUDA, Float64) -> 4, (CUDA, Int32) -> 5
    // (Metal, Float32) -> 6, (Metal, Float64) -> 7, (Metal, Int32) -> 8

    EXPECT_EQ(spaceLinearIndex(Space{}, std::make_tuple(Device::CPU, DType::Float32)),
              std::optional<size_t>{0});
    EXPECT_EQ(spaceLinearIndex(Space{}, std::make_tuple(Device::CPU, DType::Float64)),
              std::optional<size_t>{1});
    EXPECT_EQ(spaceLinearIndex(Space{}, std::make_tuple(Device::CPU, DType::Int32)),
              std::optional<size_t>{2});
    EXPECT_EQ(spaceLinearIndex(Space{}, std::make_tuple(Device::CUDA, DType::Float32)),
              std::optional<size_t>{3});
    EXPECT_EQ(spaceLinearIndex(Space{}, std::make_tuple(Device::CUDA, DType::Int32)),
              std::optional<size_t>{5});
    EXPECT_EQ(spaceLinearIndex(Space{}, std::make_tuple(Device::Metal, DType::Int32)),
              std::optional<size_t>{8});
}

TEST(SpaceLinearIndex, ThreeAxisReturnsLinearIndex) {
    using Space = ValueAxes<Values<'x', 'y'>, Values<10, 20>, Values<true, false>>;
    // 2 x 2 x 2 = 8 combinations in row-major order

    EXPECT_EQ(spaceLinearIndex(Space{}, std::make_tuple('x', 10, true)), std::optional<size_t>{0});
    EXPECT_EQ(spaceLinearIndex(Space{}, std::make_tuple('x', 10, false)), std::optional<size_t>{1});
    EXPECT_EQ(spaceLinearIndex(Space{}, std::make_tuple('x', 20, true)), std::optional<size_t>{2});
    EXPECT_EQ(spaceLinearIndex(Space{}, std::make_tuple('x', 20, false)), std::optional<size_t>{3});
    EXPECT_EQ(spaceLinearIndex(Space{}, std::make_tuple('y', 10, true)), std::optional<size_t>{4});
    EXPECT_EQ(spaceLinearIndex(Space{}, std::make_tuple('y', 10, false)), std::optional<size_t>{5});
    EXPECT_EQ(spaceLinearIndex(Space{}, std::make_tuple('y', 20, true)), std::optional<size_t>{6});
    EXPECT_EQ(spaceLinearIndex(Space{}, std::make_tuple('y', 20, false)), std::optional<size_t>{7});
}

TEST(SpaceLinearIndex, ReturnsNulloptForMissingValues) {
    using Space = ValueAxes<DeviceAxis, IntAxis>; // {CPU,CUDA,Metal} x {1,2,4,8}

    // First element missing from axis
    EXPECT_EQ(spaceLinearIndex(Space{}, std::make_tuple(Device::CPU, 3)), std::nullopt);
    EXPECT_EQ(spaceLinearIndex(Space{}, std::make_tuple(Device::CUDA, 16)), std::nullopt);

    // Both elements would need to be valid Device for type checking
    // (type mismatch would be caught at compile time via concept)
}

TEST(SpaceLinearIndex, MatchesLinearIndexFromCoord) {
    using Space  = ValueAxes<DeviceAxis, DTypeAxis>;
    using Coord1 = Values<Device::CPU, DType::Float32>;
    using Coord2 = Values<Device::CUDA, DType::Float64>;
    using Coord3 = Values<Device::Metal, DType::Int32>;

    // Verify runtime spaceLinearIndex matches compile-time LinearIndexFromCoord_v
    // for all valid coordinates
    auto coord1 = std::make_tuple(Device::CPU, DType::Float32);
    auto coord2 = std::make_tuple(Device::CUDA, DType::Float64);
    auto coord3 = std::make_tuple(Device::Metal, DType::Int32);

    // Store expected values to avoid macro comma issues with template arguments
    constexpr size_t expected1 = LinearIndexFromCoord_v<Space, Coord1>();
    constexpr size_t expected2 = LinearIndexFromCoord_v<Space, Coord2>();
    constexpr size_t expected3 = LinearIndexFromCoord_v<Space, Coord3>();

    EXPECT_EQ(spaceLinearIndex(Space{}, coord1), expected1);
    EXPECT_EQ(spaceLinearIndex(Space{}, coord2), expected2);
    EXPECT_EQ(spaceLinearIndex(Space{}, coord3), expected3);
}

TEST(SpaceLinearIndex, ConstexprUsage) {
    using Space = ValueAxes<IntAxis>;

    // Verify constexpr evaluation works
    constexpr auto idx1 = spaceLinearIndex(Space{}, std::make_tuple(4));
    static_assert(idx1.has_value());
    static_assert(*idx1 == 2);

    constexpr auto idx2 = spaceLinearIndex(Space{}, std::make_tuple(99));
    static_assert(!idx2.has_value());
}

TEST(SpaceLinearIndex, RuntimeVariableCoord) {
    using Space = ValueAxes<IntAxis>; // {1, 2, 4, 8}

    // Test with runtime-determined values
    for (int val: {1, 2, 4, 8}) {
        auto result = spaceLinearIndex(Space{}, std::make_tuple(val));
        EXPECT_TRUE(result.has_value()) << "Value " << val << " should be in space";
    }

    for (int val: {0, 3, 5, 6, 7, 9, 16}) {
        auto result = spaceLinearIndex(Space{}, std::make_tuple(val));
        EXPECT_FALSE(result.has_value()) << "Value " << val << " should not be in space";
    }
}

TEST(SpaceLinearIndex, TwoAxisRuntimeLookup) {
    using Space = ValueAxes<DeviceAxis, IntAxis>;
    // DeviceAxis = {CPU, CUDA, Metal} (3), IntAxis = {1, 2, 4, 8} (4)
    // Total: 12 combinations

    // Verify all valid combinations return correct linear indices
    size_t expectedIdx = 0;
    for (Device d: {Device::CPU, Device::CUDA, Device::Metal}) {
        for (int i: {1, 2, 4, 8}) {
            auto result = spaceLinearIndex(Space{}, std::make_tuple(d, i));
            EXPECT_EQ(result, std::optional<size_t>{expectedIdx})
                << "Failed for device=" << static_cast<int>(d) << ", int=" << i;
            ++expectedIdx;
        }
    }

    // Verify invalid int values return nullopt
    for (Device d: {Device::CPU, Device::CUDA, Device::Metal}) {
        EXPECT_EQ(spaceLinearIndex(Space{}, std::make_tuple(d, 3)), std::nullopt);
        EXPECT_EQ(spaceLinearIndex(Space{}, std::make_tuple(d, 16)), std::nullopt);
    }
}

TEST(SpaceLinearIndex, RoundTripWithCoordFromLinearIndex) {
    using Space = ValueAxes<DeviceAxis, DTypeAxis>;

    // For each linear index, convert to coord tuple and back
    for (size_t i = 0; i < Numel_v<Space>(); ++i) {
        // We need to use visit to get runtime access to compile-time coords
        size_t visitedIdx = 0;
        visit_value_space(
            [&](auto coord) {
                if (visitedIdx == i) {
                    auto tuple  = coordToTuple(coord);
                    auto result = spaceLinearIndex(Space{}, tuple);
                    EXPECT_EQ(result, std::optional<size_t>{i})
                        << "Round trip failed for linear index " << i;
                }
                ++visitedIdx;
            },
            Space{});
    }
}

} // namespace dispatch
} // namespace fvdb
