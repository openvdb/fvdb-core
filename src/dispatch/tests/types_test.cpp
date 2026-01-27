// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include "dispatch/detail.h"
#include "dispatch/iteration_policy.h"
#include "dispatch/tag_match.h"
#include "dispatch/types.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace dispatch {

// =============================================================================
// types.h - Type system primitives
// =============================================================================

//------------------------------------------------------------------------------
// extents
//------------------------------------------------------------------------------

TEST(Extents, Construction) {
    using E = extents<2, 3, 4>;
    static_assert(is_extents_v<E>());
    static_assert(!is_extents_v<int>());
}

TEST(Extents, Equality) {
    using E1 = extents<2, 3>;
    using E2 = extents<2, 3>;
    using E3 = extents<3, 2>;
    static_assert(std::is_same_v<E1, E2>);
    static_assert(!std::is_same_v<E1, E3>);
}

//------------------------------------------------------------------------------
// indices
//------------------------------------------------------------------------------

TEST(Indices, Construction) {
    using I = indices<0, 1, 2>;
    static_assert(is_indices_v<I>());
    static_assert(!is_indices_v<int>());
}

TEST(Indices, Equality) {
    using I1 = indices<0, 1>;
    using I2 = indices<0, 1>;
    using I3 = indices<1, 0>;
    static_assert(std::is_same_v<I1, I2>);
    static_assert(!std::is_same_v<I1, I3>);
}

//------------------------------------------------------------------------------
// types
//------------------------------------------------------------------------------

TEST(Types, Construction) {
    using T = types<int, float, double>;
    static_assert(is_types_v<T>());
    static_assert(!is_types_v<int>());
}

//------------------------------------------------------------------------------
// tag
//------------------------------------------------------------------------------

TEST(Tag, Construction) {
    using T = tag<1, 2, 3>;
    static_assert(is_tag_v<T>());
    static_assert(!is_tag_v<int>());
}

TEST(Tag, Equality) {
    using T1 = tag<1, 2>;
    using T2 = tag<1, 2>;
    using T3 = tag<2, 1>;
    static_assert(std::is_same_v<T1, T2>);
    static_assert(!std::is_same_v<T1, T3>);
}

//------------------------------------------------------------------------------
// axis
//------------------------------------------------------------------------------

TEST(Axis, SingleValue) {
    using A = axis<42>;
    static_assert(is_axis_v<A>());
    static_assert(extent_v<A>() == 1);
}

TEST(Axis, MultipleValues) {
    using A = axis<1, 2, 3>;
    static_assert(is_axis_v<A>());
    static_assert(extent_v<A>() == 3);
}

TEST(Axis, EnumValues) {
    using A = axis<placement::in_place, placement::out_of_place>;
    static_assert(is_axis_v<A>());
    static_assert(extent_v<A>() == 2);
}

//------------------------------------------------------------------------------
// axes
//------------------------------------------------------------------------------

TEST(Axes, Construction) {
    using A1   = axis<1, 2>;
    using A2   = axis<3, 4>;
    using Axes = axes<A1, A2>;
    static_assert(is_axes_v<Axes>());
    static_assert(!is_axes_v<int>());
}

TEST(Axes, PredefinedAxes) {
    static_assert(is_axis_v<full_placement_axis>());
    static_assert(is_axis_v<full_determinism_axis>());
    static_assert(is_axis_v<full_contiguity_axis>());
    static_assert(extent_v<full_placement_axis>() == 2);
    static_assert(extent_v<full_determinism_axis>() == 2);
    static_assert(extent_v<full_contiguity_axis>() == 2);
}

//------------------------------------------------------------------------------
// Enum types and stringification
//------------------------------------------------------------------------------

TEST(PlacementEnum, Values) {
    EXPECT_EQ(placement::in_place, placement::in_place);
    EXPECT_EQ(placement::out_of_place, placement::out_of_place);
    EXPECT_NE(placement::in_place, placement::out_of_place);
}

TEST(PlacementEnum, Stringification) {
    EXPECT_STREQ(to_string(placement::in_place), "in_place");
    EXPECT_STREQ(to_string(placement::out_of_place), "out_of_place");
}

TEST(DeterminismEnum, Values) {
    EXPECT_EQ(determinism::not_required, determinism::not_required);
    EXPECT_EQ(determinism::required, determinism::required);
    EXPECT_NE(determinism::not_required, determinism::required);
}

TEST(DeterminismEnum, Stringification) {
    EXPECT_STREQ(to_string(determinism::not_required), "not_required");
    EXPECT_STREQ(to_string(determinism::required), "required");
}

TEST(ContiguityEnum, Values) {
    EXPECT_EQ(contiguity::strided, contiguity::strided);
    EXPECT_EQ(contiguity::contiguous, contiguity::contiguous);
    EXPECT_NE(contiguity::strided, contiguity::contiguous);
}

TEST(ContiguityEnum, Stringification) {
    EXPECT_STREQ(to_string(contiguity::strided), "strided");
    EXPECT_STREQ(to_string(contiguity::contiguous), "contiguous");
}

// =============================================================================
// detail.h - Concepts
// =============================================================================

TEST(Concepts, ExtentsLike) {
    static_assert(extents_like<extents<2, 3>>);
    static_assert(!extents_like<int>);
}

TEST(Concepts, IndicesLike) {
    static_assert(indices_like<indices<0, 1>>);
    static_assert(!indices_like<int>);
}

TEST(Concepts, TypesLike) {
    static_assert(types_like<types<int, float>>);
    static_assert(!types_like<int>);
}

TEST(Concepts, TagLike) {
    static_assert(tag_like<tag<1, 2>>);
    static_assert(!tag_like<int>);
}

TEST(Concepts, AxisLike) {
    static_assert(axis_like<axis<1, 2>>);
    static_assert(!axis_like<int>);
}

TEST(Concepts, AxesLike) {
    using A = axis<1, 2>;
    static_assert(axes_like<axes<A>>);
    static_assert(!axes_like<int>);
}

TEST(Concepts, IndexSequenceLike) {
    static_assert(index_sequence_like<std::index_sequence<1, 2, 3>>);
    static_assert(!index_sequence_like<int>);
}

// =============================================================================
// detail.h - Dimensionality traits
// =============================================================================

TEST(Extent, SingleAxis) {
    using A = axis<1, 2, 3>;
    static_assert(extent_v<A>() == 3);
}

TEST(Extent, MultipleAxes) {
    using A1 = axis<1, 2>;
    using A2 = axis<3, 4>;
    static_assert(extent_v<A1>() == 2);
    static_assert(extent_v<A2>() == 2);
}

TEST(Ndim, Extents) {
    static_assert(ndim_v<extents<2, 3, 4>>() == 3);
    static_assert(ndim_v<extents<5>>() == 1);
}

TEST(Ndim, Indices) {
    static_assert(ndim_v<indices<0, 1, 2>>() == 3);
    static_assert(ndim_v<indices<0>>() == 1);
}

TEST(Ndim, Tag) {
    static_assert(ndim_v<tag<1, 2, 3>>() == 3);
    static_assert(ndim_v<tag<1>>() == 1);
}

TEST(Ndim, Axis) {
    static_assert(ndim_v<axis<1, 2, 3>>() == 1);
}

TEST(Ndim, Axes) {
    using A1   = axis<1, 2>;
    using A2   = axis<3, 4>;
    using Axes = axes<A1, A2>;
    static_assert(ndim_v<Axes>() == 2);
}

TEST(Volume, Extents) {
    static_assert(volume_v<extents<2, 3>>() == 6);
    static_assert(volume_v<extents<2, 3, 4>>() == 24);
    static_assert(volume_v<extents<5>>() == 5);
}

TEST(Volume, Axes) {
    using A1   = axis<1, 2>;
    using A2   = axis<3, 4>;
    using Axes = axes<A1, A2>;
    static_assert(volume_v<Axes>() == 4); // 2 * 2
}

TEST(IsEquidimensionalWith, Extents) {
    static_assert(is_equidimensional_with_v<extents<2, 3>, extents<4, 5>>());
    static_assert(!is_equidimensional_with_v<extents<2>, extents<2, 3>>());
}

TEST(IsEquidimensionalWith, Indices) {
    static_assert(is_equidimensional_with_v<indices<0, 1>, indices<2, 3>>());
    static_assert(!is_equidimensional_with_v<indices<0>, indices<0, 1>>());
}

// =============================================================================
// detail.h - Containment checks
// =============================================================================

TEST(IsWithin, IndexPointInIndexSpace) {
    static_assert(is_within_v<indices<1, 2>, extents<3, 4>>());
    static_assert(!is_within_v<indices<3, 2>, extents<3, 4>>()); // 3 >= 3
    static_assert(!is_within_v<indices<1, 4>, extents<3, 4>>()); // 4 >= 4
}

TEST(IsWithin, IndexSpaceInIndexSpace) {
    static_assert(is_within_v<extents<2, 3>, extents<3, 4>>());
    static_assert(!is_within_v<extents<3, 3>, extents<3, 4>>()); // 3 > 3
}

TEST(IsWithin, SingleValueInAxis) {
    using A = axis<1, 2, 3>;
    static_assert(is_within_v<axis<2>, A>());
    static_assert(!is_within_v<axis<4>, A>());
}

TEST(IsWithin, SubAxisInAxis) {
    using A = axis<1, 2, 3, 4>;
    static_assert(is_within_v<axis<2, 3>, A>());
    static_assert(!is_within_v<axis<2, 5>, A>());
}

TEST(IsWithin, TagInAxes) {
    using A1   = axis<1, 2>;
    using A2   = axis<3, 4>;
    using Axes = axes<A1, A2>;
    static_assert(is_within_v<tag<1, 3>, Axes>());
    static_assert(!is_within_v<tag<1, 5>, Axes>());
}

TEST(IsWithin, AxesInAxes) {
    using SubA1    = axis<1, 2>;
    using SubA2    = axis<3>;
    using FullA1   = axis<1, 2, 3>;
    using FullA2   = axis<3, 4>;
    using FullAxes = axes<FullA1, FullA2>;
    using SubAxes  = axes<SubA1, SubA2>;
    // SubA1 {1,2} is within FullA1 {1,2,3}, SubA2 {3} is within FullA2 {3,4}
    static_assert(is_within_v<SubAxes, FullAxes>());

    // Test case where sub is not within full
    using BadSubA1   = axis<1, 5>; // 5 is not in FullA1
    using BadSubAxes = axes<BadSubA1, SubA2>;
    static_assert(!is_within_v<BadSubAxes, FullAxes>());
}

// =============================================================================
// detail.h - Index operations
// =============================================================================

TEST(AtIndex, SingleValueAxis) {
    using A = axis<42>;
    static_assert(at_index_v<0, A>() == 42);
}

TEST(AtIndex, MultipleValueAxis) {
    using A = axis<10, 20, 30>;
    static_assert(at_index_v<0, A>() == 10);
    static_assert(at_index_v<1, A>() == 20);
    static_assert(at_index_v<2, A>() == 30);
}

TEST(AtIndices, SingleAxis) {
    using A = axis<10, 20, 30>;
    using T = at_indices_t<indices<1>, axes<A>>;
    static_assert(std::is_same_v<T, tag<20>>);
}

TEST(AtIndices, MultipleAxes) {
    using A1   = axis<1, 2>;
    using A2   = axis<3, 4>;
    using Axes = axes<A1, A2>;
    using T    = at_indices_t<indices<0, 1>, Axes>;
    static_assert(std::is_same_v<T, tag<1, 4>>);
}

TEST(IndexOf, SingleValueAxis) {
    using A = axis<42>;
    static_assert(index_of_v<42, A>() == 0);
}

TEST(IndexOf, MultipleValueAxis) {
    using A = axis<10, 20, 30>;
    static_assert(index_of_v<10, A>() == 0);
    static_assert(index_of_v<20, A>() == 1);
    static_assert(index_of_v<30, A>() == 2);
}

// Note: Runtime index_of_value function is not yet implemented
// TEST(IndexOfValue, Runtime) {
//     using A  = axis<10, 20, 30>;
//     auto opt = index_of_value(A{}, 20);
//     ASSERT_TRUE(opt.has_value());
//     EXPECT_EQ(*opt, 1u);
//
//     auto opt2 = index_of_value(A{}, 99);
//     ASSERT_FALSE(opt2.has_value());
// }

TEST(IndicesOf, SingleAxis) {
    using A = axis<10, 20, 30>;
    using I = indices_of_t<tag<20>, axes<A>>;
    static_assert(std::is_same_v<I, indices<1>>);
}

TEST(IndicesOf, MultipleAxes) {
    using A1   = axis<1, 2>;
    using A2   = axis<3, 4>;
    using Axes = axes<A1, A2>;
    using I    = indices_of_t<tag<2, 4>, Axes>;
    static_assert(std::is_same_v<I, indices<1, 1>>);
}

// =============================================================================
// detail.h - Linear indexing
// =============================================================================

TEST(LinearIndexFromIndices, SingleDimension) {
    using E = extents<5>;
    static_assert(linear_index_from_indices_v<E, indices<3>>() == 3);
    static_assert(linear_index_from_indices_v<E, indices<0>>() == 0);
}

TEST(LinearIndexFromIndices, MultipleDimensions) {
    using E = extents<2, 3>;
    static_assert(linear_index_from_indices_v<E, indices<0, 0>>() == 0);
    static_assert(linear_index_from_indices_v<E, indices<0, 1>>() == 1);
    static_assert(linear_index_from_indices_v<E, indices<1, 0>>() == 3);
    static_assert(linear_index_from_indices_v<E, indices<1, 2>>() == 5);
}

TEST(IndicesFromLinearIndex, SingleDimension) {
    using E = extents<5>;
    using I = indices_from_linear_index_t<E, 3>;
    static_assert(std::is_same_v<I, indices<3>>);
}

TEST(IndicesFromLinearIndex, MultipleDimensions) {
    using E  = extents<2, 3>;
    using I0 = indices_from_linear_index_t<E, 0>;
    using I1 = indices_from_linear_index_t<E, 1>;
    using I3 = indices_from_linear_index_t<E, 3>;
    using I5 = indices_from_linear_index_t<E, 5>;
    static_assert(std::is_same_v<I0, indices<0, 0>>);
    static_assert(std::is_same_v<I1, indices<0, 1>>);
    static_assert(std::is_same_v<I3, indices<1, 0>>);
    static_assert(std::is_same_v<I5, indices<1, 2>>);
}

TEST(LinearIndexFromTag, SingleAxis) {
    using A    = axis<10, 20, 30>;
    using Axes = axes<A>;
    static_assert(linear_index_from_tag_v<Axes, tag<10>>() == 0);
    static_assert(linear_index_from_tag_v<Axes, tag<20>>() == 1);
    static_assert(linear_index_from_tag_v<Axes, tag<30>>() == 2);
}

TEST(LinearIndexFromTag, MultipleAxes) {
    using A1   = axis<1, 2>;
    using A2   = axis<3, 4>;
    using Axes = axes<A1, A2>;
    static_assert(linear_index_from_tag_v<Axes, tag<1, 3>>() == 0);
    static_assert(linear_index_from_tag_v<Axes, tag<1, 4>>() == 1);
    static_assert(linear_index_from_tag_v<Axes, tag<2, 3>>() == 2);
    static_assert(linear_index_from_tag_v<Axes, tag<2, 4>>() == 3);
}

TEST(TagFromLinearIndex, SingleAxis) {
    using A    = axis<10, 20, 30>;
    using Axes = axes<A>;
    using T0   = tag_from_linear_index_t<Axes, 0>;
    using T1   = tag_from_linear_index_t<Axes, 1>;
    using T2   = tag_from_linear_index_t<Axes, 2>;
    static_assert(std::is_same_v<T0, tag<10>>);
    static_assert(std::is_same_v<T1, tag<20>>);
    static_assert(std::is_same_v<T2, tag<30>>);
}

TEST(TagFromLinearIndex, MultipleAxes) {
    using A1   = axis<1, 2>;
    using A2   = axis<3, 4>;
    using Axes = axes<A1, A2>;
    using T0   = tag_from_linear_index_t<Axes, 0>;
    using T1   = tag_from_linear_index_t<Axes, 1>;
    using T2   = tag_from_linear_index_t<Axes, 2>;
    using T3   = tag_from_linear_index_t<Axes, 3>;
    static_assert(std::is_same_v<T0, tag<1, 3>>);
    static_assert(std::is_same_v<T1, tag<1, 4>>);
    static_assert(std::is_same_v<T2, tag<2, 3>>);
    static_assert(std::is_same_v<T3, tag<2, 4>>);
}

TEST(LinearIndexRoundTrip, Extents) {
    using E                 = extents<2, 3, 4>;
    constexpr size_t volume = volume_v<E>();
    static_assert(volume == 24);

    // Test a few specific indices at compile time
    {
        using I               = indices_from_linear_index_t<E, 0>;
        constexpr size_t back = linear_index_from_indices_v<E, I>();
        static_assert(back == 0);
    }
    {
        using I               = indices_from_linear_index_t<E, 5>;
        constexpr size_t back = linear_index_from_indices_v<E, I>();
        static_assert(back == 5);
    }
    {
        using I               = indices_from_linear_index_t<E, 23>;
        constexpr size_t back = linear_index_from_indices_v<E, I>();
        static_assert(back == 23);
    }
    SUCCEED();
}

TEST(LinearIndexRoundTrip, Axes) {
    using A1                = axis<1, 2>;
    using A2                = axis<3, 4>;
    using Axes              = axes<A1, A2>;
    constexpr size_t volume = volume_v<Axes>();
    static_assert(volume == 4);

    // Test all indices at compile time
    {
        using T               = tag_from_linear_index_t<Axes, 0>;
        constexpr size_t back = linear_index_from_tag_v<Axes, T>();
        static_assert(back == 0);
    }
    {
        using T               = tag_from_linear_index_t<Axes, 1>;
        constexpr size_t back = linear_index_from_tag_v<Axes, T>();
        static_assert(back == 1);
    }
    {
        using T               = tag_from_linear_index_t<Axes, 2>;
        constexpr size_t back = linear_index_from_tag_v<Axes, T>();
        static_assert(back == 2);
    }
    {
        using T               = tag_from_linear_index_t<Axes, 3>;
        constexpr size_t back = linear_index_from_tag_v<Axes, T>();
        static_assert(back == 3);
    }
    SUCCEED();
}

// =============================================================================
// detail.h - Value tuple type
// =============================================================================

TEST(ValueTupleType, SingleAxis) {
    using A    = axis<1, 2>;
    using Axes = axes<A>;
    using T    = value_tuple_type_t<Axes>;
    static_assert(std::is_same_v<T, std::tuple<int>>);
}

TEST(ValueTupleType, MultipleAxes) {
    using A1   = axis<placement::in_place, placement::out_of_place>;
    using A2   = axis<determinism::required, determinism::not_required>;
    using Axes = axes<A1, A2>;
    using T    = value_tuple_type_t<Axes>;
    static_assert(std::is_same_v<T, std::tuple<placement, determinism>>);
}

// =============================================================================
// tag_match.h - is_tag_match
// =============================================================================

TEST(IsTagMatch, Basic) {
    using T = tag<1, 2, 3>;
    static_assert(is_tag_match_v<T, 1>());
    static_assert(!is_tag_match_v<T, 4>());

    // It won't match this because the first valid axis is 1, and it doesn't
    // equal 2 even though a later axis is 2.
    static_assert(!is_tag_match_v<T, 2>());

    // However, this will work because the axes get peeled off one at a time.
    static_assert(is_tag_match_v<T, 1, 2>());
}

enum class FakeDevice { CPU, GPU, PVT1 };
enum class FakeStype { F16, F32, F64, I32, I64 };
enum class FakeMethod { Clever, Quick, Lazy, Stupid };

TEST(IsTagMatch, Complex) {
    // Testing is_tag_match with realistic, mixed-type axes.

    using Tag1 = tag<FakeDevice::CPU, FakeStype::F32, 2, FakeMethod::Clever, 7>;
    // Match just FakeDevice
    static_assert(is_tag_match_v<Tag1, FakeDevice::CPU>());
    static_assert(!is_tag_match_v<Tag1, FakeDevice::GPU>());
    // Match FakeStype (skips CPU since it's a different type)
    static_assert(is_tag_match_v<Tag1, FakeDevice::CPU, FakeStype::F32>());
    static_assert(is_tag_match_v<Tag1, FakeStype::F32>());                   // skips CPU, finds F32
    static_assert(!is_tag_match_v<Tag1, FakeStype::F64>());                  // wrong value blocks
    static_assert(!is_tag_match_v<Tag1, FakeDevice::CPU, FakeStype::I32>()); // wrong stype value
    // Match int value (2), only matches if all previous types in sequence match
    static_assert(is_tag_match_v<Tag1, FakeDevice::CPU, FakeStype::F32, 2>());
    static_assert(!is_tag_match_v<Tag1, FakeDevice::CPU, FakeStype::F32, 3>());
    // Try matching method far in tag
    static_assert(is_tag_match_v<Tag1, FakeDevice::CPU, FakeStype::F32, 2, FakeMethod::Clever>());
    static_assert(!is_tag_match_v<Tag1, FakeDevice::CPU, FakeStype::F32, 2, FakeMethod::Quick>());

    // Test that repeated integer values work
    using Tag2 = tag<1, FakeDevice::GPU, 3, 1, FakeStype::F64>;
    static_assert(is_tag_match_v<Tag2, 1>());                        // matches first axis
    static_assert(!is_tag_match_v<Tag2, 2>());                       // wrong first int blocks
    static_assert(is_tag_match_v<Tag2, 1, FakeDevice::GPU>());
    static_assert(!is_tag_match_v<Tag2, 1, FakeDevice::CPU>());      // wrong device blocks
    static_assert(is_tag_match_v<Tag2, 1, FakeDevice::GPU, 3>());
    static_assert(!is_tag_match_v<Tag2, 1, FakeDevice::GPU, 5>());   // wrong second int blocks
    static_assert(is_tag_match_v<Tag2, 1, FakeDevice::GPU, 3, 1>()); // matches repeated 1, but must
                                                                     // be in order
    static_assert(
        !is_tag_match_v<Tag2, 1, FakeDevice::GPU, 1>()); // first int after GPU is 3, not 1
    static_assert(is_tag_match_v<Tag2, 1, FakeDevice::GPU, 3, 1, FakeStype::F64>());
    static_assert(
        !is_tag_match_v<Tag2, 1, FakeDevice::GPU, 3, 1, FakeStype::F32>()); // wrong stype blocks
    static_assert(!is_tag_match_v<Tag2, FakeStype::F32>()); // F64 is the only FakeStype, F32 blocks

    // Tag with shuffled types and values, only matching select axes
    using Tag3 = tag<42, FakeStype::I32, FakeMethod::Lazy, FakeDevice::PVT1, 42>;
    static_assert(is_tag_match_v<Tag3, 42>());
    static_assert(!is_tag_match_v<Tag3, 99>());                           // wrong first int blocks
    static_assert(is_tag_match_v<Tag3, 42, FakeStype::I32>());
    static_assert(!is_tag_match_v<Tag3, 42, FakeStype::F32>());           // wrong stype blocks
    static_assert(is_tag_match_v<Tag3, 42, FakeStype::I32, FakeMethod::Lazy>());
    static_assert(
        !is_tag_match_v<Tag3, 42, FakeStype::I32, FakeMethod::Clever>()); // wrong method blocks
    static_assert(is_tag_match_v<Tag3, 42, FakeStype::I32, FakeMethod::Lazy, FakeDevice::PVT1>());
    static_assert(
        !is_tag_match_v<Tag3, 42, FakeStype::I32, FakeMethod::Lazy, FakeDevice::CPU>()); // wrong
                                                                                         // device
    static_assert(
        is_tag_match_v<Tag3, FakeDevice::PVT1>()); // skips int, FakeStype, FakeMethod to find PVT1
    static_assert(!is_tag_match_v<Tag3, FakeDevice::GPU>()); // wrong device blocks

    // Matching different types in any order works (types are independent)
    static_assert(is_tag_match_v<Tag3, FakeStype::I32, 42>()); // I32 found, residual starts with 42
    static_assert(!is_tag_match_v<Tag3, FakeStype::I32, 99>()); // I32 found, but residual's first
                                                                // int is 42, not 99

    // Match last int (42)
    static_assert(
        is_tag_match_v<Tag3, 42, FakeStype::I32, FakeMethod::Lazy, FakeDevice::PVT1, 42>());
    static_assert(!is_tag_match_v<Tag3,
                                  42,
                                  FakeStype::I32,
                                  FakeMethod::Lazy,
                                  FakeDevice::PVT1,
                                  99>()); // second
                                          // int
                                          // is
                                          // 42

    // Matching subset of values (different types can be matched in any order)
    static_assert(
        is_tag_match_v<Tag3, FakeStype::I32, FakeMethod::Lazy>()); // skips int to find each type
    static_assert(!is_tag_match_v<Tag3, FakeStype::F64, FakeMethod::Lazy>()); // wrong stype blocks
    static_assert(is_tag_match_v<Tag3, 42, FakeStype::I32, FakeMethod::Lazy>());

    // Typical real-world tag: device, input stype, output stype, method
    using TagIO = tag<FakeDevice::GPU, FakeStype::F32, FakeStype::F16, FakeMethod::Quick>;

    // Match device only
    static_assert(is_tag_match_v<TagIO, FakeDevice::GPU>());
    static_assert(!is_tag_match_v<TagIO, FakeDevice::CPU>());

    // Match device + input stype
    static_assert(is_tag_match_v<TagIO, FakeDevice::GPU, FakeStype::F32>());
    static_assert(!is_tag_match_v<TagIO, FakeDevice::GPU, FakeStype::F64>());
    static_assert(is_tag_match_v<TagIO, FakeStype::F32>()); // skips GPU to find F32

    // Match input and output stypes
    static_assert(is_tag_match_v<TagIO, FakeDevice::GPU, FakeStype::F32, FakeStype::F16>());
    static_assert(!is_tag_match_v<TagIO, FakeDevice::GPU, FakeStype::F32, FakeStype::F32>());

    // Match both stypes (skips device, matches F32 then F16)
    static_assert(is_tag_match_v<TagIO, FakeStype::F32, FakeStype::F16>());
    static_assert(
        !is_tag_match_v<TagIO, FakeStype::F32, FakeStype::F64>()); // second stype is F16, not F64
    static_assert(
        !is_tag_match_v<TagIO, FakeStype::F16, FakeStype::F32>()); // first stype is F32, F16 blocks
    static_assert(is_tag_match_v<TagIO, FakeStype::F32>());        // skips GPU to find F32

    // Match device, both stypes, method
    static_assert(is_tag_match_v<TagIO,
                                 FakeDevice::GPU,
                                 FakeStype::F32,
                                 FakeStype::F16,
                                 FakeMethod::Quick>());
    static_assert(!is_tag_match_v<TagIO,
                                  FakeDevice::GPU,
                                  FakeStype::F32,
                                  FakeStype::F16,
                                  FakeMethod::Clever>());

    // Try to match output stype only (can't skip previous types)
    static_assert(!is_tag_match_v<TagIO, FakeStype::F16>());

    // Try matching device and output stype only (should block: must match axes in order)
    static_assert(!is_tag_match_v<TagIO, FakeDevice::GPU, FakeStype::F16>());

    // Match device, input stype, and method (F16 preserved in residual, not blocking)
    static_assert(is_tag_match_v<TagIO, FakeDevice::GPU, FakeStype::F32, FakeMethod::Quick>());

    // Fully match including method
    static_assert(is_tag_match_v<TagIO,
                                 FakeDevice::GPU,
                                 FakeStype::F32,
                                 FakeStype::F16,
                                 FakeMethod::Quick>());

    // Add an integer axis before input stype to check detection
    using TagIO2 = tag<FakeDevice::GPU, 8, FakeStype::F32, FakeStype::F16, FakeMethod::Quick>;
    static_assert(is_tag_match_v<TagIO2, FakeDevice::GPU, 8, FakeStype::F32>()); // correct sequence
    static_assert(
        !is_tag_match_v<TagIO2, FakeDevice::GPU, 9, FakeStype::F32>());          // wrong int blocks
    static_assert(
        !is_tag_match_v<TagIO2, FakeDevice::CPU, 8, FakeStype::F32>()); // wrong device blocks
    static_assert(
        is_tag_match_v<TagIO2, 8, FakeDevice::GPU>()); // finds 8 (skips GPU), GPU in residual
    static_assert(!is_tag_match_v<TagIO2, 8, FakeDevice::CPU>());   // GPU in residual, not CPU
    static_assert(
        is_tag_match_v<TagIO2, FakeDevice::GPU, FakeStype::F32>()); // skips 8 when looking for
                                                                    // FakeStype
    static_assert(!is_tag_match_v<TagIO2, FakeDevice::GPU, FakeStype::I64>()); // wrong stype blocks

    // Matching trailing axes only fails (first stype is F32, not F16)
    static_assert(!is_tag_match_v<TagIO2, FakeStype::F16, FakeMethod::Quick>());
    static_assert(!is_tag_match_v<TagIO2, FakeMethod::Lazy>()); // method is Quick, not Lazy
}

// =============================================================================
// iteration_policy.h - N-D iteration utilities
// =============================================================================

TEST(IterationPolicy, RowMajor_1D) {
    // 1D case: linear index == output index
    std::array<int64_t, 1> shape = {5};
    std::array<int64_t, 1> indices;

    linear_to_nd<1, row_major>(0, shape, indices);
    EXPECT_EQ(indices[0], 0);

    linear_to_nd<1, row_major>(3, shape, indices);
    EXPECT_EQ(indices[0], 3);

    linear_to_nd<1, row_major>(4, shape, indices);
    EXPECT_EQ(indices[0], 4);
}

TEST(IterationPolicy, RowMajor_2D) {
    // 2D case: shape [2, 3] means last index varies fastest
    // Linear order: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2)
    std::array<int64_t, 2> shape = {2, 3};
    std::array<int64_t, 2> indices;

    linear_to_nd<2, row_major>(0, shape, indices);
    EXPECT_EQ(indices[0], 0);
    EXPECT_EQ(indices[1], 0);

    linear_to_nd<2, row_major>(1, shape, indices);
    EXPECT_EQ(indices[0], 0);
    EXPECT_EQ(indices[1], 1);

    linear_to_nd<2, row_major>(2, shape, indices);
    EXPECT_EQ(indices[0], 0);
    EXPECT_EQ(indices[1], 2);

    linear_to_nd<2, row_major>(3, shape, indices);
    EXPECT_EQ(indices[0], 1);
    EXPECT_EQ(indices[1], 0);

    linear_to_nd<2, row_major>(5, shape, indices);
    EXPECT_EQ(indices[0], 1);
    EXPECT_EQ(indices[1], 2);
}

TEST(IterationPolicy, ColMajor_2D) {
    // 2D case with col_major: first index varies fastest
    // Linear order: (0,0), (1,0), (0,1), (1,1), (0,2), (1,2)
    std::array<int64_t, 2> shape = {2, 3};
    std::array<int64_t, 2> indices;

    linear_to_nd<2, col_major>(0, shape, indices);
    EXPECT_EQ(indices[0], 0);
    EXPECT_EQ(indices[1], 0);

    linear_to_nd<2, col_major>(1, shape, indices);
    EXPECT_EQ(indices[0], 1);
    EXPECT_EQ(indices[1], 0);

    linear_to_nd<2, col_major>(2, shape, indices);
    EXPECT_EQ(indices[0], 0);
    EXPECT_EQ(indices[1], 1);

    linear_to_nd<2, col_major>(3, shape, indices);
    EXPECT_EQ(indices[0], 1);
    EXPECT_EQ(indices[1], 1);

    linear_to_nd<2, col_major>(5, shape, indices);
    EXPECT_EQ(indices[0], 1);
    EXPECT_EQ(indices[1], 2);
}

TEST(IterationPolicy, RowMajor_3D) {
    // 3D case: shape [2, 3, 4]
    std::array<int64_t, 3> shape = {2, 3, 4};
    std::array<int64_t, 3> indices;

    // First element
    linear_to_nd<3, row_major>(0, shape, indices);
    EXPECT_EQ(indices[0], 0);
    EXPECT_EQ(indices[1], 0);
    EXPECT_EQ(indices[2], 0);

    // Last element in first "row"
    linear_to_nd<3, row_major>(3, shape, indices);
    EXPECT_EQ(indices[0], 0);
    EXPECT_EQ(indices[1], 0);
    EXPECT_EQ(indices[2], 3);

    // First element of second row in first plane
    linear_to_nd<3, row_major>(4, shape, indices);
    EXPECT_EQ(indices[0], 0);
    EXPECT_EQ(indices[1], 1);
    EXPECT_EQ(indices[2], 0);

    // First element of second plane
    linear_to_nd<3, row_major>(12, shape, indices);
    EXPECT_EQ(indices[0], 1);
    EXPECT_EQ(indices[1], 0);
    EXPECT_EQ(indices[2], 0);

    // Last element
    linear_to_nd<3, row_major>(23, shape, indices);
    EXPECT_EQ(indices[0], 1);
    EXPECT_EQ(indices[1], 2);
    EXPECT_EQ(indices[2], 3);
}

TEST(IterationPolicy, NdToLinear_RowMajor) {
    std::array<int64_t, 2> shape = {2, 3};

    EXPECT_EQ((nd_to_linear<2, row_major>(std::array<int64_t, 2>{0, 0}, shape)), 0);
    EXPECT_EQ((nd_to_linear<2, row_major>(std::array<int64_t, 2>{0, 1}, shape)), 1);
    EXPECT_EQ((nd_to_linear<2, row_major>(std::array<int64_t, 2>{0, 2}, shape)), 2);
    EXPECT_EQ((nd_to_linear<2, row_major>(std::array<int64_t, 2>{1, 0}, shape)), 3);
    EXPECT_EQ((nd_to_linear<2, row_major>(std::array<int64_t, 2>{1, 2}, shape)), 5);
}

TEST(IterationPolicy, NdToLinear_ColMajor) {
    std::array<int64_t, 2> shape = {2, 3};

    EXPECT_EQ((nd_to_linear<2, col_major>(std::array<int64_t, 2>{0, 0}, shape)), 0);
    EXPECT_EQ((nd_to_linear<2, col_major>(std::array<int64_t, 2>{1, 0}, shape)), 1);
    EXPECT_EQ((nd_to_linear<2, col_major>(std::array<int64_t, 2>{0, 1}, shape)), 2);
    EXPECT_EQ((nd_to_linear<2, col_major>(std::array<int64_t, 2>{1, 1}, shape)), 3);
    EXPECT_EQ((nd_to_linear<2, col_major>(std::array<int64_t, 2>{1, 2}, shape)), 5);
}

TEST(IterationPolicy, RoundTrip_RowMajor) {
    // Test that linear_to_nd and nd_to_linear are inverses
    std::array<int64_t, 3> shape = {2, 3, 4};
    int64_t total                = shape_volume<3>(shape);

    for (int64_t i = 0; i < total; ++i) {
        auto indices = linear_to_nd<3, row_major>(i, shape);
        auto back    = nd_to_linear<3, row_major>(indices, shape);
        EXPECT_EQ(back, i) << "RoundTrip failed for linear index " << i;
    }
}

TEST(IterationPolicy, RoundTrip_ColMajor) {
    std::array<int64_t, 3> shape = {2, 3, 4};
    int64_t total                = shape_volume<3>(shape);

    for (int64_t i = 0; i < total; ++i) {
        auto indices = linear_to_nd<3, col_major>(i, shape);
        auto back    = nd_to_linear<3, col_major>(indices, shape);
        EXPECT_EQ(back, i) << "RoundTrip failed for linear index " << i;
    }
}

TEST(IterationPolicy, ShapeVolume) {
    EXPECT_EQ((shape_volume<1>(std::array<int64_t, 1>{5})), 5);
    EXPECT_EQ((shape_volume<2>(std::array<int64_t, 2>{2, 3})), 6);
    EXPECT_EQ((shape_volume<3>(std::array<int64_t, 3>{2, 3, 4})), 24);
}

TEST(IterationPolicy, ReturnByValue) {
    // Test the return-by-value overload
    std::array<int64_t, 2> shape = {2, 3};

    auto indices = linear_to_nd<2, row_major>(4, shape);
    EXPECT_EQ(indices[0], 1);
    EXPECT_EQ(indices[1], 1);
}

} // namespace dispatch
