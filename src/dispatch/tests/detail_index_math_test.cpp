// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Tests for the index math and containment machinery in detail.h.
// These tests use axes_storage and tag_storage directly (not the normalizing
// aliases) because the index math is positional — it operates on the internal
// ordered representation regardless of normalization.
//
#include "dispatch/detail.h"
#include "dispatch/enums.h"
#include "dispatch/visit_spaces.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace dispatch {

// ============================================================================
// Test enums — each axis needs a unique value type
// ============================================================================

enum class device { cpu, gpu };
enum class stype { f32, f64 };
enum class method { fast, slow, auto_select };

template <>
struct type_label<device> {
    static consteval auto
    value() {
        return fixed_label("test.device");
    }
};

template <>
struct type_label<stype> {
    static consteval auto
    value() {
        return fixed_label("test.stype");
    }
};

template <>
struct type_label<method> {
    static consteval auto
    value() {
        return fixed_label("test.method");
    }
};

// Convenience axis types for tests
using device_axis = axis<device::cpu, device::gpu>;
using stype_axis  = axis<stype::f32, stype::f64>;
using method_axis = axis<method::fast, method::slow, method::auto_select>;

// ============================================================================
// Concepts
// ============================================================================

TEST(Concepts, ExtentsLike) {
    static_assert(extents_like<extents<2, 3>>);
    static_assert(!extents_like<int>);
}

TEST(Concepts, IndicesLike) {
    static_assert(indices_like<indices<0, 1>>);
    static_assert(!indices_like<int>);
}

TEST(Concepts, TagLike) {
    static_assert(tag_like<tag_storage<device::cpu, stype::f32>>);
    static_assert(!tag_like<int>);
}

TEST(Concepts, AxesLike) {
    static_assert(axes_like<axes_storage<device_axis, stype_axis>>);
    static_assert(!axes_like<int>);
}

TEST(Concepts, IndexSequenceLike) {
    static_assert(index_sequence_like<std::index_sequence<1, 2, 3>>);
    static_assert(!index_sequence_like<int>);
}

// ============================================================================
// Ndim
// ============================================================================

TEST(Ndim, Extents) {
    static_assert(ndim_v<extents<2, 3, 4>>() == 3);
    static_assert(ndim_v<extents<5>>() == 1);
}

TEST(Ndim, Indices) {
    static_assert(ndim_v<indices<0, 1, 2>>() == 3);
    static_assert(ndim_v<indices<0>>() == 1);
}

TEST(Ndim, TagStorage) {
    static_assert(ndim_v<tag_storage<device::cpu, stype::f32>>() == 2);
    static_assert(ndim_v<tag_storage<device::cpu>>() == 1);
}

TEST(Ndim, Axis) {
    static_assert(ndim_v<device_axis>() == 1);
    static_assert(ndim_v<method_axis>() == 1);
}

TEST(Ndim, AxesStorage) {
    static_assert(ndim_v<axes_storage<device_axis, stype_axis>>() == 2);
}

// ============================================================================
// Volume
// ============================================================================

TEST(Volume, Extents) {
    static_assert(volume_v<extents<2, 3>>() == 6);
    static_assert(volume_v<extents<2, 3, 4>>() == 24);
    static_assert(volume_v<extents<5>>() == 5);
}

TEST(Volume, AxesStorage) {
    static_assert(volume_v<axes_storage<device_axis, stype_axis>>() == 4); // 2 * 2
    static_assert(volume_v<axes_storage<device_axis, method_axis>>() == 6); // 2 * 3
}

// ============================================================================
// ExtentAt, VolumeSuffix
// ============================================================================

TEST(ExtentAt, Basic) {
    using E = extents<2, 3, 4>;
    static_assert(extent_at_v<0, E>() == 2);
    static_assert(extent_at_v<1, E>() == 3);
    static_assert(extent_at_v<2, E>() == 4);
}

TEST(VolumeSuffix, Basic) {
    using E = extents<2, 3, 4>;
    static_assert(volume_suffix_v<0, E>() == 12); // 3 * 4
    static_assert(volume_suffix_v<1, E>() == 4);  // 4
    static_assert(volume_suffix_v<2, E>() == 1);  // last
}

// ============================================================================
// Equidimensional
// ============================================================================

TEST(Equidimensional, Extents) {
    static_assert(is_equidimensional_with_v<extents<2, 3>, extents<4, 5>>());
    static_assert(!is_equidimensional_with_v<extents<2>, extents<2, 3>>());
}

TEST(Equidimensional, Indices) {
    static_assert(is_equidimensional_with_v<indices<0, 1>, indices<2, 3>>());
    static_assert(!is_equidimensional_with_v<indices<0>, indices<0, 1>>());
}

// ============================================================================
// NonEmpty
// ============================================================================

TEST(NonEmpty, Extents) {
    static_assert(non_empty<extents<2, 3>>);
}

TEST(NonEmpty, AxesStorage) {
    static_assert(non_empty<axes_storage<device_axis, stype_axis>>);
}

// ============================================================================
// Within (containment)
// ============================================================================

TEST(IsWithin, IndexPointInIndexSpace) {
    static_assert(is_within_v<indices<1, 2>, extents<3, 4>>());
    static_assert(!is_within_v<indices<3, 2>, extents<3, 4>>());
    static_assert(!is_within_v<indices<1, 4>, extents<3, 4>>());
}

TEST(IsWithin, IndexSpaceInIndexSpace) {
    static_assert(is_within_v<extents<2, 3>, extents<3, 4>>());
    static_assert(!is_within_v<extents<3, 3>, extents<3, 4>>());
}

TEST(IsWithin, SingleValueInAxis) {
    static_assert(is_within_v<axis<device::cpu>, device_axis>());
    static_assert(!is_within_v<axis<method::fast>, device_axis>()); // wrong type won't compile
}

TEST(IsWithin, SubAxisInAxis) {
    static_assert(is_within_v<axis<stype::f32>, stype_axis>());
    static_assert(is_within_v<stype_axis, stype_axis>()); // identity
}

TEST(IsWithin, TagInAxes) {
    using TestAxes = axes_storage<device_axis, stype_axis>;
    static_assert(is_within_v<tag_storage<device::cpu, stype::f32>, TestAxes>());
    static_assert(is_within_v<tag_storage<device::gpu, stype::f64>, TestAxes>());
}

TEST(IsWithin, AxesInAxes) {
    using FullAxes = axes_storage<device_axis, stype_axis>;
    using SubAxes  = axes_storage<axis<device::cpu>, axis<stype::f32>>;
    static_assert(is_within_v<SubAxes, FullAxes>());
}

// ============================================================================
// AtIndex
// ============================================================================

TEST(AtIndex, Basic) {
    static_assert(at_index_v<0, device_axis>() == device::cpu);
    static_assert(at_index_v<1, device_axis>() == device::gpu);
}

TEST(AtIndex, ThreeValues) {
    static_assert(at_index_v<0, method_axis>() == method::fast);
    static_assert(at_index_v<1, method_axis>() == method::slow);
    static_assert(at_index_v<2, method_axis>() == method::auto_select);
}

// ============================================================================
// AtIndices
// ============================================================================

TEST(AtIndices, SingleAxis) {
    using TestAxes = axes_storage<device_axis>;
    using T        = at_indices_t<indices<1>, TestAxes>;
    static_assert(std::is_same_v<T, tag_storage<device::gpu>>);
}

TEST(AtIndices, MultipleAxes) {
    using TestAxes = axes_storage<device_axis, stype_axis>;
    using T        = at_indices_t<indices<0, 1>, TestAxes>;
    static_assert(std::is_same_v<T, tag_storage<device::cpu, stype::f64>>);
}

// ============================================================================
// IndexOf
// ============================================================================

TEST(IndexOf, Basic) {
    static_assert(index_of_v<device::cpu, device_axis>() == 0);
    static_assert(index_of_v<device::gpu, device_axis>() == 1);
}

TEST(IndexOf, ThreeValues) {
    static_assert(index_of_v<method::fast, method_axis>() == 0);
    static_assert(index_of_v<method::slow, method_axis>() == 1);
    static_assert(index_of_v<method::auto_select, method_axis>() == 2);
}

// ============================================================================
// IndicesOf
// ============================================================================

TEST(IndicesOf, SingleAxis) {
    using TestAxes = axes_storage<device_axis>;
    using I        = indices_of_t<tag_storage<device::gpu>, TestAxes>;
    static_assert(std::is_same_v<I, indices<1>>);
}

TEST(IndicesOf, MultipleAxes) {
    using TestAxes = axes_storage<device_axis, stype_axis>;
    using I        = indices_of_t<tag_storage<device::gpu, stype::f64>, TestAxes>;
    static_assert(std::is_same_v<I, indices<1, 1>>);
}

// ============================================================================
// Linear index from indices
// ============================================================================

TEST(LinearIndexFromIndices, SingleDimension) {
    using E = extents<5>;
    static_assert(linear_index_from_indices_v<E, indices<0>>() == 0);
    static_assert(linear_index_from_indices_v<E, indices<3>>() == 3);
}

TEST(LinearIndexFromIndices, MultipleDimensions) {
    using E = extents<2, 3>;
    static_assert(linear_index_from_indices_v<E, indices<0, 0>>() == 0);
    static_assert(linear_index_from_indices_v<E, indices<0, 1>>() == 1);
    static_assert(linear_index_from_indices_v<E, indices<1, 0>>() == 3);
    static_assert(linear_index_from_indices_v<E, indices<1, 2>>() == 5);
}

// ============================================================================
// Indices from linear index
// ============================================================================

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

// ============================================================================
// Linear index from tag
// ============================================================================

TEST(LinearIndexFromTag, SingleAxis) {
    using TestAxes = axes_storage<device_axis>;
    static_assert(linear_index_from_tag_v<TestAxes, tag_storage<device::cpu>>() == 0);
    static_assert(linear_index_from_tag_v<TestAxes, tag_storage<device::gpu>>() == 1);
}

TEST(LinearIndexFromTag, MultipleAxes) {
    using TestAxes = axes_storage<device_axis, stype_axis>;
    static_assert(linear_index_from_tag_v<TestAxes, tag_storage<device::cpu, stype::f32>>() == 0);
    static_assert(linear_index_from_tag_v<TestAxes, tag_storage<device::cpu, stype::f64>>() == 1);
    static_assert(linear_index_from_tag_v<TestAxes, tag_storage<device::gpu, stype::f32>>() == 2);
    static_assert(linear_index_from_tag_v<TestAxes, tag_storage<device::gpu, stype::f64>>() == 3);
}

// ============================================================================
// Tag from linear index
// ============================================================================

TEST(TagFromLinearIndex, SingleAxis) {
    using TestAxes = axes_storage<device_axis>;
    using T0       = tag_from_linear_index_t<TestAxes, 0>;
    using T1       = tag_from_linear_index_t<TestAxes, 1>;
    static_assert(std::is_same_v<T0, tag_storage<device::cpu>>);
    static_assert(std::is_same_v<T1, tag_storage<device::gpu>>);
}

TEST(TagFromLinearIndex, MultipleAxes) {
    using TestAxes = axes_storage<device_axis, stype_axis>;
    using T0       = tag_from_linear_index_t<TestAxes, 0>;
    using T1       = tag_from_linear_index_t<TestAxes, 1>;
    using T2       = tag_from_linear_index_t<TestAxes, 2>;
    using T3       = tag_from_linear_index_t<TestAxes, 3>;
    static_assert(std::is_same_v<T0, tag_storage<device::cpu, stype::f32>>);
    static_assert(std::is_same_v<T1, tag_storage<device::cpu, stype::f64>>);
    static_assert(std::is_same_v<T2, tag_storage<device::gpu, stype::f32>>);
    static_assert(std::is_same_v<T3, tag_storage<device::gpu, stype::f64>>);
}

// ============================================================================
// Round-trip: linear index <-> indices
// ============================================================================

TEST(LinearIndexRoundTrip, Extents) {
    using E                 = extents<2, 3, 4>;
    constexpr size_t vol    = volume_v<E>();
    static_assert(vol == 24);

    // Spot-check a few
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
    using TestAxes          = axes_storage<device_axis, stype_axis>;
    constexpr size_t vol    = volume_v<TestAxes>();
    static_assert(vol == 4);

    {
        using T               = tag_from_linear_index_t<TestAxes, 0>;
        constexpr size_t back = linear_index_from_tag_v<TestAxes, T>();
        static_assert(back == 0);
    }
    {
        using T               = tag_from_linear_index_t<TestAxes, 3>;
        constexpr size_t back = linear_index_from_tag_v<TestAxes, T>();
        static_assert(back == 3);
    }
    SUCCEED();
}

// ============================================================================
// ExtentsOf
// ============================================================================

TEST(ExtentsOf, Basic) {
    using TestAxes = axes_storage<device_axis, method_axis>;
    using E        = extents_of_t<TestAxes>;
    static_assert(std::is_same_v<E, extents<2, 3>>);
}

// ============================================================================
// ValueTupleType
// ============================================================================

TEST(ValueTupleType, SingleAxis) {
    using TestAxes = axes_storage<device_axis>;
    using T        = value_tuple_type_t<TestAxes>;
    static_assert(std::is_same_v<T, std::tuple<device>>);
}

TEST(ValueTupleType, MultipleAxes) {
    using TestAxes = axes_storage<full_placement_axis, full_determinism_axis>;
    using T        = value_tuple_type_t<TestAxes>;
    static_assert(std::is_same_v<T, std::tuple<placement, determinism>>);
}

// ============================================================================
// VisitSpaces
// ============================================================================

TEST(VisitAxesSpace, CountsAllCoordinates) {
    using TestAxes = axes_storage<device_axis, stype_axis>;
    int count      = 0;
    auto visitor   = [&count](auto /*tag*/) { ++count; };
    visit_axes_space(visitor, TestAxes{});
    EXPECT_EQ(count, 4); // 2 * 2
}

TEST(VisitAxesSpace, ThreeAxes) {
    using TestAxes = axes_storage<device_axis, stype_axis, full_placement_axis>;
    int count      = 0;
    auto visitor   = [&count](auto /*tag*/) { ++count; };
    visit_axes_space(visitor, TestAxes{});
    EXPECT_EQ(count, 8); // 2 * 2 * 2
}

TEST(VisitExtentsSpace, Basic) {
    using E  = extents<2, 3>;
    int count = 0;
    auto visitor = [&count](auto /*indices*/) { ++count; };
    visit_extents_space(visitor, E{});
    EXPECT_EQ(count, 6); // 2 * 3
}

} // namespace dispatch
