// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Tests for the label system (fixed_label, type_label, named),
// self-normalizing tag, axis, and self-normalizing axes.
//
#include "dispatch/axes.h"
#include "dispatch/axis.h"
#include "dispatch/consteval_types.h"
#include "dispatch/enums.h"
#include "dispatch/label.h"
#include "dispatch/label_sorted.h"
#include "dispatch/tag.h"

#include <gtest/gtest.h>

#include <type_traits>

namespace dispatch {

// ============================================================================
// Test enums with type_label specializations
// ============================================================================

enum class color { red, green, blue };
enum class shape { circle, square, triangle };
enum class size { small, medium, large };

template <>
struct type_label<color> {
    static consteval auto
    value() {
        return fixed_label("test.color");
    }
};

template <>
struct type_label<shape> {
    static consteval auto
    value() {
        return fixed_label("test.shape");
    }
};

template <>
struct type_label<size> {
    static consteval auto
    value() {
        return fixed_label("test.size");
    }
};

// ============================================================================
// fixed_label
// ============================================================================

TEST(FixedLabel, Construction) {
    constexpr auto label = fixed_label("hello");
    static_assert(label.size() == 5);
    static_assert(label[0] == 'h');
    static_assert(label[4] == 'o');
}

TEST(FixedLabel, Equality) {
    constexpr auto a = fixed_label("abc");
    constexpr auto b = fixed_label("abc");
    constexpr auto c = fixed_label("xyz");
    static_assert(a == b);
    static_assert(!(a == c));
}

TEST(FixedLabel, Comparison) {
    constexpr auto a = fixed_label("alpha");
    constexpr auto b = fixed_label("beta");
    constexpr auto c = fixed_label("alpha");
    static_assert(compare_fixed_labels(a, b) < 0);
    static_assert(compare_fixed_labels(b, a) > 0);
    static_assert(compare_fixed_labels(a, c) == 0);
}

TEST(FixedLabel, CrossLengthComparison) {
    constexpr auto short_label = fixed_label("ab");
    constexpr auto long_label  = fixed_label("abc");
    static_assert(compare_fixed_labels(short_label, long_label) < 0);
    static_assert(compare_fixed_labels(long_label, short_label) > 0);
}

// ============================================================================
// type_label
// ============================================================================

TEST(TypeLabel, DispatchEnums) {
    // Built-in dispatch enums have type_label specializations
    constexpr auto p = type_label<placement>::value();
    constexpr auto d = type_label<determinism>::value();
    constexpr auto c = type_label<contiguity>::value();
    static_assert(p == fixed_label("dispatch.placement"));
    static_assert(d == fixed_label("dispatch.determinism"));
    static_assert(c == fixed_label("dispatch.contiguity"));
}

TEST(TypeLabel, TestEnums) {
    constexpr auto cl = type_label<color>::value();
    constexpr auto sh = type_label<shape>::value();
    constexpr auto sz = type_label<size>::value();
    static_assert(cl == fixed_label("test.color"));
    static_assert(sh == fixed_label("test.shape"));
    static_assert(sz == fixed_label("test.size"));
}

// ============================================================================
// named
// ============================================================================

TEST(Named, Construction) {
    constexpr auto v = named<fixed_label("input_stype"), int>{42};
    static_assert(v.value == 42);
}

TEST(Named, Equality) {
    constexpr auto a = named<fixed_label("slot"), int>{1};
    constexpr auto b = named<fixed_label("slot"), int>{1};
    constexpr auto c = named<fixed_label("slot"), int>{2};
    static_assert(a == b);
    static_assert(!(a == c));
}

TEST(Named, DistinctTypes) {
    // Same underlying value, different labels -> different types
    using A = named<fixed_label("input"), int>;
    using B = named<fixed_label("output"), int>;
    static_assert(!std::is_same_v<A, B>);
}

TEST(Named, SelfRegisteringTypeLabel) {
    // type_label for named<Label, T> returns Label
    using N = named<fixed_label("my.label"), int>;
    constexpr auto label = type_label<N>::value();
    static_assert(label == fixed_label("my.label"));
}

TEST(Named, IsNamedTrait) {
    using N = named<fixed_label("slot"), int>;
    static_assert(is_named_v<N>());
    static_assert(!is_named_v<int>());
    static_assert(!is_named_v<placement>());
}

// ============================================================================
// axis
// ============================================================================

TEST(Axis, SingleValue) {
    using A = axis<color::red>;
    static_assert(is_axis_v<A>());
    static_assert(extent_v<A>() == 1);
}

TEST(Axis, MultipleValues) {
    using A = axis<color::red, color::green, color::blue>;
    static_assert(is_axis_v<A>());
    static_assert(extent_v<A>() == 3);
}

TEST(Axis, ValueType) {
    using A = axis<placement::in_place, placement::out_of_place>;
    static_assert(std::is_same_v<axis_value_type_t<A>, placement>);
}

TEST(Axis, IsNotAxis) {
    static_assert(!is_axis_v<int>());
    static_assert(!is_axis_v<placement>());
}

TEST(Axis, PredefinedAxes) {
    static_assert(is_axis_v<full_placement_axis>());
    static_assert(is_axis_v<full_determinism_axis>());
    static_assert(is_axis_v<full_contiguity_axis>());
    static_assert(extent_v<full_placement_axis>() == 2);
    static_assert(extent_v<full_determinism_axis>() == 2);
    static_assert(extent_v<full_contiguity_axis>() == 2);
}

// ============================================================================
// tag: self-normalizing, unique types
// ============================================================================

TEST(Tag, SingleValue) {
    using T = tag<color::red>;
    static_assert(is_tag_v<T>());
}

TEST(Tag, MultipleUniqueTypes) {
    using T = tag<color::red, shape::circle, size::small>;
    static_assert(is_tag_v<T>());
}

TEST(Tag, SelfNormalizing) {
    // Different orderings produce the same type
    using T1 = tag<color::red, shape::circle>;
    using T2 = tag<shape::circle, color::red>;
    static_assert(std::is_same_v<T1, T2>);
}

TEST(Tag, SelfNormalizingThreeValues) {
    using T1 = tag<color::red, shape::circle, size::small>;
    using T2 = tag<size::small, color::red, shape::circle>;
    using T3 = tag<shape::circle, size::small, color::red>;
    static_assert(std::is_same_v<T1, T2>);
    static_assert(std::is_same_v<T2, T3>);
}

TEST(Tag, DispatchEnums) {
    using T1 = tag<placement::in_place, determinism::required, contiguity::contiguous>;
    using T2 = tag<contiguity::contiguous, placement::in_place, determinism::required>;
    static_assert(std::is_same_v<T1, T2>);
    static_assert(is_tag_v<T1>());
}

TEST(Tag, IsNotTag) {
    static_assert(!is_tag_v<int>());
    static_assert(!is_tag_v<axis<1, 2>>());
}

// ============================================================================
// axes: self-normalizing, unique axis value types
// ============================================================================

TEST(Axes, SingleAxis) {
    using A    = axis<color::red, color::green, color::blue>;
    using MyAxes = axes<A>;
    static_assert(is_axes_v<MyAxes>());
}

TEST(Axes, MultipleAxes) {
    using ColorAxis = axis<color::red, color::green>;
    using ShapeAxis = axis<shape::circle, shape::square>;
    using MyAxes    = axes<ColorAxis, ShapeAxis>;
    static_assert(is_axes_v<MyAxes>());
}

TEST(Axes, SelfNormalizing) {
    using ColorAxis = axis<color::red, color::green>;
    using ShapeAxis = axis<shape::circle, shape::square>;

    using Axes1 = axes<ColorAxis, ShapeAxis>;
    using Axes2 = axes<ShapeAxis, ColorAxis>;
    static_assert(std::is_same_v<Axes1, Axes2>);
}

TEST(Axes, SelfNormalizingThreeAxes) {
    using ColorAxis = axis<color::red, color::green>;
    using ShapeAxis = axis<shape::circle, shape::square>;
    using SizeAxis  = axis<size::small, size::large>;

    using Axes1 = axes<ColorAxis, ShapeAxis, SizeAxis>;
    using Axes2 = axes<SizeAxis, ColorAxis, ShapeAxis>;
    using Axes3 = axes<ShapeAxis, SizeAxis, ColorAxis>;
    static_assert(std::is_same_v<Axes1, Axes2>);
    static_assert(std::is_same_v<Axes2, Axes3>);
}

TEST(Axes, DispatchEnums) {
    using Axes1 = axes<full_placement_axis, full_determinism_axis, full_contiguity_axis>;
    using Axes2 = axes<full_contiguity_axis, full_placement_axis, full_determinism_axis>;
    static_assert(std::is_same_v<Axes1, Axes2>);
}

TEST(Axes, IsNotAxes) {
    static_assert(!is_axes_v<int>());
    static_assert(!is_axes_v<axis<1, 2>>());
}

// ============================================================================
// Enum stringification
// ============================================================================

TEST(PlacementEnum, Values) {
    EXPECT_EQ(placement::in_place, placement::in_place);
    EXPECT_NE(placement::in_place, placement::out_of_place);
}

TEST(PlacementEnum, Stringification) {
    EXPECT_STREQ(to_string(placement::in_place), "in_place");
    EXPECT_STREQ(to_string(placement::out_of_place), "out_of_place");
}

TEST(DeterminismEnum, Stringification) {
    EXPECT_STREQ(to_string(determinism::not_required), "not_required");
    EXPECT_STREQ(to_string(determinism::required), "required");
}

TEST(ContiguityEnum, Stringification) {
    EXPECT_STREQ(to_string(contiguity::strided), "strided");
    EXPECT_STREQ(to_string(contiguity::contiguous), "contiguous");
}

} // namespace dispatch
