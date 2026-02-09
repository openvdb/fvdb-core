// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Tests for compile-time visitation over axes and extents spaces.
//
#include "dispatch/detail/core_types.h"
#include "dispatch/detail/index_math.h"
#include "dispatch/detail/visit_spaces.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <set>
#include <tuple>
#include <type_traits>
#include <vector>

namespace dispatch {

// ============================================================================
// Test enums with unique value types
// ============================================================================

enum class fruit { apple, banana, cherry };
enum class drink { water, juice, tea };

template <> struct type_label<fruit> {
    static consteval auto
    value() {
        return fixed_label("test.fruit");
    }
};

template <> struct type_label<drink> {
    static consteval auto
    value() {
        return fixed_label("test.drink");
    }
};

using fruit_axis = axis<fruit::apple, fruit::banana, fruit::cherry>;
using drink_axis = axis<drink::water, drink::juice, drink::tea>;

// ============================================================================
// Extents visitation
// ============================================================================

TEST(VisitExtentsSpace, VisitsAllIndices) {
    using E = extents<2, 3>;
    std::vector<std::tuple<size_t, size_t>> visited;

    auto visitor = [&visited](auto idx) {
        if constexpr (std::is_same_v<decltype(idx), indices<0, 0>>) {
            visited.push_back({0, 0});
        } else if constexpr (std::is_same_v<decltype(idx), indices<0, 1>>) {
            visited.push_back({0, 1});
        } else if constexpr (std::is_same_v<decltype(idx), indices<0, 2>>) {
            visited.push_back({0, 2});
        } else if constexpr (std::is_same_v<decltype(idx), indices<1, 0>>) {
            visited.push_back({1, 0});
        } else if constexpr (std::is_same_v<decltype(idx), indices<1, 1>>) {
            visited.push_back({1, 1});
        } else if constexpr (std::is_same_v<decltype(idx), indices<1, 2>>) {
            visited.push_back({1, 2});
        }
    };

    visit_extents_space(visitor, E{});

    EXPECT_EQ(visited.size(), 6u);

    std::set<std::tuple<size_t, size_t>> expected = {
        {0, 0}, {0, 1}, {0, 2}, {1, 0}, {1, 1}, {1, 2}};
    std::set<std::tuple<size_t, size_t>> actual(visited.begin(), visited.end());
    EXPECT_EQ(actual, expected);
}

TEST(VisitExtentsSpace, SingleDimension) {
    using E      = extents<3>;
    int count    = 0;
    auto visitor = [&count](auto /*idx*/) { ++count; };
    visit_extents_space(visitor, E{});
    EXPECT_EQ(count, 3);
}

TEST(VisitExtentsSpace, LargeSpace) {
    using E      = extents<10, 10>;
    int count    = 0;
    auto visitor = [&count](auto /*idx*/) { ++count; };
    visit_extents_space(visitor, E{});
    EXPECT_EQ(count, 100);
}

TEST(VisitExtentsSpace, CountEqualsVolume) {
    using E      = extents<3, 4, 5>;
    int count    = 0;
    auto visitor = [&count](auto /*idx*/) { ++count; };
    visit_extents_space(visitor, E{});
    EXPECT_EQ(count, 60);
}

TEST(VisitExtentsSpaces, VisitsMultipleSpaces) {
    using E1     = extents<2>;
    using E2     = extents<3>;
    int count    = 0;
    auto visitor = [&count](auto /*idx*/) { ++count; };
    visit_extents_spaces(visitor, E1{}, E2{});
    EXPECT_EQ(count, 5);
}

TEST(VisitExtentsSpace, EachCoordinateVisitedOnce) {
    using E = extents<2, 2>;
    std::set<std::tuple<size_t, size_t>> visited;

    auto visitor = [&visited](auto idx) {
        if constexpr (std::is_same_v<decltype(idx), indices<0, 0>>) {
            visited.insert({0, 0});
        } else if constexpr (std::is_same_v<decltype(idx), indices<0, 1>>) {
            visited.insert({0, 1});
        } else if constexpr (std::is_same_v<decltype(idx), indices<1, 0>>) {
            visited.insert({1, 0});
        } else if constexpr (std::is_same_v<decltype(idx), indices<1, 1>>) {
            visited.insert({1, 1});
        }
    };

    visit_extents_space(visitor, E{});
    EXPECT_EQ(visited.size(), 4u);
}

// ============================================================================
// Axes visitation (using uniquely-typed axes)
// ============================================================================

TEST(VisitAxesSpace, VisitsAllTags) {
    using TestAxes = axes<full_placement_axis, full_determinism_axis>;
    int count      = 0;
    auto visitor   = [&count](auto /*coord*/) { ++count; };
    visit_axes_space(visitor, TestAxes{});
    EXPECT_EQ(count, 4); // 2 * 2
}

TEST(VisitAxesSpace, SingleAxis) {
    using TestAxes = axes<fruit_axis>;
    int count      = 0;
    auto visitor   = [&count](auto /*coord*/) { ++count; };
    visit_axes_space(visitor, TestAxes{});
    EXPECT_EQ(count, 3);
}

TEST(VisitAxesSpace, EnumValues) {
    using TestAxes = axes<full_placement_axis, full_determinism_axis>;
    int count      = 0;

    auto visitor = [&count](auto coord) {
        ++count;
        // These are all tag_storage types produced by the visit machinery
        using coord_type = decltype(coord);
        static_assert(tag_like<coord_type>);
    };

    visit_axes_space(visitor, TestAxes{});
    EXPECT_EQ(count, 4);
}

TEST(VisitAxesSpace, CountEqualsVolume) {
    using TestAxes = axes<fruit_axis, drink_axis>;
    int count      = 0;
    auto visitor   = [&count](auto /*coord*/) { ++count; };
    visit_axes_space(visitor, TestAxes{});
    EXPECT_EQ(count, 9); // 3 * 3
}

TEST(VisitAxesSpaces, VisitsMultipleSpaces) {
    using Axes1  = axes<full_placement_axis>;
    using Axes2  = axes<full_determinism_axis>;
    int count    = 0;
    auto visitor = [&count](auto /*coord*/) { ++count; };
    visit_axes_spaces(visitor, Axes1{}, Axes2{});
    EXPECT_EQ(count, 4); // 2 + 2
}

} // namespace dispatch
