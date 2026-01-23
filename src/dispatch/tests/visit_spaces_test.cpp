// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include "dispatch/types.h"
#include "dispatch/visit_spaces.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <set>
#include <vector>

namespace dispatch {

// =============================================================================
// Extents visitation
// =============================================================================

TEST(VisitExtentsSpace, VisitsAllIndices) {
    using E = extents<2, 3>;
    std::vector<std::tuple<size_t, size_t>> visited;

    auto visitor = [&visited](auto indices) {
        if constexpr (std::is_same_v<decltype(indices), indices<0, 0>>) {
            visited.push_back({0, 0});
        } else if constexpr (std::is_same_v<decltype(indices), indices<0, 1>>) {
            visited.push_back({0, 1});
        } else if constexpr (std::is_same_v<decltype(indices), indices<0, 2>>) {
            visited.push_back({0, 2});
        } else if constexpr (std::is_same_v<decltype(indices), indices<1, 0>>) {
            visited.push_back({1, 0});
        } else if constexpr (std::is_same_v<decltype(indices), indices<1, 1>>) {
            visited.push_back({1, 1});
        } else if constexpr (std::is_same_v<decltype(indices), indices<1, 2>>) {
            visited.push_back({1, 2});
        }
    };

    visit_extents_space(visitor, E{});

    EXPECT_EQ(visited.size(), 6u); // 2 * 3 = 6

    // Check all expected indices are present
    std::set<std::tuple<size_t, size_t>> expected = {
        {0, 0}, {0, 1}, {0, 2}, {1, 0}, {1, 1}, {1, 2}};
    std::set<std::tuple<size_t, size_t>> actual(visited.begin(), visited.end());
    EXPECT_EQ(actual, expected);
}

TEST(VisitExtentsSpace, SingleDimension) {
    using E   = extents<3>;
    int count = 0;

    auto visitor = [&count](auto indices) {
        ++count;
        if constexpr (std::is_same_v<decltype(indices), indices<0>>) {
            // Expected
        } else if constexpr (std::is_same_v<decltype(indices), indices<1>>) {
            // Expected
        } else if constexpr (std::is_same_v<decltype(indices), indices<2>>) {
            // Expected
        }
    };

    visit_extents_space(visitor, E{});

    EXPECT_EQ(count, 3);
}

TEST(VisitExtentsSpace, LargeSpace) {
    using E   = extents<10, 10>;
    int count = 0;

    auto visitor = [&count](auto /*indices*/) { ++count; };

    visit_extents_space(visitor, E{});

    EXPECT_EQ(count, 100); // 10 * 10 = 100
}

TEST(VisitExtentsSpaces, VisitsMultipleSpaces) {
    using E1  = extents<2>;
    using E2  = extents<3>;
    int count = 0;

    auto visitor = [&count](auto /*indices*/) { ++count; };

    visit_extents_spaces(visitor, E1{}, E2{});

    EXPECT_EQ(count, 5); // 2 + 3 = 5
}

TEST(VisitExtentsSpaces, ThreeSpaces) {
    using E1  = extents<2>;
    using E2  = extents<3>;
    using E3  = extents<4>;
    int count = 0;

    auto visitor = [&count](auto /*indices*/) { ++count; };

    visit_extents_spaces(visitor, E1{}, E2{}, E3{});

    EXPECT_EQ(count, 9); // 2 + 3 + 4 = 9
}

// =============================================================================
// Axes visitation
// =============================================================================

TEST(VisitAxesSpace, VisitsAllTags) {
    using A1   = axis<1, 2>;
    using A2   = axis<3, 4>;
    using Axes = axes<A1, A2>;
    std::vector<std::tuple<int, int>> visited;

    auto visitor = [&visited](auto tag) {
        if constexpr (std::is_same_v<decltype(tag), tag<1, 3>>) {
            visited.push_back({1, 3});
        } else if constexpr (std::is_same_v<decltype(tag), tag<1, 4>>) {
            visited.push_back({1, 4});
        } else if constexpr (std::is_same_v<decltype(tag), tag<2, 3>>) {
            visited.push_back({2, 3});
        } else if constexpr (std::is_same_v<decltype(tag), tag<2, 4>>) {
            visited.push_back({2, 4});
        }
    };

    visit_axes_space(visitor, Axes{});

    EXPECT_EQ(visited.size(), 4u); // 2 * 2 = 4

    // Check all expected tags are present
    std::set<std::tuple<int, int>> expected = {{1, 3}, {1, 4}, {2, 3}, {2, 4}};
    std::set<std::tuple<int, int>> actual(visited.begin(), visited.end());
    EXPECT_EQ(actual, expected);
}

TEST(VisitAxesSpace, SingleAxis) {
    using A    = axis<10, 20, 30>;
    using Axes = axes<A>;
    int count  = 0;
    std::set<int> values;

    auto visitor = [&count, &values](auto tag) {
        ++count;
        if constexpr (std::is_same_v<decltype(tag), tag<10>>) {
            values.insert(10);
        } else if constexpr (std::is_same_v<decltype(tag), tag<20>>) {
            values.insert(20);
        } else if constexpr (std::is_same_v<decltype(tag), tag<30>>) {
            values.insert(30);
        }
    };

    visit_axes_space(visitor, Axes{});

    EXPECT_EQ(count, 3);
    EXPECT_EQ(values.size(), 3u);
    EXPECT_TRUE(values.count(10));
    EXPECT_TRUE(values.count(20));
    EXPECT_TRUE(values.count(30));
}

TEST(VisitAxesSpace, EnumValues) {
    using Axes = axes<full_placement_axis, full_determinism_axis>;
    int count  = 0;

    auto visitor = [&count](auto tag) {
        ++count;
        // Should visit all 4 combinations
        if constexpr (std::is_same_v<decltype(tag),
                                     tag<placement::in_place, determinism::not_required>>) {
            // Expected
        } else if constexpr (std::is_same_v<decltype(tag),
                                            tag<placement::in_place, determinism::required>>) {
            // Expected
        } else if constexpr (std::is_same_v<
                                 decltype(tag),
                                 tag<placement::out_of_place, determinism::not_required>>) {
            // Expected
        } else if constexpr (std::is_same_v<decltype(tag),
                                            tag<placement::out_of_place, determinism::required>>) {
            // Expected
        }
    };

    visit_axes_space(visitor, Axes{});

    EXPECT_EQ(count, 4); // 2 * 2 = 4
}

TEST(VisitAxesSpaces, VisitsMultipleSpaces) {
    using A1    = axis<1, 2>;
    using A2    = axis<3>;
    using Axes1 = axes<A1>;
    using Axes2 = axes<A2>;
    int count   = 0;

    auto visitor = [&count](auto /*tag*/) { ++count; };

    visit_axes_spaces(visitor, Axes1{}, Axes2{});

    EXPECT_EQ(count, 3); // 2 + 1 = 3
}

TEST(VisitAxesSpaces, ThreeSpaces) {
    using A1    = axis<1>;
    using A2    = axis<2>;
    using A3    = axis<3>;
    using Axes1 = axes<A1>;
    using Axes2 = axes<A2>;
    using Axes3 = axes<A3>;
    int count   = 0;

    auto visitor = [&count](auto /*tag*/) { ++count; };

    visit_axes_spaces(visitor, Axes1{}, Axes2{}, Axes3{});

    EXPECT_EQ(count, 3); // 1 + 1 + 1 = 3
}

// =============================================================================
// Visitor correctness
// =============================================================================

TEST(VisitExtentsSpace, CountEqualsVolume) {
    using E   = extents<3, 4, 5>;
    int count = 0;

    auto visitor = [&count](auto /*indices*/) { ++count; };

    visit_extents_space(visitor, E{});

    EXPECT_EQ(count, 60); // 3 * 4 * 5 = 60
}

TEST(VisitAxesSpace, CountEqualsVolume) {
    using A1   = axis<1, 2, 3>;
    using A2   = axis<4, 5>;
    using Axes = axes<A1, A2>;
    int count  = 0;

    auto visitor = [&count](auto /*tag*/) { ++count; };

    visit_axes_space(visitor, Axes{});

    EXPECT_EQ(count, 6); // 3 * 2 = 6
}

TEST(VisitExtentsSpace, EachCoordinateVisitedOnce) {
    using E = extents<2, 2>;
    std::set<std::tuple<size_t, size_t>> visited;

    auto visitor = [&visited](auto indices) {
        if constexpr (std::is_same_v<decltype(indices), indices<0, 0>>) {
            visited.insert({0, 0});
        } else if constexpr (std::is_same_v<decltype(indices), indices<0, 1>>) {
            visited.insert({0, 1});
        } else if constexpr (std::is_same_v<decltype(indices), indices<1, 0>>) {
            visited.insert({1, 0});
        } else if constexpr (std::is_same_v<decltype(indices), indices<1, 1>>) {
            visited.insert({1, 1});
        }
    };

    visit_extents_space(visitor, E{});

    EXPECT_EQ(visited.size(), 4u); // All 4 coordinates visited exactly once
}

TEST(VisitAxesSpace, EachTagVisitedOnce) {
    using A1   = axis<1, 2>;
    using A2   = axis<3, 4>;
    using Axes = axes<A1, A2>;
    std::set<std::tuple<int, int>> visited;

    auto visitor = [&visited](auto tag) {
        if constexpr (std::is_same_v<decltype(tag), tag<1, 3>>) {
            visited.insert({1, 3});
        } else if constexpr (std::is_same_v<decltype(tag), tag<1, 4>>) {
            visited.insert({1, 4});
        } else if constexpr (std::is_same_v<decltype(tag), tag<2, 3>>) {
            visited.insert({2, 3});
        } else if constexpr (std::is_same_v<decltype(tag), tag<2, 4>>) {
            visited.insert({2, 4});
        }
    };

    visit_axes_space(visitor, Axes{});

    EXPECT_EQ(visited.size(), 4u); // All 4 tags visited exactly once
}

} // namespace dispatch
