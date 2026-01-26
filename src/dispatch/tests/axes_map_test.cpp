// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include "dispatch/axes_map.h"
#include "dispatch/types.h"

#include <gtest/gtest.h>

#include <functional>
#include <stdexcept>
#include <tuple>

namespace dispatch {

// =============================================================================
// Basic operations
// =============================================================================

TEST(AxesMap, DefaultConstruction) {
    using A1   = axis<1, 2>;
    using A2   = axis<3, 4>;
    using Axes = axes<A1, A2>;
    axes_map<Axes, int> map;
    EXPECT_EQ(map.size(), 0u);
    EXPECT_TRUE(map.empty());
}

TEST(AxesMap, CreateAndStoreSingleCoordinate) {
    using A1   = axis<1, 2>;
    using A2   = axis<3, 4>;
    using Axes = axes<A1, A2>;
    axes_map<Axes, int> map;

    auto factory = [](auto coord) -> int {
        if constexpr (std::is_same_v<decltype(coord), tag<1, 3>>) {
            return 10;
        }
        return 0;
    };

    create_and_store(map, factory, tag<1, 3>{});

    EXPECT_EQ(map.size(), 1u);
    auto it = map.find(std::make_tuple(1, 3));
    ASSERT_NE(it, map.end());
    EXPECT_EQ(it->second, 10);
}

TEST(AxesMap, CreateAndStoreSubspace) {
    using A1   = axis<1, 2>;
    using A2   = axis<3, 4>;
    using Axes = axes<A1, A2>;
    axes_map<Axes, int> map;

    int call_count = 0;
    auto factory   = [&call_count](auto coord) -> int {
        ++call_count;
        return 42;
    };

    // Store all coordinates in the full space
    create_and_store(map, factory, Axes{});

    EXPECT_EQ(map.size(), 4u); // 2 * 2 = 4
    EXPECT_EQ(call_count, 4);
}

TEST(AxesMap, InsertValidCoordinate) {
    using A1   = axis<1, 2>;
    using A2   = axis<3, 4>;
    using Axes = axes<A1, A2>;
    axes_map<Axes, int> map;

    map.emplace(tag<1, 3>{}, 10);
    map.emplace(tag<2, 4>{}, 20);

    EXPECT_EQ(map.size(), 2u);
    EXPECT_EQ((map[axes_map_key<Axes>{tag<1, 3>{}}]), 10);
    EXPECT_EQ((map[axes_map_key<Axes>{tag<2, 4>{}}]), 20);
}

TEST(AxesMap, FindExistingKey) {
    using A1   = axis<1, 2>;
    using A2   = axis<3, 4>;
    using Axes = axes<A1, A2>;
    axes_map<Axes, int> map;

    map.emplace(tag<1, 3>{}, 10);

    auto it = map.find(std::make_tuple(1, 3));
    ASSERT_NE(it, map.end());
    EXPECT_EQ(it->second, 10);
}

TEST(AxesMap, FindMissingKey) {
    using A1   = axis<1, 2>;
    using A2   = axis<3, 4>;
    using Axes = axes<A1, A2>;
    axes_map<Axes, int> map;

    map.emplace(tag<1, 3>{}, 10);

    auto it = map.find(std::make_tuple(2, 4));
    EXPECT_EQ(it, map.end());
}

TEST(AxesMap, Iteration) {
    using A1   = axis<1, 2>;
    using A2   = axis<3, 4>;
    using Axes = axes<A1, A2>;
    axes_map<Axes, int> map;

    map.emplace(tag<1, 3>{}, 10);
    map.emplace(tag<1, 4>{}, 20);
    map.emplace(tag<2, 3>{}, 30);
    map.emplace(tag<2, 4>{}, 40);

    int sum = 0;
    for (auto const &[key, value]: map) {
        sum += value;
    }
    EXPECT_EQ(sum, 100);
}

// =============================================================================
// Error handling
// =============================================================================

TEST(AxesMap, InsertThrowsForInvalidCoordinate) {
    using A1   = axis<1, 2>;
    using A2   = axis<3, 4>;
    using Axes = axes<A1, A2>;
    axes_map<Axes, int> map;

    // Try to insert coordinate (99, 3) which is not in the space
    EXPECT_THROW(map.emplace(std::make_tuple(99, 3), 10), std::runtime_error);
}

TEST(AxesMap, FindReturnsEndForOutOfSpaceCoordinate) {
    using A1   = axis<1, 2>;
    using A2   = axis<3, 4>;
    using Axes = axes<A1, A2>;
    axes_map<Axes, int> map;

    map.emplace(tag<1, 3>{}, 10);

    // Find with invalid coordinate - should return end() gracefully
    auto it = map.find(std::make_tuple(99, 3));
    EXPECT_EQ(it, map.end());
}

// =============================================================================
// Tuple-based lookups
// =============================================================================

TEST(AxesMap, TransparentFindWithTuple) {
    using A1   = axis<1, 2>;
    using A2   = axis<3, 4>;
    using Axes = axes<A1, A2>;
    axes_map<Axes, int> map;

    map.emplace(tag<1, 3>{}, 10);

    // Find using tuple without creating a key
    auto it = map.find(std::make_tuple(1, 3));
    ASSERT_NE(it, map.end());
    EXPECT_EQ(it->second, 10);
}

TEST(AxesMap, TransparentFindWithTag) {
    using A1   = axis<1, 2>;
    using A2   = axis<3, 4>;
    using Axes = axes<A1, A2>;
    axes_map<Axes, int> map;

    map.emplace(tag<1, 3>{}, 10);

    // Find using tag directly
    auto it = map.find(tag<1, 3>{});
    ASSERT_NE(it, map.end());
    EXPECT_EQ(it->second, 10);
}

// =============================================================================
// Edge cases
// =============================================================================

TEST(AxesMap, SingleElementAxis) {
    using A    = axis<42>;
    using Axes = axes<A>;
    axes_map<Axes, int> map;

    map.emplace(tag<42>{}, 100);
    EXPECT_EQ(map.size(), 1u);

    auto it = map.find(std::make_tuple(42));
    ASSERT_NE(it, map.end());
    EXPECT_EQ(it->second, 100);
}

TEST(AxesMap, MultiAxisWithVaryingDimensions) {
    using A1   = axis<1>;       // 1 value
    using A2   = axis<2, 3, 4>; // 3 values
    using A3   = axis<5, 6>;    // 2 values
    using Axes = axes<A1, A2, A3>;
    axes_map<Axes, int> map;

    auto factory = [](auto coord) -> int { return 42; };
    create_and_store(map, factory, Axes{});

    EXPECT_EQ(map.size(), 6u); // 1 * 3 * 2 = 6
}

TEST(AxesMap, EmptyMapOperations) {
    using A1   = axis<1, 2>;
    using A2   = axis<3, 4>;
    using Axes = axes<A1, A2>;
    axes_map<Axes, int> map;

    EXPECT_TRUE(map.empty());
    EXPECT_EQ(map.size(), 0u);

    auto it = map.find(std::make_tuple(1, 3));
    EXPECT_EQ(it, map.end());
}

TEST(AxesMap, OperatorBracket) {
    using A1   = axis<1, 2>;
    using A2   = axis<3, 4>;
    using Axes = axes<A1, A2>;
    axes_map<Axes, int> map;

    map[axes_map_key<Axes>{tag<1, 3>{}}] = 10;
    EXPECT_EQ((map[axes_map_key<Axes>{tag<1, 3>{}}]), 10);
    EXPECT_EQ(map.size(), 1u);
}

TEST(AxesMap, AtMethod) {
    using A1   = axis<1, 2>;
    using A2   = axis<3, 4>;
    using Axes = axes<A1, A2>;
    axes_map<Axes, int> map;

    map.emplace(tag<1, 3>{}, 10);

    EXPECT_EQ((map.at(axes_map_key<Axes>{tag<1, 3>{}})), 10);
    EXPECT_THROW((map.at(axes_map_key<Axes>{tag<2, 4>{}})), std::out_of_range);
}

TEST(AxesMap, Clear) {
    using A1   = axis<1, 2>;
    using A2   = axis<3, 4>;
    using Axes = axes<A1, A2>;
    axes_map<Axes, int> map;

    map.emplace(tag<1, 3>{}, 10);
    EXPECT_EQ(map.size(), 1u);

    map.clear();
    EXPECT_TRUE(map.empty());
    EXPECT_EQ(map.size(), 0u);
}

// =============================================================================
// insert_or_assign free function
// =============================================================================

TEST(AxesMap, InsertOrAssignInsertsWithTag) {
    using A1   = axis<1, 2>;
    using A2   = axis<3, 4>;
    using Axes = axes<A1, A2>;
    axes_map<Axes, int> map;

    insert_or_assign(map, tag<1, 3>{}, 10);

    EXPECT_EQ(map.size(), 1u);
    auto it = map.find(tag<1, 3>{});
    ASSERT_NE(it, map.end());
    EXPECT_EQ(it->second, 10);
}

TEST(AxesMap, InsertOrAssignInsertsWithTuple) {
    using A1   = axis<1, 2>;
    using A2   = axis<3, 4>;
    using Axes = axes<A1, A2>;
    axes_map<Axes, int> map;

    insert_or_assign(map, std::make_tuple(1, 3), 10);

    EXPECT_EQ(map.size(), 1u);
    auto it = map.find(std::make_tuple(1, 3));
    ASSERT_NE(it, map.end());
    EXPECT_EQ(it->second, 10);
}

TEST(AxesMap, InsertOrAssignOverwritesExisting) {
    using A1   = axis<1, 2>;
    using A2   = axis<3, 4>;
    using Axes = axes<A1, A2>;
    axes_map<Axes, int> map;

    insert_or_assign(map, tag<1, 3>{}, 10);
    EXPECT_EQ(map.find(tag<1, 3>{})->second, 10);

    insert_or_assign(map, tag<1, 3>{}, 99);
    EXPECT_EQ(map.size(), 1u); // Still only one entry
    EXPECT_EQ(map.find(tag<1, 3>{})->second, 99);
}

TEST(AxesMap, InsertOrAssignOverwritesWithTuple) {
    using A1   = axis<1, 2>;
    using A2   = axis<3, 4>;
    using Axes = axes<A1, A2>;
    axes_map<Axes, int> map;

    insert_or_assign(map, std::make_tuple(2, 4), 100);
    insert_or_assign(map, std::make_tuple(2, 4), 200);

    EXPECT_EQ(map.size(), 1u);
    EXPECT_EQ(map.find(std::make_tuple(2, 4))->second, 200);
}

TEST(AxesMap, InsertOrAssignThrowsForOutOfSpaceTuple) {
    using A1   = axis<1, 2>;
    using A2   = axis<3, 4>;
    using Axes = axes<A1, A2>;
    axes_map<Axes, int> map;

    // Tuple with values outside the axes should throw
    EXPECT_THROW(insert_or_assign(map, std::make_tuple(99, 3), 10), std::runtime_error);
    EXPECT_THROW(insert_or_assign(map, std::make_tuple(1, 99), 10), std::runtime_error);
    EXPECT_TRUE(map.empty());
}

} // namespace dispatch
