// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Tests for axes_map: sparse dispatch map with transparent lookup.
//
#include "dispatch/axes_map.h"
#include "dispatch/enums.h"
#include "dispatch/label.h"

#include <gtest/gtest.h>

#include <functional>
#include <stdexcept>
#include <tuple>

namespace dispatch {

// ============================================================================
// Test enums with unique value types
// ============================================================================

enum class color { red, green, blue };
enum class shape { circle, square };

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

using color_axis = axis<color::red, color::green, color::blue>;
using shape_axis = axis<shape::circle, shape::square>;

// The axes alias normalizes, so we need to know the canonical order for tuples.
// Since "test.color" < "test.shape" lexicographically, the canonical order
// is axes_storage<color_axis, shape_axis>, so tuple is (color, shape).
using test_axes = axes<color_axis, shape_axis>;

// ============================================================================
// Basic operations
// ============================================================================

TEST(AxesMap, DefaultConstruction) {
    axes_map<test_axes, int> map;
    EXPECT_EQ(map.size(), 0u);
    EXPECT_TRUE(map.empty());
}

TEST(AxesMap, CreateAndStoreSingleCoordinate) {
    axes_map<test_axes, int> map;

    auto factory = [](auto coord) -> int {
        if constexpr (std::is_same_v<decltype(coord),
                                     tag_storage<color::red, shape::circle>>) {
            return 10;
        }
        return 0;
    };

    create_and_store(map, factory, tag<color::red, shape::circle>{});

    EXPECT_EQ(map.size(), 1u);
    auto it = map.find(std::make_tuple(color::red, shape::circle));
    ASSERT_NE(it, map.end());
    EXPECT_EQ(it->second, 10);
}

TEST(AxesMap, CreateAndStoreSubspace) {
    axes_map<test_axes, int> map;

    int call_count = 0;
    auto factory   = [&call_count](auto coord) -> int {
        ++call_count;
        return 42;
    };

    create_and_store(map, factory, test_axes{});

    EXPECT_EQ(map.size(), 6u); // 3 * 2 = 6
    EXPECT_EQ(call_count, 6);
}

TEST(AxesMap, InsertValidCoordinate) {
    axes_map<test_axes, int> map;

    map.emplace(tag<color::red, shape::circle>{}, 10);
    map.emplace(tag<color::green, shape::square>{}, 20);

    EXPECT_EQ(map.size(), 2u);
}

TEST(AxesMap, FindExistingKey) {
    axes_map<test_axes, int> map;

    map.emplace(tag<color::red, shape::circle>{}, 10);

    auto it = map.find(std::make_tuple(color::red, shape::circle));
    ASSERT_NE(it, map.end());
    EXPECT_EQ(it->second, 10);
}

TEST(AxesMap, FindMissingKey) {
    axes_map<test_axes, int> map;

    map.emplace(tag<color::red, shape::circle>{}, 10);

    auto it = map.find(std::make_tuple(color::green, shape::square));
    EXPECT_EQ(it, map.end());
}

TEST(AxesMap, Iteration) {
    axes_map<test_axes, int> map;

    map.emplace(tag<color::red, shape::circle>{}, 10);
    map.emplace(tag<color::red, shape::square>{}, 20);
    map.emplace(tag<color::green, shape::circle>{}, 30);
    map.emplace(tag<color::green, shape::square>{}, 40);

    int sum = 0;
    for (auto const &[key, value]: map) {
        sum += value;
    }
    EXPECT_EQ(sum, 100);
}

// ============================================================================
// Error handling
// ============================================================================

TEST(AxesMap, InsertThrowsForInvalidCoordinate) {
    // Single-axis map to test out-of-range
    using single_axes = axes<color_axis>;
    axes_map<single_axes, int> map;

    // Enum value not in the axis — runtime failure via axes_map_key constructor
    auto const bad_color = static_cast<color>(99);
    EXPECT_THROW(map.emplace(std::make_tuple(bad_color), 10), std::runtime_error);
}

TEST(AxesMap, FindReturnsEndForOutOfSpaceCoordinate) {
    using single_axes = axes<color_axis>;
    axes_map<single_axes, int> map;

    map.emplace(tag<color::red>{}, 10);

    auto const bad_color = static_cast<color>(99);
    auto it = map.find(std::make_tuple(bad_color));
    EXPECT_EQ(it, map.end());
}

// ============================================================================
// Transparent lookups
// ============================================================================

TEST(AxesMap, TransparentFindWithTuple) {
    axes_map<test_axes, int> map;

    map.emplace(tag<color::red, shape::circle>{}, 10);

    auto it = map.find(std::make_tuple(color::red, shape::circle));
    ASSERT_NE(it, map.end());
    EXPECT_EQ(it->second, 10);
}

TEST(AxesMap, TransparentFindWithTag) {
    axes_map<test_axes, int> map;

    map.emplace(tag<color::red, shape::circle>{}, 10);

    auto it = map.find(tag<color::red, shape::circle>{});
    ASSERT_NE(it, map.end());
    EXPECT_EQ(it->second, 10);
}

// ============================================================================
// Edge cases
// ============================================================================

TEST(AxesMap, SingleElementAxis) {
    using single_axes = axes<axis<placement::in_place>>;
    axes_map<single_axes, int> map;

    map.emplace(tag<placement::in_place>{}, 100);
    EXPECT_EQ(map.size(), 1u);

    auto it = map.find(std::make_tuple(placement::in_place));
    ASSERT_NE(it, map.end());
    EXPECT_EQ(it->second, 100);
}

TEST(AxesMap, ThreeAxes) {
    using three_axes = axes<full_placement_axis, full_determinism_axis, full_contiguity_axis>;
    axes_map<three_axes, int> map;

    auto factory = [](auto coord) -> int { return 42; };
    create_and_store(map, factory, three_axes{});

    EXPECT_EQ(map.size(), 8u); // 2 * 2 * 2 = 8
}

TEST(AxesMap, EmptyMapOperations) {
    axes_map<test_axes, int> map;

    EXPECT_TRUE(map.empty());
    EXPECT_EQ(map.size(), 0u);

    auto it = map.find(std::make_tuple(color::red, shape::circle));
    EXPECT_EQ(it, map.end());
}

TEST(AxesMap, Clear) {
    axes_map<test_axes, int> map;

    map.emplace(tag<color::red, shape::circle>{}, 10);
    EXPECT_EQ(map.size(), 1u);

    map.clear();
    EXPECT_TRUE(map.empty());
}

// ============================================================================
// insert_or_assign
// ============================================================================

TEST(AxesMap, InsertOrAssignInsertsWithTag) {
    axes_map<test_axes, int> map;

    insert_or_assign(map, tag<color::red, shape::circle>{}, 10);

    EXPECT_EQ(map.size(), 1u);
    auto it = map.find(tag<color::red, shape::circle>{});
    ASSERT_NE(it, map.end());
    EXPECT_EQ(it->second, 10);
}

TEST(AxesMap, InsertOrAssignInsertsWithTuple) {
    axes_map<test_axes, int> map;

    insert_or_assign(map, std::make_tuple(color::red, shape::circle), 10);

    EXPECT_EQ(map.size(), 1u);
    auto it = map.find(std::make_tuple(color::red, shape::circle));
    ASSERT_NE(it, map.end());
    EXPECT_EQ(it->second, 10);
}

TEST(AxesMap, InsertOrAssignOverwritesExisting) {
    axes_map<test_axes, int> map;

    insert_or_assign(map, tag<color::red, shape::circle>{}, 10);
    EXPECT_EQ(map.find(tag<color::red, shape::circle>{})->second, 10);

    insert_or_assign(map, tag<color::red, shape::circle>{}, 99);
    EXPECT_EQ(map.size(), 1u);
    EXPECT_EQ(map.find(tag<color::red, shape::circle>{})->second, 99);
}

TEST(AxesMap, InsertOrAssignThrowsForOutOfSpaceTuple) {
    using single_axes = axes<color_axis>;
    axes_map<single_axes, int> map;

    auto const bad_color = static_cast<color>(99);
    EXPECT_THROW(insert_or_assign(map, std::make_tuple(bad_color), 10), std::runtime_error);
    EXPECT_TRUE(map.empty());
}

} // namespace dispatch
