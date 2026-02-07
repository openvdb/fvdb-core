// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Tests for dispatch_table: sparse subspace instantiation, select/try_select
// with dispatch_set, functional update, and factory patterns.
//
#include "dispatch/dispatch_set.h"
#include "dispatch/dispatch_table.h"
#include "dispatch/enums.h"
#include "dispatch/label.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <type_traits>

namespace dispatch {

// ============================================================================
// Test enums with unique value types
// ============================================================================

enum class color { red, green, blue };
enum class shape { circle, square, triangle };

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
using shape_axis = axis<shape::circle, shape::square, shape::triangle>;

// ============================================================================
// from_op: struct with static op() method
// ============================================================================

struct test_op {
    template <typename Coord>
    static int
    op(Coord, int x) {
        if constexpr (std::is_same_v<Coord, tag<color::red>>) {
            return x * 2;
        } else if constexpr (std::is_same_v<Coord, tag<color::green>>) {
            return x * 3;
        }
        return 0;
    }
};

TEST(DispatchTable, FromOpSelectInvoke) {
    using TestAxes = axes<color_axis>;
    using SubAxis  = axis<color::red, color::green>;
    using SubAxes  = axes<SubAxis>;
    using Table    = dispatch_table<TestAxes, int(int)>;

    Table table("test_from_op", Table::from_op<test_op>(), SubAxes{});

    auto fn_red   = table.select(dispatch_set{color::red});
    auto fn_green = table.select(dispatch_set{color::green});

    EXPECT_EQ(fn_red(5), 10);   // 5 * 2
    EXPECT_EQ(fn_green(5), 15); // 5 * 3

    // blue is not in the subspace — should throw
    EXPECT_THROW(table.select(dispatch_set{color::blue}), dispatch_lookup_error);
}

TEST(DispatchTable, TrySelectReturnsNullptr) {
    using TestAxes = axes<color_axis>;
    using SubAxis  = axis<color::red>;
    using SubAxes  = axes<SubAxis>;
    using Table    = dispatch_table<TestAxes, int(int)>;

    Table table("test_try", Table::from_op<test_op>(), SubAxes{});

    auto fn = table.try_select(dispatch_set{color::red});
    ASSERT_NE(fn, nullptr);
    EXPECT_EQ(fn(5), 10);

    auto fn_missing = table.try_select(dispatch_set{color::green});
    EXPECT_EQ(fn_missing, nullptr);
}

// ============================================================================
// from_visitor: wrap overloaded free functions
// ============================================================================

int
test_free_function(tag<color::red>, int x) {
    return x * 10;
}
int
test_free_function(tag<color::green>, int x) {
    return x * 20;
}

template <color C>
int
test_free_function(tag<C>, int) {
    return 0;
}

TEST(DispatchTable, FromVisitor) {
    using TestAxes = axes<color_axis>;
    using SubAxis  = axis<color::red, color::green, color::blue>;
    using SubAxes  = axes<SubAxis>;
    using Table    = dispatch_table<TestAxes, int(int)>;

    Table table("test_visitor",
                Table::from_visitor([](auto c, int x) { return test_free_function(c, x); }),
                SubAxes{});

    EXPECT_EQ(table.select(dispatch_set{color::red})(3), 30);  // 3 * 10
    EXPECT_EQ(table.select(dispatch_set{color::green})(3), 60); // 3 * 20
    EXPECT_EQ(table.select(dispatch_set{color::blue})(3), 0);   // default
}

// ============================================================================
// Multi-axis dispatch
// ============================================================================

struct multi_axis_op {
    template <color C, shape S>
    static int
    op(tag<C, S>) {
        if constexpr (C == color::red && S == shape::circle) {
            return 1;
        } else if constexpr (C == color::green && S == shape::square) {
            return 2;
        }
        return 0;
    }
};

TEST(DispatchTable, MultiAxisSelectWithDispatchSet) {
    using TestAxes = axes<color_axis, shape_axis>;
    using Table    = dispatch_table<TestAxes, int()>;

    // Only instantiate a subset
    using SubAxes = axes<axis<color::red, color::green>, axis<shape::circle, shape::square>>;
    Table table("multi_axis", Table::from_op<multi_axis_op>(), SubAxes{});

    // Order doesn't matter in dispatch_set — matched by type
    EXPECT_EQ(table.select(dispatch_set{color::red, shape::circle})(), 1);
    EXPECT_EQ(table.select(dispatch_set{shape::circle, color::red})(), 1);  // reversed order
    EXPECT_EQ(table.select(dispatch_set{color::green, shape::square})(), 2);

    // Not in subspace
    EXPECT_THROW(table.select(dispatch_set{color::blue, shape::circle}), dispatch_lookup_error);
    EXPECT_THROW(table.select(dispatch_set{color::red, shape::triangle}), dispatch_lookup_error);
}

// ============================================================================
// Functional update (with)
// ============================================================================

TEST(DispatchTable, WithReturnsNewTable) {
    using TestAxes = axes<color_axis>;
    using Table    = dispatch_table<TestAxes, int()>;

    auto factory1 = [](auto) -> int (*)() { return []() { return 10; }; };
    auto factory2 = [](auto) -> int (*)() { return []() { return 20; }; };

    Table table1("with_test", factory1, axes<axis<color::red>>{});
    Table table2 = table1.with(factory2, axes<axis<color::green>>{});

    // table1 only has red
    EXPECT_EQ(table1.select(dispatch_set{color::red})(), 10);
    EXPECT_THROW(table1.select(dispatch_set{color::green}), dispatch_lookup_error);

    // table2 has both
    EXPECT_EQ(table2.select(dispatch_set{color::red})(), 10);    // inherited
    EXPECT_EQ(table2.select(dispatch_set{color::green})(), 20);  // added
}

TEST(DispatchTable, OriginalUnchangedAfterWith) {
    using TestAxes = axes<color_axis>;
    using Table    = dispatch_table<TestAxes, int()>;

    auto factory1 = [](auto) -> int (*)() { return []() { return 100; }; };
    auto factory2 = [](auto) -> int (*)() { return []() { return 200; }; };

    Table table1("unchanged_test", factory1, axes<color_axis>{});
    Table table2 = table1.with(factory2, axes<color_axis>{});

    EXPECT_EQ(table1.select(dispatch_set{color::red})(), 100);
    EXPECT_EQ(table2.select(dispatch_set{color::red})(), 200);
}

// ============================================================================
// Single tag coordinate
// ============================================================================

TEST(DispatchTable, SingleTagCoordinate) {
    using TestAxes = axes<color_axis, shape_axis>;
    using Table    = dispatch_table<TestAxes, int()>;

    auto factory = [](auto) -> int (*)() { return []() { return 42; }; };

    // Instantiate a single point
    Table table("single_tag", factory, tag<color::red, shape::circle>{});

    EXPECT_EQ(table.select(dispatch_set{color::red, shape::circle})(), 42);
    EXPECT_THROW(table.select(dispatch_set{color::red, shape::square}), dispatch_lookup_error);
    EXPECT_THROW(table.select(dispatch_set{color::green, shape::circle}), dispatch_lookup_error);
}

// ============================================================================
// Multiple arguments
// ============================================================================

TEST(DispatchTable, MultipleArguments) {
    using TestAxes = axes<color_axis>;
    using Table    = dispatch_table<TestAxes, int(int, int)>;

    auto factory = [](auto c) -> int (*)(int, int) {
        if constexpr (std::is_same_v<decltype(c), tag<color::red>>) {
            return [](int a, int b) { return a + b; };
        } else {
            return [](int a, int b) { return a * b; };
        }
    };

    Table table("multi_args", factory, axes<color_axis>{});

    EXPECT_EQ(table.select(dispatch_set{color::red})(3, 4), 7);   // 3 + 4
    EXPECT_EQ(table.select(dispatch_set{color::green})(3, 4), 12); // 3 * 4
}

// ============================================================================
// Table name
// ============================================================================

TEST(DispatchTable, NameAccessor) {
    using TestAxes = axes<color_axis>;
    using Table    = dispatch_table<TestAxes, int()>;

    Table table("my_table_name");
    EXPECT_EQ(table.name(), "my_table_name");
}

TEST(DispatchTable, ErrorMessageContainsName) {
    using TestAxes = axes<color_axis>;
    using Table    = dispatch_table<TestAxes, int()>;

    Table table("test_error_name");

    try {
        table.select(dispatch_set{color::red});
        FAIL() << "Expected dispatch_lookup_error";
    } catch (dispatch_lookup_error const &e) {
        std::string msg = e.what();
        EXPECT_TRUE(msg.find("test_error_name") != std::string::npos)
            << "Error message should contain table name, got: " << msg;
    }
}

// ============================================================================
// dispatch_table_from_op
// ============================================================================

struct simple_op {
    template <typename Coord>
    static int
    op(Coord, int x) {
        return x * 2;
    }

    using space      = axes<color_axis>;
    using subspaces  = coverage<space>;
    using dispatcher = dispatch_table<space, int(int)>;
};

TEST(DispatchTableFromOp, Basic) {
    auto table = dispatch_table_from_op<simple_op>("simple_op");
    EXPECT_EQ(table.select(dispatch_set{color::red})(5), 10);
    EXPECT_EQ(table.select(dispatch_set{color::green})(5), 10);
    EXPECT_EQ(table.select(dispatch_set{color::blue})(5), 10);
}

} // namespace dispatch
