// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include "dispatch/dispatch_table.h"
#include "dispatch/types.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <tuple>

namespace dispatch {

// Helper to reduce std::make_tuple boilerplate in tests
template <typename... Ts>
auto
coord(Ts... vs) {
    return std::make_tuple(vs...);
}

// =============================================================================
// from_op: struct with static op() method
// =============================================================================

// Test op struct: returns x*2 for tag<1>, x*3 for tag<2>, 0 otherwise.
struct TestOp {
    template <typename Coord>
    static int
    op(Coord, int x) {
        if constexpr (std::is_same_v<Coord, tag<1>>) {
            return x * 2;
        } else if constexpr (std::is_same_v<Coord, tag<2>>) {
            return x * 3;
        }
        return 0;
    }
};

TEST(DispatchTable, FromOp) {
    using Axis    = axis<1, 2, 3, 4, 5, 6, 7, 8>;
    using Axes    = axes<Axis>;
    using SubAxis = axis<1, 2, 5>;
    using SubAxes = axes<SubAxis>;
    using Table   = dispatch_table<Axes, int(int)>;

    Table table(Table::from_op<TestOp>(), SubAxes{});

    EXPECT_EQ(table(coord(1), 5), 10); // 5 * 2
    EXPECT_EQ(table(coord(2), 5), 15); // 5 * 3
    EXPECT_EQ(table(coord(5), 5), 0);  // default

    // These should throw for all points not in the subspace
    for (int i: {3, 4, 6, 7, 8}) {
        EXPECT_THROW(table(coord(i), 5), std::runtime_error);
    }

    // Also outside the original space
    for (int i: {0, 9, 10}) {
        EXPECT_THROW(table(coord(i), 5), std::runtime_error);
    }
}

// =============================================================================
// from_visitor: wrap overloaded free functions
// =============================================================================

int
test_free_function(tag<1>, int x) {
    return x * 10;
}
int
test_free_function(tag<2>, int x) {
    return x * 20;
}

template <int I>
int
test_free_function(tag<I>, int) {
    return 0;
}

TEST(DispatchTable, FromVisitor) {
    using Axis    = axis<1, 2, 3, 4, 5, 6, 7, 8>;
    using Axes    = axes<Axis>;
    using SubAxis = axis<1, 2, 5>;
    using SubAxes = axes<SubAxis>;
    using Table   = dispatch_table<Axes, int(int)>;

    // Table defined over full Axes, but only instantiated for SubAxes
    Table table(Table::from_visitor([](auto c, int x) { return test_free_function(c, x); }),
                SubAxes{});

    EXPECT_EQ(table(coord(1), 3), 30); // 3 * 10
    EXPECT_EQ(table(coord(2), 3), 60); // 3 * 20
    EXPECT_EQ(table(coord(5), 3), 0);  // default

    // These should throw for points not in the subspace
    for (int i: {3, 4, 6, 7, 8}) {
        EXPECT_THROW(table(coord(i), 3), std::runtime_error);
    }

    // Also outside the original space
    for (int i: {0, 9, 10}) {
        EXPECT_THROW(table(coord(i), 3), std::runtime_error);
    }
}

TEST(DispatchTable, ThrowsForMissingCoordinates) {
    using A     = axis<1, 2>;
    using Axes  = axes<A>;
    using Table = dispatch_table<Axes, int()>;

    auto factory = [](auto) -> int (*)() { return []() { return 42; }; };

    Table table(factory, Axes{});

    EXPECT_EQ(table(coord(1)), 42);

    // Invalid coordinate - should throw
    EXPECT_THROW(table(coord(99)), std::runtime_error);
}

// =============================================================================
// Functional update
// =============================================================================

TEST(DispatchTable, WithReturnsNewTable) {
    using Axis     = axis<1, 2, 3, 4, 5, 6, 7, 8>;
    using Axes     = axes<Axis>;
    using SubAxis1 = axis<3, 4>;
    using SubAxes1 = axes<SubAxis1>;
    using SubAxis2 = axis<1, 2, 3, 5>;
    using SubAxes2 = axes<SubAxis2>;
    using Table    = dispatch_table<Axes, int()>;

    auto factory1 = [](auto) -> int (*)() { return []() { return 10; }; };
    Table table1(factory1, SubAxes1{});

    auto factory2 = [](auto c) -> int (*)() {
        if constexpr (std::is_same_v<decltype(c), tag<1>>) {
            return []() { return 10; };
        } else if constexpr (std::is_same_v<decltype(c), tag<2>>) {
            return []() { return 20; };
        } else {
            return nullptr;
        }
    };

    Table table2 = table1.with(factory2, SubAxes2{});

    // table1: {3,4}->10, table2: inherits 4->10, adds 1->10, 2->20, overwrites 3&5 with nullptr
    EXPECT_EQ(table1(coord(3)), 10);
    EXPECT_EQ(table1(coord(4)), 10);
    for (int i: {1, 2, 5, 6, 7, 8}) {
        EXPECT_THROW(table1(coord(i)), std::runtime_error);
    }

    // Table 2: 1->10, 2->20, 4->10 (inherited), 3&5 throw (nullptr), 6-8 throw (missing)
    EXPECT_EQ(table2(coord(1)), 10);
    EXPECT_EQ(table2(coord(2)), 20);
    EXPECT_EQ(table2(coord(4)), 10);
    for (int i: {3, 5, 6, 7, 8}) {
        EXPECT_THROW(table2(coord(i)), std::runtime_error);
    }
}

TEST(DispatchTable, OriginalTableUnchangedAfterWith) {
    using A     = axis<1, 2>;
    using Axes  = axes<A>;
    using Table = dispatch_table<Axes, int()>;

    auto factory1 = [](auto) -> int (*)() { return []() { return 100; }; };
    Table table1(factory1, Axes{});

    auto factory2 = [](auto) -> int (*)() { return []() { return 200; }; };

    Table table2 = table1.with(factory2, Axes{});

    // Original should still have original values
    EXPECT_EQ(table1(coord(1)), 100);
    EXPECT_EQ(table1(coord(2)), 100);

    // New table should have new values
    EXPECT_EQ(table2(coord(1)), 200);
    EXPECT_EQ(table2(coord(2)), 200);
}

// =============================================================================
// Subspaces
// =============================================================================

TEST(DispatchTable, TablesBuiltFromSubspaces) {
    using A1       = axis<1, 2>;
    using A2       = axis<3, 4>;
    using FullAxes = axes<A1, A2>;

    // Subspace: only A1=1
    using SubA1   = axis<1>;
    using SubAxes = axes<SubA1, A2>;

    using Table = dispatch_table<FullAxes, int()>;

    int call_count = 0;
    auto factory   = [&call_count](auto) -> int (*)() {
        ++call_count;
        return []() { return 42; };
    };

    Table table(factory, SubAxes{});

    // Should only have called factory for 2 coordinates (1 * 2 = 2)
    EXPECT_EQ(call_count, 2);

    // Should be able to dispatch to those coordinates
    EXPECT_EQ(table(coord(1, 3)), 42);
    EXPECT_EQ(table(coord(1, 4)), 42);

    // Should not be able to dispatch to coordinates not in subspace
    EXPECT_THROW(table(coord(2, 3)), std::runtime_error);
}

TEST(DispatchTable, OnlyValidCombinationsInstantiated) {
    using A1       = axis<1, 2>;
    using A2       = axis<3, 4>;
    using FullAxes = axes<A1, A2>;

    using Table = dispatch_table<FullAxes, int()>;

    int call_count = 0;
    auto factory   = [&call_count](auto) -> int (*)() {
        ++call_count;
        return []() { return 42; };
    };

    Table table(factory, tag<1, 3>{});

    // Should only have called factory once
    EXPECT_EQ(call_count, 1);

    // Should be able to dispatch to that coordinate
    EXPECT_EQ(table(coord(1, 3)), 42);

    // Should not be able to dispatch to other coordinates
    EXPECT_THROW(table(coord(1, 4)), std::runtime_error);
    EXPECT_THROW(table(coord(2, 3)), std::runtime_error);
}

// =============================================================================
// Multiple arguments
// =============================================================================

TEST(DispatchTable, MultipleArguments) {
    using A     = axis<1, 2>;
    using Axes  = axes<A>;
    using Table = dispatch_table<Axes, int(int, int)>;

    auto factory = [](auto c) -> int (*)(int, int) {
        if constexpr (std::is_same_v<decltype(c), tag<1>>) {
            return [](int a, int b) { return a + b; };
        } else {
            return [](int a, int b) { return a * b; };
        }
    };

    Table table(factory, Axes{});

    EXPECT_EQ(table(coord(1), 3, 4), 7);  // 3 + 4
    EXPECT_EQ(table(coord(2), 3, 4), 12); // 3 * 4
}

TEST(DispatchTable, NoArguments) {
    using A     = axis<1>;
    using Axes  = axes<A>;
    using Table = dispatch_table<Axes, int()>;

    auto factory = [](auto) -> int (*)() { return []() { return 42; }; };

    Table table(factory, Axes{});

    EXPECT_EQ(table(coord(1)), 42);
}

} // namespace dispatch
