// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include "dispatch/dispatch_table.h"
#include "dispatch/types.h"

#include <gtest/gtest.h>

#include <functional>
#include <stdexcept>
#include <string>
#include <tuple>

namespace dispatch {

// =============================================================================
// Factory methods
// =============================================================================

// Test op struct
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
    using A     = axis<1, 2>;
    using Axes  = axes<A>;
    using Table = dispatch_table<Axes, int(int)>;

    auto factory = Table::from_op<TestOp>();
    Table table(factory, Axes{});

    auto result1 = table(std::make_tuple(1), 5);
    EXPECT_EQ(result1, 10); // 5 * 2

    auto result2 = table(std::make_tuple(2), 5);
    EXPECT_EQ(result2, 15); // 5 * 3
}

// Test visitor (functor)
struct TestVisitor {
    int
    operator()(tag<1>, int x) const {
        return x * 10;
    }
    int
    operator()(tag<2>, int x) const {
        return x * 20;
    }
};

TEST(DispatchTable, FromVisitor) {
    using A     = axis<1, 2>;
    using Axes  = axes<A>;
    using Table = dispatch_table<Axes, int(int)>;

    TestVisitor visitor;
    auto factory = Table::from_visitor(visitor);
    Table table(factory, Axes{});

    auto result1 = table(std::make_tuple(1), 3);
    EXPECT_EQ(result1, 30); // 3 * 10

    auto result2 = table(std::make_tuple(2), 3);
    EXPECT_EQ(result2, 60); // 3 * 20
}

// Test visitor lambda
TEST(DispatchTable, FromVisitorLambda) {
    using A     = axis<1, 2>;
    using Axes  = axes<A>;
    using Table = dispatch_table<Axes, int(int)>;

    int captured = 100;
    auto factory = Table::from_visitor([captured](auto coord, int x) -> int {
        if constexpr (std::is_same_v<decltype(coord), tag<1>>) {
            return x + captured;
        } else {
            return x - captured;
        }
    });
    Table table(factory, Axes{});

    auto result1 = table(std::make_tuple(1), 5);
    EXPECT_EQ(result1, 105); // 5 + 100

    auto result2 = table(std::make_tuple(2), 5);
    EXPECT_EQ(result2, -95); // 5 - 100
}

// =============================================================================
// Dispatch
// =============================================================================

TEST(DispatchTable, InvokesCorrectHandler) {
    using A     = axis<1, 2, 3>;
    using Axes  = axes<A>;
    using Table = dispatch_table<Axes, std::string()>;

    auto factory = [](auto coord) -> std::string (*)() {
        if constexpr (std::is_same_v<decltype(coord), tag<1>>) {
            return []() { return std::string("one"); };
        } else if constexpr (std::is_same_v<decltype(coord), tag<2>>) {
            return []() { return std::string("two"); };
        } else {
            return []() { return std::string("three"); };
        }
    };

    Table table(factory, Axes{});

    EXPECT_EQ(table(std::make_tuple(1)), "one");
    EXPECT_EQ(table(std::make_tuple(2)), "two");
    EXPECT_EQ(table(std::make_tuple(3)), "three");
}

TEST(DispatchTable, ThrowsForMissingCoordinates) {
    using A     = axis<1, 2>;
    using Axes  = axes<A>;
    using Table = dispatch_table<Axes, int()>;

    auto factory = [](auto coord) -> int (*)() { return []() { return 42; }; };

    Table table(factory, Axes{});

    // Valid coordinate
    EXPECT_EQ(table(std::make_tuple(1)), 42);

    // Invalid coordinate - should throw
    EXPECT_THROW(table(std::make_tuple(99)), std::runtime_error);
}

TEST(DispatchTable, ThrowsForNullHandlers) {
    using A     = axis<1, 2>;
    using Axes  = axes<A>;
    using Table = dispatch_table<Axes, int()>;

    // Create table with null handler
    axes_map<Axes, int (*)()> map;
    map.emplace(tag<1>{}, nullptr);
    map.emplace(tag<2>{}, []() { return 42; });

    // Manually create table with null handler
    auto data = std::make_shared<Table::map_type>(std::move(map));
    Table table(data);

    // Valid coordinate with null handler - should throw
    EXPECT_THROW(table(std::make_tuple(1)), std::runtime_error);

    // Valid coordinate with non-null handler - should work
    EXPECT_EQ(table(std::make_tuple(2)), 42);
}

// =============================================================================
// Functional update
// =============================================================================

TEST(DispatchTable, WithReturnsNewTable) {
    using A     = axis<1, 2>;
    using Axes  = axes<A>;
    using Table = dispatch_table<Axes, int()>;

    auto factory1 = [](auto coord) -> int (*)() { return []() { return 10; }; };
    Table table1(factory1, Axes{});

    auto factory2 = [](auto coord) -> int (*)() {
        if constexpr (std::is_same_v<decltype(coord), tag<2>>) {
            return []() { return 20; };
        }
        return nullptr;
    };

    Table table2 = table1.with(factory2, tag<2>{});

    // Original table unchanged
    EXPECT_EQ(table1(std::make_tuple(1)), 10);
    EXPECT_EQ(table1(std::make_tuple(2)), 10);

    // New table has override
    EXPECT_EQ(table2(std::make_tuple(1)), 10);
    EXPECT_EQ(table2(std::make_tuple(2)), 20);
}

TEST(DispatchTable, OriginalTableUnchangedAfterWith) {
    using A     = axis<1, 2>;
    using Axes  = axes<A>;
    using Table = dispatch_table<Axes, int()>;

    auto factory1 = [](auto coord) -> int (*)() { return []() { return 100; }; };
    Table table1(factory1, Axes{});

    auto factory2 = [](auto coord) -> int (*)() { return []() { return 200; }; };

    Table table2 = table1.with(factory2, Axes{});

    // Original should still have original values
    EXPECT_EQ(table1(std::make_tuple(1)), 100);
    EXPECT_EQ(table1(std::make_tuple(2)), 100);

    // New table should have new values
    EXPECT_EQ(table2(std::make_tuple(1)), 200);
    EXPECT_EQ(table2(std::make_tuple(2)), 200);
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
    auto factory   = [&call_count](auto coord) -> int (*)() {
        ++call_count;
        return []() { return 42; };
    };

    Table table(factory, SubAxes{});

    // Should only have called factory for 2 coordinates (1 * 2 = 2)
    EXPECT_EQ(call_count, 2);

    // Should be able to dispatch to those coordinates
    EXPECT_EQ(table(std::make_tuple(1, 3)), 42);
    EXPECT_EQ(table(std::make_tuple(1, 4)), 42);

    // Should not be able to dispatch to coordinates not in subspace
    EXPECT_THROW(table(std::make_tuple(2, 3)), std::runtime_error);
}

TEST(DispatchTable, OnlyValidCombinationsInstantiated) {
    using A1       = axis<1, 2>;
    using A2       = axis<3, 4>;
    using FullAxes = axes<A1, A2>;

    using Table = dispatch_table<FullAxes, int()>;

    int call_count = 0;
    auto factory   = [&call_count](auto coord) -> int (*)() {
        ++call_count;
        return []() { return call_count; };
    };

    // Create table with only one coordinate
    Table table(factory, tag<1, 3>{});

    // Should only have called factory once
    EXPECT_EQ(call_count, 1);

    // Should be able to dispatch to that coordinate
    EXPECT_EQ(table(std::make_tuple(1, 3)), 1);

    // Should not be able to dispatch to other coordinates
    EXPECT_THROW(table(std::make_tuple(1, 4)), std::runtime_error);
    EXPECT_THROW(table(std::make_tuple(2, 3)), std::runtime_error);
}

// =============================================================================
// Multiple arguments
// =============================================================================

TEST(DispatchTable, MultipleArguments) {
    using A     = axis<1, 2>;
    using Axes  = axes<A>;
    using Table = dispatch_table<Axes, int(int, int)>;

    auto factory = [](auto coord) -> int (*)(int, int) {
        if constexpr (std::is_same_v<decltype(coord), tag<1>>) {
            return [](int a, int b) { return a + b; };
        } else {
            return [](int a, int b) { return a * b; };
        }
    };

    Table table(factory, Axes{});

    EXPECT_EQ(table(std::make_tuple(1), 3, 4), 7);  // 3 + 4
    EXPECT_EQ(table(std::make_tuple(2), 3, 4), 12); // 3 * 4
}

TEST(DispatchTable, NoArguments) {
    using A     = axis<1>;
    using Axes  = axes<A>;
    using Table = dispatch_table<Axes, int()>;

    auto factory = [](auto coord) -> int (*)() { return []() { return 42; }; };

    Table table(factory, Axes{});

    EXPECT_EQ(table(std::make_tuple(1)), 42);
}

// =============================================================================
// Copy semantics
// =============================================================================

TEST(DispatchTable, CopyConstruction) {
    using A     = axis<1>;
    using Axes  = axes<A>;
    using Table = dispatch_table<Axes, int()>;

    auto factory = [](auto coord) -> int (*)() { return []() { return 100; }; };

    Table table1(factory, Axes{});
    Table table2(table1);

    EXPECT_EQ(table1(std::make_tuple(1)), 100);
    EXPECT_EQ(table2(std::make_tuple(1)), 100);
}

TEST(DispatchTable, CopyAssignment) {
    using A     = axis<1>;
    using Axes  = axes<A>;
    using Table = dispatch_table<Axes, int()>;

    auto factory1 = [](auto coord) -> int (*)() { return []() { return 100; }; };
    auto factory2 = [](auto coord) -> int (*)() { return []() { return 200; }; };

    Table table1(factory1, Axes{});
    Table table2(factory2, Axes{});

    table2 = table1;

    EXPECT_EQ(table1(std::make_tuple(1)), 100);
    EXPECT_EQ(table2(std::make_tuple(1)), 100);
}

TEST(DispatchTable, MoveConstruction) {
    using A     = axis<1>;
    using Axes  = axes<A>;
    using Table = dispatch_table<Axes, int()>;

    auto factory = [](auto coord) -> int (*)() { return []() { return 100; }; };

    Table table1(factory, Axes{});
    Table table2(std::move(table1));

    EXPECT_EQ(table2(std::make_tuple(1)), 100);
}

TEST(DispatchTable, MoveAssignment) {
    using A     = axis<1>;
    using Axes  = axes<A>;
    using Table = dispatch_table<Axes, int()>;

    auto factory1 = [](auto coord) -> int (*)() { return []() { return 100; }; };
    auto factory2 = [](auto coord) -> int (*)() { return []() { return 200; }; };

    Table table1(factory1, Axes{});
    Table table2(factory2, Axes{});

    table2 = std::move(table1);

    EXPECT_EQ(table2(std::make_tuple(1)), 100);
}

} // namespace dispatch
