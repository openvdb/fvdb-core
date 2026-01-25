// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Negative compilation tests for dispatch system.
// Each test case is activated via a preprocessor define and should fail to compile.
//
#include "dispatch/detail.h"
#include "dispatch/dispatch_table.h"

int
main() {
#if defined(TEST_MIXED_AXIS_TYPES)
    // ERROR: "axis values must be the same type"
    using bad = dispatch::axis<1, 2.0f>;
    (void)bad{};
#endif

#if defined(TEST_SUBSPACE_NOT_WITHIN)
    // ERROR: "Subs must be within the axes"
    using MainAxes   = dispatch::axes<dispatch::axis<1, 2, 3>>;
    using BadSubAxes = dispatch::axes<dispatch::axis<1, 99>>;
    using Table      = dispatch::dispatch_table<MainAxes, int()>;
    Table table([](auto) -> int (*)() { return nullptr; }, BadSubAxes{});
#endif

#if defined(TEST_OP_MISSING_OVERLOAD)
    // ERROR: Template instantiation failure - no matching op() for tag<2>
    struct IncompleteOp {
        static int
        op(dispatch::tag<1>, int x) {
            return x;
        }
    };
    using Axes  = dispatch::axes<dispatch::axis<1, 2>>;
    using Table = dispatch::dispatch_table<Axes, int(int)>;
    Table table(Table::from_op<IncompleteOp>(), Axes{});
#endif

#if defined(TEST_WRONG_TUPLE_TYPE)
    // ERROR: "value type mismatch"
    using Axes  = dispatch::axes<dispatch::axis<1, 2>>;
    auto result = dispatch::linear_index_from_value_tuple(Axes{}, std::make_tuple(1.0f));
    (void)result;
#endif

    return 0;
}
