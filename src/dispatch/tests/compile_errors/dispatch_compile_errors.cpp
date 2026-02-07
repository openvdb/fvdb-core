// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Negative compilation tests for dispatch system.
// Each test case is activated via a preprocessor define and should fail to compile.
//
#include "dispatch/detail.h"
#include "dispatch/dispatch_table.h"
#include "dispatch/enums.h"

int
main() {
#if defined(TEST_MIXED_AXIS_TYPES)
    // ERROR: "axis values must be the same type"
    using bad = dispatch::axis<1, 2.0f>;
    (void)bad{};
#endif

#if defined(TEST_SUBSPACE_NOT_WITHIN)
    // ERROR: "Subs must be within the axes"
    using main_axes = dispatch::axes<dispatch::full_placement_axis>;
    using bad_sub   = dispatch::axes<dispatch::axis<dispatch::determinism::required>>;
    using table     = dispatch::dispatch_table<main_axes, int()>;
    table t("compile_error_test", [](auto) -> int (*)() { return nullptr; }, bad_sub{});
#endif

#if defined(TEST_OP_MISSING_OVERLOAD)
    // ERROR: Template instantiation failure - no matching op() for tag<out_of_place>
    struct incomplete_op {
        static int
        op(dispatch::tag<dispatch::placement::in_place>, int x) {
            return x;
        }
    };
    using test_axes = dispatch::axes<dispatch::full_placement_axis>;
    using table     = dispatch::dispatch_table<test_axes, int(int)>;
    table t("compile_error_test", table::from_op<incomplete_op>(), test_axes{});
#endif

#if defined(TEST_WRONG_TUPLE_TYPE)
    // ERROR: "value type mismatch"
    using test_axes = dispatch::axes<dispatch::full_placement_axis>;
    auto result     = dispatch::linear_index_from_value_tuple(test_axes{}, std::make_tuple(1.0f));
    (void)result;
#endif

    return 0;
}
