// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Negative compilation tests for dispatch system.
// Each test case is activated via a preprocessor define and should fail to compile.
//
// Existing tests validate axis/dispatch_table invariants.
// New tests validate tag uniqueness and axes label uniqueness.
//
#include "dispatch/detail.h"
#include "dispatch/dispatch_table.h"
#include "dispatch/enums.h"
#include "dispatch/with_value.h"

int
main() {
    // ============================================================================
    // Axis invariants
    // ============================================================================

#if defined(TEST_MIXED_AXIS_TYPES)
    // ERROR: "axis values must be the same type"
    // axis requires all values to share a single type.
    using bad = dispatch::axis<dispatch::placement::in_place, dispatch::determinism::required>;
    (void)bad{};
#endif

    // ============================================================================
    // Tag invariants
    // ============================================================================

#if defined(TEST_TAG_DUPLICATE_TYPES)
    // ERROR: "tag values must have unique types"
    // tag<A, B> requires decltype(A) != decltype(B).
    using bad = dispatch::tag<dispatch::placement::in_place, dispatch::placement::out_of_place>;
    (void)bad{};
#endif

    // ============================================================================
    // Axes invariants
    // ============================================================================

#if defined(TEST_AXES_DUPLICATE_VALUE_TYPES)
    // ERROR: "All axes must have unique value types"
    // axes<A, B> requires each axis to have a distinct value type.
    using axis_a = dispatch::axis<dispatch::placement::in_place, dispatch::placement::out_of_place>;
    using axis_b = dispatch::axis<dispatch::placement::in_place>;
    using bad    = dispatch::axes<axis_a, axis_b>;
    (void)bad{};
#endif

    // ============================================================================
    // Dispatch table invariants
    // ============================================================================

#if defined(TEST_SUBSPACE_NOT_WITHIN)
    // ERROR: "Subs must be within the axes"
    // A subspace axis type must exist in the table's space.
    using main_axes = dispatch::axes<dispatch::full_placement_axis>;
    using bad_sub   = dispatch::axes<dispatch::axis<dispatch::determinism::required>>;
    using table     = dispatch::dispatch_table<main_axes, int()>;
    table t("compile_error_test", [](auto) -> int (*)() { return nullptr; }, bad_sub{});
#endif

#if defined(TEST_OP_MISSING_OVERLOAD)
    // ERROR: Template instantiation failure - no matching op() for tag<out_of_place>
    // from_op requires Op::op to be callable for every point in the subspace.
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
    // linear_index_from_value_tuple requires tuple element types to match axis value types.
    using test_axes = dispatch::axes<dispatch::full_placement_axis>;
    auto result     = dispatch::linear_index_from_value_tuple(test_axes{}, std::make_tuple(1.0f));
    (void)result;
#endif

    return 0;
}
