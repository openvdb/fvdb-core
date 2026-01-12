// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <fvdb/detail/dispatch/Traits.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <type_traits>

namespace fvdb {
namespace dispatch {

// =============================================================================
// strides_helper_t Tests
// =============================================================================

// For a shape <D0, D1, D2, ...>, strides are <D1*D2*..., D2*D3*..., ..., 1>
// i.e., row-major (C-order) strides where stride[i] = product of dims[i+1:]

TEST(strides_helper_t, SingleDimension) {
    // Shape <10> → Strides <1>
    using Shape    = std::index_sequence<10>;
    using Strides  = strides_helper_t<Shape>;
    using Expected = std::index_sequence<1>;
    EXPECT_TRUE((std::is_same_v<Strides, Expected>));
}

TEST(strides_helper_t, TwoDimensions) {
    // Shape <3, 4> → Strides <4, 1>
    using Shape    = std::index_sequence<3, 4>;
    using Strides  = strides_helper_t<Shape>;
    using Expected = std::index_sequence<4, 1>;
    EXPECT_TRUE((std::is_same_v<Strides, Expected>));
}

TEST(strides_helper_t, ThreeDimensions) {
    // Shape <2, 3, 4> → Strides <12, 4, 1>
    // stride[0] = 3*4 = 12
    // stride[1] = 4
    // stride[2] = 1
    using Shape    = std::index_sequence<2, 3, 4>;
    using Strides  = strides_helper_t<Shape>;
    using Expected = std::index_sequence<12, 4, 1>;
    EXPECT_TRUE((std::is_same_v<Strides, Expected>));
}

TEST(strides_helper_t, FourDimensions) {
    // Shape <2, 3, 4, 5> → Strides <60, 20, 5, 1>
    // stride[0] = 3*4*5 = 60
    // stride[1] = 4*5 = 20
    // stride[2] = 5
    // stride[3] = 1
    using Shape    = std::index_sequence<2, 3, 4, 5>;
    using Strides  = strides_helper_t<Shape>;
    using Expected = std::index_sequence<60, 20, 5, 1>;
    EXPECT_TRUE((std::is_same_v<Strides, Expected>));
}

TEST(strides_helper_t, AllOnes) {
    // Shape <1, 1, 1> → Strides <1, 1, 1>
    using Shape    = std::index_sequence<1, 1, 1>;
    using Strides  = strides_helper_t<Shape>;
    using Expected = std::index_sequence<1, 1, 1>;
    EXPECT_TRUE((std::is_same_v<Strides, Expected>));
}

TEST(strides_helper_t, LargeDimensions) {
    // Shape <100, 200> → Strides <200, 1>
    using Shape    = std::index_sequence<100, 200>;
    using Strides  = strides_helper_t<Shape>;
    using Expected = std::index_sequence<200, 1>;
    EXPECT_TRUE((std::is_same_v<Strides, Expected>));
}

TEST(strides_helper_t, NonUniformDimensions) {
    // Shape <5, 1, 7> → Strides <7, 7, 1>
    // stride[0] = 1*7 = 7
    // stride[1] = 7
    // stride[2] = 1
    using Shape    = std::index_sequence<5, 1, 7>;
    using Strides  = strides_helper_t<Shape>;
    using Expected = std::index_sequence<7, 7, 1>;
    EXPECT_TRUE((std::is_same_v<Strides, Expected>));
}

// =============================================================================
// array_from_indices Tests
// =============================================================================

TEST(array_from_indices, EmptySequence) {
    using Seq          = std::index_sequence<>;
    constexpr auto arr = array_from_indices<Seq>::value;
    EXPECT_EQ(arr.size(), size_t{0});
}

TEST(array_from_indices, SingleElement) {
    using Seq          = std::index_sequence<42>;
    constexpr auto arr = array_from_indices<Seq>::value;
    EXPECT_EQ(arr.size(), size_t{1});
    EXPECT_EQ(arr[0], size_t{42});
}

TEST(array_from_indices, MultipleElements) {
    using Seq          = std::index_sequence<1, 2, 3, 4, 5>;
    constexpr auto arr = array_from_indices<Seq>::value;
    EXPECT_EQ(arr.size(), size_t{5});
    EXPECT_EQ(arr[0], size_t{1});
    EXPECT_EQ(arr[1], size_t{2});
    EXPECT_EQ(arr[2], size_t{3});
    EXPECT_EQ(arr[3], size_t{4});
    EXPECT_EQ(arr[4], size_t{5});
}

TEST(array_from_indices, ZeroValues) {
    using Seq          = std::index_sequence<0, 0, 0>;
    constexpr auto arr = array_from_indices<Seq>::value;
    EXPECT_EQ(arr.size(), size_t{3});
    EXPECT_EQ(arr[0], size_t{0});
    EXPECT_EQ(arr[1], size_t{0});
    EXPECT_EQ(arr[2], size_t{0});
}

TEST(array_from_indices, LargeValues) {
    using Seq          = std::index_sequence<1000000, 2000000>;
    constexpr auto arr = array_from_indices<Seq>::value;
    EXPECT_EQ(arr.size(), size_t{2});
    EXPECT_EQ(arr[0], size_t{1000000});
    EXPECT_EQ(arr[1], size_t{2000000});
}

TEST(array_from_indices, IsConstexpr) {
    // Verify the array is usable at compile time
    using Seq          = std::index_sequence<10, 20, 30>;
    constexpr auto arr = array_from_indices<Seq>::value;
    static_assert(arr.size() == 3, "Size should be 3");
    static_assert(arr[0] == 10, "First element should be 10");
    static_assert(arr[1] == 20, "Second element should be 20");
    static_assert(arr[2] == 30, "Third element should be 30");
    SUCCEED(); // Test passes if compilation succeeds
}

// =============================================================================
// Integration: strides_helper_t with array_from_indices
// =============================================================================

TEST(TraitsIntegration, StridesToArray) {
    // Convert strides to an array for runtime use
    using Shape                = std::index_sequence<2, 3, 4>;
    using Strides              = strides_helper_t<Shape>;
    constexpr auto strides_arr = array_from_indices<Strides>::value;

    EXPECT_EQ(strides_arr.size(), size_t{3});
    EXPECT_EQ(strides_arr[0], size_t{12}); // 3*4
    EXPECT_EQ(strides_arr[1], size_t{4});  // 4
    EXPECT_EQ(strides_arr[2], size_t{1});  // 1
}

TEST(TraitsIntegration, LinearIndexComputation) {
    // Verify strides can be used for linear index computation
    // For shape <2, 3, 4>, element at [i, j, k] has linear index: i*12 + j*4 + k
    using Shape            = std::index_sequence<2, 3, 4>;
    using Strides          = strides_helper_t<Shape>;
    constexpr auto strides = array_from_indices<Strides>::value;

    // Element [0, 0, 0] → linear index 0
    EXPECT_EQ(0 * strides[0] + 0 * strides[1] + 0 * strides[2], size_t{0});

    // Element [1, 0, 0] → linear index 12
    EXPECT_EQ(1 * strides[0] + 0 * strides[1] + 0 * strides[2], size_t{12});

    // Element [0, 1, 0] → linear index 4
    EXPECT_EQ(0 * strides[0] + 1 * strides[1] + 0 * strides[2], size_t{4});

    // Element [0, 0, 1] → linear index 1
    EXPECT_EQ(0 * strides[0] + 0 * strides[1] + 1 * strides[2], size_t{1});

    // Element [1, 2, 3] → linear index 1*12 + 2*4 + 3 = 23
    EXPECT_EQ(1 * strides[0] + 2 * strides[1] + 3 * strides[2], size_t{23});

    // Last element [1, 2, 3] should be total_size - 1 = 2*3*4 - 1 = 23
    EXPECT_EQ(1 * strides[0] + 2 * strides[1] + 3 * strides[2], size_t{2 * 3 * 4 - 1});
}

} // namespace dispatch
} // namespace fvdb
