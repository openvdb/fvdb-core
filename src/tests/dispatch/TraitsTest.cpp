// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <fvdb/detail/dispatch/Traits.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace fvdb {
namespace dispatch {

// =============================================================================
// is_index_sequence Tests
// =============================================================================

TEST(is_index_sequence, DetectsIndexSequence) {
    EXPECT_TRUE((is_index_sequence<std::index_sequence<>>::value));
    EXPECT_TRUE((is_index_sequence<std::index_sequence<1>>::value));
    EXPECT_TRUE((is_index_sequence<std::index_sequence<1, 2, 3>>::value));
}

TEST(is_index_sequence, RejectsNonIndexSequence) {
    EXPECT_FALSE((is_index_sequence<int>::value));
    EXPECT_FALSE((is_index_sequence<std::tuple<int>>::value));
    EXPECT_FALSE((is_index_sequence<void>::value));
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
// tuple_head Tests
// =============================================================================

TEST(tuple_head, SingleElement) {
    auto t      = std::make_tuple(42);
    auto result = tuple_head(t);
    EXPECT_EQ(result, 42);
}

TEST(tuple_head, MultipleElements) {
    auto t      = std::make_tuple(1, 2.0, "three");
    auto result = tuple_head(t);
    EXPECT_EQ(result, 1);
}

TEST(tuple_head, DifferentTypes) {
    auto t      = std::make_tuple(std::string("hello"), 42, 3.14);
    auto result = tuple_head(t);
    EXPECT_EQ(result, "hello");
}

TEST(tuple_head, IsConstexpr) {
    constexpr auto t      = std::make_tuple(10, 20, 30);
    constexpr auto result = tuple_head(t);
    static_assert(result == 10, "Head should be 10");
    SUCCEED();
}

// =============================================================================
// tuple_tail Tests
// =============================================================================

TEST(tuple_tail, TwoElements) {
    auto t        = std::make_tuple(1, 2);
    auto result   = tuple_tail(t);
    auto expected = std::make_tuple(2);
    EXPECT_EQ(result, expected);
}

TEST(tuple_tail, ThreeElements) {
    auto t        = std::make_tuple(1, 2, 3);
    auto result   = tuple_tail(t);
    auto expected = std::make_tuple(2, 3);
    EXPECT_EQ(result, expected);
}

TEST(tuple_tail, SingleElementReturnsEmpty) {
    auto t      = std::make_tuple(42);
    auto result = tuple_tail(t);
    EXPECT_EQ(std::tuple_size_v<decltype(result)>, size_t{0});
}

TEST(tuple_tail, MixedTypes) {
    auto t        = std::make_tuple(1, 2.5, std::string("three"));
    auto result   = tuple_tail(t);
    auto expected = std::make_tuple(2.5, std::string("three"));
    EXPECT_EQ(result, expected);
}

TEST(tuple_tail, IsConstexpr) {
    constexpr auto t        = std::make_tuple(10, 20, 30);
    constexpr auto result   = tuple_tail(t);
    constexpr auto expected = std::make_tuple(20, 30);
    static_assert(result == expected, "Tail should be (20, 30)");
    SUCCEED();
}

// =============================================================================
// TupleHead_t Tests
// =============================================================================

TEST(TupleHead_t, SingleElement) {
    using Tuple    = std::tuple<int>;
    using Expected = int;
    EXPECT_TRUE((std::is_same_v<TupleHead_t<Tuple>, Expected>));
}

TEST(TupleHead_t, MultipleElements) {
    using Tuple    = std::tuple<double, int, char>;
    using Expected = double;
    EXPECT_TRUE((std::is_same_v<TupleHead_t<Tuple>, Expected>));
}

TEST(TupleHead_t, ComplexTypes) {
    using Tuple    = std::tuple<std::string, int, double>;
    using Expected = std::string;
    EXPECT_TRUE((std::is_same_v<TupleHead_t<Tuple>, Expected>));
}

// =============================================================================
// TupleTail_t Tests
// =============================================================================

TEST(TupleTail_t, TwoElements) {
    using Tuple    = std::tuple<int, double>;
    using Expected = std::tuple<double>;
    EXPECT_TRUE((std::is_same_v<TupleTail_t<Tuple>, Expected>));
}

TEST(TupleTail_t, ThreeElements) {
    using Tuple    = std::tuple<int, double, char>;
    using Expected = std::tuple<double, char>;
    EXPECT_TRUE((std::is_same_v<TupleTail_t<Tuple>, Expected>));
}

TEST(TupleTail_t, SingleElementReturnsEmpty) {
    using Tuple    = std::tuple<int>;
    using Expected = std::tuple<>;
    EXPECT_TRUE((std::is_same_v<TupleTail_t<Tuple>, Expected>));
}

TEST(TupleTail_t, ComplexTypes) {
    using Tuple    = std::tuple<std::string, int, std::vector<double>>;
    using Expected = std::tuple<int, std::vector<double>>;
    EXPECT_TRUE((std::is_same_v<TupleTail_t<Tuple>, Expected>));
}

// =============================================================================
// Integration: tuple_head and tuple_tail together
// =============================================================================

TEST(TupleIntegration, HeadAndTailCoverAll) {
    auto t    = std::make_tuple(1, 2, 3);
    auto head = tuple_head(t);
    auto tail = tuple_tail(t);

    EXPECT_EQ(head, 1);
    EXPECT_EQ(tail, std::make_tuple(2, 3));
}

TEST(TupleIntegration, RecursiveDecomposition) {
    // Decompose tuple step by step
    auto t = std::make_tuple(1, 2, 3, 4);

    auto h1 = tuple_head(t);
    auto t1 = tuple_tail(t);
    EXPECT_EQ(h1, 1);

    auto h2 = tuple_head(t1);
    auto t2 = tuple_tail(t1);
    EXPECT_EQ(h2, 2);

    auto h3 = tuple_head(t2);
    auto t3 = tuple_tail(t2);
    EXPECT_EQ(h3, 3);

    auto h4 = tuple_head(t3);
    auto t4 = tuple_tail(t3);
    EXPECT_EQ(h4, 4);
    EXPECT_EQ(std::tuple_size_v<decltype(t4)>, size_t{0});
}

} // namespace dispatch
} // namespace fvdb
