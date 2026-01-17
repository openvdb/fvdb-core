// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <fvdb/detail/dispatch/example/ScanLib.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <numeric>
#include <vector>

namespace scanlib {
namespace {

// =============================================================================
// Test utilities
// =============================================================================

/// Compute expected inclusive scan result
template <typename T>
std::vector<T>
expected_scan(std::vector<T> const &input) {
    std::vector<T> result(input.size());
    if (!input.empty()) {
        std::partial_sum(input.begin(), input.end(), result.begin());
    }
    return result;
}

/// Extract strided elements from a buffer
template <typename T>
std::vector<T>
extract_strided(T const *data, int64_t stride, int64_t n) {
    std::vector<T> result(n);
    for (int64_t i = 0; i < n; ++i) {
        result[i] = data[i * stride];
    }
    return result;
}

/// Create test input: [1, 2, 3, ..., n] cast to T
template <typename T>
std::vector<T>
make_input(int64_t n) {
    std::vector<T> v(n);
    for (int64_t i = 0; i < n; ++i) {
        v[i] = static_cast<T>(i + 1);
    }
    return v;
}

// =============================================================================
// Typed test fixture for all scalar types
// =============================================================================

template <typename T> class ScanLibHostTest : public ::testing::Test {};

using TestTypes = ::testing::Types<float, double, int32_t, int64_t>;
TYPED_TEST_SUITE(ScanLibHostTest, TestTypes);

// =============================================================================
// Serial out-of-place tests
// =============================================================================

TYPED_TEST(ScanLibHostTest, Serial_Contiguous) {
    using T                    = TypeParam;
    std::vector<T> const input = make_input<T>(100);
    std::vector<T> output(100);

    inclusive_scan_serial(input.data(), 1, output.data(), 1, 100);

    EXPECT_EQ(output, expected_scan(input));
}

TYPED_TEST(ScanLibHostTest, Serial_Strided_Input) {
    using T = TypeParam;
    // Input with stride 2: use every other element
    std::vector<T> input_buf(200);
    for (int64_t i = 0; i < 100; ++i) {
        input_buf[i * 2] = static_cast<T>(i + 1);
    }
    std::vector<T> output(100);

    inclusive_scan_serial(input_buf.data(), 2, output.data(), 1, 100);

    EXPECT_EQ(output, expected_scan(make_input<T>(100)));
}

TYPED_TEST(ScanLibHostTest, Serial_Strided_Output) {
    using T                    = TypeParam;
    std::vector<T> const input = make_input<T>(50);
    std::vector<T> output_buf(150, T{0}); // stride 3

    inclusive_scan_serial(input.data(), 1, output_buf.data(), 3, 50);

    EXPECT_EQ(extract_strided(output_buf.data(), 3, 50), expected_scan(input));
}

TYPED_TEST(ScanLibHostTest, Serial_Empty) {
    using T = TypeParam;
    std::vector<T> output(1, T{99});

    inclusive_scan_serial<T>(nullptr, 1, output.data(), 1, 0);

    EXPECT_EQ(output[0], T{99}); // unchanged
}

// =============================================================================
// Serial in-place tests
// =============================================================================

TYPED_TEST(ScanLibHostTest, SerialInplace_Contiguous) {
    using T             = TypeParam;
    std::vector<T> data = make_input<T>(100);
    auto expected       = expected_scan(data);

    inclusive_scan_serial_inplace(data.data(), 1, 100);

    EXPECT_EQ(data, expected);
}

TYPED_TEST(ScanLibHostTest, SerialInplace_Strided) {
    using T = TypeParam;
    std::vector<T> buf(200, T{0});
    for (int64_t i = 0; i < 100; ++i) {
        buf[i * 2] = static_cast<T>(i + 1);
    }

    inclusive_scan_serial_inplace(buf.data(), 2, 100);

    EXPECT_EQ(extract_strided(buf.data(), 2, 100), expected_scan(make_input<T>(100)));
}

// =============================================================================
// Parallel out-of-place tests
// =============================================================================

TYPED_TEST(ScanLibHostTest, Parallel_Contiguous) {
    using T                    = TypeParam;
    std::vector<T> const input = make_input<T>(1000); // larger for parallelism
    std::vector<T> output(1000);

    inclusive_scan_parallel(input.data(), 1, output.data(), 1, 1000);

    EXPECT_EQ(output, expected_scan(input));
}

TYPED_TEST(ScanLibHostTest, Parallel_Strided) {
    using T = TypeParam;
    std::vector<T> input_buf(2000);
    for (int64_t i = 0; i < 1000; ++i) {
        input_buf[i * 2] = static_cast<T>(i + 1);
    }
    std::vector<T> output(1000);

    inclusive_scan_parallel(input_buf.data(), 2, output.data(), 1, 1000);

    EXPECT_EQ(output, expected_scan(make_input<T>(1000)));
}

TYPED_TEST(ScanLibHostTest, Parallel_Small_FallsBackToSerial) {
    using T                    = TypeParam;
    std::vector<T> const input = make_input<T>(10); // small, should use serial path
    std::vector<T> output(10);

    inclusive_scan_parallel(input.data(), 1, output.data(), 1, 10);

    EXPECT_EQ(output, expected_scan(input));
}

// =============================================================================
// Edge cases (applied to serial as representative)
// =============================================================================

TYPED_TEST(ScanLibHostTest, SingleElement) {
    using T              = TypeParam;
    std::vector<T> input = {T{42}};
    std::vector<T> output(1);

    inclusive_scan_serial(input.data(), 1, output.data(), 1, 1);

    EXPECT_EQ(output[0], T{42});
}

TYPED_TEST(ScanLibHostTest, TwoElements) {
    using T              = TypeParam;
    std::vector<T> input = {T{3}, T{5}};
    std::vector<T> output(2);

    inclusive_scan_serial(input.data(), 1, output.data(), 1, 2);

    EXPECT_EQ(output[0], T{3});
    EXPECT_EQ(output[1], T{8});
}

} // namespace
} // namespace scanlib
