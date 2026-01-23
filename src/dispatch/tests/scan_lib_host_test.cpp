// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include "examples/scan_lib.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

namespace scan_lib {
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

TYPED_TEST(ScanLibHostTest, Serial_Strided_Both) {
    using T = TypeParam;
    std::vector<T> input_buf(200);
    for (int64_t i = 0; i < 100; ++i) {
        input_buf[i * 2] = static_cast<T>(i + 1);
    }
    std::vector<T> output_buf(300, T{0}); // stride 3

    inclusive_scan_serial(input_buf.data(), 2, output_buf.data(), 3, 100);

    EXPECT_EQ(extract_strided(output_buf.data(), 3, 100), expected_scan(make_input<T>(100)));
}

TYPED_TEST(ScanLibHostTest, Serial_Empty) {
    using T = TypeParam;
    std::vector<T> output(1, T{99});

    inclusive_scan_serial<T>(nullptr, 1, output.data(), 1, 0);

    EXPECT_EQ(output[0], T{99}); // unchanged
}

TYPED_TEST(ScanLibHostTest, Serial_SingleElement) {
    using T                    = TypeParam;
    std::vector<T> const input = {T{42}};
    std::vector<T> output(1);

    inclusive_scan_serial(input.data(), 1, output.data(), 1, 1);

    EXPECT_EQ(output[0], T{42});
}

TYPED_TEST(ScanLibHostTest, Serial_LargeInput) {
    using T                    = TypeParam;
    constexpr int64_t n        = 100000;
    std::vector<T> const input = make_input<T>(n);
    std::vector<T> output(n);

    inclusive_scan_serial(input.data(), 1, output.data(), 1, n);

    EXPECT_EQ(output, expected_scan(input));
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

TYPED_TEST(ScanLibHostTest, SerialInplace_Empty) {
    using T             = TypeParam;
    std::vector<T> data = {T{99}};

    inclusive_scan_serial_inplace<T>(data.data(), 1, 0);

    EXPECT_EQ(data[0], T{99}); // unchanged
}

TYPED_TEST(ScanLibHostTest, SerialInplace_SingleElement) {
    using T             = TypeParam;
    std::vector<T> data = {T{42}};

    inclusive_scan_serial_inplace(data.data(), 1, 1);

    EXPECT_EQ(data[0], T{42});
}

// =============================================================================
// Parallel out-of-place tests
// =============================================================================

TYPED_TEST(ScanLibHostTest, Parallel_Contiguous) {
    using T                    = TypeParam;
    std::vector<T> const input = make_input<T>(10000); // larger for parallelism
    std::vector<T> output(10000);

    inclusive_scan_parallel(input.data(), 1, output.data(), 1, 10000);

    // For integers, should match exactly
    if constexpr (std::is_integral_v<T>) {
        EXPECT_EQ(output, expected_scan(input));
    } else {
        // For floats, check approximate equality (non-deterministic due to parallel reduction)
        ASSERT_EQ(output.size(), expected_scan(input).size());
        for (size_t i = 0; i < output.size(); ++i) {
            EXPECT_NEAR(output[i], expected_scan(input)[i], 1e-4);
        }
    }
}

TYPED_TEST(ScanLibHostTest, Parallel_Strided) {
    using T = TypeParam;
    std::vector<T> input_buf(20000);
    for (int64_t i = 0; i < 10000; ++i) {
        input_buf[i * 2] = static_cast<T>(i + 1);
    }
    std::vector<T> output(10000);

    inclusive_scan_parallel(input_buf.data(), 2, output.data(), 1, 10000);

    auto expected = expected_scan(make_input<T>(10000));
    if constexpr (std::is_integral_v<T>) {
        EXPECT_EQ(output, expected);
    } else {
        ASSERT_EQ(output.size(), expected.size());
        for (size_t i = 0; i < output.size(); ++i) {
            EXPECT_NEAR(output[i], expected[i], 1e-4);
        }
    }
}

TYPED_TEST(ScanLibHostTest, Parallel_Empty) {
    using T = TypeParam;
    std::vector<T> output(1, T{99});

    inclusive_scan_parallel<T>(nullptr, 1, output.data(), 1, 0);

    EXPECT_EQ(output[0], T{99}); // unchanged
}

TYPED_TEST(ScanLibHostTest, Parallel_SingleElement) {
    using T                    = TypeParam;
    std::vector<T> const input = {T{42}};
    std::vector<T> output(1);

    inclusive_scan_parallel(input.data(), 1, output.data(), 1, 1);

    EXPECT_EQ(output[0], T{42});
}

TYPED_TEST(ScanLibHostTest, Parallel_LargeInput) {
    using T                    = TypeParam;
    constexpr int64_t n        = 100000;
    std::vector<T> const input = make_input<T>(n);
    std::vector<T> output(n);

    inclusive_scan_parallel(input.data(), 1, output.data(), 1, n);

    auto expected = expected_scan(input);
    if constexpr (std::is_integral_v<T>) {
        EXPECT_EQ(output, expected);
    } else {
        ASSERT_EQ(output.size(), expected.size());
        for (size_t i = 0; i < output.size(); ++i) {
            EXPECT_NEAR(output[i], expected[i], 1e-4);
        }
    }
}

// =============================================================================
// Determinism tests
// =============================================================================

TYPED_TEST(ScanLibHostTest, Serial_Deterministic) {
    using T                    = TypeParam;
    std::vector<T> const input = make_input<T>(100);
    std::vector<T> output1(100);
    std::vector<T> output2(100);

    inclusive_scan_serial(input.data(), 1, output1.data(), 1, 100);
    inclusive_scan_serial(input.data(), 1, output2.data(), 1, 100);

    EXPECT_EQ(output1, output2); // Should be identical
}

TYPED_TEST(ScanLibHostTest, Parallel_IntegerDeterministic) {
    using T = TypeParam;
    if constexpr (std::is_integral_v<T>) {
        std::vector<T> const input = make_input<T>(10000);
        std::vector<T> output1(10000);
        std::vector<T> output2(10000);

        inclusive_scan_parallel(input.data(), 1, output1.data(), 1, 10000);
        inclusive_scan_parallel(input.data(), 1, output2.data(), 1, 10000);

        EXPECT_EQ(output1, output2); // Integers should be deterministic
    } else {
        GTEST_SKIP() << "Float types are non-deterministic in parallel";
    }
}

} // namespace
} // namespace scan_lib
