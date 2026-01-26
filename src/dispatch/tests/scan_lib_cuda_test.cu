// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include "examples/scan_lib.h"
#include "test_utils.cuh"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace scan_lib {
namespace {

// =============================================================================
// CUDA utilities
// =============================================================================

/// RAII wrapper for raw device memory (for temp storage)
class DeviceRawBuffer {
    void *ptr_    = nullptr;
    size_t bytes_ = 0;

  public:
    explicit DeviceRawBuffer(size_t bytes) : bytes_(bytes) {
        if (bytes > 0) {
            cudaError_t err = cudaMalloc(&ptr_, bytes);
            if (err != cudaSuccess) {
                throw std::runtime_error(std::string("cudaMalloc failed: ") +
                                         cudaGetErrorString(err));
            }
        }
    }
    ~DeviceRawBuffer() {
        if (ptr_)
            cudaFree(ptr_);
    }
    DeviceRawBuffer(DeviceRawBuffer const &)            = delete;
    DeviceRawBuffer &operator=(DeviceRawBuffer const &) = delete;

    void *
    get() {
        return ptr_;
    }
    size_t
    size() const {
        return bytes_;
    }
};

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

/// Create test input with cyclic pattern: [1, 2, ..., k, 1, 2, ..., k, ...]
/// Using bounded cyclic values avoids integer overflow in scan results for large n,
/// while still providing varied data that can catch index/ordering bugs.
/// For n=100000 with k=100, the maximum scan sum is ~5 million, safely within int32_t.
template <typename T>
std::vector<T>
make_input(int64_t n) {
    constexpr int64_t k = 100; // cycle length
    std::vector<T> v(n);
    for (int64_t i = 0; i < n; ++i) {
        v[i] = static_cast<T>((i % k) + 1);
    }
    return v;
}

/// Compare scan results with appropriate tolerance for floating-point types.
/// Parallel scans are not deterministic for floats due to non-associativity,
/// so we use relative tolerance. Integers are compared exactly.
template <typename T>
void
expect_scan_equal(std::vector<T> const &actual, std::vector<T> const &expected) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        if constexpr (std::is_floating_point_v<T>) {
            // Relative tolerance + small absolute tolerance for values near zero
            T const rel_tol = T{1e-4};
            T const abs_tol = T{1e-6};
            T const tol     = std::abs(expected[i]) * rel_tol + abs_tol;
            EXPECT_NEAR(actual[i], expected[i], tol) << "at index " << i;
        } else {
            EXPECT_EQ(actual[i], expected[i]) << "at index " << i;
        }
    }
}

// =============================================================================
// Typed test fixture for all scalar types
// =============================================================================

template <typename T> class ScanLibCudaTest : public ::testing::Test {};

using TestTypes = ::testing::Types<float, double, int32_t, int64_t>;
TYPED_TEST_SUITE(ScanLibCudaTest, TestTypes);

// =============================================================================
// CUDA scan tests
// =============================================================================

TYPED_TEST(ScanLibCudaTest, Basic) {
    using T         = TypeParam;
    int64_t const n = 1000;

    std::vector<T> const input = make_input<T>(n);
    std::vector<T> output(n);

    dispatch::test::DeviceBuffer<T> d_in(n);
    dispatch::test::DeviceBuffer<T> d_out(n);
    dispatch::test::copy_to_device(input.data(), d_in.get(), n);

    size_t temp_bytes = inclusive_scan_cuda_temp_bytes<T>(n);
    DeviceRawBuffer temp(temp_bytes);

    inclusive_scan_cuda(d_in.get(), d_out.get(), n, temp.get(), temp.size(), nullptr);
    cudaDeviceSynchronize();

    dispatch::test::copy_to_host(d_out.get(), output.data(), n);

    expect_scan_equal(output, expected_scan(input));
}

TYPED_TEST(ScanLibCudaTest, Large) {
    using T         = TypeParam;
    int64_t const n = 100000;

    std::vector<T> const input = make_input<T>(n);
    std::vector<T> output(n);

    dispatch::test::DeviceBuffer<T> d_in(n);
    dispatch::test::DeviceBuffer<T> d_out(n);
    dispatch::test::copy_to_device(input.data(), d_in.get(), n);

    size_t temp_bytes = inclusive_scan_cuda_temp_bytes<T>(n);
    DeviceRawBuffer temp(temp_bytes);

    inclusive_scan_cuda(d_in.get(), d_out.get(), n, temp.get(), temp.size(), nullptr);
    cudaDeviceSynchronize();

    dispatch::test::copy_to_host(d_out.get(), output.data(), n);

    expect_scan_equal(output, expected_scan(input));
}

TYPED_TEST(ScanLibCudaTest, SingleElement) {
    using T               = TypeParam;
    std::vector<T> input  = {T{42}};
    std::vector<T> output = {T{0}};

    dispatch::test::DeviceBuffer<T> d_in(1);
    dispatch::test::DeviceBuffer<T> d_out(1);
    dispatch::test::copy_to_device(input.data(), d_in.get(), 1);

    size_t temp_bytes = inclusive_scan_cuda_temp_bytes<T>(1);
    DeviceRawBuffer temp(temp_bytes);

    inclusive_scan_cuda(d_in.get(), d_out.get(), 1, temp.get(), temp.size(), nullptr);
    cudaDeviceSynchronize();

    dispatch::test::copy_to_host(d_out.get(), output.data(), 1);

    EXPECT_EQ(output[0], T{42});
}

TYPED_TEST(ScanLibCudaTest, Empty) {
    using T = TypeParam;

    // Should not crash
    size_t temp_bytes = inclusive_scan_cuda_temp_bytes<T>(0);
    EXPECT_EQ(temp_bytes, 0u);

    inclusive_scan_cuda<T>(nullptr, nullptr, 0, nullptr, 0, nullptr);
    // No crash = success
}

TYPED_TEST(ScanLibCudaTest, WithStream) {
    using T         = TypeParam;
    int64_t const n = 500;

    std::vector<T> const input = make_input<T>(n);
    std::vector<T> output(n);

    dispatch::test::DeviceBuffer<T> d_in(n);
    dispatch::test::DeviceBuffer<T> d_out(n);
    dispatch::test::copy_to_device(input.data(), d_in.get(), n);

    size_t temp_bytes = inclusive_scan_cuda_temp_bytes<T>(n);
    DeviceRawBuffer temp(temp_bytes);

    cudaStream_t stream;
    cudaError_t stream_err = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, stream_err)
        << "cudaStreamCreate failed: " << cudaGetErrorString(stream_err);

    inclusive_scan_cuda(d_in.get(), d_out.get(), n, temp.get(), temp.size(), stream);
    cudaStreamSynchronize(stream);

    dispatch::test::copy_to_host(d_out.get(), output.data(), n);
    cudaStreamDestroy(stream);

    expect_scan_equal(output, expected_scan(input));
}

// =============================================================================
// Temp bytes query test
// =============================================================================

TYPED_TEST(ScanLibCudaTest, TempBytesIncreases) {
    using T = TypeParam;

    size_t bytes_small  = inclusive_scan_cuda_temp_bytes<T>(100);
    size_t bytes_medium = inclusive_scan_cuda_temp_bytes<T>(10000);
    size_t bytes_large  = inclusive_scan_cuda_temp_bytes<T>(1000000);

    // Temp storage should be non-zero for non-empty
    EXPECT_GT(bytes_small, 0u);
    // Generally increases with size (or stays same for CUB's implementation)
    EXPECT_GE(bytes_large, bytes_small);
    EXPECT_GE(bytes_medium, bytes_small);
}

} // namespace
} // namespace scan_lib
