// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Purpose: Test CUDA scan implementations and verify nvcc compilation of scan_lib templates.

#include "examples/scan_lib.h"
#include "test_utils.cuh"

#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include <cmath>
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

// RAII wrapper for raw device memory (for temp storage)
class DeviceRawBuffer {
    void *ptr_    = nullptr;
    size_t bytes_ = 0;

  public:
    explicit DeviceRawBuffer(size_t bytes) : bytes_(bytes) {
        if (bytes > 0) {
            cudaMalloc(&ptr_, bytes);
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
// Typed test fixture for all scalar types
// =============================================================================

template <typename T> class ScanLibCudaTest : public ::testing::Test {};

using TestTypes = ::testing::Types<float, double, int32_t, int64_t>;
TYPED_TEST_SUITE(ScanLibCudaTest, TestTypes);

// =============================================================================
// nvcc template verification
// =============================================================================

TEST(ScanLibCudaNvcc, TempBytesInstantiations) {
    // Verify all inclusive_scan_cuda_temp_bytes instantiations compile under nvcc
    static_assert(std::is_same_v<decltype(inclusive_scan_cuda_temp_bytes<float>(100)), size_t>);
    static_assert(std::is_same_v<decltype(inclusive_scan_cuda_temp_bytes<double>(100)), size_t>);
    static_assert(std::is_same_v<decltype(inclusive_scan_cuda_temp_bytes<int32_t>(100)), size_t>);
    static_assert(std::is_same_v<decltype(inclusive_scan_cuda_temp_bytes<int64_t>(100)), size_t>);
    SUCCEED();
}

TEST(ScanLibCudaNvcc, ScanInstantiations) {
    // Verify all inclusive_scan_cuda instantiations compile under nvcc
    float *f_in       = nullptr;
    float *f_out      = nullptr;
    void *temp        = nullptr;
    size_t temp_bytes = 0;

    // Should compile
    inclusive_scan_cuda(f_in, f_out, 100, temp, temp_bytes, nullptr);
    SUCCEED();
}

// =============================================================================
// Temp storage query
// =============================================================================

TYPED_TEST(ScanLibCudaTest, TempBytes_NonZero) {
    using T         = TypeParam;
    int64_t const n = 1000;

    size_t temp_bytes = inclusive_scan_cuda_temp_bytes<T>(n);
    EXPECT_GT(temp_bytes, 0u);
}

TYPED_TEST(ScanLibCudaTest, TempBytes_Zero) {
    using T           = TypeParam;
    size_t temp_bytes = inclusive_scan_cuda_temp_bytes<T>(0);
    // May be 0 or non-zero, both are valid
    SUCCEED();
}

// =============================================================================
// CUDA scan
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
    int64_t const n = 1000000; // 1M elements for GPU

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

    dispatch::test::DeviceBuffer<T> d_in(1);
    dispatch::test::DeviceBuffer<T> d_out(1);

    size_t temp_bytes = inclusive_scan_cuda_temp_bytes<T>(0);
    DeviceRawBuffer temp(temp_bytes);

    // Should not crash
    inclusive_scan_cuda(d_in.get(), d_out.get(), 0, temp.get(), temp.size(), nullptr);
    cudaDeviceSynchronize();

    SUCCEED();
}

// =============================================================================
// Stream support
// =============================================================================

TYPED_TEST(ScanLibCudaTest, ExplicitStream) {
    using T         = TypeParam;
    int64_t const n = 1000;

    std::vector<T> const input = make_input<T>(n);
    std::vector<T> output(n);

    dispatch::test::DeviceBuffer<T> d_in(n);
    dispatch::test::DeviceBuffer<T> d_out(n);
    dispatch::test::copy_to_device(input.data(), d_in.get(), n);

    size_t temp_bytes = inclusive_scan_cuda_temp_bytes<T>(n);
    DeviceRawBuffer temp(temp_bytes);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    inclusive_scan_cuda(d_in.get(), d_out.get(), n, temp.get(), temp.size(), stream);
    cudaStreamSynchronize(stream);

    dispatch::test::copy_to_host(d_out.get(), output.data(), n);

    expect_scan_equal(output, expected_scan(input));

    cudaStreamDestroy(stream);
}

TYPED_TEST(ScanLibCudaTest, DefaultStream) {
    using T         = TypeParam;
    int64_t const n = 1000;

    std::vector<T> const input = make_input<T>(n);
    std::vector<T> output(n);

    dispatch::test::DeviceBuffer<T> d_in(n);
    dispatch::test::DeviceBuffer<T> d_out(n);
    dispatch::test::copy_to_device(input.data(), d_in.get(), n);

    size_t temp_bytes = inclusive_scan_cuda_temp_bytes<T>(n);
    DeviceRawBuffer temp(temp_bytes);

    // nullptr for default stream
    inclusive_scan_cuda(d_in.get(), d_out.get(), n, temp.get(), temp.size(), nullptr);
    cudaDeviceSynchronize();

    dispatch::test::copy_to_host(d_out.get(), output.data(), n);

    expect_scan_equal(output, expected_scan(input));
}

// =============================================================================
// Determinism tests
// =============================================================================

TYPED_TEST(ScanLibCudaTest, IntegerDeterministic) {
    using T = TypeParam;
    if constexpr (std::is_integral_v<T>) {
        int64_t const n            = 10000;
        std::vector<T> const input = make_input<T>(n);
        std::vector<T> output1(n);
        std::vector<T> output2(n);

        dispatch::test::DeviceBuffer<T> d_in(n);
        dispatch::test::DeviceBuffer<T> d_out1(n);
        dispatch::test::DeviceBuffer<T> d_out2(n);
        dispatch::test::copy_to_device(input.data(), d_in.get(), n);

        size_t temp_bytes = inclusive_scan_cuda_temp_bytes<T>(n);
        DeviceRawBuffer temp1(temp_bytes);
        DeviceRawBuffer temp2(temp_bytes);

        inclusive_scan_cuda(d_in.get(), d_out1.get(), n, temp1.get(), temp1.size(), nullptr);
        cudaDeviceSynchronize();
        dispatch::test::copy_to_host(d_out1.get(), output1.data(), n);

        dispatch::test::copy_to_device(input.data(), d_in.get(), n);
        inclusive_scan_cuda(d_in.get(), d_out2.get(), n, temp2.get(), temp2.size(), nullptr);
        cudaDeviceSynchronize();
        dispatch::test::copy_to_host(d_out2.get(), output2.data(), n);

        EXPECT_EQ(output1, output2); // Integers should be deterministic
    } else {
        GTEST_SKIP() << "Float types are non-deterministic in CUDA";
    }
}

} // namespace
} // namespace scan_lib
