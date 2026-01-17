// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <fvdb/detail/dispatch/example/ScanLib.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <numeric>
#include <type_traits>
#include <vector>

namespace scanlib {
namespace {

// =============================================================================
// CUDA utilities
// =============================================================================

/// RAII wrapper for device memory
template <typename T> class DeviceBuffer {
    T *ptr_   = nullptr;
    size_t n_ = 0;

  public:
    explicit DeviceBuffer(size_t n) : n_(n) {
        if (n > 0) {
            cudaMalloc(&ptr_, n * sizeof(T));
        }
    }
    ~DeviceBuffer() {
        if (ptr_)
            cudaFree(ptr_);
    }
    DeviceBuffer(DeviceBuffer const &)            = delete;
    DeviceBuffer &operator=(DeviceBuffer const &) = delete;

    T *
    get() {
        return ptr_;
    }
    T const *
    get() const {
        return ptr_;
    }
    size_t
    size() const {
        return n_;
    }

    void
    copyFromHost(T const *host_data) {
        cudaMemcpy(ptr_, host_data, n_ * sizeof(T), cudaMemcpyHostToDevice);
    }
    void
    copyToHost(T *host_data) const {
        cudaMemcpy(host_data, ptr_, n_ * sizeof(T), cudaMemcpyDeviceToHost);
    }
};

/// RAII wrapper for raw device memory (for temp storage)
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

    DeviceBuffer<T> d_in(n);
    DeviceBuffer<T> d_out(n);
    d_in.copyFromHost(input.data());

    size_t temp_bytes = inclusive_scan_cuda_temp_bytes<T>(n);
    DeviceRawBuffer temp(temp_bytes);

    inclusive_scan_cuda(d_in.get(), d_out.get(), n, temp.get(), temp.size(), nullptr);
    cudaDeviceSynchronize();

    d_out.copyToHost(output.data());

    expect_scan_equal(output, expected_scan(input));
}

TYPED_TEST(ScanLibCudaTest, Large) {
    using T         = TypeParam;
    int64_t const n = 100000;

    std::vector<T> const input = make_input<T>(n);
    std::vector<T> output(n);

    DeviceBuffer<T> d_in(n);
    DeviceBuffer<T> d_out(n);
    d_in.copyFromHost(input.data());

    size_t temp_bytes = inclusive_scan_cuda_temp_bytes<T>(n);
    DeviceRawBuffer temp(temp_bytes);

    inclusive_scan_cuda(d_in.get(), d_out.get(), n, temp.get(), temp.size(), nullptr);
    cudaDeviceSynchronize();

    d_out.copyToHost(output.data());

    expect_scan_equal(output, expected_scan(input));
}

TYPED_TEST(ScanLibCudaTest, SingleElement) {
    using T               = TypeParam;
    std::vector<T> input  = {T{42}};
    std::vector<T> output = {T{0}};

    DeviceBuffer<T> d_in(1);
    DeviceBuffer<T> d_out(1);
    d_in.copyFromHost(input.data());

    size_t temp_bytes = inclusive_scan_cuda_temp_bytes<T>(1);
    DeviceRawBuffer temp(temp_bytes);

    inclusive_scan_cuda(d_in.get(), d_out.get(), 1, temp.get(), temp.size(), nullptr);
    cudaDeviceSynchronize();

    d_out.copyToHost(output.data());

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

    DeviceBuffer<T> d_in(n);
    DeviceBuffer<T> d_out(n);
    d_in.copyFromHost(input.data());

    size_t temp_bytes = inclusive_scan_cuda_temp_bytes<T>(n);
    DeviceRawBuffer temp(temp_bytes);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    inclusive_scan_cuda(d_in.get(), d_out.get(), n, temp.get(), temp.size(), stream);
    cudaStreamSynchronize(stream);

    d_out.copyToHost(output.data());
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
} // namespace scanlib
