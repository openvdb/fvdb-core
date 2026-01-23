// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef TESTS_DISPATCH_TEST_UTILS_H
#define TESTS_DISPATCH_TEST_UTILS_H

#include <torch/torch.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

namespace dispatch {
namespace test {

// =============================================================================
// Tensor creation utilities
// =============================================================================

/// Create a contiguous 1D tensor with values [1, 2, 3, ..., n] using torch::arange
/// Works directly on CPU or CUDA without manual memory copies
inline torch::Tensor
make_arange_tensor(int64_t n, torch::ScalarType dtype, torch::Device device = torch::kCPU) {
    auto options = torch::TensorOptions().dtype(dtype).device(device);
    // arange is [0, n), add 1 to get [1, n]
    return torch::arange(1, n + 1, options);
}

/// Create a strided (non-contiguous) 1D tensor with values [1, 2, 3, ..., n]
/// Uses slicing to create stride=2, works on CPU or CUDA
inline torch::Tensor
make_strided_arange_tensor(int64_t n, torch::ScalarType dtype, torch::Device device = torch::kCPU) {
    auto options = torch::TensorOptions().dtype(dtype).device(device);
    // Create [1, 0, 2, 0, 3, 0, ...] then slice every other element
    auto full = torch::zeros({n * 2}, options);
    // Use scatter or slice assignment: full[::2] = arange(1, n+1)
    full.slice(0, 0, n * 2, 2).copy_(torch::arange(1, n + 1, options));
    return full.slice(0, 0, n * 2, 2); // stride of 2
}

// =============================================================================
// Expected result computation
// =============================================================================

/// Compute expected inclusive scan for [1, 2, 3, ..., n]: result is [1, 3, 6, 10, ...]
/// This is the triangular number sequence: T_i = i*(i+1)/2
template <typename T>
std::vector<T>
expected_arange_scan(int64_t n) {
    std::vector<T> result(n);
    for (int64_t i = 0; i < n; ++i) {
        // Inclusive scan of [1, 2, ..., i+1] = (i+1)*(i+2)/2
        result[i] = static_cast<T>((i + 1) * (i + 2) / 2);
    }
    return result;
}

// =============================================================================
// Test assertion utilities
// =============================================================================

/// Compare scan results with appropriate tolerance for floating-point types
template <typename T>
void
expect_scan_equal(torch::Tensor actual, std::vector<T> const &expected) {
    ASSERT_EQ(actual.size(0), static_cast<int64_t>(expected.size()));
    auto actual_cpu = actual.cpu().contiguous();
    for (size_t i = 0; i < expected.size(); ++i) {
        T actual_val = actual_cpu.template data_ptr<T>()[i];
        if constexpr (std::is_floating_point_v<T>) {
            T const rel_tol = T{1e-4};
            T const abs_tol = T{1e-6};
            T const tol     = std::abs(expected[i]) * rel_tol + abs_tol;
            EXPECT_NEAR(actual_val, expected[i], tol) << "at index " << i;
        } else {
            EXPECT_EQ(actual_val, expected[i]) << "at index " << i;
        }
    }
}

/// Compare ReLU results with appropriate tolerance
void
expect_relu_equal(torch::Tensor actual, torch::Tensor expected, double tol = 1e-5) {
    ASSERT_EQ(actual.sizes(), expected.sizes());
    ASSERT_EQ(actual.dtype(), expected.dtype());
    auto actual_cpu   = actual.cpu().contiguous();
    auto expected_cpu = expected.cpu().contiguous();

    if (actual.dtype() == torch::kFloat16 || actual.dtype() == torch::kBFloat16) {
        // Half precision needs larger tolerance
        tol = 1e-2;
    }

    auto actual_ptr   = actual_cpu.data_ptr<float>();
    auto expected_ptr = expected_cpu.data_ptr<float>();
    for (int64_t i = 0; i < actual_cpu.numel(); ++i) {
        EXPECT_NEAR(actual_ptr[i], expected_ptr[i], tol) << "at index " << i;
    }
}

// =============================================================================
// CUDA availability check
// =============================================================================

inline void
skip_if_no_cuda() {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }
}

} // namespace test
} // namespace dispatch

#endif // TESTS_DISPATCH_TEST_UTILS_H
