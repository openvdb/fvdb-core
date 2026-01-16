// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <fvdb/detail/dispatch/example/Common.h>
#include <fvdb/detail/dispatch/example/Functional.h>

#include <torch/torch.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

namespace fvdb {
namespace dispatch {
namespace example {
namespace {

// =============================================================================
// Test utilities
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

// =============================================================================
// CPU Dispatch Tests - Verify correct dispatch path via notes
// =============================================================================

// -----------------------------------------------------------------------------
// Float types on CPU
// -----------------------------------------------------------------------------

TEST(FunctionalDispatchCPU, Float_Contiguous_OutOfPlace_Deterministic_UsesSerial) {
    int64_t const n = 100;
    auto tensor     = make_arange_tensor(n, torch::kFloat);

    auto result =
        inclusiveScanFunctional(tensor, Placement::OutOfPlace, Determinism::Deterministic);

    EXPECT_EQ(result.notes, "cpu_serial_out_of_place");
    expect_scan_equal<float>(result.tensor, expected_arange_scan<float>(n));
}

TEST(FunctionalDispatchCPU, Float_Contiguous_OutOfPlace_NonDeterministic_UsesParallel) {
    int64_t const n = 100;
    auto tensor     = make_arange_tensor(n, torch::kFloat);

    auto result =
        inclusiveScanFunctional(tensor, Placement::OutOfPlace, Determinism::NonDeterministic);

    EXPECT_EQ(result.notes, "cpu_parallel_float_nondeterministic");
    expect_scan_equal<float>(result.tensor, expected_arange_scan<float>(n));
}

TEST(FunctionalDispatchCPU, Float_Contiguous_InPlace_Deterministic_UsesSerialInPlace) {
    int64_t const n = 100;
    auto tensor     = make_arange_tensor(n, torch::kFloat);

    auto result = inclusiveScanFunctional(tensor, Placement::InPlace, Determinism::Deterministic);

    EXPECT_EQ(result.notes, "cpu_serial_in_place");
    // For in-place, result.tensor should be same as input tensor
    EXPECT_TRUE(result.tensor.data_ptr() == tensor.data_ptr());
    expect_scan_equal<float>(result.tensor, expected_arange_scan<float>(n));
}

TEST(FunctionalDispatchCPU, Float_Contiguous_InPlace_NonDeterministic_UsesSerialInPlace) {
    int64_t const n = 100;
    auto tensor     = make_arange_tensor(n, torch::kFloat);

    auto result =
        inclusiveScanFunctional(tensor, Placement::InPlace, Determinism::NonDeterministic);

    EXPECT_EQ(result.notes, "cpu_serial_in_place");
    EXPECT_TRUE(result.tensor.data_ptr() == tensor.data_ptr());
    expect_scan_equal<float>(result.tensor, expected_arange_scan<float>(n));
}

TEST(FunctionalDispatchCPU, Float_Strided_OutOfPlace_NonDeterministic_UsesParallel) {
    int64_t const n = 100;
    auto tensor     = make_strided_arange_tensor(n, torch::kFloat);

    ASSERT_FALSE(tensor.is_contiguous());

    auto result =
        inclusiveScanFunctional(tensor, Placement::OutOfPlace, Determinism::NonDeterministic);

    EXPECT_EQ(result.notes, "cpu_parallel_float_nondeterministic");
    expect_scan_equal<float>(result.tensor, expected_arange_scan<float>(n));
}

TEST(FunctionalDispatchCPU, Float_Strided_InPlace_UsesSerialInPlace) {
    int64_t const n = 50;
    auto tensor     = make_strided_arange_tensor(n, torch::kFloat);

    ASSERT_FALSE(tensor.is_contiguous());

    auto result = inclusiveScanFunctional(tensor, Placement::InPlace, Determinism::Deterministic);

    EXPECT_EQ(result.notes, "cpu_serial_in_place");
    expect_scan_equal<float>(result.tensor, expected_arange_scan<float>(n));
}

// -----------------------------------------------------------------------------
// Double type on CPU
// -----------------------------------------------------------------------------

TEST(FunctionalDispatchCPU, Double_Contiguous_OutOfPlace_Deterministic_UsesSerial) {
    int64_t const n = 100;
    auto tensor     = make_arange_tensor(n, torch::kDouble);

    auto result =
        inclusiveScanFunctional(tensor, Placement::OutOfPlace, Determinism::Deterministic);

    EXPECT_EQ(result.notes, "cpu_serial_out_of_place");
    expect_scan_equal<double>(result.tensor, expected_arange_scan<double>(n));
}

TEST(FunctionalDispatchCPU, Double_Contiguous_OutOfPlace_NonDeterministic_UsesParallel) {
    int64_t const n = 100;
    auto tensor     = make_arange_tensor(n, torch::kDouble);

    auto result =
        inclusiveScanFunctional(tensor, Placement::OutOfPlace, Determinism::NonDeterministic);

    EXPECT_EQ(result.notes, "cpu_parallel_float_nondeterministic");
    expect_scan_equal<double>(result.tensor, expected_arange_scan<double>(n));
}

// -----------------------------------------------------------------------------
// Integer types on CPU - determinism is promoted to Deterministic
// -----------------------------------------------------------------------------

TEST(FunctionalDispatchCPU, Int_Contiguous_OutOfPlace_Deterministic_UsesParallel) {
    int64_t const n = 100;
    auto tensor     = make_arange_tensor(n, torch::kInt);

    auto result =
        inclusiveScanFunctional(tensor, Placement::OutOfPlace, Determinism::Deterministic);

    EXPECT_EQ(result.notes, "cpu_parallel_int_deterministic");
    expect_scan_equal<int32_t>(result.tensor, expected_arange_scan<int32_t>(n));
}

TEST(FunctionalDispatchCPU, Int_Contiguous_OutOfPlace_NonDeterministic_PromotedToDeterministic) {
    // Integers are promoted to deterministic, so NonDeterministic should still use parallel
    int64_t const n = 100;
    auto tensor     = make_arange_tensor(n, torch::kInt);

    auto result =
        inclusiveScanFunctional(tensor, Placement::OutOfPlace, Determinism::NonDeterministic);

    // Even though NonDeterministic was requested, integers get promoted to Deterministic
    EXPECT_EQ(result.notes, "cpu_parallel_int_deterministic");
    expect_scan_equal<int32_t>(result.tensor, expected_arange_scan<int32_t>(n));
}

TEST(FunctionalDispatchCPU, Int_Contiguous_InPlace_UsesSerialInPlace) {
    int64_t const n = 100;
    auto tensor     = make_arange_tensor(n, torch::kInt);

    auto result = inclusiveScanFunctional(tensor, Placement::InPlace, Determinism::Deterministic);

    EXPECT_EQ(result.notes, "cpu_serial_in_place");
    EXPECT_TRUE(result.tensor.data_ptr() == tensor.data_ptr());
    expect_scan_equal<int32_t>(result.tensor, expected_arange_scan<int32_t>(n));
}

TEST(FunctionalDispatchCPU, Int_Strided_OutOfPlace_UsesParallel) {
    int64_t const n = 100;
    auto tensor     = make_strided_arange_tensor(n, torch::kInt);

    ASSERT_FALSE(tensor.is_contiguous());

    auto result =
        inclusiveScanFunctional(tensor, Placement::OutOfPlace, Determinism::Deterministic);

    EXPECT_EQ(result.notes, "cpu_parallel_int_deterministic");
    expect_scan_equal<int32_t>(result.tensor, expected_arange_scan<int32_t>(n));
}

// -----------------------------------------------------------------------------
// Long (int64_t) type on CPU
// -----------------------------------------------------------------------------

TEST(FunctionalDispatchCPU, Long_Contiguous_OutOfPlace_UsesParallel) {
    int64_t const n = 100;
    auto tensor     = make_arange_tensor(n, torch::kLong);

    auto result =
        inclusiveScanFunctional(tensor, Placement::OutOfPlace, Determinism::Deterministic);

    EXPECT_EQ(result.notes, "cpu_parallel_int_deterministic");
    expect_scan_equal<int64_t>(result.tensor, expected_arange_scan<int64_t>(n));
}

TEST(FunctionalDispatchCPU, Long_Strided_InPlace_UsesSerialInPlace) {
    int64_t const n = 50;
    auto tensor     = make_strided_arange_tensor(n, torch::kLong);

    ASSERT_FALSE(tensor.is_contiguous());

    auto result = inclusiveScanFunctional(tensor, Placement::InPlace, Determinism::Deterministic);

    EXPECT_EQ(result.notes, "cpu_serial_in_place");
    expect_scan_equal<int64_t>(result.tensor, expected_arange_scan<int64_t>(n));
}

// =============================================================================
// Input Validation Tests
// =============================================================================

TEST(FunctionalValidation, Rank0_Throws) {
    auto tensor = torch::tensor(42.0f); // scalar (0D)

    EXPECT_THROW(inclusiveScanFunctional(tensor, Placement::OutOfPlace, Determinism::Deterministic),
                 c10::Error);
}

TEST(FunctionalValidation, Rank2_Throws) {
    auto tensor = torch::empty({10, 10}, torch::kFloat);

    EXPECT_THROW(inclusiveScanFunctional(tensor, Placement::OutOfPlace, Determinism::Deterministic),
                 c10::Error);
}

TEST(FunctionalValidation, Rank3_Throws) {
    auto tensor = torch::empty({2, 3, 4}, torch::kFloat);

    EXPECT_THROW(inclusiveScanFunctional(tensor, Placement::OutOfPlace, Determinism::Deterministic),
                 c10::Error);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST(FunctionalDispatchCPU, EmptyTensor_OutOfPlace) {
    auto tensor = torch::empty({0}, torch::kFloat);

    auto result =
        inclusiveScanFunctional(tensor, Placement::OutOfPlace, Determinism::Deterministic);

    EXPECT_EQ(result.notes, "empty_out_of_place");
    EXPECT_EQ(result.tensor.size(0), 0);
    // Out of place should return a different tensor (use is_same, not data_ptr which is NULL)
    EXPECT_FALSE(result.tensor.is_same(tensor));
}

TEST(FunctionalDispatchCPU, EmptyTensor_InPlace) {
    auto tensor = torch::empty({0}, torch::kFloat);

    auto result = inclusiveScanFunctional(tensor, Placement::InPlace, Determinism::Deterministic);

    EXPECT_EQ(result.notes, "empty_in_place");
    EXPECT_EQ(result.tensor.size(0), 0);
    // In place should return the same tensor
    EXPECT_TRUE(result.tensor.is_same(tensor));
}

TEST(FunctionalDispatchCPU, SingleElement) {
    auto tensor = torch::tensor({42.0f});

    auto result =
        inclusiveScanFunctional(tensor, Placement::OutOfPlace, Determinism::Deterministic);

    EXPECT_EQ(result.tensor.size(0), 1);
    EXPECT_FLOAT_EQ(result.tensor[0].item<float>(), 42.0f);
}

TEST(FunctionalDispatchCPU, SingleElementInPlace) {
    auto tensor = torch::tensor({42.0f});

    auto result = inclusiveScanFunctional(tensor, Placement::InPlace, Determinism::Deterministic);

    EXPECT_EQ(result.notes, "cpu_serial_in_place");
    EXPECT_EQ(result.tensor.size(0), 1);
    EXPECT_FLOAT_EQ(result.tensor[0].item<float>(), 42.0f);
}

TEST(FunctionalDispatchCPU, LargeInput) {
    int64_t const n = 100000;
    auto tensor     = make_arange_tensor(n, torch::kFloat);

    auto result =
        inclusiveScanFunctional(tensor, Placement::OutOfPlace, Determinism::NonDeterministic);

    EXPECT_EQ(result.notes, "cpu_parallel_float_nondeterministic");
    expect_scan_equal<float>(result.tensor, expected_arange_scan<float>(n));
}

// =============================================================================
// Correctness Tests - Verify scan output is mathematically correct
// =============================================================================

TEST(FunctionalCorrectness, ScanSumIsCorrect_Float) {
    int64_t const n = 5;
    auto tensor     = make_arange_tensor(n, torch::kFloat); // [1, 2, 3, 4, 5]

    auto result =
        inclusiveScanFunctional(tensor, Placement::OutOfPlace, Determinism::Deterministic);

    std::vector<float> expected = {1.0f, 3.0f, 6.0f, 10.0f, 15.0f};
    expect_scan_equal<float>(result.tensor, expected);
}

TEST(FunctionalCorrectness, ScanSumIsCorrect_Int) {
    int64_t const n = 5;
    auto tensor     = make_arange_tensor(n, torch::kInt); // [1, 2, 3, 4, 5]

    auto result =
        inclusiveScanFunctional(tensor, Placement::OutOfPlace, Determinism::Deterministic);

    std::vector<int32_t> expected = {1, 3, 6, 10, 15};
    expect_scan_equal<int32_t>(result.tensor, expected);
}

TEST(FunctionalCorrectness, InPlaceModifiesInput) {
    int64_t const n   = 5;
    auto tensor       = make_arange_tensor(n, torch::kFloat); // [1, 2, 3, 4, 5]
    auto original_ptr = tensor.data_ptr<float>();

    auto result = inclusiveScanFunctional(tensor, Placement::InPlace, Determinism::Deterministic);

    // Should be same memory
    EXPECT_EQ(result.tensor.data_ptr<float>(), original_ptr);
    // Original tensor should be modified
    std::vector<float> expected = {1.0f, 3.0f, 6.0f, 10.0f, 15.0f};
    expect_scan_equal<float>(tensor, expected);
}

TEST(FunctionalCorrectness, OutOfPlaceDoesNotModifyInput) {
    int64_t const n = 5;
    auto tensor     = make_arange_tensor(n, torch::kFloat); // [1, 2, 3, 4, 5]
    auto original   = tensor.clone();                       // Save original values

    auto result =
        inclusiveScanFunctional(tensor, Placement::OutOfPlace, Determinism::Deterministic);

    // Input should be unchanged
    EXPECT_TRUE(torch::equal(tensor, original));
    // Result should be different memory
    EXPECT_NE(result.tensor.data_ptr<float>(), tensor.data_ptr<float>());
}

// =============================================================================
// CUDA Dispatch Tests
// =============================================================================

TEST(FunctionalDispatchCUDA, Float_Contiguous_OutOfPlace_NonDeterministic_UsesCub) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    int64_t const n = 1000;
    auto tensor     = make_arange_tensor(n, torch::kFloat, torch::kCUDA);

    auto result =
        inclusiveScanFunctional(tensor, Placement::OutOfPlace, Determinism::NonDeterministic);

    EXPECT_EQ(result.notes, "cuda_cub");
    expect_scan_equal<float>(result.tensor, expected_arange_scan<float>(n));
}

TEST(FunctionalDispatchCUDA, Double_Contiguous_OutOfPlace_NonDeterministic_UsesCub) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    int64_t const n = 1000;
    auto tensor     = make_arange_tensor(n, torch::kDouble, torch::kCUDA);

    auto result =
        inclusiveScanFunctional(tensor, Placement::OutOfPlace, Determinism::NonDeterministic);

    EXPECT_EQ(result.notes, "cuda_cub");
    expect_scan_equal<double>(result.tensor, expected_arange_scan<double>(n));
}

TEST(FunctionalDispatchCUDA, Int_Contiguous_OutOfPlace_Deterministic_UsesCub) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    int64_t const n = 1000;
    auto tensor     = make_arange_tensor(n, torch::kInt, torch::kCUDA);

    auto result =
        inclusiveScanFunctional(tensor, Placement::OutOfPlace, Determinism::Deterministic);

    EXPECT_EQ(result.notes, "cuda_cub");
    expect_scan_equal<int32_t>(result.tensor, expected_arange_scan<int32_t>(n));
}

TEST(FunctionalDispatchCUDA, Long_Contiguous_OutOfPlace_UsesCub) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    int64_t const n = 1000;
    auto tensor     = make_arange_tensor(n, torch::kLong, torch::kCUDA);

    // Integers are promoted to Deterministic
    auto result =
        inclusiveScanFunctional(tensor, Placement::OutOfPlace, Determinism::NonDeterministic);

    EXPECT_EQ(result.notes, "cuda_cub");
    expect_scan_equal<int64_t>(result.tensor, expected_arange_scan<int64_t>(n));
}

TEST(FunctionalDispatchCUDA, Float_Contiguous_OutOfPlace_Deterministic_ThrowsUnsupported) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto tensor = make_arange_tensor(100, torch::kFloat, torch::kCUDA);

    // Float + Deterministic is not supported on CUDA (floats are non-deterministic)
    EXPECT_THROW(inclusiveScanFunctional(tensor, Placement::OutOfPlace, Determinism::Deterministic),
                 c10::Error);
}

TEST(FunctionalDispatchCUDA, Float_Contiguous_InPlace_ThrowsUnsupported) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto tensor = make_arange_tensor(100, torch::kFloat, torch::kCUDA);

    // InPlace is not supported on CUDA
    EXPECT_THROW(inclusiveScanFunctional(tensor, Placement::InPlace, Determinism::NonDeterministic),
                 c10::Error);
}

TEST(FunctionalDispatchCUDA, Float_Strided_ThrowsUnsupported) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    // Create strided tensor on CUDA using our utility
    auto tensor = make_strided_arange_tensor(100, torch::kFloat, torch::kCUDA);

    ASSERT_FALSE(tensor.is_contiguous());

    // Strided is not supported on CUDA
    EXPECT_THROW(
        inclusiveScanFunctional(tensor, Placement::OutOfPlace, Determinism::NonDeterministic),
        c10::Error);
}

TEST(FunctionalDispatchCUDA, LargeInput) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    int64_t const n = 100000;
    auto tensor     = make_arange_tensor(n, torch::kFloat, torch::kCUDA);

    auto result =
        inclusiveScanFunctional(tensor, Placement::OutOfPlace, Determinism::NonDeterministic);

    EXPECT_EQ(result.notes, "cuda_cub");
    expect_scan_equal<float>(result.tensor, expected_arange_scan<float>(n));
}

TEST(FunctionalDispatchCUDA, SingleElement) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto tensor = make_arange_tensor(1, torch::kFloat, torch::kCUDA); // [1.0]

    auto result =
        inclusiveScanFunctional(tensor, Placement::OutOfPlace, Determinism::NonDeterministic);

    EXPECT_EQ(result.notes, "cuda_cub");
    EXPECT_EQ(result.tensor.size(0), 1);
    EXPECT_FLOAT_EQ(result.tensor.cpu()[0].item<float>(), 1.0f);
}

} // namespace
} // namespace example
} // namespace dispatch
} // namespace fvdb
