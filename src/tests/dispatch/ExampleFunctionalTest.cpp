// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <fvdb/detail/dispatch/example/Common.h>
#include <fvdb/detail/dispatch/example/Functional.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace fvdb {
namespace dispatch {
namespace example {
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

/// Create a contiguous tensor from a vector
template <typename T>
torch::Tensor
make_tensor(std::vector<T> const &data, torch::Device device = torch::kCPU) {
    auto options =
        torch::TensorOptions().dtype(torch::CppTypeToScalarType<T>::value).device(device);
    auto tensor = torch::empty({static_cast<int64_t>(data.size())}, options);
    std::memcpy(tensor.data_ptr<T>(), data.data(), data.size() * sizeof(T));
    return tensor;
}

/// Create a strided (non-contiguous) tensor - every other element
template <typename T>
torch::Tensor
make_strided_tensor(std::vector<T> const &data, torch::Device device = torch::kCPU) {
    auto options =
        torch::TensorOptions().dtype(torch::CppTypeToScalarType<T>::value).device(device);
    // Create tensor with twice the elements, then slice to get stride=2
    auto full = torch::empty({static_cast<int64_t>(data.size() * 2)}, options);
    for (int64_t i = 0; i < static_cast<int64_t>(data.size()); ++i) {
        full[i * 2] = data[i];
    }
    return full.slice(0, 0, data.size() * 2, 2); // stride of 2
}

/// Compare scan results with appropriate tolerance for floating-point types
template <typename T>
void
expect_scan_equal(torch::Tensor actual, std::vector<T> const &expected) {
    ASSERT_EQ(actual.size(0), static_cast<int64_t>(expected.size()));
    auto actual_cpu = actual.cpu().contiguous();
    for (size_t i = 0; i < expected.size(); ++i) {
        T actual_val = actual_cpu.data_ptr<T>()[i];
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
    auto input  = make_input<float>(100);
    auto tensor = make_tensor(input);

    auto result =
        inclusiveScanFunctional(tensor, Placement::OutOfPlace, Determinism::Deterministic);

    EXPECT_EQ(result.notes, "cpu_serial_out_of_place");
    expect_scan_equal<float>(result.tensor, expected_scan(input));
}

TEST(FunctionalDispatchCPU, Float_Contiguous_OutOfPlace_NonDeterministic_UsesParallel) {
    auto input  = make_input<float>(100);
    auto tensor = make_tensor(input);

    auto result =
        inclusiveScanFunctional(tensor, Placement::OutOfPlace, Determinism::NonDeterministic);

    EXPECT_EQ(result.notes, "cpu_parallel_float_nondeterministic");
    expect_scan_equal<float>(result.tensor, expected_scan(input));
}

TEST(FunctionalDispatchCPU, Float_Contiguous_InPlace_Deterministic_UsesSerialInPlace) {
    auto input  = make_input<float>(100);
    auto tensor = make_tensor(input);

    auto result = inclusiveScanFunctional(tensor, Placement::InPlace, Determinism::Deterministic);

    EXPECT_EQ(result.notes, "cpu_serial_in_place");
    // For in-place, result.tensor should be same as input tensor
    EXPECT_TRUE(result.tensor.data_ptr() == tensor.data_ptr());
    expect_scan_equal<float>(result.tensor, expected_scan(input));
}

TEST(FunctionalDispatchCPU, Float_Contiguous_InPlace_NonDeterministic_UsesSerialInPlace) {
    auto input  = make_input<float>(100);
    auto tensor = make_tensor(input);

    auto result =
        inclusiveScanFunctional(tensor, Placement::InPlace, Determinism::NonDeterministic);

    EXPECT_EQ(result.notes, "cpu_serial_in_place");
    EXPECT_TRUE(result.tensor.data_ptr() == tensor.data_ptr());
    expect_scan_equal<float>(result.tensor, expected_scan(input));
}

TEST(FunctionalDispatchCPU, Float_Strided_OutOfPlace_NonDeterministic_UsesParallel) {
    auto input  = make_input<float>(100);
    auto tensor = make_strided_tensor(input);

    ASSERT_FALSE(tensor.is_contiguous());

    auto result =
        inclusiveScanFunctional(tensor, Placement::OutOfPlace, Determinism::NonDeterministic);

    EXPECT_EQ(result.notes, "cpu_parallel_float_nondeterministic");
    expect_scan_equal<float>(result.tensor, expected_scan(input));
}

TEST(FunctionalDispatchCPU, Float_Strided_InPlace_UsesSerialInPlace) {
    auto input  = make_input<float>(50);
    auto tensor = make_strided_tensor(input);

    ASSERT_FALSE(tensor.is_contiguous());

    auto result = inclusiveScanFunctional(tensor, Placement::InPlace, Determinism::Deterministic);

    EXPECT_EQ(result.notes, "cpu_serial_in_place");
    expect_scan_equal<float>(result.tensor, expected_scan(input));
}

// -----------------------------------------------------------------------------
// Double type on CPU
// -----------------------------------------------------------------------------

TEST(FunctionalDispatchCPU, Double_Contiguous_OutOfPlace_Deterministic_UsesSerial) {
    auto input  = make_input<double>(100);
    auto tensor = make_tensor(input);

    auto result =
        inclusiveScanFunctional(tensor, Placement::OutOfPlace, Determinism::Deterministic);

    EXPECT_EQ(result.notes, "cpu_serial_out_of_place");
    expect_scan_equal<double>(result.tensor, expected_scan(input));
}

TEST(FunctionalDispatchCPU, Double_Contiguous_OutOfPlace_NonDeterministic_UsesParallel) {
    auto input  = make_input<double>(100);
    auto tensor = make_tensor(input);

    auto result =
        inclusiveScanFunctional(tensor, Placement::OutOfPlace, Determinism::NonDeterministic);

    EXPECT_EQ(result.notes, "cpu_parallel_float_nondeterministic");
    expect_scan_equal<double>(result.tensor, expected_scan(input));
}

// -----------------------------------------------------------------------------
// Integer types on CPU - determinism is promoted to Deterministic
// -----------------------------------------------------------------------------

TEST(FunctionalDispatchCPU, Int_Contiguous_OutOfPlace_Deterministic_UsesParallel) {
    auto input  = make_input<int32_t>(100);
    auto tensor = make_tensor(input);

    auto result =
        inclusiveScanFunctional(tensor, Placement::OutOfPlace, Determinism::Deterministic);

    EXPECT_EQ(result.notes, "cpu_parallel_int_deterministic");
    expect_scan_equal<int32_t>(result.tensor, expected_scan(input));
}

TEST(FunctionalDispatchCPU, Int_Contiguous_OutOfPlace_NonDeterministic_PromotedToDeterministic) {
    // Integers are promoted to deterministic, so NonDeterministic should still use parallel
    auto input  = make_input<int32_t>(100);
    auto tensor = make_tensor(input);

    auto result =
        inclusiveScanFunctional(tensor, Placement::OutOfPlace, Determinism::NonDeterministic);

    // Even though NonDeterministic was requested, integers get promoted to Deterministic
    EXPECT_EQ(result.notes, "cpu_parallel_int_deterministic");
    expect_scan_equal<int32_t>(result.tensor, expected_scan(input));
}

TEST(FunctionalDispatchCPU, Int_Contiguous_InPlace_UsesSerialInPlace) {
    auto input  = make_input<int32_t>(100);
    auto tensor = make_tensor(input);

    auto result = inclusiveScanFunctional(tensor, Placement::InPlace, Determinism::Deterministic);

    EXPECT_EQ(result.notes, "cpu_serial_in_place");
    EXPECT_TRUE(result.tensor.data_ptr() == tensor.data_ptr());
    expect_scan_equal<int32_t>(result.tensor, expected_scan(input));
}

TEST(FunctionalDispatchCPU, Int_Strided_OutOfPlace_UsesParallel) {
    auto input  = make_input<int32_t>(100);
    auto tensor = make_strided_tensor(input);

    ASSERT_FALSE(tensor.is_contiguous());

    auto result =
        inclusiveScanFunctional(tensor, Placement::OutOfPlace, Determinism::Deterministic);

    EXPECT_EQ(result.notes, "cpu_parallel_int_deterministic");
    expect_scan_equal<int32_t>(result.tensor, expected_scan(input));
}

// -----------------------------------------------------------------------------
// Long (int64_t) type on CPU
// -----------------------------------------------------------------------------

TEST(FunctionalDispatchCPU, Long_Contiguous_OutOfPlace_UsesParallel) {
    auto input  = make_input<int64_t>(100);
    auto tensor = make_tensor(input);

    auto result =
        inclusiveScanFunctional(tensor, Placement::OutOfPlace, Determinism::Deterministic);

    EXPECT_EQ(result.notes, "cpu_parallel_int_deterministic");
    expect_scan_equal<int64_t>(result.tensor, expected_scan(input));
}

TEST(FunctionalDispatchCPU, Long_Strided_InPlace_UsesSerialInPlace) {
    auto input  = make_input<int64_t>(50);
    auto tensor = make_strided_tensor(input);

    ASSERT_FALSE(tensor.is_contiguous());

    auto result = inclusiveScanFunctional(tensor, Placement::InPlace, Determinism::Deterministic);

    EXPECT_EQ(result.notes, "cpu_serial_in_place");
    expect_scan_equal<int64_t>(result.tensor, expected_scan(input));
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
    // Out of place should return a different tensor
    EXPECT_NE(result.tensor.data_ptr(), tensor.data_ptr());
}

TEST(FunctionalDispatchCPU, EmptyTensor_InPlace) {
    auto tensor = torch::empty({0}, torch::kFloat);

    auto result = inclusiveScanFunctional(tensor, Placement::InPlace, Determinism::Deterministic);

    EXPECT_EQ(result.notes, "empty_in_place");
    EXPECT_EQ(result.tensor.size(0), 0);
    // In place should return the same tensor
    EXPECT_EQ(result.tensor.data_ptr(), tensor.data_ptr());
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
    auto input  = make_input<float>(100000);
    auto tensor = make_tensor(input);

    auto result =
        inclusiveScanFunctional(tensor, Placement::OutOfPlace, Determinism::NonDeterministic);

    EXPECT_EQ(result.notes, "cpu_parallel_float_nondeterministic");
    expect_scan_equal<float>(result.tensor, expected_scan(input));
}

// =============================================================================
// Correctness Tests - Verify scan output is mathematically correct
// =============================================================================

TEST(FunctionalCorrectness, ScanSumIsCorrect_Float) {
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    auto tensor              = make_tensor(input);

    auto result =
        inclusiveScanFunctional(tensor, Placement::OutOfPlace, Determinism::Deterministic);

    std::vector<float> expected = {1.0f, 3.0f, 6.0f, 10.0f, 15.0f};
    expect_scan_equal<float>(result.tensor, expected);
}

TEST(FunctionalCorrectness, ScanSumIsCorrect_Int) {
    std::vector<int32_t> input = {1, 2, 3, 4, 5};
    auto tensor                = make_tensor(input);

    auto result =
        inclusiveScanFunctional(tensor, Placement::OutOfPlace, Determinism::Deterministic);

    std::vector<int32_t> expected = {1, 3, 6, 10, 15};
    expect_scan_equal<int32_t>(result.tensor, expected);
}

TEST(FunctionalCorrectness, InPlaceModifiesInput) {
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    auto tensor              = make_tensor(input);
    auto original_ptr        = tensor.data_ptr<float>();

    auto result = inclusiveScanFunctional(tensor, Placement::InPlace, Determinism::Deterministic);

    // Should be same memory
    EXPECT_EQ(result.tensor.data_ptr<float>(), original_ptr);
    // Original tensor should be modified
    std::vector<float> expected = {1.0f, 3.0f, 6.0f, 10.0f, 15.0f};
    expect_scan_equal<float>(tensor, expected);
}

TEST(FunctionalCorrectness, OutOfPlaceDoesNotModifyInput) {
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    auto tensor              = make_tensor(input);

    auto result =
        inclusiveScanFunctional(tensor, Placement::OutOfPlace, Determinism::Deterministic);

    // Input should be unchanged
    for (size_t i = 0; i < input.size(); ++i) {
        EXPECT_FLOAT_EQ(tensor[i].item<float>(), input[i]);
    }
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

    auto input  = make_input<float>(1000);
    auto tensor = make_tensor(input, torch::kCUDA);

    auto result =
        inclusiveScanFunctional(tensor, Placement::OutOfPlace, Determinism::NonDeterministic);

    EXPECT_EQ(result.notes, "cuda_cub");
    expect_scan_equal<float>(result.tensor, expected_scan(input));
}

TEST(FunctionalDispatchCUDA, Double_Contiguous_OutOfPlace_NonDeterministic_UsesCub) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto input  = make_input<double>(1000);
    auto tensor = make_tensor(input, torch::kCUDA);

    auto result =
        inclusiveScanFunctional(tensor, Placement::OutOfPlace, Determinism::NonDeterministic);

    EXPECT_EQ(result.notes, "cuda_cub");
    expect_scan_equal<double>(result.tensor, expected_scan(input));
}

TEST(FunctionalDispatchCUDA, Int_Contiguous_OutOfPlace_Deterministic_UsesCub) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto input  = make_input<int32_t>(1000);
    auto tensor = make_tensor(input, torch::kCUDA);

    auto result =
        inclusiveScanFunctional(tensor, Placement::OutOfPlace, Determinism::Deterministic);

    EXPECT_EQ(result.notes, "cuda_cub");
    expect_scan_equal<int32_t>(result.tensor, expected_scan(input));
}

TEST(FunctionalDispatchCUDA, Long_Contiguous_OutOfPlace_UsesCub) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto input  = make_input<int64_t>(1000);
    auto tensor = make_tensor(input, torch::kCUDA);

    // Integers are promoted to Deterministic
    auto result =
        inclusiveScanFunctional(tensor, Placement::OutOfPlace, Determinism::NonDeterministic);

    EXPECT_EQ(result.notes, "cuda_cub");
    expect_scan_equal<int64_t>(result.tensor, expected_scan(input));
}

TEST(FunctionalDispatchCUDA, Float_Contiguous_OutOfPlace_Deterministic_ThrowsUnsupported) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto input  = make_input<float>(100);
    auto tensor = make_tensor(input, torch::kCUDA);

    // Float + Deterministic is not supported on CUDA (floats are non-deterministic)
    EXPECT_THROW(inclusiveScanFunctional(tensor, Placement::OutOfPlace, Determinism::Deterministic),
                 c10::Error);
}

TEST(FunctionalDispatchCUDA, Float_Contiguous_InPlace_ThrowsUnsupported) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto input  = make_input<float>(100);
    auto tensor = make_tensor(input, torch::kCUDA);

    // InPlace is not supported on CUDA
    EXPECT_THROW(inclusiveScanFunctional(tensor, Placement::InPlace, Determinism::NonDeterministic),
                 c10::Error);
}

TEST(FunctionalDispatchCUDA, Float_Strided_ThrowsUnsupported) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    // Create strided tensor on CUDA
    auto options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    auto full    = torch::arange(200, options);
    auto tensor  = full.slice(0, 0, 200, 2); // stride of 2

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

    auto input  = make_input<float>(100000);
    auto tensor = make_tensor(input, torch::kCUDA);

    auto result =
        inclusiveScanFunctional(tensor, Placement::OutOfPlace, Determinism::NonDeterministic);

    EXPECT_EQ(result.notes, "cuda_cub");
    expect_scan_equal<float>(result.tensor, expected_scan(input));
}

TEST(FunctionalDispatchCUDA, SingleElement) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto tensor = torch::tensor({42.0f}, torch::kCUDA);

    auto result =
        inclusiveScanFunctional(tensor, Placement::OutOfPlace, Determinism::NonDeterministic);

    EXPECT_EQ(result.notes, "cuda_cub");
    EXPECT_EQ(result.tensor.size(0), 1);
    EXPECT_FLOAT_EQ(result.tensor.cpu()[0].item<float>(), 42.0f);
}

} // namespace
} // namespace example
} // namespace dispatch
} // namespace fvdb
