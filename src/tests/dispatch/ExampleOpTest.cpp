// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <fvdb/detail/dispatch/example/Common.h>
#include <fvdb/detail/dispatch/example/Op.h>

#include <gtest/gtest.h>
#include <tests/dispatch/ExampleUtils.h>

namespace fvdb {
namespace dispatch {
namespace example {
namespace {

// =============================================================================
// CPU Dispatch Tests - Verify correct dispatch path via notes
// =============================================================================

// -----------------------------------------------------------------------------
// Float types on CPU
// -----------------------------------------------------------------------------

TEST(OpDispatchCPU, Float_Contiguous_OutOfPlace_Deterministic_UsesSerial) {
    int64_t const n = 100;
    auto tensor     = make_arange_tensor(n, torch::kFloat);

    auto result = inclusiveScanOp(tensor, Placement::OutOfPlace, Determinism::Deterministic);

    EXPECT_EQ(result.notes, "cpu_serial_out_of_place");
    expect_scan_equal<float>(result.tensor, expected_arange_scan<float>(n));
}

TEST(OpDispatchCPU, Float_Contiguous_OutOfPlace_NonDeterministic_UsesParallel) {
    int64_t const n = 100;
    auto tensor     = make_arange_tensor(n, torch::kFloat);

    auto result = inclusiveScanOp(tensor, Placement::OutOfPlace, Determinism::NonDeterministic);

    EXPECT_EQ(result.notes, "cpu_parallel_deterministic");
    expect_scan_equal<float>(result.tensor, expected_arange_scan<float>(n));
}

TEST(OpDispatchCPU, Float_Contiguous_InPlace_Deterministic_UsesSerialInPlace) {
    int64_t const n = 100;
    auto tensor     = make_arange_tensor(n, torch::kFloat);

    auto result = inclusiveScanOp(tensor, Placement::InPlace, Determinism::Deterministic);

    EXPECT_EQ(result.notes, "cpu_serial_in_place");
    // For in-place, result.tensor should be same as input tensor
    EXPECT_TRUE(result.tensor.data_ptr() == tensor.data_ptr());
    expect_scan_equal<float>(result.tensor, expected_arange_scan<float>(n));
}

TEST(OpDispatchCPU, Float_Contiguous_InPlace_NonDeterministic_UsesSerialInPlace) {
    int64_t const n = 100;
    auto tensor     = make_arange_tensor(n, torch::kFloat);

    auto result = inclusiveScanOp(tensor, Placement::InPlace, Determinism::NonDeterministic);

    EXPECT_EQ(result.notes, "cpu_serial_in_place");
    EXPECT_TRUE(result.tensor.data_ptr() == tensor.data_ptr());
    expect_scan_equal<float>(result.tensor, expected_arange_scan<float>(n));
}

TEST(OpDispatchCPU, Float_Strided_OutOfPlace_NonDeterministic_UsesParallel) {
    int64_t const n = 100;
    auto tensor     = make_strided_arange_tensor(n, torch::kFloat);

    ASSERT_FALSE(tensor.is_contiguous());

    auto result = inclusiveScanOp(tensor, Placement::OutOfPlace, Determinism::NonDeterministic);

    EXPECT_EQ(result.notes, "cpu_parallel_deterministic");
    expect_scan_equal<float>(result.tensor, expected_arange_scan<float>(n));
}

TEST(OpDispatchCPU, Float_Strided_InPlace_UsesSerialInPlace) {
    int64_t const n = 50;
    auto tensor     = make_strided_arange_tensor(n, torch::kFloat);

    ASSERT_FALSE(tensor.is_contiguous());

    auto result = inclusiveScanOp(tensor, Placement::InPlace, Determinism::Deterministic);

    EXPECT_EQ(result.notes, "cpu_serial_in_place");
    expect_scan_equal<float>(result.tensor, expected_arange_scan<float>(n));
}

// -----------------------------------------------------------------------------
// Double type on CPU
// -----------------------------------------------------------------------------

TEST(OpDispatchCPU, Double_Contiguous_OutOfPlace_Deterministic_UsesSerial) {
    int64_t const n = 100;
    auto tensor     = make_arange_tensor(n, torch::kDouble);

    auto result = inclusiveScanOp(tensor, Placement::OutOfPlace, Determinism::Deterministic);

    EXPECT_EQ(result.notes, "cpu_serial_out_of_place");
    expect_scan_equal<double>(result.tensor, expected_arange_scan<double>(n));
}

TEST(OpDispatchCPU, Double_Contiguous_OutOfPlace_NonDeterministic_UsesParallel) {
    int64_t const n = 100;
    auto tensor     = make_arange_tensor(n, torch::kDouble);

    auto result = inclusiveScanOp(tensor, Placement::OutOfPlace, Determinism::NonDeterministic);

    EXPECT_EQ(result.notes, "cpu_parallel_deterministic");
    expect_scan_equal<double>(result.tensor, expected_arange_scan<double>(n));
}

// -----------------------------------------------------------------------------
// Integer types on CPU - determinism is promoted to Deterministic
// -----------------------------------------------------------------------------

TEST(OpDispatchCPU, Int_Contiguous_OutOfPlace_Deterministic_UsesParallel) {
    int64_t const n = 100;
    auto tensor     = make_arange_tensor(n, torch::kInt);

    auto result = inclusiveScanOp(tensor, Placement::OutOfPlace, Determinism::Deterministic);

    EXPECT_EQ(result.notes, "cpu_parallel_deterministic");
    expect_scan_equal<int32_t>(result.tensor, expected_arange_scan<int32_t>(n));
}

TEST(OpDispatchCPU, Int_Contiguous_OutOfPlace_NonDeterministic_PromotedToDeterministic) {
    // Integers are promoted to deterministic, so NonDeterministic should still use parallel
    int64_t const n = 100;
    auto tensor     = make_arange_tensor(n, torch::kInt);

    auto result = inclusiveScanOp(tensor, Placement::OutOfPlace, Determinism::NonDeterministic);

    // Even though NonDeterministic was requested, integers get promoted to Deterministic
    EXPECT_EQ(result.notes, "cpu_parallel_deterministic");
    expect_scan_equal<int32_t>(result.tensor, expected_arange_scan<int32_t>(n));
}

TEST(OpDispatchCPU, Int_Contiguous_InPlace_UsesSerialInPlace) {
    int64_t const n = 100;
    auto tensor     = make_arange_tensor(n, torch::kInt);

    auto result = inclusiveScanOp(tensor, Placement::InPlace, Determinism::Deterministic);

    EXPECT_EQ(result.notes, "cpu_serial_in_place");
    EXPECT_TRUE(result.tensor.data_ptr() == tensor.data_ptr());
    expect_scan_equal<int32_t>(result.tensor, expected_arange_scan<int32_t>(n));
}

TEST(OpDispatchCPU, Int_Strided_OutOfPlace_UsesParallel) {
    int64_t const n = 100;
    auto tensor     = make_strided_arange_tensor(n, torch::kInt);

    ASSERT_FALSE(tensor.is_contiguous());

    auto result = inclusiveScanOp(tensor, Placement::OutOfPlace, Determinism::Deterministic);

    EXPECT_EQ(result.notes, "cpu_parallel_deterministic");
    expect_scan_equal<int32_t>(result.tensor, expected_arange_scan<int32_t>(n));
}

// -----------------------------------------------------------------------------
// Long (int64_t) type on CPU
// -----------------------------------------------------------------------------

TEST(OpDispatchCPU, Long_Contiguous_OutOfPlace_UsesParallel) {
    int64_t const n = 100;
    auto tensor     = make_arange_tensor(n, torch::kLong);

    auto result = inclusiveScanOp(tensor, Placement::OutOfPlace, Determinism::Deterministic);

    EXPECT_EQ(result.notes, "cpu_parallel_deterministic");
    expect_scan_equal<int64_t>(result.tensor, expected_arange_scan<int64_t>(n));
}

TEST(OpDispatchCPU, Long_Strided_InPlace_UsesSerialInPlace) {
    int64_t const n = 50;
    auto tensor     = make_strided_arange_tensor(n, torch::kLong);

    ASSERT_FALSE(tensor.is_contiguous());

    auto result = inclusiveScanOp(tensor, Placement::InPlace, Determinism::Deterministic);

    EXPECT_EQ(result.notes, "cpu_serial_in_place");
    expect_scan_equal<int64_t>(result.tensor, expected_arange_scan<int64_t>(n));
}

// =============================================================================
// Input Validation Tests
// =============================================================================

TEST(OpValidation, Rank0_Throws) {
    auto tensor = torch::tensor(42.0f); // scalar (0D)

    EXPECT_THROW(inclusiveScanOp(tensor, Placement::OutOfPlace, Determinism::Deterministic),
                 c10::Error);
}

TEST(OpValidation, Rank2_Throws) {
    auto tensor = torch::empty({10, 10}, torch::kFloat);

    EXPECT_THROW(inclusiveScanOp(tensor, Placement::OutOfPlace, Determinism::Deterministic),
                 c10::Error);
}

TEST(OpValidation, Rank3_Throws) {
    auto tensor = torch::empty({2, 3, 4}, torch::kFloat);

    EXPECT_THROW(inclusiveScanOp(tensor, Placement::OutOfPlace, Determinism::Deterministic),
                 c10::Error);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST(OpDispatchCPU, EmptyTensor_OutOfPlace) {
    auto tensor = torch::empty({0}, torch::kFloat);

    auto result = inclusiveScanOp(tensor, Placement::OutOfPlace, Determinism::Deterministic);

    EXPECT_EQ(result.notes, "empty_out_of_place");
    EXPECT_EQ(result.tensor.size(0), 0);
    // Out of place should return a different tensor (use is_same, not data_ptr which is NULL)
    EXPECT_FALSE(result.tensor.is_same(tensor));
}

TEST(OpDispatchCPU, EmptyTensor_InPlace) {
    auto tensor = torch::empty({0}, torch::kFloat);

    auto result = inclusiveScanOp(tensor, Placement::InPlace, Determinism::Deterministic);

    EXPECT_EQ(result.notes, "empty_in_place");
    EXPECT_EQ(result.tensor.size(0), 0);
    // In place should return the same tensor
    EXPECT_TRUE(result.tensor.is_same(tensor));
}

TEST(OpDispatchCPU, SingleElement) {
    auto tensor = torch::tensor({42.0f});

    auto result = inclusiveScanOp(tensor, Placement::OutOfPlace, Determinism::Deterministic);

    EXPECT_EQ(result.tensor.size(0), 1);
    EXPECT_FLOAT_EQ(result.tensor[0].item<float>(), 42.0f);
}

TEST(OpDispatchCPU, SingleElementInPlace) {
    auto tensor = torch::tensor({42.0f});

    auto result = inclusiveScanOp(tensor, Placement::InPlace, Determinism::Deterministic);

    EXPECT_EQ(result.notes, "cpu_serial_in_place");
    EXPECT_EQ(result.tensor.size(0), 1);
    EXPECT_FLOAT_EQ(result.tensor[0].item<float>(), 42.0f);
}

TEST(OpDispatchCPU, LargeInput) {
    int64_t const n = 100000;
    auto tensor     = make_arange_tensor(n, torch::kFloat);

    auto result = inclusiveScanOp(tensor, Placement::OutOfPlace, Determinism::NonDeterministic);

    EXPECT_EQ(result.notes, "cpu_parallel_deterministic");
    expect_scan_equal<float>(result.tensor, expected_arange_scan<float>(n));
}

// =============================================================================
// Correctness Tests - Verify scan output is mathematically correct
// =============================================================================

TEST(OpCorrectness, ScanSumIsCorrect_Float) {
    int64_t const n = 5;
    auto tensor     = make_arange_tensor(n, torch::kFloat); // [1, 2, 3, 4, 5]

    auto result = inclusiveScanOp(tensor, Placement::OutOfPlace, Determinism::Deterministic);

    std::vector<float> expected = {1.0f, 3.0f, 6.0f, 10.0f, 15.0f};
    expect_scan_equal<float>(result.tensor, expected);
}

TEST(OpCorrectness, ScanSumIsCorrect_Int) {
    int64_t const n = 5;
    auto tensor     = make_arange_tensor(n, torch::kInt); // [1, 2, 3, 4, 5]

    auto result = inclusiveScanOp(tensor, Placement::OutOfPlace, Determinism::Deterministic);

    std::vector<int32_t> expected = {1, 3, 6, 10, 15};
    expect_scan_equal<int32_t>(result.tensor, expected);
}

TEST(OpCorrectness, InPlaceModifiesInput) {
    int64_t const n   = 5;
    auto tensor       = make_arange_tensor(n, torch::kFloat); // [1, 2, 3, 4, 5]
    auto original_ptr = tensor.data_ptr<float>();

    auto result = inclusiveScanOp(tensor, Placement::InPlace, Determinism::Deterministic);

    // Should be same memory
    EXPECT_EQ(result.tensor.data_ptr<float>(), original_ptr);
    // Original tensor should be modified
    std::vector<float> expected = {1.0f, 3.0f, 6.0f, 10.0f, 15.0f};
    expect_scan_equal<float>(tensor, expected);
}

TEST(OpCorrectness, OutOfPlaceDoesNotModifyInput) {
    int64_t const n = 5;
    auto tensor     = make_arange_tensor(n, torch::kFloat); // [1, 2, 3, 4, 5]
    auto original   = tensor.clone();                       // Save original values

    auto result = inclusiveScanOp(tensor, Placement::OutOfPlace, Determinism::Deterministic);

    // Input should be unchanged
    EXPECT_TRUE(torch::equal(tensor, original));
    // Result should be different memory
    EXPECT_NE(result.tensor.data_ptr<float>(), tensor.data_ptr<float>());
}

// =============================================================================
// CUDA Dispatch Tests
// =============================================================================

TEST(OpDispatchCUDA, Float_Contiguous_OutOfPlace_NonDeterministic_UsesCuda) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    int64_t const n = 1000;
    auto tensor     = make_arange_tensor(n, torch::kFloat, torch::kCUDA);

    auto result = inclusiveScanOp(tensor, Placement::OutOfPlace, Determinism::NonDeterministic);

    EXPECT_EQ(result.notes, "cuda_non_deterministic");
    expect_scan_equal<float>(result.tensor, expected_arange_scan<float>(n));
}

TEST(OpDispatchCUDA, Double_Contiguous_OutOfPlace_NonDeterministic_UsesCuda) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    int64_t const n = 1000;
    auto tensor     = make_arange_tensor(n, torch::kDouble, torch::kCUDA);

    auto result = inclusiveScanOp(tensor, Placement::OutOfPlace, Determinism::NonDeterministic);

    EXPECT_EQ(result.notes, "cuda_non_deterministic");
    expect_scan_equal<double>(result.tensor, expected_arange_scan<double>(n));
}

TEST(OpDispatchCUDA, Int_Contiguous_OutOfPlace_Deterministic_UsesCuda) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    int64_t const n = 1000;
    auto tensor     = make_arange_tensor(n, torch::kInt, torch::kCUDA);

    auto result = inclusiveScanOp(tensor, Placement::OutOfPlace, Determinism::Deterministic);

    EXPECT_EQ(result.notes, "cuda_non_deterministic");
    expect_scan_equal<int32_t>(result.tensor, expected_arange_scan<int32_t>(n));
}

TEST(OpDispatchCUDA, Long_Contiguous_OutOfPlace_UsesCuda) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    int64_t const n = 1000;
    auto tensor     = make_arange_tensor(n, torch::kLong, torch::kCUDA);

    // Integers are promoted to Deterministic
    auto result = inclusiveScanOp(tensor, Placement::OutOfPlace, Determinism::NonDeterministic);

    EXPECT_EQ(result.notes, "cuda_non_deterministic");
    expect_scan_equal<int64_t>(result.tensor, expected_arange_scan<int64_t>(n));
}

TEST(OpDispatchCUDA, Float_Contiguous_OutOfPlace_Deterministic_ThrowsUnsupported) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto tensor = make_arange_tensor(100, torch::kFloat, torch::kCUDA);

    // Float + Deterministic is not supported on CUDA (floats are non-deterministic)
    EXPECT_THROW(inclusiveScanOp(tensor, Placement::OutOfPlace, Determinism::Deterministic),
                 c10::Error);
}

TEST(OpDispatchCUDA, Float_Contiguous_InPlace_ThrowsUnsupported) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto tensor = make_arange_tensor(100, torch::kFloat, torch::kCUDA);

    // InPlace is not supported on CUDA
    EXPECT_THROW(inclusiveScanOp(tensor, Placement::InPlace, Determinism::NonDeterministic),
                 c10::Error);
}

TEST(OpDispatchCUDA, Float_Strided_ThrowsUnsupported) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    // Create strided tensor on CUDA using our utility
    auto tensor = make_strided_arange_tensor(100, torch::kFloat, torch::kCUDA);

    ASSERT_FALSE(tensor.is_contiguous());

    // Strided is not supported on CUDA
    EXPECT_THROW(inclusiveScanOp(tensor, Placement::OutOfPlace, Determinism::NonDeterministic),
                 c10::Error);
}

TEST(OpDispatchCUDA, LargeInput) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    int64_t const n = 100000;
    auto tensor     = make_arange_tensor(n, torch::kFloat, torch::kCUDA);

    auto result = inclusiveScanOp(tensor, Placement::OutOfPlace, Determinism::NonDeterministic);

    EXPECT_EQ(result.notes, "cuda_non_deterministic");
    expect_scan_equal<float>(result.tensor, expected_arange_scan<float>(n));
}

TEST(OpDispatchCUDA, SingleElement) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto tensor = make_arange_tensor(1, torch::kFloat, torch::kCUDA); // [1.0]

    auto result = inclusiveScanOp(tensor, Placement::OutOfPlace, Determinism::NonDeterministic);

    EXPECT_EQ(result.notes, "cuda_non_deterministic");
    EXPECT_EQ(result.tensor.size(0), 1);
    EXPECT_FLOAT_EQ(result.tensor.cpu()[0].item<float>(), 1.0f);
}

} // namespace
} // namespace example
} // namespace dispatch
} // namespace fvdb
