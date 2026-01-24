// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Tests for functional.cu and op.cu examples - parameterized over implementation

#include "examples/common.h"
#include "examples/functional.h"
#include "examples/op.h"
#include "test_utils.h"

#include <torch/torch.h>

#include <gtest/gtest.h>

namespace dispatch_examples {
namespace {

using namespace dispatch;
using namespace dispatch::test;

// =============================================================================
// Parameterized test fixture
// =============================================================================

enum class Implementation { Functional, Op };

class InclusiveScanExamplesTest
    : public ::testing::TestWithParam<
          std::tuple<Implementation, torch::ScalarType, placement, determinism>> {
  protected:
    tensor_with_notes
    call_scan(torch::Tensor input, placement plc, determinism det) {
        auto [impl, stype, expected_plc, expected_det] = GetParam();
        (void)stype;
        (void)expected_plc;
        (void)expected_det;

        if (impl == Implementation::Functional) {
            return inclusive_scan_functional(input, plc, det);
        } else {
            return inclusive_scan_op(input, plc, det);
        }
    }
};

// =============================================================================
// CPU float dispatch
// =============================================================================

class CpuFloatScanTest : public ::testing::TestWithParam<
                             std::tuple<Implementation, contiguity, placement, determinism>> {
  protected:
    tensor_with_notes
    call_scan(torch::Tensor input, placement plc, determinism det) {
        auto [impl, cont, expected_plc, expected_det] = GetParam();
        (void)cont;
        (void)expected_plc;
        (void)expected_det;

        if (impl == Implementation::Functional) {
            return inclusive_scan_functional(input, plc, det);
        } else {
            return inclusive_scan_op(input, plc, det);
        }
    }
};

TEST_P(CpuFloatScanTest, Float_AllCombinations) {
    auto [impl, cont, plc, det] = GetParam();
    int64_t const n             = 100;

    torch::Tensor input;
    if (cont == contiguity::contiguous) {
        input = make_arange_tensor(n, torch::kFloat, torch::kCPU);
    } else {
        input = make_strided_arange_tensor(n, torch::kFloat, torch::kCPU);
    }

    auto result = call_scan(input, plc, det);

    // Verify scan correctness
    expect_scan_equal<float>(result.tensor, expected_arange_scan<float>(n));

    // Verify algorithm selection via notes
    if (plc == placement::in_place) {
        EXPECT_EQ(result.notes, "cpu_serial_in_place");
        EXPECT_TRUE(result.tensor.data_ptr() == input.data_ptr());
    } else if (det == determinism::required) {
        EXPECT_EQ(result.notes, "cpu_serial_out_of_place");
    } else {
        EXPECT_EQ(result.notes, "cpu_parallel_float_nondeterministic");
    }
}

INSTANTIATE_TEST_SUITE_P(
    CpuFloat,
    CpuFloatScanTest,
    ::testing::Combine(::testing::Values(Implementation::Functional, Implementation::Op),
                       ::testing::Values(contiguity::contiguous, contiguity::strided),
                       ::testing::Values(placement::in_place, placement::out_of_place),
                       ::testing::Values(determinism::required, determinism::not_required)));

// =============================================================================
// CPU integer dispatch
// =============================================================================

TEST_P(CpuFloatScanTest, Int_AllCombinations) {
    auto [impl, cont, plc, det] = GetParam();
    int64_t const n             = 100;

    torch::Tensor input;
    if (cont == contiguity::contiguous) {
        input = make_arange_tensor(n, torch::kInt, torch::kCPU);
    } else {
        input = make_strided_arange_tensor(n, torch::kInt, torch::kCPU);
    }

    auto result = call_scan(input, plc, det);

    // Verify scan correctness
    expect_scan_equal<int32_t>(result.tensor, expected_arange_scan<int32_t>(n));

    // Verify algorithm selection
    if (plc == placement::in_place) {
        EXPECT_EQ(result.notes, "cpu_serial_in_place");
        EXPECT_TRUE(result.tensor.data_ptr() == input.data_ptr());
    } else {
        // Integers are always deterministic, so NonDeterministic gets promoted
        EXPECT_EQ(result.notes, "cpu_parallel_int_deterministic");
    }
}

// =============================================================================
// CUDA float dispatch
// =============================================================================

class CudaFloatScanTest : public ::testing::TestWithParam<Implementation> {
  protected:
    void
    SetUp() override {
        skip_if_no_cuda();
    }

    tensor_with_notes
    call_scan(torch::Tensor input, placement plc, determinism det) {
        auto impl = GetParam();
        if (impl == Implementation::Functional) {
            return inclusive_scan_functional(input, plc, det);
        } else {
            return inclusive_scan_op(input, plc, det);
        }
    }
};

TEST_P(CudaFloatScanTest, Float_Valid_Contiguous_OutOfPlace_NonDeterministic) {
    int64_t const n = 1000;

    auto input = make_arange_tensor(n, torch::kFloat, torch::kCUDA);

    auto result = call_scan(input, placement::out_of_place, determinism::not_required);

    EXPECT_EQ(result.notes, "cuda_float_nondeterministic");
    expect_scan_equal<float>(result.tensor, expected_arange_scan<float>(n));
}

TEST_P(CudaFloatScanTest, Float_Error_Deterministic) {
    int64_t const n = 100;

    auto input = make_arange_tensor(n, torch::kFloat, torch::kCUDA);

    // CUDA float with deterministic should throw
    EXPECT_THROW(call_scan(input, placement::out_of_place, determinism::required), c10::Error);
}

TEST_P(CudaFloatScanTest, Float_Error_InPlace) {
    int64_t const n = 100;

    auto input = make_arange_tensor(n, torch::kFloat, torch::kCUDA);

    // CUDA with in-place should throw
    EXPECT_THROW(call_scan(input, placement::in_place, determinism::not_required), c10::Error);
}

TEST_P(CudaFloatScanTest, Float_Error_Strided) {
    int64_t const n = 100;

    auto input = make_strided_arange_tensor(n, torch::kFloat, torch::kCUDA);

    // CUDA with strided should throw
    EXPECT_THROW(call_scan(input, placement::out_of_place, determinism::not_required), c10::Error);
}

INSTANTIATE_TEST_SUITE_P(CudaFloat,
                         CudaFloatScanTest,
                         ::testing::Values(Implementation::Functional, Implementation::Op));

// =============================================================================
// CUDA integer dispatch
// =============================================================================

TEST_P(CudaFloatScanTest, Int_Valid_Contiguous_OutOfPlace_Deterministic) {
    int64_t const n = 1000;

    auto input = make_arange_tensor(n, torch::kInt, torch::kCUDA);

    auto result = call_scan(input, placement::out_of_place, determinism::required);

    EXPECT_EQ(result.notes, "cuda_int_deterministic");
    expect_scan_equal<int32_t>(result.tensor, expected_arange_scan<int32_t>(n));
}

TEST_P(CudaFloatScanTest, Int_NonDeterministic_PromotedToDeterministic) {
    int64_t const n = 1000;

    auto input = make_arange_tensor(n, torch::kInt, torch::kCUDA);

    // NonDeterministic should be promoted to Deterministic for integers
    auto result = call_scan(input, placement::out_of_place, determinism::not_required);

    EXPECT_EQ(result.notes, "cuda_int_deterministic");
    expect_scan_equal<int32_t>(result.tensor, expected_arange_scan<int32_t>(n));
}

// =============================================================================
// Input validation
// =============================================================================

class InputValidationTest : public ::testing::TestWithParam<Implementation> {
  protected:
    tensor_with_notes
    call_scan(torch::Tensor input, placement plc, determinism det) {
        auto impl = GetParam();
        if (impl == Implementation::Functional) {
            return inclusive_scan_functional(input, plc, det);
        } else {
            return inclusive_scan_op(input, plc, det);
        }
    }
};

TEST_P(InputValidationTest, Rank0_Scalar_Throws) {
    auto scalar = torch::tensor(42.0f);

    EXPECT_THROW(call_scan(scalar, placement::out_of_place, determinism::not_required), c10::Error);
}

TEST_P(InputValidationTest, Rank2_Throws) {
    auto tensor = torch::zeros({10, 20});

    EXPECT_THROW(call_scan(tensor, placement::out_of_place, determinism::not_required), c10::Error);
}

TEST_P(InputValidationTest, Rank1_Supported) {
    auto tensor = make_arange_tensor(100, torch::kFloat, torch::kCPU);

    // Should not throw
    auto result = call_scan(tensor, placement::out_of_place, determinism::not_required);
    EXPECT_GT(result.tensor.numel(), 0);
}

INSTANTIATE_TEST_SUITE_P(InputValidation,
                         InputValidationTest,
                         ::testing::Values(Implementation::Functional, Implementation::Op));

// =============================================================================
// Edge cases
// =============================================================================

TEST_P(InputValidationTest, EmptyTensor) {
    auto tensor = torch::zeros({0}, torch::TensorOptions().dtype(torch::kFloat));

    auto result = call_scan(tensor, placement::out_of_place, determinism::not_required);
    EXPECT_EQ(result.tensor.size(0), 0);
}

TEST_P(InputValidationTest, SingleElement) {
    auto tensor = torch::tensor({42.0f});

    auto result = call_scan(tensor, placement::out_of_place, determinism::not_required);
    EXPECT_EQ(result.tensor.size(0), 1);
    EXPECT_FLOAT_EQ(result.tensor.cpu().data_ptr<float>()[0], 42.0f);
}

TEST_P(InputValidationTest, LargeInput) {
    constexpr int64_t n = 100000;
    auto tensor         = make_arange_tensor(n, torch::kFloat, torch::kCPU);

    auto result = call_scan(tensor, placement::out_of_place, determinism::not_required);
    expect_scan_equal<float>(result.tensor, expected_arange_scan<float>(n));
}

// =============================================================================
// Functional vs Op equivalence
// =============================================================================

TEST(InclusiveScanEquivalence, SameResults) {
    int64_t const n = 1000;
    auto input      = make_arange_tensor(n, torch::kFloat, torch::kCPU);

    auto result_func =
        inclusive_scan_functional(input, placement::out_of_place, determinism::not_required);
    auto result_op = inclusive_scan_op(input, placement::out_of_place, determinism::not_required);

    // Both should produce same results
    expect_scan_equal<float>(result_func.tensor, expected_arange_scan<float>(n));
    expect_scan_equal<float>(result_op.tensor, expected_arange_scan<float>(n));

    // Notes may differ slightly but both should indicate parallel
    EXPECT_TRUE(result_func.notes.find("parallel") != std::string::npos ||
                result_func.notes.find("cpu_parallel") != std::string::npos);
    EXPECT_TRUE(result_op.notes.find("parallel") != std::string::npos ||
                result_op.notes.find("cpu_parallel") != std::string::npos);
}

} // namespace
} // namespace dispatch_examples
