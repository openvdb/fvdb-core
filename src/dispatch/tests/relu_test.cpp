// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include "examples/relu.h"
#include "test_utils.h"

#include <torch/torch.h>

#include <gtest/gtest.h>

namespace dispatch_examples {
namespace {

using namespace dispatch::test;

// =============================================================================
// Test utilities
// =============================================================================

/// Create a 1D tensor with mixed positive and negative values: [-n/2, ..., -1, 0, 1, ..., n/2]
inline torch::Tensor
make_relu_test_tensor(int64_t n, torch::ScalarType dtype, torch::Device device = torch::kCPU) {
    auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto values        = torch::arange(0, n, float_options) - static_cast<float>(n / 2);
    return values.to(dtype);
}

/// Create a strided (non-contiguous) tensor with mixed values for ReLU testing
inline torch::Tensor
make_strided_relu_test_tensor(int64_t n,
                              torch::ScalarType dtype,
                              torch::Device device = torch::kCPU) {
    auto options       = torch::TensorOptions().dtype(dtype).device(device);
    auto full          = torch::zeros({n * 2}, options);
    auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto values        = torch::arange(0, n, float_options) - static_cast<float>(n / 2);
    full.slice(0, 0, n * 2, 2).copy_(values);
    return full.slice(0, 0, n * 2, 2); // stride of 2
}

/// Compute expected ReLU output: max(0, x) for each element
inline torch::Tensor
compute_expected_relu(torch::Tensor input) {
    auto float_input = input.to(torch::kFloat32);
    auto result      = torch::relu(float_input);
    return result.to(input.scalar_type());
}

// =============================================================================
// Parameterized tests
// =============================================================================

struct ReluTestParams {
    torch::ScalarType dtype;
    torch::Device device;
    placement placement;
    bool contiguous;
};

class ReluTest : public ::testing::TestWithParam<ReluTestParams> {
  protected:
    void
    SetUp() override {
        auto params = GetParam();
        if (params.device.is_cuda()) {
            skip_if_no_cuda();
        }
    }

    torch::Tensor
    createInputTensor(int64_t n) {
        auto params = GetParam();
        if (params.contiguous) {
            return make_relu_test_tensor(n, params.dtype, params.device);
        } else {
            return make_strided_relu_test_tensor(n, params.dtype, params.device);
        }
    }
};

TEST_P(ReluTest, Correctness) {
    auto params     = GetParam();
    int64_t const n = 100;
    auto input      = createInputTensor(n);
    auto expected   = compute_expected_relu(input);

    torch::Tensor result;
    if (params.placement == placement::in_place) {
        result = relu_(input);
    } else {
        result = relu(input);
    }

    // Verify ReLU math: max(0, x)
    double tol = 1e-5;
    if (params.dtype == torch::kFloat16 || params.dtype == torch::kBFloat16) {
        tol = 1e-2; // Half precision needs larger tolerance
    }
    expect_relu_equal(result, expected, tol);
}

TEST_P(ReluTest, InPlaceModifiesInput) {
    auto params        = GetParam();
    int64_t const n    = 100;
    auto input         = createInputTensor(n);
    auto input_clone   = input.clone();
    void *original_ptr = input.data_ptr();

    if (params.placement == placement::in_place) {
        auto result = relu_(input);

        // Should return same tensor
        EXPECT_EQ(result.data_ptr(), original_ptr);
        // Input should be modified
        EXPECT_FALSE(torch::equal(input.to(torch::kFloat32), input_clone.to(torch::kFloat32)));
    }
}

TEST_P(ReluTest, OutOfPlaceDoesNotModifyInput) {
    auto params      = GetParam();
    int64_t const n  = 100;
    auto input       = createInputTensor(n);
    auto input_clone = input.clone();

    if (params.placement == placement::out_of_place) {
        auto result = relu(input);

        // Should return different tensor
        EXPECT_NE(result.data_ptr(), input.data_ptr());
        // Input should be unchanged
        EXPECT_TRUE(torch::equal(input.to(torch::kFloat32), input_clone.to(torch::kFloat32)));
    }
}

TEST_P(ReluTest, DevicePreservation) {
    auto params     = GetParam();
    int64_t const n = 100;
    auto input      = createInputTensor(n);

    torch::Tensor result;
    if (params.placement == placement::in_place) {
        result = relu_(input);
    } else {
        result = relu(input);
    }

    EXPECT_EQ(result.device().type(), params.device.type());
}

TEST_P(ReluTest, DtypePreservation) {
    auto params     = GetParam();
    int64_t const n = 100;
    auto input      = createInputTensor(n);

    torch::Tensor result;
    if (params.placement == placement::in_place) {
        result = relu_(input);
    } else {
        result = relu(input);
    }

    EXPECT_EQ(result.scalar_type(), params.dtype);
}

// Generate all parameter combinations
std::vector<ReluTestParams>
GenerateAllTests() {
    std::vector<ReluTestParams> params;

    torch::ScalarType dtypes[] = {
        torch::kFloat32, torch::kFloat64, torch::kFloat16, torch::kBFloat16};
    torch::Device devices[] = {torch::kCPU, torch::kCUDA};
    placement placements[]  = {placement::in_place, placement::out_of_place};
    bool contiguities[]     = {true, false};

    for (auto device: devices) {
        for (auto dtype: dtypes) {
            for (auto plc: placements) {
                for (bool cont: contiguities) {
                    params.push_back({dtype, device, plc, cont});
                }
            }
        }
    }

    return params;
}

INSTANTIATE_TEST_SUITE_P(AllCombinations, ReluTest, ::testing::ValuesIn(GenerateAllTests()));

// =============================================================================
// Edge cases
// =============================================================================

TEST(ReluEdgeCases, EmptyTensor) {
    auto tensor = torch::zeros({0}, torch::TensorOptions().dtype(torch::kFloat32));
    auto result = relu(tensor);
    EXPECT_EQ(result.size(0), 0);
}

TEST(ReluEdgeCases, SingleElement_Positive) {
    auto tensor = torch::tensor({5.0f});
    auto result = relu(tensor);
    EXPECT_FLOAT_EQ(result.cpu().data_ptr<float>()[0], 5.0f);
}

TEST(ReluEdgeCases, SingleElement_Negative) {
    auto tensor = torch::tensor({-5.0f});
    auto result = relu(tensor);
    EXPECT_FLOAT_EQ(result.cpu().data_ptr<float>()[0], 0.0f);
}

TEST(ReluEdgeCases, SingleElement_Zero) {
    auto tensor = torch::tensor({0.0f});
    auto result = relu(tensor);
    EXPECT_FLOAT_EQ(result.cpu().data_ptr<float>()[0], 0.0f);
}

TEST(ReluEdgeCases, AllNegative) {
    auto tensor   = torch::tensor({-1.0f, -2.0f, -3.0f});
    auto result   = relu(tensor);
    auto expected = torch::zeros_like(tensor);
    EXPECT_TRUE(torch::equal(result, expected));
}

TEST(ReluEdgeCases, AllPositive) {
    auto tensor = torch::tensor({1.0f, 2.0f, 3.0f});
    auto result = relu(tensor);
    EXPECT_TRUE(torch::equal(result, tensor));
}

TEST(ReluEdgeCases, AllZero) {
    auto tensor = torch::zeros({10});
    auto result = relu(tensor);
    EXPECT_TRUE(torch::equal(result, tensor));
}

TEST(ReluEdgeCases, LargeInput) {
    constexpr int64_t n = 100000;
    auto tensor         = make_relu_test_tensor(n, torch::kFloat32, torch::kCPU);
    auto result         = relu(tensor);
    auto expected       = compute_expected_relu(tensor);
    expect_relu_equal(result, expected);
}

// =============================================================================
// Half precision tolerance
// =============================================================================

TEST(ReluHalfPrecision, Float16) {
    auto tensor   = make_relu_test_tensor(100, torch::kFloat16, torch::kCPU);
    auto result   = relu(tensor);
    auto expected = compute_expected_relu(tensor);
    expect_relu_equal(result, expected, 1e-2); // Larger tolerance for half
}

TEST(ReluHalfPrecision, BFloat16) {
    auto tensor   = make_relu_test_tensor(100, torch::kBFloat16, torch::kCPU);
    auto result   = relu(tensor);
    auto expected = compute_expected_relu(tensor);
    expect_relu_equal(result, expected, 1e-2); // Larger tolerance for half
}

} // namespace
} // namespace dispatch_examples
