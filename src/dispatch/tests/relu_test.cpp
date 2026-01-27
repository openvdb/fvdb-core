// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include "examples/relu.h"
#include "test_utils.h"

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

#include <dispatch/types.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace dispatch_examples {
namespace {

using dispatch::test::expect_relu_equal;
using dispatch::test::skip_if_no_cuda;

// =============================================================================
// Test utilities
// =============================================================================

/// Create a 1D tensor with mixed positive and negative values: [-2, -1, 0, 1, 2, ...]
/// For testing ReLU, we want negative values that become 0 and positive values that stay.
inline torch::Tensor
make_relu_test_tensor(int64_t n, torch::ScalarType dtype, torch::Device device = torch::kCPU) {
    // Create [-n/2, ..., -1, 0, 1, ..., n/2] style values
    // Use float computation then convert to target dtype
    auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto values        = torch::arange(0, n, float_options) - static_cast<float>(n / 2);
    return values.to(dtype);
}

/// Create a strided (non-contiguous) tensor with mixed values for ReLU testing
inline torch::Tensor
make_strided_relu_test_tensor(int64_t n,
                              torch::ScalarType dtype,
                              torch::Device device = torch::kCPU) {
    // Create full tensor directly in target dtype, then slice to get stride=2
    // Note: We must NOT call .to(dtype) on a strided tensor - it creates a contiguous copy!
    auto options = torch::TensorOptions().dtype(dtype).device(device);
    auto full    = torch::zeros({n * 2}, options);
    // Create values in float32 for the arithmetic, then copy_ handles conversion
    auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto values        = torch::arange(0, n, float_options) - static_cast<float>(n / 2);
    full.slice(0, 0, n * 2, 2).copy_(values); // copy_ converts dtype automatically
    return full.slice(0, 0, n * 2, 2);        // stride of 2, stays in target dtype
}

/// Compute expected ReLU output: max(0, x) for each element
inline torch::Tensor
compute_expected_relu(torch::Tensor input) {
    // Use float for computation to avoid issues with half types
    auto float_input = input.to(torch::kFloat32);
    auto result      = torch::relu(float_input);
    return result.to(input.scalar_type());
}

// =============================================================================
// Parameterized test base for ReLU
// =============================================================================

struct ReluTestParams {
    torch::ScalarType dtype;
    torch::Device device;
    dispatch::placement placement;
    bool contiguous;
};

class ReluTest : public ::testing::TestWithParam<ReluTestParams> {
  protected:
    void
    SetUp() override {
        auto params = GetParam();
        if (params.device.is_cuda() && !torch::cuda::is_available()) {
            GTEST_SKIP() << "CUDA not available";
        }
    }

    torch::Tensor
    create_input_tensor(int64_t n) {
        auto params = GetParam();
        if (params.contiguous) {
            return make_relu_test_tensor(n, params.dtype, params.device);
        } else {
            return make_strided_relu_test_tensor(n, params.dtype, params.device);
        }
    }

    // Call the appropriate ReLU function based on placement
    torch::Tensor
    call_relu(torch::Tensor input) {
        auto params = GetParam();
        if (params.placement == dispatch::placement::in_place) {
            return example_relu_(input);
        } else {
            return example_relu(input);
        }
    }
};

// =============================================================================
// Test implementation shared across all parameterized tests
// =============================================================================

TEST_P(ReluTest, CorrectnessAndPlacement) {
    auto params        = GetParam();
    int64_t const n    = 100;
    auto input         = create_input_tensor(n);
    auto input_clone   = input.clone(); // Save original for out-of-place verification
    auto expected      = compute_expected_relu(input);
    void *original_ptr = input.data_ptr();
    bool is_contiguous = input.is_contiguous();

    // Build context string for error messages
    std::string context;
    context += (params.device.is_cuda() ? "CUDA_" : "CPU_");
    context += c10::toString(params.dtype);
    context += (params.placement == dispatch::placement::in_place ? "_InPlace" : "_OutOfPlace");
    context += (params.contiguous ? "_Contiguous" : "_Strided");

    ASSERT_EQ(is_contiguous, params.contiguous) << "Test setup error: contiguity mismatch";

    auto result = call_relu(input);

    // Verify correctness
    expect_relu_equal(result, expected);

    // Verify placement semantics
    if (params.placement == dispatch::placement::in_place) {
        EXPECT_EQ(result.data_ptr(), original_ptr)
            << context << ": InPlace should return same memory";
        // Input should be modified
        expect_relu_equal(input, expected);
    } else {
        EXPECT_NE(result.data_ptr(), original_ptr)
            << context << ": OutOfPlace should return different memory";
        // Input should be unchanged
        EXPECT_TRUE(torch::equal(input.to(torch::kFloat32), input_clone.to(torch::kFloat32)))
            << context << ": OutOfPlace should not modify input";
    }

    // Verify output is on correct device
    EXPECT_EQ(result.device().type(), params.device.type()) << context;

    // Verify output dtype matches input
    EXPECT_EQ(result.scalar_type(), params.dtype) << context;
}

// =============================================================================
// Generate all test parameter combinations
// =============================================================================

// Helper to generate test name from params
std::string
test_name_generator(::testing::TestParamInfo<ReluTestParams> const &info) {
    auto const &p = info.param;
    std::string name;
    name += (p.device.is_cuda() ? "CUDA_" : "CPU_");
    name += c10::toString(p.dtype);
    name += (p.placement == dispatch::placement::in_place ? "_InPlace" : "_OutOfPlace");
    name += (p.contiguous ? "_Contiguous" : "_Strided");
    return name;
}

// All float types including half precision
static torch::ScalarType const kAllFloatTypes[] = {
    torch::kFloat32, torch::kFloat64, torch::kFloat16, torch::kBFloat16};

std::vector<ReluTestParams>
generate_all_tests() {
    std::vector<ReluTestParams> params;

    // Generate all combinations: device x dtype x placement x contiguity
    torch::Device devices[] = {torch::kCPU, torch::kCUDA};

    for (auto device: devices) {
        for (auto dtype: kAllFloatTypes) {
            for (auto placement:
                 {dispatch::placement::in_place, dispatch::placement::out_of_place}) {
                for (bool contiguous: {true, false}) {
                    params.push_back({dtype, device, placement, contiguous});
                }
            }
        }
    }
    return params;
}

INSTANTIATE_TEST_SUITE_P(AllConfigurations,
                         ReluTest,
                         ::testing::ValuesIn(generate_all_tests()),
                         test_name_generator);

// =============================================================================
// Edge case tests
// =============================================================================

class ReluEdgeCases : public ::testing::Test {};

TEST_F(ReluEdgeCases, EmptyTensor_OutOfPlace) {
    auto tensor = torch::empty({0}, torch::kFloat32);

    auto result = example_relu(tensor);

    EXPECT_EQ(result.size(0), 0);
    EXPECT_FALSE(result.is_same(tensor));
}

TEST_F(ReluEdgeCases, EmptyTensor_InPlace) {
    auto tensor = torch::empty({0}, torch::kFloat32);

    auto result = example_relu_(tensor);

    EXPECT_EQ(result.size(0), 0);
    EXPECT_TRUE(result.is_same(tensor));
}

TEST_F(ReluEdgeCases, SingleElement_Positive) {
    auto tensor = torch::tensor({5.0f});

    auto result = example_relu(tensor);

    EXPECT_EQ(result.size(0), 1);
    EXPECT_FLOAT_EQ(result[0].item<float>(), 5.0f);
}

TEST_F(ReluEdgeCases, SingleElement_Negative) {
    auto tensor = torch::tensor({-5.0f});

    auto result = example_relu(tensor);

    EXPECT_EQ(result.size(0), 1);
    EXPECT_FLOAT_EQ(result[0].item<float>(), 0.0f);
}

TEST_F(ReluEdgeCases, SingleElement_Zero) {
    auto tensor = torch::tensor({0.0f});

    auto result = example_relu(tensor);

    EXPECT_EQ(result.size(0), 1);
    EXPECT_FLOAT_EQ(result[0].item<float>(), 0.0f);
}

TEST_F(ReluEdgeCases, AllNegative) {
    auto tensor = torch::tensor({-1.0f, -2.0f, -3.0f, -100.0f});

    auto result = example_relu(tensor);

    for (int64_t i = 0; i < result.size(0); ++i) {
        EXPECT_FLOAT_EQ(result[i].item<float>(), 0.0f) << "at index " << i;
    }
}

TEST_F(ReluEdgeCases, AllPositive) {
    auto tensor   = torch::tensor({1.0f, 2.0f, 3.0f, 100.0f});
    auto original = tensor.clone();

    auto result = example_relu(tensor);

    EXPECT_TRUE(torch::equal(result, original));
}

TEST_F(ReluEdgeCases, LargeInput_CPU) {
    int64_t const n = 100000;
    auto tensor     = make_relu_test_tensor(n, torch::kFloat32);
    auto expected   = compute_expected_relu(tensor);

    auto result = example_relu(tensor);

    expect_relu_equal(result, expected);
}

TEST_F(ReluEdgeCases, LargeInput_CUDA) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    int64_t const n = 100000;
    auto tensor     = make_relu_test_tensor(n, torch::kFloat32, torch::kCUDA);
    auto expected   = compute_expected_relu(tensor);

    auto result = example_relu(tensor);

    expect_relu_equal(result, expected);
}

// =============================================================================
// Half precision specific tests
// =============================================================================

class ReluHalfPrecision : public ::testing::Test {};

TEST_F(ReluHalfPrecision, Float16_SpecialValues) {
    // Test with values that exercise float16 range
    auto options = torch::TensorOptions().dtype(torch::kFloat16);
    auto tensor  = torch::tensor({-1.0f, -0.5f, 0.0f, 0.5f, 1.0f}, options);

    auto result = example_relu(tensor);

    auto result_float = result.to(torch::kFloat32);
    EXPECT_NEAR(result_float[0].item<float>(), 0.0f, 1e-3f);
    EXPECT_NEAR(result_float[1].item<float>(), 0.0f, 1e-3f);
    EXPECT_NEAR(result_float[2].item<float>(), 0.0f, 1e-3f);
    EXPECT_NEAR(result_float[3].item<float>(), 0.5f, 1e-3f);
    EXPECT_NEAR(result_float[4].item<float>(), 1.0f, 1e-3f);
}

TEST_F(ReluHalfPrecision, BFloat16_SpecialValues) {
    auto options = torch::TensorOptions().dtype(torch::kBFloat16);
    auto tensor  = torch::tensor({-1.0f, -0.5f, 0.0f, 0.5f, 1.0f}, options);

    auto result = example_relu(tensor);

    auto result_float = result.to(torch::kFloat32);
    EXPECT_NEAR(result_float[0].item<float>(), 0.0f, 1e-2f);
    EXPECT_NEAR(result_float[1].item<float>(), 0.0f, 1e-2f);
    EXPECT_NEAR(result_float[2].item<float>(), 0.0f, 1e-2f);
    EXPECT_NEAR(result_float[3].item<float>(), 0.5f, 1e-2f);
    EXPECT_NEAR(result_float[4].item<float>(), 1.0f, 1e-2f);
}

TEST_F(ReluHalfPrecision, Float16_CUDA) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
    auto tensor  = torch::tensor({-2.0f, -1.0f, 0.0f, 1.0f, 2.0f}, options);

    auto result = example_relu(tensor);

    auto result_float = result.cpu().to(torch::kFloat32);
    EXPECT_NEAR(result_float[0].item<float>(), 0.0f, 1e-3f);
    EXPECT_NEAR(result_float[1].item<float>(), 0.0f, 1e-3f);
    EXPECT_NEAR(result_float[2].item<float>(), 0.0f, 1e-3f);
    EXPECT_NEAR(result_float[3].item<float>(), 1.0f, 1e-3f);
    EXPECT_NEAR(result_float[4].item<float>(), 2.0f, 1e-3f);
}

TEST_F(ReluHalfPrecision, BFloat16_CUDA) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto options = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
    auto tensor  = torch::tensor({-2.0f, -1.0f, 0.0f, 1.0f, 2.0f}, options);

    auto result = example_relu(tensor);

    auto result_float = result.cpu().to(torch::kFloat32);
    EXPECT_NEAR(result_float[0].item<float>(), 0.0f, 1e-2f);
    EXPECT_NEAR(result_float[1].item<float>(), 0.0f, 1e-2f);
    EXPECT_NEAR(result_float[2].item<float>(), 0.0f, 1e-2f);
    EXPECT_NEAR(result_float[3].item<float>(), 1.0f, 1e-2f);
    EXPECT_NEAR(result_float[4].item<float>(), 2.0f, 1e-2f);
}

} // namespace
} // namespace dispatch_examples
