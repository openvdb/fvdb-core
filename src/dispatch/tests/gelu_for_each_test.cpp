// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include "examples/gelu_for_each.h"
#include "test_utils.h"

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

#include <dispatch/types.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace dispatch_examples {
namespace {

using dispatch::test::skip_if_no_cuda;

// =============================================================================
// Test utilities
// =============================================================================

/// Compare GELU results with appropriate tolerance
inline void
expect_gelu_equal(torch::Tensor actual, torch::Tensor expected, double tol = 1e-5) {
    ASSERT_EQ(actual.sizes(), expected.sizes());

    // Convert to float32 on CPU for comparison (handles half types correctly)
    auto actual_float   = actual.cpu().to(torch::kFloat32).contiguous();
    auto expected_float = expected.cpu().to(torch::kFloat32).contiguous();

    if (actual.dtype() == torch::kFloat16 || actual.dtype() == torch::kBFloat16) {
        // Half precision needs larger tolerance
        tol = 1e-2;
    }

    auto actual_ptr   = actual_float.data_ptr<float>();
    auto expected_ptr = expected_float.data_ptr<float>();
    for (int64_t i = 0; i < actual_float.numel(); ++i) {
        EXPECT_NEAR(actual_ptr[i], expected_ptr[i], tol) << "at index " << i;
    }
}

/// Create a 1D tensor with mixed positive and negative values: [-2, -1, 0, 1, 2, ...]
/// For testing GELU, we want a range of values including negative, zero, and positive.
inline torch::Tensor
make_gelu_test_tensor(int64_t n, torch::ScalarType dtype, torch::Device device = torch::kCPU) {
    // Create [-n/2, ..., -1, 0, 1, ..., n/2] style values
    // Use float computation then convert to target dtype
    auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto values        = torch::arange(0, n, float_options) - static_cast<float>(n / 2);
    return values.to(dtype);
}

/// Create a strided (non-contiguous) tensor with mixed values for GELU testing
inline torch::Tensor
make_strided_gelu_test_tensor(int64_t n,
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

/// Compute expected GELU output using PyTorch's gelu function
inline torch::Tensor
compute_expected_gelu(torch::Tensor input) {
    // Use float for computation to avoid issues with half types
    auto float_input = input.to(torch::kFloat32);
    auto result      = torch::gelu(float_input);
    return result.to(input.scalar_type());
}

// =============================================================================
// Parameterized test base for GELU
// =============================================================================

struct GeluForEachTestParams {
    torch::ScalarType dtype;
    torch::Device device;
    dispatch::placement placement;
    bool contiguous;
};

class GeluForEachTest : public ::testing::TestWithParam<GeluForEachTestParams> {
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
            return make_gelu_test_tensor(n, params.dtype, params.device);
        } else {
            return make_strided_gelu_test_tensor(n, params.dtype, params.device);
        }
    }

    // Call the appropriate GELU function based on placement
    torch::Tensor
    call_gelu(torch::Tensor input) {
        auto params = GetParam();
        if (params.placement == dispatch::placement::in_place) {
            return example_gelu_for_each_(input);
        } else {
            return example_gelu_for_each(input);
        }
    }
};

// =============================================================================
// Test implementation shared across all parameterized tests
// =============================================================================

TEST_P(GeluForEachTest, CorrectnessAndPlacement) {
    auto params        = GetParam();
    int64_t const n    = 100;
    auto input         = create_input_tensor(n);
    auto input_clone   = input.clone(); // Save original for out-of-place verification
    auto expected      = compute_expected_gelu(input);
    void *original_ptr = input.data_ptr();
    bool is_contiguous = input.is_contiguous();

    // Build context string for error messages
    std::string context;
    context += (params.device.is_cuda() ? "CUDA_" : "CPU_");
    context += c10::toString(params.dtype);
    context += (params.placement == dispatch::placement::in_place ? "_InPlace" : "_OutOfPlace");
    context += (params.contiguous ? "_Contiguous" : "_Strided");

    ASSERT_EQ(is_contiguous, params.contiguous) << "Test setup error: contiguity mismatch";

    auto result = call_gelu(input);

    // Verify correctness
    expect_gelu_equal(result, expected);

    // Verify placement semantics
    if (params.placement == dispatch::placement::in_place) {
        EXPECT_EQ(result.data_ptr(), original_ptr)
            << context << ": InPlace should return same memory";
        // Input should be modified
        expect_gelu_equal(input, expected);
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
test_name_generator(::testing::TestParamInfo<GeluForEachTestParams> const &info) {
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

std::vector<GeluForEachTestParams>
generate_all_tests() {
    std::vector<GeluForEachTestParams> params;

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
                         GeluForEachTest,
                         ::testing::ValuesIn(generate_all_tests()),
                         test_name_generator);

// =============================================================================
// Edge case tests
// =============================================================================

class GeluForEachEdgeCases : public ::testing::Test {};

TEST_F(GeluForEachEdgeCases, EmptyTensor_OutOfPlace) {
    auto tensor = torch::empty({0}, torch::kFloat32);

    auto result = example_gelu_for_each(tensor);

    EXPECT_EQ(result.size(0), 0);
    EXPECT_FALSE(result.is_same(tensor));
}

TEST_F(GeluForEachEdgeCases, EmptyTensor_InPlace) {
    auto tensor = torch::empty({0}, torch::kFloat32);

    auto result = example_gelu_for_each_(tensor);

    EXPECT_EQ(result.size(0), 0);
    EXPECT_TRUE(result.is_same(tensor));
}

TEST_F(GeluForEachEdgeCases, SingleElement_Positive) {
    auto tensor   = torch::tensor({2.0f});
    auto expected = torch::gelu(tensor);

    auto result = example_gelu_for_each(tensor);

    EXPECT_EQ(result.size(0), 1);
    EXPECT_NEAR(result[0].item<float>(), expected[0].item<float>(), 1e-5f);
}

TEST_F(GeluForEachEdgeCases, SingleElement_Negative) {
    auto tensor   = torch::tensor({-2.0f});
    auto expected = torch::gelu(tensor);

    auto result = example_gelu_for_each(tensor);

    EXPECT_EQ(result.size(0), 1);
    EXPECT_NEAR(result[0].item<float>(), expected[0].item<float>(), 1e-5f);
}

TEST_F(GeluForEachEdgeCases, SingleElement_Zero) {
    auto tensor = torch::tensor({0.0f});

    auto result = example_gelu_for_each(tensor);

    EXPECT_EQ(result.size(0), 1);
    EXPECT_FLOAT_EQ(result[0].item<float>(), 0.0f); // GELU(0) = 0
}

TEST_F(GeluForEachEdgeCases, AllNegative) {
    auto tensor   = torch::tensor({-1.0f, -2.0f, -3.0f, -4.0f});
    auto expected = torch::gelu(tensor);

    auto result = example_gelu_for_each(tensor);

    for (int64_t i = 0; i < result.size(0); ++i) {
        EXPECT_NEAR(result[i].item<float>(), expected[i].item<float>(), 1e-5f) << "at index " << i;
    }
}

TEST_F(GeluForEachEdgeCases, AllPositive) {
    auto tensor   = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f});
    auto expected = torch::gelu(tensor);

    auto result = example_gelu_for_each(tensor);

    for (int64_t i = 0; i < result.size(0); ++i) {
        EXPECT_NEAR(result[i].item<float>(), expected[i].item<float>(), 1e-5f) << "at index " << i;
    }
}

TEST_F(GeluForEachEdgeCases, LargeInput_CPU) {
    int64_t const n = 100000;
    auto tensor     = make_gelu_test_tensor(n, torch::kFloat32);
    auto expected   = compute_expected_gelu(tensor);

    auto result = example_gelu_for_each(tensor);

    expect_gelu_equal(result, expected);
}

TEST_F(GeluForEachEdgeCases, LargeInput_CUDA) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    int64_t const n = 100000;
    auto tensor     = make_gelu_test_tensor(n, torch::kFloat32, torch::kCUDA);
    auto expected   = compute_expected_gelu(tensor);

    auto result = example_gelu_for_each(tensor);

    expect_gelu_equal(result, expected);
}

// =============================================================================
// Half precision specific tests
// =============================================================================

class GeluForEachHalfPrecision : public ::testing::Test {};

TEST_F(GeluForEachHalfPrecision, Float16_SpecialValues) {
    // Test with values that exercise float16 range
    auto options  = torch::TensorOptions().dtype(torch::kFloat16);
    auto tensor   = torch::tensor({-2.0f, -1.0f, 0.0f, 1.0f, 2.0f}, options);
    auto expected = torch::gelu(tensor.to(torch::kFloat32)).to(torch::kFloat16);

    auto result = example_gelu_for_each(tensor);

    auto result_float   = result.to(torch::kFloat32);
    auto expected_float = expected.to(torch::kFloat32);
    for (int64_t i = 0; i < result.size(0); ++i) {
        EXPECT_NEAR(result_float[i].item<float>(), expected_float[i].item<float>(), 1e-2f)
            << "at index " << i;
    }
}

TEST_F(GeluForEachHalfPrecision, BFloat16_SpecialValues) {
    auto options  = torch::TensorOptions().dtype(torch::kBFloat16);
    auto tensor   = torch::tensor({-2.0f, -1.0f, 0.0f, 1.0f, 2.0f}, options);
    auto expected = torch::gelu(tensor.to(torch::kFloat32)).to(torch::kBFloat16);

    auto result = example_gelu_for_each(tensor);

    auto result_float   = result.to(torch::kFloat32);
    auto expected_float = expected.to(torch::kFloat32);
    for (int64_t i = 0; i < result.size(0); ++i) {
        EXPECT_NEAR(result_float[i].item<float>(), expected_float[i].item<float>(), 1e-2f)
            << "at index " << i;
    }
}

TEST_F(GeluForEachHalfPrecision, Float16_CUDA) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto options  = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
    auto tensor   = torch::tensor({-2.0f, -1.0f, 0.0f, 1.0f, 2.0f}, options);
    auto expected = torch::gelu(tensor.to(torch::kFloat32)).to(torch::kFloat16);

    auto result = example_gelu_for_each(tensor);

    auto result_float   = result.cpu().to(torch::kFloat32);
    auto expected_float = expected.cpu().to(torch::kFloat32);
    for (int64_t i = 0; i < result.size(0); ++i) {
        EXPECT_NEAR(result_float[i].item<float>(), expected_float[i].item<float>(), 1e-2f)
            << "at index " << i;
    }
}

TEST_F(GeluForEachHalfPrecision, BFloat16_CUDA) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto options  = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
    auto tensor   = torch::tensor({-2.0f, -1.0f, 0.0f, 1.0f, 2.0f}, options);
    auto expected = torch::gelu(tensor.to(torch::kFloat32)).to(torch::kBFloat16);

    auto result = example_gelu_for_each(tensor);

    auto result_float   = result.cpu().to(torch::kFloat32);
    auto expected_float = expected.cpu().to(torch::kFloat32);
    for (int64_t i = 0; i < result.size(0); ++i) {
        EXPECT_NEAR(result_float[i].item<float>(), expected_float[i].item<float>(), 1e-2f)
            << "at index " << i;
    }
}

} // namespace
} // namespace dispatch_examples
