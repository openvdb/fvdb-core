// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Tests for ternary affine transformation: y = R @ x + T
//
// Test coverage:
//   - Basic correctness with fully specified tensors
//   - Broadcast on R (same rotation for all points)
//   - Broadcast on T (same translation for all points)
//   - Views from packed storage (strided, non-contiguous)
//   - CPU and CUDA devices
//   - All float types (float16, bfloat16, float32, float64)
//

#include "examples/affine_xform_ternary.h"
#include "test_utils.h"

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <vector>

namespace dispatch_examples {
namespace {

using dispatch::test::skip_if_no_cuda;

//==============================================================================
// Test utilities
//==============================================================================

/// Create a rotation matrix around the Z axis
inline torch::Tensor
make_rotation_z(float angle_rad, torch::ScalarType dtype, torch::Device device) {
    float const c = std::cos(angle_rad);
    float const s = std::sin(angle_rad);

    // clang-format off
    auto R = torch::tensor({
        { c, -s, 0.0f},
        { s,  c, 0.0f},
        {0.0f, 0.0f, 1.0f}
    }, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    // clang-format on

    return R.to(dtype);
}

/// Create N different rotation matrices (rotations around Z by varying angles)
inline torch::Tensor
make_rotation_matrices(int64_t N, torch::ScalarType dtype, torch::Device device) {
    std::vector<torch::Tensor> rotations;
    rotations.reserve(N);

    for (int64_t i = 0; i < N; ++i) {
        float const angle = static_cast<float>(i) * 0.1f; // varying angles
        rotations.push_back(make_rotation_z(angle, dtype, device));
    }

    return torch::stack(rotations, 0); // (N, 3, 3)
}

/// Create test positions
inline torch::Tensor
make_positions(int64_t N, torch::ScalarType dtype, torch::Device device) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    // Create positions like (1, 0, 0), (2, 0, 0), ...
    auto x         = torch::zeros({N, 3}, options);
    x.select(1, 0) = torch::arange(1, N + 1, options);
    return x.to(dtype);
}

/// Create test translations
inline torch::Tensor
make_translations(int64_t N, torch::ScalarType dtype, torch::Device device) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto T       = torch::zeros({N, 3}, options);
    // T[i] = (0, i, 0)
    T.select(1, 1) = torch::arange(0, N, options);
    return T.to(dtype);
}

/// Compute expected result using PyTorch operations
inline torch::Tensor
compute_expected(torch::Tensor R, torch::Tensor T, torch::Tensor x) {
    // Use float32 for computation to avoid issues with half types
    auto R_float = R.to(torch::kFloat32);
    auto T_float = T.to(torch::kFloat32);
    auto x_float = x.to(torch::kFloat32);

    int64_t const N = x_float.size(0);

    // Handle broadcast
    if (R_float.size(0) == 1 && N > 1) {
        R_float = R_float.expand({N, 3, 3});
    }
    if (T_float.size(0) == 1 && N > 1) {
        T_float = T_float.expand({N, 3});
    }

    // y = R @ x + T
    // x has shape (N, 3), need to make it (N, 3, 1) for bmm
    auto x_col  = x_float.unsqueeze(2);       // (N, 3, 1)
    auto Rx     = torch::bmm(R_float, x_col); // (N, 3, 1)
    auto result = Rx.squeeze(2) + T_float;    // (N, 3)

    return result.to(x.scalar_type());
}

/// Compare results with tolerance
inline void
expect_affine_equal(torch::Tensor actual, torch::Tensor expected, double tol = 1e-5) {
    ASSERT_EQ(actual.sizes(), expected.sizes());

    auto actual_float   = actual.cpu().to(torch::kFloat32).contiguous();
    auto expected_float = expected.cpu().to(torch::kFloat32).contiguous();

    if (actual.dtype() == torch::kFloat16) {
        tol = 1e-2; // Larger tolerance for half precision
    } else if (actual.dtype() == torch::kBFloat16) {
        tol = 0.1;  // BFloat16 has only 7 mantissa bits, needs larger tolerance
    }

    auto actual_ptr   = actual_float.data_ptr<float>();
    auto expected_ptr = expected_float.data_ptr<float>();

    for (int64_t i = 0; i < actual_float.numel(); ++i) {
        EXPECT_NEAR(actual_ptr[i], expected_ptr[i], tol) << "at index " << i;
    }
}

//==============================================================================
// Parameterized test base
//==============================================================================

struct AffineXformTestParams {
    torch::ScalarType dtype;
    torch::Device device;
    bool R_broadcast; // R is (1, 3, 3) instead of (N, 3, 3)
    bool T_broadcast; // T is (1, 3) instead of (N, 3)
    bool strided;     // Use strided (non-contiguous) tensors
};

class AffineXformTest : public ::testing::TestWithParam<AffineXformTestParams> {
  protected:
    void
    SetUp() override {
        auto params = GetParam();
        if (params.device.is_cuda() && !torch::cuda::is_available()) {
            GTEST_SKIP() << "CUDA not available";
        }
    }
};

TEST_P(AffineXformTest, Correctness) {
    auto params     = GetParam();
    int64_t const N = 10;

    // Create test tensors
    auto x = make_positions(N, params.dtype, params.device);
    auto R = params.R_broadcast ? make_rotation_z(0.5f, params.dtype, params.device).unsqueeze(0)
                                : make_rotation_matrices(N, params.dtype, params.device);
    auto T = params.T_broadcast
                 ? torch::tensor({1.0f, 2.0f, 3.0f},
                                 torch::TensorOptions().dtype(params.dtype).device(params.device))
                       .unsqueeze(0)
                 : make_translations(N, params.dtype, params.device);

    // Make strided if requested
    if (params.strided) {
        // Create strided versions by slicing on dimension 0
        // Note: For broadcast tensors (size 1 on dim 0), we make inner dims strided instead

        if (!params.R_broadcast) {
            // R has size N on dim 0, can stride there
            auto R_full = torch::zeros({R.size(0) * 2, 3, 3}, R.options());
            R_full.slice(0, 0, R.size(0) * 2, 2).copy_(R);
            R = R_full.slice(0, 0, R.size(0) * 2, 2);
        } else {
            // R has size 1 on dim 0, stride on dim 1 instead
            auto R_full = torch::zeros({1, 6, 3}, R.options());
            R_full.slice(1, 0, 6, 2).copy_(R);
            R = R_full.slice(1, 0, 6, 2);
        }

        if (!params.T_broadcast) {
            // T has size N on dim 0, can stride there
            auto T_full = torch::zeros({T.size(0) * 2, 3}, T.options());
            T_full.slice(0, 0, T.size(0) * 2, 2).copy_(T);
            T = T_full.slice(0, 0, T.size(0) * 2, 2);
        } else {
            // T has size 1 on dim 0, stride on dim 1 instead
            auto T_full = torch::zeros({1, 6}, T.options());
            T_full.slice(1, 0, 6, 2).copy_(T);
            T = T_full.slice(1, 0, 6, 2);
        }

        // x always has size N on dim 0
        auto x_full = torch::zeros({N * 2, 3}, x.options());
        x_full.slice(0, 0, N * 2, 2).copy_(x);
        x = x_full.slice(0, 0, N * 2, 2);

        ASSERT_FALSE(R.is_contiguous()) << "R should be strided";
        ASSERT_FALSE(T.is_contiguous()) << "T should be strided";
        ASSERT_FALSE(x.is_contiguous()) << "x should be strided";
    }

    // Compute expected
    auto expected = compute_expected(R, T, x);

    // Call our implementation
    auto result = example_affine_xform(R, T, x);

    // Verify
    EXPECT_EQ(result.sizes(), expected.sizes());
    EXPECT_EQ(result.device().type(), params.device.type());
    EXPECT_EQ(result.scalar_type(), params.dtype);

    expect_affine_equal(result, expected);
}

//==============================================================================
// Generate test parameter combinations
//==============================================================================

std::string
test_name_generator(::testing::TestParamInfo<AffineXformTestParams> const &info) {
    auto const &p = info.param;
    std::string name;
    name += (p.device.is_cuda() ? "CUDA_" : "CPU_");
    name += c10::toString(p.dtype);
    name += (p.R_broadcast ? "_Rbroadcast" : "_Rfull");
    name += (p.T_broadcast ? "_Tbroadcast" : "_Tfull");
    name += (p.strided ? "_Strided" : "_Contiguous");
    return name;
}

std::vector<AffineXformTestParams>
generate_all_tests() {
    std::vector<AffineXformTestParams> params;

    torch::ScalarType dtypes[] = {
        torch::kFloat32, torch::kFloat64, torch::kFloat16, torch::kBFloat16};
    torch::Device devices[] = {torch::kCPU, torch::kCUDA};

    for (auto device: devices) {
        for (auto dtype: dtypes) {
            for (bool R_broadcast: {false, true}) {
                for (bool T_broadcast: {false, true}) {
                    for (bool strided: {false, true}) {
                        params.push_back({dtype, device, R_broadcast, T_broadcast, strided});
                    }
                }
            }
        }
    }
    return params;
}

INSTANTIATE_TEST_SUITE_P(AllConfigurations,
                         AffineXformTest,
                         ::testing::ValuesIn(generate_all_tests()),
                         test_name_generator);

//==============================================================================
// Edge case tests
//==============================================================================

class AffineXformEdgeCases : public ::testing::Test {};

TEST_F(AffineXformEdgeCases, EmptyTensor) {
    auto R = torch::zeros({0, 3, 3}, torch::kFloat32);
    auto T = torch::zeros({0, 3}, torch::kFloat32);
    auto x = torch::zeros({0, 3}, torch::kFloat32);

    auto result = example_affine_xform(R, T, x);

    EXPECT_EQ(result.size(0), 0);
    EXPECT_EQ(result.size(1), 3);
}

TEST_F(AffineXformEdgeCases, SinglePoint) {
    // Identity rotation, zero translation
    auto R = torch::eye(3, torch::kFloat32).unsqueeze(0); // (1, 3, 3)
    auto T = torch::zeros({1, 3}, torch::kFloat32);
    auto x = torch::tensor({{1.0f, 2.0f, 3.0f}});         // (1, 3)

    auto result = example_affine_xform(R, T, x);

    EXPECT_EQ(result.size(0), 1);
    EXPECT_FLOAT_EQ(result[0][0].item<float>(), 1.0f);
    EXPECT_FLOAT_EQ(result[0][1].item<float>(), 2.0f);
    EXPECT_FLOAT_EQ(result[0][2].item<float>(), 3.0f);
}

TEST_F(AffineXformEdgeCases, IdentityTransform) {
    int64_t const N = 5;
    auto R          = torch::eye(3, torch::kFloat32).unsqueeze(0).expand({N, 3, 3}).contiguous();
    auto T          = torch::zeros({N, 3}, torch::kFloat32);
    auto x          = torch::randn({N, 3}, torch::kFloat32);

    auto result = example_affine_xform(R, T, x);

    expect_affine_equal(result, x);
}

TEST_F(AffineXformEdgeCases, TranslationOnly) {
    int64_t const N = 5;
    auto R          = torch::eye(3, torch::kFloat32).unsqueeze(0); // (1, 3, 3) broadcast
    auto T          = torch::tensor({{10.0f, 20.0f, 30.0f}});      // (1, 3) broadcast
    auto x          = torch::zeros({N, 3}, torch::kFloat32);

    auto result   = example_affine_xform(R, T, x);
    auto expected = T.expand({N, 3});

    expect_affine_equal(result, expected);
}

TEST_F(AffineXformEdgeCases, Rotation90DegreesZ) {
    // 90 degree rotation around Z axis: (1, 0, 0) -> (0, 1, 0)
    float const angle = std::numbers::pi_v<float> / 2.0f;
    auto R            = make_rotation_z(angle, torch::kFloat32, torch::kCPU).unsqueeze(0);
    auto T            = torch::zeros({1, 3}, torch::kFloat32);
    auto x            = torch::tensor({{1.0f, 0.0f, 0.0f}});

    auto result = example_affine_xform(R, T, x);

    EXPECT_NEAR(result[0][0].item<float>(), 0.0f, 1e-5f);
    EXPECT_NEAR(result[0][1].item<float>(), 1.0f, 1e-5f);
    EXPECT_NEAR(result[0][2].item<float>(), 0.0f, 1e-5f);
}

TEST_F(AffineXformEdgeCases, CUDA_SinglePoint) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto R = torch::eye(3, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
                 .unsqueeze(0);
    auto T = torch::tensor({{1.0f, 2.0f, 3.0f}},
                           torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto x = torch::tensor({{10.0f, 20.0f, 30.0f}},
                           torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    auto result = example_affine_xform(R, T, x);

    auto result_cpu = result.cpu();
    EXPECT_FLOAT_EQ(result_cpu[0][0].item<float>(), 11.0f);
    EXPECT_FLOAT_EQ(result_cpu[0][1].item<float>(), 22.0f);
    EXPECT_FLOAT_EQ(result_cpu[0][2].item<float>(), 33.0f);
}

//==============================================================================
// Views from packed storage test
//==============================================================================

class AffineXformPackedStorageTest : public ::testing::Test {};

TEST_F(AffineXformPackedStorageTest, ViewsFromPacked_CPU) {
    // Create (N, 12) packed storage containing R (9 values) and T (3 values)
    int64_t const N = 5;
    auto packed     = torch::randn({N, 12}, torch::kFloat32);

    // Create views: R from first 9 columns, T from last 3
    // Note: these are strided views!
    auto R_flat = packed.narrow(1, 0, 9); // (N, 9)
    auto R      = R_flat.view({N, 3, 3}); // (N, 3, 3) - this creates a view
    auto T      = packed.narrow(1, 9, 3); // (N, 3)
    auto x      = torch::randn({N, 3}, torch::kFloat32);

    // Verify they're views (non-contiguous or sharing storage)
    ASSERT_EQ(R.data_ptr<float>(), packed.data_ptr<float>());
    ASSERT_EQ(T.data_ptr<float>(), packed.data_ptr<float>() + 9);

    // Compute expected
    auto expected = compute_expected(R, T, x);

    // Our implementation should handle these views correctly
    auto result = example_affine_xform(R, T, x);

    expect_affine_equal(result, expected);
}

TEST_F(AffineXformPackedStorageTest, ViewsFromPacked_CUDA) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    int64_t const N = 5;
    auto options    = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto packed     = torch::randn({N, 12}, options);

    auto R_flat = packed.narrow(1, 0, 9);
    auto R      = R_flat.view({N, 3, 3});
    auto T      = packed.narrow(1, 9, 3);
    auto x      = torch::randn({N, 3}, options);

    auto expected = compute_expected(R, T, x);
    auto result   = example_affine_xform(R, T, x);

    expect_affine_equal(result, expected);
}

//==============================================================================
// Validation tests (expected failures)
//==============================================================================

class AffineXformValidation : public ::testing::Test {};

TEST_F(AffineXformValidation, WrongRRank) {
    auto R = torch::randn({5, 3}, torch::kFloat32); // 2D instead of 3D
    auto T = torch::randn({5, 3}, torch::kFloat32);
    auto x = torch::randn({5, 3}, torch::kFloat32);

    EXPECT_THROW(example_affine_xform(R, T, x), c10::Error);
}

TEST_F(AffineXformValidation, WrongRElementShape) {
    auto R = torch::randn({5, 3, 4}, torch::kFloat32); // (3, 4) instead of (3, 3)
    auto T = torch::randn({5, 3}, torch::kFloat32);
    auto x = torch::randn({5, 3}, torch::kFloat32);

    EXPECT_THROW(example_affine_xform(R, T, x), c10::Error);
}

TEST_F(AffineXformValidation, IncompatibleIterationDim) {
    auto R = torch::randn({5, 3, 3}, torch::kFloat32);
    auto T = torch::randn({3, 3}, torch::kFloat32); // 3 instead of 5 or 1
    auto x = torch::randn({5, 3}, torch::kFloat32);

    EXPECT_THROW(example_affine_xform(R, T, x), c10::Error);
}

TEST_F(AffineXformValidation, MixedDtypes) {
    auto R = torch::randn({5, 3, 3}, torch::kFloat32);
    auto T = torch::randn({5, 3}, torch::kFloat64); // Different dtype
    auto x = torch::randn({5, 3}, torch::kFloat32);

    EXPECT_THROW(example_affine_xform(R, T, x), c10::Error);
}

TEST_F(AffineXformValidation, MixedDevices) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto R = torch::randn({5, 3, 3}, torch::kFloat32);
    auto T =
        torch::randn({5, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto x = torch::randn({5, 3}, torch::kFloat32);

    EXPECT_THROW(example_affine_xform(R, T, x), c10::Error);
}

} // namespace
} // namespace dispatch_examples
