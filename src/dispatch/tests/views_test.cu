// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Tests for dispatch view types: flat_view, vector_view, matrix_view.
//

// Suppress nvcc warning about test_info_ member in GoogleTest macros
#ifdef __NVCC__
#pragma nv_diag_suppress 177
#endif

#include "dispatch/torch/for_each.h"
#include "dispatch/torch/views.h"
#include "test_utils.h"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace dispatch {
namespace {

using test::skip_if_no_cuda;

// =============================================================================
// Reference GELU implementation for testing
// =============================================================================

template <typename T>
T
reference_gelu(T x) {
    // GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    constexpr T sqrt2_inv = T{0.7071067811865476};
    return x * T{0.5} * (T{1} + std::erf(x * sqrt2_inv));
}

// =============================================================================
// CUDA helper functions (device lambdas must be in free functions)
// =============================================================================

using CudaFloatTag  = tag<torch::kCUDA, torch::kFloat32>;
using CudaDoubleTag = tag<torch::kCUDA, torch::kFloat64>;

void
cuda_flat_view_gelu(torch::Tensor input, torch::Tensor output) {
    flat_const_view<torch::kCUDA, torch::kFloat32> in{input};
    flat_mutable_view<torch::kCUDA, torch::kFloat32> out{output};

    for_each(CudaFloatTag{}, in.numel, [=] __device__(CudaFloatTag, int64_t i) {
        float const x        = in[i];
        float const sqrt2inv = 0.7071067811865476f;
        out[i]               = x * 0.5f * (1.0f + erff(x * sqrt2inv));
    });
}

void
cuda_flat_view_double(torch::Tensor input, torch::Tensor output) {
    flat_const_view<torch::kCUDA, torch::kFloat64> in{input};
    flat_mutable_view<torch::kCUDA, torch::kFloat64> out{output};

    for_each(CudaDoubleTag{}, in.numel, [=] __device__(CudaDoubleTag, int64_t i) {
        double const x        = in[i];
        double const sqrt2inv = 0.7071067811865476;
        out[i]                = x * 0.5 * (1.0 + erf(x * sqrt2inv));
    });
}

// =============================================================================
// flat_view tests
// =============================================================================

class FlatViewTest : public ::testing::Test {};

TEST_F(FlatViewTest, CPU_1D_Contiguous) {
    using Tag       = tag<torch::kCPU, torch::kFloat32>;
    int64_t const n = 100;
    auto input      = torch::randn({n}, torch::kFloat32);
    auto output     = torch::zeros({n}, torch::kFloat32);

    flat_const_view<torch::kCPU, torch::kFloat32> in{input};
    flat_mutable_view<torch::kCPU, torch::kFloat32> out{output};

    EXPECT_EQ(in.numel, n);
    EXPECT_EQ(in.rank, 1);
    EXPECT_TRUE(in.is_contiguous);

    for_each(Tag{}, in.numel, [=](Tag, int64_t i) { out[i] = reference_gelu(in[i]); });

    auto expected = torch::nn::functional::gelu(input);
    for (int64_t i = 0; i < n; ++i) {
        EXPECT_NEAR(output[i].item<float>(), expected[i].item<float>(), 1e-5f) << "at index " << i;
    }
}

TEST_F(FlatViewTest, CPU_2D_Contiguous) {
    using Tag           = tag<torch::kCPU, torch::kFloat32>;
    int64_t const rows  = 10;
    int64_t const cols  = 20;
    int64_t const numel = rows * cols;
    auto input          = torch::randn({rows, cols}, torch::kFloat32);
    auto output         = torch::zeros({rows, cols}, torch::kFloat32);

    flat_const_view<torch::kCPU, torch::kFloat32> in{input};
    flat_mutable_view<torch::kCPU, torch::kFloat32> out{output};

    EXPECT_EQ(in.numel, numel);
    EXPECT_EQ(in.rank, 2);
    EXPECT_TRUE(in.is_contiguous);

    for_each(Tag{}, in.numel, [=](Tag, int64_t i) { out[i] = reference_gelu(in[i]); });

    auto expected = torch::nn::functional::gelu(input);
    auto out_flat = output.flatten();
    auto exp_flat = expected.flatten();
    for (int64_t i = 0; i < numel; ++i) {
        EXPECT_NEAR(out_flat[i].item<float>(), exp_flat[i].item<float>(), 1e-5f)
            << "at index " << i;
    }
}

TEST_F(FlatViewTest, CPU_3D_Contiguous) {
    using Tag           = tag<torch::kCPU, torch::kFloat32>;
    int64_t const d0    = 4;
    int64_t const d1    = 5;
    int64_t const d2    = 6;
    int64_t const numel = d0 * d1 * d2;
    auto input          = torch::randn({d0, d1, d2}, torch::kFloat32);
    auto output         = torch::zeros({d0, d1, d2}, torch::kFloat32);

    flat_const_view<torch::kCPU, torch::kFloat32> in{input};
    flat_mutable_view<torch::kCPU, torch::kFloat32> out{output};

    EXPECT_EQ(in.numel, numel);
    EXPECT_EQ(in.rank, 3);
    EXPECT_TRUE(in.is_contiguous);

    for_each(Tag{}, in.numel, [=](Tag, int64_t i) { out[i] = reference_gelu(in[i]); });

    auto expected = torch::nn::functional::gelu(input);
    auto out_flat = output.flatten();
    auto exp_flat = expected.flatten();
    for (int64_t i = 0; i < numel; ++i) {
        EXPECT_NEAR(out_flat[i].item<float>(), exp_flat[i].item<float>(), 1e-5f)
            << "at index " << i;
    }
}

TEST_F(FlatViewTest, CPU_4D_Contiguous) {
    using Tag           = tag<torch::kCPU, torch::kFloat32>;
    int64_t const d0    = 2;
    int64_t const d1    = 3;
    int64_t const d2    = 4;
    int64_t const d3    = 5;
    int64_t const numel = d0 * d1 * d2 * d3;
    auto input          = torch::randn({d0, d1, d2, d3}, torch::kFloat32);
    auto output         = torch::zeros({d0, d1, d2, d3}, torch::kFloat32);

    flat_const_view<torch::kCPU, torch::kFloat32> in{input};
    flat_mutable_view<torch::kCPU, torch::kFloat32> out{output};

    EXPECT_EQ(in.numel, numel);
    EXPECT_EQ(in.rank, 4);
    EXPECT_TRUE(in.is_contiguous);

    for_each(Tag{}, in.numel, [=](Tag, int64_t i) { out[i] = reference_gelu(in[i]); });

    auto expected = torch::nn::functional::gelu(input);
    auto out_flat = output.flatten();
    auto exp_flat = expected.flatten();
    for (int64_t i = 0; i < numel; ++i) {
        EXPECT_NEAR(out_flat[i].item<float>(), exp_flat[i].item<float>(), 1e-5f)
            << "at index " << i;
    }
}

TEST_F(FlatViewTest, CPU_2D_Transposed) {
    // Transposed tensor is non-contiguous
    using Tag          = tag<torch::kCPU, torch::kFloat32>;
    int64_t const rows = 10;
    int64_t const cols = 20;
    auto input_orig    = torch::randn({rows, cols}, torch::kFloat32);
    auto input         = input_orig.t(); // Now (cols x rows) with non-contiguous strides
    auto output_orig   = torch::zeros({rows, cols}, torch::kFloat32);
    auto output        = output_orig.t();

    EXPECT_FALSE(input.is_contiguous());

    flat_const_view<torch::kCPU, torch::kFloat32> in{input};
    flat_mutable_view<torch::kCPU, torch::kFloat32> out{output};

    EXPECT_EQ(in.numel, rows * cols);
    EXPECT_EQ(in.rank, 2);
    EXPECT_FALSE(in.is_contiguous);

    for_each(Tag{}, in.numel, [=](Tag, int64_t i) { out[i] = reference_gelu(in[i]); });

    // Compare with torch's gelu on the transposed tensor
    auto expected = torch::nn::functional::gelu(input);
    for (int64_t r = 0; r < cols; ++r) {
        for (int64_t c = 0; c < rows; ++c) {
            EXPECT_NEAR(output[r][c].item<float>(), expected[r][c].item<float>(), 1e-5f)
                << "at (" << r << ", " << c << ")";
        }
    }
}

TEST_F(FlatViewTest, CPU_1D_Strided) {
    // Create a strided 1D view via slicing
    using Tag        = tag<torch::kCPU, torch::kFloat32>;
    int64_t const n  = 50;
    auto full        = torch::randn({n * 2}, torch::kFloat32);
    auto input       = full.slice(0, 0, n * 2, 2); // Every other element, stride=2
    auto output_full = torch::zeros({n * 2}, torch::kFloat32);
    auto output      = output_full.slice(0, 0, n * 2, 2);

    EXPECT_FALSE(input.is_contiguous());
    EXPECT_EQ(input.stride(0), 2);

    flat_const_view<torch::kCPU, torch::kFloat32> in{input};
    flat_mutable_view<torch::kCPU, torch::kFloat32> out{output};

    EXPECT_EQ(in.numel, n);
    EXPECT_FALSE(in.is_contiguous);

    for_each(Tag{}, in.numel, [=](Tag, int64_t i) { out[i] = reference_gelu(in[i]); });

    auto expected = torch::nn::functional::gelu(input);
    for (int64_t i = 0; i < n; ++i) {
        EXPECT_NEAR(output[i].item<float>(), expected[i].item<float>(), 1e-5f) << "at index " << i;
    }
}

TEST_F(FlatViewTest, CPU_Scalar) {
    // 0-dimensional tensor (scalar)
    using Tag   = tag<torch::kCPU, torch::kFloat32>;
    auto input  = torch::tensor(2.5f);
    auto output = torch::zeros({});

    flat_const_view<torch::kCPU, torch::kFloat32> in{input};
    flat_mutable_view<torch::kCPU, torch::kFloat32> out{output};

    EXPECT_EQ(in.numel, 1);
    EXPECT_EQ(in.rank, 0);
    EXPECT_TRUE(in.is_contiguous);

    for_each(Tag{}, in.numel, [=](Tag, int64_t i) { out[i] = reference_gelu(in[i]); });

    auto expected = torch::nn::functional::gelu(input);
    EXPECT_NEAR(output.item<float>(), expected.item<float>(), 1e-5f);
}

TEST_F(FlatViewTest, CPU_Empty) {
    // Empty tensor
    using Tag   = tag<torch::kCPU, torch::kFloat32>;
    auto input  = torch::zeros({0}, torch::kFloat32);
    auto output = torch::zeros({0}, torch::kFloat32);

    flat_const_view<torch::kCPU, torch::kFloat32> in{input};
    flat_mutable_view<torch::kCPU, torch::kFloat32> out{output};

    EXPECT_EQ(in.numel, 0);

    // Should not crash
    for_each(
        Tag{}, in.numel, [=](Tag, int64_t) { FAIL() << "Should not iterate on empty tensor"; });
}

TEST_F(FlatViewTest, CPU_Double) {
    using Tag       = tag<torch::kCPU, torch::kFloat64>;
    int64_t const n = 100;
    auto input      = torch::randn({n}, torch::kFloat64);
    auto output     = torch::zeros({n}, torch::kFloat64);

    flat_const_view<torch::kCPU, torch::kFloat64> in{input};
    flat_mutable_view<torch::kCPU, torch::kFloat64> out{output};

    for_each(Tag{}, in.numel, [=](Tag, int64_t i) { out[i] = reference_gelu(in[i]); });

    auto expected = torch::nn::functional::gelu(input);
    for (int64_t i = 0; i < n; ++i) {
        EXPECT_NEAR(output[i].item<double>(), expected[i].item<double>(), 1e-10)
            << "at index " << i;
    }
}

TEST_F(FlatViewTest, CUDA_2D_Contiguous) {
    skip_if_no_cuda();

    int64_t const rows  = 10;
    int64_t const cols  = 20;
    int64_t const numel = rows * cols;
    auto input          = torch::randn({rows, cols},
                              torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto output         = torch::zeros({rows, cols},
                               torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    cuda_flat_view_gelu(input, output);

    auto expected   = torch::nn::functional::gelu(input);
    auto output_cpu = output.cpu().flatten();
    auto exp_cpu    = expected.cpu().flatten();
    for (int64_t i = 0; i < numel; ++i) {
        EXPECT_NEAR(output_cpu[i].item<float>(), exp_cpu[i].item<float>(), 1e-5f)
            << "at index " << i;
    }
}

TEST_F(FlatViewTest, CUDA_3D_Contiguous) {
    skip_if_no_cuda();

    int64_t const d0    = 4;
    int64_t const d1    = 5;
    int64_t const d2    = 6;
    int64_t const numel = d0 * d1 * d2;
    auto input          = torch::randn({d0, d1, d2},
                              torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto output         = torch::zeros({d0, d1, d2},
                               torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    cuda_flat_view_gelu(input, output);

    auto expected   = torch::nn::functional::gelu(input);
    auto output_cpu = output.cpu().flatten();
    auto exp_cpu    = expected.cpu().flatten();
    for (int64_t i = 0; i < numel; ++i) {
        EXPECT_NEAR(output_cpu[i].item<float>(), exp_cpu[i].item<float>(), 1e-5f)
            << "at index " << i;
    }
}

TEST_F(FlatViewTest, CUDA_Double) {
    skip_if_no_cuda();

    int64_t const n = 100;
    auto input =
        torch::randn({n}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA));
    auto output =
        torch::zeros({n}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA));

    cuda_flat_view_double(input, output);

    auto expected   = torch::nn::functional::gelu(input);
    auto output_cpu = output.cpu();
    auto exp_cpu    = expected.cpu();
    for (int64_t i = 0; i < n; ++i) {
        EXPECT_NEAR(output_cpu[i].item<double>(), exp_cpu[i].item<double>(), 1e-10)
            << "at index " << i;
    }
}

// =============================================================================
// vector_view tests
// =============================================================================

class VectorViewTest : public ::testing::Test {};

TEST_F(VectorViewTest, CPU_Contiguous) {
    int64_t const n = 100;
    auto tensor     = torch::arange(n, torch::kFloat32);

    vector_const_view<torch::kCPU, torch::kFloat32> view{tensor};

    EXPECT_EQ(view.count, n);
    EXPECT_EQ(view.stride, 1);

    for (int64_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(view[i], static_cast<float>(i));
    }
}

TEST_F(VectorViewTest, CPU_Strided) {
    int64_t const n = 50;
    auto full       = torch::arange(n * 2, torch::kFloat32);
    auto tensor     = full.slice(0, 0, n * 2, 2); // stride=2

    vector_const_view<torch::kCPU, torch::kFloat32> view{tensor};

    EXPECT_EQ(view.count, n);
    EXPECT_EQ(view.stride, 2);

    for (int64_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(view[i], static_cast<float>(i * 2));
    }
}

// =============================================================================
// matrix_view tests
// =============================================================================

class MatrixViewTest : public ::testing::Test {};

TEST_F(MatrixViewTest, CPU_Basic) {
    int64_t const rows = 4;
    int64_t const cols = 5;
    auto tensor        = torch::zeros({rows, cols}, torch::kFloat32);

    // Fill with row * 10 + col pattern
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
            tensor[r][c] = static_cast<float>(r * 10 + c);
        }
    }

    matrix_const_view<torch::kCPU, torch::kFloat32> view{tensor};

    EXPECT_EQ(view.rows, rows);
    EXPECT_EQ(view.cols, cols);

    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
            EXPECT_FLOAT_EQ(view(r, c), static_cast<float>(r * 10 + c))
                << "at (" << r << ", " << c << ")";
        }
    }
}

TEST_F(MatrixViewTest, CPU_Mutable) {
    int64_t const rows = 4;
    int64_t const cols = 5;
    auto tensor        = torch::zeros({rows, cols}, torch::kFloat32);

    matrix_mutable_view<torch::kCPU, torch::kFloat32> view{tensor};

    // Write through view
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
            view(r, c) = static_cast<float>(r * 10 + c);
        }
    }

    // Verify in tensor
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
            EXPECT_FLOAT_EQ(tensor[r][c].item<float>(), static_cast<float>(r * 10 + c))
                << "at (" << r << ", " << c << ")";
        }
    }
}

} // namespace
} // namespace dispatch
