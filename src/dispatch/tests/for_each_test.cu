// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Tests for dispatch::for_each and dispatch::for_each_nd primitives.
//

// Suppress nvcc warning about test_info_ member in GoogleTest macros
#ifdef __NVCC__
#pragma nv_diag_suppress 177
#endif

#include "dispatch/torch/for_each.h"
#include "dispatch/torch/types.h"
#include "test_utils.h"

#include <gtest/gtest.h>

#include <array>
#include <vector>

namespace dispatch {
namespace {

using test::skip_if_no_cuda;

// =============================================================================
// Helper functions for CUDA tests (must be free functions, not in TEST_F)
// Extended __device__ lambdas cannot be in private/protected class methods.
// =============================================================================

using CudaFloatTag  = tag<torch::kCUDA, torch::kFloat32>;
using CudaDoubleTag = tag<torch::kCUDA, torch::kFloat64>;

void
cuda_for_each_increment(float *ptr, int64_t n) {
    for_each(CudaFloatTag{}, n, [=] __device__(CudaFloatTag, int64_t i) { ptr[i] = i + 1.0f; });
}

void
cuda_for_each_set_one(float *ptr, int64_t n) {
    for_each(CudaFloatTag{}, n, [=] __device__(CudaFloatTag, int64_t i) { ptr[i] = 1.0f; });
}

void
cuda_for_each_double(double *ptr, int64_t n) {
    for_each(CudaDoubleTag{}, n, [=] __device__(CudaDoubleTag, int64_t i) { ptr[i] = i * 0.5; });
}

void
cuda_for_each_nd_2d(float *ptr, int64_t rows, int64_t cols) {
    std::array<int64_t, 2> shape = {rows, cols};
    for_each_nd<2, row_major>(
        CudaFloatTag{}, shape, [=] __device__(CudaFloatTag, std::array<int64_t, 2> const &idx) {
            auto const r      = idx[0];
            auto const c      = idx[1];
            ptr[r * cols + c] = static_cast<float>(r * 100 + c);
        });
}

void
cuda_for_each_nd_large(float *ptr, int64_t rows, int64_t cols) {
    std::array<int64_t, 2> shape = {rows, cols};
    for_each_nd<2, row_major>(
        CudaFloatTag{}, shape, [=] __device__(CudaFloatTag, std::array<int64_t, 2> const &idx) {
            ptr[idx[0] * cols + idx[1]] = 1.0f;
        });
}

void
cuda_for_each_empty() {
    // Empty iteration - should not crash
    for_each(CudaFloatTag{}, 0, [=] __device__(CudaFloatTag, int64_t) {});
}

// =============================================================================
// for_each tests (1D iteration)
// =============================================================================

class ForEachTest : public ::testing::Test {};

TEST_F(ForEachTest, CPU_IncrementElements) {
    using Tag       = tag<torch::kCPU, torch::kFloat32>;
    int64_t const n = 100;
    auto tensor     = torch::zeros({n}, torch::kFloat32);
    float *ptr      = tensor.data_ptr<float>();

    for_each(Tag{}, n, [=](Tag, int64_t i) { ptr[i] = i + 1.0f; });

    for (int64_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(ptr[i], static_cast<float>(i + 1)) << "at index " << i;
    }
}

TEST_F(ForEachTest, CPU_EmptyTensor) {
    using Tag = tag<torch::kCPU, torch::kFloat32>;

    // Should not crash on empty tensor
    for_each(Tag{}, 0, [](Tag, int64_t) { FAIL() << "Should not be called for empty range"; });
}

TEST_F(ForEachTest, CPU_LargeCount) {
    using Tag       = tag<torch::kCPU, torch::kFloat32>;
    int64_t const n = 100000;
    auto tensor     = torch::zeros({n}, torch::kFloat32);
    float *ptr      = tensor.data_ptr<float>();

    for_each(Tag{}, n, [=](Tag, int64_t i) { ptr[i] = 1.0f; });

    float const sum = tensor.sum().item<float>();
    EXPECT_FLOAT_EQ(sum, static_cast<float>(n));
}

TEST_F(ForEachTest, CUDA_IncrementElements) {
    skip_if_no_cuda();

    int64_t const n = 100;
    auto tensor =
        torch::zeros({n}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    float *ptr = tensor.data_ptr<float>();

    cuda_for_each_increment(ptr, n);

    auto cpu_tensor = tensor.cpu();
    float *cpu_ptr  = cpu_tensor.data_ptr<float>();
    for (int64_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(cpu_ptr[i], static_cast<float>(i + 1)) << "at index " << i;
    }
}

TEST_F(ForEachTest, CUDA_EmptyTensor) {
    skip_if_no_cuda();

    // Should not crash on empty tensor - just call with count=0
    // Uses helper function because device lambdas can't be in TEST_F body
    cuda_for_each_empty();
}

TEST_F(ForEachTest, CUDA_LargeCount) {
    skip_if_no_cuda();

    int64_t const n = 100000;
    auto tensor =
        torch::zeros({n}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    float *ptr = tensor.data_ptr<float>();

    cuda_for_each_set_one(ptr, n);

    float const sum = tensor.sum().item<float>();
    EXPECT_FLOAT_EQ(sum, static_cast<float>(n));
}

// =============================================================================
// for_each_nd tests (N-dimensional iteration)
// =============================================================================

class ForEachNdTest : public ::testing::Test {};

TEST_F(ForEachNdTest, CPU_2D_RowMajor) {
    using Tag          = tag<torch::kCPU, torch::kFloat32>;
    int64_t const rows = 10;
    int64_t const cols = 20;
    auto tensor        = torch::zeros({rows, cols}, torch::kFloat32);
    float *ptr         = tensor.data_ptr<float>();

    std::array<int64_t, 2> shape = {rows, cols};

    for_each_nd<2, row_major>(Tag{}, shape, [=](Tag, std::array<int64_t, 2> const &idx) {
        auto const [r, c] = idx;
        // Store row * 100 + col to verify indices
        ptr[r * cols + c] = static_cast<float>(r * 100 + c);
    });

    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
            EXPECT_FLOAT_EQ(ptr[r * cols + c], static_cast<float>(r * 100 + c))
                << "at (" << r << ", " << c << ")";
        }
    }
}

TEST_F(ForEachNdTest, CPU_3D_RowMajor) {
    using Tag        = tag<torch::kCPU, torch::kFloat32>;
    int64_t const d0 = 5;
    int64_t const d1 = 6;
    int64_t const d2 = 7;
    auto tensor      = torch::zeros({d0, d1, d2}, torch::kFloat32);
    float *ptr       = tensor.data_ptr<float>();

    std::array<int64_t, 3> shape = {d0, d1, d2};

    for_each_nd<3, row_major>(Tag{}, shape, [=](Tag, std::array<int64_t, 3> const &idx) {
        auto const [i, j, k]       = idx;
        ptr[(i * d1 + j) * d2 + k] = static_cast<float>(i * 10000 + j * 100 + k);
    });

    for (int64_t i = 0; i < d0; ++i) {
        for (int64_t j = 0; j < d1; ++j) {
            for (int64_t k = 0; k < d2; ++k) {
                EXPECT_FLOAT_EQ(ptr[(i * d1 + j) * d2 + k],
                                static_cast<float>(i * 10000 + j * 100 + k))
                    << "at (" << i << ", " << j << ", " << k << ")";
            }
        }
    }
}

TEST_F(ForEachNdTest, CPU_EmptyShape) {
    using Tag                    = tag<torch::kCPU, torch::kFloat32>;
    std::array<int64_t, 2> shape = {0, 10};

    for_each_nd<2, row_major>(Tag{}, shape, [](Tag, std::array<int64_t, 2> const &) {
        FAIL() << "Should not be called for empty shape";
    });
}

TEST_F(ForEachNdTest, CUDA_2D_RowMajor) {
    skip_if_no_cuda();

    int64_t const rows = 10;
    int64_t const cols = 20;
    auto tensor        = torch::zeros({rows, cols},
                               torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    float *ptr         = tensor.data_ptr<float>();

    cuda_for_each_nd_2d(ptr, rows, cols);

    auto cpu_tensor = tensor.cpu();
    float *cpu_ptr  = cpu_tensor.data_ptr<float>();
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
            EXPECT_FLOAT_EQ(cpu_ptr[r * cols + c], static_cast<float>(r * 100 + c))
                << "at (" << r << ", " << c << ")";
        }
    }
}

TEST_F(ForEachNdTest, CUDA_LargeShape) {
    skip_if_no_cuda();

    int64_t const rows = 100;
    int64_t const cols = 1000;
    auto tensor        = torch::zeros({rows, cols},
                               torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    float *ptr         = tensor.data_ptr<float>();

    cuda_for_each_nd_large(ptr, rows, cols);

    float const sum = tensor.sum().item<float>();
    EXPECT_FLOAT_EQ(sum, static_cast<float>(rows * cols));
}

// =============================================================================
// Double precision tests
// =============================================================================

TEST_F(ForEachTest, CPU_Double) {
    using Tag       = tag<torch::kCPU, torch::kFloat64>;
    int64_t const n = 100;
    auto tensor     = torch::zeros({n}, torch::kFloat64);
    double *ptr     = tensor.data_ptr<double>();

    for_each(Tag{}, n, [=](Tag, int64_t i) { ptr[i] = i * 0.5; });

    for (int64_t i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(ptr[i], i * 0.5) << "at index " << i;
    }
}

TEST_F(ForEachTest, CUDA_Double) {
    skip_if_no_cuda();

    int64_t const n = 100;
    auto tensor =
        torch::zeros({n}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA));
    double *ptr = tensor.data_ptr<double>();

    cuda_for_each_double(ptr, n);

    auto cpu_tensor = tensor.cpu();
    double *cpu_ptr = cpu_tensor.data_ptr<double>();
    for (int64_t i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(cpu_ptr[i], i * 0.5) << "at index " << i;
    }
}

} // namespace
} // namespace dispatch
