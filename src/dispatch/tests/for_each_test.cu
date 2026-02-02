// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Tests for dispatch::for_each and dispatch::for_each_nd primitives.
//

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
// for_each tests (1D iteration)
// =============================================================================

class ForEachTest : public ::testing::Test {};

TEST_F(ForEachTest, CPU_IncrementElements) {
    int64_t const n = 100;
    auto tensor     = torch::zeros({n}, torch::kFloat32);
    float *ptr      = tensor.data_ptr<float>();

    for_each(tag<torch::kCPU, torch::kFloat32>{}, n, [=](auto, int64_t i) { ptr[i] = i + 1.0f; });

    for (int64_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(ptr[i], static_cast<float>(i + 1)) << "at index " << i;
    }
}

TEST_F(ForEachTest, CPU_EmptyTensor) {
    auto tensor = torch::zeros({0}, torch::kFloat32);

    // Should not crash on empty tensor
    for_each(tag<torch::kCPU, torch::kFloat32>{}, 0, [](auto, int64_t) {
        FAIL() << "Should not be called for empty range";
    });
}

TEST_F(ForEachTest, CPU_LargeCount) {
    int64_t const n = 100000;
    auto tensor     = torch::zeros({n}, torch::kFloat32);
    float *ptr      = tensor.data_ptr<float>();

    for_each(tag<torch::kCPU, torch::kFloat32>{}, n, [=](auto, int64_t i) { ptr[i] = 1.0f; });

    float const sum = tensor.sum().item<float>();
    EXPECT_FLOAT_EQ(sum, static_cast<float>(n));
}

TEST_F(ForEachTest, CUDA_IncrementElements) {
    skip_if_no_cuda();

    int64_t const n = 100;
    auto tensor =
        torch::zeros({n}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    float *ptr = tensor.data_ptr<float>();

    for_each(tag<torch::kCUDA, torch::kFloat32>{}, n, [=] __device__(auto, int64_t i) {
        ptr[i] = i + 1.0f;
    });

    auto cpu_tensor = tensor.cpu();
    float *cpu_ptr  = cpu_tensor.data_ptr<float>();
    for (int64_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(cpu_ptr[i], static_cast<float>(i + 1)) << "at index " << i;
    }
}

TEST_F(ForEachTest, CUDA_EmptyTensor) {
    skip_if_no_cuda();

    // Should not crash on empty tensor
    for_each(tag<torch::kCUDA, torch::kFloat32>{}, 0, [=] __device__(auto, int64_t) {
        // Should never be called
    });
}

TEST_F(ForEachTest, CUDA_LargeCount) {
    skip_if_no_cuda();

    int64_t const n = 100000;
    auto tensor =
        torch::zeros({n}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    float *ptr = tensor.data_ptr<float>();

    for_each(tag<torch::kCUDA, torch::kFloat32>{}, n, [=] __device__(auto, int64_t i) {
        ptr[i] = 1.0f;
    });

    float const sum = tensor.sum().item<float>();
    EXPECT_FLOAT_EQ(sum, static_cast<float>(n));
}

// =============================================================================
// for_each_nd tests (N-dimensional iteration)
// =============================================================================

class ForEachNdTest : public ::testing::Test {};

TEST_F(ForEachNdTest, CPU_2D_RowMajor) {
    int64_t const rows = 10;
    int64_t const cols = 20;
    auto tensor        = torch::zeros({rows, cols}, torch::kFloat32);
    float *ptr         = tensor.data_ptr<float>();

    std::array<int64_t, 2> shape = {rows, cols};

    for_each_nd<2, row_major>(
        tag<torch::kCPU, torch::kFloat32>{}, shape, [=](auto, std::array<int64_t, 2> const &idx) {
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
    int64_t const d0 = 5;
    int64_t const d1 = 6;
    int64_t const d2 = 7;
    auto tensor      = torch::zeros({d0, d1, d2}, torch::kFloat32);
    float *ptr       = tensor.data_ptr<float>();

    std::array<int64_t, 3> shape = {d0, d1, d2};

    for_each_nd<3, row_major>(
        tag<torch::kCPU, torch::kFloat32>{}, shape, [=](auto, std::array<int64_t, 3> const &idx) {
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
    std::array<int64_t, 2> shape = {0, 10};

    for_each_nd<2, row_major>(
        tag<torch::kCPU, torch::kFloat32>{}, shape, [](auto, std::array<int64_t, 2> const &) {
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

    std::array<int64_t, 2> shape = {rows, cols};

    for_each_nd<2, row_major>(tag<torch::kCUDA, torch::kFloat32>{},
                              shape,
                              [=] __device__(auto, std::array<int64_t, 2> const &idx) {
                                  auto const r      = idx[0];
                                  auto const c      = idx[1];
                                  ptr[r * cols + c] = static_cast<float>(r * 100 + c);
                              });

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

    std::array<int64_t, 2> shape = {rows, cols};

    for_each_nd<2, row_major>(tag<torch::kCUDA, torch::kFloat32>{},
                              shape,
                              [=] __device__(auto, std::array<int64_t, 2> const &idx) {
                                  ptr[idx[0] * cols + idx[1]] = 1.0f;
                              });

    float const sum = tensor.sum().item<float>();
    EXPECT_FLOAT_EQ(sum, static_cast<float>(rows * cols));
}

// =============================================================================
// Double precision tests
// =============================================================================

TEST_F(ForEachTest, CPU_Double) {
    int64_t const n = 100;
    auto tensor     = torch::zeros({n}, torch::kFloat64);
    double *ptr     = tensor.data_ptr<double>();

    for_each(tag<torch::kCPU, torch::kFloat64>{}, n, [=](auto, int64_t i) { ptr[i] = i * 0.5; });

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

    for_each(tag<torch::kCUDA, torch::kFloat64>{}, n, [=] __device__(auto, int64_t i) {
        ptr[i] = i * 0.5;
    });

    auto cpu_tensor = tensor.cpu();
    double *cpu_ptr = cpu_tensor.data_ptr<double>();
    for (int64_t i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(cpu_ptr[i], i * 0.5) << "at index " << i;
    }
}

} // namespace
} // namespace dispatch
