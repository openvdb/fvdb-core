// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Tests for dispatch::gather and dispatch::scatter_add primitives.
//

// Suppress nvcc warning about test_info_ member in GoogleTest macros
#ifdef __NVCC__
#pragma nv_diag_suppress 177
#endif

#include "dispatch/torch/gather_scatter.h"
#include "dispatch/torch/types.h"
#include "test_utils.h"

#include <gtest/gtest.h>

#include <vector>

namespace dispatch {
namespace {

using test::skip_if_no_cuda;

// =============================================================================
// Helper functions to create views from tensors
// =============================================================================

template <torch::DeviceType Dev, torch::ScalarType Stype>
matrix_const_view<Dev, Stype>
make_const_matrix_view(torch::Tensor t) {
    using T = torch_scalar_cpp_type_t<Stype>;
    return matrix_const_view<Dev, Stype>(t.data_ptr<T>(), t.size(0), t.size(1));
}

template <torch::DeviceType Dev, torch::ScalarType Stype>
matrix_mutable_view<Dev, Stype>
make_mutable_matrix_view(torch::Tensor t) {
    using T = torch_scalar_cpp_type_t<Stype>;
    return matrix_mutable_view<Dev, Stype>(t.data_ptr<T>(), t.size(0), t.size(1));
}

template <torch::DeviceType Dev>
vector_const_view<Dev, torch::kInt32>
make_index_view(torch::Tensor t) {
    return vector_const_view<Dev, torch::kInt32>(t.data_ptr<int32_t>(), t.size(0), t.stride(0));
}

// =============================================================================
// Gather tests
// =============================================================================

class GatherTest : public ::testing::Test {};

TEST_F(GatherTest, CPU_BasicGather) {
    // Create source matrix (4x3) with known values
    auto src = torch::zeros({4, 3}, torch::kFloat32);
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 3; ++c) {
            src[r][c] = static_cast<float>(r * 10 + c);
        }
    }

    // Indices to gather: rows 2, 0, 3
    auto indices = torch::tensor({2, 0, 3}, torch::kInt32);

    // Output (3x3)
    auto dst = torch::zeros({3, 3}, torch::kFloat32);

    using Tag     = tag<torch::kCPU, torch::kFloat32>;
    auto src_view = make_const_matrix_view<torch::kCPU, torch::kFloat32>(src);
    auto dst_view = make_mutable_matrix_view<torch::kCPU, torch::kFloat32>(dst);
    auto idx_view = make_index_view<torch::kCPU>(indices);

    gather(Tag{}, src_view, dst_view, idx_view);

    // Verify: dst[0,:] = src[2,:], dst[1,:] = src[0,:], dst[2,:] = src[3,:]
    for (int c = 0; c < 3; ++c) {
        EXPECT_FLOAT_EQ(dst[0][c].item<float>(), src[2][c].item<float>());
        EXPECT_FLOAT_EQ(dst[1][c].item<float>(), src[0][c].item<float>());
        EXPECT_FLOAT_EQ(dst[2][c].item<float>(), src[3][c].item<float>());
    }
}

TEST_F(GatherTest, CPU_NegativeIndicesSkipped) {
    auto src = torch::ones({4, 3}, torch::kFloat32);

    // Some indices are negative (should skip those rows)
    auto indices = torch::tensor({1, -1, 2}, torch::kInt32);

    // Pre-fill dst with sentinel values
    auto dst = torch::full({3, 3}, -999.0f, torch::kFloat32);

    using Tag     = tag<torch::kCPU, torch::kFloat32>;
    auto src_view = make_const_matrix_view<torch::kCPU, torch::kFloat32>(src);
    auto dst_view = make_mutable_matrix_view<torch::kCPU, torch::kFloat32>(dst);
    auto idx_view = make_index_view<torch::kCPU>(indices);

    gather(Tag{}, src_view, dst_view, idx_view);

    // Row 0: gathered from src[1,:] -> all 1s
    // Row 1: skipped (index -1) -> unchanged (-999)
    // Row 2: gathered from src[2,:] -> all 1s
    for (int c = 0; c < 3; ++c) {
        EXPECT_FLOAT_EQ(dst[0][c].item<float>(), 1.0f);
        EXPECT_FLOAT_EQ(dst[1][c].item<float>(), -999.0f); // unchanged
        EXPECT_FLOAT_EQ(dst[2][c].item<float>(), 1.0f);
    }
}

TEST_F(GatherTest, CUDA_BasicGather) {
    skip_if_no_cuda();

    auto src_cpu = torch::zeros({4, 3}, torch::kFloat32);
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 3; ++c) {
            src_cpu[r][c] = static_cast<float>(r * 10 + c);
        }
    }
    auto src     = src_cpu.cuda();
    auto indices = torch::tensor({2, 0, 3}, torch::kInt32).cuda();
    auto dst =
        torch::zeros({3, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    using Tag     = tag<torch::kCUDA, torch::kFloat32>;
    auto src_view = make_const_matrix_view<torch::kCUDA, torch::kFloat32>(src);
    auto dst_view = make_mutable_matrix_view<torch::kCUDA, torch::kFloat32>(dst);
    auto idx_view = make_index_view<torch::kCUDA>(indices);

    gather(Tag{}, src_view, dst_view, idx_view);

    auto dst_cpu = dst.cpu();
    for (int c = 0; c < 3; ++c) {
        EXPECT_FLOAT_EQ(dst_cpu[0][c].item<float>(), src_cpu[2][c].item<float>());
        EXPECT_FLOAT_EQ(dst_cpu[1][c].item<float>(), src_cpu[0][c].item<float>());
        EXPECT_FLOAT_EQ(dst_cpu[2][c].item<float>(), src_cpu[3][c].item<float>());
    }
}

// =============================================================================
// ScatterAdd tests
// =============================================================================

class ScatterAddTest : public ::testing::Test {};

TEST_F(ScatterAddTest, CPU_BasicScatterAdd) {
    // Source (3x3) with all 1s
    auto src = torch::ones({3, 3}, torch::kFloat32);

    // Scatter to rows 0, 2, 1 of dst
    auto indices = torch::tensor({0, 2, 1}, torch::kInt32);

    // Destination (4x3), initially zeros
    auto dst = torch::zeros({4, 3}, torch::kFloat32);

    using Tag     = tag<torch::kCPU, torch::kFloat32>;
    auto src_view = make_const_matrix_view<torch::kCPU, torch::kFloat32>(src);
    auto dst_view = make_mutable_matrix_view<torch::kCPU, torch::kFloat32>(dst);
    auto idx_view = make_index_view<torch::kCPU>(indices);

    scatter_add(Tag{}, src_view, dst_view, idx_view);

    // dst[0,:] += 1, dst[2,:] += 1, dst[1,:] += 1 -> all should be 1
    for (int c = 0; c < 3; ++c) {
        EXPECT_FLOAT_EQ(dst[0][c].item<float>(), 1.0f);
        EXPECT_FLOAT_EQ(dst[1][c].item<float>(), 1.0f);
        EXPECT_FLOAT_EQ(dst[2][c].item<float>(), 1.0f);
        EXPECT_FLOAT_EQ(dst[3][c].item<float>(), 0.0f); // untouched
    }
}

TEST_F(ScatterAddTest, CPU_NegativeIndicesSkipped) {
    auto src     = torch::ones({3, 3}, torch::kFloat32);
    auto indices = torch::tensor({0, -1, 1}, torch::kInt32);
    auto dst     = torch::zeros({4, 3}, torch::kFloat32);

    using Tag     = tag<torch::kCPU, torch::kFloat32>;
    auto src_view = make_const_matrix_view<torch::kCPU, torch::kFloat32>(src);
    auto dst_view = make_mutable_matrix_view<torch::kCPU, torch::kFloat32>(dst);
    auto idx_view = make_index_view<torch::kCPU>(indices);

    scatter_add(Tag{}, src_view, dst_view, idx_view);

    for (int c = 0; c < 3; ++c) {
        EXPECT_FLOAT_EQ(dst[0][c].item<float>(), 1.0f);
        EXPECT_FLOAT_EQ(dst[1][c].item<float>(), 1.0f);
        // Rows 2 and 3 never touched
        EXPECT_FLOAT_EQ(dst[2][c].item<float>(), 0.0f);
        EXPECT_FLOAT_EQ(dst[3][c].item<float>(), 0.0f);
    }
}

TEST_F(ScatterAddTest, CUDA_BasicScatterAdd) {
    skip_if_no_cuda();

    auto src =
        torch::ones({3, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto indices = torch::tensor({0, 2, 1}, torch::kInt32).cuda();
    auto dst =
        torch::zeros({4, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    using Tag     = tag<torch::kCUDA, torch::kFloat32>;
    auto src_view = make_const_matrix_view<torch::kCUDA, torch::kFloat32>(src);
    auto dst_view = make_mutable_matrix_view<torch::kCUDA, torch::kFloat32>(dst);
    auto idx_view = make_index_view<torch::kCUDA>(indices);

    scatter_add(Tag{}, src_view, dst_view, idx_view);

    auto dst_cpu = dst.cpu();
    for (int c = 0; c < 3; ++c) {
        EXPECT_FLOAT_EQ(dst_cpu[0][c].item<float>(), 1.0f);
        EXPECT_FLOAT_EQ(dst_cpu[1][c].item<float>(), 1.0f);
        EXPECT_FLOAT_EQ(dst_cpu[2][c].item<float>(), 1.0f);
        EXPECT_FLOAT_EQ(dst_cpu[3][c].item<float>(), 0.0f);
    }
}

TEST_F(ScatterAddTest, CUDA_ManyCollisions) {
    // Test atomic correctness: many threads scatter to same row
    skip_if_no_cuda();

    int64_t const n_rows = 1000;
    auto src             = torch::ones({n_rows, 3},
                           torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    // All scatter to row 0
    auto indices = torch::zeros({n_rows}, torch::kInt32).cuda();
    auto dst =
        torch::zeros({1, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    using Tag     = tag<torch::kCUDA, torch::kFloat32>;
    auto src_view = make_const_matrix_view<torch::kCUDA, torch::kFloat32>(src);
    auto dst_view = make_mutable_matrix_view<torch::kCUDA, torch::kFloat32>(dst);
    auto idx_view = make_index_view<torch::kCUDA>(indices);

    scatter_add(Tag{}, src_view, dst_view, idx_view);

    auto dst_cpu = dst.cpu();
    for (int c = 0; c < 3; ++c) {
        EXPECT_FLOAT_EQ(dst_cpu[0][c].item<float>(), static_cast<float>(n_rows));
    }
}

// =============================================================================
// Double precision tests
// =============================================================================

TEST_F(GatherTest, CPU_Double) {
    auto src = torch::zeros({4, 3}, torch::kFloat64);
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 3; ++c) {
            src[r][c] = static_cast<double>(r * 10 + c) + 0.5;
        }
    }

    auto indices = torch::tensor({2, 0}, torch::kInt32);
    auto dst     = torch::zeros({2, 3}, torch::kFloat64);

    using Tag     = tag<torch::kCPU, torch::kFloat64>;
    auto src_view = make_const_matrix_view<torch::kCPU, torch::kFloat64>(src);
    auto dst_view = make_mutable_matrix_view<torch::kCPU, torch::kFloat64>(dst);
    auto idx_view = make_index_view<torch::kCPU>(indices);

    gather(Tag{}, src_view, dst_view, idx_view);

    for (int c = 0; c < 3; ++c) {
        EXPECT_DOUBLE_EQ(dst[0][c].item<double>(), src[2][c].item<double>());
        EXPECT_DOUBLE_EQ(dst[1][c].item<double>(), src[0][c].item<double>());
    }
}

TEST_F(ScatterAddTest, CPU_Double) {
    auto src     = torch::full({2, 3}, 0.5, torch::kFloat64);
    auto indices = torch::tensor({1, 1}, torch::kInt32);
    auto dst     = torch::zeros({3, 3}, torch::kFloat64);

    using Tag     = tag<torch::kCPU, torch::kFloat64>;
    auto src_view = make_const_matrix_view<torch::kCPU, torch::kFloat64>(src);
    auto dst_view = make_mutable_matrix_view<torch::kCPU, torch::kFloat64>(dst);
    auto idx_view = make_index_view<torch::kCPU>(indices);

    scatter_add(Tag{}, src_view, dst_view, idx_view);

    for (int c = 0; c < 3; ++c) {
        EXPECT_DOUBLE_EQ(dst[0][c].item<double>(), 0.0);
        EXPECT_DOUBLE_EQ(dst[1][c].item<double>(), 1.0); // 0.5 + 0.5
        EXPECT_DOUBLE_EQ(dst[2][c].item<double>(), 0.0);
    }
}

} // namespace
} // namespace dispatch
