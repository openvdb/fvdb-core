// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Tests for dispatch::gather and dispatch::scatter_add primitives.
//

#include "dispatch/torch/gather_scatter.h"
#include "dispatch/torch/types.h"
#include "test_utils.h"

#include <gtest/gtest.h>

#include <vector>

namespace dispatch {
namespace {

using test::skip_if_no_cuda;

// =============================================================================
// Helper to create views from tensors
// =============================================================================

template <torch::DeviceType Dev, torch::ScalarType Stype>
matrix_const_view<Dev, Stype>
make_const_matrix_view(torch::Tensor t) {
    using T = torch_scalar_cpp_type_t<Stype>;
    return {t.data_ptr<T>(), t.size(0), t.size(1)};
}

template <torch::DeviceType Dev, torch::ScalarType Stype>
matrix_mutable_view<Dev, Stype>
make_mutable_matrix_view(torch::Tensor t) {
    using T = torch_scalar_cpp_type_t<Stype>;
    return {t.data_ptr<T>(), t.size(0), t.size(1)};
}

template <torch::DeviceType Dev>
vector_const_view<Dev, torch::kInt32>
make_index_view(torch::Tensor t) {
    return {t.data_ptr<int32_t>(), t.size(0), t.stride(0)};
}

// =============================================================================
// gather tests
// =============================================================================

class GatherTest : public ::testing::Test {};

TEST_F(GatherTest, CPU_BasicGather) {
    // Source: 4x3 matrix with values row*10 + col
    // [[ 0,  1,  2],
    //  [10, 11, 12],
    //  [20, 21, 22],
    //  [30, 31, 32]]
    auto src = torch::zeros({4, 3}, torch::kFloat32);
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 3; ++c) {
            src[r][c] = static_cast<float>(r * 10 + c);
        }
    }

    // Indices: [2, 0, 3] -> gather rows 2, 0, 3
    auto indices = torch::tensor({2, 0, 3}, torch::kInt32);

    // Destination: 3x3 matrix
    auto dst = torch::zeros({3, 3}, torch::kFloat32);

    auto src_view = make_const_matrix_view<torch::kCPU, torch::kFloat32>(src);
    auto dst_view = make_mutable_matrix_view<torch::kCPU, torch::kFloat32>(dst);
    auto idx_view = make_index_view<torch::kCPU>(indices);

    gather(tag<torch::kCPU, torch::kFloat32>{}, src_view, dst_view, idx_view);

    // Expected:
    // dst[0] = src[2] = [20, 21, 22]
    // dst[1] = src[0] = [ 0,  1,  2]
    // dst[2] = src[3] = [30, 31, 32]
    EXPECT_FLOAT_EQ(dst[0][0].item<float>(), 20.0f);
    EXPECT_FLOAT_EQ(dst[0][1].item<float>(), 21.0f);
    EXPECT_FLOAT_EQ(dst[0][2].item<float>(), 22.0f);
    EXPECT_FLOAT_EQ(dst[1][0].item<float>(), 0.0f);
    EXPECT_FLOAT_EQ(dst[1][1].item<float>(), 1.0f);
    EXPECT_FLOAT_EQ(dst[1][2].item<float>(), 2.0f);
    EXPECT_FLOAT_EQ(dst[2][0].item<float>(), 30.0f);
    EXPECT_FLOAT_EQ(dst[2][1].item<float>(), 31.0f);
    EXPECT_FLOAT_EQ(dst[2][2].item<float>(), 32.0f);
}

TEST_F(GatherTest, CPU_NegativeIndicesSkipped) {
    auto src     = torch::ones({4, 3}, torch::kFloat32) * 5.0f;
    auto indices = torch::tensor({0, -1, 2}, torch::kInt32); // -1 should skip
    auto dst     = torch::zeros({3, 3}, torch::kFloat32);

    auto src_view = make_const_matrix_view<torch::kCPU, torch::kFloat32>(src);
    auto dst_view = make_mutable_matrix_view<torch::kCPU, torch::kFloat32>(dst);
    auto idx_view = make_index_view<torch::kCPU>(indices);

    gather(tag<torch::kCPU, torch::kFloat32>{}, src_view, dst_view, idx_view);

    // Row 0: gathered from src[0] = [5, 5, 5]
    EXPECT_FLOAT_EQ(dst[0][0].item<float>(), 5.0f);
    // Row 1: index is -1, should remain zeros
    EXPECT_FLOAT_EQ(dst[1][0].item<float>(), 0.0f);
    EXPECT_FLOAT_EQ(dst[1][1].item<float>(), 0.0f);
    EXPECT_FLOAT_EQ(dst[1][2].item<float>(), 0.0f);
    // Row 2: gathered from src[2] = [5, 5, 5]
    EXPECT_FLOAT_EQ(dst[2][0].item<float>(), 5.0f);
}

TEST_F(GatherTest, CUDA_BasicGather) {
    skip_if_no_cuda();

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto src     = torch::zeros({4, 3}, options);
    // Fill on CPU then move to CUDA for simplicity
    auto src_cpu = torch::zeros({4, 3}, torch::kFloat32);
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 3; ++c) {
            src_cpu[r][c] = static_cast<float>(r * 10 + c);
        }
    }
    src.copy_(src_cpu);

    auto indices =
        torch::tensor({2, 0, 3}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    auto dst = torch::zeros({3, 3}, options);

    auto src_view = make_const_matrix_view<torch::kCUDA, torch::kFloat32>(src);
    auto dst_view = make_mutable_matrix_view<torch::kCUDA, torch::kFloat32>(dst);
    auto idx_view = make_index_view<torch::kCUDA>(indices);

    gather(tag<torch::kCUDA, torch::kFloat32>{}, src_view, dst_view, idx_view);

    auto dst_cpu = dst.cpu();
    EXPECT_FLOAT_EQ(dst_cpu[0][0].item<float>(), 20.0f);
    EXPECT_FLOAT_EQ(dst_cpu[0][1].item<float>(), 21.0f);
    EXPECT_FLOAT_EQ(dst_cpu[0][2].item<float>(), 22.0f);
    EXPECT_FLOAT_EQ(dst_cpu[1][0].item<float>(), 0.0f);
    EXPECT_FLOAT_EQ(dst_cpu[1][1].item<float>(), 1.0f);
    EXPECT_FLOAT_EQ(dst_cpu[1][2].item<float>(), 2.0f);
    EXPECT_FLOAT_EQ(dst_cpu[2][0].item<float>(), 30.0f);
    EXPECT_FLOAT_EQ(dst_cpu[2][1].item<float>(), 31.0f);
    EXPECT_FLOAT_EQ(dst_cpu[2][2].item<float>(), 32.0f);
}

// =============================================================================
// scatter_add tests
// =============================================================================

class ScatterAddTest : public ::testing::Test {};

TEST_F(ScatterAddTest, CPU_BasicScatterAdd) {
    // Source: 3x2 matrix
    // [[1, 2],
    //  [3, 4],
    //  [5, 6]]
    auto src = torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}});

    // Indices: [0, 2, 0] -> scatter src rows to dst rows 0, 2, 0
    // Row 0 accumulates src[0] + src[2] = [1+5, 2+6] = [6, 8]
    // Row 2 gets src[1] = [3, 4]
    auto indices = torch::tensor({0, 2, 0}, torch::kInt32);

    auto dst = torch::zeros({4, 2}, torch::kFloat32);

    auto src_view = make_const_matrix_view<torch::kCPU, torch::kFloat32>(src);
    auto dst_view = make_mutable_matrix_view<torch::kCPU, torch::kFloat32>(dst);
    auto idx_view = make_index_view<torch::kCPU>(indices);

    scatter_add(tag<torch::kCPU, torch::kFloat32>{}, src_view, dst_view, idx_view);

    // dst[0] = src[0] + src[2] = [6, 8]
    EXPECT_FLOAT_EQ(dst[0][0].item<float>(), 6.0f);
    EXPECT_FLOAT_EQ(dst[0][1].item<float>(), 8.0f);
    // dst[1] = unchanged = [0, 0]
    EXPECT_FLOAT_EQ(dst[1][0].item<float>(), 0.0f);
    EXPECT_FLOAT_EQ(dst[1][1].item<float>(), 0.0f);
    // dst[2] = src[1] = [3, 4]
    EXPECT_FLOAT_EQ(dst[2][0].item<float>(), 3.0f);
    EXPECT_FLOAT_EQ(dst[2][1].item<float>(), 4.0f);
    // dst[3] = unchanged = [0, 0]
    EXPECT_FLOAT_EQ(dst[3][0].item<float>(), 0.0f);
    EXPECT_FLOAT_EQ(dst[3][1].item<float>(), 0.0f);
}

TEST_F(ScatterAddTest, CPU_NegativeIndicesSkipped) {
    auto src     = torch::ones({3, 2}, torch::kFloat32);
    auto indices = torch::tensor({0, -1, 1}, torch::kInt32); // -1 should skip
    auto dst     = torch::zeros({3, 2}, torch::kFloat32);

    auto src_view = make_const_matrix_view<torch::kCPU, torch::kFloat32>(src);
    auto dst_view = make_mutable_matrix_view<torch::kCPU, torch::kFloat32>(dst);
    auto idx_view = make_index_view<torch::kCPU>(indices);

    scatter_add(tag<torch::kCPU, torch::kFloat32>{}, src_view, dst_view, idx_view);

    // dst[0] = src[0] = [1, 1]
    EXPECT_FLOAT_EQ(dst[0][0].item<float>(), 1.0f);
    EXPECT_FLOAT_EQ(dst[0][1].item<float>(), 1.0f);
    // dst[1] = src[2] = [1, 1]
    EXPECT_FLOAT_EQ(dst[1][0].item<float>(), 1.0f);
    EXPECT_FLOAT_EQ(dst[1][1].item<float>(), 1.0f);
    // dst[2] = unchanged (src[1] had index -1)
    EXPECT_FLOAT_EQ(dst[2][0].item<float>(), 0.0f);
    EXPECT_FLOAT_EQ(dst[2][1].item<float>(), 0.0f);
}

TEST_F(ScatterAddTest, CUDA_BasicScatterAdd) {
    skip_if_no_cuda();

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto src     = torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}}, options);
    auto indices =
        torch::tensor({0, 2, 0}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    auto dst = torch::zeros({4, 2}, options);

    auto src_view = make_const_matrix_view<torch::kCUDA, torch::kFloat32>(src);
    auto dst_view = make_mutable_matrix_view<torch::kCUDA, torch::kFloat32>(dst);
    auto idx_view = make_index_view<torch::kCUDA>(indices);

    scatter_add(tag<torch::kCUDA, torch::kFloat32>{}, src_view, dst_view, idx_view);

    auto dst_cpu = dst.cpu();
    EXPECT_FLOAT_EQ(dst_cpu[0][0].item<float>(), 6.0f);
    EXPECT_FLOAT_EQ(dst_cpu[0][1].item<float>(), 8.0f);
    EXPECT_FLOAT_EQ(dst_cpu[1][0].item<float>(), 0.0f);
    EXPECT_FLOAT_EQ(dst_cpu[1][1].item<float>(), 0.0f);
    EXPECT_FLOAT_EQ(dst_cpu[2][0].item<float>(), 3.0f);
    EXPECT_FLOAT_EQ(dst_cpu[2][1].item<float>(), 4.0f);
    EXPECT_FLOAT_EQ(dst_cpu[3][0].item<float>(), 0.0f);
    EXPECT_FLOAT_EQ(dst_cpu[3][1].item<float>(), 0.0f);
}

TEST_F(ScatterAddTest, CUDA_ManyCollisions) {
    skip_if_no_cuda();

    // Test atomic correctness: many rows scatter to same destination
    int64_t const n_src = 1000;
    int64_t const cols  = 4;

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto src     = torch::ones({n_src, cols}, options);
    // All rows scatter to row 0
    auto indices =
        torch::zeros({n_src}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    auto dst = torch::zeros({1, cols}, options);

    auto src_view = make_const_matrix_view<torch::kCUDA, torch::kFloat32>(src);
    auto dst_view = make_mutable_matrix_view<torch::kCUDA, torch::kFloat32>(dst);
    auto idx_view = make_index_view<torch::kCUDA>(indices);

    scatter_add(tag<torch::kCUDA, torch::kFloat32>{}, src_view, dst_view, idx_view);

    auto dst_cpu = dst.cpu();
    for (int64_t c = 0; c < cols; ++c) {
        EXPECT_FLOAT_EQ(dst_cpu[0][c].item<float>(), static_cast<float>(n_src))
            << "at column " << c;
    }
}

// =============================================================================
// Double precision tests
// =============================================================================

TEST_F(GatherTest, CPU_Double) {
    auto src     = torch::tensor({{1.0, 2.0}, {3.0, 4.0}}, torch::kFloat64);
    auto indices = torch::tensor({1, 0}, torch::kInt32);
    auto dst     = torch::zeros({2, 2}, torch::kFloat64);

    auto src_view = make_const_matrix_view<torch::kCPU, torch::kFloat64>(src);
    auto dst_view = make_mutable_matrix_view<torch::kCPU, torch::kFloat64>(dst);
    auto idx_view = make_index_view<torch::kCPU>(indices);

    gather(tag<torch::kCPU, torch::kFloat64>{}, src_view, dst_view, idx_view);

    EXPECT_DOUBLE_EQ(dst[0][0].item<double>(), 3.0);
    EXPECT_DOUBLE_EQ(dst[0][1].item<double>(), 4.0);
    EXPECT_DOUBLE_EQ(dst[1][0].item<double>(), 1.0);
    EXPECT_DOUBLE_EQ(dst[1][1].item<double>(), 2.0);
}

TEST_F(ScatterAddTest, CPU_Double) {
    auto src     = torch::tensor({{1.0, 2.0}, {3.0, 4.0}}, torch::kFloat64);
    auto indices = torch::tensor({0, 0}, torch::kInt32);
    auto dst     = torch::zeros({1, 2}, torch::kFloat64);

    auto src_view = make_const_matrix_view<torch::kCPU, torch::kFloat64>(src);
    auto dst_view = make_mutable_matrix_view<torch::kCPU, torch::kFloat64>(dst);
    auto idx_view = make_index_view<torch::kCPU>(indices);

    scatter_add(tag<torch::kCPU, torch::kFloat64>{}, src_view, dst_view, idx_view);

    EXPECT_DOUBLE_EQ(dst[0][0].item<double>(), 4.0); // 1 + 3
    EXPECT_DOUBLE_EQ(dst[0][1].item<double>(), 6.0); // 2 + 4
}

} // namespace
} // namespace dispatch
