// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// GroupedGemmConvTest.cu -- Tests for the CUTLASS grouped-GEMM sparse convolution op.
//
// Every test compares the GroupedGemm backend output against the GatherScatter
// (GEMM) backend output, which serves as the reference implementation.
// The GroupedGemm backend is CUDA-only and float32-only, so all tests run
// exclusively on CUDA with float32 data.
//
#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/GridBatchImpl.h>
#include <fvdb/detail/ops/convolution/GatherScatter.h>
#include <fvdb/detail/ops/convolution/GroupedGemm.h>

#include <torch/types.h>

#include <gtest/gtest.h>

#include <cmath>
#include <tuple>
#include <vector>

using namespace fvdb;
using namespace fvdb::detail;

// =============================================================================
// Helpers
// =============================================================================

// Tolerances for float32 comparison.  The GroupedGemm path uses TF32
// tensorcores (reduced mantissa) so we allow slightly more slack than
// exact float32.
static constexpr double kRtol = 1e-3;
static constexpr double kAtol = 1e-3;

static c10::intrusive_ptr<GridBatchImpl>
makeGrid(torch::Tensor ijk_2d, torch::Device device) {
    auto ijk_dev = ijk_2d.to(device);
    JaggedTensor jt(ijk_dev);
    std::vector<nanovdb::Vec3d> voxel_sizes = {{1.0, 1.0, 1.0}};
    std::vector<nanovdb::Vec3d> origins     = {{0.0, 0.0, 0.0}};
    return GridBatchImpl::createFromIjk(jt, voxel_sizes, origins);
}

static c10::intrusive_ptr<GridBatchImpl>
makeStridedTestGrid(torch::Device device) {
    std::vector<std::vector<int32_t>> coords;
    for (int i = 4; i < 7; ++i)
        for (int j = 4; j < 7; ++j)
            for (int k = 4; k < 7; ++k)
                coords.push_back({i, j, k});

    int64_t N = static_cast<int64_t>(coords.size());
    auto ijk  = torch::zeros({N, 3}, torch::dtype(torch::kInt32));
    for (int64_t i = 0; i < N; ++i) {
        ijk[i][0] = coords[i][0];
        ijk[i][1] = coords[i][1];
        ijk[i][2] = coords[i][2];
    }
    return makeGrid(ijk, device);
}

static bool
cudaIsAvailable() {
    int count = 0;
    auto err  = cudaGetDeviceCount(&count);
    return err == cudaSuccess && count > 0;
}

// =============================================================================
// Forward tests -- GroupedGemm vs GatherScatter (GEMM) reference
// =============================================================================

class GroupedGemmForwardTest : public ::testing::TestWithParam<int64_t> {};

TEST_P(GroupedGemmForwardTest, SameGridStride1) {
    if (!cudaIsAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }
    torch::Device device(torch::kCUDA, 0);
    int64_t const C = GetParam(); // C_in = C_out = C

    // 2x2x2 block
    auto ijk = torch::tensor(
        {{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1}, {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}},
        torch::dtype(torch::kInt32));
    auto grid = makeGrid(ijk, device);

    int64_t N = 8;
    nanovdb::Coord kernel_size(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);

    // Build both topologies
    auto topo_gemm    = ops::gatherScatterSparseConvTopology(*grid, *grid, kernel_size, stride);
    auto topo_grouped = ops::groupedGemmSparseConvTopology(*grid, *grid, kernel_size, stride);

    torch::manual_seed(42);
    auto features = torch::randn({N, C}, torch::dtype(torch::kFloat32).device(device));
    auto weights  = torch::randn({C, C, 3, 3, 3}, torch::dtype(torch::kFloat32).device(device));

    auto ref = ops::gatherScatterSparseConv(features, weights, topo_gemm);
    auto out = ops::groupedGemmSparseConv(features, weights, topo_grouped);

    EXPECT_EQ(out.sizes(), ref.sizes());
    EXPECT_TRUE(torch::allclose(out.cpu(), ref.cpu(), kRtol, kAtol))
        << "Forward mismatch at C=" << C;
}

TEST_P(GroupedGemmForwardTest, IdentityConv1x1x1) {
    if (!cudaIsAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }
    torch::Device device(torch::kCUDA, 0);
    int64_t const C = GetParam();

    auto ijk = torch::tensor(
        {{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1}, {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}},
        torch::dtype(torch::kInt32));
    auto grid = makeGrid(ijk, device);
    int64_t N = 8;

    nanovdb::Coord kernel_size(1, 1, 1);
    nanovdb::Coord stride(1, 1, 1);

    auto topo = ops::groupedGemmSparseConvTopology(*grid, *grid, kernel_size, stride);

    auto features = torch::randn({N, C}, torch::dtype(torch::kFloat32).device(device));
    auto weights =
        torch::eye(C, torch::dtype(torch::kFloat32).device(device)).reshape({C, C, 1, 1, 1});

    auto out = ops::groupedGemmSparseConv(features, weights, topo);

    // With identity weights, output should equal input (within TF32 tolerance)
    EXPECT_TRUE(torch::allclose(out.cpu(), features.cpu(), kRtol, kAtol))
        << "Identity conv mismatch at C=" << C;
}

TEST_P(GroupedGemmForwardTest, DifferentGrids) {
    if (!cudaIsAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }
    torch::Device device(torch::kCUDA, 0);
    int64_t const C = GetParam();

    // Single voxel at (5,5,5) -- convOutput will produce 27 output voxels
    auto ijk      = torch::tensor({{5, 5, 5}}, torch::dtype(torch::kInt32));
    auto src_grid = makeGrid(ijk, device);

    nanovdb::Coord kernel_size(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);
    auto dst_grid = src_grid->convolutionOutput(kernel_size, stride);

    auto topo_gemm =
        ops::gatherScatterSparseConvTopology(*src_grid, *dst_grid, kernel_size, stride);
    auto topo_grouped =
        ops::groupedGemmSparseConvTopology(*src_grid, *dst_grid, kernel_size, stride);

    torch::manual_seed(99);
    auto features = torch::randn({1, C}, torch::dtype(torch::kFloat32).device(device));
    auto weights  = torch::randn({C, C, 3, 3, 3}, torch::dtype(torch::kFloat32).device(device));

    auto ref = ops::gatherScatterSparseConv(features, weights, topo_gemm);
    auto out = ops::groupedGemmSparseConv(features, weights, topo_grouped);

    EXPECT_EQ(out.sizes(), ref.sizes());
    EXPECT_TRUE(torch::allclose(out.cpu(), ref.cpu(), kRtol, kAtol))
        << "Different-grid forward mismatch at C=" << C;
}

// =============================================================================
// Backward tests -- grad_features and grad_weights vs GatherScatter reference
// =============================================================================

class GroupedGemmBackwardTest : public ::testing::TestWithParam<int64_t> {};

TEST_P(GroupedGemmBackwardTest, SameGridStride1) {
    if (!cudaIsAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }
    torch::Device device(torch::kCUDA, 0);
    int64_t const C = GetParam();

    auto ijk = torch::tensor(
        {{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1}, {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}},
        torch::dtype(torch::kInt32));
    auto grid = makeGrid(ijk, device);
    int64_t N = 8;

    nanovdb::Coord kernel_size(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);

    auto topo_gemm    = ops::gatherScatterSparseConvTopology(*grid, *grid, kernel_size, stride);
    auto topo_grouped = ops::groupedGemmSparseConvTopology(*grid, *grid, kernel_size, stride);

    torch::manual_seed(77);
    auto features    = torch::randn({N, C}, torch::dtype(torch::kFloat32).device(device));
    auto weights     = torch::randn({C, C, 3, 3, 3}, torch::dtype(torch::kFloat32).device(device));
    auto grad_output = torch::randn({N, C}, torch::dtype(torch::kFloat32).device(device));

    auto [gf_ref, gw_ref] =
        ops::gatherScatterSparseConvBackward(grad_output, features, weights, topo_gemm);
    auto [gf_out, gw_out] =
        ops::groupedGemmSparseConvBackward(grad_output, features, weights, topo_grouped);

    EXPECT_EQ(gf_out.sizes(), gf_ref.sizes());
    EXPECT_EQ(gw_out.sizes(), gw_ref.sizes());

    EXPECT_TRUE(torch::allclose(gf_out.cpu(), gf_ref.cpu(), kRtol, kAtol))
        << "Backward grad_features mismatch at C=" << C;
    EXPECT_TRUE(torch::allclose(gw_out.cpu(), gw_ref.cpu(), kRtol, kAtol))
        << "Backward grad_weights mismatch at C=" << C;
}

// =============================================================================
// Transposed forward tests
// =============================================================================

class GroupedGemmTransposeTest : public ::testing::TestWithParam<int64_t> {};

TEST_P(GroupedGemmTransposeTest, TransposeForwardSameGrid) {
    if (!cudaIsAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }
    torch::Device device(torch::kCUDA, 0);
    int64_t const C = GetParam();

    auto ijk = torch::tensor(
        {{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1}, {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}},
        torch::dtype(torch::kInt32));
    auto grid = makeGrid(ijk, device);
    int64_t N = 8;

    nanovdb::Coord kernel_size(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);

    auto topo_gemm =
        ops::gatherScatterSparseConvTransposeTopology(*grid, *grid, kernel_size, stride);
    auto topo_grouped =
        ops::groupedGemmSparseConvTransposeTopology(*grid, *grid, kernel_size, stride);

    torch::manual_seed(55);
    auto features = torch::randn({N, C}, torch::dtype(torch::kFloat32).device(device));
    auto weights  = torch::randn({C, C, 3, 3, 3}, torch::dtype(torch::kFloat32).device(device));

    auto ref = ops::gatherScatterSparseConvTranspose(features, weights, topo_gemm);
    auto out = ops::groupedGemmSparseConvTranspose(features, weights, topo_grouped);

    EXPECT_EQ(out.sizes(), ref.sizes());
    EXPECT_TRUE(torch::allclose(out.cpu(), ref.cpu(), kRtol, kAtol))
        << "Transpose forward mismatch at C=" << C;
}

// =============================================================================
// Transposed backward tests
// =============================================================================

TEST_P(GroupedGemmTransposeTest, TransposeBackwardSameGrid) {
    if (!cudaIsAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }
    torch::Device device(torch::kCUDA, 0);
    int64_t const C = GetParam();

    auto ijk = torch::tensor(
        {{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1}, {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}},
        torch::dtype(torch::kInt32));
    auto grid = makeGrid(ijk, device);
    int64_t N = 8;

    nanovdb::Coord kernel_size(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);

    auto topo_gemm =
        ops::gatherScatterSparseConvTransposeTopology(*grid, *grid, kernel_size, stride);
    auto topo_grouped =
        ops::groupedGemmSparseConvTransposeTopology(*grid, *grid, kernel_size, stride);

    torch::manual_seed(66);
    auto features    = torch::randn({N, C}, torch::dtype(torch::kFloat32).device(device));
    auto weights     = torch::randn({C, C, 3, 3, 3}, torch::dtype(torch::kFloat32).device(device));
    auto grad_output = torch::randn({N, C}, torch::dtype(torch::kFloat32).device(device));

    auto [gf_ref, gw_ref] =
        ops::gatherScatterSparseConvTransposeBackward(grad_output, features, weights, topo_gemm);
    auto [gf_out, gw_out] =
        ops::groupedGemmSparseConvTransposeBackward(grad_output, features, weights, topo_grouped);

    EXPECT_EQ(gf_out.sizes(), gf_ref.sizes());
    EXPECT_EQ(gw_out.sizes(), gw_ref.sizes());

    EXPECT_TRUE(torch::allclose(gf_out.cpu(), gf_ref.cpu(), kRtol, kAtol))
        << "Transpose backward grad_features mismatch at C=" << C;
    EXPECT_TRUE(torch::allclose(gw_out.cpu(), gw_ref.cpu(), kRtol, kAtol))
        << "Transpose backward grad_weights mismatch at C=" << C;
}

// =============================================================================
// Strided forward test
// =============================================================================

class GroupedGemmStridedTest : public ::testing::TestWithParam<int64_t> {};

TEST_P(GroupedGemmStridedTest, StridedForward) {
    if (!cudaIsAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }
    torch::Device device(torch::kCUDA, 0);
    int64_t const C = GetParam();

    auto src_grid = makeStridedTestGrid(device);

    nanovdb::Coord kernel_size(3, 3, 3);
    nanovdb::Coord stride(2, 2, 2);

    auto dst_grid = src_grid->convolutionOutput(kernel_size, stride);

    auto topo_gemm =
        ops::gatherScatterSparseConvTopology(*src_grid, *dst_grid, kernel_size, stride);
    auto topo_grouped =
        ops::groupedGemmSparseConvTopology(*src_grid, *dst_grid, kernel_size, stride);

    int64_t S = topo_gemm.feature_total_voxels;

    torch::manual_seed(88);
    auto features = torch::randn({S, C}, torch::dtype(torch::kFloat32).device(device));
    auto weights  = torch::randn({C, C, 3, 3, 3}, torch::dtype(torch::kFloat32).device(device));

    auto ref = ops::gatherScatterSparseConv(features, weights, topo_gemm);
    auto out = ops::groupedGemmSparseConv(features, weights, topo_grouped);

    EXPECT_EQ(out.sizes(), ref.sizes());
    EXPECT_TRUE(torch::allclose(out.cpu(), ref.cpu(), kRtol, kAtol))
        << "Strided forward mismatch at C=" << C;
}

TEST_P(GroupedGemmStridedTest, StridedBackward) {
    if (!cudaIsAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }
    torch::Device device(torch::kCUDA, 0);
    int64_t const C = GetParam();

    auto src_grid = makeStridedTestGrid(device);

    nanovdb::Coord kernel_size(3, 3, 3);
    nanovdb::Coord stride(2, 2, 2);

    auto dst_grid = src_grid->convolutionOutput(kernel_size, stride);

    auto topo_gemm =
        ops::gatherScatterSparseConvTopology(*src_grid, *dst_grid, kernel_size, stride);
    auto topo_grouped =
        ops::groupedGemmSparseConvTopology(*src_grid, *dst_grid, kernel_size, stride);

    int64_t S = topo_gemm.feature_total_voxels;
    int64_t D = topo_gemm.output_total_voxels;

    torch::manual_seed(89);
    auto features    = torch::randn({S, C}, torch::dtype(torch::kFloat32).device(device));
    auto weights     = torch::randn({C, C, 3, 3, 3}, torch::dtype(torch::kFloat32).device(device));
    auto grad_output = torch::randn({D, C}, torch::dtype(torch::kFloat32).device(device));

    auto [gf_ref, gw_ref] =
        ops::gatherScatterSparseConvBackward(grad_output, features, weights, topo_gemm);
    auto [gf_out, gw_out] =
        ops::groupedGemmSparseConvBackward(grad_output, features, weights, topo_grouped);

    EXPECT_EQ(gf_out.sizes(), gf_ref.sizes());
    EXPECT_EQ(gw_out.sizes(), gw_ref.sizes());

    EXPECT_TRUE(torch::allclose(gf_out.cpu(), gf_ref.cpu(), kRtol, kAtol))
        << "Strided backward grad_features mismatch at C=" << C;
    EXPECT_TRUE(torch::allclose(gw_out.cpu(), gw_ref.cpu(), kRtol, kAtol))
        << "Strided backward grad_weights mismatch at C=" << C;
}

// =============================================================================
// Test instantiation -- parameterized by channel count (multiples of 32)
// =============================================================================

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    ChannelSizes,
    GroupedGemmForwardTest,
    ::testing::Values(32, 64, 128),
    [](const auto &info) { return "C" + std::to_string(info.param); });

INSTANTIATE_TEST_SUITE_P(
    ChannelSizes,
    GroupedGemmBackwardTest,
    ::testing::Values(32, 64, 128),
    [](const auto &info) { return "C" + std::to_string(info.param); });

INSTANTIATE_TEST_SUITE_P(
    ChannelSizes,
    GroupedGemmTransposeTest,
    ::testing::Values(32, 64, 128),
    [](const auto &info) { return "C" + std::to_string(info.param); });

INSTANTIATE_TEST_SUITE_P(
    ChannelSizes,
    GroupedGemmStridedTest,
    ::testing::Values(32, 64, 128),
    [](const auto &info) { return "C" + std::to_string(info.param); });
// clang-format on
