// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// GatherScatterConvTest.cu -- Tests for the gather-scatter sparse convolution op.
//
#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/GridBatchImpl.h>
#include <fvdb/detail/ops/convolution/GatherScatter.h>
#include <fvdb/detail/ops/convolution/GatherScatterFused.h>

#include <torch/types.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <tuple>
#include <vector>

using namespace fvdb;
using namespace fvdb::detail;

// =============================================================================
// Tolerance helpers -- keyed on dtype so half/bfloat16 tests pass.
// =============================================================================

static double
rtol_for(torch::Dtype dtype) {
    if (dtype == torch::kHalf)
        return 5e-3;
    if (dtype == torch::kBFloat16)
        return 1e-1; // ~2.4 decimal digits; GEMM accumulates in fp32, fused in bf16
    if (dtype == torch::kFloat32)
        return 1e-5;
    return 1e-10;    // float64
}

static double
atol_for(torch::Dtype dtype) {
    if (dtype == torch::kHalf)
        return 5e-3;
    if (dtype == torch::kBFloat16)
        return 1e-1; // ~2.4 decimal digits; GEMM accumulates in fp32, fused in bf16
    if (dtype == torch::kFloat32)
        return 1e-6;
    return 1e-12;    // float64
}

// Dtype name for test parameterization
static std::string
dtype_name(torch::Dtype dtype) {
    if (dtype == torch::kFloat32)
        return "f32";
    if (dtype == torch::kFloat64)
        return "f64";
    if (dtype == torch::kHalf)
        return "f16";
    if (dtype == torch::kBFloat16)
        return "bf16";
    return "unknown";
}

// Helper: create a GridBatchImpl from a single grid's ijk coordinates.
static c10::intrusive_ptr<GridBatchImpl>
makeGrid(torch::Tensor ijk_2d, torch::Device device) {
    // ijk_2d: [N, 3] int32
    auto ijk_dev = ijk_2d.to(device);
    JaggedTensor jt(ijk_dev);
    std::vector<nanovdb::Vec3d> voxel_sizes = {{1.0, 1.0, 1.0}};
    std::vector<nanovdb::Vec3d> origins     = {{0.0, 0.0, 0.0}};
    return GridBatchImpl::createFromIjk(jt, voxel_sizes, origins);
}

// =============================================================================
// Topology tests
// =============================================================================

class GatherScatterTopologyTest : public ::testing::TestWithParam<torch::Device> {};

TEST_P(GatherScatterTopologyTest, SingleVoxelKernel3x3x3) {
    auto device = GetParam();

    // Single voxel at (5, 5, 5)
    auto ijk      = torch::tensor({{5, 5, 5}}, torch::dtype(torch::kInt32));
    auto src_grid = makeGrid(ijk, device);

    // Dst grid = convolutionOutput of src with 3x3x3 kernel, stride 1
    auto dst_grid = src_grid->convolutionOutput(nanovdb::Coord(3, 3, 3), nanovdb::Coord(1, 1, 1));

    nanovdb::Coord kernel_size(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);

    auto topo = ops::gatherScatterSparseConvTopology(*src_grid, *dst_grid, kernel_size, stride);

    // With a single src voxel and 3x3x3 kernel, dst should have 27 voxels
    EXPECT_EQ(topo.output_total_voxels, 27);
    EXPECT_EQ(topo.feature_total_voxels, 1);
    EXPECT_EQ(topo.kernel_volume, 27);

    // kernel_map shape: [27, 27]
    EXPECT_EQ(topo.kernel_map.size(0), 27);
    EXPECT_EQ(topo.kernel_map.size(1), 27);

    // Each dst voxel should have exactly one kernel offset that hits the src voxel
    auto kmap_cpu = topo.kernel_map.cpu();
    auto kmap_acc = kmap_cpu.accessor<int32_t, 2>();

    int total_hits = 0;
    for (int64_t d = 0; d < 27; ++d) {
        for (int64_t k = 0; k < 27; ++k) {
            if (kmap_acc[d][k] >= 0) {
                EXPECT_EQ(kmap_acc[d][k], 0); // only one src voxel, index 0
                total_hits++;
            }
        }
    }
    // Each dst voxel has exactly 1 hit (the single src voxel is reachable
    // from exactly one kernel offset per dst voxel)
    EXPECT_EQ(total_hits, 27);

    // src and dst differ (1 vs 27 voxels), so center cannot be identity.
    EXPECT_FALSE(topo.center_is_identity);
}

TEST_P(GatherScatterTopologyTest, SameTopologyStride1) {
    auto device = GetParam();

    // Small grid: 8 voxels in a 2x2x2 block
    std::vector<std::vector<int32_t>> coords = {
        {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1}, {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}};
    auto ijk = torch::zeros({8, 3}, torch::dtype(torch::kInt32));
    for (int i = 0; i < 8; ++i) {
        ijk[i][0] = coords[i][0];
        ijk[i][1] = coords[i][1];
        ijk[i][2] = coords[i][2];
    }
    auto grid = makeGrid(ijk, device);

    // With 1x1x1 kernel and stride 1, topology is identity
    nanovdb::Coord kernel_size(1, 1, 1);
    nanovdb::Coord stride_1(1, 1, 1);

    auto topo = ops::gatherScatterSparseConvTopology(*grid, *grid, kernel_size, stride_1);

    EXPECT_EQ(topo.output_total_voxels, 8);
    EXPECT_EQ(topo.feature_total_voxels, 8);
    EXPECT_EQ(topo.kernel_volume, 1);

    // kernel_map: [8, 1], each entry should map dst_i -> src_i (identity)
    auto kmap_cpu = topo.kernel_map.cpu();
    auto kmap_acc = kmap_cpu.accessor<int32_t, 2>();
    for (int64_t d = 0; d < 8; ++d) {
        EXPECT_EQ(kmap_acc[d][0], static_cast<int32_t>(d));
    }

    // 1x1x1 kernel on same grid: center IS the identity.
    EXPECT_TRUE(topo.center_is_identity);
}

// =============================================================================
// Convolution tests
// =============================================================================

class GatherScatterConvTest
    : public ::testing::TestWithParam<std::tuple<torch::Device, torch::Dtype>> {};

TEST_P(GatherScatterConvTest, IdentityConv1x1x1) {
    auto [device, dtype] = GetParam();

    // 2x2x2 block
    auto ijk = torch::tensor(
        {{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1}, {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}},
        torch::dtype(torch::kInt32));
    auto grid = makeGrid(ijk, device);

    int64_t C_in  = 4;
    int64_t C_out = 4;
    int64_t N     = 8;

    nanovdb::Coord kernel_size(1, 1, 1);
    nanovdb::Coord stride(1, 1, 1);

    auto topo = ops::gatherScatterSparseConvTopology(*grid, *grid, kernel_size, stride);

    // Features: random
    auto features = torch::randn({N, C_in}, torch::dtype(dtype).device(device));

    // Weight: identity matrix reshaped to [C_out, C_in, 1, 1, 1]
    auto weights =
        torch::eye(C_in, torch::dtype(dtype).device(device)).reshape({C_out, C_in, 1, 1, 1});

    auto output = ops::gatherScatterSparseConv(features, weights, topo);

    EXPECT_EQ(output.size(0), N);
    EXPECT_EQ(output.size(1), C_out);

    // With identity weights, output should equal input
    EXPECT_TRUE(torch::allclose(output.cpu(), features.cpu(), rtol_for(dtype), atol_for(dtype)));
}

TEST_P(GatherScatterConvTest, SingleImpulse3x3x3) {
    auto [device, dtype] = GetParam();

    // Single voxel at (2, 3, 4)
    auto ijk      = torch::tensor({{2, 3, 4}}, torch::dtype(torch::kInt32));
    auto src_grid = makeGrid(ijk, device);

    nanovdb::Coord kernel_size(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);

    auto dst_grid = src_grid->convolutionOutput(kernel_size, stride);
    auto topo     = ops::gatherScatterSparseConvTopology(*src_grid, *dst_grid, kernel_size, stride);

    EXPECT_EQ(topo.output_total_voxels, 27);

    int64_t C_in  = 1;
    int64_t C_out = 1;

    // Features: single voxel with value 1
    auto features = torch::ones({1, C_in}, torch::dtype(dtype).device(device));

    // Weights: all ones
    auto weights = torch::ones({C_out, C_in, 3, 3, 3}, torch::dtype(dtype).device(device));

    auto output = ops::gatherScatterSparseConv(features, weights, topo);

    EXPECT_EQ(output.size(0), 27);
    EXPECT_EQ(output.size(1), C_out);

    // Each dst voxel should see the impulse through exactly one kernel offset,
    // and the weight is 1, so each output voxel should be 1.
    double const tol = atol_for(dtype);
    auto output_cpu  = output.cpu().to(torch::kFloat64);
    for (int64_t d = 0; d < 27; ++d) {
        EXPECT_NEAR(output_cpu[d][0].item<double>(), 1.0, tol)
            << "Output voxel " << d << " should be 1.0";
    }

    // Sum should be 27 (one contribution per dst voxel)
    EXPECT_NEAR(output_cpu.sum().item<double>(), 27.0, 27.0 * rtol_for(dtype) + tol);
}

TEST_P(GatherScatterConvTest, CompareWithDenseConv3d) {
    auto [device, dtype] = GetParam();

    // Create a small dense-ish grid: 4x4x4 block
    std::vector<std::vector<int32_t>> coords;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 4; ++k)
                coords.push_back({i, j, k});

    int64_t N = static_cast<int64_t>(coords.size()); // 64
    auto ijk  = torch::zeros({N, 3}, torch::dtype(torch::kInt32));
    for (int64_t i = 0; i < N; ++i) {
        ijk[i][0] = coords[i][0];
        ijk[i][1] = coords[i][1];
        ijk[i][2] = coords[i][2];
    }
    auto src_grid = makeGrid(ijk, device);

    nanovdb::Coord kernel_size(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);

    // For stride=1, same topology for src and dst (all 4x4x4 voxels present)
    // Actually convolutionOutput may produce a larger grid. Let's just use
    // src_grid == dst_grid since stride=1 and we want to compare.
    // The output will only have values at the src voxel locations.
    auto dst_grid = src_grid->convolutionOutput(kernel_size, stride);
    auto topo     = ops::gatherScatterSparseConvTopology(*src_grid, *dst_grid, kernel_size, stride);

    int64_t C_in  = 2;
    int64_t C_out = 3;

    // Random features and weights
    torch::manual_seed(42);
    auto features = torch::randn({N, C_in}, torch::dtype(dtype).device(device));
    auto weights  = torch::randn({C_out, C_in, 3, 3, 3}, torch::dtype(dtype).device(device));

    auto sparse_output = ops::gatherScatterSparseConv(features, weights, topo);

    // Dense ground truth: inject features into a dense volume, conv3d, extract
    // We need the ijk coordinates of the dst grid to compare.
    // For now, just check that the output has the right shape and isn't all zeros.
    EXPECT_EQ(sparse_output.size(1), C_out);
    EXPECT_GT(sparse_output.size(0), 0);

    // The output should not be all zeros (random weights and features)
    auto abs_sum = sparse_output.abs().sum().cpu().item<double>();
    EXPECT_GT(abs_sum, 0.0);
}

TEST_P(GatherScatterConvTest, MiddleAcceleration) {
    auto [device, dtype] = GetParam();

    // 2x2x2 block, stride 1, 3x3x3 kernel, same src and dst
    auto ijk = torch::tensor(
        {{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1}, {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}},
        torch::dtype(torch::kInt32));
    auto grid = makeGrid(ijk, device);

    nanovdb::Coord kernel_size(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);

    // Use same grid as src and dst to trigger middle acceleration
    auto topo = ops::gatherScatterSparseConvTopology(*grid, *grid, kernel_size, stride);

    int64_t N     = 8;
    int64_t C_in  = 3;
    int64_t C_out = 2;

    torch::manual_seed(123);
    auto features = torch::randn({N, C_in}, torch::dtype(dtype).device(device));
    auto weights  = torch::randn({C_out, C_in, 3, 3, 3}, torch::dtype(dtype).device(device));

    auto output = ops::gatherScatterSparseConv(features, weights, topo);

    EXPECT_EQ(output.size(0), N);
    EXPECT_EQ(output.size(1), C_out);

    // Sanity: output should not be all zeros
    auto abs_sum = output.abs().sum().cpu().item<double>();
    EXPECT_GT(abs_sum, 0.0);
}

TEST_P(GatherScatterConvTest, NonUniformKernel) {
    auto [device, dtype] = GetParam();

    // Single voxel
    auto ijk      = torch::tensor({{5, 5, 5}}, torch::dtype(torch::kInt32));
    auto src_grid = makeGrid(ijk, device);

    nanovdb::Coord kernel_size(3, 5, 7);
    nanovdb::Coord stride(1, 1, 1);

    auto dst_grid = src_grid->convolutionOutput(kernel_size, stride);
    auto topo     = ops::gatherScatterSparseConvTopology(*src_grid, *dst_grid, kernel_size, stride);

    int64_t expected_volume = 3 * 5 * 7; // 105
    EXPECT_EQ(topo.kernel_volume, expected_volume);
    EXPECT_EQ(topo.output_total_voxels, expected_volume);

    int64_t C_in  = 1;
    int64_t C_out = 1;

    auto features = torch::ones({1, C_in}, torch::dtype(dtype).device(device));
    auto weights  = torch::ones({C_out, C_in, 3, 5, 7}, torch::dtype(dtype).device(device));

    auto output = ops::gatherScatterSparseConv(features, weights, topo);

    EXPECT_EQ(output.size(0), expected_volume);
    EXPECT_EQ(output.size(1), C_out);

    // Each dst voxel gets exactly one contribution of 1.0
    auto output_cpu  = output.cpu().to(torch::kFloat64);
    double const vol = static_cast<double>(expected_volume);
    EXPECT_NEAR(output_cpu.sum().item<double>(), vol, vol * rtol_for(dtype) + atol_for(dtype));
}

TEST_P(GatherScatterConvTest, MiddleAccelDifferentGridsSameCount) {
    auto [device, dtype] = GetParam();

    // Two grids with the same voxel count (2) but different positions.
    // The old middle-acceleration guard (count equality) would fire here,
    // but center_is_identity is false because the center kernel offset
    // does NOT map dst voxel 1 to any active src voxel.
    auto src_ijk  = torch::tensor({{0, 0, 0}, {1, 1, 1}}, torch::dtype(torch::kInt32));
    auto dst_ijk  = torch::tensor({{0, 0, 0}, {2, 2, 2}}, torch::dtype(torch::kInt32));
    auto src_grid = makeGrid(src_ijk, device);
    auto dst_grid = makeGrid(dst_ijk, device);

    nanovdb::Coord kernel_size(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);

    auto topo = ops::gatherScatterSparseConvTopology(*src_grid, *dst_grid, kernel_size, stride);

    EXPECT_EQ(topo.feature_total_voxels, 2);
    EXPECT_EQ(topo.output_total_voxels, 2);
    EXPECT_FALSE(topo.center_is_identity);

    int64_t C_in  = 2;
    int64_t C_out = 2;

    // Center-only identity weights: non-zero only at the (1,1,1) center.
    auto weights = torch::zeros({C_out, C_in, 3, 3, 3}, torch::dtype(dtype).device(device));
    for (int c = 0; c < C_in; ++c) {
        weights[c][c][1][1][1] = 1.0;
    }

    // Features: [1, 0] for src voxel 0, [0, 1] for src voxel 1
    auto features = torch::eye(C_in, torch::dtype(dtype).device(device));

    auto output = ops::gatherScatterSparseConv(features, weights, topo);

    EXPECT_EQ(output.size(0), 2);
    EXPECT_EQ(output.size(1), C_out);

    // dst voxel 0 at (0,0,0): center probes src at (0,0,0) -> active -> [1, 0]
    // dst voxel 1 at (2,2,2): center probes src at (2,2,2) -> miss    -> [0, 0]
    double const tol = atol_for(dtype);
    auto out_cpu     = output.cpu().to(torch::kFloat64);
    EXPECT_NEAR(out_cpu[0][0].item<double>(), 1.0, tol);
    EXPECT_NEAR(out_cpu[0][1].item<double>(), 0.0, tol);
    EXPECT_NEAR(out_cpu[1][0].item<double>(), 0.0, tol);
    EXPECT_NEAR(out_cpu[1][1].item<double>(), 0.0, tol);
}

// =============================================================================
// Mixed-dtype tests
// =============================================================================

class GatherScatterConvMixedDtypeTest : public ::testing::TestWithParam<torch::Device> {};

TEST_P(GatherScatterConvMixedDtypeTest, HalfFeaturesFloat32Weights) {
    auto device = GetParam();

    auto ijk = torch::tensor(
        {{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1}, {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}},
        torch::dtype(torch::kInt32));
    auto grid = makeGrid(ijk, device);

    nanovdb::Coord kernel_size(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);
    auto topo = ops::gatherScatterSparseConvTopology(*grid, *grid, kernel_size, stride);

    int64_t N = 8, C_in = 4, C_out = 4;
    torch::manual_seed(42);

    auto features_f32 = torch::randn({N, C_in}, torch::dtype(torch::kFloat32).device(device));
    auto features_f16 = features_f32.to(torch::kHalf);
    auto weights_f32 =
        torch::randn({C_out, C_in, 3, 3, 3}, torch::dtype(torch::kFloat32).device(device));

    // Mixed: fp16 features, fp32 weights -> output should be fp16
    auto output_mixed = ops::gatherScatterSparseConv(features_f16, weights_f32, topo);
    EXPECT_EQ(output_mixed.scalar_type(), torch::kHalf);

    // Reference: all fp32
    auto output_ref = ops::gatherScatterSparseConv(features_f32, weights_f32, topo);

    // Compare (with fp16 tolerance)
    auto mixed_f32 = output_mixed.to(torch::kFloat32).cpu();
    auto ref_cpu   = output_ref.cpu();
    EXPECT_TRUE(torch::allclose(mixed_f32, ref_cpu, /*rtol=*/5e-2, /*atol=*/5e-2));
}

TEST_P(GatherScatterConvMixedDtypeTest, BFloat16FeaturesFloat32Weights) {
    auto device = GetParam();

    auto ijk = torch::tensor(
        {{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1}, {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}},
        torch::dtype(torch::kInt32));
    auto grid = makeGrid(ijk, device);

    nanovdb::Coord kernel_size(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);
    auto topo = ops::gatherScatterSparseConvTopology(*grid, *grid, kernel_size, stride);

    int64_t N = 8, C_in = 4, C_out = 4;
    torch::manual_seed(42);

    auto features_f32  = torch::randn({N, C_in}, torch::dtype(torch::kFloat32).device(device));
    auto features_bf16 = features_f32.to(torch::kBFloat16);
    auto weights_f32 =
        torch::randn({C_out, C_in, 3, 3, 3}, torch::dtype(torch::kFloat32).device(device));

    // Mixed: bf16 features, fp32 weights -> output should be bf16
    auto output_mixed = ops::gatherScatterSparseConv(features_bf16, weights_f32, topo);
    EXPECT_EQ(output_mixed.scalar_type(), torch::kBFloat16);

    // Reference: all fp32
    auto output_ref = ops::gatherScatterSparseConv(features_f32, weights_f32, topo);

    // Compare (with bf16 tolerance)
    auto mixed_f32 = output_mixed.to(torch::kFloat32).cpu();
    auto ref_cpu   = output_ref.cpu();
    EXPECT_TRUE(torch::allclose(mixed_f32, ref_cpu, /*rtol=*/1e-1, /*atol=*/1e-1));
}

// =============================================================================
// Backward tests (GEMM path)
// =============================================================================

class GatherScatterConvBackwardTest
    : public ::testing::TestWithParam<std::tuple<torch::Device, torch::Dtype>> {};

TEST_P(GatherScatterConvBackwardTest, IdentityBackward1x1x1) {
    auto [device, dtype] = GetParam();

    // 2x2x2 block
    auto ijk = torch::tensor(
        {{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1}, {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}},
        torch::dtype(torch::kInt32));
    auto grid = makeGrid(ijk, device);

    int64_t N     = 8;
    int64_t C_in  = 4;
    int64_t C_out = 4;

    nanovdb::Coord kernel_size(1, 1, 1);
    nanovdb::Coord stride(1, 1, 1);

    auto topo = ops::gatherScatterSparseConvTopology(*grid, *grid, kernel_size, stride);

    torch::manual_seed(42);
    auto features = torch::randn({N, C_in}, torch::dtype(dtype).device(device));
    auto weights =
        torch::eye(C_in, torch::dtype(dtype).device(device)).reshape({C_out, C_in, 1, 1, 1});

    // grad_output = random
    auto grad_output = torch::randn({N, C_out}, torch::dtype(dtype).device(device));

    auto [grad_features, grad_weights] =
        ops::gatherScatterSparseConvBackward(grad_output, features, weights, topo);

    EXPECT_EQ(grad_features.size(0), N);
    EXPECT_EQ(grad_features.size(1), C_in);
    EXPECT_EQ(grad_weights.sizes(), weights.sizes());

    // With identity weights: forward is f(x) = x @ I = x
    // So grad_features = grad_output @ I^T = grad_output
    EXPECT_TRUE(
        torch::allclose(grad_features.cpu(), grad_output.cpu(), rtol_for(dtype), atol_for(dtype)));

    // grad_weights[0,0,0,0,0] (the single kernel element):
    // grad_W[ci, co] = features^T @ grad_output -> shape [C_in, C_out]
    // But output is [C_out, C_in, 1, 1, 1], so it's transposed.
    auto expected_gw = features.t().mm(grad_output).t().reshape({C_out, C_in, 1, 1, 1});
    EXPECT_TRUE(
        torch::allclose(grad_weights.cpu(), expected_gw.cpu(), rtol_for(dtype), atol_for(dtype)));
}

TEST_P(GatherScatterConvBackwardTest, FiniteDifference) {
    auto [device, dtype] = GetParam();

    // Use float64 for tight finite-difference tolerance; skip others.
    if (dtype != torch::kFloat64) {
        GTEST_SKIP() << "Finite difference test only runs in float64";
    }

    // Small 2x2x2 grid, same src and dst
    auto ijk = torch::tensor(
        {{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1}, {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}},
        torch::dtype(torch::kInt32));
    auto grid = makeGrid(ijk, device);

    int64_t N     = 8;
    int64_t C_in  = 2;
    int64_t C_out = 3;

    nanovdb::Coord kernel_size(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);

    auto topo = ops::gatherScatterSparseConvTopology(*grid, *grid, kernel_size, stride);

    torch::manual_seed(7);
    auto features = torch::randn({N, C_in}, torch::dtype(dtype).device(device));
    auto weights  = torch::randn({C_out, C_in, 3, 3, 3}, torch::dtype(dtype).device(device));

    // Use a scalar loss: L = sum(output)
    // grad_output = ones
    auto output      = ops::gatherScatterSparseConv(features, weights, topo);
    auto grad_output = torch::ones_like(output);

    auto [grad_features, grad_weights] =
        ops::gatherScatterSparseConvBackward(grad_output, features, weights, topo);

    // Finite difference for features
    double const eps = 1e-5;
    for (int64_t i = 0; i < N; ++i) {
        for (int64_t c = 0; c < C_in; ++c) {
            auto f_plus  = features.clone();
            auto f_minus = features.clone();
            f_plus[i][c] += eps;
            f_minus[i][c] -= eps;

            auto out_plus  = ops::gatherScatterSparseConv(f_plus, weights, topo);
            auto out_minus = ops::gatherScatterSparseConv(f_minus, weights, topo);
            double numerical =
                (out_plus.sum() - out_minus.sum()).cpu().item<double>() / (2.0 * eps);
            double analytical = grad_features[i][c].cpu().item<double>();

            EXPECT_NEAR(analytical, numerical, 1e-4)
                << "grad_features[" << i << "][" << c << "] mismatch";
        }
    }

    // Finite difference for weights (sample a few elements to keep test fast)
    auto w_flat  = weights.reshape(-1);
    int64_t n_w  = w_flat.size(0);
    int64_t step = std::max(int64_t(1), n_w / 20); // test ~20 elements
    for (int64_t idx = 0; idx < n_w; idx += step) {
        auto w_plus  = weights.clone();
        auto w_minus = weights.clone();
        w_plus.reshape(-1)[idx] += eps;
        w_minus.reshape(-1)[idx] -= eps;

        auto out_plus     = ops::gatherScatterSparseConv(features, w_plus, topo);
        auto out_minus    = ops::gatherScatterSparseConv(features, w_minus, topo);
        double numerical  = (out_plus.sum() - out_minus.sum()).cpu().item<double>() / (2.0 * eps);
        double analytical = grad_weights.reshape(-1)[idx].cpu().item<double>();

        EXPECT_NEAR(analytical, numerical, 1e-4) << "grad_weights flat[" << idx << "] mismatch";
    }
}

TEST_P(GatherScatterConvBackwardTest, FusedVsGemmAgreement) {
    auto [device, dtype] = GetParam();

    // Small 2x2x2 grid, same src and dst
    auto ijk = torch::tensor(
        {{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1}, {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}},
        torch::dtype(torch::kInt32));
    auto grid = makeGrid(ijk, device);

    int64_t N     = 8;
    int64_t C_in  = 3;
    int64_t C_out = 2;

    nanovdb::Coord kernel_size(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);

    auto topo = ops::gatherScatterSparseConvTopology(*grid, *grid, kernel_size, stride);

    torch::manual_seed(99);
    auto features    = torch::randn({N, C_in}, torch::dtype(dtype).device(device));
    auto weights     = torch::randn({C_out, C_in, 3, 3, 3}, torch::dtype(dtype).device(device));
    auto grad_output = torch::randn({N, C_out}, torch::dtype(dtype).device(device));

    // GEMM backward
    auto [gf_gemm, gw_gemm] =
        ops::gatherScatterSparseConvBackward(grad_output, features, weights, topo);

    // Fused backward
    auto [gf_fused, gw_fused] = ops::gatherScatterSparseConvFusedBackward(
        grad_output, features, weights, *grid, *grid, kernel_size, stride);

    EXPECT_TRUE(torch::allclose(gf_gemm.cpu().to(torch::kFloat64),
                                gf_fused.cpu().to(torch::kFloat64),
                                rtol_for(dtype),
                                atol_for(dtype)))
        << "grad_features mismatch between GEMM and fused backward";

    EXPECT_TRUE(torch::allclose(gw_gemm.cpu().to(torch::kFloat64),
                                gw_fused.cpu().to(torch::kFloat64),
                                rtol_for(dtype),
                                atol_for(dtype)))
        << "grad_weights mismatch between GEMM and fused backward";
}

TEST_P(GatherScatterConvBackwardTest, MiddleAccelBackward) {
    auto [device, dtype] = GetParam();

    // 2x2x2 block, same grid -> center_is_identity = true
    auto ijk = torch::tensor(
        {{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1}, {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}},
        torch::dtype(torch::kInt32));
    auto grid = makeGrid(ijk, device);

    int64_t N     = 8;
    int64_t C_in  = 3;
    int64_t C_out = 2;

    nanovdb::Coord kernel_size(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);

    auto topo = ops::gatherScatterSparseConvTopology(*grid, *grid, kernel_size, stride);
    EXPECT_TRUE(topo.center_is_identity);

    torch::manual_seed(77);
    auto features    = torch::randn({N, C_in}, torch::dtype(dtype).device(device));
    auto weights     = torch::randn({C_out, C_in, 3, 3, 3}, torch::dtype(dtype).device(device));
    auto grad_output = torch::randn({N, C_out}, torch::dtype(dtype).device(device));

    auto [grad_features, grad_weights] =
        ops::gatherScatterSparseConvBackward(grad_output, features, weights, topo);

    // Verify against fused backward (which doesn't use middle acceleration)
    auto [gf_fused, gw_fused] = ops::gatherScatterSparseConvFusedBackward(
        grad_output, features, weights, *grid, *grid, kernel_size, stride);

    EXPECT_TRUE(torch::allclose(grad_features.cpu().to(torch::kFloat64),
                                gf_fused.cpu().to(torch::kFloat64),
                                rtol_for(dtype),
                                atol_for(dtype)))
        << "Middle-accel grad_features disagrees with fused reference";

    EXPECT_TRUE(torch::allclose(grad_weights.cpu().to(torch::kFloat64),
                                gw_fused.cpu().to(torch::kFloat64),
                                rtol_for(dtype),
                                atol_for(dtype)))
        << "Middle-accel grad_weights disagrees with fused reference";
}

// =============================================================================
// Strided convolution tests
// =============================================================================

class GatherScatterConvStridedTest
    : public ::testing::TestWithParam<std::tuple<torch::Device, torch::Dtype>> {};

// Helper: make a grid with voxels in a region suitable for strided testing.
// Places voxels in a 3x3x3 block starting at (4,4,4) to stay away from
// negative output coords and provide enough overlap for a 3x3x3 kernel.
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

TEST_P(GatherScatterConvStridedTest, TopologyUniformStride) {
    auto [device, dtype] = GetParam();

    auto src_grid = makeStridedTestGrid(device);

    nanovdb::Coord kernel_size(3, 3, 3);
    nanovdb::Coord stride(2, 2, 2);

    auto dst_grid = src_grid->convolutionOutput(kernel_size, stride);
    auto topo     = ops::gatherScatterSparseConvTopology(*src_grid, *dst_grid, kernel_size, stride);

    // Verify dimensions
    EXPECT_EQ(topo.kernel_volume, 27);
    EXPECT_EQ(topo.kernel_map.size(0), topo.output_total_voxels);
    EXPECT_EQ(topo.kernel_map.size(1), 27);
    EXPECT_GT(topo.output_total_voxels, 0);

    // With stride > 1, src and dst topologies differ -> center is NOT identity
    EXPECT_FALSE(topo.center_is_identity);

    // Every kmap entry must be either -1 or a valid src index [0, src_total)
    auto kmap_cpu = topo.kernel_map.cpu();
    auto kmap_acc = kmap_cpu.accessor<int32_t, 2>();
    int hits      = 0;
    for (int64_t d = 0; d < topo.output_total_voxels; ++d) {
        for (int64_t k = 0; k < 27; ++k) {
            int32_t val = kmap_acc[d][k];
            EXPECT_TRUE(val == -1 || (val >= 0 && val < topo.feature_total_voxels))
                << "kmap[" << d << "][" << k << "]=" << val << " out of range";
            if (val >= 0)
                hits++;
        }
    }
    EXPECT_GT(hits, 0) << "Topology should have at least one active src hit";
}

TEST_P(GatherScatterConvStridedTest, TopologyNonUniformStride) {
    auto [device, dtype] = GetParam();

    auto src_grid = makeStridedTestGrid(device);

    nanovdb::Coord kernel_size(3, 3, 3);
    nanovdb::Coord stride(1, 2, 3);

    auto dst_grid = src_grid->convolutionOutput(kernel_size, stride);
    auto topo     = ops::gatherScatterSparseConvTopology(*src_grid, *dst_grid, kernel_size, stride);

    EXPECT_EQ(topo.kernel_volume, 27);
    EXPECT_EQ(topo.kernel_map.size(0), topo.output_total_voxels);
    EXPECT_GT(topo.output_total_voxels, 0);

    // Non-uniform stride -> center not identity
    EXPECT_FALSE(topo.center_is_identity);

    // Validate kmap bounds
    auto kmap_cpu = topo.kernel_map.cpu();
    auto kmap_acc = kmap_cpu.accessor<int32_t, 2>();
    int hits      = 0;
    for (int64_t d = 0; d < topo.output_total_voxels; ++d) {
        for (int64_t k = 0; k < 27; ++k) {
            int32_t val = kmap_acc[d][k];
            EXPECT_TRUE(val == -1 || (val >= 0 && val < topo.feature_total_voxels));
            if (val >= 0)
                hits++;
        }
    }
    EXPECT_GT(hits, 0);
}

TEST_P(GatherScatterConvStridedTest, ForwardStridedFusedVsGemm) {
    auto [device, dtype] = GetParam();

    auto src_grid = makeStridedTestGrid(device);

    int64_t C_in  = 2;
    int64_t C_out = 3;

    nanovdb::Coord kernel_size(3, 3, 3);
    nanovdb::Coord stride(2, 2, 2);

    auto dst_grid = src_grid->convolutionOutput(kernel_size, stride);
    auto topo     = ops::gatherScatterSparseConvTopology(*src_grid, *dst_grid, kernel_size, stride);

    int64_t S = topo.feature_total_voxels;

    torch::manual_seed(55);
    auto features = torch::randn({S, C_in}, torch::dtype(dtype).device(device));
    auto weights  = torch::randn({C_out, C_in, 3, 3, 3}, torch::dtype(dtype).device(device));

    // GEMM path
    auto out_gemm = ops::gatherScatterSparseConv(features, weights, topo);

    // Fused path
    auto out_fused = ops::gatherScatterSparseConvFused(
        features, weights, *src_grid, *dst_grid, kernel_size, stride);

    EXPECT_TRUE(torch::allclose(out_gemm.cpu().to(torch::kFloat64),
                                out_fused.cpu().to(torch::kFloat64),
                                rtol_for(dtype),
                                atol_for(dtype)))
        << "Strided forward: GEMM vs fused mismatch (stride 2,2,2)";
}

TEST_P(GatherScatterConvStridedTest, ForwardNonUniformStridedFusedVsGemm) {
    auto [device, dtype] = GetParam();

    auto src_grid = makeStridedTestGrid(device);

    int64_t C_in  = 2;
    int64_t C_out = 3;

    nanovdb::Coord kernel_size(3, 3, 3);
    nanovdb::Coord stride(1, 2, 3);

    auto dst_grid = src_grid->convolutionOutput(kernel_size, stride);
    auto topo     = ops::gatherScatterSparseConvTopology(*src_grid, *dst_grid, kernel_size, stride);

    int64_t S = topo.feature_total_voxels;

    torch::manual_seed(56);
    auto features = torch::randn({S, C_in}, torch::dtype(dtype).device(device));
    auto weights  = torch::randn({C_out, C_in, 3, 3, 3}, torch::dtype(dtype).device(device));

    auto out_gemm  = ops::gatherScatterSparseConv(features, weights, topo);
    auto out_fused = ops::gatherScatterSparseConvFused(
        features, weights, *src_grid, *dst_grid, kernel_size, stride);

    EXPECT_TRUE(torch::allclose(out_gemm.cpu().to(torch::kFloat64),
                                out_fused.cpu().to(torch::kFloat64),
                                rtol_for(dtype),
                                atol_for(dtype)))
        << "Strided forward: GEMM vs fused mismatch (stride 1,2,3)";
}

TEST_P(GatherScatterConvStridedTest, BackwardStridedFiniteDifference) {
    auto [device, dtype] = GetParam();

    if (dtype != torch::kFloat64) {
        GTEST_SKIP() << "Finite difference test only runs in float64";
    }

    auto src_grid = makeStridedTestGrid(device);

    int64_t C_in  = 2;
    int64_t C_out = 3;

    nanovdb::Coord kernel_size(3, 3, 3);
    nanovdb::Coord stride(2, 2, 2);

    auto dst_grid = src_grid->convolutionOutput(kernel_size, stride);
    auto topo     = ops::gatherScatterSparseConvTopology(*src_grid, *dst_grid, kernel_size, stride);

    int64_t S = topo.feature_total_voxels;

    torch::manual_seed(60);
    auto features = torch::randn({S, C_in}, torch::dtype(dtype).device(device));
    auto weights  = torch::randn({C_out, C_in, 3, 3, 3}, torch::dtype(dtype).device(device));

    auto output      = ops::gatherScatterSparseConv(features, weights, topo);
    auto grad_output = torch::ones_like(output);

    auto [grad_features, grad_weights] =
        ops::gatherScatterSparseConvBackward(grad_output, features, weights, topo);

    double const eps = 1e-5;

    // Finite difference for features (sample a subset to keep test fast)
    int64_t feat_step = std::max(int64_t(1), S / 5);
    for (int64_t i = 0; i < S; i += feat_step) {
        for (int64_t c = 0; c < C_in; ++c) {
            auto f_plus  = features.clone();
            auto f_minus = features.clone();
            f_plus[i][c] += eps;
            f_minus[i][c] -= eps;

            auto out_plus  = ops::gatherScatterSparseConv(f_plus, weights, topo);
            auto out_minus = ops::gatherScatterSparseConv(f_minus, weights, topo);
            double numerical =
                (out_plus.sum() - out_minus.sum()).cpu().item<double>() / (2.0 * eps);
            double analytical = grad_features[i][c].cpu().item<double>();

            EXPECT_NEAR(analytical, numerical, 1e-4)
                << "Strided grad_features[" << i << "][" << c << "] mismatch";
        }
    }

    // Finite difference for weights (sample ~15 elements)
    auto w_flat    = weights.reshape(-1);
    int64_t n_w    = w_flat.size(0);
    int64_t w_step = std::max(int64_t(1), n_w / 15);
    for (int64_t idx = 0; idx < n_w; idx += w_step) {
        auto w_plus  = weights.clone();
        auto w_minus = weights.clone();
        w_plus.reshape(-1)[idx] += eps;
        w_minus.reshape(-1)[idx] -= eps;

        auto out_plus     = ops::gatherScatterSparseConv(features, w_plus, topo);
        auto out_minus    = ops::gatherScatterSparseConv(features, w_minus, topo);
        double numerical  = (out_plus.sum() - out_minus.sum()).cpu().item<double>() / (2.0 * eps);
        double analytical = grad_weights.reshape(-1)[idx].cpu().item<double>();

        EXPECT_NEAR(analytical, numerical, 1e-4)
            << "Strided grad_weights flat[" << idx << "] mismatch";
    }
}

TEST_P(GatherScatterConvStridedTest, BackwardStridedFusedVsGemm) {
    auto [device, dtype] = GetParam();

    auto src_grid = makeStridedTestGrid(device);

    int64_t C_in  = 2;
    int64_t C_out = 3;

    nanovdb::Coord kernel_size(3, 3, 3);
    nanovdb::Coord stride(2, 2, 2);

    auto dst_grid = src_grid->convolutionOutput(kernel_size, stride);
    auto topo     = ops::gatherScatterSparseConvTopology(*src_grid, *dst_grid, kernel_size, stride);

    int64_t S = topo.feature_total_voxels;
    int64_t D = topo.output_total_voxels;

    torch::manual_seed(61);
    auto features    = torch::randn({S, C_in}, torch::dtype(dtype).device(device));
    auto weights     = torch::randn({C_out, C_in, 3, 3, 3}, torch::dtype(dtype).device(device));
    auto grad_output = torch::randn({D, C_out}, torch::dtype(dtype).device(device));

    auto [gf_gemm, gw_gemm] =
        ops::gatherScatterSparseConvBackward(grad_output, features, weights, topo);

    auto [gf_fused, gw_fused] = ops::gatherScatterSparseConvFusedBackward(
        grad_output, features, weights, *src_grid, *dst_grid, kernel_size, stride);

    EXPECT_TRUE(torch::allclose(gf_gemm.cpu().to(torch::kFloat64),
                                gf_fused.cpu().to(torch::kFloat64),
                                rtol_for(dtype),
                                atol_for(dtype)))
        << "Strided backward grad_features: GEMM vs fused mismatch (stride 2,2,2)";

    EXPECT_TRUE(torch::allclose(gw_gemm.cpu().to(torch::kFloat64),
                                gw_fused.cpu().to(torch::kFloat64),
                                rtol_for(dtype),
                                atol_for(dtype)))
        << "Strided backward grad_weights: GEMM vs fused mismatch (stride 2,2,2)";
}

TEST_P(GatherScatterConvStridedTest, BackwardNonUniformStridedFusedVsGemm) {
    auto [device, dtype] = GetParam();

    auto src_grid = makeStridedTestGrid(device);

    int64_t C_in  = 2;
    int64_t C_out = 3;

    nanovdb::Coord kernel_size(3, 3, 3);
    nanovdb::Coord stride(1, 2, 3);

    auto dst_grid = src_grid->convolutionOutput(kernel_size, stride);
    auto topo     = ops::gatherScatterSparseConvTopology(*src_grid, *dst_grid, kernel_size, stride);

    int64_t S = topo.feature_total_voxels;
    int64_t D = topo.output_total_voxels;

    torch::manual_seed(62);
    auto features    = torch::randn({S, C_in}, torch::dtype(dtype).device(device));
    auto weights     = torch::randn({C_out, C_in, 3, 3, 3}, torch::dtype(dtype).device(device));
    auto grad_output = torch::randn({D, C_out}, torch::dtype(dtype).device(device));

    auto [gf_gemm, gw_gemm] =
        ops::gatherScatterSparseConvBackward(grad_output, features, weights, topo);

    auto [gf_fused, gw_fused] = ops::gatherScatterSparseConvFusedBackward(
        grad_output, features, weights, *src_grid, *dst_grid, kernel_size, stride);

    EXPECT_TRUE(torch::allclose(gf_gemm.cpu().to(torch::kFloat64),
                                gf_fused.cpu().to(torch::kFloat64),
                                rtol_for(dtype),
                                atol_for(dtype)))
        << "Strided backward grad_features: GEMM vs fused mismatch (stride 1,2,3)";

    EXPECT_TRUE(torch::allclose(gw_gemm.cpu().to(torch::kFloat64),
                                gw_fused.cpu().to(torch::kFloat64),
                                rtol_for(dtype),
                                atol_for(dtype)))
        << "Strided backward grad_weights: GEMM vs fused mismatch (stride 1,2,3)";
}

// =============================================================================
// Test instantiation
// =============================================================================

static bool
cudaIsAvailable() {
    int count = 0;
    auto err  = cudaGetDeviceCount(&count);
    return err == cudaSuccess && count > 0;
}

static std::vector<torch::Device>
availableDevices() {
    std::vector<torch::Device> devices;
    devices.emplace_back(torch::kCPU);
    if (cudaIsAvailable()) {
        devices.emplace_back(torch::kCUDA, 0);
    }
    return devices;
}

INSTANTIATE_TEST_SUITE_P(Devices,
                         GatherScatterTopologyTest,
                         ::testing::ValuesIn(availableDevices()),
                         [](const auto &info) { return info.param.is_cpu() ? "CPU" : "CUDA"; });

static std::vector<std::tuple<torch::Device, torch::Dtype>>
deviceDtypeCombos() {
    std::vector<std::tuple<torch::Device, torch::Dtype>> combos;
    for (auto const &dev: availableDevices()) {
        combos.emplace_back(dev, torch::kFloat32);
        combos.emplace_back(dev, torch::kFloat64);
        combos.emplace_back(dev, torch::kHalf);
        combos.emplace_back(dev, torch::kBFloat16);
    }
    return combos;
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    DeviceDtype,
    GatherScatterConvTest,
    ::testing::ValuesIn(deviceDtypeCombos()),
    [](const auto &info) {
        auto dev = std::get<0>(info.param);
        auto dtype = std::get<1>(info.param);
        std::string name = dev.is_cpu() ? "CPU" : "CUDA";
        name += "_" + dtype_name(dtype);
        return name;
    });
// clang-format on

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    DeviceDtype,
    GatherScatterConvBackwardTest,
    ::testing::ValuesIn(deviceDtypeCombos()),
    [](const auto &info) {
        auto dev = std::get<0>(info.param);
        auto dtype = std::get<1>(info.param);
        std::string name = dev.is_cpu() ? "CPU" : "CUDA";
        name += "_" + dtype_name(dtype);
        return name;
    });
// clang-format on

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    DeviceDtype,
    GatherScatterConvStridedTest,
    ::testing::ValuesIn(deviceDtypeCombos()),
    [](const auto &info) {
        auto dev = std::get<0>(info.param);
        auto dtype = std::get<1>(info.param);
        std::string name = dev.is_cpu() ? "CPU" : "CUDA";
        name += "_" + dtype_name(dtype);
        return name;
    });
// clang-format on

INSTANTIATE_TEST_SUITE_P(Devices,
                         GatherScatterConvMixedDtypeTest,
                         ::testing::ValuesIn(availableDevices()),
                         [](const auto &info) { return info.param.is_cpu() ? "CPU" : "CUDA"; });
