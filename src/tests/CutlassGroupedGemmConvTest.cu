// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// CutlassGroupedGemmConvTest.cu -- Tests for the CUTLASS-accelerated sparse
// convolution.
//
// Phase 1: Topology validation only.  Compare cutlassConvTopology against
// gatherScatterSparseConvTopology (reference) for total_pairs and per-offset
// counts across all kernel size and stride combinations.
//
#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/GridBatchImpl.h>
#include <fvdb/detail/ops/convolution/CutlassGroupedGemm.h>
#include <fvdb/detail/ops/convolution/GatherScatter.h>

#include <torch/types.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <set>
#include <string>
#include <tuple>
#include <vector>

using namespace fvdb;
using namespace fvdb::detail;

// =============================================================================
// Helpers
// =============================================================================

static bool
cudaIsAvailable() {
    int count = 0;
    auto err  = cudaGetDeviceCount(&count);
    return err == cudaSuccess && count > 0;
}

static void
skipIfCudaUnavailable() {
    if (!cudaIsAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }
}

static torch::Device
cudaDevice() {
    return torch::Device(torch::kCUDA, 0);
}

static c10::intrusive_ptr<GridBatchImpl>
makeGrid(torch::Tensor ijk_2d, torch::Device device) {
    auto ijk_dev = ijk_2d.to(device);
    JaggedTensor jt(ijk_dev);
    std::vector<nanovdb::Vec3d> voxel_sizes = {{1.0, 1.0, 1.0}};
    std::vector<nanovdb::Vec3d> origins     = {{0.0, 0.0, 0.0}};
    return GridBatchImpl::createFromIjk(jt, voxel_sizes, origins);
}

// Sphere shell grid: voxels near the surface of a sphere of radius R.
// Produces ~4*pi*R^2 voxels (realistic SDF-like sparsity).
static c10::intrusive_ptr<GridBatchImpl>
makeSphereShellGrid(int R, torch::Device device) {
    std::vector<int32_t> flat;
    int const R_inner_sq = (R - 1) * (R - 1);
    int const R_outer_sq = R * R;
    for (int i = -R; i <= R; ++i)
        for (int j = -R; j <= R; ++j)
            for (int k = -R; k <= R; ++k) {
                int const r2 = i * i + j * j + k * k;
                if (r2 >= R_inner_sq && r2 <= R_outer_sq) {
                    flat.push_back(i + R);
                    flat.push_back(j + R);
                    flat.push_back(k + R);
                }
            }
    int64_t N = static_cast<int64_t>(flat.size()) / 3;
    if (N == 0) {
        auto ijk = torch::tensor({{R, R, R}}, torch::kInt32);
        return makeGrid(ijk, device);
    }
    auto ijk = torch::from_blob(flat.data(), {N, 3}, torch::kInt32).clone();
    return makeGrid(ijk, device);
}

static c10::intrusive_ptr<GridBatchImpl>
makeDenseGrid(int dim, torch::Device device) {
    std::vector<int32_t> flat;
    flat.reserve(dim * dim * dim * 3);
    for (int i = 0; i < dim; ++i)
        for (int j = 0; j < dim; ++j)
            for (int k = 0; k < dim; ++k) {
                flat.push_back(i);
                flat.push_back(j);
                flat.push_back(k);
            }
    int64_t N = static_cast<int64_t>(dim) * dim * dim;
    auto ijk  = torch::from_blob(flat.data(), {N, 3}, torch::kInt32).clone();
    return makeGrid(ijk, device);
}

// =============================================================================
// Topology validation helper
// =============================================================================
//
// Builds topology with both cutlassConvTopology and gatherScatterSparseConvTopology,
// then compares:
//   1. total_pairs must match
//   2. per-offset counts must match
//   3. offsets array is well-formed (non-decreasing, starts at 0, ends at total_pairs)
//   4. gather_indices are in [0, NA), scatter_indices are in [0, NB)
//   5. For each offset k, the SET of (gather, scatter) pairs matches the reference
//      (order within an offset may differ due to atomic fill, but the set must be identical)

static void
validateTopology(c10::intrusive_ptr<GridBatchImpl> const &feature_grid,
                 c10::intrusive_ptr<GridBatchImpl> const &output_grid,
                 nanovdb::Coord kernel_size,
                 nanovdb::Coord stride,
                 std::string const &label) {
    int64_t const NA = feature_grid->totalVoxels();
    int64_t const NB = output_grid->totalVoxels();
    int32_t const K =
        static_cast<int32_t>(kernel_size[0]) * kernel_size[1] * kernel_size[2];

    // -- Build both topologies --
    auto cutlass_topo =
        ops::cutlassConvTopology(*feature_grid, *output_grid, kernel_size, stride);
    auto ref_topo =
        ops::gatherScatterSparseConvTopology(*feature_grid, *output_grid, kernel_size, stride);

    // -- Reference: extract per-offset counts from dense kernel_map[O, K] --
    // kernel_map is [output_total_voxels, kernel_volume], transposed to [K, O] for iteration.
    auto kmap_t = ref_topo.kernel_map.t().contiguous(); // [K, O]
    auto mask   = kmap_t != -1;                          // [K, O] bool

    // Per-offset reference counts
    auto ref_counts_tensor =
        torch::sum(mask, /*dim=*/-1, /*keepdim=*/false, torch::kInt64).cpu(); // [K]
    auto ref_counts = ref_counts_tensor.accessor<int64_t, 1>();

    int64_t ref_total_pairs = 0;
    for (int32_t k = 0; k < K; ++k)
        ref_total_pairs += ref_counts[k];

    // -- 1. Total pairs --
    EXPECT_EQ(cutlass_topo.total_pairs, ref_total_pairs)
        << label << ": total_pairs mismatch (cutlass=" << cutlass_topo.total_pairs
        << " ref=" << ref_total_pairs << ")";

    // -- 2. Offsets well-formedness --
    EXPECT_EQ(cutlass_topo.offsets.size(0), K + 1) << label << ": offsets wrong size";
    auto off = cutlass_topo.offsets.accessor<int64_t, 1>();
    EXPECT_EQ(off[0], 0) << label << ": offsets[0] != 0";
    for (int32_t k = 0; k < K; ++k) {
        EXPECT_LE(off[k], off[k + 1])
            << label << ": offsets not non-decreasing at k=" << k;
    }
    EXPECT_EQ(off[K], cutlass_topo.total_pairs)
        << label << ": offsets[K] != total_pairs";

    // -- 3. Per-offset counts match --
    for (int32_t k = 0; k < K; ++k) {
        int64_t cutlass_count = off[k + 1] - off[k];
        EXPECT_EQ(cutlass_count, ref_counts[k])
            << label << ": count mismatch at offset k=" << k
            << " (cutlass=" << cutlass_count << " ref=" << ref_counts[k] << ")";
    }

    if (cutlass_topo.total_pairs == 0)
        return;

    // -- 4. Index range checks --
    auto gather_cpu  = cutlass_topo.gather_indices.cpu();
    auto scatter_cpu = cutlass_topo.scatter_indices.cpu();
    auto g_acc       = gather_cpu.accessor<int32_t, 1>();
    auto s_acc       = scatter_cpu.accessor<int32_t, 1>();

    for (int64_t i = 0; i < cutlass_topo.total_pairs; ++i) {
        EXPECT_GE(g_acc[i], 0) << label << ": gather_indices[" << i << "] < 0";
        EXPECT_LT(g_acc[i], NA) << label << ": gather_indices[" << i << "] >= NA";
        EXPECT_GE(s_acc[i], 0) << label << ": scatter_indices[" << i << "] < 0";
        EXPECT_LT(s_acc[i], NB) << label << ": scatter_indices[" << i << "] >= NB";
    }

    // -- 5. Per-offset pair sets match --
    // For each offset k, build the set of (gather, scatter) pairs from both
    // topologies and verify they are identical (order-independent).
    auto kmap_cpu = ref_topo.kernel_map.cpu(); // [O, K] int32
    auto km_acc   = kmap_cpu.accessor<int32_t, 2>();

    for (int32_t k = 0; k < K; ++k) {
        int64_t start = off[k];
        int64_t end   = off[k + 1];

        // Cutlass pairs for this offset
        std::set<std::pair<int32_t, int32_t>> cutlass_pairs;
        for (int64_t i = start; i < end; ++i) {
            cutlass_pairs.insert({g_acc[i], s_acc[i]});
        }

        // Reference pairs for this offset: iterate output voxels
        std::set<std::pair<int32_t, int32_t>> ref_pairs;
        for (int64_t o = 0; o < NB; ++o) {
            int32_t feat_idx = km_acc[o][k];
            if (feat_idx != -1) {
                ref_pairs.insert({feat_idx, static_cast<int32_t>(o)});
            }
        }

        EXPECT_EQ(cutlass_pairs, ref_pairs)
            << label << ": pair set mismatch at offset k=" << k
            << " (cutlass has " << cutlass_pairs.size()
            << " pairs, ref has " << ref_pairs.size() << ")";
    }
}

// =============================================================================
// Topology tests -- parameterized by kernel size, same grid, stride 1
// =============================================================================

using KernelParam = std::tuple<int, int, int>;

static std::string
kernelParamName(::testing::TestParamInfo<KernelParam> const &info) {
    auto [k0, k1, k2] = info.param;
    return "k" + std::to_string(k0) + "x" + std::to_string(k1) + "x" + std::to_string(k2);
}

class TopoKernelSizeTest : public ::testing::TestWithParam<KernelParam> {};

TEST_P(TopoKernelSizeTest, SameGridStride1) {
    skipIfCudaUnavailable();
    auto [k0, k1, k2] = GetParam();

    int dim   = std::max({k0, k1, k2, 4}) + 2;
    auto grid = makeDenseGrid(dim, cudaDevice());

    nanovdb::Coord ks(k0, k1, k2);
    validateTopology(grid, grid, ks, {1, 1, 1},
                     "topo_k" + std::to_string(k0) + "x" + std::to_string(k1) +
                         "x" + std::to_string(k2));
}

static std::vector<KernelParam>
kernelSizeConfigs() {
    return {
        // Uniform odd
        {1, 1, 1},
        {3, 3, 3},
        {5, 5, 5},
        // Uniform even
        {2, 2, 2},
        {4, 4, 4},
        // Non-uniform odd
        {3, 5, 1},
        {1, 3, 5},
        {5, 1, 3},
        // Non-uniform even
        {2, 4, 2},
        {4, 2, 4},
        // Mixed odd/even
        {3, 4, 5},
        {2, 3, 4},
        {4, 5, 2},
        {1, 2, 3},
    };
}

INSTANTIATE_TEST_SUITE_P(KernelSizes,
                         TopoKernelSizeTest,
                         ::testing::ValuesIn(kernelSizeConfigs()),
                         kernelParamName);

// =============================================================================
// Topology tests -- parameterized by kernel + stride, different grids
// =============================================================================

using StrideParam = std::tuple<int, int, int, int, int, int>;

static std::string
strideParamName(::testing::TestParamInfo<StrideParam> const &info) {
    auto [k0, k1, k2, s0, s1, s2] = info.param;
    return "k" + std::to_string(k0) + "x" + std::to_string(k1) + "x" + std::to_string(k2) +
           "_s" + std::to_string(s0) + "x" + std::to_string(s1) + "x" + std::to_string(s2);
}

class TopoStrideTest : public ::testing::TestWithParam<StrideParam> {};

TEST_P(TopoStrideTest, StridedDifferentGrids) {
    skipIfCudaUnavailable();
    auto [k0, k1, k2, s0, s1, s2] = GetParam();

    auto device   = cudaDevice();
    auto src_grid = makeDenseGrid(8, device);

    nanovdb::Coord ks(k0, k1, k2);
    nanovdb::Coord stride(s0, s1, s2);
    auto dst_grid = src_grid->convolutionOutput(ks, stride);

    validateTopology(
        src_grid, dst_grid, ks, stride,
        "topo_k" + std::to_string(k0) + "x" + std::to_string(k1) + "x" + std::to_string(k2) +
            "_s" + std::to_string(s0) + "x" + std::to_string(s1) + "x" + std::to_string(s2));
}

static std::vector<StrideParam>
strideConfigs() {
    return {
        // Uniform stride with odd kernel
        {3, 3, 3, 2, 2, 2},
        {3, 3, 3, 3, 3, 3},
        {5, 5, 5, 2, 2, 2},
        // Uniform stride with even kernel
        {2, 2, 2, 2, 2, 2},
        {4, 4, 4, 2, 2, 2},
        // Non-uniform stride with uniform odd kernel
        {3, 3, 3, 1, 2, 1},
        {3, 3, 3, 2, 1, 2},
        {3, 3, 3, 1, 1, 2},
        // Non-uniform stride with non-uniform kernel
        {3, 5, 3, 1, 2, 1},
        {2, 3, 4, 2, 1, 2},
        // Non-uniform stride with even kernel
        {4, 2, 4, 2, 1, 2},
        // Non-uniform stride with mixed odd/even kernel
        {3, 4, 5, 1, 2, 3},
    };
}

INSTANTIATE_TEST_SUITE_P(Strides,
                         TopoStrideTest,
                         ::testing::ValuesIn(strideConfigs()),
                         strideParamName);

// =============================================================================
// Topology edge cases
// =============================================================================

TEST(TopoEdgeCases, EmptyTopology) {
    skipIfCudaUnavailable();
    auto device = cudaDevice();

    auto ijk      = torch::tensor({{0, 0, 0}}, torch::kInt32);
    auto src_grid = makeGrid(ijk, device);

    auto far_ijk  = torch::tensor({{100, 100, 100}}, torch::kInt32);
    auto dst_grid = makeGrid(far_ijk, device);

    auto topo = ops::cutlassConvTopology(*src_grid, *dst_grid, {3, 3, 3}, {1, 1, 1});
    EXPECT_EQ(topo.total_pairs, 0);
    EXPECT_EQ(topo.gather_indices.size(0), 0);
    EXPECT_EQ(topo.scatter_indices.size(0), 0);
}

TEST(TopoEdgeCases, SingleVoxelSameGrid) {
    skipIfCudaUnavailable();
    auto device = cudaDevice();

    auto ijk  = torch::tensor({{5, 5, 5}}, torch::kInt32);
    auto grid = makeGrid(ijk, device);

    // 1 voxel, 3x3x3 kernel, same grid: only the center offset (k=13) should hit
    validateTopology(grid, grid, {3, 3, 3}, {1, 1, 1}, "single_voxel_same_grid");
}

TEST(TopoEdgeCases, SingleVoxelConvOutput) {
    skipIfCudaUnavailable();
    auto device = cudaDevice();

    auto ijk      = torch::tensor({{5, 5, 5}}, torch::kInt32);
    auto src_grid = makeGrid(ijk, device);
    auto dst_grid = src_grid->convolutionOutput({3, 3, 3}, {1, 1, 1});

    // 1 src voxel expanded to 27 dst voxels: each dst hits exactly 1 src
    validateTopology(src_grid, dst_grid, {3, 3, 3}, {1, 1, 1}, "single_voxel_conv_output");
}

// =============================================================================
// Forward: identity 1x1x1 smoke test (exercises CUTLASS GEMM path)
// =============================================================================

TEST(CutlassForwardSmoke, Identity1x1x1_C64) {
    skipIfCudaUnavailable();
    int64_t const C = 64;

    auto device = cudaDevice();
    auto grid   = makeDenseGrid(4, device);
    int64_t NV  = grid->totalVoxels();

    auto features =
        torch::randn({NV, C}, torch::dtype(torch::kFloat16).device(device)) * 0.1;
    auto weights =
        torch::eye(C, torch::dtype(torch::kFloat16).device(device)).reshape({C, C, 1, 1, 1});

    auto topo = ops::cutlassConvTopology(*grid, *grid, {1, 1, 1}, {1, 1, 1});
    auto out  = ops::cutlassConv(features, weights, topo);

    EXPECT_EQ(out.sizes(), features.sizes());

    auto out_f32  = out.to(torch::kFloat32).cpu();
    auto feat_f32 = features.to(torch::kFloat32).cpu();

    EXPECT_TRUE(torch::allclose(out_f32, feat_f32, 0.01, 0.01))
        << "Identity 1x1x1 conv mismatch\n"
        << "  max abs diff = "
        << (out_f32 - feat_f32).abs().max().item<double>() << "\n"
        << "  mean abs diff = "
        << (out_f32 - feat_f32).abs().mean().item<double>();
}

// 1x1x1 with random weights = pure matmul, including non-square (Cin != Cout)
TEST(CutlassForwardSmoke, Matmul1x1x1_C64) {
    skipIfCudaUnavailable();
    int64_t const C = 64;

    auto device = cudaDevice();
    auto grid   = makeDenseGrid(4, device);
    int64_t NV  = grid->totalVoxels();

    torch::manual_seed(123);
    auto features =
        torch::randn({NV, C}, torch::dtype(torch::kFloat16).device(device)) * 0.1;
    auto weights =
        torch::randn({C, C, 1, 1, 1}, torch::dtype(torch::kFloat16).device(device)) * 0.1;

    auto ref_topo = ops::gatherScatterSparseConvTopology(*grid, *grid, {1, 1, 1}, {1, 1, 1});
    auto ref_out  = ops::gatherScatterSparseConv(
        features.to(torch::kFloat32), weights.to(torch::kFloat32), ref_topo);

    auto topo = ops::cutlassConvTopology(*grid, *grid, {1, 1, 1}, {1, 1, 1});
    auto out  = ops::cutlassConv(features, weights, topo);

    auto out_f32 = out.to(torch::kFloat32).cpu();
    auto ref_f32 = ref_out.cpu();

    EXPECT_TRUE(torch::allclose(out_f32, ref_f32, 0.05, 0.05))
        << "1x1x1 matmul mismatch (square C=64)\n"
        << "  max abs diff = " << (out_f32 - ref_f32).abs().max().item<double>() << "\n"
        << "  mean abs diff = " << (out_f32 - ref_f32).abs().mean().item<double>();
}

TEST(CutlassForwardSmoke, Matmul1x1x1_Cin32_Cout64) {
    skipIfCudaUnavailable();

    auto device = cudaDevice();
    auto grid   = makeDenseGrid(4, device);
    int64_t NV  = grid->totalVoxels();

    torch::manual_seed(456);
    auto features =
        torch::randn({NV, 32}, torch::dtype(torch::kFloat16).device(device)) * 0.1;
    auto weights =
        torch::randn({64, 32, 1, 1, 1}, torch::dtype(torch::kFloat16).device(device)) * 0.1;

    auto ref_topo = ops::gatherScatterSparseConvTopology(*grid, *grid, {1, 1, 1}, {1, 1, 1});
    auto ref_out  = ops::gatherScatterSparseConv(
        features.to(torch::kFloat32), weights.to(torch::kFloat32), ref_topo);

    auto topo = ops::cutlassConvTopology(*grid, *grid, {1, 1, 1}, {1, 1, 1});
    auto out  = ops::cutlassConv(features, weights, topo);

    auto out_f32 = out.to(torch::kFloat32).cpu();
    auto ref_f32 = ref_out.cpu();

    EXPECT_EQ(out.size(0), NV);
    EXPECT_EQ(out.size(1), 64);

    EXPECT_TRUE(torch::allclose(out_f32, ref_f32, 0.05, 0.05))
        << "1x1x1 matmul mismatch (Cin=32, Cout=64)\n"
        << "  max abs diff = " << (out_f32 - ref_f32).abs().max().item<double>() << "\n"
        << "  mean abs diff = " << (out_f32 - ref_f32).abs().mean().item<double>();
}

// 3x3x3 with random weights, same grid -- exercises 27-group GEMM + scatter
TEST(CutlassForwardSmoke, Forward3x3x3_C64) {
    skipIfCudaUnavailable();
    int64_t const C = 64;

    auto device = cudaDevice();
    auto grid   = makeDenseGrid(6, device);
    int64_t NV  = grid->totalVoxels();

    torch::manual_seed(789);
    auto features =
        torch::randn({NV, C}, torch::dtype(torch::kFloat16).device(device)) * 0.1;
    auto weights =
        torch::randn({C, C, 3, 3, 3}, torch::dtype(torch::kFloat16).device(device)) * 0.1;

    auto ref_topo = ops::gatherScatterSparseConvTopology(*grid, *grid, {3, 3, 3}, {1, 1, 1});
    auto ref_out  = ops::gatherScatterSparseConv(
        features.to(torch::kFloat32), weights.to(torch::kFloat32), ref_topo);

    auto topo = ops::cutlassConvTopology(*grid, *grid, {3, 3, 3}, {1, 1, 1});
    auto out  = ops::cutlassConv(features, weights, topo);

    auto out_f32 = out.to(torch::kFloat32).cpu();
    auto ref_f32 = ref_out.cpu();

    EXPECT_TRUE(torch::allclose(out_f32, ref_f32, 0.05, 0.05))
        << "3x3x3 forward mismatch (C=64)\n"
        << "  max abs diff = " << (out_f32 - ref_f32).abs().max().item<double>() << "\n"
        << "  mean abs diff = " << (out_f32 - ref_f32).abs().mean().item<double>();
}

// 3x3x3 strided with different src/dst grids
TEST(CutlassForwardSmoke, Forward3x3x3_Stride2_C64) {
    skipIfCudaUnavailable();
    int64_t const C = 64;

    auto device   = cudaDevice();
    auto src_grid = makeDenseGrid(8, device);

    nanovdb::Coord ks(3, 3, 3);
    nanovdb::Coord stride(2, 2, 2);
    auto dst_grid = src_grid->convolutionOutput(ks, stride);

    int64_t NA = src_grid->totalVoxels();

    torch::manual_seed(101);
    auto features =
        torch::randn({NA, C}, torch::dtype(torch::kFloat16).device(device)) * 0.1;
    auto weights =
        torch::randn({C, C, 3, 3, 3}, torch::dtype(torch::kFloat16).device(device)) * 0.1;

    auto ref_topo =
        ops::gatherScatterSparseConvTopology(*src_grid, *dst_grid, ks, stride);
    auto ref_out = ops::gatherScatterSparseConv(
        features.to(torch::kFloat32), weights.to(torch::kFloat32), ref_topo);

    auto topo = ops::cutlassConvTopology(*src_grid, *dst_grid, ks, stride);
    auto out  = ops::cutlassConv(features, weights, topo);

    auto out_f32 = out.to(torch::kFloat32).cpu();
    auto ref_f32 = ref_out.cpu();

    EXPECT_TRUE(torch::allclose(out_f32, ref_f32, 0.05, 0.05))
        << "3x3x3 stride-2 forward mismatch (C=64)\n"
        << "  max abs diff = " << (out_f32 - ref_f32).abs().max().item<double>() << "\n"
        << "  mean abs diff = " << (out_f32 - ref_f32).abs().mean().item<double>();
}

// Non-uniform kernel: 3x5x1 (all odd, asymmetric), same grid
TEST(CutlassForwardSmoke, Forward3x5x1_C64) {
    skipIfCudaUnavailable();
    int64_t const C = 64;

    auto device = cudaDevice();
    auto grid   = makeDenseGrid(7, device);
    int64_t NV  = grid->totalVoxels();

    torch::manual_seed(202);
    auto features =
        torch::randn({NV, C}, torch::dtype(torch::kFloat16).device(device)) * 0.1;
    auto weights =
        torch::randn({C, C, 3, 5, 1}, torch::dtype(torch::kFloat16).device(device)) * 0.1;

    auto ref_topo = ops::gatherScatterSparseConvTopology(*grid, *grid, {3, 5, 1}, {1, 1, 1});
    auto ref_out  = ops::gatherScatterSparseConv(
        features.to(torch::kFloat32), weights.to(torch::kFloat32), ref_topo);

    auto topo = ops::cutlassConvTopology(*grid, *grid, {3, 5, 1}, {1, 1, 1});
    auto out  = ops::cutlassConv(features, weights, topo);

    auto out_f32 = out.to(torch::kFloat32).cpu();
    auto ref_f32 = ref_out.cpu();

    EXPECT_TRUE(torch::allclose(out_f32, ref_f32, 0.05, 0.05))
        << "3x5x1 forward mismatch (C=64)\n"
        << "  max abs diff = " << (out_f32 - ref_f32).abs().max().item<double>() << "\n"
        << "  mean abs diff = " << (out_f32 - ref_f32).abs().mean().item<double>();
}

// Even kernel: 2x2x2, same grid (previously failed with hash map topology)
TEST(CutlassForwardSmoke, Forward2x2x2_C64) {
    skipIfCudaUnavailable();
    int64_t const C = 64;

    auto device = cudaDevice();
    auto grid   = makeDenseGrid(6, device);
    int64_t NV  = grid->totalVoxels();

    torch::manual_seed(303);
    auto features =
        torch::randn({NV, C}, torch::dtype(torch::kFloat16).device(device)) * 0.1;
    auto weights =
        torch::randn({C, C, 2, 2, 2}, torch::dtype(torch::kFloat16).device(device)) * 0.1;

    auto ref_topo = ops::gatherScatterSparseConvTopology(*grid, *grid, {2, 2, 2}, {1, 1, 1});
    auto ref_out  = ops::gatherScatterSparseConv(
        features.to(torch::kFloat32), weights.to(torch::kFloat32), ref_topo);

    auto topo = ops::cutlassConvTopology(*grid, *grid, {2, 2, 2}, {1, 1, 1});
    auto out  = ops::cutlassConv(features, weights, topo);

    auto out_f32 = out.to(torch::kFloat32).cpu();
    auto ref_f32 = ref_out.cpu();

    EXPECT_TRUE(torch::allclose(out_f32, ref_f32, 0.05, 0.05))
        << "2x2x2 forward mismatch (C=64)\n"
        << "  max abs diff = " << (out_f32 - ref_f32).abs().max().item<double>() << "\n"
        << "  mean abs diff = " << (out_f32 - ref_f32).abs().mean().item<double>();
}

// Large channels: C=128, 3x3x3 -- tests bigger GEMM tiles
TEST(CutlassForwardSmoke, Forward3x3x3_C128) {
    skipIfCudaUnavailable();
    int64_t const C = 128;

    auto device = cudaDevice();
    auto grid   = makeDenseGrid(6, device);
    int64_t NV  = grid->totalVoxels();

    torch::manual_seed(404);
    auto features =
        torch::randn({NV, C}, torch::dtype(torch::kFloat16).device(device)) * 0.1;
    auto weights =
        torch::randn({C, C, 3, 3, 3}, torch::dtype(torch::kFloat16).device(device)) * 0.1;

    auto ref_topo = ops::gatherScatterSparseConvTopology(*grid, *grid, {3, 3, 3}, {1, 1, 1});
    auto ref_out  = ops::gatherScatterSparseConv(
        features.to(torch::kFloat32), weights.to(torch::kFloat32), ref_topo);

    auto topo = ops::cutlassConvTopology(*grid, *grid, {3, 3, 3}, {1, 1, 1});
    auto out  = ops::cutlassConv(features, weights, topo);

    auto out_f32 = out.to(torch::kFloat32).cpu();
    auto ref_f32 = ref_out.cpu();

    EXPECT_TRUE(torch::allclose(out_f32, ref_f32, 0.05, 0.05))
        << "3x3x3 forward mismatch (C=128)\n"
        << "  max abs diff = " << (out_f32 - ref_f32).abs().max().item<double>() << "\n"
        << "  mean abs diff = " << (out_f32 - ref_f32).abs().mean().item<double>();
}

// Non-uniform stride: 3x3x3 kernel, stride {1,2,1}
TEST(CutlassForwardSmoke, Forward3x3x3_Stride1x2x1_C64) {
    skipIfCudaUnavailable();
    int64_t const C = 64;

    auto device   = cudaDevice();
    auto src_grid = makeDenseGrid(8, device);

    nanovdb::Coord ks(3, 3, 3);
    nanovdb::Coord stride(1, 2, 1);
    auto dst_grid = src_grid->convolutionOutput(ks, stride);

    int64_t NA = src_grid->totalVoxels();

    torch::manual_seed(505);
    auto features =
        torch::randn({NA, C}, torch::dtype(torch::kFloat16).device(device)) * 0.1;
    auto weights =
        torch::randn({C, C, 3, 3, 3}, torch::dtype(torch::kFloat16).device(device)) * 0.1;

    auto ref_topo =
        ops::gatherScatterSparseConvTopology(*src_grid, *dst_grid, ks, stride);
    auto ref_out = ops::gatherScatterSparseConv(
        features.to(torch::kFloat32), weights.to(torch::kFloat32), ref_topo);

    auto topo = ops::cutlassConvTopology(*src_grid, *dst_grid, ks, stride);
    auto out  = ops::cutlassConv(features, weights, topo);

    auto out_f32 = out.to(torch::kFloat32).cpu();
    auto ref_f32 = ref_out.cpu();

    EXPECT_TRUE(torch::allclose(out_f32, ref_f32, 0.05, 0.05))
        << "3x3x3 stride {1,2,1} forward mismatch (C=64)\n"
        << "  max abs diff = " << (out_f32 - ref_f32).abs().max().item<double>() << "\n"
        << "  mean abs diff = " << (out_f32 - ref_f32).abs().mean().item<double>();
}

// =============================================================================
// Full parameterized forward tests
// =============================================================================

static void
compareForward(c10::intrusive_ptr<GridBatchImpl> const &feature_grid,
               c10::intrusive_ptr<GridBatchImpl> const &output_grid,
               nanovdb::Coord kernel_size,
               nanovdb::Coord stride,
               int64_t Cin,
               int64_t Cout,
               double rtol,
               double atol,
               std::string const &label) {
    auto device = cudaDevice();

    int64_t NA = feature_grid->totalVoxels();
    int64_t NB = output_grid->totalVoxels();

    torch::manual_seed(42);
    auto features_f16 = torch::randn({NA, Cin}, torch::dtype(torch::kFloat16).device(device)) * 0.1;
    auto weights_f16 =
        torch::randn({Cout, Cin, kernel_size[0], kernel_size[1], kernel_size[2]},
                     torch::dtype(torch::kFloat16).device(device)) *
        0.1;

    auto ref_topo =
        ops::gatherScatterSparseConvTopology(*feature_grid, *output_grid, kernel_size, stride);
    auto ref_out = ops::gatherScatterSparseConv(
        features_f16.to(torch::kFloat32), weights_f16.to(torch::kFloat32), ref_topo);

    auto cutlass_topo = ops::cutlassConvTopology(*feature_grid, *output_grid, kernel_size, stride);
    auto cutlass_out  = ops::cutlassConv(features_f16, weights_f16, cutlass_topo);

    EXPECT_EQ(cutlass_out.sizes(), ref_out.sizes()) << label << ": shape mismatch";

    auto cutlass_f32 = cutlass_out.to(torch::kFloat32).cpu();
    auto ref_f32     = ref_out.cpu();

    EXPECT_TRUE(torch::allclose(cutlass_f32, ref_f32, rtol, atol))
        << label << ": numerical mismatch\n"
        << "  max abs diff = " << (cutlass_f32 - ref_f32).abs().max().item<double>() << "\n"
        << "  mean abs diff = " << (cutlass_f32 - ref_f32).abs().mean().item<double>();
}

// -- Channel sweep: 3x3x3 + identity, multiple Cin/Cout combos --

using ChannelParam = std::tuple<int64_t, int64_t>;

static std::string
channelParamName(::testing::TestParamInfo<ChannelParam> const &info) {
    auto Cin  = std::get<0>(info.param);
    auto Cout = std::get<1>(info.param);
    return "Cin" + std::to_string(Cin) + "_Cout" + std::to_string(Cout);
}

class CutlassChannelTest : public ::testing::TestWithParam<ChannelParam> {};

TEST_P(CutlassChannelTest, Forward3x3x3) {
    skipIfCudaUnavailable();
    auto [Cin, Cout] = GetParam();
    auto grid        = makeDenseGrid(6, cudaDevice());
    compareForward(grid, grid, {3, 3, 3}, {1, 1, 1}, Cin, Cout, 0.05, 0.05, "3x3x3_channel");
}

static std::vector<ChannelParam>
channelConfigs() {
    return {{32, 32}, {64, 64}, {128, 128}, {32, 64}, {64, 128}};
}

INSTANTIATE_TEST_SUITE_P(Channels, CutlassChannelTest,
    ::testing::ValuesIn(channelConfigs()), channelParamName);

// -- Full kernel-size sweep --

class CutlassKernelSizeTest : public ::testing::TestWithParam<KernelParam> {};

TEST_P(CutlassKernelSizeTest, ForwardSameGrid) {
    skipIfCudaUnavailable();
    auto [k0, k1, k2] = GetParam();
    int dim   = std::max({k0, k1, k2, 4}) + 2;
    auto grid = makeDenseGrid(dim, cudaDevice());
    nanovdb::Coord ks(k0, k1, k2);
    compareForward(grid, grid, ks, {1, 1, 1}, 64, 64, 0.05, 0.05,
                   "kernel_" + std::to_string(k0) + "x" + std::to_string(k1) + "x" + std::to_string(k2));
}

INSTANTIATE_TEST_SUITE_P(FwdKernelSizes, CutlassKernelSizeTest,
    ::testing::ValuesIn(kernelSizeConfigs()), kernelParamName);

// -- Full stride sweep --

class CutlassStrideTest : public ::testing::TestWithParam<StrideParam> {};

TEST_P(CutlassStrideTest, ForwardStrided) {
    skipIfCudaUnavailable();
    auto [k0, k1, k2, s0, s1, s2] = GetParam();
    auto device   = cudaDevice();
    auto src_grid = makeDenseGrid(8, device);
    nanovdb::Coord ks(k0, k1, k2);
    nanovdb::Coord stride(s0, s1, s2);
    auto dst_grid = src_grid->convolutionOutput(ks, stride);
    compareForward(src_grid, dst_grid, ks, stride, 64, 64, 0.05, 0.05,
                   "stride_k" + std::to_string(k0) + "x" + std::to_string(k1) + "x" + std::to_string(k2) +
                   "_s" + std::to_string(s0) + "x" + std::to_string(s1) + "x" + std::to_string(s2));
}

INSTANTIATE_TEST_SUITE_P(FwdStrides, CutlassStrideTest,
    ::testing::ValuesIn(strideConfigs()), strideParamName);

// -- Edge-case forward tests --

TEST(CutlassEdgeCases, SingleVoxelForward) {
    skipIfCudaUnavailable();
    auto device = cudaDevice();
    auto ijk  = torch::tensor({{5, 5, 5}}, torch::kInt32);
    auto grid = makeGrid(ijk, device);
    auto dst_grid = grid->convolutionOutput({3, 3, 3}, {1, 1, 1});
    compareForward(grid, dst_grid, {3, 3, 3}, {1, 1, 1}, 32, 32, 0.05, 0.05, "single_voxel");
}

TEST(CutlassEdgeCases, EmptyOutputForward) {
    skipIfCudaUnavailable();
    auto device = cudaDevice();
    auto ijk      = torch::tensor({{0, 0, 0}}, torch::kInt32);
    auto src_grid = makeGrid(ijk, device);
    auto far_ijk  = torch::tensor({{100, 100, 100}}, torch::kInt32);
    auto dst_grid = makeGrid(far_ijk, device);
    auto topo = ops::cutlassConvTopology(*src_grid, *dst_grid, {3, 3, 3}, {1, 1, 1});
    EXPECT_EQ(topo.total_pairs, 0);
    auto features = torch::randn({1, 32}, torch::dtype(torch::kFloat16).device(device));
    auto weights  = torch::randn({32, 32, 3, 3, 3}, torch::dtype(torch::kFloat16).device(device));
    auto out = ops::cutlassConv(features, weights, topo);
    EXPECT_EQ(out.size(0), 1);
    EXPECT_EQ(out.size(1), 32);
    EXPECT_TRUE(out.abs().max().item<float>() == 0.0f);
}

// =============================================================================
// Sparse grid tests -- sphere shells with realistic sparsity patterns
// =============================================================================
//
// Sphere shells have boundary voxels with fewer neighbors per offset than
// interior voxels, so the per-offset Mk values vary widely.  This exercises
// the grouped GEMM with heterogeneous group sizes.

// Small sphere: R=10, ~1.2K voxels, 3x3x3, same grid
TEST(CutlassSphereTests, Sphere_R10_3x3x3_C64) {
    skipIfCudaUnavailable();
    auto device = cudaDevice();
    auto grid   = makeSphereShellGrid(10, device);

    compareForward(grid, grid, {3, 3, 3}, {1, 1, 1}, 64, 64, 0.05, 0.05,
                   "sphere_R10_3x3x3");
}

// Medium sphere: R=20, ~5K voxels, 3x3x3, same grid
TEST(CutlassSphereTests, Sphere_R20_3x3x3_C64) {
    skipIfCudaUnavailable();
    auto device = cudaDevice();
    auto grid   = makeSphereShellGrid(20, device);

    compareForward(grid, grid, {3, 3, 3}, {1, 1, 1}, 64, 64, 0.05, 0.05,
                   "sphere_R20_3x3x3");
}

// Larger sphere: R=50, ~30K voxels, 3x3x3, same grid, C=128
TEST(CutlassSphereTests, Sphere_R50_3x3x3_C128) {
    skipIfCudaUnavailable();
    auto device = cudaDevice();
    auto grid   = makeSphereShellGrid(50, device);

    compareForward(grid, grid, {3, 3, 3}, {1, 1, 1}, 128, 128, 0.05, 0.05,
                   "sphere_R50_3x3x3_C128");
}

// Sphere with non-uniform kernel: 3x5x1
TEST(CutlassSphereTests, Sphere_R20_3x5x1_C64) {
    skipIfCudaUnavailable();
    auto device = cudaDevice();
    auto grid   = makeSphereShellGrid(20, device);

    compareForward(grid, grid, {3, 5, 1}, {1, 1, 1}, 64, 64, 0.05, 0.05,
                   "sphere_R20_3x5x1");
}

// Sphere with even kernel: 2x2x2
TEST(CutlassSphereTests, Sphere_R20_2x2x2_C64) {
    skipIfCudaUnavailable();
    auto device = cudaDevice();
    auto grid   = makeSphereShellGrid(20, device);

    compareForward(grid, grid, {2, 2, 2}, {1, 1, 1}, 64, 64, 0.05, 0.05,
                   "sphere_R20_2x2x2");
}

// Sphere with stride: 3x3x3, stride 2, different src/dst grids
TEST(CutlassSphereTests, Sphere_R20_3x3x3_Stride2_C64) {
    skipIfCudaUnavailable();
    auto device   = cudaDevice();
    auto src_grid = makeSphereShellGrid(20, device);

    nanovdb::Coord ks(3, 3, 3);
    nanovdb::Coord stride(2, 2, 2);
    auto dst_grid = src_grid->convolutionOutput(ks, stride);

    compareForward(src_grid, dst_grid, ks, stride, 64, 64, 0.05, 0.05,
                   "sphere_R20_3x3x3_stride2");
}

// Sphere with Cin != Cout: channel expansion
TEST(CutlassSphereTests, Sphere_R20_3x3x3_Cin32_Cout128) {
    skipIfCudaUnavailable();
    auto device = cudaDevice();
    auto grid   = makeSphereShellGrid(20, device);

    compareForward(grid, grid, {3, 3, 3}, {1, 1, 1}, 32, 128, 0.05, 0.05,
                   "sphere_R20_Cin32_Cout128");
}

// Sphere with mixed odd/even kernel and non-uniform stride
TEST(CutlassSphereTests, Sphere_R20_3x4x5_Stride1x2x3_C64) {
    skipIfCudaUnavailable();
    auto device   = cudaDevice();
    auto src_grid = makeSphereShellGrid(20, device);

    nanovdb::Coord ks(3, 4, 5);
    nanovdb::Coord stride(1, 2, 3);
    auto dst_grid = src_grid->convolutionOutput(ks, stride);

    compareForward(src_grid, dst_grid, ks, stride, 64, 64, 0.05, 0.05,
                   "sphere_R20_3x4x5_s1x2x3");
}
