// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// CutlassGroupedGemmConvTest.cu -- Tests for the CUTLASS-accelerated sparse
// convolution.
//
// Test sections:
//   1. Topology validation (cutlassConvTopology vs gatherScatterDefaultSparseConvTopology)
//   2. GPU topology builder comparison (GPU two-pass vs GroupedGemm CSR reference)
//   3. Forward numerical correctness (CUTLASS vs GatherScatterDefault reference)
//   4. Sparse grid tests (sphere shells with realistic sparsity patterns)
//
#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/GridBatchImpl.h>
#include <fvdb/detail/ops/convolution/CutlassGroupedGemm.h>
#include <fvdb/detail/ops/convolution/GatherScatterDefault.h>
#include <fvdb/detail/ops/convolution/GroupedGemm.h>

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
// Builds topology with both cutlassConvTopology and gatherScatterDefaultSparseConvTopology,
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
    int32_t const K  = static_cast<int32_t>(kernel_size[0]) * kernel_size[1] * kernel_size[2];

    auto cutlass_topo = ops::cutlassConvTopology(*feature_grid, *output_grid, kernel_size, stride);
    auto ref_topo     = ops::gatherScatterDefaultSparseConvTopology(
        *feature_grid, *output_grid, kernel_size, stride);

    // 1. Total pairs
    EXPECT_EQ(cutlass_topo.total_pairs, ref_topo.totalPairs) << label << ": total_pairs mismatch";

    // 2. Offsets well-formedness (cutlass)
    EXPECT_EQ(cutlass_topo.offsets.size(0), K + 1) << label << ": offsets wrong size";
    auto c_off = cutlass_topo.offsets.accessor<int64_t, 1>();
    EXPECT_EQ(c_off[0], 0) << label << ": cutlass offsets[0] != 0";
    for (int32_t k = 0; k < K; ++k) {
        EXPECT_LE(c_off[k], c_off[k + 1])
            << label << ": cutlass offsets not non-decreasing at k=" << k;
    }
    EXPECT_EQ(c_off[K], cutlass_topo.total_pairs) << label << ": cutlass offsets[K] != total_pairs";

    // 3. Per-offset counts match
    auto r_off = ref_topo.offsets.accessor<int64_t, 1>();
    for (int32_t k = 0; k < K; ++k) {
        int64_t cutlass_count = c_off[k + 1] - c_off[k];
        int64_t ref_count     = r_off[k + 1] - r_off[k];
        EXPECT_EQ(cutlass_count, ref_count) << label << ": count mismatch at offset k=" << k;
    }

    if (cutlass_topo.total_pairs == 0)
        return;

    // 4. Index range checks (cutlass)
    auto c_gather_cpu  = cutlass_topo.gather_indices.cpu();
    auto c_scatter_cpu = cutlass_topo.scatter_indices.cpu();
    auto cg_acc        = c_gather_cpu.accessor<int32_t, 1>();
    auto cs_acc        = c_scatter_cpu.accessor<int32_t, 1>();

    for (int64_t i = 0; i < cutlass_topo.total_pairs; ++i) {
        EXPECT_GE(cg_acc[i], 0) << label << ": gather_indices[" << i << "] < 0";
        EXPECT_LT(cg_acc[i], NA) << label << ": gather_indices[" << i << "] >= NA";
        EXPECT_GE(cs_acc[i], 0) << label << ": scatter_indices[" << i << "] < 0";
        EXPECT_LT(cs_acc[i], NB) << label << ": scatter_indices[" << i << "] >= NB";
    }

    // 5. Per-offset pair sets match (order-independent)
    auto r_gather_cpu  = ref_topo.gatherIndices.cpu();
    auto r_scatter_cpu = ref_topo.scatterIndices.cpu();
    auto rg_acc        = r_gather_cpu.accessor<int32_t, 1>();
    auto rs_acc        = r_scatter_cpu.accessor<int32_t, 1>();

    for (int32_t k = 0; k < K; ++k) {
        int64_t c_start = c_off[k], c_end = c_off[k + 1];
        int64_t r_start = r_off[k], r_end = r_off[k + 1];

        std::set<std::pair<int32_t, int32_t>> cutlass_pairs;
        for (int64_t i = c_start; i < c_end; ++i)
            cutlass_pairs.insert({cg_acc[i], cs_acc[i]});

        std::set<std::pair<int32_t, int32_t>> ref_pairs;
        for (int64_t i = r_start; i < r_end; ++i)
            ref_pairs.insert({rg_acc[i], rs_acc[i]});

        EXPECT_EQ(cutlass_pairs, ref_pairs) << label << ": pair set mismatch at offset k=" << k;
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
    validateTopology(grid,
                     grid,
                     ks,
                     {1, 1, 1},
                     "topo_k" + std::to_string(k0) + "x" + std::to_string(k1) + "x" +
                         std::to_string(k2));
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
    return "k" + std::to_string(k0) + "x" + std::to_string(k1) + "x" + std::to_string(k2) + "_s" +
           std::to_string(s0) + "x" + std::to_string(s1) + "x" + std::to_string(s2);
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

    validateTopology(src_grid,
                     dst_grid,
                     ks,
                     stride,
                     "topo_k" + std::to_string(k0) + "x" + std::to_string(k1) + "x" +
                         std::to_string(k2) + "_s" + std::to_string(s0) + "x" + std::to_string(s1) +
                         "x" + std::to_string(s2));
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
// GPU topology builder vs reference comparison tests
// =============================================================================
//
// Compare the GPU two-pass topology builder (called via cutlassConvTopology)
// against the GroupedGemm CSR reference (groupedGemmSparseConvTopology).
// All values are integer -- must match exactly.  Within each offset segment
// the pair ordering may differ (atomicAdd non-determinism), so we compare
// sorted pair sets.

static void
compareGpuVsRefTopology(c10::intrusive_ptr<GridBatchImpl> const &feature_grid,
                        c10::intrusive_ptr<GridBatchImpl> const &output_grid,
                        nanovdb::Coord kernel_size,
                        nanovdb::Coord stride,
                        std::string const &label) {
    // GPU two-pass topology (internally also validates against reference via TORCH_CHECK)
    auto gpu = ops::cutlassConvTopology(*feature_grid, *output_grid, kernel_size, stride);

    // Reference CSR topology (independent call)
    auto ref = ops::groupedGemmSparseConvTopology(*feature_grid, *output_grid, kernel_size, stride);

    // -- Metadata must match exactly --
    EXPECT_EQ(gpu.kernel_volume, ref.kernel_volume) << label << ": kernel_volume";
    EXPECT_EQ(gpu.feature_total_voxels, ref.feature_total_voxels)
        << label << ": feature_total_voxels";
    EXPECT_EQ(gpu.output_total_voxels, ref.output_total_voxels) << label << ": output_total_voxels";
    EXPECT_EQ(gpu.total_pairs, ref.total_pairs)
        << label << ": total_pairs (gpu=" << gpu.total_pairs << " ref=" << ref.total_pairs << ")";

    // -- Offsets must match exactly --
    ASSERT_EQ(gpu.offsets.size(0), ref.offsets.size(0)) << label << ": offsets size";
    EXPECT_TRUE(torch::equal(gpu.offsets, ref.offsets)) << label << ": offsets content";

    if (gpu.total_pairs == 0)
        return;

    // -- Per-segment pair-set equality --
    auto gpu_off = gpu.offsets.accessor<int64_t, 1>();

    auto gpu_g_cpu = gpu.gather_indices.cpu();
    auto gpu_s_cpu = gpu.scatter_indices.cpu();
    auto ref_g_cpu = ref.gather_indices.cpu();
    auto ref_s_cpu = ref.scatter_indices.cpu();

    auto ga  = gpu_g_cpu.accessor<int32_t, 1>();
    auto sa  = gpu_s_cpu.accessor<int32_t, 1>();
    auto rga = ref_g_cpu.accessor<int32_t, 1>();
    auto rsa = ref_s_cpu.accessor<int32_t, 1>();

    for (int64_t k = 0; k < gpu.kernel_volume; ++k) {
        int64_t start = gpu_off[k];
        int64_t end   = gpu_off[k + 1];
        if (end == start)
            continue;

        std::set<std::pair<int32_t, int32_t>> gpu_pairs, ref_pairs;
        for (int64_t i = start; i < end; ++i) {
            gpu_pairs.insert({ga[i], sa[i]});
            ref_pairs.insert({rga[i], rsa[i]});
        }

        EXPECT_EQ(gpu_pairs, ref_pairs)
            << label << ": pair set mismatch at offset k=" << k << " (gpu has " << gpu_pairs.size()
            << " pairs, ref has " << ref_pairs.size() << ")";
    }
}

// -- Parameterized by kernel size, same grid, stride 1 --

class GpuTopoKernelSizeTest : public ::testing::TestWithParam<KernelParam> {};

TEST_P(GpuTopoKernelSizeTest, DenseGridStride1) {
    skipIfCudaUnavailable();
    auto [k0, k1, k2] = GetParam();
    int dim           = std::max({k0, k1, k2, 4}) + 2;
    auto grid         = makeDenseGrid(dim, cudaDevice());
    nanovdb::Coord ks(k0, k1, k2);
    compareGpuVsRefTopology(grid,
                            grid,
                            ks,
                            {1, 1, 1},
                            "gpu_topo_k" + std::to_string(k0) + "x" + std::to_string(k1) + "x" +
                                std::to_string(k2));
}

INSTANTIATE_TEST_SUITE_P(GpuTopoKernelSizes,
                         GpuTopoKernelSizeTest,
                         ::testing::ValuesIn(kernelSizeConfigs()),
                         kernelParamName);

// -- Parameterized by stride, different grids --

class GpuTopoStrideTest : public ::testing::TestWithParam<StrideParam> {};

TEST_P(GpuTopoStrideTest, DenseGridStrided) {
    skipIfCudaUnavailable();
    auto [k0, k1, k2, s0, s1, s2] = GetParam();
    auto device                   = cudaDevice();
    auto src_grid                 = makeDenseGrid(8, device);
    nanovdb::Coord ks(k0, k1, k2);
    nanovdb::Coord stride(s0, s1, s2);
    auto dst_grid = src_grid->convolutionOutput(ks, stride);
    compareGpuVsRefTopology(src_grid,
                            dst_grid,
                            ks,
                            stride,
                            "gpu_topo_k" + std::to_string(k0) + "x" + std::to_string(k1) + "x" +
                                std::to_string(k2) + "_s" + std::to_string(s0) + "x" +
                                std::to_string(s1) + "x" + std::to_string(s2));
}

INSTANTIATE_TEST_SUITE_P(GpuTopoStrides,
                         GpuTopoStrideTest,
                         ::testing::ValuesIn(strideConfigs()),
                         strideParamName);

// -- Sphere shells (realistic sparsity) --

TEST(GpuTopoSphere, Sphere_R10_3x3x3) {
    skipIfCudaUnavailable();
    auto grid = makeSphereShellGrid(10, cudaDevice());
    compareGpuVsRefTopology(grid, grid, {3, 3, 3}, {1, 1, 1}, "gpu_topo_sphere_R10_3x3x3");
}

TEST(GpuTopoSphere, Sphere_R20_3x3x3) {
    skipIfCudaUnavailable();
    auto grid = makeSphereShellGrid(20, cudaDevice());
    compareGpuVsRefTopology(grid, grid, {3, 3, 3}, {1, 1, 1}, "gpu_topo_sphere_R20_3x3x3");
}

TEST(GpuTopoSphere, Sphere_R50_3x3x3) {
    skipIfCudaUnavailable();
    auto grid = makeSphereShellGrid(50, cudaDevice());
    compareGpuVsRefTopology(grid, grid, {3, 3, 3}, {1, 1, 1}, "gpu_topo_sphere_R50_3x3x3");
}

TEST(GpuTopoSphere, Sphere_R20_3x5x1) {
    skipIfCudaUnavailable();
    auto grid = makeSphereShellGrid(20, cudaDevice());
    compareGpuVsRefTopology(grid, grid, {3, 5, 1}, {1, 1, 1}, "gpu_topo_sphere_R20_3x5x1");
}

TEST(GpuTopoSphere, Sphere_R20_2x2x2) {
    skipIfCudaUnavailable();
    auto grid = makeSphereShellGrid(20, cudaDevice());
    compareGpuVsRefTopology(grid, grid, {2, 2, 2}, {1, 1, 1}, "gpu_topo_sphere_R20_2x2x2");
}

TEST(GpuTopoSphere, Sphere_R20_3x4x5_Stride1x2x3) {
    skipIfCudaUnavailable();
    auto device   = cudaDevice();
    auto src_grid = makeSphereShellGrid(20, device);
    nanovdb::Coord ks(3, 4, 5);
    nanovdb::Coord stride(1, 2, 3);
    auto dst_grid = src_grid->convolutionOutput(ks, stride);
    compareGpuVsRefTopology(src_grid, dst_grid, ks, stride, "gpu_topo_sphere_R20_3x4x5_s1x2x3");
}

// -- Edge cases --

TEST(GpuTopoEdgeCases, EmptyTopology) {
    skipIfCudaUnavailable();
    auto device   = cudaDevice();
    auto src_grid = makeGrid(torch::tensor({{0, 0, 0}}, torch::kInt32), device);
    auto dst_grid = makeGrid(torch::tensor({{100, 100, 100}}, torch::kInt32), device);
    compareGpuVsRefTopology(src_grid, dst_grid, {3, 3, 3}, {1, 1, 1}, "gpu_topo_empty");
}

TEST(GpuTopoEdgeCases, SingleVoxelSameGrid) {
    skipIfCudaUnavailable();
    auto grid = makeGrid(torch::tensor({{5, 5, 5}}, torch::kInt32), cudaDevice());
    compareGpuVsRefTopology(grid, grid, {3, 3, 3}, {1, 1, 1}, "gpu_topo_single_same");
}

TEST(GpuTopoEdgeCases, SingleVoxelConvOutput) {
    skipIfCudaUnavailable();
    auto device   = cudaDevice();
    auto src_grid = makeGrid(torch::tensor({{5, 5, 5}}, torch::kInt32), device);
    auto dst_grid = src_grid->convolutionOutput({3, 3, 3}, {1, 1, 1});
    compareGpuVsRefTopology(
        src_grid, dst_grid, {3, 3, 3}, {1, 1, 1}, "gpu_topo_single_conv_output");
}

TEST(GpuTopoEdgeCases, SingleVoxel1x1x1) {
    skipIfCudaUnavailable();
    auto grid = makeGrid(torch::tensor({{5, 5, 5}}, torch::kInt32), cudaDevice());
    compareGpuVsRefTopology(grid, grid, {1, 1, 1}, {1, 1, 1}, "gpu_topo_single_1x1x1");
}

TEST(GpuTopoEdgeCases, LargeEvenKernel4x4x4) {
    skipIfCudaUnavailable();
    auto grid = makeDenseGrid(8, cudaDevice());
    compareGpuVsRefTopology(grid, grid, {4, 4, 4}, {1, 1, 1}, "gpu_topo_even_4x4x4");
}

TEST(GpuTopoEdgeCases, LargeEvenKernel4x4x4_Stride2) {
    skipIfCudaUnavailable();
    auto device   = cudaDevice();
    auto src_grid = makeDenseGrid(8, device);
    nanovdb::Coord ks(4, 4, 4);
    nanovdb::Coord stride(2, 2, 2);
    auto dst_grid = src_grid->convolutionOutput(ks, stride);
    compareGpuVsRefTopology(src_grid, dst_grid, ks, stride, "gpu_topo_even_4x4x4_s2");
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

    auto features = torch::randn({NV, C}, torch::dtype(torch::kFloat16).device(device)) * 0.1;
    auto weights =
        torch::eye(C, torch::dtype(torch::kFloat16).device(device)).reshape({C, C, 1, 1, 1});

    auto topo = ops::cutlassConvTopology(*grid, *grid, {1, 1, 1}, {1, 1, 1});
    auto out  = ops::cutlassConv(features, weights, topo);

    EXPECT_EQ(out.sizes(), features.sizes());

    auto out_f32  = out.to(torch::kFloat32).cpu();
    auto feat_f32 = features.to(torch::kFloat32).cpu();

    EXPECT_TRUE(torch::allclose(out_f32, feat_f32, 0.01, 0.01))
        << "Identity 1x1x1 conv mismatch\n"
        << "  max abs diff = " << (out_f32 - feat_f32).abs().max().item<double>() << "\n"
        << "  mean abs diff = " << (out_f32 - feat_f32).abs().mean().item<double>();
}

// 1x1x1 with random weights = pure matmul, including non-square (Cin != Cout)
TEST(CutlassForwardSmoke, Matmul1x1x1_C64) {
    skipIfCudaUnavailable();
    int64_t const C = 64;

    auto device = cudaDevice();
    auto grid   = makeDenseGrid(4, device);
    int64_t NV  = grid->totalVoxels();

    torch::manual_seed(123);
    auto features = torch::randn({NV, C}, torch::dtype(torch::kFloat16).device(device)) * 0.1;
    auto weights =
        torch::randn({C, C, 1, 1, 1}, torch::dtype(torch::kFloat16).device(device)) * 0.1;

    auto ref_topo = ops::gatherScatterDefaultSparseConvTopology(*grid, *grid, {1, 1, 1}, {1, 1, 1});
    auto ref_out  = ops::gatherScatterDefaultSparseConv(
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
    auto features = torch::randn({NV, 32}, torch::dtype(torch::kFloat16).device(device)) * 0.1;
    auto weights =
        torch::randn({64, 32, 1, 1, 1}, torch::dtype(torch::kFloat16).device(device)) * 0.1;

    auto ref_topo = ops::gatherScatterDefaultSparseConvTopology(*grid, *grid, {1, 1, 1}, {1, 1, 1});
    auto ref_out  = ops::gatherScatterDefaultSparseConv(
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
    auto features = torch::randn({NV, C}, torch::dtype(torch::kFloat16).device(device)) * 0.1;
    auto weights =
        torch::randn({C, C, 3, 3, 3}, torch::dtype(torch::kFloat16).device(device)) * 0.1;

    auto ref_topo = ops::gatherScatterDefaultSparseConvTopology(*grid, *grid, {3, 3, 3}, {1, 1, 1});
    auto ref_out  = ops::gatherScatterDefaultSparseConv(
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
    auto features = torch::randn({NA, C}, torch::dtype(torch::kFloat16).device(device)) * 0.1;
    auto weights =
        torch::randn({C, C, 3, 3, 3}, torch::dtype(torch::kFloat16).device(device)) * 0.1;

    auto ref_topo = ops::gatherScatterDefaultSparseConvTopology(*src_grid, *dst_grid, ks, stride);
    auto ref_out  = ops::gatherScatterDefaultSparseConv(
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
    auto features = torch::randn({NV, C}, torch::dtype(torch::kFloat16).device(device)) * 0.1;
    auto weights =
        torch::randn({C, C, 3, 5, 1}, torch::dtype(torch::kFloat16).device(device)) * 0.1;

    auto ref_topo = ops::gatherScatterDefaultSparseConvTopology(*grid, *grid, {3, 5, 1}, {1, 1, 1});
    auto ref_out  = ops::gatherScatterDefaultSparseConv(
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
    auto features = torch::randn({NV, C}, torch::dtype(torch::kFloat16).device(device)) * 0.1;
    auto weights =
        torch::randn({C, C, 2, 2, 2}, torch::dtype(torch::kFloat16).device(device)) * 0.1;

    auto ref_topo = ops::gatherScatterDefaultSparseConvTopology(*grid, *grid, {2, 2, 2}, {1, 1, 1});
    auto ref_out  = ops::gatherScatterDefaultSparseConv(
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
    auto features = torch::randn({NV, C}, torch::dtype(torch::kFloat16).device(device)) * 0.1;
    auto weights =
        torch::randn({C, C, 3, 3, 3}, torch::dtype(torch::kFloat16).device(device)) * 0.1;

    auto ref_topo = ops::gatherScatterDefaultSparseConvTopology(*grid, *grid, {3, 3, 3}, {1, 1, 1});
    auto ref_out  = ops::gatherScatterDefaultSparseConv(
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
    auto features = torch::randn({NA, C}, torch::dtype(torch::kFloat16).device(device)) * 0.1;
    auto weights =
        torch::randn({C, C, 3, 3, 3}, torch::dtype(torch::kFloat16).device(device)) * 0.1;

    auto ref_topo = ops::gatherScatterDefaultSparseConvTopology(*src_grid, *dst_grid, ks, stride);
    auto ref_out  = ops::gatherScatterDefaultSparseConv(
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
    auto weights_f16  = torch::randn({Cout, Cin, kernel_size[0], kernel_size[1], kernel_size[2]},
                                    torch::dtype(torch::kFloat16).device(device)) *
                       0.1;

    auto ref_topo = ops::gatherScatterDefaultSparseConvTopology(
        *feature_grid, *output_grid, kernel_size, stride);
    auto ref_out = ops::gatherScatterDefaultSparseConv(
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

INSTANTIATE_TEST_SUITE_P(Channels,
                         CutlassChannelTest,
                         ::testing::ValuesIn(channelConfigs()),
                         channelParamName);

// -- Full kernel-size sweep --

class CutlassKernelSizeTest : public ::testing::TestWithParam<KernelParam> {};

TEST_P(CutlassKernelSizeTest, ForwardSameGrid) {
    skipIfCudaUnavailable();
    auto [k0, k1, k2] = GetParam();
    int dim           = std::max({k0, k1, k2, 4}) + 2;
    auto grid         = makeDenseGrid(dim, cudaDevice());
    nanovdb::Coord ks(k0, k1, k2);
    compareForward(grid,
                   grid,
                   ks,
                   {1, 1, 1},
                   64,
                   64,
                   0.05,
                   0.05,
                   "kernel_" + std::to_string(k0) + "x" + std::to_string(k1) + "x" +
                       std::to_string(k2));
}

INSTANTIATE_TEST_SUITE_P(FwdKernelSizes,
                         CutlassKernelSizeTest,
                         ::testing::ValuesIn(kernelSizeConfigs()),
                         kernelParamName);

// -- Full stride sweep --

class CutlassStrideTest : public ::testing::TestWithParam<StrideParam> {};

TEST_P(CutlassStrideTest, ForwardStrided) {
    skipIfCudaUnavailable();
    auto [k0, k1, k2, s0, s1, s2] = GetParam();
    auto device                   = cudaDevice();
    auto src_grid                 = makeDenseGrid(8, device);
    nanovdb::Coord ks(k0, k1, k2);
    nanovdb::Coord stride(s0, s1, s2);
    auto dst_grid = src_grid->convolutionOutput(ks, stride);
    compareForward(src_grid,
                   dst_grid,
                   ks,
                   stride,
                   64,
                   64,
                   0.05,
                   0.05,
                   "stride_k" + std::to_string(k0) + "x" + std::to_string(k1) + "x" +
                       std::to_string(k2) + "_s" + std::to_string(s0) + "x" + std::to_string(s1) +
                       "x" + std::to_string(s2));
}

INSTANTIATE_TEST_SUITE_P(FwdStrides,
                         CutlassStrideTest,
                         ::testing::ValuesIn(strideConfigs()),
                         strideParamName);

// -- Edge-case forward tests --

TEST(CutlassEdgeCases, SingleVoxelForward) {
    skipIfCudaUnavailable();
    auto device   = cudaDevice();
    auto ijk      = torch::tensor({{5, 5, 5}}, torch::kInt32);
    auto grid     = makeGrid(ijk, device);
    auto dst_grid = grid->convolutionOutput({3, 3, 3}, {1, 1, 1});
    compareForward(grid, dst_grid, {3, 3, 3}, {1, 1, 1}, 32, 32, 0.05, 0.05, "single_voxel");
}

TEST(CutlassEdgeCases, EmptyOutputForward) {
    skipIfCudaUnavailable();
    auto device   = cudaDevice();
    auto ijk      = torch::tensor({{0, 0, 0}}, torch::kInt32);
    auto src_grid = makeGrid(ijk, device);
    auto far_ijk  = torch::tensor({{100, 100, 100}}, torch::kInt32);
    auto dst_grid = makeGrid(far_ijk, device);
    auto topo     = ops::cutlassConvTopology(*src_grid, *dst_grid, {3, 3, 3}, {1, 1, 1});
    EXPECT_EQ(topo.total_pairs, 0);
    auto features = torch::randn({1, 32}, torch::dtype(torch::kFloat16).device(device));
    auto weights  = torch::randn({32, 32, 3, 3, 3}, torch::dtype(torch::kFloat16).device(device));
    auto out      = ops::cutlassConv(features, weights, topo);
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

    compareForward(grid, grid, {3, 3, 3}, {1, 1, 1}, 64, 64, 0.05, 0.05, "sphere_R10_3x3x3");
}

// Medium sphere: R=20, ~5K voxels, 3x3x3, same grid
TEST(CutlassSphereTests, Sphere_R20_3x3x3_C64) {
    skipIfCudaUnavailable();
    auto device = cudaDevice();
    auto grid   = makeSphereShellGrid(20, device);

    compareForward(grid, grid, {3, 3, 3}, {1, 1, 1}, 64, 64, 0.05, 0.05, "sphere_R20_3x3x3");
}

// Larger sphere: R=50, ~30K voxels, 3x3x3, same grid, C=128
TEST(CutlassSphereTests, Sphere_R50_3x3x3_C128) {
    skipIfCudaUnavailable();
    auto device = cudaDevice();
    auto grid   = makeSphereShellGrid(50, device);

    compareForward(grid, grid, {3, 3, 3}, {1, 1, 1}, 128, 128, 0.05, 0.05, "sphere_R50_3x3x3_C128");
}

// Sphere with non-uniform kernel: 3x5x1
TEST(CutlassSphereTests, Sphere_R20_3x5x1_C64) {
    skipIfCudaUnavailable();
    auto device = cudaDevice();
    auto grid   = makeSphereShellGrid(20, device);

    compareForward(grid, grid, {3, 5, 1}, {1, 1, 1}, 64, 64, 0.05, 0.05, "sphere_R20_3x5x1");
}

// Sphere with even kernel: 2x2x2
TEST(CutlassSphereTests, Sphere_R20_2x2x2_C64) {
    skipIfCudaUnavailable();
    auto device = cudaDevice();
    auto grid   = makeSphereShellGrid(20, device);

    compareForward(grid, grid, {2, 2, 2}, {1, 1, 1}, 64, 64, 0.05, 0.05, "sphere_R20_2x2x2");
}

// Sphere with stride: 3x3x3, stride 2, different src/dst grids
TEST(CutlassSphereTests, Sphere_R20_3x3x3_Stride2_C64) {
    skipIfCudaUnavailable();
    auto device   = cudaDevice();
    auto src_grid = makeSphereShellGrid(20, device);

    nanovdb::Coord ks(3, 3, 3);
    nanovdb::Coord stride(2, 2, 2);
    auto dst_grid = src_grid->convolutionOutput(ks, stride);

    compareForward(src_grid, dst_grid, ks, stride, 64, 64, 0.05, 0.05, "sphere_R20_3x3x3_stride2");
}

// Sphere with Cin != Cout: channel expansion
TEST(CutlassSphereTests, Sphere_R20_3x3x3_Cin32_Cout128) {
    skipIfCudaUnavailable();
    auto device = cudaDevice();
    auto grid   = makeSphereShellGrid(20, device);

    compareForward(
        grid, grid, {3, 3, 3}, {1, 1, 1}, 32, 128, 0.05, 0.05, "sphere_R20_Cin32_Cout128");
}

// Sphere with mixed odd/even kernel and non-uniform stride
TEST(CutlassSphereTests, Sphere_R20_3x4x5_Stride1x2x3_C64) {
    skipIfCudaUnavailable();
    auto device   = cudaDevice();
    auto src_grid = makeSphereShellGrid(20, device);

    nanovdb::Coord ks(3, 4, 5);
    nanovdb::Coord stride(1, 2, 3);
    auto dst_grid = src_grid->convolutionOutput(ks, stride);

    compareForward(src_grid, dst_grid, ks, stride, 64, 64, 0.05, 0.05, "sphere_R20_3x4x5_s1x2x3");
}

// =============================================================================
// Backward tests -- compare cutlassConvBackward against GatherScatterDefault reference
// =============================================================================
//
// The GatherScatterDefault backward runs in fp32 (reference). The CUTLASS backward
// runs in fp16 with fp32 accumulate.  We compare with tolerances matching
// the forward tests.

static void
compareBackward(c10::intrusive_ptr<GridBatchImpl> const &feature_grid,
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
    auto weights_f16  = torch::randn({Cout, Cin, kernel_size[0], kernel_size[1], kernel_size[2]},
                                    torch::dtype(torch::kFloat16).device(device)) *
                       0.1;
    auto grad_output_f16 =
        torch::randn({NB, Cout}, torch::dtype(torch::kFloat16).device(device)) * 0.1;

    // Reference backward (fp32)
    auto ref_topo = ops::gatherScatterDefaultSparseConvTopology(
        *feature_grid, *output_grid, kernel_size, stride);
    auto [ref_grad_feat, ref_grad_w] =
        ops::gatherScatterDefaultSparseConvBackward(grad_output_f16.to(torch::kFloat32),
                                                    features_f16.to(torch::kFloat32),
                                                    weights_f16.to(torch::kFloat32),
                                                    ref_topo);

    // CUTLASS backward (fp16 with fp32 accumulate)
    auto cutlass_topo = ops::cutlassConvTopology(*feature_grid, *output_grid, kernel_size, stride);
    auto [cutlass_grad_feat, cutlass_grad_w] =
        ops::cutlassConvBackward(grad_output_f16, features_f16, weights_f16, cutlass_topo);

    // -- grad_features --
    EXPECT_EQ(cutlass_grad_feat.sizes(), ref_grad_feat.sizes())
        << label << ": grad_features shape mismatch";

    auto cutlass_gf_f32 = cutlass_grad_feat.to(torch::kFloat32).cpu();
    auto ref_gf_f32     = ref_grad_feat.cpu();

    EXPECT_TRUE(torch::allclose(cutlass_gf_f32, ref_gf_f32, rtol, atol))
        << label << ": grad_features mismatch\n"
        << "  max abs diff = " << (cutlass_gf_f32 - ref_gf_f32).abs().max().item<double>() << "\n"
        << "  mean abs diff = " << (cutlass_gf_f32 - ref_gf_f32).abs().mean().item<double>();

    // -- grad_weights --
    EXPECT_EQ(cutlass_grad_w.sizes(), ref_grad_w.sizes())
        << label << ": grad_weights shape mismatch";

    auto cutlass_gw_f32 = cutlass_grad_w.to(torch::kFloat32).cpu();
    auto ref_gw_f32     = ref_grad_w.cpu();

    EXPECT_TRUE(torch::allclose(cutlass_gw_f32, ref_gw_f32, rtol, atol))
        << label << ": grad_weights mismatch\n"
        << "  max abs diff = " << (cutlass_gw_f32 - ref_gw_f32).abs().max().item<double>() << "\n"
        << "  mean abs diff = " << (cutlass_gw_f32 - ref_gw_f32).abs().mean().item<double>();
}

// -- Backward smoke tests --

TEST(CutlassBackwardSmoke, Backward1x1x1_C64) {
    skipIfCudaUnavailable();
    auto grid = makeDenseGrid(4, cudaDevice());
    compareBackward(grid, grid, {1, 1, 1}, {1, 1, 1}, 64, 64, 0.05, 0.05, "bwd_1x1x1");
}

TEST(CutlassBackwardSmoke, Backward3x3x3_C64) {
    skipIfCudaUnavailable();
    auto grid = makeDenseGrid(6, cudaDevice());
    compareBackward(grid, grid, {3, 3, 3}, {1, 1, 1}, 64, 64, 0.05, 0.05, "bwd_3x3x3");
}

TEST(CutlassBackwardSmoke, Backward3x3x3_Stride2_C64) {
    skipIfCudaUnavailable();
    auto device   = cudaDevice();
    auto src_grid = makeDenseGrid(8, device);
    nanovdb::Coord ks(3, 3, 3);
    nanovdb::Coord stride(2, 2, 2);
    auto dst_grid = src_grid->convolutionOutput(ks, stride);
    compareBackward(src_grid, dst_grid, ks, stride, 64, 64, 0.05, 0.05, "bwd_3x3x3_stride2");
}

TEST(CutlassBackwardSmoke, Backward2x2x2_C64) {
    skipIfCudaUnavailable();
    auto grid = makeDenseGrid(6, cudaDevice());
    compareBackward(grid, grid, {2, 2, 2}, {1, 1, 1}, 64, 64, 0.05, 0.05, "bwd_2x2x2");
}

TEST(CutlassBackwardSmoke, Backward3x5x1_C64) {
    skipIfCudaUnavailable();
    auto grid = makeDenseGrid(7, cudaDevice());
    compareBackward(grid, grid, {3, 5, 1}, {1, 1, 1}, 64, 64, 0.05, 0.05, "bwd_3x5x1");
}

TEST(CutlassBackwardSmoke, Backward_Cin32_Cout64) {
    skipIfCudaUnavailable();
    auto grid = makeDenseGrid(6, cudaDevice());
    compareBackward(grid, grid, {3, 3, 3}, {1, 1, 1}, 32, 64, 0.05, 0.05, "bwd_Cin32_Cout64");
}

TEST(CutlassBackwardSmoke, Backward3x3x3_C128) {
    skipIfCudaUnavailable();
    auto grid = makeDenseGrid(6, cudaDevice());
    compareBackward(grid, grid, {3, 3, 3}, {1, 1, 1}, 128, 128, 0.05, 0.05, "bwd_3x3x3_C128");
}

// -- Backward parameterized by channel --

class CutlassBackwardChannelTest : public ::testing::TestWithParam<ChannelParam> {};

TEST_P(CutlassBackwardChannelTest, Backward3x3x3) {
    skipIfCudaUnavailable();
    auto [Cin, Cout] = GetParam();
    auto grid        = makeDenseGrid(6, cudaDevice());
    compareBackward(grid, grid, {3, 3, 3}, {1, 1, 1}, Cin, Cout, 0.05, 0.05, "bwd_3x3x3_channel");
}

INSTANTIATE_TEST_SUITE_P(BwdChannels,
                         CutlassBackwardChannelTest,
                         ::testing::ValuesIn(channelConfigs()),
                         channelParamName);

// -- Backward parameterized by kernel size --

class CutlassBackwardKernelSizeTest : public ::testing::TestWithParam<KernelParam> {};

TEST_P(CutlassBackwardKernelSizeTest, BackwardSameGrid) {
    skipIfCudaUnavailable();
    auto [k0, k1, k2] = GetParam();
    int dim           = std::max({k0, k1, k2, 4}) + 2;
    auto grid         = makeDenseGrid(dim, cudaDevice());
    nanovdb::Coord ks(k0, k1, k2);
    compareBackward(grid,
                    grid,
                    ks,
                    {1, 1, 1},
                    64,
                    64,
                    0.05,
                    0.05,
                    "bwd_k" + std::to_string(k0) + "x" + std::to_string(k1) + "x" +
                        std::to_string(k2));
}

INSTANTIATE_TEST_SUITE_P(BwdKernelSizes,
                         CutlassBackwardKernelSizeTest,
                         ::testing::ValuesIn(kernelSizeConfigs()),
                         kernelParamName);

// -- Backward parameterized by stride --

class CutlassBackwardStrideTest : public ::testing::TestWithParam<StrideParam> {};

TEST_P(CutlassBackwardStrideTest, BackwardStrided) {
    skipIfCudaUnavailable();
    auto [k0, k1, k2, s0, s1, s2] = GetParam();
    auto device                   = cudaDevice();
    auto src_grid                 = makeDenseGrid(8, device);
    nanovdb::Coord ks(k0, k1, k2);
    nanovdb::Coord stride(s0, s1, s2);
    auto dst_grid = src_grid->convolutionOutput(ks, stride);
    compareBackward(src_grid,
                    dst_grid,
                    ks,
                    stride,
                    64,
                    64,
                    0.05,
                    0.05,
                    "bwd_k" + std::to_string(k0) + "x" + std::to_string(k1) + "x" +
                        std::to_string(k2) + "_s" + std::to_string(s0) + "x" + std::to_string(s1) +
                        "x" + std::to_string(s2));
}

INSTANTIATE_TEST_SUITE_P(BwdStrides,
                         CutlassBackwardStrideTest,
                         ::testing::ValuesIn(strideConfigs()),
                         strideParamName);

// -- Backward sphere shell tests --

TEST(CutlassBackwardSphere, Sphere_R10_3x3x3_C64) {
    skipIfCudaUnavailable();
    auto grid = makeSphereShellGrid(10, cudaDevice());
    compareBackward(grid, grid, {3, 3, 3}, {1, 1, 1}, 64, 64, 0.05, 0.05, "bwd_sphere_R10_3x3x3");
}

TEST(CutlassBackwardSphere, Sphere_R20_3x3x3_C64) {
    skipIfCudaUnavailable();
    auto grid = makeSphereShellGrid(20, cudaDevice());
    compareBackward(grid, grid, {3, 3, 3}, {1, 1, 1}, 64, 64, 0.05, 0.05, "bwd_sphere_R20_3x3x3");
}

TEST(CutlassBackwardSphere, Sphere_R20_2x2x2_C64) {
    skipIfCudaUnavailable();
    auto grid = makeSphereShellGrid(20, cudaDevice());
    compareBackward(grid, grid, {2, 2, 2}, {1, 1, 1}, 64, 64, 0.05, 0.05, "bwd_sphere_R20_2x2x2");
}

TEST(CutlassBackwardSphere, Sphere_R20_3x3x3_Stride2_C64) {
    skipIfCudaUnavailable();
    auto device   = cudaDevice();
    auto src_grid = makeSphereShellGrid(20, device);
    nanovdb::Coord ks(3, 3, 3);
    nanovdb::Coord stride(2, 2, 2);
    auto dst_grid = src_grid->convolutionOutput(ks, stride);
    compareBackward(
        src_grid, dst_grid, ks, stride, 64, 64, 0.05, 0.05, "bwd_sphere_R20_3x3x3_stride2");
}

TEST(CutlassBackwardSphere, Sphere_R20_Cin32_Cout128) {
    skipIfCudaUnavailable();
    auto grid = makeSphereShellGrid(20, cudaDevice());
    compareBackward(
        grid, grid, {3, 3, 3}, {1, 1, 1}, 32, 128, 0.05, 0.05, "bwd_sphere_R20_Cin32_Cout128");
}

TEST(CutlassBackwardSphere, Sphere_R20_3x5x1_C64) {
    skipIfCudaUnavailable();
    auto grid = makeSphereShellGrid(20, cudaDevice());
    compareBackward(grid, grid, {3, 5, 1}, {1, 1, 1}, 64, 64, 0.05, 0.05, "bwd_sphere_R20_3x5x1");
}

TEST(CutlassBackwardSphere, Sphere_R20_3x4x5_Stride1x2x3_C64) {
    skipIfCudaUnavailable();
    auto device   = cudaDevice();
    auto src_grid = makeSphereShellGrid(20, device);
    nanovdb::Coord ks(3, 4, 5);
    nanovdb::Coord stride(1, 2, 3);
    auto dst_grid = src_grid->convolutionOutput(ks, stride);
    compareBackward(
        src_grid, dst_grid, ks, stride, 64, 64, 0.05, 0.05, "bwd_sphere_R20_3x4x5_s1x2x3");
}

// -- Backward edge cases --

TEST(CutlassBackwardEdgeCases, SingleVoxelBackward) {
    skipIfCudaUnavailable();
    auto device   = cudaDevice();
    auto grid     = makeGrid(torch::tensor({{5, 5, 5}}, torch::kInt32), device);
    auto dst_grid = grid->convolutionOutput({3, 3, 3}, {1, 1, 1});
    compareBackward(grid, dst_grid, {3, 3, 3}, {1, 1, 1}, 32, 32, 0.05, 0.05, "bwd_single_voxel");
}

TEST(CutlassBackwardEdgeCases, EmptyOutputBackward) {
    skipIfCudaUnavailable();
    auto device   = cudaDevice();
    auto src_grid = makeGrid(torch::tensor({{0, 0, 0}}, torch::kInt32), device);
    auto dst_grid = makeGrid(torch::tensor({{100, 100, 100}}, torch::kInt32), device);

    auto topo = ops::cutlassConvTopology(*src_grid, *dst_grid, {3, 3, 3}, {1, 1, 1});
    EXPECT_EQ(topo.total_pairs, 0);

    auto features = torch::randn({1, 32}, torch::dtype(torch::kFloat16).device(device));
    auto weights  = torch::randn({32, 32, 3, 3, 3}, torch::dtype(torch::kFloat16).device(device));
    auto grad_output = torch::randn({1, 32}, torch::dtype(torch::kFloat16).device(device));

    auto [grad_feat, grad_w] = ops::cutlassConvBackward(grad_output, features, weights, topo);

    EXPECT_EQ(grad_feat.size(0), 1);
    EXPECT_EQ(grad_feat.size(1), 32);
    EXPECT_TRUE(grad_feat.abs().max().item<float>() == 0.0f);

    EXPECT_EQ(grad_w.sizes(), weights.sizes());
    EXPECT_TRUE(grad_w.abs().max().item<float>() == 0.0f);
}

// =============================================================================
// Transposed convolution tests
// =============================================================================
//
// Transposed conv uses a different topology (probe = (ijk - offset) / stride)
// but identical GEMM operations.  Grid relationship: the transposed output
// grid is built via convolutionTransposeOutput (upsampling).

// -- Transposed topology comparison --

static void
compareTransposeTopology(c10::intrusive_ptr<GridBatchImpl> const &feature_grid,
                         c10::intrusive_ptr<GridBatchImpl> const &output_grid,
                         nanovdb::Coord kernel_size,
                         nanovdb::Coord stride,
                         std::string const &label) {
    auto gpu = ops::cutlassConvTransposeTopology(*feature_grid, *output_grid, kernel_size, stride);
    auto ref = ops::groupedGemmSparseConvTransposeTopology(
        *feature_grid, *output_grid, kernel_size, stride);

    EXPECT_EQ(gpu.kernel_volume, ref.kernel_volume) << label << ": kernel_volume";
    EXPECT_EQ(gpu.feature_total_voxels, ref.feature_total_voxels)
        << label << ": feature_total_voxels";
    EXPECT_EQ(gpu.output_total_voxels, ref.output_total_voxels) << label << ": output_total_voxels";
    EXPECT_EQ(gpu.total_pairs, ref.total_pairs)
        << label << ": total_pairs (gpu=" << gpu.total_pairs << " ref=" << ref.total_pairs << ")";
    ASSERT_EQ(gpu.offsets.size(0), ref.offsets.size(0)) << label << ": offsets size";
    EXPECT_TRUE(torch::equal(gpu.offsets, ref.offsets)) << label << ": offsets content";

    if (gpu.total_pairs == 0)
        return;

    auto gpu_off   = gpu.offsets.accessor<int64_t, 1>();
    auto gpu_g_cpu = gpu.gather_indices.cpu();
    auto gpu_s_cpu = gpu.scatter_indices.cpu();
    auto ref_g_cpu = ref.gather_indices.cpu();
    auto ref_s_cpu = ref.scatter_indices.cpu();
    auto ga        = gpu_g_cpu.accessor<int32_t, 1>();
    auto sa        = gpu_s_cpu.accessor<int32_t, 1>();
    auto rga       = ref_g_cpu.accessor<int32_t, 1>();
    auto rsa       = ref_s_cpu.accessor<int32_t, 1>();

    for (int64_t k = 0; k < gpu.kernel_volume; ++k) {
        int64_t start = gpu_off[k];
        int64_t end   = gpu_off[k + 1];
        if (end == start)
            continue;
        std::set<std::pair<int32_t, int32_t>> gpu_pairs, ref_pairs;
        for (int64_t i = start; i < end; ++i) {
            gpu_pairs.insert({ga[i], sa[i]});
            ref_pairs.insert({rga[i], rsa[i]});
        }
        EXPECT_EQ(gpu_pairs, ref_pairs) << label << ": pair set mismatch at offset k=" << k;
    }
}

class TransposeTopoStrideTest : public ::testing::TestWithParam<StrideParam> {};

TEST_P(TransposeTopoStrideTest, TransposeTopology) {
    skipIfCudaUnavailable();
    auto [k0, k1, k2, s0, s1, s2] = GetParam();
    auto device                   = cudaDevice();
    auto src_grid                 = makeDenseGrid(6, device);
    nanovdb::Coord ks(k0, k1, k2);
    nanovdb::Coord stride(s0, s1, s2);
    auto dst_grid = src_grid->convolutionTransposeOutput(ks, stride);
    compareTransposeTopology(src_grid,
                             dst_grid,
                             ks,
                             stride,
                             "tr_topo_k" + std::to_string(k0) + "x" + std::to_string(k1) + "x" +
                                 std::to_string(k2) + "_s" + std::to_string(s0) + "x" +
                                 std::to_string(s1) + "x" + std::to_string(s2));
}

INSTANTIATE_TEST_SUITE_P(TransposeTopoStrides,
                         TransposeTopoStrideTest,
                         ::testing::ValuesIn(strideConfigs()),
                         strideParamName);

class TransposeTopoKernelSizeTest : public ::testing::TestWithParam<KernelParam> {};

TEST_P(TransposeTopoKernelSizeTest, TransposeTopologySameGrid) {
    skipIfCudaUnavailable();
    auto [k0, k1, k2] = GetParam();
    int dim           = std::max({k0, k1, k2, 4}) + 2;
    auto grid         = makeDenseGrid(dim, cudaDevice());
    nanovdb::Coord ks(k0, k1, k2);
    // stride=1: transpose output == input grid for dense grids
    compareTransposeTopology(grid,
                             grid,
                             ks,
                             {1, 1, 1},
                             "tr_topo_k" + std::to_string(k0) + "x" + std::to_string(k1) + "x" +
                                 std::to_string(k2));
}

INSTANTIATE_TEST_SUITE_P(TransposeTopoKernelSizes,
                         TransposeTopoKernelSizeTest,
                         ::testing::ValuesIn(kernelSizeConfigs()),
                         kernelParamName);

// -- Transposed forward tests --

static void
compareTransposeForward(c10::intrusive_ptr<GridBatchImpl> const &feature_grid,
                        c10::intrusive_ptr<GridBatchImpl> const &output_grid,
                        nanovdb::Coord kernel_size,
                        nanovdb::Coord stride,
                        int64_t Cin,
                        int64_t Cout,
                        double rtol,
                        double atol,
                        std::string const &label) {
    auto device = cudaDevice();
    int64_t NA  = feature_grid->totalVoxels();

    torch::manual_seed(42);
    auto features_f16 = torch::randn({NA, Cin}, torch::dtype(torch::kFloat16).device(device)) * 0.1;
    auto weights_f16  = torch::randn({Cout, Cin, kernel_size[0], kernel_size[1], kernel_size[2]},
                                    torch::dtype(torch::kFloat16).device(device)) *
                       0.1;

    auto ref_topo = ops::gatherScatterDefaultSparseConvTransposeTopology(
        *feature_grid, *output_grid, kernel_size, stride);
    auto ref_out = ops::gatherScatterDefaultSparseConvTranspose(
        features_f16.to(torch::kFloat32), weights_f16.to(torch::kFloat32), ref_topo);

    auto cutlass_topo =
        ops::cutlassConvTransposeTopology(*feature_grid, *output_grid, kernel_size, stride);
    auto cutlass_out = ops::cutlassConvTranspose(features_f16, weights_f16, cutlass_topo);

    EXPECT_EQ(cutlass_out.sizes(), ref_out.sizes()) << label << ": shape mismatch";

    auto cutlass_f32 = cutlass_out.to(torch::kFloat32).cpu();
    auto ref_f32     = ref_out.cpu();

    EXPECT_TRUE(torch::allclose(cutlass_f32, ref_f32, rtol, atol))
        << label << ": numerical mismatch\n"
        << "  max abs diff = " << (cutlass_f32 - ref_f32).abs().max().item<double>() << "\n"
        << "  mean abs diff = " << (cutlass_f32 - ref_f32).abs().mean().item<double>();
}

TEST(CutlassTransposeForward, Transpose3x3x3_Stride2_C64) {
    skipIfCudaUnavailable();
    auto device   = cudaDevice();
    auto src_grid = makeDenseGrid(4, device);
    nanovdb::Coord ks(3, 3, 3);
    nanovdb::Coord stride(2, 2, 2);
    auto dst_grid = src_grid->convolutionTransposeOutput(ks, stride);
    compareTransposeForward(src_grid, dst_grid, ks, stride, 64, 64, 0.05, 0.05, "tr_fwd_3x3x3_s2");
}

TEST(CutlassTransposeForward, Transpose3x3x3_SameGrid_C64) {
    skipIfCudaUnavailable();
    auto grid = makeDenseGrid(6, cudaDevice());
    compareTransposeForward(
        grid, grid, {3, 3, 3}, {1, 1, 1}, 64, 64, 0.05, 0.05, "tr_fwd_3x3x3_same");
}

TEST(CutlassTransposeForward, Transpose2x2x2_Stride2_C64) {
    skipIfCudaUnavailable();
    auto device   = cudaDevice();
    auto src_grid = makeDenseGrid(4, device);
    nanovdb::Coord ks(2, 2, 2);
    nanovdb::Coord stride(2, 2, 2);
    auto dst_grid = src_grid->convolutionTransposeOutput(ks, stride);
    compareTransposeForward(src_grid, dst_grid, ks, stride, 64, 64, 0.05, 0.05, "tr_fwd_2x2x2_s2");
}

TEST(CutlassTransposeForward, Transpose3x4x5_Stride1x2x3_C64) {
    skipIfCudaUnavailable();
    auto device   = cudaDevice();
    auto src_grid = makeDenseGrid(4, device);
    nanovdb::Coord ks(3, 4, 5);
    nanovdb::Coord stride(1, 2, 3);
    auto dst_grid = src_grid->convolutionTransposeOutput(ks, stride);
    compareTransposeForward(
        src_grid, dst_grid, ks, stride, 64, 64, 0.05, 0.05, "tr_fwd_3x4x5_s1x2x3");
}

TEST(CutlassTransposeForward, Transpose_Cin32_Cout128) {
    skipIfCudaUnavailable();
    auto device   = cudaDevice();
    auto src_grid = makeDenseGrid(4, device);
    nanovdb::Coord ks(3, 3, 3);
    nanovdb::Coord stride(2, 2, 2);
    auto dst_grid = src_grid->convolutionTransposeOutput(ks, stride);
    compareTransposeForward(
        src_grid, dst_grid, ks, stride, 32, 128, 0.05, 0.05, "tr_fwd_Cin32_Cout128");
}

TEST(CutlassTransposeForward, Transpose_Sphere_R10_3x3x3_Stride2_C64) {
    skipIfCudaUnavailable();
    auto device   = cudaDevice();
    auto src_grid = makeSphereShellGrid(10, device);
    nanovdb::Coord ks(3, 3, 3);
    nanovdb::Coord stride(2, 2, 2);
    auto dst_grid = src_grid->convolutionTransposeOutput(ks, stride);
    compareTransposeForward(
        src_grid, dst_grid, ks, stride, 64, 64, 0.05, 0.05, "tr_fwd_sphere_R10_3x3x3_s2");
}

// -- Transposed backward tests --

static void
compareTransposeBackward(c10::intrusive_ptr<GridBatchImpl> const &feature_grid,
                         c10::intrusive_ptr<GridBatchImpl> const &output_grid,
                         nanovdb::Coord kernel_size,
                         nanovdb::Coord stride,
                         int64_t Cin,
                         int64_t Cout,
                         double rtol,
                         double atol,
                         std::string const &label) {
    auto device = cudaDevice();
    int64_t NA  = feature_grid->totalVoxels();
    int64_t NB  = output_grid->totalVoxels();

    torch::manual_seed(42);
    auto features_f16 = torch::randn({NA, Cin}, torch::dtype(torch::kFloat16).device(device)) * 0.1;
    auto weights_f16  = torch::randn({Cout, Cin, kernel_size[0], kernel_size[1], kernel_size[2]},
                                    torch::dtype(torch::kFloat16).device(device)) *
                       0.1;
    auto grad_output_f16 =
        torch::randn({NB, Cout}, torch::dtype(torch::kFloat16).device(device)) * 0.1;

    auto ref_topo = ops::gatherScatterDefaultSparseConvTransposeTopology(
        *feature_grid, *output_grid, kernel_size, stride);
    auto [ref_grad_feat, ref_grad_w] =
        ops::gatherScatterDefaultSparseConvTransposeBackward(grad_output_f16.to(torch::kFloat32),
                                                             features_f16.to(torch::kFloat32),
                                                             weights_f16.to(torch::kFloat32),
                                                             ref_topo);

    auto cutlass_topo =
        ops::cutlassConvTransposeTopology(*feature_grid, *output_grid, kernel_size, stride);
    auto [cutlass_grad_feat, cutlass_grad_w] =
        ops::cutlassConvTransposeBackward(grad_output_f16, features_f16, weights_f16, cutlass_topo);

    EXPECT_EQ(cutlass_grad_feat.sizes(), ref_grad_feat.sizes())
        << label << ": grad_features shape mismatch";
    auto cutlass_gf_f32 = cutlass_grad_feat.to(torch::kFloat32).cpu();
    auto ref_gf_f32     = ref_grad_feat.cpu();
    EXPECT_TRUE(torch::allclose(cutlass_gf_f32, ref_gf_f32, rtol, atol))
        << label << ": grad_features mismatch\n"
        << "  max abs diff = " << (cutlass_gf_f32 - ref_gf_f32).abs().max().item<double>() << "\n"
        << "  mean abs diff = " << (cutlass_gf_f32 - ref_gf_f32).abs().mean().item<double>();

    EXPECT_EQ(cutlass_grad_w.sizes(), ref_grad_w.sizes())
        << label << ": grad_weights shape mismatch";
    auto cutlass_gw_f32 = cutlass_grad_w.to(torch::kFloat32).cpu();
    auto ref_gw_f32     = ref_grad_w.cpu();
    EXPECT_TRUE(torch::allclose(cutlass_gw_f32, ref_gw_f32, rtol, atol))
        << label << ": grad_weights mismatch\n"
        << "  max abs diff = " << (cutlass_gw_f32 - ref_gw_f32).abs().max().item<double>() << "\n"
        << "  mean abs diff = " << (cutlass_gw_f32 - ref_gw_f32).abs().mean().item<double>();
}

TEST(CutlassTransposeBackward, Transpose3x3x3_Stride2_C64) {
    skipIfCudaUnavailable();
    auto device   = cudaDevice();
    auto src_grid = makeDenseGrid(4, device);
    nanovdb::Coord ks(3, 3, 3);
    nanovdb::Coord stride(2, 2, 2);
    auto dst_grid = src_grid->convolutionTransposeOutput(ks, stride);
    compareTransposeBackward(src_grid, dst_grid, ks, stride, 64, 64, 0.05, 0.05, "tr_bwd_3x3x3_s2");
}

TEST(CutlassTransposeBackward, Transpose3x3x3_SameGrid_C64) {
    skipIfCudaUnavailable();
    auto grid = makeDenseGrid(6, cudaDevice());
    compareTransposeBackward(
        grid, grid, {3, 3, 3}, {1, 1, 1}, 64, 64, 0.05, 0.05, "tr_bwd_3x3x3_same");
}

TEST(CutlassTransposeBackward, Transpose2x2x2_Stride2_C64) {
    skipIfCudaUnavailable();
    auto device   = cudaDevice();
    auto src_grid = makeDenseGrid(4, device);
    nanovdb::Coord ks(2, 2, 2);
    nanovdb::Coord stride(2, 2, 2);
    auto dst_grid = src_grid->convolutionTransposeOutput(ks, stride);
    compareTransposeBackward(src_grid, dst_grid, ks, stride, 64, 64, 0.05, 0.05, "tr_bwd_2x2x2_s2");
}

TEST(CutlassTransposeBackward, Transpose3x4x5_Stride1x2x3_C64) {
    skipIfCudaUnavailable();
    auto device   = cudaDevice();
    auto src_grid = makeDenseGrid(4, device);
    nanovdb::Coord ks(3, 4, 5);
    nanovdb::Coord stride(1, 2, 3);
    auto dst_grid = src_grid->convolutionTransposeOutput(ks, stride);
    compareTransposeBackward(
        src_grid, dst_grid, ks, stride, 64, 64, 0.05, 0.05, "tr_bwd_3x4x5_s1x2x3");
}

TEST(CutlassTransposeBackward, Transpose_Cin32_Cout128) {
    skipIfCudaUnavailable();
    auto device   = cudaDevice();
    auto src_grid = makeDenseGrid(4, device);
    nanovdb::Coord ks(3, 3, 3);
    nanovdb::Coord stride(2, 2, 2);
    auto dst_grid = src_grid->convolutionTransposeOutput(ks, stride);
    compareTransposeBackward(
        src_grid, dst_grid, ks, stride, 32, 128, 0.05, 0.05, "tr_bwd_Cin32_Cout128");
}

TEST(CutlassTransposeBackward, Transpose_Sphere_R10_3x3x3_Stride2_C64) {
    skipIfCudaUnavailable();
    auto device   = cudaDevice();
    auto src_grid = makeSphereShellGrid(10, device);
    nanovdb::Coord ks(3, 3, 3);
    nanovdb::Coord stride(2, 2, 2);
    auto dst_grid = src_grid->convolutionTransposeOutput(ks, stride);
    compareTransposeBackward(
        src_grid, dst_grid, ks, stride, 64, 64, 0.05, 0.05, "tr_bwd_sphere_R10_3x3x3_s2");
}

// -- Transposed edge cases --

TEST(CutlassTransposeEdgeCases, SingleVoxelTranspose) {
    skipIfCudaUnavailable();
    auto device   = cudaDevice();
    auto grid     = makeGrid(torch::tensor({{5, 5, 5}}, torch::kInt32), device);
    auto dst_grid = grid->convolutionTransposeOutput({3, 3, 3}, {2, 2, 2});
    compareTransposeForward(
        grid, dst_grid, {3, 3, 3}, {2, 2, 2}, 32, 32, 0.05, 0.05, "tr_fwd_single_voxel");
    compareTransposeBackward(
        grid, dst_grid, {3, 3, 3}, {2, 2, 2}, 32, 32, 0.05, 0.05, "tr_bwd_single_voxel");
}

TEST(CutlassTransposeEdgeCases, EmptyTranspose) {
    skipIfCudaUnavailable();
    auto device   = cudaDevice();
    auto src_grid = makeGrid(torch::tensor({{0, 0, 0}}, torch::kInt32), device);
    auto dst_grid = makeGrid(torch::tensor({{100, 100, 100}}, torch::kInt32), device);

    auto topo = ops::cutlassConvTransposeTopology(*src_grid, *dst_grid, {3, 3, 3}, {1, 1, 1});
    EXPECT_EQ(topo.total_pairs, 0);
}
