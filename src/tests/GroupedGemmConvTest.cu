// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// GroupedGemmConvTest.cu -- Tests for the compacted gather-scatter sparse convolution op.
//
// Every test compares the GroupedGemm backend output against the GatherScatter
// (GEMM) backend output, which serves as the reference implementation.
//
// Tests are parameterized by (device, scalar type, channel count) and cover
// CPU and CUDA with float32 and float64.  A separate kernel-size test suite
// exercises non-cubic and varied kernel dimensions.
//
#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/GridBatchImpl.h>
#include <fvdb/detail/ops/convolution/GatherScatter.h>
#include <fvdb/detail/ops/convolution/GroupedGemm.h>

#include <torch/types.h>

#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <tuple>
#include <vector>

using namespace fvdb;
using namespace fvdb::detail;

// =============================================================================
// Test parameter: (device_type, scalar_type, channel_count)
// =============================================================================

using GroupedGemmParam = std::tuple<torch::DeviceType, torch::ScalarType, int64_t>;

// Shorthand constructors -- keep INSTANTIATE_TEST_SUITE_P lines readable
// without triggering the unused-variable warning that Combine causes in
// gtest's macro-generated code (which crashes NVCC via -Werror).
static GroupedGemmParam
gp(torch::DeviceType d, torch::ScalarType s, int64_t c) {
    return std::make_tuple(d, s, c);
}

static std::string
paramName(::testing::TestParamInfo<GroupedGemmParam> const &info) {
    auto dev   = std::get<0>(info.param);
    auto dtype = std::get<1>(info.param);
    auto C     = std::get<2>(info.param);

    std::string dev_str   = (dev == torch::kCPU) ? "CPU" : "CUDA";
    std::string dtype_str = (dtype == torch::kFloat32) ? "f32" : "f64";
    return dev_str + "_" + dtype_str + "_C" + std::to_string(C);
}

// =============================================================================
// Helpers
// =============================================================================

// Tolerances: float64 is much tighter; float32 allows TF32 drift on Ampere+.
static std::pair<double, double>
tolerances(torch::ScalarType dtype) {
    if (dtype == torch::kFloat64) {
        return {1e-8, 1e-8};
    }
    return {1e-3, 1e-3};
}

static bool
cudaIsAvailable() {
    int count = 0;
    auto err  = cudaGetDeviceCount(&count);
    return err == cudaSuccess && count > 0;
}

static void
skipIfCudaUnavailable(torch::DeviceType dev) {
    if (dev == torch::kCUDA && !cudaIsAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }
}

static torch::Device
makeDevice(torch::DeviceType dev) {
    return (dev == torch::kCUDA) ? torch::Device(torch::kCUDA, 0) : torch::Device(torch::kCPU);
}

static c10::intrusive_ptr<GridBatchImpl>
makeGrid(torch::Tensor ijk_2d, torch::Device device) {
    auto ijk_dev = ijk_2d.to(device);
    JaggedTensor jt(ijk_dev);
    std::vector<nanovdb::Vec3d> voxel_sizes = {{1.0, 1.0, 1.0}};
    std::vector<nanovdb::Vec3d> origins     = {{0.0, 0.0, 0.0}};
    return GridBatchImpl::createFromIjk(jt, voxel_sizes, origins);
}

// Dense cube grid of side length `dim` from explicit ijk coordinates.
static c10::intrusive_ptr<GridBatchImpl>
makeDenseTestGrid(int dim, torch::Device device) {
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

static torch::TensorOptions
opts(torch::Device device, torch::ScalarType dtype) {
    return torch::dtype(dtype).device(device);
}

// =============================================================================
// Forward tests -- GroupedGemm vs GatherScatter (GEMM) reference
// =============================================================================

class GroupedGemmForwardTest : public ::testing::TestWithParam<GroupedGemmParam> {};

TEST_P(GroupedGemmForwardTest, SameGridStride1) {
    auto [dev_type, dtype, C] = GetParam();
    skipIfCudaUnavailable(dev_type);
    auto device       = makeDevice(dev_type);
    auto [rtol, atol] = tolerances(dtype);

    // 2x2x2 block
    auto ijk = torch::tensor(
        {{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1}, {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}},
        torch::dtype(torch::kInt32));
    auto grid = makeGrid(ijk, device);

    int64_t N = 8;
    nanovdb::Coord kernel_size(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);

    auto topo_gemm    = ops::gatherScatterSparseConvTopology(*grid, *grid, kernel_size, stride);
    auto topo_grouped = ops::groupedGemmSparseConvTopology(*grid, *grid, kernel_size, stride);

    torch::manual_seed(42);
    auto features = torch::randn({N, C}, opts(device, dtype));
    auto weights  = torch::randn({C, C, 3, 3, 3}, opts(device, dtype));

    auto ref = ops::gatherScatterSparseConv(features, weights, topo_gemm);
    auto out = ops::groupedGemmSparseConv(features, weights, topo_grouped);

    EXPECT_EQ(out.sizes(), ref.sizes());
    EXPECT_TRUE(torch::allclose(out.cpu(), ref.cpu(), rtol, atol)) << "Forward mismatch at C=" << C;
}

TEST_P(GroupedGemmForwardTest, IdentityConv1x1x1) {
    auto [dev_type, dtype, C] = GetParam();
    skipIfCudaUnavailable(dev_type);
    auto device       = makeDevice(dev_type);
    auto [rtol, atol] = tolerances(dtype);

    auto ijk = torch::tensor(
        {{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1}, {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}},
        torch::dtype(torch::kInt32));
    auto grid = makeGrid(ijk, device);
    int64_t N = 8;

    nanovdb::Coord kernel_size(1, 1, 1);
    nanovdb::Coord stride(1, 1, 1);

    auto topo = ops::groupedGemmSparseConvTopology(*grid, *grid, kernel_size, stride);

    auto features = torch::randn({N, C}, opts(device, dtype));
    auto weights  = torch::eye(C, opts(device, dtype)).reshape({C, C, 1, 1, 1});

    auto out = ops::groupedGemmSparseConv(features, weights, topo);

    // With identity weights, output should equal input
    EXPECT_TRUE(torch::allclose(out.cpu(), features.cpu(), rtol, atol))
        << "Identity conv mismatch at C=" << C;
}

TEST_P(GroupedGemmForwardTest, DifferentGrids) {
    auto [dev_type, dtype, C] = GetParam();
    skipIfCudaUnavailable(dev_type);
    auto device       = makeDevice(dev_type);
    auto [rtol, atol] = tolerances(dtype);

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
    auto features = torch::randn({1, C}, opts(device, dtype));
    auto weights  = torch::randn({C, C, 3, 3, 3}, opts(device, dtype));

    auto ref = ops::gatherScatterSparseConv(features, weights, topo_gemm);
    auto out = ops::groupedGemmSparseConv(features, weights, topo_grouped);

    EXPECT_EQ(out.sizes(), ref.sizes());
    EXPECT_TRUE(torch::allclose(out.cpu(), ref.cpu(), rtol, atol))
        << "Different-grid forward mismatch at C=" << C;
}

// =============================================================================
// Backward tests -- grad_features and grad_weights vs GatherScatter reference
// =============================================================================

class GroupedGemmBackwardTest : public ::testing::TestWithParam<GroupedGemmParam> {};

TEST_P(GroupedGemmBackwardTest, SameGridStride1) {
    auto [dev_type, dtype, C] = GetParam();
    skipIfCudaUnavailable(dev_type);
    auto device       = makeDevice(dev_type);
    auto [rtol, atol] = tolerances(dtype);

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
    auto features    = torch::randn({N, C}, opts(device, dtype));
    auto weights     = torch::randn({C, C, 3, 3, 3}, opts(device, dtype));
    auto grad_output = torch::randn({N, C}, opts(device, dtype));

    auto [gf_ref, gw_ref] =
        ops::gatherScatterSparseConvBackward(grad_output, features, weights, topo_gemm);
    auto [gf_out, gw_out] =
        ops::groupedGemmSparseConvBackward(grad_output, features, weights, topo_grouped);

    EXPECT_EQ(gf_out.sizes(), gf_ref.sizes());
    EXPECT_EQ(gw_out.sizes(), gw_ref.sizes());

    EXPECT_TRUE(torch::allclose(gf_out.cpu(), gf_ref.cpu(), rtol, atol))
        << "Backward grad_features mismatch at C=" << C;
    EXPECT_TRUE(torch::allclose(gw_out.cpu(), gw_ref.cpu(), rtol, atol))
        << "Backward grad_weights mismatch at C=" << C;
}

// =============================================================================
// Transposed forward tests
// =============================================================================

class GroupedGemmTransposeTest : public ::testing::TestWithParam<GroupedGemmParam> {};

TEST_P(GroupedGemmTransposeTest, TransposeForwardSameGrid) {
    auto [dev_type, dtype, C] = GetParam();
    skipIfCudaUnavailable(dev_type);
    auto device       = makeDevice(dev_type);
    auto [rtol, atol] = tolerances(dtype);

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
    auto features = torch::randn({N, C}, opts(device, dtype));
    auto weights  = torch::randn({C, C, 3, 3, 3}, opts(device, dtype));

    auto ref = ops::gatherScatterSparseConvTranspose(features, weights, topo_gemm);
    auto out = ops::groupedGemmSparseConvTranspose(features, weights, topo_grouped);

    EXPECT_EQ(out.sizes(), ref.sizes());
    EXPECT_TRUE(torch::allclose(out.cpu(), ref.cpu(), rtol, atol))
        << "Transpose forward mismatch at C=" << C;
}

// =============================================================================
// Transposed backward tests
// =============================================================================

TEST_P(GroupedGemmTransposeTest, TransposeBackwardSameGrid) {
    auto [dev_type, dtype, C] = GetParam();
    skipIfCudaUnavailable(dev_type);
    auto device       = makeDevice(dev_type);
    auto [rtol, atol] = tolerances(dtype);

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
    auto features    = torch::randn({N, C}, opts(device, dtype));
    auto weights     = torch::randn({C, C, 3, 3, 3}, opts(device, dtype));
    auto grad_output = torch::randn({N, C}, opts(device, dtype));

    auto [gf_ref, gw_ref] =
        ops::gatherScatterSparseConvTransposeBackward(grad_output, features, weights, topo_gemm);
    auto [gf_out, gw_out] =
        ops::groupedGemmSparseConvTransposeBackward(grad_output, features, weights, topo_grouped);

    EXPECT_EQ(gf_out.sizes(), gf_ref.sizes());
    EXPECT_EQ(gw_out.sizes(), gw_ref.sizes());

    EXPECT_TRUE(torch::allclose(gf_out.cpu(), gf_ref.cpu(), rtol, atol))
        << "Transpose backward grad_features mismatch at C=" << C;
    EXPECT_TRUE(torch::allclose(gw_out.cpu(), gw_ref.cpu(), rtol, atol))
        << "Transpose backward grad_weights mismatch at C=" << C;
}

// =============================================================================
// Strided tests
// =============================================================================

class GroupedGemmStridedTest : public ::testing::TestWithParam<GroupedGemmParam> {};

TEST_P(GroupedGemmStridedTest, StridedForward) {
    auto [dev_type, dtype, C] = GetParam();
    skipIfCudaUnavailable(dev_type);
    auto device       = makeDevice(dev_type);
    auto [rtol, atol] = tolerances(dtype);

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
    auto features = torch::randn({S, C}, opts(device, dtype));
    auto weights  = torch::randn({C, C, 3, 3, 3}, opts(device, dtype));

    auto ref = ops::gatherScatterSparseConv(features, weights, topo_gemm);
    auto out = ops::groupedGemmSparseConv(features, weights, topo_grouped);

    EXPECT_EQ(out.sizes(), ref.sizes());
    EXPECT_TRUE(torch::allclose(out.cpu(), ref.cpu(), rtol, atol))
        << "Strided forward mismatch at C=" << C;
}

TEST_P(GroupedGemmStridedTest, StridedBackward) {
    auto [dev_type, dtype, C] = GetParam();
    skipIfCudaUnavailable(dev_type);
    auto device       = makeDevice(dev_type);
    auto [rtol, atol] = tolerances(dtype);

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
    auto features    = torch::randn({S, C}, opts(device, dtype));
    auto weights     = torch::randn({C, C, 3, 3, 3}, opts(device, dtype));
    auto grad_output = torch::randn({D, C}, opts(device, dtype));

    auto [gf_ref, gw_ref] =
        ops::gatherScatterSparseConvBackward(grad_output, features, weights, topo_gemm);
    auto [gf_out, gw_out] =
        ops::groupedGemmSparseConvBackward(grad_output, features, weights, topo_grouped);

    EXPECT_EQ(gf_out.sizes(), gf_ref.sizes());
    EXPECT_EQ(gw_out.sizes(), gw_ref.sizes());

    EXPECT_TRUE(torch::allclose(gf_out.cpu(), gf_ref.cpu(), rtol, atol))
        << "Strided backward grad_features mismatch at C=" << C;
    EXPECT_TRUE(torch::allclose(gw_out.cpu(), gw_ref.cpu(), rtol, atol))
        << "Strided backward grad_weights mismatch at C=" << C;
}

// =============================================================================
// Kernel-size tests -- varied and asymmetric kernel dimensions
// =============================================================================
//
// Exercises kernel sizes beyond 3x3x3: larger cubes, asymmetric shapes, and
// even-sized kernels.  Uses a 4x4x4 dense grid and fixed C=16 to keep the
// combinatorial space manageable while covering a wide range of shapes.

using KernelSizeParam = std::tuple<torch::DeviceType, torch::ScalarType, int, int, int>;

static KernelSizeParam
kp(torch::DeviceType d, torch::ScalarType s, int k0, int k1, int k2) {
    return std::make_tuple(d, s, k0, k1, k2);
}

static std::string
kernelSizeParamName(::testing::TestParamInfo<KernelSizeParam> const &info) {
    auto dev = std::get<0>(info.param);
    auto dt  = std::get<1>(info.param);
    auto k0  = std::get<2>(info.param);
    auto k1  = std::get<3>(info.param);
    auto k2  = std::get<4>(info.param);

    std::string dev_str   = (dev == torch::kCPU) ? "CPU" : "CUDA";
    std::string dtype_str = (dt == torch::kFloat32) ? "f32" : "f64";
    return dev_str + "_" + dtype_str + "_k" + std::to_string(k0) + "x" + std::to_string(k1) + "x" +
           std::to_string(k2);
}

class GroupedGemmKernelSizeTest : public ::testing::TestWithParam<KernelSizeParam> {};

TEST_P(GroupedGemmKernelSizeTest, ForwardAndBackward) {
    auto [dev_type, dtype, k0, k1, k2] = GetParam();
    skipIfCudaUnavailable(dev_type);
    auto device       = makeDevice(dev_type);
    auto [rtol, atol] = tolerances(dtype);

    int64_t const C = 16;

    // 4x4x4 grid: large enough for kernels up to 5x5x5
    auto grid       = makeDenseTestGrid(4, device);
    int64_t const N = 64;

    nanovdb::Coord kernel_size(k0, k1, k2);
    nanovdb::Coord stride(1, 1, 1);

    auto topo_gemm    = ops::gatherScatterSparseConvTopology(*grid, *grid, kernel_size, stride);
    auto topo_grouped = ops::groupedGemmSparseConvTopology(*grid, *grid, kernel_size, stride);

    torch::manual_seed(42);
    auto features = torch::randn({N, C}, opts(device, dtype));
    auto weights  = torch::randn({C, C, k0, k1, k2}, opts(device, dtype));

    // Forward
    auto ref = ops::gatherScatterSparseConv(features, weights, topo_gemm);
    auto out = ops::groupedGemmSparseConv(features, weights, topo_grouped);

    EXPECT_EQ(out.sizes(), ref.sizes());
    EXPECT_TRUE(torch::allclose(out.cpu(), ref.cpu(), rtol, atol))
        << "Forward mismatch for kernel " << k0 << "x" << k1 << "x" << k2;

    // Backward
    auto grad_output = torch::randn_like(ref);

    auto [gf_ref, gw_ref] =
        ops::gatherScatterSparseConvBackward(grad_output, features, weights, topo_gemm);
    auto [gf_out, gw_out] =
        ops::groupedGemmSparseConvBackward(grad_output, features, weights, topo_grouped);

    EXPECT_EQ(gf_out.sizes(), gf_ref.sizes());
    EXPECT_EQ(gw_out.sizes(), gw_ref.sizes());

    EXPECT_TRUE(torch::allclose(gf_out.cpu(), gf_ref.cpu(), rtol, atol))
        << "Backward grad_features mismatch for kernel " << k0 << "x" << k1 << "x" << k2;
    EXPECT_TRUE(torch::allclose(gw_out.cpu(), gw_ref.cpu(), rtol, atol))
        << "Backward grad_weights mismatch for kernel " << k0 << "x" << k1 << "x" << k2;
}

// =============================================================================
// Test instantiation -- parameterized by (device, dtype, channel count)
// =============================================================================

// clang-format off
//
// Uses ValuesIn() with runtime vectors instead of Values() to avoid an
// unused-variable warning in gtest's INSTANTIATE_TEST_SUITE_P macro
// that triggers -Werror under NVCC's host compiler (GCC).

static std::vector<GroupedGemmParam>
channelConfigs() {
    std::vector<GroupedGemmParam> v;
    for (auto dev : {torch::kCPU, torch::kCUDA}) {
        for (auto dt : {torch::kFloat32, torch::kFloat64}) {
            for (int64_t C : {4, 16, 32, 64}) {
                v.push_back(gp(dev, dt, C));
            }
        }
        // Extra large channel count for CUDA float32
        v.push_back(gp(dev, torch::kFloat32, 128));
    }
    return v;
}

static std::vector<KernelSizeParam>
kernelSizeConfigs() {
    std::vector<KernelSizeParam> v;
    for (auto dev : {torch::kCPU, torch::kCUDA}) {
        // float32: full matrix of kernel shapes
        for (auto [k0, k1, k2] : std::vector<std::tuple<int,int,int>>{
                 {1,1,1}, {3,3,3}, {5,5,5}, {3,5,1}, {1,3,5}, {3,1,3}, {2,2,2}, {4,4,4}}) {
            v.push_back(kp(dev, torch::kFloat32, k0, k1, k2));
        }
        // float64: representative subset
        for (auto [k0, k1, k2] : std::vector<std::tuple<int,int,int>>{
                 {3,3,3}, {5,5,5}, {3,5,1}}) {
            v.push_back(kp(dev, torch::kFloat64, k0, k1, k2));
        }
    }
    return v;
}

INSTANTIATE_TEST_SUITE_P(Configs, GroupedGemmForwardTest,
    ::testing::ValuesIn(channelConfigs()), paramName);

INSTANTIATE_TEST_SUITE_P(Configs, GroupedGemmBackwardTest,
    ::testing::ValuesIn(channelConfigs()), paramName);

INSTANTIATE_TEST_SUITE_P(Configs, GroupedGemmTransposeTest,
    ::testing::ValuesIn(channelConfigs()), paramName);

INSTANTIATE_TEST_SUITE_P(Configs, GroupedGemmStridedTest,
    ::testing::ValuesIn(channelConfigs()), paramName);

INSTANTIATE_TEST_SUITE_P(KernelSizes, GroupedGemmKernelSizeTest,
    ::testing::ValuesIn(kernelSizeConfigs()), kernelSizeParamName);

// clang-format on
