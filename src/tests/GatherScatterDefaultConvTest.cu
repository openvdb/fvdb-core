// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// GatherScatterDefaultConvTest.cu -- Tests for the default gather-scatter sparse convolution op.
//
// Tests are parameterized by (device, scalar type, channel count) and cover
// CPU and CUDA with float32 and float64.  A separate kernel-size test suite
// exercises non-cubic and varied kernel dimensions.
//
#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/GridBatchImpl.h>
#include <fvdb/detail/ops/convolution/GatherScatterDefault.h>

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

using ConvParam = std::tuple<torch::DeviceType, torch::ScalarType, int64_t>;

static ConvParam
cp(torch::DeviceType d, torch::ScalarType s, int64_t c) {
    return std::make_tuple(d, s, c);
}

static std::string
paramName(::testing::TestParamInfo<ConvParam> const &info) {
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
// Forward tests
// =============================================================================

class GatherScatterDefaultForwardTest : public ::testing::TestWithParam<ConvParam> {};

TEST_P(GatherScatterDefaultForwardTest, SameGridStride1) {
    auto [dev_type, dtype, C] = GetParam();
    skipIfCudaUnavailable(dev_type);
    auto device = makeDevice(dev_type);

    auto ijk = torch::tensor(
        {{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1}, {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}},
        torch::dtype(torch::kInt32));
    auto grid = makeGrid(ijk, device);

    int64_t N = 8;
    nanovdb::Coord kernel_size(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);

    auto topo = ops::gatherScatterDefaultSparseConvTopology(*grid, *grid, kernel_size, stride);

    torch::manual_seed(42);
    auto features = torch::randn({N, C}, opts(device, dtype));
    auto weights  = torch::randn({C, C, 3, 3, 3}, opts(device, dtype));

    auto out = ops::gatherScatterDefaultSparseConv(features, weights, topo);

    EXPECT_EQ(out.dim(), 2);
    EXPECT_EQ(out.size(0), N);
    EXPECT_EQ(out.size(1), C);
    EXPECT_TRUE(out.is_floating_point());
}

TEST_P(GatherScatterDefaultForwardTest, IdentityConv1x1x1) {
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

    auto topo = ops::gatherScatterDefaultSparseConvTopology(*grid, *grid, kernel_size, stride);

    auto features = torch::randn({N, C}, opts(device, dtype));
    auto weights  = torch::eye(C, opts(device, dtype)).reshape({C, C, 1, 1, 1});

    auto out = ops::gatherScatterDefaultSparseConv(features, weights, topo);

    EXPECT_TRUE(torch::allclose(out.cpu(), features.cpu(), rtol, atol))
        << "Identity conv mismatch at C=" << C;
}

TEST_P(GatherScatterDefaultForwardTest, DifferentGrids) {
    auto [dev_type, dtype, C] = GetParam();
    skipIfCudaUnavailable(dev_type);
    auto device = makeDevice(dev_type);

    auto ijk      = torch::tensor({{5, 5, 5}}, torch::dtype(torch::kInt32));
    auto src_grid = makeGrid(ijk, device);

    nanovdb::Coord kernel_size(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);
    auto dst_grid = src_grid->convolutionOutput(kernel_size, stride);

    auto topo =
        ops::gatherScatterDefaultSparseConvTopology(*src_grid, *dst_grid, kernel_size, stride);

    torch::manual_seed(99);
    auto features = torch::randn({1, C}, opts(device, dtype));
    auto weights  = torch::randn({C, C, 3, 3, 3}, opts(device, dtype));

    auto out = ops::gatherScatterDefaultSparseConv(features, weights, topo);

    int64_t expected_output_voxels = dst_grid->totalVoxels();
    EXPECT_EQ(out.size(0), expected_output_voxels);
    EXPECT_EQ(out.size(1), C);
}

// =============================================================================
// Backward tests
// =============================================================================

class GatherScatterDefaultBackwardTest : public ::testing::TestWithParam<ConvParam> {};

TEST_P(GatherScatterDefaultBackwardTest, SameGridStride1) {
    auto [dev_type, dtype, C] = GetParam();
    skipIfCudaUnavailable(dev_type);
    auto device = makeDevice(dev_type);

    auto ijk = torch::tensor(
        {{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1}, {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}},
        torch::dtype(torch::kInt32));
    auto grid = makeGrid(ijk, device);
    int64_t N = 8;

    nanovdb::Coord kernel_size(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);

    auto topo = ops::gatherScatterDefaultSparseConvTopology(*grid, *grid, kernel_size, stride);

    torch::manual_seed(77);
    auto features    = torch::randn({N, C}, opts(device, dtype));
    auto weights     = torch::randn({C, C, 3, 3, 3}, opts(device, dtype));
    auto grad_output = torch::randn({N, C}, opts(device, dtype));

    auto [gf, gw] =
        ops::gatherScatterDefaultSparseConvBackward(grad_output, features, weights, topo);

    EXPECT_EQ(gf.sizes(), features.sizes());
    EXPECT_EQ(gw.sizes(), weights.sizes());
    EXPECT_TRUE(gf.is_floating_point());
    EXPECT_TRUE(gw.is_floating_point());
}

// =============================================================================
// Transposed tests
// =============================================================================

class GatherScatterDefaultTransposeTest : public ::testing::TestWithParam<ConvParam> {};

TEST_P(GatherScatterDefaultTransposeTest, TransposeForwardSameGrid) {
    auto [dev_type, dtype, C] = GetParam();
    skipIfCudaUnavailable(dev_type);
    auto device = makeDevice(dev_type);

    auto ijk = torch::tensor(
        {{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1}, {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}},
        torch::dtype(torch::kInt32));
    auto grid = makeGrid(ijk, device);
    int64_t N = 8;

    nanovdb::Coord kernel_size(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);

    auto topo =
        ops::gatherScatterDefaultSparseConvTransposeTopology(*grid, *grid, kernel_size, stride);

    torch::manual_seed(55);
    auto features = torch::randn({N, C}, opts(device, dtype));
    auto weights  = torch::randn({C, C, 3, 3, 3}, opts(device, dtype));

    auto out = ops::gatherScatterDefaultSparseConvTranspose(features, weights, topo);

    EXPECT_EQ(out.dim(), 2);
    EXPECT_EQ(out.size(0), N);
    EXPECT_EQ(out.size(1), C);
}

TEST_P(GatherScatterDefaultTransposeTest, TransposeBackwardSameGrid) {
    auto [dev_type, dtype, C] = GetParam();
    skipIfCudaUnavailable(dev_type);
    auto device = makeDevice(dev_type);

    auto ijk = torch::tensor(
        {{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1}, {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}},
        torch::dtype(torch::kInt32));
    auto grid = makeGrid(ijk, device);
    int64_t N = 8;

    nanovdb::Coord kernel_size(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);

    auto topo =
        ops::gatherScatterDefaultSparseConvTransposeTopology(*grid, *grid, kernel_size, stride);

    torch::manual_seed(66);
    auto features    = torch::randn({N, C}, opts(device, dtype));
    auto weights     = torch::randn({C, C, 3, 3, 3}, opts(device, dtype));
    auto grad_output = torch::randn({N, C}, opts(device, dtype));

    auto [gf, gw] =
        ops::gatherScatterDefaultSparseConvTransposeBackward(grad_output, features, weights, topo);

    EXPECT_EQ(gf.sizes(), features.sizes());
    EXPECT_EQ(gw.sizes(), weights.sizes());
}

// =============================================================================
// Strided tests
// =============================================================================

class GatherScatterDefaultStridedTest : public ::testing::TestWithParam<ConvParam> {};

TEST_P(GatherScatterDefaultStridedTest, StridedForward) {
    auto [dev_type, dtype, C] = GetParam();
    skipIfCudaUnavailable(dev_type);
    auto device = makeDevice(dev_type);

    auto src_grid = makeStridedTestGrid(device);

    nanovdb::Coord kernel_size(3, 3, 3);
    nanovdb::Coord stride(2, 2, 2);

    auto dst_grid = src_grid->convolutionOutput(kernel_size, stride);

    auto topo =
        ops::gatherScatterDefaultSparseConvTopology(*src_grid, *dst_grid, kernel_size, stride);

    int64_t S = topo.feature_total_voxels;

    torch::manual_seed(88);
    auto features = torch::randn({S, C}, opts(device, dtype));
    auto weights  = torch::randn({C, C, 3, 3, 3}, opts(device, dtype));

    auto out = ops::gatherScatterDefaultSparseConv(features, weights, topo);

    EXPECT_EQ(out.size(0), topo.output_total_voxels);
    EXPECT_EQ(out.size(1), C);
}

TEST_P(GatherScatterDefaultStridedTest, StridedBackward) {
    auto [dev_type, dtype, C] = GetParam();
    skipIfCudaUnavailable(dev_type);
    auto device = makeDevice(dev_type);

    auto src_grid = makeStridedTestGrid(device);

    nanovdb::Coord kernel_size(3, 3, 3);
    nanovdb::Coord stride(2, 2, 2);

    auto dst_grid = src_grid->convolutionOutput(kernel_size, stride);

    auto topo =
        ops::gatherScatterDefaultSparseConvTopology(*src_grid, *dst_grid, kernel_size, stride);

    int64_t S = topo.feature_total_voxels;
    int64_t D = topo.output_total_voxels;

    torch::manual_seed(89);
    auto features    = torch::randn({S, C}, opts(device, dtype));
    auto weights     = torch::randn({C, C, 3, 3, 3}, opts(device, dtype));
    auto grad_output = torch::randn({D, C}, opts(device, dtype));

    auto [gf, gw] =
        ops::gatherScatterDefaultSparseConvBackward(grad_output, features, weights, topo);

    EXPECT_EQ(gf.sizes(), features.sizes());
    EXPECT_EQ(gw.sizes(), weights.sizes());
}

// =============================================================================
// Kernel-size tests -- varied and asymmetric kernel dimensions
// =============================================================================

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

class GatherScatterDefaultKernelSizeTest : public ::testing::TestWithParam<KernelSizeParam> {};

TEST_P(GatherScatterDefaultKernelSizeTest, ForwardAndBackward) {
    auto [dev_type, dtype, k0, k1, k2] = GetParam();
    skipIfCudaUnavailable(dev_type);
    auto device = makeDevice(dev_type);

    int64_t const C = 16;

    auto grid       = makeDenseTestGrid(4, device);
    int64_t const N = 64;

    nanovdb::Coord kernel_size(k0, k1, k2);
    nanovdb::Coord stride(1, 1, 1);

    auto topo = ops::gatherScatterDefaultSparseConvTopology(*grid, *grid, kernel_size, stride);

    torch::manual_seed(42);
    auto features = torch::randn({N, C}, opts(device, dtype));
    auto weights  = torch::randn({C, C, k0, k1, k2}, opts(device, dtype));

    auto out = ops::gatherScatterDefaultSparseConv(features, weights, topo);

    EXPECT_EQ(out.dim(), 2);
    EXPECT_EQ(out.size(0), N);
    EXPECT_EQ(out.size(1), C);

    auto grad_output = torch::randn_like(out);

    auto [gf, gw] =
        ops::gatherScatterDefaultSparseConvBackward(grad_output, features, weights, topo);

    EXPECT_EQ(gf.sizes(), features.sizes());
    EXPECT_EQ(gw.sizes(), weights.sizes());
}

// =============================================================================
// Test instantiation
// =============================================================================

// clang-format off

static std::vector<ConvParam>
channelConfigs() {
    std::vector<ConvParam> v;
    for (auto dev : {torch::kCPU, torch::kCUDA}) {
        for (auto dt : {torch::kFloat32, torch::kFloat64}) {
            for (int64_t C : {4, 16, 32, 64}) {
                v.push_back(cp(dev, dt, C));
            }
        }
        v.push_back(cp(dev, torch::kFloat32, 128));
    }
    return v;
}

static std::vector<KernelSizeParam>
kernelSizeConfigs() {
    std::vector<KernelSizeParam> v;
    for (auto dev : {torch::kCPU, torch::kCUDA}) {
        for (auto [k0, k1, k2] : std::vector<std::tuple<int,int,int>>{
                 {1,1,1}, {3,3,3}, {5,5,5}, {3,5,1}, {1,3,5}, {3,1,3}, {2,2,2}, {4,4,4}}) {
            v.push_back(kp(dev, torch::kFloat32, k0, k1, k2));
        }
        for (auto [k0, k1, k2] : std::vector<std::tuple<int,int,int>>{
                 {3,3,3}, {5,5,5}, {3,5,1}}) {
            v.push_back(kp(dev, torch::kFloat64, k0, k1, k2));
        }
    }
    return v;
}

INSTANTIATE_TEST_SUITE_P(Configs, GatherScatterDefaultForwardTest,
    ::testing::ValuesIn(channelConfigs()), paramName);

INSTANTIATE_TEST_SUITE_P(Configs, GatherScatterDefaultBackwardTest,
    ::testing::ValuesIn(channelConfigs()), paramName);

INSTANTIATE_TEST_SUITE_P(Configs, GatherScatterDefaultTransposeTest,
    ::testing::ValuesIn(channelConfigs()), paramName);

INSTANTIATE_TEST_SUITE_P(Configs, GatherScatterDefaultStridedTest,
    ::testing::ValuesIn(channelConfigs()), paramName);

INSTANTIATE_TEST_SUITE_P(KernelSizes, GatherScatterDefaultKernelSizeTest,
    ::testing::ValuesIn(kernelSizeConfigs()), kernelSizeParamName);

// clang-format on
