// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// GatherScatterDefaultConvTest.cu -- Tests for the default gather-scatter sparse convolution op.
//
// =============================================================================
// TEST PHILOSOPHY
// =============================================================================
//
// The GatherScatterDefault convolution is a correctness-critical operator that
// replaces a long series of faulty prior implementations.  These tests are
// therefore designed to be rigorous, not just smoke-tested.
//
// Tests are organized in three layers:
//
//   STRUCTURAL (topology invariants)
//     Verify that the compacted CSR topology has monotonic offsets, valid
//     index ranges, and correct pair counts.  These do not exercise the
//     GEMM path at all.
//
//   SMOKE (shapes and types)
//     Parameterized across devices, dtypes, channel counts, and kernel
//     sizes.  Verify output tensor shapes, dtypes, and absence of NaN/Inf.
//     These catch crashes and gross misconfigurations but not subtle value
//     errors.
//
//   VALUE (numerical correctness)
//     Verified via two complementary, independent strategies:
//
//     1. Naive CPU triple-loop reference -- uses the already-validated
//        topology structure but computes the gather-GEMM-scatter via
//        scalar loops.  Catches GEMM dispatch bugs, weight layout
//        errors, and scatter-add mistakes.
//
//     2. Adjoint identity -- for a linear operator L and its claimed
//        adjoint L*, the identity <gy, L(x)> == <L*(gy), x> must hold
//        for all x, gy.  This is checked for both the feature gradient
//        and the weight gradient, catching backward-pass bugs without
//        needing a reference backward implementation.
//
//   CROSS-CHECKS
//     conv_transpose(x, W) == conv(x, flip(W)) at stride 1 for odd
//     kernels.  This is a fundamental mathematical identity that ties
//     the forward and transposed code paths together.
//
//   ERROR PATHS
//     TORCH_CHECK guards are exercised with intentionally invalid inputs
//     to confirm they reject rather than silently produce garbage.
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
dtypeStr(torch::ScalarType dtype) {
    if (dtype == torch::kFloat64)
        return "f64";
    if (dtype == torch::kFloat32)
        return "f32";
    if (dtype == torch::kFloat16)
        return "f16";
    if (dtype == torch::kBFloat16)
        return "bf16";
    return "unknown";
}

static std::string
paramName(::testing::TestParamInfo<ConvParam> const &info) {
    auto dev   = std::get<0>(info.param);
    auto dtype = std::get<1>(info.param);
    auto C     = std::get<2>(info.param);

    std::string dev_str = (dev == torch::kCPU) ? "CPU" : "CUDA";
    return dev_str + "_" + dtypeStr(dtype) + "_C" + std::to_string(C);
}

// =============================================================================
// Helpers
// =============================================================================

static std::pair<double, double>
tolerances(torch::ScalarType dtype) {
    if (dtype == torch::kFloat64)
        return {1e-8, 1e-8};
    if (dtype == torch::kFloat32)
        return {1e-3, 1e-3};
    // float16 and bfloat16 have ~3 decimal digits of precision
    return {1e-1, 1e-1};
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
// Verification helpers
// =============================================================================

static void
assertNoNanInf(torch::Tensor t, char const *label) {
    auto t_cpu = t.cpu().to(torch::kFloat64);
    EXPECT_FALSE(t_cpu.isnan().any().item<bool>()) << label << " contains NaN";
    EXPECT_FALSE(t_cpu.isinf().any().item<bool>()) << label << " contains Inf";
}

// =============================================================================
// Naive CPU reference implementation
// =============================================================================
//
// Uses the topology structure (independently validated by SanityChecks) but
// computes the gather-GEMM-scatter via scalar loops on CPU float64.  This is
// intentionally simple and unoptimized to serve as a ground-truth oracle.

static torch::Tensor
naiveConvForward(torch::Tensor features,
                 torch::Tensor weights,
                 ops::GatherScatterDefaultTopology const &topo) {
    int64_t const O     = topo.outputTotalVoxels;
    int64_t const K     = topo.kernelVolume;
    int64_t const C_in  = weights.size(1);
    int64_t const C_out = weights.size(0);

    auto feat = features.cpu().to(torch::kFloat64);
    auto W    = weights.cpu()
                 .to(torch::kFloat64)
                 .permute({2, 3, 4, 1, 0})
                 .reshape({K, C_in, C_out})
                 .contiguous();

    auto output = torch::zeros({O, C_out}, torch::kFloat64);

    auto off_acc = topo.offsets.accessor<int64_t, 1>();
    auto gi      = topo.gatherIndices.cpu();
    auto si      = topo.scatterIndices.cpu();
    auto gi_acc  = gi.accessor<int32_t, 1>();
    auto si_acc  = si.accessor<int32_t, 1>();
    auto feat_a  = feat.accessor<double, 2>();
    auto W_a     = W.accessor<double, 3>();
    auto out_a   = output.accessor<double, 2>();

    for (int64_t k = 0; k < K; ++k) {
        for (int64_t p = off_acc[k]; p < off_acc[k + 1]; ++p) {
            int32_t fi = gi_acc[p];
            int32_t oi = si_acc[p];
            for (int64_t co = 0; co < C_out; ++co) {
                for (int64_t ci = 0; ci < C_in; ++ci) {
                    out_a[oi][co] += feat_a[fi][ci] * W_a[k][ci][co];
                }
            }
        }
    }
    return output;
}

static std::tuple<torch::Tensor, torch::Tensor>
naiveConvBackward(torch::Tensor grad_output,
                  torch::Tensor features,
                  torch::Tensor weights,
                  ops::GatherScatterDefaultTopology const &topo) {
    int64_t const F     = topo.featureTotalVoxels;
    int64_t const K     = topo.kernelVolume;
    int64_t const C_in  = weights.size(1);
    int64_t const C_out = weights.size(0);

    auto go   = grad_output.cpu().to(torch::kFloat64);
    auto feat = features.cpu().to(torch::kFloat64);
    auto W    = weights.cpu()
                 .to(torch::kFloat64)
                 .permute({2, 3, 4, 1, 0})
                 .reshape({K, C_in, C_out})
                 .contiguous();

    auto grad_feat   = torch::zeros({F, C_in}, torch::kFloat64);
    auto grad_W_flat = torch::zeros({K, C_in, C_out}, torch::kFloat64);

    auto off_acc = topo.offsets.accessor<int64_t, 1>();
    auto gi      = topo.gatherIndices.cpu();
    auto si      = topo.scatterIndices.cpu();
    auto gi_acc  = gi.accessor<int32_t, 1>();
    auto si_acc  = si.accessor<int32_t, 1>();
    auto go_a    = go.accessor<double, 2>();
    auto feat_a  = feat.accessor<double, 2>();
    auto W_a     = W.accessor<double, 3>();
    auto gf_a    = grad_feat.accessor<double, 2>();
    auto gW_a    = grad_W_flat.accessor<double, 3>();

    for (int64_t k = 0; k < K; ++k) {
        for (int64_t p = off_acc[k]; p < off_acc[k + 1]; ++p) {
            int32_t fi = gi_acc[p];
            int32_t oi = si_acc[p];
            for (int64_t ci = 0; ci < C_in; ++ci) {
                for (int64_t co = 0; co < C_out; ++co) {
                    gf_a[fi][ci] += go_a[oi][co] * W_a[k][ci][co];
                    gW_a[k][ci][co] += feat_a[fi][ci] * go_a[oi][co];
                }
            }
        }
    }

    auto ks           = topo.kernelSize;
    auto grad_weights = grad_W_flat.reshape({ks[0], ks[1], ks[2], C_in, C_out})
                            .permute({4, 3, 0, 1, 2})
                            .contiguous();
    return {grad_feat, grad_weights};
}

// =============================================================================
// Forward smoke tests
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
    assertNoNanInf(out, "SameGridStride1 forward output");
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
    assertNoNanInf(out, "IdentityConv1x1x1 output");

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
    assertNoNanInf(out, "DifferentGrids forward output");

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
    assertNoNanInf(gf, "SameGridStride1 backward grad_features");
    assertNoNanInf(gw, "SameGridStride1 backward grad_weights");
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
    assertNoNanInf(out, "TransposeForwardSameGrid output");
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
    assertNoNanInf(gf, "TransposeBackwardSameGrid grad_features");
    assertNoNanInf(gw, "TransposeBackwardSameGrid grad_weights");
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

    int64_t S = topo.featureTotalVoxels;

    torch::manual_seed(88);
    auto features = torch::randn({S, C}, opts(device, dtype));
    auto weights  = torch::randn({C, C, 3, 3, 3}, opts(device, dtype));

    auto out = ops::gatherScatterDefaultSparseConv(features, weights, topo);

    EXPECT_EQ(out.size(0), topo.outputTotalVoxels);
    EXPECT_EQ(out.size(1), C);
    assertNoNanInf(out, "StridedForward output");
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

    int64_t S = topo.featureTotalVoxels;
    int64_t D = topo.outputTotalVoxels;

    torch::manual_seed(89);
    auto features    = torch::randn({S, C}, opts(device, dtype));
    auto weights     = torch::randn({C, C, 3, 3, 3}, opts(device, dtype));
    auto grad_output = torch::randn({D, C}, opts(device, dtype));

    auto [gf, gw] =
        ops::gatherScatterDefaultSparseConvBackward(grad_output, features, weights, topo);

    EXPECT_EQ(gf.sizes(), features.sizes());
    EXPECT_EQ(gw.sizes(), weights.sizes());
    assertNoNanInf(gf, "StridedBackward grad_features");
    assertNoNanInf(gw, "StridedBackward grad_weights");
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

    std::string dev_str = (dev == torch::kCPU) ? "CPU" : "CUDA";
    return dev_str + "_" + dtypeStr(dt) + "_k" + std::to_string(k0) + "x" + std::to_string(k1) +
           "x" + std::to_string(k2);
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
    assertNoNanInf(out, "KernelSize forward output");

    auto grad_output = torch::randn_like(out);

    auto [gf, gw] =
        ops::gatherScatterDefaultSparseConvBackward(grad_output, features, weights, topo);

    EXPECT_EQ(gf.sizes(), features.sizes());
    EXPECT_EQ(gw.sizes(), weights.sizes());
    assertNoNanInf(gf, "KernelSize backward grad_features");
    assertNoNanInf(gw, "KernelSize backward grad_weights");
}

// =============================================================================
// Helper: multi-batch grid (2 dense cubes as separate batch entries)
// =============================================================================

static c10::intrusive_ptr<GridBatchImpl>
makeMultiBatchGrid(int dim0, int dim1, torch::Device device) {
    auto makeDenseIjk = [](int dim) {
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
        return torch::from_blob(flat.data(), {N, 3}, torch::kInt32).clone();
    };

    auto ijk0 = makeDenseIjk(dim0).to(device);
    auto ijk1 = makeDenseIjk(dim1).to(device);
    JaggedTensor jt({ijk0, ijk1});
    std::vector<nanovdb::Vec3d> voxel_sizes = {{1.0, 1.0, 1.0}, {1.0, 1.0, 1.0}};
    std::vector<nanovdb::Vec3d> origins     = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
    return GridBatchImpl::createFromIjk(jt, voxel_sizes, origins);
}

// =============================================================================
// Non-parameterized targeted tests
// =============================================================================

TEST(GatherScatterDefaultTopology, SanityChecks) {
    for (auto dev_type: {torch::kCPU, torch::kCUDA}) {
        skipIfCudaUnavailable(dev_type);
        auto device = makeDevice(dev_type);

        auto grid = makeDenseTestGrid(16, device); // 4096 voxels, 8 NanoVDB leaves

        nanovdb::Coord kernel_size(3, 3, 3);
        nanovdb::Coord stride(1, 1, 1);
        int64_t const K = 27;

        for (bool transposed: {false, true}) {
            auto topo = transposed ? ops::gatherScatterDefaultSparseConvTransposeTopology(
                                         *grid, *grid, kernel_size, stride)
                                   : ops::gatherScatterDefaultSparseConvTopology(
                                         *grid, *grid, kernel_size, stride);

            EXPECT_GT(topo.totalPairs, 0);
            EXPECT_EQ(topo.kernelVolume, K);
            EXPECT_EQ(topo.featureTotalVoxels, 4096);
            EXPECT_EQ(topo.outputTotalVoxels, 4096);

            auto off = topo.offsets.accessor<int64_t, 1>();
            EXPECT_EQ(topo.offsets.size(0), K + 1);
            EXPECT_EQ(off[0], 0);
            EXPECT_EQ(off[K], topo.totalPairs);
            for (int64_t k = 0; k < K; ++k) {
                EXPECT_LE(off[k], off[k + 1]) << "offsets not monotonic at k=" << k;
            }

            EXPECT_EQ(topo.gatherIndices.size(0), topo.totalPairs);
            EXPECT_EQ(topo.scatterIndices.size(0), topo.totalPairs);

            auto gi = topo.gatherIndices.cpu();
            auto si = topo.scatterIndices.cpu();
            EXPECT_GE(gi.min().item<int32_t>(), 0);
            EXPECT_LT(gi.max().item<int32_t>(), topo.featureTotalVoxels);
            EXPECT_GE(si.min().item<int32_t>(), 0);
            EXPECT_LT(si.max().item<int32_t>(), topo.outputTotalVoxels);
        }
    }
}

TEST(GatherScatterDefaultBackward, IdentityGradients) {
    for (auto dev_type: {torch::kCPU, torch::kCUDA}) {
        skipIfCudaUnavailable(dev_type);
        auto device = makeDevice(dev_type);

        int64_t const C = 8;
        auto grid       = makeDenseTestGrid(10, device); // 1000 voxels
        int64_t const N = grid->totalVoxels();

        nanovdb::Coord kernel_size(1, 1, 1);
        nanovdb::Coord stride(1, 1, 1);

        auto topo = ops::gatherScatterDefaultSparseConvTopology(*grid, *grid, kernel_size, stride);

        torch::manual_seed(123);
        auto features    = torch::randn({N, C}, opts(device, torch::kFloat64));
        auto weights     = torch::eye(C, opts(device, torch::kFloat64)).reshape({C, C, 1, 1, 1});
        auto grad_output = torch::randn({N, C}, opts(device, torch::kFloat64));

        auto [gf, gw] =
            ops::gatherScatterDefaultSparseConvBackward(grad_output, features, weights, topo);

        EXPECT_TRUE(torch::allclose(gf.cpu(), grad_output.cpu(), 1e-8, 1e-8))
            << "grad_features should equal grad_output for identity kernel";

        auto expected_gw = (grad_output.t().mm(features)).reshape({C, C, 1, 1, 1});
        EXPECT_TRUE(torch::allclose(gw.cpu(), expected_gw.cpu(), 1e-6, 1e-6))
            << "grad_weights mismatch for identity kernel";
    }
}

TEST(GatherScatterDefaultForward, AsymmetricChannels) {
    for (auto dev_type: {torch::kCPU, torch::kCUDA}) {
        skipIfCudaUnavailable(dev_type);
        auto device = makeDevice(dev_type);

        auto grid       = makeDenseTestGrid(10, device); // 1000 voxels
        int64_t const N = grid->totalVoxels();

        nanovdb::Coord kernel_size(3, 3, 3);
        nanovdb::Coord stride(1, 1, 1);

        auto topo = ops::gatherScatterDefaultSparseConvTopology(*grid, *grid, kernel_size, stride);

        for (auto [C_in, C_out]: std::vector<std::pair<int64_t, int64_t>>{{8, 16}, {16, 8}}) {
            torch::manual_seed(42);
            auto features    = torch::randn({N, C_in}, opts(device, torch::kFloat32));
            auto weights     = torch::randn({C_out, C_in, 3, 3, 3}, opts(device, torch::kFloat32));
            auto grad_output = torch::randn({N, C_out}, opts(device, torch::kFloat32));

            auto out = ops::gatherScatterDefaultSparseConv(features, weights, topo);
            EXPECT_EQ(out.size(0), N);
            EXPECT_EQ(out.size(1), C_out);

            auto [gf, gw] =
                ops::gatherScatterDefaultSparseConvBackward(grad_output, features, weights, topo);
            EXPECT_EQ(gf.size(0), N);
            EXPECT_EQ(gf.size(1), C_in);
            EXPECT_EQ(gw.size(0), C_out);
            EXPECT_EQ(gw.size(1), C_in);
        }
    }
}

TEST(GatherScatterDefaultForward, MultiBatchGrid) {
    for (auto dev_type: {torch::kCPU, torch::kCUDA}) {
        skipIfCudaUnavailable(dev_type);
        auto device = makeDevice(dev_type);

        // Batch 0: 8x8x8 = 512 voxels, Batch 1: 10x10x10 = 1000 voxels
        auto grid       = makeMultiBatchGrid(8, 10, device);
        int64_t const N = grid->totalVoxels();
        EXPECT_EQ(N, 1512);

        int64_t const C = 4;
        nanovdb::Coord kernel_size(3, 3, 3);
        nanovdb::Coord stride(1, 1, 1);

        // Forward topology
        auto topo = ops::gatherScatterDefaultSparseConvTopology(*grid, *grid, kernel_size, stride);
        EXPECT_EQ(topo.featureTotalVoxels, N);
        EXPECT_EQ(topo.outputTotalVoxels, N);
        EXPECT_GT(topo.totalPairs, 0);

        torch::manual_seed(77);
        auto features = torch::randn({N, C}, opts(device, torch::kFloat32));
        auto weights  = torch::randn({C, C, 3, 3, 3}, opts(device, torch::kFloat32));

        auto out = ops::gatherScatterDefaultSparseConv(features, weights, topo);
        EXPECT_EQ(out.size(0), N);
        EXPECT_EQ(out.size(1), C);
        assertNoNanInf(out, "MultiBatch forward output");

        auto grad_output = torch::randn({N, C}, opts(device, torch::kFloat32));
        auto [gf, gw] =
            ops::gatherScatterDefaultSparseConvBackward(grad_output, features, weights, topo);
        EXPECT_EQ(gf.sizes(), features.sizes());
        EXPECT_EQ(gw.sizes(), weights.sizes());
        assertNoNanInf(gf, "MultiBatch backward grad_features");
        assertNoNanInf(gw, "MultiBatch backward grad_weights");

        // Transposed topology
        auto topo_t =
            ops::gatherScatterDefaultSparseConvTransposeTopology(*grid, *grid, kernel_size, stride);

        auto out_t = ops::gatherScatterDefaultSparseConvTranspose(features, weights, topo_t);
        EXPECT_EQ(out_t.size(0), N);
        EXPECT_EQ(out_t.size(1), C);
        assertNoNanInf(out_t, "MultiBatch transpose forward output");

        auto [gf_t, gw_t] = ops::gatherScatterDefaultSparseConvTransposeBackward(
            grad_output, features, weights, topo_t);
        EXPECT_EQ(gf_t.sizes(), features.sizes());
        EXPECT_EQ(gw_t.sizes(), weights.sizes());
        assertNoNanInf(gf_t, "MultiBatch transpose backward grad_features");
        assertNoNanInf(gw_t, "MultiBatch transpose backward grad_weights");
    }
}

TEST(GatherScatterDefaultTopology, EmptyGrid) {
    for (auto dev_type: {torch::kCPU, torch::kCUDA}) {
        skipIfCudaUnavailable(dev_type);
        auto device = makeDevice(dev_type);

        auto grid = GridBatchImpl::createFromEmpty(device, {1.0, 1.0, 1.0}, {0.0, 0.0, 0.0});

        nanovdb::Coord kernel_size(3, 3, 3);
        nanovdb::Coord stride(1, 1, 1);

        auto topo = ops::gatherScatterDefaultSparseConvTopology(*grid, *grid, kernel_size, stride);
        EXPECT_EQ(topo.totalPairs, 0);
        EXPECT_EQ(topo.gatherIndices.size(0), 0);
        EXPECT_EQ(topo.scatterIndices.size(0), 0);
        EXPECT_EQ(topo.featureTotalVoxels, 0);
        EXPECT_EQ(topo.outputTotalVoxels, 0);

        int64_t const C = 4;
        auto features   = torch::zeros({0, C}, opts(device, torch::kFloat32));
        auto weights    = torch::randn({C, C, 3, 3, 3}, opts(device, torch::kFloat32));

        auto out = ops::gatherScatterDefaultSparseConv(features, weights, topo);
        EXPECT_EQ(out.size(0), 0);
        EXPECT_EQ(out.size(1), C);
    }
}

// =============================================================================
// Forward value correctness tests (naive reference comparison)
// =============================================================================

TEST(GatherScatterDefaultValue, SmallDenseGrid) {
    for (auto dev_type: {torch::kCPU, torch::kCUDA}) {
        skipIfCudaUnavailable(dev_type);
        auto device = makeDevice(dev_type);

        auto grid       = makeDenseTestGrid(4, device);
        int64_t const N = 64;

        nanovdb::Coord kernel_size(3, 3, 3);
        nanovdb::Coord stride(1, 1, 1);
        auto topo = ops::gatherScatterDefaultSparseConvTopology(*grid, *grid, kernel_size, stride);

        int64_t const C_in  = 4;
        int64_t const C_out = 8;

        torch::manual_seed(42);
        auto features = torch::randn({N, C_in}, opts(device, torch::kFloat64));
        auto weights  = torch::randn({C_out, C_in, 3, 3, 3}, opts(device, torch::kFloat64));

        auto out = ops::gatherScatterDefaultSparseConv(features, weights, topo);
        auto ref = naiveConvForward(features, weights, topo);

        assertNoNanInf(out, "Value SmallDenseGrid forward");
        EXPECT_TRUE(torch::allclose(out.cpu().to(torch::kFloat64), ref, 1e-8, 1e-8))
            << "Forward value mismatch on " << (dev_type == torch::kCPU ? "CPU" : "CUDA");
    }
}

TEST(GatherScatterDefaultValue, AsymmetricKernel) {
    for (auto dev_type: {torch::kCPU, torch::kCUDA}) {
        skipIfCudaUnavailable(dev_type);
        auto device = makeDevice(dev_type);

        auto grid       = makeDenseTestGrid(4, device);
        int64_t const N = 64;

        nanovdb::Coord kernel_size(3, 5, 1);
        nanovdb::Coord stride(1, 1, 1);
        auto topo = ops::gatherScatterDefaultSparseConvTopology(*grid, *grid, kernel_size, stride);

        int64_t const C_in  = 4;
        int64_t const C_out = 8;

        torch::manual_seed(43);
        auto features = torch::randn({N, C_in}, opts(device, torch::kFloat64));
        auto weights  = torch::randn({C_out, C_in, 3, 5, 1}, opts(device, torch::kFloat64));

        auto out = ops::gatherScatterDefaultSparseConv(features, weights, topo);
        auto ref = naiveConvForward(features, weights, topo);

        assertNoNanInf(out, "Value AsymmetricKernel forward");
        EXPECT_TRUE(torch::allclose(out.cpu().to(torch::kFloat64), ref, 1e-8, 1e-8))
            << "Asymmetric kernel value mismatch on " << (dev_type == torch::kCPU ? "CPU" : "CUDA");
    }
}

TEST(GatherScatterDefaultValue, StridedConv) {
    for (auto dev_type: {torch::kCPU, torch::kCUDA}) {
        skipIfCudaUnavailable(dev_type);
        auto device = makeDevice(dev_type);

        auto src_grid = makeStridedTestGrid(device);
        nanovdb::Coord kernel_size(3, 3, 3);
        nanovdb::Coord stride(2, 2, 2);
        auto dst_grid = src_grid->convolutionOutput(kernel_size, stride);

        auto topo =
            ops::gatherScatterDefaultSparseConvTopology(*src_grid, *dst_grid, kernel_size, stride);

        int64_t const S     = topo.featureTotalVoxels;
        int64_t const C_in  = 4;
        int64_t const C_out = 8;

        torch::manual_seed(44);
        auto features = torch::randn({S, C_in}, opts(device, torch::kFloat64));
        auto weights  = torch::randn({C_out, C_in, 3, 3, 3}, opts(device, torch::kFloat64));

        auto out = ops::gatherScatterDefaultSparseConv(features, weights, topo);
        auto ref = naiveConvForward(features, weights, topo);

        assertNoNanInf(out, "Value StridedConv forward");
        EXPECT_TRUE(torch::allclose(out.cpu().to(torch::kFloat64), ref, 1e-8, 1e-8))
            << "Strided conv value mismatch on " << (dev_type == torch::kCPU ? "CPU" : "CUDA");
    }
}

TEST(GatherScatterDefaultValue, SingleVoxel) {
    for (auto dev_type: {torch::kCPU, torch::kCUDA}) {
        skipIfCudaUnavailable(dev_type);
        auto device = makeDevice(dev_type);

        auto ijk      = torch::tensor({{5, 5, 5}}, torch::dtype(torch::kInt32));
        auto src_grid = makeGrid(ijk, device);
        nanovdb::Coord kernel_size(3, 3, 3);
        nanovdb::Coord stride(1, 1, 1);
        auto dst_grid = src_grid->convolutionOutput(kernel_size, stride);

        auto topo =
            ops::gatherScatterDefaultSparseConvTopology(*src_grid, *dst_grid, kernel_size, stride);

        int64_t const C_in  = 4;
        int64_t const C_out = 8;

        torch::manual_seed(45);
        auto features = torch::randn({1, C_in}, opts(device, torch::kFloat64));
        auto weights  = torch::randn({C_out, C_in, 3, 3, 3}, opts(device, torch::kFloat64));

        auto out = ops::gatherScatterDefaultSparseConv(features, weights, topo);
        auto ref = naiveConvForward(features, weights, topo);

        assertNoNanInf(out, "Value SingleVoxel forward");
        EXPECT_TRUE(torch::allclose(out.cpu().to(torch::kFloat64), ref, 1e-8, 1e-8))
            << "Single voxel value mismatch on " << (dev_type == torch::kCPU ? "CPU" : "CUDA");
    }
}

// =============================================================================
// Backward adjoint identity tests
// =============================================================================
//
// For a linear operator L with claimed adjoint L*:
//   <gy, L(x)> == <L*(gy), x>       (feature adjoint)
//   <gy, L(x)> == <gW, W>           (weight adjoint, treating L as linear in W)
//
// Both equalities must hold to machine precision in float64.

static void
checkAdjointIdentity(torch::Tensor x,
                     torch::Tensor W,
                     torch::Tensor gy,
                     torch::Tensor y,
                     torch::Tensor gx,
                     torch::Tensor gW,
                     char const *label) {
    auto x_d  = x.cpu().to(torch::kFloat64).flatten();
    auto W_d  = W.cpu().to(torch::kFloat64).flatten();
    auto gy_d = gy.cpu().to(torch::kFloat64).flatten();
    auto y_d  = y.cpu().to(torch::kFloat64).flatten();
    auto gx_d = gx.cpu().to(torch::kFloat64).flatten();
    auto gW_d = gW.cpu().to(torch::kFloat64).flatten();

    double lhs      = torch::dot(gy_d, y_d).item<double>();
    double rhs_feat = torch::dot(gx_d, x_d).item<double>();
    double rhs_wt   = torch::dot(gW_d, W_d).item<double>();

    double scale_feat = std::max(std::abs(lhs), std::abs(rhs_feat));
    double scale_wt   = std::max(std::abs(lhs), std::abs(rhs_wt));
    double tol        = 1e-8;

    EXPECT_NEAR(lhs, rhs_feat, tol * scale_feat + tol)
        << label << ": feature adjoint identity violated";
    EXPECT_NEAR(lhs, rhs_wt, tol * scale_wt + tol) << label << ": weight adjoint identity violated";
}

TEST(GatherScatterDefaultAdjoint, ForwardAdjoint) {
    for (auto dev_type: {torch::kCPU, torch::kCUDA}) {
        skipIfCudaUnavailable(dev_type);
        auto device = makeDevice(dev_type);

        auto grid           = makeDenseTestGrid(4, device);
        int64_t const N     = 64;
        int64_t const C_in  = 4;
        int64_t const C_out = 8;

        nanovdb::Coord kernel_size(3, 3, 3);
        nanovdb::Coord stride(1, 1, 1);
        auto topo = ops::gatherScatterDefaultSparseConvTopology(*grid, *grid, kernel_size, stride);

        torch::manual_seed(50);
        auto x  = torch::randn({N, C_in}, opts(device, torch::kFloat64));
        auto W  = torch::randn({C_out, C_in, 3, 3, 3}, opts(device, torch::kFloat64));
        auto gy = torch::randn({N, C_out}, opts(device, torch::kFloat64));

        auto y        = ops::gatherScatterDefaultSparseConv(x, W, topo);
        auto [gx, gW] = ops::gatherScatterDefaultSparseConvBackward(gy, x, W, topo);

        checkAdjointIdentity(x, W, gy, y, gx, gW, "ForwardAdjoint");
    }
}

TEST(GatherScatterDefaultAdjoint, ForwardAdjointStrided) {
    for (auto dev_type: {torch::kCPU, torch::kCUDA}) {
        skipIfCudaUnavailable(dev_type);
        auto device = makeDevice(dev_type);

        auto src_grid = makeStridedTestGrid(device);
        nanovdb::Coord kernel_size(3, 3, 3);
        nanovdb::Coord stride(2, 2, 2);
        auto dst_grid = src_grid->convolutionOutput(kernel_size, stride);
        auto topo =
            ops::gatherScatterDefaultSparseConvTopology(*src_grid, *dst_grid, kernel_size, stride);

        int64_t const S     = topo.featureTotalVoxels;
        int64_t const D     = topo.outputTotalVoxels;
        int64_t const C_in  = 4;
        int64_t const C_out = 8;

        torch::manual_seed(51);
        auto x  = torch::randn({S, C_in}, opts(device, torch::kFloat64));
        auto W  = torch::randn({C_out, C_in, 3, 3, 3}, opts(device, torch::kFloat64));
        auto gy = torch::randn({D, C_out}, opts(device, torch::kFloat64));

        auto y        = ops::gatherScatterDefaultSparseConv(x, W, topo);
        auto [gx, gW] = ops::gatherScatterDefaultSparseConvBackward(gy, x, W, topo);

        checkAdjointIdentity(x, W, gy, y, gx, gW, "ForwardAdjointStrided");
    }
}

TEST(GatherScatterDefaultAdjoint, TransposeAdjoint) {
    for (auto dev_type: {torch::kCPU, torch::kCUDA}) {
        skipIfCudaUnavailable(dev_type);
        auto device = makeDevice(dev_type);

        auto grid           = makeDenseTestGrid(4, device);
        int64_t const N     = 64;
        int64_t const C_in  = 4;
        int64_t const C_out = 8;

        nanovdb::Coord kernel_size(3, 3, 3);
        nanovdb::Coord stride(1, 1, 1);
        auto topo =
            ops::gatherScatterDefaultSparseConvTransposeTopology(*grid, *grid, kernel_size, stride);

        torch::manual_seed(52);
        auto x  = torch::randn({N, C_in}, opts(device, torch::kFloat64));
        auto W  = torch::randn({C_out, C_in, 3, 3, 3}, opts(device, torch::kFloat64));
        auto gy = torch::randn({N, C_out}, opts(device, torch::kFloat64));

        auto y        = ops::gatherScatterDefaultSparseConvTranspose(x, W, topo);
        auto [gx, gW] = ops::gatherScatterDefaultSparseConvTransposeBackward(gy, x, W, topo);

        checkAdjointIdentity(x, W, gy, y, gx, gW, "TransposeAdjoint");
    }
}

TEST(GatherScatterDefaultAdjoint, StridedTransposeAdjoint) {
    for (auto dev_type: {torch::kCPU, torch::kCUDA}) {
        skipIfCudaUnavailable(dev_type);
        auto device = makeDevice(dev_type);

        auto src_grid = makeStridedTestGrid(device);
        nanovdb::Coord kernel_size(3, 3, 3);
        nanovdb::Coord stride(2, 2, 2);
        auto fwd_dst = src_grid->convolutionOutput(kernel_size, stride);

        auto topo = ops::gatherScatterDefaultSparseConvTransposeTopology(
            *fwd_dst, *src_grid, kernel_size, stride);

        int64_t const S     = topo.featureTotalVoxels;
        int64_t const D     = topo.outputTotalVoxels;
        int64_t const C_in  = 4;
        int64_t const C_out = 8;

        torch::manual_seed(53);
        auto x  = torch::randn({S, C_in}, opts(device, torch::kFloat64));
        auto W  = torch::randn({C_out, C_in, 3, 3, 3}, opts(device, torch::kFloat64));
        auto gy = torch::randn({D, C_out}, opts(device, torch::kFloat64));

        auto y        = ops::gatherScatterDefaultSparseConvTranspose(x, W, topo);
        auto [gx, gW] = ops::gatherScatterDefaultSparseConvTransposeBackward(gy, x, W, topo);

        checkAdjointIdentity(x, W, gy, y, gx, gW, "StridedTransposeAdjoint");
    }
}

// =============================================================================
// Backward value correctness tests (naive reference comparison)
// =============================================================================

TEST(GatherScatterDefaultBackwardValue, SmallDenseGrid) {
    for (auto dev_type: {torch::kCPU, torch::kCUDA}) {
        skipIfCudaUnavailable(dev_type);
        auto device = makeDevice(dev_type);

        auto grid           = makeDenseTestGrid(4, device);
        int64_t const N     = 64;
        int64_t const C_in  = 4;
        int64_t const C_out = 8;

        nanovdb::Coord kernel_size(3, 3, 3);
        nanovdb::Coord stride(1, 1, 1);
        auto topo = ops::gatherScatterDefaultSparseConvTopology(*grid, *grid, kernel_size, stride);

        torch::manual_seed(60);
        auto features    = torch::randn({N, C_in}, opts(device, torch::kFloat64));
        auto weights     = torch::randn({C_out, C_in, 3, 3, 3}, opts(device, torch::kFloat64));
        auto grad_output = torch::randn({N, C_out}, opts(device, torch::kFloat64));

        auto [gf, gw] =
            ops::gatherScatterDefaultSparseConvBackward(grad_output, features, weights, topo);
        auto [ref_gf, ref_gw] = naiveConvBackward(grad_output, features, weights, topo);

        assertNoNanInf(gf, "BackwardValue SmallDenseGrid grad_features");
        assertNoNanInf(gw, "BackwardValue SmallDenseGrid grad_weights");

        EXPECT_TRUE(torch::allclose(gf.cpu().to(torch::kFloat64), ref_gf, 1e-8, 1e-8))
            << "grad_features mismatch on " << (dev_type == torch::kCPU ? "CPU" : "CUDA");
        EXPECT_TRUE(torch::allclose(gw.cpu().to(torch::kFloat64), ref_gw, 1e-6, 1e-6))
            << "grad_weights mismatch on " << (dev_type == torch::kCPU ? "CPU" : "CUDA");
    }
}

TEST(GatherScatterDefaultBackwardValue, AsymmetricChannels) {
    for (auto dev_type: {torch::kCPU, torch::kCUDA}) {
        skipIfCudaUnavailable(dev_type);
        auto device = makeDevice(dev_type);

        auto grid           = makeDenseTestGrid(4, device);
        int64_t const N     = 64;
        int64_t const C_in  = 4;
        int64_t const C_out = 16;

        nanovdb::Coord kernel_size(3, 3, 3);
        nanovdb::Coord stride(1, 1, 1);
        auto topo = ops::gatherScatterDefaultSparseConvTopology(*grid, *grid, kernel_size, stride);

        torch::manual_seed(61);
        auto features    = torch::randn({N, C_in}, opts(device, torch::kFloat64));
        auto weights     = torch::randn({C_out, C_in, 3, 3, 3}, opts(device, torch::kFloat64));
        auto grad_output = torch::randn({N, C_out}, opts(device, torch::kFloat64));

        auto [gf, gw] =
            ops::gatherScatterDefaultSparseConvBackward(grad_output, features, weights, topo);
        auto [ref_gf, ref_gw] = naiveConvBackward(grad_output, features, weights, topo);

        assertNoNanInf(gf, "BackwardValue AsymmetricChannels grad_features");
        assertNoNanInf(gw, "BackwardValue AsymmetricChannels grad_weights");

        EXPECT_TRUE(torch::allclose(gf.cpu().to(torch::kFloat64), ref_gf, 1e-8, 1e-8))
            << "grad_features mismatch on " << (dev_type == torch::kCPU ? "CPU" : "CUDA");
        EXPECT_TRUE(torch::allclose(gw.cpu().to(torch::kFloat64), ref_gw, 1e-6, 1e-6))
            << "grad_weights mismatch on " << (dev_type == torch::kCPU ? "CPU" : "CUDA");
    }
}

// =============================================================================
// Transpose flip cross-check: conv_transpose(x, W) == conv(x, flip(W))
// =============================================================================

TEST(GatherScatterDefaultTransposeFlip, FlipIdentity3x3x3) {
    for (auto dev_type: {torch::kCPU, torch::kCUDA}) {
        skipIfCudaUnavailable(dev_type);
        auto device = makeDevice(dev_type);

        auto grid       = makeDenseTestGrid(4, device);
        int64_t const N = 64;
        int64_t const C = 8;

        nanovdb::Coord kernel_size(3, 3, 3);
        nanovdb::Coord stride(1, 1, 1);

        auto topo_f =
            ops::gatherScatterDefaultSparseConvTopology(*grid, *grid, kernel_size, stride);
        auto topo_t =
            ops::gatherScatterDefaultSparseConvTransposeTopology(*grid, *grid, kernel_size, stride);

        torch::manual_seed(70);
        auto features = torch::randn({N, C}, opts(device, torch::kFloat64));
        auto weights  = torch::randn({C, C, 3, 3, 3}, opts(device, torch::kFloat64));
        auto W_flip   = torch::flip(weights, {2, 3, 4});

        auto out_t = ops::gatherScatterDefaultSparseConvTranspose(features, weights, topo_t);
        auto out_f = ops::gatherScatterDefaultSparseConv(features, W_flip, topo_f);

        assertNoNanInf(out_t, "FlipIdentity3x3x3 transpose output");
        assertNoNanInf(out_f, "FlipIdentity3x3x3 forward output");
        EXPECT_TRUE(torch::allclose(out_t.cpu(), out_f.cpu(), 1e-8, 1e-8))
            << "conv_transpose(x,W) != conv(x,flip(W)) on "
            << (dev_type == torch::kCPU ? "CPU" : "CUDA");
    }
}

TEST(GatherScatterDefaultTransposeFlip, FlipIdentity3x5x7) {
    for (auto dev_type: {torch::kCPU, torch::kCUDA}) {
        skipIfCudaUnavailable(dev_type);
        auto device = makeDevice(dev_type);

        auto grid       = makeDenseTestGrid(4, device);
        int64_t const N = 64;
        int64_t const C = 4;

        nanovdb::Coord kernel_size(3, 5, 7);
        nanovdb::Coord stride(1, 1, 1);

        auto topo_f =
            ops::gatherScatterDefaultSparseConvTopology(*grid, *grid, kernel_size, stride);
        auto topo_t =
            ops::gatherScatterDefaultSparseConvTransposeTopology(*grid, *grid, kernel_size, stride);

        torch::manual_seed(71);
        auto features = torch::randn({N, C}, opts(device, torch::kFloat64));
        auto weights  = torch::randn({C, C, 3, 5, 7}, opts(device, torch::kFloat64));
        auto W_flip   = torch::flip(weights, {2, 3, 4});

        auto out_t = ops::gatherScatterDefaultSparseConvTranspose(features, weights, topo_t);
        auto out_f = ops::gatherScatterDefaultSparseConv(features, W_flip, topo_f);

        assertNoNanInf(out_t, "FlipIdentity3x5x7 transpose output");
        assertNoNanInf(out_f, "FlipIdentity3x5x7 forward output");
        EXPECT_TRUE(torch::allclose(out_t.cpu(), out_f.cpu(), 1e-8, 1e-8))
            << "conv_transpose(x,W) != conv(x,flip(W)) for 3x5x7 on "
            << (dev_type == torch::kCPU ? "CPU" : "CUDA");
    }
}

// =============================================================================
// Strided transposed convolution tests (parameterized)
// =============================================================================

class GatherScatterDefaultStridedTransposeTest : public ::testing::TestWithParam<ConvParam> {};

TEST_P(GatherScatterDefaultStridedTransposeTest, StridedTransposeForward) {
    auto [dev_type, dtype, C] = GetParam();
    skipIfCudaUnavailable(dev_type);
    auto device = makeDevice(dev_type);

    auto src_grid = makeStridedTestGrid(device);
    nanovdb::Coord kernel_size(3, 3, 3);
    nanovdb::Coord stride(2, 2, 2);
    auto fwd_dst = src_grid->convolutionOutput(kernel_size, stride);

    auto topo = ops::gatherScatterDefaultSparseConvTransposeTopology(
        *fwd_dst, *src_grid, kernel_size, stride);

    int64_t S = topo.featureTotalVoxels;
    int64_t D = topo.outputTotalVoxels;

    torch::manual_seed(80);
    auto features = torch::randn({S, C}, opts(device, dtype));
    auto weights  = torch::randn({C, C, 3, 3, 3}, opts(device, dtype));

    auto out = ops::gatherScatterDefaultSparseConvTranspose(features, weights, topo);

    EXPECT_EQ(out.size(0), D);
    EXPECT_EQ(out.size(1), C);
    assertNoNanInf(out, "StridedTransposeForward output");
}

TEST_P(GatherScatterDefaultStridedTransposeTest, StridedTransposeBackward) {
    auto [dev_type, dtype, C] = GetParam();
    skipIfCudaUnavailable(dev_type);
    auto device = makeDevice(dev_type);

    auto src_grid = makeStridedTestGrid(device);
    nanovdb::Coord kernel_size(3, 3, 3);
    nanovdb::Coord stride(2, 2, 2);
    auto fwd_dst = src_grid->convolutionOutput(kernel_size, stride);

    auto topo = ops::gatherScatterDefaultSparseConvTransposeTopology(
        *fwd_dst, *src_grid, kernel_size, stride);

    int64_t S = topo.featureTotalVoxels;
    int64_t D = topo.outputTotalVoxels;

    torch::manual_seed(81);
    auto features    = torch::randn({S, C}, opts(device, dtype));
    auto weights     = torch::randn({C, C, 3, 3, 3}, opts(device, dtype));
    auto grad_output = torch::randn({D, C}, opts(device, dtype));

    auto [gf, gw] =
        ops::gatherScatterDefaultSparseConvTransposeBackward(grad_output, features, weights, topo);

    EXPECT_EQ(gf.sizes(), features.sizes());
    EXPECT_EQ(gw.sizes(), weights.sizes());
    assertNoNanInf(gf, "StridedTransposeBackward grad_features");
    assertNoNanInf(gw, "StridedTransposeBackward grad_weights");
}

// =============================================================================
// Asymmetric stride tests
// =============================================================================

TEST(GatherScatterDefaultAsymmetricStride, ForwardBackwardAdjoint) {
    for (auto dev_type: {torch::kCPU, torch::kCUDA}) {
        skipIfCudaUnavailable(dev_type);
        auto device = makeDevice(dev_type);

        auto src_grid = makeDenseTestGrid(6, device);
        nanovdb::Coord kernel_size(3, 3, 3);
        nanovdb::Coord stride(1, 2, 3);
        auto dst_grid = src_grid->convolutionOutput(kernel_size, stride);

        auto topo =
            ops::gatherScatterDefaultSparseConvTopology(*src_grid, *dst_grid, kernel_size, stride);

        int64_t const S     = topo.featureTotalVoxels;
        int64_t const D     = topo.outputTotalVoxels;
        int64_t const C_in  = 4;
        int64_t const C_out = 8;

        EXPECT_GT(D, 0) << "Asymmetric stride produced empty output grid";

        torch::manual_seed(90);
        auto x  = torch::randn({S, C_in}, opts(device, torch::kFloat64));
        auto W  = torch::randn({C_out, C_in, 3, 3, 3}, opts(device, torch::kFloat64));
        auto gy = torch::randn({D, C_out}, opts(device, torch::kFloat64));

        auto y        = ops::gatherScatterDefaultSparseConv(x, W, topo);
        auto [gx, gW] = ops::gatherScatterDefaultSparseConvBackward(gy, x, W, topo);

        assertNoNanInf(y, "AsymmetricStride forward");
        assertNoNanInf(gx, "AsymmetricStride grad_features");
        assertNoNanInf(gW, "AsymmetricStride grad_weights");

        EXPECT_EQ(y.size(0), D);
        EXPECT_EQ(y.size(1), C_out);
        EXPECT_EQ(gx.sizes(), x.sizes());
        EXPECT_EQ(gW.sizes(), W.sizes());

        auto ref = naiveConvForward(x, W, topo);
        EXPECT_TRUE(torch::allclose(y.cpu().to(torch::kFloat64), ref, 1e-8, 1e-8))
            << "Asymmetric stride value mismatch on " << (dev_type == torch::kCPU ? "CPU" : "CUDA");

        checkAdjointIdentity(x, W, gy, y, gx, gW, "AsymmetricStride");
    }
}

// =============================================================================
// Error path tests
// =============================================================================

TEST(GatherScatterDefaultError, WrongTopologyDirection) {
    auto device     = torch::Device(torch::kCPU);
    auto grid       = makeDenseTestGrid(4, device);
    int64_t const N = 64;
    int64_t const C = 4;
    nanovdb::Coord ks(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);

    auto fwd_topo = ops::gatherScatterDefaultSparseConvTopology(*grid, *grid, ks, stride);
    auto t_topo   = ops::gatherScatterDefaultSparseConvTransposeTopology(*grid, *grid, ks, stride);

    auto features = torch::randn({N, C}, torch::kFloat32);
    auto weights  = torch::randn({C, C, 3, 3, 3}, torch::kFloat32);
    auto grad_out = torch::randn({N, C}, torch::kFloat32);

    EXPECT_THROW(ops::gatherScatterDefaultSparseConvTranspose(features, weights, fwd_topo),
                 c10::Error);
    EXPECT_THROW(ops::gatherScatterDefaultSparseConv(features, weights, t_topo), c10::Error);
    EXPECT_THROW(ops::gatherScatterDefaultSparseConvBackward(grad_out, features, weights, t_topo),
                 c10::Error);
    EXPECT_THROW(
        ops::gatherScatterDefaultSparseConvTransposeBackward(grad_out, features, weights, fwd_topo),
        c10::Error);
}

TEST(GatherScatterDefaultError, FeaturesNot2D) {
    auto device     = torch::Device(torch::kCPU);
    auto grid       = makeDenseTestGrid(4, device);
    int64_t const C = 4;
    nanovdb::Coord ks(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);
    auto topo = ops::gatherScatterDefaultSparseConvTopology(*grid, *grid, ks, stride);

    auto features_3d = torch::randn({1, 64, C}, torch::kFloat32);
    auto weights     = torch::randn({C, C, 3, 3, 3}, torch::kFloat32);

    EXPECT_THROW(ops::gatherScatterDefaultSparseConv(features_3d, weights, topo), c10::Error);
}

TEST(GatherScatterDefaultError, NonContiguousFeatures) {
    auto device     = torch::Device(torch::kCPU);
    auto grid       = makeDenseTestGrid(4, device);
    int64_t const N = 64;
    int64_t const C = 4;
    nanovdb::Coord ks(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);
    auto topo = ops::gatherScatterDefaultSparseConvTopology(*grid, *grid, ks, stride);

    auto features_nc = torch::randn({N, C}, torch::kFloat32).t().contiguous().t();
    auto weights     = torch::randn({C, C, 3, 3, 3}, torch::kFloat32);

    EXPECT_FALSE(features_nc.is_contiguous());
    EXPECT_THROW(ops::gatherScatterDefaultSparseConv(features_nc, weights, topo), c10::Error);
}

TEST(GatherScatterDefaultError, ChannelMismatch) {
    auto device     = torch::Device(torch::kCPU);
    auto grid       = makeDenseTestGrid(4, device);
    int64_t const N = 64;
    nanovdb::Coord ks(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);
    auto topo = ops::gatherScatterDefaultSparseConvTopology(*grid, *grid, ks, stride);

    auto features    = torch::randn({N, 4}, torch::kFloat32);
    auto weights_bad = torch::randn({8, 6, 3, 3, 3}, torch::kFloat32);

    EXPECT_THROW(ops::gatherScatterDefaultSparseConv(features, weights_bad, topo), c10::Error);
}

TEST(GatherScatterDefaultError, SpatialDimMismatch) {
    auto device     = torch::Device(torch::kCPU);
    auto grid       = makeDenseTestGrid(4, device);
    int64_t const N = 64;
    int64_t const C = 4;
    nanovdb::Coord ks(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);
    auto topo = ops::gatherScatterDefaultSparseConvTopology(*grid, *grid, ks, stride);

    auto features    = torch::randn({N, C}, torch::kFloat32);
    auto weights_bad = torch::randn({C, C, 5, 5, 5}, torch::kFloat32);

    EXPECT_THROW(ops::gatherScatterDefaultSparseConv(features, weights_bad, topo), c10::Error);
}

TEST(GatherScatterDefaultError, DeviceMismatch) {
    if (!cudaIsAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }
    auto device     = torch::Device(torch::kCPU);
    auto grid       = makeDenseTestGrid(4, device);
    int64_t const N = 64;
    int64_t const C = 4;
    nanovdb::Coord ks(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);
    auto topo = ops::gatherScatterDefaultSparseConvTopology(*grid, *grid, ks, stride);

    auto features = torch::randn({N, C}, torch::kFloat32);
    auto weights_gpu =
        torch::randn({C, C, 3, 3, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));

    EXPECT_THROW(ops::gatherScatterDefaultSparseConv(features, weights_gpu, topo), c10::Error);
}

// =============================================================================
// Mixed scalar-type promotion tests
// =============================================================================

TEST(GatherScatterDefaultMixedType, PromotionPolicy) {
    for (auto dev_type: {torch::kCPU, torch::kCUDA}) {
        skipIfCudaUnavailable(dev_type);
        auto device = makeDevice(dev_type);

        auto grid       = makeDenseTestGrid(4, device);
        int64_t const N = 64;
        int64_t const C = 4;

        nanovdb::Coord kernel_size(3, 3, 3);
        nanovdb::Coord stride(1, 1, 1);
        auto topo = ops::gatherScatterDefaultSparseConvTopology(*grid, *grid, kernel_size, stride);

        struct Case {
            torch::ScalarType feat_type;
            torch::ScalarType weight_type;
            torch::ScalarType expected_out;
        };
        std::vector<Case> cases = {
            {torch::kFloat16, torch::kFloat32, torch::kFloat32},
            {torch::kBFloat16, torch::kFloat64, torch::kFloat64},
            {torch::kFloat32, torch::kFloat16, torch::kFloat32},
            {torch::kFloat16, torch::kFloat64, torch::kFloat64},
            {torch::kBFloat16, torch::kFloat32, torch::kFloat32},
            {torch::kFloat32, torch::kBFloat16, torch::kFloat32},
        };

        for (auto const &[ft, wt, expected]: cases) {
            auto features = torch::randn({N, C}, opts(device, ft));
            auto weights  = torch::randn({C, C, 3, 3, 3}, opts(device, wt));

            auto out = ops::gatherScatterDefaultSparseConv(features, weights, topo);

            EXPECT_EQ(out.scalar_type(), expected)
                << "features=" << dtypeStr(ft) << " weights=" << dtypeStr(wt)
                << " expected=" << dtypeStr(expected) << " got=" << dtypeStr(out.scalar_type())
                << " on " << (dev_type == torch::kCPU ? "CPU" : "CUDA");

            assertNoNanInf(out, "MixedType forward output");
        }
    }
}

TEST(GatherScatterDefaultMixedType, BackwardPromotionPolicy) {
    for (auto dev_type: {torch::kCPU, torch::kCUDA}) {
        skipIfCudaUnavailable(dev_type);
        auto device = makeDevice(dev_type);

        auto grid       = makeDenseTestGrid(4, device);
        int64_t const N = 64;
        int64_t const C = 4;

        nanovdb::Coord kernel_size(3, 3, 3);
        nanovdb::Coord stride(1, 1, 1);
        auto topo = ops::gatherScatterDefaultSparseConvTopology(*grid, *grid, kernel_size, stride);

        auto features    = torch::randn({N, C}, opts(device, torch::kFloat16));
        auto weights     = torch::randn({C, C, 3, 3, 3}, opts(device, torch::kFloat32));
        auto grad_output = torch::randn({N, C}, opts(device, torch::kFloat32));

        auto [gf, gw] =
            ops::gatherScatterDefaultSparseConvBackward(grad_output, features, weights, topo);

        EXPECT_EQ(gf.scalar_type(), torch::kFloat32);
        EXPECT_EQ(gw.scalar_type(), torch::kFloat32);
        assertNoNanInf(gf, "MixedType backward grad_features");
        assertNoNanInf(gw, "MixedType backward grad_weights");
    }
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
        for (auto dt : {torch::kFloat16, torch::kBFloat16}) {
            for (int64_t C : {4, 16, 32}) {
                v.push_back(cp(dev, dt, C));
            }
        }
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
        for (auto [k0, k1, k2] : std::vector<std::tuple<int,int,int>>{
                 {3,3,3}, {3,5,1}}) {
            v.push_back(kp(dev, torch::kFloat16, k0, k1, k2));
            v.push_back(kp(dev, torch::kBFloat16, k0, k1, k2));
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

INSTANTIATE_TEST_SUITE_P(Configs, GatherScatterDefaultStridedTransposeTest,
    ::testing::ValuesIn(channelConfigs()), paramName);

INSTANTIATE_TEST_SUITE_P(KernelSizes, GatherScatterDefaultKernelSizeTest,
    ::testing::ValuesIn(kernelSizeConfigs()), kernelSizeParamName);

// clang-format on
