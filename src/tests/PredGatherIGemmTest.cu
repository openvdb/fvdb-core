// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// PredGatherIGemmTest.cu -- GTest for PredGatherIGemm sparse convolution,
// validating against GatherScatterDefault as ground truth.

#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/GridBatchData.h>
#include <fvdb/detail/ops/BuildGridFromIjk.h>
#include <fvdb/detail/ops/convolution/GatherScatterDefault.h>
#include <fvdb/detail/ops/convolution/PredGatherIGemm.h>

#include <c10/cuda/CUDAStream.h>
#include <torch/types.h>

#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

using namespace fvdb;
using namespace fvdb::detail;

// =============================================================================
// Helpers
// =============================================================================

static bool
deviceSupportsSm80() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
        return false;
    int major = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
    return major >= 8;
}

static torch::Device
makeDevice() {
    return torch::Device(torch::kCUDA, 0);
}

static torch::TensorOptions
topts(torch::Device device, torch::ScalarType dtype = torch::kFloat32) {
    return torch::dtype(dtype).device(device);
}

static uint32_t
coordinateBitpack(uint32_t x) {
    x &= 0x49249249;
    x |= (x >> 2);
    x &= 0xc30c30c3;
    x |= (x >> 4);
    x &= 0x0f00f00f;
    x |= (x >> 8);
    x &= 0xff0000ff;
    x |= (x >> 16);
    x &= 0x0000ffff;
    return x;
}

static std::pair<std::vector<nanovdb::Coord>, std::vector<nanovdb::Coord>>
generateCoordinates(
    int ambient_voxels, float input_occ, float output_occ, float overlap, int seed = 12345) {
    std::mt19937 generator(seed);
    std::uniform_int_distribution<int> distribution(0, ambient_voxels - 1);
    std::vector<bool> voxmap(ambient_voxels, false);

    int target_input = static_cast<int>(input_occ * static_cast<float>(ambient_voxels));
    int active       = 0;
    while (active < target_input) {
        int i = distribution(generator);
        if (!voxmap[i]) {
            voxmap[i] = true;
            ++active;
        }
    }

    std::vector<nanovdb::Coord> inputPoints;
    for (int i = 0; i < ambient_voxels; ++i)
        if (voxmap[i]) {
            int x = static_cast<int>(coordinateBitpack(i & 0x49249249));
            int y = static_cast<int>(coordinateBitpack((i >> 1) & 0x49249249));
            int z = static_cast<int>(coordinateBitpack((i >> 2) & 0x49249249));
            inputPoints.emplace_back(x, y, z);
        }

    int target_overlap = static_cast<int>(overlap * static_cast<float>(ambient_voxels));
    while (active > target_overlap) {
        int i = distribution(generator);
        if (voxmap[i]) {
            voxmap[i] = false;
            --active;
        }
    }
    int target_output = static_cast<int>(output_occ * static_cast<float>(ambient_voxels));
    while (active < target_output) {
        int i = distribution(generator);
        if (!voxmap[i]) {
            voxmap[i] = true;
            ++active;
        }
    }

    std::vector<nanovdb::Coord> outputPoints;
    for (int i = 0; i < ambient_voxels; ++i)
        if (voxmap[i]) {
            int x = static_cast<int>(coordinateBitpack(i & 0x49249249));
            int y = static_cast<int>(coordinateBitpack((i >> 1) & 0x49249249));
            int z = static_cast<int>(coordinateBitpack((i >> 2) & 0x49249249));
            outputPoints.emplace_back(x, y, z);
        }

    return {inputPoints, outputPoints};
}

static c10::intrusive_ptr<GridBatchData>
makeFvdbGrid(const std::vector<nanovdb::Coord> &coords, torch::Device device) {
    const int64_t N = static_cast<int64_t>(coords.size());
    auto ijk_cpu    = torch::empty({N, 3}, torch::dtype(torch::kInt32));
    auto *ptr       = ijk_cpu.data_ptr<int32_t>();
    for (int64_t i = 0; i < N; ++i) {
        ptr[i * 3 + 0] = coords[i][0];
        ptr[i * 3 + 1] = coords[i][1];
        ptr[i * 3 + 2] = coords[i][2];
    }
    auto ijk_dev = ijk_cpu.to(device);
    JaggedTensor jt(ijk_dev);
    std::vector<nanovdb::Vec3d> vs      = {{1.0, 1.0, 1.0}};
    std::vector<nanovdb::Vec3d> origins = {{0.0, 0.0, 0.0}};
    return ops::createNanoGridFromIJK(jt, vs, origins);
}

struct CudaTimer {
    cudaEvent_t start_, stop_;

    CudaTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }
    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    CudaTimer(const CudaTimer &)            = delete;
    CudaTimer &operator=(const CudaTimer &) = delete;

    void
    recordStart() {
        cudaEventRecord(start_);
    }

    float
    recordStopMs() {
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }
};

struct BenchConfig {
    const char *label;
    int ambient_voxels;
    float input_occ;
    float output_occ;
    float overlap;
};

static void
runBenchmark(const BenchConfig &cfg, torch::Device device) {
    auto [inputCoords, outputCoords] =
        generateCoordinates(cfg.ambient_voxels, cfg.input_occ, cfg.output_occ, cfg.overlap);

    auto input_grid  = makeFvdbGrid(inputCoords, device);
    auto output_grid = makeFvdbGrid(outputCoords, device);

    const int64_t N_in  = input_grid->totalVoxels();
    const int64_t N_out = output_grid->totalVoxels();
    ASSERT_GT(N_in, 0);
    ASSERT_GT(N_out, 0);

    const int64_t Cin = 64, Cout = 128;
    torch::manual_seed(9999);

    auto features     = torch::randn({N_in, Cin}, topts(device));
    auto fvdb_weights = torch::randn({Cout, Cin, 3, 3, 3}, topts(device));

    uint32_t leafCount = output_grid->numLeavesAt(0);

    nanovdb::Coord ks(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);

    std::cout << "  [" << cfg.label << "] N_in=" << N_in << "  N_out=" << N_out
              << "  leaves=" << leafCount << "  leaf_occ=" << (100.f * N_out / (leafCount * 512))
              << "%" << std::endl;

    constexpr int kWarmup = 2;
    constexpr int kIters  = 5;
    CudaTimer timer;

    // --- PredGatherIGemm ---
    for (int i = 0; i < kWarmup; ++i)
        ops::predGatherIGemmSparseConv(features, fvdb_weights, *input_grid, *output_grid, 3, 1);
    timer.recordStart();
    for (int i = 0; i < kIters; ++i)
        ops::predGatherIGemmSparseConv(features, fvdb_weights, *input_grid, *output_grid, 3, 1);
    float igemm_ms = timer.recordStopMs() / kIters;

    // --- GatherScatterDefault (with topology build) ---
    for (int i = 0; i < kWarmup; ++i) {
        auto t = ops::gatherScatterDefaultSparseConvTopology(*input_grid, *output_grid, ks, stride);
        ops::gatherScatterDefaultSparseConv(features, fvdb_weights, t);
    }
    timer.recordStart();
    for (int i = 0; i < kIters; ++i) {
        auto t = ops::gatherScatterDefaultSparseConvTopology(*input_grid, *output_grid, ks, stride);
        ops::gatherScatterDefaultSparseConv(features, fvdb_weights, t);
    }
    float gs_with_topo_ms = timer.recordStopMs() / kIters;

    // --- GatherScatterDefault (topology pre-computed) ---
    auto topo = ops::gatherScatterDefaultSparseConvTopology(*input_grid, *output_grid, ks, stride);
    for (int i = 0; i < kWarmup; ++i)
        ops::gatherScatterDefaultSparseConv(features, fvdb_weights, topo);
    timer.recordStart();
    for (int i = 0; i < kIters; ++i)
        ops::gatherScatterDefaultSparseConv(features, fvdb_weights, topo);
    float gs_no_topo_ms = timer.recordStopMs() / kIters;

    std::cout << "    PredGatherIGemm: " << igemm_ms << " ms"
              << "  |  GS+topo: " << gs_with_topo_ms << " ms"
              << "  |  GS only: " << gs_no_topo_ms << " ms" << std::endl;

    // -- Correctness sanity check --
    auto igemm_output =
        ops::predGatherIGemmSparseConv(features, fvdb_weights, *input_grid, *output_grid, 3, 1);
    auto gs_output = ops::gatherScatterDefaultSparseConv(features, fvdb_weights, topo);

    auto igemm_f64 = igemm_output.cpu().to(torch::kFloat64);
    auto gs_f64    = gs_output.cpu().to(torch::kFloat64);
    auto diff      = (igemm_f64 - gs_f64).abs();

    EXPECT_TRUE(torch::allclose(igemm_f64, gs_f64, /*rtol=*/5e-3, /*atol=*/0.5))
        << cfg.label << " correctness failed, max diff=" << diff.max().item<double>()
        << ", mean diff=" << diff.mean().item<double>();
}

// =============================================================================
// Tests
// =============================================================================

TEST(PredGatherIGemm, MatchesGatherScatterDefault) {
    if (!deviceSupportsSm80()) {
        GTEST_SKIP() << "Requires SM80+";
    }
    auto device = makeDevice();

    auto [inputCoords, outputCoords] = generateCoordinates(16 * 1024, 0.75f, 0.75f, 0.65f);

    auto input_grid  = makeFvdbGrid(inputCoords, device);
    auto output_grid = makeFvdbGrid(outputCoords, device);

    const int64_t N_in  = input_grid->totalVoxels();
    const int64_t N_out = output_grid->totalVoxels();
    ASSERT_GT(N_in, 0);
    ASSERT_GT(N_out, 0);

    const int64_t Cin = 64, Cout = 128;

    struct GeomConfig {
        int kernel_size, stride;
    };
    GeomConfig geom_configs[] = {{3, 1}, {3, 2}, {5, 1}, {5, 2}, {7, 1}, {7, 2}};

    for (const auto &gc: geom_configs) {
        torch::manual_seed(8888);

        auto features     = torch::randn({N_in, Cin}, topts(device));
        auto fvdb_weights = torch::randn(
            {Cout, Cin, gc.kernel_size, gc.kernel_size, gc.kernel_size}, topts(device));

        auto igemm_output = ops::predGatherIGemmSparseConv(
            features, fvdb_weights, *input_grid, *output_grid, gc.kernel_size, gc.stride);

        nanovdb::Coord ks(gc.kernel_size, gc.kernel_size, gc.kernel_size);
        nanovdb::Coord st(gc.stride, gc.stride, gc.stride);
        auto topo = ops::gatherScatterDefaultSparseConvTopology(*input_grid, *output_grid, ks, st);
        auto gs_output = ops::gatherScatterDefaultSparseConv(features, fvdb_weights, topo);

        auto igemm_f64 = igemm_output.cpu().to(torch::kFloat64);
        auto gs_f64    = gs_output.cpu().to(torch::kFloat64);
        auto diff      = (igemm_f64 - gs_f64).abs();

        ASSERT_EQ(igemm_f64.size(0), gs_f64.size(0))
            << "kernel=" << gc.kernel_size << " stride=" << gc.stride;
        ASSERT_EQ(igemm_f64.size(1), gs_f64.size(1))
            << "kernel=" << gc.kernel_size << " stride=" << gc.stride;

        // TF32 tensor-core arithmetic (10-bit mantissa) vs full f32 cuBLAS in
        // GatherScatterDefault.  With randn inputs the expected per-element error
        // is ~0.3% relative; observed max diff is ~0.14 on outputs of magnitude ~40.
        EXPECT_TRUE(torch::allclose(igemm_f64, gs_f64, /*rtol=*/5e-3, /*atol=*/0.5))
            << "kernel=" << gc.kernel_size << " stride=" << gc.stride
            << ": max diff=" << diff.max().item<double>()
            << ", mean diff=" << diff.mean().item<double>();
    }
}

TEST(PredGatherIGemm, MatchesGatherScatterDefaultMultipleChannels) {
    if (!deviceSupportsSm80()) {
        GTEST_SKIP() << "Requires SM80+";
    }
    auto device = makeDevice();

    auto [inputCoords, outputCoords] = generateCoordinates(8 * 1024, 0.75f, 0.75f, 0.65f, 54321);

    auto input_grid  = makeFvdbGrid(inputCoords, device);
    auto output_grid = makeFvdbGrid(outputCoords, device);

    const int64_t N_in  = input_grid->totalVoxels();
    const int64_t N_out = output_grid->totalVoxels();
    ASSERT_GT(N_in, 0);
    ASSERT_GT(N_out, 0);

    struct ChannelConfig {
        int64_t cin, cout;
    };
    ChannelConfig channel_configs[] = {
        {32, 32},
        {32, 64},
        {64, 64},
        {128, 64},
        {64, 128},
        {128, 128},
        {256, 128},
    };

    struct GeomConfig {
        int kernel_size, stride;
    };
    GeomConfig geom_configs[] = {{3, 1}, {3, 2}, {5, 1}, {5, 2}, {7, 1}, {7, 2}};

    for (const auto &gc: geom_configs) {
        for (const auto &cfg: channel_configs) {
            torch::manual_seed(42);

            auto features     = torch::randn({N_in, cfg.cin}, topts(device));
            auto fvdb_weights = torch::randn(
                {cfg.cout, cfg.cin, gc.kernel_size, gc.kernel_size, gc.kernel_size}, topts(device));

            auto igemm_output = ops::predGatherIGemmSparseConv(
                features, fvdb_weights, *input_grid, *output_grid, gc.kernel_size, gc.stride);

            nanovdb::Coord ks(gc.kernel_size, gc.kernel_size, gc.kernel_size);
            nanovdb::Coord st(gc.stride, gc.stride, gc.stride);
            auto topo =
                ops::gatherScatterDefaultSparseConvTopology(*input_grid, *output_grid, ks, st);
            auto gs_output = ops::gatherScatterDefaultSparseConv(features, fvdb_weights, topo);

            auto igemm_f64 = igemm_output.cpu().to(torch::kFloat64);
            auto gs_f64    = gs_output.cpu().to(torch::kFloat64);
            auto diff      = (igemm_f64 - gs_f64).abs();

            ASSERT_EQ(igemm_f64.size(0), gs_f64.size(0))
                << "kernel=" << gc.kernel_size << " stride=" << gc.stride << " C=" << cfg.cin
                << " K=" << cfg.cout;
            ASSERT_EQ(igemm_f64.size(1), gs_f64.size(1))
                << "kernel=" << gc.kernel_size << " stride=" << gc.stride << " C=" << cfg.cin
                << " K=" << cfg.cout;

            EXPECT_TRUE(torch::allclose(igemm_f64, gs_f64, /*rtol=*/5e-3, /*atol=*/0.5))
                << "kernel=" << gc.kernel_size << " stride=" << gc.stride << " C=" << cfg.cin
                << " K=" << cfg.cout << ": max diff=" << diff.max().item<double>()
                << ", mean diff=" << diff.mean().item<double>();
        }
    }
}

// =============================================================================
// Speed comparison
// =============================================================================

TEST(PredGatherIGemm, SpeedComparison) {
    if (!deviceSupportsSm80()) {
        GTEST_SKIP() << "Requires SM80+";
    }
    auto device = makeDevice();

    // clang-format off
    BenchConfig configs[] = {
        {"1M dense",     1024 * 1024,     0.75f, 0.75f, 0.65f},
        {"2M dense",     2 * 1024 * 1024, 0.75f, 0.75f, 0.65f},
        {"4M sparse-25", 4 * 1024 * 1024, 0.25f, 0.25f, 0.20f},
        {"8M sparse-10", 8 * 1024 * 1024, 0.10f, 0.10f, 0.08f},
    };
    // clang-format on

    std::cout << "SpeedComparison (Cin=64, Cout=128, kernel 3x3x3):" << std::endl;
    for (const auto &cfg: configs)
        runBenchmark(cfg, device);
}
