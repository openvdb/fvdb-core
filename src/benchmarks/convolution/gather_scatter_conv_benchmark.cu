// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// gather_scatter_conv_benchmark.cu -- Performance benchmarks for the
// GatherScatterDefault sparse convolution operator.
//
// Benchmark groups:
//
//   TopologyBuild   Topology construction cost (two-pass atomic counting).
//   Forward         End-to-end forward sparse convolution (topology amortized).
//   Backward        End-to-end backward sparse convolution (topology amortized).
//   DenseBaseline   Equivalent dense torch::conv3d via cuDNN / MKL-DNN.
//   Sparsity        Sparse forward at varying occupancy vs fixed dense cost.
//   ChannelScale    Forward conv with varying channel width at fixed grid size.
//
// All tensors and topologies are pre-allocated outside the timing loop.
// CUDA benchmarks synchronize after warmup and after every iteration.
//

#ifdef __NVCC__
#pragma nv_diag_suppress 177
#endif

#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/GridBatchImpl.h>
#include <fvdb/detail/ops/convolution/GatherScatterDefault.h>

#include <torch/torch.h>

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

using namespace fvdb;
using namespace fvdb::detail;

namespace F = torch::nn::functional;

// ============================================================================
// Grid factory helpers
// ============================================================================

static c10::intrusive_ptr<GridBatchImpl>
makeGrid(torch::Tensor ijk_2d, torch::Device device) {
    auto ijk_dev = ijk_2d.to(device);
    JaggedTensor jt(ijk_dev);
    std::vector<nanovdb::Vec3d> voxel_sizes = {{1.0, 1.0, 1.0}};
    std::vector<nanovdb::Vec3d> origins     = {{0.0, 0.0, 0.0}};
    return GridBatchImpl::createFromIjk(jt, voxel_sizes, origins);
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

static c10::intrusive_ptr<GridBatchImpl>
makeSparseGrid(int bbox_dim, int occupancy_pct, torch::Device device) {
    int64_t total = static_cast<int64_t>(bbox_dim) * bbox_dim * bbox_dim;
    int64_t N     = std::max<int64_t>(1, total * occupancy_pct / 100);

    torch::manual_seed(42);
    auto perm     = torch::randperm(total, torch::kInt64);
    auto selected = perm.slice(0, 0, N);

    auto ijk     = torch::zeros({N, 3}, torch::kInt32);
    auto sel_acc = selected.accessor<int64_t, 1>();
    auto ijk_acc = ijk.accessor<int32_t, 2>();
    for (int64_t i = 0; i < N; ++i) {
        int64_t idx   = sel_acc[i];
        ijk_acc[i][0] = static_cast<int32_t>(idx / (bbox_dim * bbox_dim));
        ijk_acc[i][1] = static_cast<int32_t>((idx / bbox_dim) % bbox_dim);
        ijk_acc[i][2] = static_cast<int32_t>(idx % bbox_dim);
    }

    return makeGrid(ijk, device);
}

static torch::TensorOptions
topts(torch::Device device) {
    return torch::dtype(torch::kFloat32).device(device);
}

// ============================================================================
// A. TopologyBuild
// ============================================================================

static void
BM_Conv_TopologyBuild_CUDA(benchmark::State &state) {
    if (!torch::cuda::is_available()) {
        state.SkipWithError("CUDA not available");
        return;
    }
    int dim     = static_cast<int>(state.range(0));
    auto device = torch::Device(torch::kCUDA, 0);
    auto grid   = makeDenseGrid(dim, device);

    nanovdb::Coord ks(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);

    // Warmup
    auto topo = ops::gatherScatterDefaultSparseConvTopology(*grid, *grid, ks, stride);
    torch::cuda::synchronize();

    for (auto _: state) {
        auto t = ops::gatherScatterDefaultSparseConvTopology(*grid, *grid, ks, stride);
        torch::cuda::synchronize();
        benchmark::DoNotOptimize(t.totalPairs);
    }
    state.SetItemsProcessed(state.iterations() * grid->totalVoxels());
    state.counters["voxels"] = static_cast<double>(grid->totalVoxels());
}

static void
BM_Conv_TopologyBuild_CPU(benchmark::State &state) {
    int dim     = static_cast<int>(state.range(0));
    auto device = torch::Device(torch::kCPU);
    auto grid   = makeDenseGrid(dim, device);

    nanovdb::Coord ks(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);

    auto topo = ops::gatherScatterDefaultSparseConvTopology(*grid, *grid, ks, stride);

    for (auto _: state) {
        auto t = ops::gatherScatterDefaultSparseConvTopology(*grid, *grid, ks, stride);
        benchmark::DoNotOptimize(t.totalPairs);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * grid->totalVoxels());
    state.counters["voxels"] = static_cast<double>(grid->totalVoxels());
}

// ============================================================================
// B. Forward convolution
// ============================================================================

static void
BM_Conv_Forward_CUDA_C32(benchmark::State &state) {
    if (!torch::cuda::is_available()) {
        state.SkipWithError("CUDA not available");
        return;
    }
    int dim     = static_cast<int>(state.range(0));
    auto device = torch::Device(torch::kCUDA, 0);
    auto grid   = makeDenseGrid(dim, device);

    nanovdb::Coord ks(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);
    auto topo = ops::gatherScatterDefaultSparseConvTopology(*grid, *grid, ks, stride);

    int64_t N     = grid->totalVoxels();
    int64_t C     = 32;
    auto features = torch::randn({N, C}, topts(device));
    auto weights  = torch::randn({C, C, 3, 3, 3}, topts(device));

    auto out = ops::gatherScatterDefaultSparseConv(features, weights, topo);
    torch::cuda::synchronize();

    for (auto _: state) {
        auto o = ops::gatherScatterDefaultSparseConv(features, weights, topo);
        torch::cuda::synchronize();
        benchmark::DoNotOptimize(o.data_ptr<float>());
    }
    state.SetItemsProcessed(state.iterations() * N);
    state.counters["voxels"] = static_cast<double>(N);
    state.counters["pairs"]  = static_cast<double>(topo.totalPairs);
}

static void
BM_Conv_Forward_CPU_C32(benchmark::State &state) {
    int dim     = static_cast<int>(state.range(0));
    auto device = torch::Device(torch::kCPU);
    auto grid   = makeDenseGrid(dim, device);

    nanovdb::Coord ks(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);
    auto topo = ops::gatherScatterDefaultSparseConvTopology(*grid, *grid, ks, stride);

    int64_t N     = grid->totalVoxels();
    int64_t C     = 32;
    auto features = torch::randn({N, C}, topts(device));
    auto weights  = torch::randn({C, C, 3, 3, 3}, topts(device));

    auto out = ops::gatherScatterDefaultSparseConv(features, weights, topo);

    for (auto _: state) {
        auto o = ops::gatherScatterDefaultSparseConv(features, weights, topo);
        benchmark::DoNotOptimize(o.data_ptr<float>());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * N);
    state.counters["voxels"] = static_cast<double>(N);
    state.counters["pairs"]  = static_cast<double>(topo.totalPairs);
}

// ============================================================================
// B. Backward convolution
// ============================================================================

static void
BM_Conv_Backward_CUDA_C32(benchmark::State &state) {
    if (!torch::cuda::is_available()) {
        state.SkipWithError("CUDA not available");
        return;
    }
    int dim     = static_cast<int>(state.range(0));
    auto device = torch::Device(torch::kCUDA, 0);
    auto grid   = makeDenseGrid(dim, device);

    nanovdb::Coord ks(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);
    auto topo = ops::gatherScatterDefaultSparseConvTopology(*grid, *grid, ks, stride);

    int64_t N        = grid->totalVoxels();
    int64_t C        = 32;
    auto features    = torch::randn({N, C}, topts(device));
    auto weights     = torch::randn({C, C, 3, 3, 3}, topts(device));
    auto grad_output = torch::randn({N, C}, topts(device));

    auto [gf, gw] =
        ops::gatherScatterDefaultSparseConvBackward(grad_output, features, weights, topo);
    torch::cuda::synchronize();

    for (auto _: state) {
        auto [gf2, gw2] =
            ops::gatherScatterDefaultSparseConvBackward(grad_output, features, weights, topo);
        torch::cuda::synchronize();
        benchmark::DoNotOptimize(gf2.data_ptr<float>());
    }
    state.SetItemsProcessed(state.iterations() * N);
    state.counters["voxels"] = static_cast<double>(N);
    state.counters["pairs"]  = static_cast<double>(topo.totalPairs);
}

static void
BM_Conv_Backward_CPU_C32(benchmark::State &state) {
    int dim     = static_cast<int>(state.range(0));
    auto device = torch::Device(torch::kCPU);
    auto grid   = makeDenseGrid(dim, device);

    nanovdb::Coord ks(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);
    auto topo = ops::gatherScatterDefaultSparseConvTopology(*grid, *grid, ks, stride);

    int64_t N        = grid->totalVoxels();
    int64_t C        = 32;
    auto features    = torch::randn({N, C}, topts(device));
    auto weights     = torch::randn({C, C, 3, 3, 3}, topts(device));
    auto grad_output = torch::randn({N, C}, topts(device));

    auto [gf, gw] =
        ops::gatherScatterDefaultSparseConvBackward(grad_output, features, weights, topo);

    for (auto _: state) {
        auto [gf2, gw2] =
            ops::gatherScatterDefaultSparseConvBackward(grad_output, features, weights, topo);
        benchmark::DoNotOptimize(gf2.data_ptr<float>());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * N);
    state.counters["voxels"] = static_cast<double>(N);
    state.counters["pairs"]  = static_cast<double>(topo.totalPairs);
}

// ============================================================================
// C. Dense conv3d baseline (cuDNN / MKL-DNN)
// ============================================================================

static void
BM_Conv_DenseBaseline_CUDA_C32(benchmark::State &state) {
    if (!torch::cuda::is_available()) {
        state.SkipWithError("CUDA not available");
        return;
    }
    int dim     = static_cast<int>(state.range(0));
    auto device = torch::Device(torch::kCUDA, 0);

    int64_t C   = 32;
    auto input  = torch::randn({1, C, dim, dim, dim}, topts(device));
    auto kernel = torch::randn({C, C, 3, 3, 3}, topts(device));

    auto out = F::conv3d(input, kernel, F::Conv3dFuncOptions().padding(1));
    torch::cuda::synchronize();

    for (auto _: state) {
        auto o = F::conv3d(input, kernel, F::Conv3dFuncOptions().padding(1));
        torch::cuda::synchronize();
        benchmark::DoNotOptimize(o.data_ptr<float>());
    }
    int64_t N = static_cast<int64_t>(dim) * dim * dim;
    state.SetItemsProcessed(state.iterations() * N);
    state.counters["voxels"] = static_cast<double>(N);
}

static void
BM_Conv_DenseBaseline_CPU_C32(benchmark::State &state) {
    int dim     = static_cast<int>(state.range(0));
    auto device = torch::Device(torch::kCPU);

    int64_t C   = 32;
    auto input  = torch::randn({1, C, dim, dim, dim}, topts(device));
    auto kernel = torch::randn({C, C, 3, 3, 3}, topts(device));

    auto out = F::conv3d(input, kernel, F::Conv3dFuncOptions().padding(1));

    for (auto _: state) {
        auto o = F::conv3d(input, kernel, F::Conv3dFuncOptions().padding(1));
        benchmark::DoNotOptimize(o.data_ptr<float>());
        benchmark::ClobberMemory();
    }
    int64_t N = static_cast<int64_t>(dim) * dim * dim;
    state.SetItemsProcessed(state.iterations() * N);
    state.counters["voxels"] = static_cast<double>(N);
}

// ============================================================================
// D. Sparsity breakeven
// ============================================================================

static void
BM_Conv_Sparsity_CUDA_C32(benchmark::State &state) {
    if (!torch::cuda::is_available()) {
        state.SkipWithError("CUDA not available");
        return;
    }
    int occupancy_pct = static_cast<int>(state.range(0));
    int bbox_dim      = 32;
    auto device       = torch::Device(torch::kCUDA, 0);

    auto grid = (occupancy_pct >= 100) ? makeDenseGrid(bbox_dim, device)
                                       : makeSparseGrid(bbox_dim, occupancy_pct, device);

    nanovdb::Coord ks(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);
    auto topo = ops::gatherScatterDefaultSparseConvTopology(*grid, *grid, ks, stride);

    int64_t N     = grid->totalVoxels();
    int64_t C     = 32;
    auto features = torch::randn({N, C}, topts(device));
    auto weights  = torch::randn({C, C, 3, 3, 3}, topts(device));

    auto out = ops::gatherScatterDefaultSparseConv(features, weights, topo);
    torch::cuda::synchronize();

    for (auto _: state) {
        auto o = ops::gatherScatterDefaultSparseConv(features, weights, topo);
        torch::cuda::synchronize();
        benchmark::DoNotOptimize(o.data_ptr<float>());
    }
    state.SetItemsProcessed(state.iterations() * N);
    state.counters["occupancy"] = static_cast<double>(occupancy_pct);
    state.counters["voxels"]    = static_cast<double>(N);
    state.counters["pairs"]     = static_cast<double>(topo.totalPairs);
}

static void
BM_Conv_DenseRef_CUDA_C32(benchmark::State &state) {
    if (!torch::cuda::is_available()) {
        state.SkipWithError("CUDA not available");
        return;
    }
    int bbox_dim = 32;
    auto device  = torch::Device(torch::kCUDA, 0);
    int64_t C    = 32;

    auto input  = torch::randn({1, C, bbox_dim, bbox_dim, bbox_dim}, topts(device));
    auto kernel = torch::randn({C, C, 3, 3, 3}, topts(device));

    auto out = F::conv3d(input, kernel, F::Conv3dFuncOptions().padding(1));
    torch::cuda::synchronize();

    for (auto _: state) {
        auto o = F::conv3d(input, kernel, F::Conv3dFuncOptions().padding(1));
        torch::cuda::synchronize();
        benchmark::DoNotOptimize(o.data_ptr<float>());
    }
    int64_t N = static_cast<int64_t>(bbox_dim) * bbox_dim * bbox_dim;
    state.SetItemsProcessed(state.iterations() * N);
    state.counters["voxels"] = static_cast<double>(N);
}

// ============================================================================
// D2. Large-scale sparsity breakeven
// ============================================================================
//
// At bbox=32 (32K voxels), dense cuDNN conv3d is so fast that sparse can never
// win -- the sparse path's fixed overhead from 27 sequential torch::mm launches
// dominates.  Realistic sparse 3D workloads (SDF surfaces, point clouds) have
// bounding boxes of 128-512+ with 1-10% occupancy.  At those scales the dense
// volume doesn't fit in memory or takes far longer than the sparse path.
//
// These benchmarks sweep bbox_dim x occupancy to find the crossover point.

static void
BM_Conv_SparsityLarge_CUDA_C32(benchmark::State &state) {
    if (!torch::cuda::is_available()) {
        state.SkipWithError("CUDA not available");
        return;
    }
    int bbox_dim      = static_cast<int>(state.range(0));
    int occupancy_pct = static_cast<int>(state.range(1));
    auto device       = torch::Device(torch::kCUDA, 0);

    auto grid = (occupancy_pct >= 100) ? makeDenseGrid(bbox_dim, device)
                                       : makeSparseGrid(bbox_dim, occupancy_pct, device);

    nanovdb::Coord ks(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);
    auto topo = ops::gatherScatterDefaultSparseConvTopology(*grid, *grid, ks, stride);

    int64_t N     = grid->totalVoxels();
    int64_t C     = 32;
    auto features = torch::randn({N, C}, topts(device));
    auto weights  = torch::randn({C, C, 3, 3, 3}, topts(device));

    auto out = ops::gatherScatterDefaultSparseConv(features, weights, topo);
    torch::cuda::synchronize();

    for (auto _: state) {
        auto o = ops::gatherScatterDefaultSparseConv(features, weights, topo);
        torch::cuda::synchronize();
        benchmark::DoNotOptimize(o.data_ptr<float>());
    }
    state.SetItemsProcessed(state.iterations() * N);
    state.counters["bbox"]      = static_cast<double>(bbox_dim);
    state.counters["occupancy"] = static_cast<double>(occupancy_pct);
    state.counters["voxels"]    = static_cast<double>(N);
    state.counters["pairs"]     = static_cast<double>(topo.totalPairs);
}

static void
BM_Conv_DenseRefLarge_CUDA_C32(benchmark::State &state) {
    if (!torch::cuda::is_available()) {
        state.SkipWithError("CUDA not available");
        return;
    }
    int bbox_dim = static_cast<int>(state.range(0));
    auto device  = torch::Device(torch::kCUDA, 0);
    int64_t C    = 32;

    auto input  = torch::randn({1, C, bbox_dim, bbox_dim, bbox_dim}, topts(device));
    auto kernel = torch::randn({C, C, 3, 3, 3}, topts(device));

    auto out = F::conv3d(input, kernel, F::Conv3dFuncOptions().padding(1));
    torch::cuda::synchronize();

    for (auto _: state) {
        auto o = F::conv3d(input, kernel, F::Conv3dFuncOptions().padding(1));
        torch::cuda::synchronize();
        benchmark::DoNotOptimize(o.data_ptr<float>());
    }
    int64_t N = static_cast<int64_t>(bbox_dim) * bbox_dim * bbox_dim;
    state.SetItemsProcessed(state.iterations() * N);
    state.counters["bbox"]   = static_cast<double>(bbox_dim);
    state.counters["voxels"] = static_cast<double>(N);
}

// ============================================================================
// E. Channel scaling
// ============================================================================

static void
BM_Conv_ChannelScale_CUDA(benchmark::State &state) {
    if (!torch::cuda::is_available()) {
        state.SkipWithError("CUDA not available");
        return;
    }
    int64_t C   = state.range(0);
    int dim     = 16;
    auto device = torch::Device(torch::kCUDA, 0);
    auto grid   = makeDenseGrid(dim, device);

    nanovdb::Coord ks(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);
    auto topo = ops::gatherScatterDefaultSparseConvTopology(*grid, *grid, ks, stride);

    int64_t N     = grid->totalVoxels();
    auto features = torch::randn({N, C}, topts(device));
    auto weights  = torch::randn({C, C, 3, 3, 3}, topts(device));

    auto out = ops::gatherScatterDefaultSparseConv(features, weights, topo);
    torch::cuda::synchronize();

    for (auto _: state) {
        auto o = ops::gatherScatterDefaultSparseConv(features, weights, topo);
        torch::cuda::synchronize();
        benchmark::DoNotOptimize(o.data_ptr<float>());
    }
    state.SetItemsProcessed(state.iterations() * N);
    state.counters["channels"] = static_cast<double>(C);
    state.counters["voxels"]   = static_cast<double>(N);
    state.counters["pairs"]    = static_cast<double>(topo.totalPairs);
}

static void
BM_Conv_ChannelScale_CPU(benchmark::State &state) {
    int64_t C   = state.range(0);
    int dim     = 16;
    auto device = torch::Device(torch::kCPU);
    auto grid   = makeDenseGrid(dim, device);

    nanovdb::Coord ks(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);
    auto topo = ops::gatherScatterDefaultSparseConvTopology(*grid, *grid, ks, stride);

    int64_t N     = grid->totalVoxels();
    auto features = torch::randn({N, C}, topts(device));
    auto weights  = torch::randn({C, C, 3, 3, 3}, topts(device));

    auto out = ops::gatherScatterDefaultSparseConv(features, weights, topo);

    for (auto _: state) {
        auto o = ops::gatherScatterDefaultSparseConv(features, weights, topo);
        benchmark::DoNotOptimize(o.data_ptr<float>());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * N);
    state.counters["channels"] = static_cast<double>(C);
    state.counters["voxels"]   = static_cast<double>(N);
    state.counters["pairs"]    = static_cast<double>(topo.totalPairs);
}

// ============================================================================
// Register benchmarks
// ============================================================================

// clang-format off

#define CONV_GRID_SIZES \
    ->Args({8})->Args({16})->Args({24})->Args({32})->UseRealTime()

#define CONV_SPARSITY \
    ->Args({1})->Args({5})->Args({10})->Args({25})->Args({50})->Args({100})->UseRealTime()

#define CONV_CHANNELS \
    ->Args({4})->Args({16})->Args({32})->Args({64})->Args({128})->Args({256})->UseRealTime()

// Topology build
BENCHMARK(BM_Conv_TopologyBuild_CUDA)       CONV_GRID_SIZES;
BENCHMARK(BM_Conv_TopologyBuild_CPU)        CONV_GRID_SIZES;

// Forward
BENCHMARK(BM_Conv_Forward_CUDA_C32)         CONV_GRID_SIZES;
BENCHMARK(BM_Conv_Forward_CPU_C32)          CONV_GRID_SIZES;

// Backward
BENCHMARK(BM_Conv_Backward_CUDA_C32)        CONV_GRID_SIZES;
BENCHMARK(BM_Conv_Backward_CPU_C32)         CONV_GRID_SIZES;

// Dense baseline
BENCHMARK(BM_Conv_DenseBaseline_CUDA_C32)   CONV_GRID_SIZES;
BENCHMARK(BM_Conv_DenseBaseline_CPU_C32)    CONV_GRID_SIZES;

// Sparsity breakeven -- small scale (CUDA only)
BENCHMARK(BM_Conv_Sparsity_CUDA_C32)        CONV_SPARSITY;
BENCHMARK(BM_Conv_DenseRef_CUDA_C32)->UseRealTime();

// Sparsity breakeven -- large scale (the key comparison)
// bbox=128: 2M cells.  Sparse at 1-10% = 20K-200K voxels.
// bbox=256: 16.7M cells.  Sparse at 1-10% = 167K-1.67M voxels.
BENCHMARK(BM_Conv_SparsityLarge_CUDA_C32)
    ->Args({128, 1})->Args({128, 5})->Args({128, 10})->Args({128, 25})->Args({128, 50})
    ->Args({256, 1})->Args({256, 5})->Args({256, 10})->Args({256, 25})
    ->UseRealTime()->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Conv_DenseRefLarge_CUDA_C32)
    ->Args({128})->Args({256})
    ->UseRealTime()->Unit(benchmark::kMillisecond);

// Channel scaling
BENCHMARK(BM_Conv_ChannelScale_CUDA)        CONV_CHANNELS;
BENCHMARK(BM_Conv_ChannelScale_CPU)         CONV_CHANNELS;

// clang-format on

} // namespace
