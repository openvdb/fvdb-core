// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// cutlass_benchmark.cu -- Benchmark comparing CUTLASS (fp16 TensorCore) vs
// GroupedGemm (fp32 cuBLAS) sparse convolution backends.
//
// Only the intersection of supported configurations is tested:
//   - GPU only (CUTLASS has no CPU path)
//   - Channel counts that are multiples of 32 (CUTLASS requirement)
//   - 3x3x3 kernel, stride 1 (dominant case in real networks)
//
// Three phases are benchmarked separately:
//   1. Topology construction (CSR index building, no GEMM)
//   2. Forward convolution   (topology pre-built, excluded from timing)
//   3. Backward convolution  (topology pre-built, grad_output pre-allocated)
//
// Each backend runs at its native precision:
//   - GroupedGemm: fp32 features/weights, fp32 output
//   - CUTLASS:     fp16 features/weights, fp16 output (fp32 accumulate)
//
// Correctness is verified before timing: CUTLASS output is compared against
// the GroupedGemm fp32 reference (within fp16 tolerance).  Any divergence
// skips the benchmark with an error so that misleading timings are never
// reported.

#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/GridBatchImpl.h>
#include <fvdb/detail/ops/convolution/CutlassGroupedGemm.h>
#include <fvdb/detail/ops/convolution/GatherScatter.h>
#include <fvdb/detail/ops/convolution/GroupedGemm.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>

using namespace fvdb;
using namespace fvdb::detail;

// =============================================================================
// Enums
// =============================================================================

enum class Backend { GroupedGemm, Cutlass };
enum class GridShape { Dense, Sphere };
enum class Phase { Topology, Forward, Backward };

// =============================================================================
// Grid creation helpers
// =============================================================================

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
    nanovdb::Coord denseDims(dim, dim, dim);
    nanovdb::Coord ijkMin(0, 0, 0);
    std::vector<nanovdb::Vec3d> voxelSizes = {{1.0, 1.0, 1.0}};
    std::vector<nanovdb::Vec3d> origins    = {{0.0, 0.0, 0.0}};
    return GridBatchImpl::dense(1, device, denseDims, ijkMin, voxelSizes, origins, std::nullopt);
}

static c10::intrusive_ptr<GridBatchImpl>
makeSphereShellGrid(int R, torch::Device device) {
    std::vector<int32_t> flat_coords;
    int const R_inner_sq = (R - 1) * (R - 1);
    int const R_outer_sq = R * R;
    for (int i = -R; i <= R; ++i)
        for (int j = -R; j <= R; ++j)
            for (int k = -R; k <= R; ++k) {
                int const r2 = i * i + j * j + k * k;
                if (r2 >= R_inner_sq && r2 <= R_outer_sq) {
                    flat_coords.push_back(i + R);
                    flat_coords.push_back(j + R);
                    flat_coords.push_back(k + R);
                }
            }
    int64_t N = static_cast<int64_t>(flat_coords.size()) / 3;
    if (N == 0) {
        auto ijk = torch::tensor({{R, R, R}}, torch::dtype(torch::kInt32));
        return makeGrid(ijk, device);
    }
    auto ijk = torch::from_blob(flat_coords.data(), {N, 3}, torch::dtype(torch::kInt32)).clone();
    return makeGrid(ijk, device);
}

static c10::intrusive_ptr<GridBatchImpl>
createGrid(GridShape shape, int dim, torch::Device device) {
    switch (shape) {
    case GridShape::Dense: return makeDenseGrid(dim, device);
    case GridShape::Sphere: return makeSphereShellGrid(dim, device);
    }
    return makeDenseGrid(dim, device);
}

// =============================================================================
// CUDA availability
// =============================================================================

static bool
cudaIsAvailable() {
    int count = 0;
    auto err  = cudaGetDeviceCount(&count);
    return err == cudaSuccess && count > 0;
}

// =============================================================================
// Correctness verification
// =============================================================================
//
// CUTLASS (fp16) output is checked against GroupedGemm (fp32) reference.
// Tolerance is generous for fp16: rtol=0.05, atol=0.05.

static constexpr double kVerifyRtol = 0.05;
static constexpr double kVerifyAtol = 0.05;

static bool
verifyOutput(benchmark::State &state,
             torch::Tensor actual_fp16,
             torch::Tensor reference_fp32,
             char const *label) {
    auto actual_f32 = actual_fp16.to(torch::kFloat32);
    auto diff       = (actual_f32 - reference_fp32).abs();
    double maxae    = diff.max().cpu().item<double>();
    double refmx    = reference_fp32.abs().max().cpu().item<double>();
    double relae    = (refmx > 0) ? maxae / refmx : maxae;

    if (maxae > kVerifyAtol && relae > kVerifyRtol) {
        std::string msg = std::string(label) +
                          " output DIVERGES: max_abs_err=" + std::to_string(maxae) +
                          " rel_err=" + std::to_string(relae);
        state.SkipWithError(msg.c_str());
        return false;
    }
    return true;
}

// =============================================================================
// Benchmark argument encoding
// =============================================================================
//
// Convolution args: {grid_dim_or_radius, C_in, C_out}
// Topology args:    {grid_dim_or_radius}

// Dense: dim in {10, 20, 40, 60, 80}, channels (mul of 32) in {32, 64, 128}
// Also include non-square: {32,64} and {64,128}
// dim=60 -> 216K voxels, dim=80 -> 512K voxels (realistic point cloud scale)
#define CUTLASS_CONV_DENSE_SIZES \
    ->Args({10, 32, 32})         \
    ->Args({10, 64, 64})         \
    ->Args({10, 128, 128})       \
    ->Args({10, 32, 64})         \
    ->Args({10, 64, 128})        \
    ->Args({20, 32, 32})         \
    ->Args({20, 64, 64})         \
    ->Args({20, 128, 128})       \
    ->Args({20, 32, 64})         \
    ->Args({20, 64, 128})        \
    ->Args({40, 32, 32})         \
    ->Args({40, 64, 64})         \
    ->Args({40, 128, 128})       \
    ->Args({40, 32, 64})         \
    ->Args({40, 64, 128})        \
    ->Args({60, 32, 32})         \
    ->Args({60, 64, 64})         \
    ->Args({60, 128, 128})       \
    ->Args({60, 32, 64})         \
    ->Args({60, 64, 128})        \
    ->Args({80, 32, 32})         \
    ->Args({80, 64, 64})         \
    ->Args({80, 128, 128})       \
    ->Args({80, 32, 64})         \
    ->Args({80, 64, 128})        \
    ->UseRealTime()              \
    ->Unit(benchmark::kMillisecond)

// Sphere: R in {8, 16, 32, 50}
// R=50 -> ~30K voxels (realistic SDF surface)
#define CUTLASS_CONV_SPHERE_SIZES \
    ->Args({8, 32, 32})           \
    ->Args({8, 64, 64})           \
    ->Args({8, 128, 128})         \
    ->Args({8, 32, 64})           \
    ->Args({8, 64, 128})          \
    ->Args({16, 32, 32})          \
    ->Args({16, 64, 64})          \
    ->Args({16, 128, 128})        \
    ->Args({16, 32, 64})          \
    ->Args({16, 64, 128})         \
    ->Args({32, 32, 32})          \
    ->Args({32, 64, 64})          \
    ->Args({32, 128, 128})        \
    ->Args({32, 32, 64})          \
    ->Args({32, 64, 128})         \
    ->Args({50, 32, 32})          \
    ->Args({50, 64, 64})          \
    ->Args({50, 128, 128})        \
    ->Args({50, 32, 64})          \
    ->Args({50, 64, 128})         \
    ->UseRealTime()               \
    ->Unit(benchmark::kMillisecond)

// Topology: single arg {dim_or_radius}
#define CUTLASS_TOPO_DENSE_SIZES \
    ->Args({10})->Args({20})->Args({40})->Args({60})->Args({80}) \
    ->UseRealTime()->Unit(benchmark::kMillisecond)

#define CUTLASS_TOPO_SPHERE_SIZES \
    ->Args({8})->Args({16})->Args({32})->Args({50}) \
    ->UseRealTime()->Unit(benchmark::kMillisecond)

// =============================================================================
// Topology benchmark
// =============================================================================

static void
BM_Topology(benchmark::State &state, Backend backend, GridShape shape) {
    if (!cudaIsAvailable()) {
        state.SkipWithError("CUDA not available");
        return;
    }

    torch::Device const device(torch::kCUDA, 0);
    int const dim = static_cast<int>(state.range(0));

    nanovdb::Coord kernel_size(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);

    auto grid       = createGrid(shape, dim, device);
    int64_t const N = static_cast<int64_t>(grid->totalVoxels());

    // Warmup
    if (backend == Backend::GroupedGemm) {
        auto warmup = ops::groupedGemmSparseConvTopology(*grid, *grid, kernel_size, stride);
        torch::cuda::synchronize();
    } else {
        auto warmup = ops::cutlassConvTopology(*grid, *grid, kernel_size, stride);
        torch::cuda::synchronize();
    }

    for (auto _ : state) {
        if (backend == Backend::GroupedGemm) {
            auto topo = ops::groupedGemmSparseConvTopology(*grid, *grid, kernel_size, stride);
            torch::cuda::synchronize();
            benchmark::DoNotOptimize(topo.gather_indices.data_ptr<int32_t>());
        } else {
            auto topo = ops::cutlassConvTopology(*grid, *grid, kernel_size, stride);
            torch::cuda::synchronize();
            benchmark::DoNotOptimize(topo.gather_indices.data_ptr<int32_t>());
        }
    }

    state.SetItemsProcessed(state.iterations() * N);
    state.counters["Voxels"] = benchmark::Counter(static_cast<double>(N));
}

// =============================================================================
// Forward benchmark
// =============================================================================

static void
BM_Forward(benchmark::State &state, Backend backend, GridShape shape) {
    if (!cudaIsAvailable()) {
        state.SkipWithError("CUDA not available");
        return;
    }

    torch::Device const device(torch::kCUDA, 0);
    int const dim       = static_cast<int>(state.range(0));
    int64_t const C_in  = state.range(1);
    int64_t const C_out = state.range(2);

    nanovdb::Coord kernel_size(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);

    auto grid       = createGrid(shape, dim, device);
    int64_t const N = static_cast<int64_t>(grid->totalVoxels());

    torch::manual_seed(42);

    if (backend == Backend::GroupedGemm) {
        auto features =
            torch::randn({N, C_in}, torch::dtype(torch::kFloat32).device(device));
        auto weights =
            torch::randn({C_out, C_in, 3, 3, 3}, torch::dtype(torch::kFloat32).device(device));
        auto topo = ops::groupedGemmSparseConvTopology(*grid, *grid, kernel_size, stride);

        // Warmup
        auto out = ops::groupedGemmSparseConv(features, weights, topo);
        torch::cuda::synchronize();

        for (auto _ : state) {
            auto result = ops::groupedGemmSparseConv(features, weights, topo);
            torch::cuda::synchronize();
            benchmark::DoNotOptimize(result.data_ptr<float>());
        }
    } else {
        auto features_f16 =
            torch::randn({N, C_in}, torch::dtype(torch::kFloat16).device(device));
        auto weights_f16 =
            torch::randn({C_out, C_in, 3, 3, 3}, torch::dtype(torch::kFloat16).device(device));
        auto topo = ops::cutlassConvTopology(*grid, *grid, kernel_size, stride);

        // Warmup
        auto out = ops::cutlassConv(features_f16, weights_f16, topo);
        torch::cuda::synchronize();

        // Verify against fp32 GroupedGemm reference
        {
            auto features_f32 = features_f16.to(torch::kFloat32);
            auto weights_f32  = weights_f16.to(torch::kFloat32);
            auto gg_topo = ops::groupedGemmSparseConvTopology(*grid, *grid, kernel_size, stride);
            auto ref     = ops::groupedGemmSparseConv(features_f32, weights_f32, gg_topo);
            torch::cuda::synchronize();
            if (!verifyOutput(state, out, ref, "Cutlass Forward"))
                return;
        }

        for (auto _ : state) {
            auto result = ops::cutlassConv(features_f16, weights_f16, topo);
            torch::cuda::synchronize();
            benchmark::DoNotOptimize(result.data_ptr<at::Half>());
        }
    }

    state.SetItemsProcessed(state.iterations() * N);
    state.counters["Voxels"]   = benchmark::Counter(static_cast<double>(N));
    state.counters["Channels"] = benchmark::Counter(static_cast<double>(C_in));
}

// =============================================================================
// Backward benchmark
// =============================================================================

static void
BM_Backward(benchmark::State &state, Backend backend, GridShape shape) {
    if (!cudaIsAvailable()) {
        state.SkipWithError("CUDA not available");
        return;
    }

    torch::Device const device(torch::kCUDA, 0);
    int const dim       = static_cast<int>(state.range(0));
    int64_t const C_in  = state.range(1);
    int64_t const C_out = state.range(2);

    nanovdb::Coord kernel_size(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);

    auto grid       = createGrid(shape, dim, device);
    int64_t const N = static_cast<int64_t>(grid->totalVoxels());

    torch::manual_seed(42);

    if (backend == Backend::GroupedGemm) {
        auto features =
            torch::randn({N, C_in}, torch::dtype(torch::kFloat32).device(device));
        auto weights =
            torch::randn({C_out, C_in, 3, 3, 3}, torch::dtype(torch::kFloat32).device(device));
        auto grad_output =
            torch::randn({N, C_out}, torch::dtype(torch::kFloat32).device(device));
        auto topo = ops::groupedGemmSparseConvTopology(*grid, *grid, kernel_size, stride);

        // Warmup
        auto [gf, gw] = ops::groupedGemmSparseConvBackward(grad_output, features, weights, topo);
        torch::cuda::synchronize();

        for (auto _ : state) {
            auto [grad_feat, grad_w] =
                ops::groupedGemmSparseConvBackward(grad_output, features, weights, topo);
            torch::cuda::synchronize();
            benchmark::DoNotOptimize(grad_feat.data_ptr<float>());
        }
    } else {
        auto features_f16 =
            torch::randn({N, C_in}, torch::dtype(torch::kFloat16).device(device));
        auto weights_f16 =
            torch::randn({C_out, C_in, 3, 3, 3}, torch::dtype(torch::kFloat16).device(device));
        auto grad_output_f16 =
            torch::randn({N, C_out}, torch::dtype(torch::kFloat16).device(device));
        auto topo = ops::cutlassConvTopology(*grid, *grid, kernel_size, stride);

        // Warmup
        auto [gf, gw] =
            ops::cutlassConvBackward(grad_output_f16, features_f16, weights_f16, topo);
        torch::cuda::synchronize();

        // Verify grad_features against fp32 reference
        {
            auto features_f32    = features_f16.to(torch::kFloat32);
            auto weights_f32     = weights_f16.to(torch::kFloat32);
            auto grad_output_f32 = grad_output_f16.to(torch::kFloat32);
            auto gg_topo = ops::groupedGemmSparseConvTopology(*grid, *grid, kernel_size, stride);
            auto [ref_gf, ref_gw] =
                ops::groupedGemmSparseConvBackward(grad_output_f32, features_f32, weights_f32, gg_topo);
            torch::cuda::synchronize();
            if (!verifyOutput(state, gf, ref_gf, "Cutlass Backward grad_features"))
                return;
        }

        for (auto _ : state) {
            auto [grad_feat, grad_w] =
                ops::cutlassConvBackward(grad_output_f16, features_f16, weights_f16, topo);
            torch::cuda::synchronize();
            benchmark::DoNotOptimize(grad_feat.data_ptr<at::Half>());
        }
    }

    state.SetItemsProcessed(state.iterations() * N);
    state.counters["Voxels"]   = benchmark::Counter(static_cast<double>(N));
    state.counters["Channels"] = benchmark::Counter(static_cast<double>(C_in));
}

// =============================================================================
// Wrapper definitions and registration
// =============================================================================

// clang-format off

// --- Topology ---
static void BM_Cutlass_Topo_Dense_GroupedGemm(benchmark::State &s)  { BM_Topology(s, Backend::GroupedGemm, GridShape::Dense); }
static void BM_Cutlass_Topo_Dense_Cutlass(benchmark::State &s)      { BM_Topology(s, Backend::Cutlass, GridShape::Dense); }
static void BM_Cutlass_Topo_Sphere_GroupedGemm(benchmark::State &s)  { BM_Topology(s, Backend::GroupedGemm, GridShape::Sphere); }
static void BM_Cutlass_Topo_Sphere_Cutlass(benchmark::State &s)      { BM_Topology(s, Backend::Cutlass, GridShape::Sphere); }

// --- Forward Dense ---
static void BM_Cutlass_Fwd_Dense_GroupedGemm(benchmark::State &s)  { BM_Forward(s, Backend::GroupedGemm, GridShape::Dense); }
static void BM_Cutlass_Fwd_Dense_Cutlass(benchmark::State &s)      { BM_Forward(s, Backend::Cutlass, GridShape::Dense); }

// --- Forward Sphere ---
static void BM_Cutlass_Fwd_Sphere_GroupedGemm(benchmark::State &s)  { BM_Forward(s, Backend::GroupedGemm, GridShape::Sphere); }
static void BM_Cutlass_Fwd_Sphere_Cutlass(benchmark::State &s)      { BM_Forward(s, Backend::Cutlass, GridShape::Sphere); }

// --- Backward Dense ---
static void BM_Cutlass_Bwd_Dense_GroupedGemm(benchmark::State &s)  { BM_Backward(s, Backend::GroupedGemm, GridShape::Dense); }
static void BM_Cutlass_Bwd_Dense_Cutlass(benchmark::State &s)      { BM_Backward(s, Backend::Cutlass, GridShape::Dense); }

// --- Backward Sphere ---
static void BM_Cutlass_Bwd_Sphere_GroupedGemm(benchmark::State &s)  { BM_Backward(s, Backend::GroupedGemm, GridShape::Sphere); }
static void BM_Cutlass_Bwd_Sphere_Cutlass(benchmark::State &s)      { BM_Backward(s, Backend::Cutlass, GridShape::Sphere); }

// clang-format on

// --- Register: Topology ---
BENCHMARK(BM_Cutlass_Topo_Dense_GroupedGemm) CUTLASS_TOPO_DENSE_SIZES;
BENCHMARK(BM_Cutlass_Topo_Dense_Cutlass) CUTLASS_TOPO_DENSE_SIZES;
BENCHMARK(BM_Cutlass_Topo_Sphere_GroupedGemm) CUTLASS_TOPO_SPHERE_SIZES;
BENCHMARK(BM_Cutlass_Topo_Sphere_Cutlass) CUTLASS_TOPO_SPHERE_SIZES;

// --- Register: Forward ---
BENCHMARK(BM_Cutlass_Fwd_Dense_GroupedGemm) CUTLASS_CONV_DENSE_SIZES;
BENCHMARK(BM_Cutlass_Fwd_Dense_Cutlass) CUTLASS_CONV_DENSE_SIZES;
BENCHMARK(BM_Cutlass_Fwd_Sphere_GroupedGemm) CUTLASS_CONV_SPHERE_SIZES;
BENCHMARK(BM_Cutlass_Fwd_Sphere_Cutlass) CUTLASS_CONV_SPHERE_SIZES;

// --- Register: Backward ---
BENCHMARK(BM_Cutlass_Bwd_Dense_GroupedGemm) CUTLASS_CONV_DENSE_SIZES;
BENCHMARK(BM_Cutlass_Bwd_Dense_Cutlass) CUTLASS_CONV_DENSE_SIZES;
BENCHMARK(BM_Cutlass_Bwd_Sphere_GroupedGemm) CUTLASS_CONV_SPHERE_SIZES;
BENCHMARK(BM_Cutlass_Bwd_Sphere_Cutlass) CUTLASS_CONV_SPHERE_SIZES;
