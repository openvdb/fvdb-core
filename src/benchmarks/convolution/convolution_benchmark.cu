// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// convolution_benchmark.cu -- Benchmark comparing old SparseConvolutionKernelMap
// backend with new GatherScatter and GatherScatterFused backends.
//
// Three backends are compared on forward sparse convolution:
//   1. Old:   dispatchSparseConvolutionKernelMap (gather->GEMM->scatter, packed neighbor map)
//   2. GEMM:  gatherScatterSparseConv            (gather->GEMM->scatter, dense kernel map)
//   3. Fused: gatherScatterSparseConvFused        (single-kernel, no intermediates)
//
// Grid generation methods:
//   - Dense cube:    GridBatchImpl::dense() for peak throughput baseline
//   - Sphere shell:  Procedural voxels near a spherical surface for realistic sparsity
//

#include <fvdb/JaggedTensor.h>
#include <fvdb/Types.h>
#include <fvdb/detail/GridBatchImpl.h>
#include <fvdb/detail/ops/convolution/GatherScatter.h>
#include <fvdb/detail/ops/convolution/GatherScatterFused.h>
#include <fvdb/detail/ops/convolution/GroupedGemm.h>
#include <fvdb/detail/ops/convolution/backend/SparseConvolutionKernelMap.h>
#include <fvdb/detail/ops/convolution/pack_info/ConvolutionKernelMap.h>
#include <fvdb/detail/utils/Utils.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>

#include <benchmark/benchmark.h>

#include <cmath>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

using namespace fvdb;
using namespace fvdb::detail;

// =============================================================================
// Grid creation helpers
// =============================================================================

// Create a GridBatchImpl from explicit [N, 3] int32 ijk coordinates.
static c10::intrusive_ptr<GridBatchImpl>
makeGrid(torch::Tensor ijk_2d, torch::Device device) {
    auto ijk_dev = ijk_2d.to(device);
    JaggedTensor jt(ijk_dev);
    std::vector<nanovdb::Vec3d> voxel_sizes = {{1.0, 1.0, 1.0}};
    std::vector<nanovdb::Vec3d> origins     = {{0.0, 0.0, 0.0}};
    return GridBatchImpl::createFromIjk(jt, voxel_sizes, origins);
}

// Create a dense cube grid of side length `dim` (total voxels = dim^3).
static c10::intrusive_ptr<GridBatchImpl>
makeDenseGrid(int dim, torch::Device device) {
    nanovdb::Coord denseDims(dim, dim, dim);
    nanovdb::Coord ijkMin(0, 0, 0);
    std::vector<nanovdb::Vec3d> voxelSizes = {{1.0, 1.0, 1.0}};
    std::vector<nanovdb::Vec3d> origins    = {{0.0, 0.0, 0.0}};
    return GridBatchImpl::dense(1, device, denseDims, ijkMin, voxelSizes, origins, std::nullopt);
}

// Create a sphere-shell grid: voxels near the surface of a sphere of radius R.
// Produces roughly 4*pi*R^2 voxels (realistic SDF-like sparsity pattern).
static c10::intrusive_ptr<GridBatchImpl>
makeSphereShellGrid(int R, torch::Device device) {
    std::vector<int32_t> flat_coords;
    int const R_inner_sq = (R - 1) * (R - 1);
    int const R_outer_sq = R * R;
    for (int i = -R; i <= R; ++i) {
        for (int j = -R; j <= R; ++j) {
            for (int k = -R; k <= R; ++k) {
                int const r2 = i * i + j * j + k * k;
                if (r2 >= R_inner_sq && r2 <= R_outer_sq) {
                    flat_coords.push_back(i + R);
                    flat_coords.push_back(j + R);
                    flat_coords.push_back(k + R);
                }
            }
        }
    }
    int64_t N = static_cast<int64_t>(flat_coords.size()) / 3;
    if (N == 0) {
        // Degenerate case: return a single-voxel grid
        auto ijk = torch::tensor({{R, R, R}}, torch::dtype(torch::kInt32));
        return makeGrid(ijk, device);
    }
    auto ijk = torch::from_blob(flat_coords.data(), {N, 3}, torch::dtype(torch::kInt32)).clone();
    return makeGrid(ijk, device);
}

// =============================================================================
// Old backend helper: convert dense kernel_map to packed neighbor_map format
// =============================================================================
//
// Replicates the Python build_kernel_map conversion from Bindings.cpp.
//
// Input:  kmap [output_total, K] int32, where kmap[o, k] = source_idx or -1
// Output: (neighbor_map [#pairs, 2] int32, neighbor_sizes [K] int32)
//
// neighbor_map rows are (source_idx, output_idx) grouped by kernel position.
// neighbor_sizes[k] = count of active pairs for kernel position k.

struct OldBackendData {
    torch::Tensor neighbor_map;   // [#pairs, 2] int32
    torch::Tensor neighbor_sizes; // [K] int32 on CPU
};

static OldBackendData
buildNeighborMap(torch::Tensor kmap) {
    // kmap: [output_total, K] on device
    auto kmap_t = kmap.t().contiguous();               // [K, output_total]

    auto kmask   = kmap_t != -1;                       // [K, output_total]
    auto nbsizes = torch::sum(kmask, /*dim=*/-1);      // [K]
    auto nbmap   = torch::nonzero(kmask).contiguous(); // [#pairs, 2] with [k_idx, output_idx]

    if (nbmap.size(0) == 0) {
        return {nbmap.to(torch::kInt32), nbsizes.to(torch::kInt32).cpu()};
    }

    // Replace column 0 (k_idx) with the source voxel index from kmap_t
    auto indices = nbmap.index({torch::indexing::Slice(), 0}) * kmap_t.size(1) +
                   nbmap.index({torch::indexing::Slice(), 1});
    nbmap.index_put_({torch::indexing::Slice(), 0}, kmap_t.reshape({-1}).index({indices}));

    return {nbmap.to(torch::kInt32), nbsizes.to(torch::kInt32).cpu()};
}

// Build the old-style kernel map from source and target grids.
static OldBackendData
buildOldKernelMap(GridBatchImpl const &source,
                  GridBatchImpl const &target,
                  nanovdb::Coord kernel_size,
                  nanovdb::Coord stride) {
    int64_t const K = static_cast<int64_t>(kernel_size[0]) * kernel_size[1] * kernel_size[2];
    auto kmap       = torch::full({static_cast<int64_t>(target.totalVoxels()), K},
                            -1,
                            torch::TensorOptions().dtype(torch::kInt32).device(target.device()));

    Vec3iOrScalar ks_arg(kernel_size);
    Vec3iOrScalar st_arg(stride);
    FVDB_DISPATCH_KERNEL_DEVICE(target.device(), [&]() {
        ops::dispatchConvolutionKernelMap<DeviceTag>(source, target, kmap, ks_arg, st_arg);
    });

    return buildNeighborMap(kmap);
}

// =============================================================================
// CUDA availability check
// =============================================================================

static bool
cudaIsAvailable() {
    int count = 0;
    auto err  = cudaGetDeviceCount(&count);
    return err == cudaSuccess && count > 0;
}

// =============================================================================
// Correctness verification (runs once during setup, outside the timed loop)
// =============================================================================
//
// Each benchmark computes a reference output using the GatherScatter GEMM
// backend and checks that the backend under test produces close-enough results.
// If the outputs diverge beyond tolerance the benchmark is skipped with an
// error so that misleading timings are never reported.
//
// Tolerance: float32 GEMM accumulation can give small differences between
// the gather-GEMM-scatter path and the fused per-element path, so we use
// a generous rtol/atol suitable for fp32.

static constexpr double kVerifyRtol = 1e-4;
static constexpr double kVerifyAtol = 1e-4;

// Compute the GatherScatter (GEMM) reference output for a same-topology,
// stride-1 forward convolution.  This is the "gold" reference.
static torch::Tensor
computeGemmReference(GridBatchImpl const &grid,
                     torch::Tensor features,
                     torch::Tensor weights,
                     nanovdb::Coord kernel_size,
                     nanovdb::Coord stride) {
    auto topo = ops::gatherScatterSparseConvTopology(grid, grid, kernel_size, stride);
    auto ref  = ops::gatherScatterSparseConv(features, weights, topo);
    torch::cuda::synchronize();
    return ref;
}

// Returns true if outputs match within tolerance.  On mismatch, fills
// `state` with a descriptive error and returns false.
static bool
verifyAgainstReference(benchmark::State &state,
                       torch::Tensor actual,
                       torch::Tensor reference,
                       char const *backend_name) {
    auto diff    = (actual - reference).abs();
    double maxae = diff.max().cpu().item<double>();
    double refmx = reference.abs().max().cpu().item<double>();
    double relae = (refmx > 0) ? maxae / refmx : maxae;

    if (maxae > kVerifyAtol && relae > kVerifyRtol) {
        std::string msg = std::string(backend_name) + " output DIVERGES from GEMM reference: " +
                          "max_abs_err=" + std::to_string(maxae) +
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
// Args layout: {grid_dim_or_radius, C_in, C_out}
//
// For dense benchmarks, arg 0 is the cube side length (total voxels = dim^3).
// For sphere benchmarks, arg 0 is the sphere radius R.

#define CONV_BENCH_SIZES     \
    ->Args({10, 4, 4})       \
        ->Args({10, 16, 16}) \
        ->Args({10, 64, 64}) \
        ->Args({20, 4, 4})   \
        ->Args({20, 16, 16}) \
        ->Args({20, 64, 64}) \
        ->Args({40, 4, 4})   \
        ->Args({40, 16, 16}) \
        ->Args({40, 64, 64}) \
        ->UseRealTime()      \
        ->Unit(benchmark::kMillisecond)

// Sphere benchmarks use smaller radii to keep voxel counts comparable to dense.
// R=8  -> ~800 voxels,  R=16 -> ~3200 voxels,  R=32 -> ~12800 voxels
#define SPHERE_BENCH_SIZES   \
    ->Args({8, 4, 4})        \
        ->Args({8, 16, 16})  \
        ->Args({8, 64, 64})  \
        ->Args({16, 4, 4})   \
        ->Args({16, 16, 16}) \
        ->Args({16, 64, 64}) \
        ->Args({32, 4, 4})   \
        ->Args({32, 16, 16}) \
        ->Args({32, 64, 64}) \
        ->UseRealTime()      \
        ->Unit(benchmark::kMillisecond)

// =============================================================================
// Dense-grid forward convolution benchmarks
// =============================================================================

static void
BM_Conv_Dense_Old(benchmark::State &state) {
    if (!cudaIsAvailable()) {
        state.SkipWithError("CUDA not available");
        return;
    }
    torch::Device device(torch::kCUDA, 0);
    int const dim       = static_cast<int>(state.range(0));
    int64_t const C_in  = state.range(1);
    int64_t const C_out = state.range(2);

    nanovdb::Coord kernel_size(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);

    // Build grids (stride=1 -> same src and dst topology)
    auto src_grid = makeDenseGrid(dim, device);
    auto dst_grid = src_grid; // stride=1 same-topology

    int64_t const N = static_cast<int64_t>(src_grid->totalVoxels());
    int64_t const K = 27;     // 3x3x3

    // Build old-style neighbor map
    auto old_data = buildOldKernelMap(*src_grid, *dst_grid, kernel_size, stride);

    // Features and pre-reshaped weights [K, C_in, C_out]
    torch::manual_seed(42);
    auto features = torch::randn({N, C_in}, torch::dtype(torch::kFloat32).device(device));
    auto weights =
        torch::randn({C_out, C_in, 3, 3, 3}, torch::dtype(torch::kFloat32).device(device));
    auto weights_reshaped = weights.permute({2, 3, 4, 1, 0}).reshape({K, C_in, C_out}).contiguous();

    auto output            = torch::zeros({N, C_out}, torch::dtype(torch::kFloat32).device(device));
    bool const middleAccel = true; // stride=1, same grid

    // Warmup
    ops::dispatchSparseConvolutionKernelMap<torch::kCUDA>(features,
                                                          output,
                                                          weights_reshaped,
                                                          old_data.neighbor_map,
                                                          old_data.neighbor_sizes,
                                                          /*transpose=*/false,
                                                          middleAccel);
    torch::cuda::synchronize();

    // Verify: compare old output to GatherScatter GEMM reference
    auto ref = computeGemmReference(*src_grid, features, weights, kernel_size, stride);
    if (!verifyAgainstReference(state, output, ref, "Old"))
        return;

    for (auto _: state) {
        output.zero_();
        ops::dispatchSparseConvolutionKernelMap<torch::kCUDA>(features,
                                                              output,
                                                              weights_reshaped,
                                                              old_data.neighbor_map,
                                                              old_data.neighbor_sizes,
                                                              /*transpose=*/false,
                                                              middleAccel);
        torch::cuda::synchronize();
        benchmark::DoNotOptimize(output.data_ptr<float>());
    }
    state.SetItemsProcessed(state.iterations() * N);
}

static void
BM_Conv_Dense_GatherScatter(benchmark::State &state) {
    if (!cudaIsAvailable()) {
        state.SkipWithError("CUDA not available");
        return;
    }
    torch::Device device(torch::kCUDA, 0);
    int const dim       = static_cast<int>(state.range(0));
    int64_t const C_in  = state.range(1);
    int64_t const C_out = state.range(2);

    nanovdb::Coord kernel_size(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);

    auto src_grid = makeDenseGrid(dim, device);
    auto topo     = ops::gatherScatterSparseConvTopology(*src_grid, *src_grid, kernel_size, stride);

    int64_t const N = topo.feature_total_voxels;

    torch::manual_seed(42);
    auto features = torch::randn({N, C_in}, torch::dtype(torch::kFloat32).device(device));
    auto weights =
        torch::randn({C_out, C_in, 3, 3, 3}, torch::dtype(torch::kFloat32).device(device));

    // Warmup
    auto out = ops::gatherScatterSparseConv(features, weights, topo);
    torch::cuda::synchronize();

    // Verify: cross-check GEMM output against the fused path
    auto fused_ref = ops::gatherScatterSparseConvFused(
        features, weights, *src_grid, *src_grid, kernel_size, stride);
    torch::cuda::synchronize();
    if (!verifyAgainstReference(state, out, fused_ref, "GatherScatter"))
        return;

    for (auto _: state) {
        auto result = ops::gatherScatterSparseConv(features, weights, topo);
        torch::cuda::synchronize();
        benchmark::DoNotOptimize(result.data_ptr<float>());
    }
    state.SetItemsProcessed(state.iterations() * N);
}

static void
BM_Conv_Dense_GatherScatterFused(benchmark::State &state) {
    if (!cudaIsAvailable()) {
        state.SkipWithError("CUDA not available");
        return;
    }
    torch::Device device(torch::kCUDA, 0);
    int const dim       = static_cast<int>(state.range(0));
    int64_t const C_in  = state.range(1);
    int64_t const C_out = state.range(2);

    nanovdb::Coord kernel_size(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);

    auto src_grid   = makeDenseGrid(dim, device);
    int64_t const N = static_cast<int64_t>(src_grid->totalVoxels());

    torch::manual_seed(42);
    auto features = torch::randn({N, C_in}, torch::dtype(torch::kFloat32).device(device));
    auto weights =
        torch::randn({C_out, C_in, 3, 3, 3}, torch::dtype(torch::kFloat32).device(device));

    // Warmup
    auto out = ops::gatherScatterSparseConvFused(
        features, weights, *src_grid, *src_grid, kernel_size, stride);
    torch::cuda::synchronize();

    // Verify: compare fused output to GatherScatter GEMM reference
    auto ref = computeGemmReference(*src_grid, features, weights, kernel_size, stride);
    if (!verifyAgainstReference(state, out, ref, "GatherScatterFused"))
        return;

    for (auto _: state) {
        auto result = ops::gatherScatterSparseConvFused(
            features, weights, *src_grid, *src_grid, kernel_size, stride);
        torch::cuda::synchronize();
        benchmark::DoNotOptimize(result.data_ptr<float>());
    }
    state.SetItemsProcessed(state.iterations() * N);
}

// =============================================================================
// Sphere-shell forward convolution benchmarks
// =============================================================================

static void
BM_Conv_Sphere_Old(benchmark::State &state) {
    if (!cudaIsAvailable()) {
        state.SkipWithError("CUDA not available");
        return;
    }
    torch::Device device(torch::kCUDA, 0);
    int const R         = static_cast<int>(state.range(0));
    int64_t const C_in  = state.range(1);
    int64_t const C_out = state.range(2);

    nanovdb::Coord kernel_size(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);

    auto src_grid = makeSphereShellGrid(R, device);
    auto dst_grid = src_grid; // stride=1 same-topology

    int64_t const N = static_cast<int64_t>(src_grid->totalVoxels());
    int64_t const K = 27;

    auto old_data = buildOldKernelMap(*src_grid, *dst_grid, kernel_size, stride);

    torch::manual_seed(42);
    auto features = torch::randn({N, C_in}, torch::dtype(torch::kFloat32).device(device));
    auto weights =
        torch::randn({C_out, C_in, 3, 3, 3}, torch::dtype(torch::kFloat32).device(device));
    auto weights_reshaped = weights.permute({2, 3, 4, 1, 0}).reshape({K, C_in, C_out}).contiguous();

    auto output            = torch::zeros({N, C_out}, torch::dtype(torch::kFloat32).device(device));
    bool const middleAccel = true;

    // Warmup
    ops::dispatchSparseConvolutionKernelMap<torch::kCUDA>(features,
                                                          output,
                                                          weights_reshaped,
                                                          old_data.neighbor_map,
                                                          old_data.neighbor_sizes,
                                                          /*transpose=*/false,
                                                          middleAccel);
    torch::cuda::synchronize();

    // Verify: compare old output to GatherScatter GEMM reference
    auto ref = computeGemmReference(*src_grid, features, weights, kernel_size, stride);
    if (!verifyAgainstReference(state, output, ref, "Old"))
        return;

    for (auto _: state) {
        output.zero_();
        ops::dispatchSparseConvolutionKernelMap<torch::kCUDA>(features,
                                                              output,
                                                              weights_reshaped,
                                                              old_data.neighbor_map,
                                                              old_data.neighbor_sizes,
                                                              /*transpose=*/false,
                                                              middleAccel);
        torch::cuda::synchronize();
        benchmark::DoNotOptimize(output.data_ptr<float>());
    }
    state.SetItemsProcessed(state.iterations() * N);
}

static void
BM_Conv_Sphere_GatherScatter(benchmark::State &state) {
    if (!cudaIsAvailable()) {
        state.SkipWithError("CUDA not available");
        return;
    }
    torch::Device device(torch::kCUDA, 0);
    int const R         = static_cast<int>(state.range(0));
    int64_t const C_in  = state.range(1);
    int64_t const C_out = state.range(2);

    nanovdb::Coord kernel_size(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);

    auto src_grid = makeSphereShellGrid(R, device);
    auto topo     = ops::gatherScatterSparseConvTopology(*src_grid, *src_grid, kernel_size, stride);

    int64_t const N = topo.feature_total_voxels;

    torch::manual_seed(42);
    auto features = torch::randn({N, C_in}, torch::dtype(torch::kFloat32).device(device));
    auto weights =
        torch::randn({C_out, C_in, 3, 3, 3}, torch::dtype(torch::kFloat32).device(device));

    // Warmup
    auto out = ops::gatherScatterSparseConv(features, weights, topo);
    torch::cuda::synchronize();

    // Verify: cross-check GEMM output against the fused path
    auto fused_ref = ops::gatherScatterSparseConvFused(
        features, weights, *src_grid, *src_grid, kernel_size, stride);
    torch::cuda::synchronize();
    if (!verifyAgainstReference(state, out, fused_ref, "GatherScatter"))
        return;

    for (auto _: state) {
        auto result = ops::gatherScatterSparseConv(features, weights, topo);
        torch::cuda::synchronize();
        benchmark::DoNotOptimize(result.data_ptr<float>());
    }
    state.SetItemsProcessed(state.iterations() * N);
}

static void
BM_Conv_Sphere_GatherScatterFused(benchmark::State &state) {
    if (!cudaIsAvailable()) {
        state.SkipWithError("CUDA not available");
        return;
    }
    torch::Device device(torch::kCUDA, 0);
    int const R         = static_cast<int>(state.range(0));
    int64_t const C_in  = state.range(1);
    int64_t const C_out = state.range(2);

    nanovdb::Coord kernel_size(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);

    auto src_grid   = makeSphereShellGrid(R, device);
    int64_t const N = static_cast<int64_t>(src_grid->totalVoxels());

    torch::manual_seed(42);
    auto features = torch::randn({N, C_in}, torch::dtype(torch::kFloat32).device(device));
    auto weights =
        torch::randn({C_out, C_in, 3, 3, 3}, torch::dtype(torch::kFloat32).device(device));

    // Warmup
    auto out = ops::gatherScatterSparseConvFused(
        features, weights, *src_grid, *src_grid, kernel_size, stride);
    torch::cuda::synchronize();

    // Verify: compare fused output to GatherScatter GEMM reference
    auto ref = computeGemmReference(*src_grid, features, weights, kernel_size, stride);
    if (!verifyAgainstReference(state, out, ref, "GatherScatterFused"))
        return;

    for (auto _: state) {
        auto result = ops::gatherScatterSparseConvFused(
            features, weights, *src_grid, *src_grid, kernel_size, stride);
        torch::cuda::synchronize();
        benchmark::DoNotOptimize(result.data_ptr<float>());
    }
    state.SetItemsProcessed(state.iterations() * N);
}

// =============================================================================
// Topology / kernel-map construction benchmarks
// =============================================================================

static void
BM_Topology_Dense_Old(benchmark::State &state) {
    if (!cudaIsAvailable()) {
        state.SkipWithError("CUDA not available");
        return;
    }
    torch::Device device(torch::kCUDA, 0);
    int const dim = static_cast<int>(state.range(0));

    nanovdb::Coord kernel_size(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);

    auto src_grid   = makeDenseGrid(dim, device);
    int64_t const N = static_cast<int64_t>(src_grid->totalVoxels());

    // Warmup
    auto warmup = buildOldKernelMap(*src_grid, *src_grid, kernel_size, stride);
    torch::cuda::synchronize();

    for (auto _: state) {
        auto data = buildOldKernelMap(*src_grid, *src_grid, kernel_size, stride);
        torch::cuda::synchronize();
        benchmark::DoNotOptimize(data.neighbor_map.data_ptr<int32_t>());
    }
    state.SetItemsProcessed(state.iterations() * N);
}

static void
BM_Topology_Dense_GatherScatter(benchmark::State &state) {
    if (!cudaIsAvailable()) {
        state.SkipWithError("CUDA not available");
        return;
    }
    torch::Device device(torch::kCUDA, 0);
    int const dim = static_cast<int>(state.range(0));

    nanovdb::Coord kernel_size(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);

    auto src_grid   = makeDenseGrid(dim, device);
    int64_t const N = static_cast<int64_t>(src_grid->totalVoxels());

    // Warmup
    auto warmup = ops::gatherScatterSparseConvTopology(*src_grid, *src_grid, kernel_size, stride);
    torch::cuda::synchronize();

    for (auto _: state) {
        auto topo = ops::gatherScatterSparseConvTopology(*src_grid, *src_grid, kernel_size, stride);
        torch::cuda::synchronize();
        benchmark::DoNotOptimize(topo.kernel_map.data_ptr<int32_t>());
    }
    state.SetItemsProcessed(state.iterations() * N);
}

// =============================================================================
// CUTLASS grouped-GEMM forward convolution benchmarks
// =============================================================================
//
// GroupedGemm requires C_in and C_out to be multiples of 32, so we use a
// separate size set that starts at 32.

#define GROUPED_GEMM_BENCH_SIZES \
    ->Args({10, 32, 32})         \
        ->Args({10, 64, 64})     \
        ->Args({10, 128, 128})   \
        ->Args({20, 32, 32})     \
        ->Args({20, 64, 64})     \
        ->Args({20, 128, 128})   \
        ->Args({40, 32, 32})     \
        ->Args({40, 64, 64})     \
        ->Args({40, 128, 128})   \
        ->UseRealTime()          \
        ->Unit(benchmark::kMillisecond)

#define GROUPED_GEMM_SPHERE_SIZES \
    ->Args({8, 32, 32})           \
        ->Args({8, 64, 64})       \
        ->Args({8, 128, 128})     \
        ->Args({16, 32, 32})      \
        ->Args({16, 64, 64})      \
        ->Args({16, 128, 128})    \
        ->Args({32, 32, 32})      \
        ->Args({32, 64, 64})      \
        ->Args({32, 128, 128})    \
        ->UseRealTime()           \
        ->Unit(benchmark::kMillisecond)

static void
BM_Conv_Dense_GroupedGemm(benchmark::State &state) {
    if (!cudaIsAvailable()) {
        state.SkipWithError("CUDA not available");
        return;
    }
    torch::Device device(torch::kCUDA, 0);
    int const dim       = static_cast<int>(state.range(0));
    int64_t const C_in  = state.range(1);
    int64_t const C_out = state.range(2);

    nanovdb::Coord kernel_size(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);

    auto src_grid   = makeDenseGrid(dim, device);
    auto topo       = ops::groupedGemmSparseConvTopology(*src_grid, *src_grid, kernel_size, stride);
    int64_t const N = topo.feature_total_voxels;

    torch::manual_seed(42);
    auto features = torch::randn({N, C_in}, torch::dtype(torch::kFloat32).device(device));
    auto weights =
        torch::randn({C_out, C_in, 3, 3, 3}, torch::dtype(torch::kFloat32).device(device));

    // Warmup
    auto out = ops::groupedGemmSparseConv(features, weights, topo);
    torch::cuda::synchronize();

    // Verify against GatherScatter GEMM reference
    auto ref = computeGemmReference(*src_grid, features, weights, kernel_size, stride);
    if (!verifyAgainstReference(state, out, ref, "GroupedGemm"))
        return;

    for (auto _: state) {
        auto result = ops::groupedGemmSparseConv(features, weights, topo);
        torch::cuda::synchronize();
        benchmark::DoNotOptimize(result.data_ptr<float>());
    }
    state.SetItemsProcessed(state.iterations() * N);
}

static void
BM_Conv_Sphere_GroupedGemm(benchmark::State &state) {
    if (!cudaIsAvailable()) {
        state.SkipWithError("CUDA not available");
        return;
    }
    torch::Device device(torch::kCUDA, 0);
    int const R         = static_cast<int>(state.range(0));
    int64_t const C_in  = state.range(1);
    int64_t const C_out = state.range(2);

    nanovdb::Coord kernel_size(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);

    auto src_grid   = makeSphereShellGrid(R, device);
    auto topo       = ops::groupedGemmSparseConvTopology(*src_grid, *src_grid, kernel_size, stride);
    int64_t const N = topo.feature_total_voxels;

    torch::manual_seed(42);
    auto features = torch::randn({N, C_in}, torch::dtype(torch::kFloat32).device(device));
    auto weights =
        torch::randn({C_out, C_in, 3, 3, 3}, torch::dtype(torch::kFloat32).device(device));

    // Warmup
    auto out = ops::groupedGemmSparseConv(features, weights, topo);
    torch::cuda::synchronize();

    // Verify against GatherScatter GEMM reference
    auto ref = computeGemmReference(*src_grid, features, weights, kernel_size, stride);
    if (!verifyAgainstReference(state, out, ref, "GroupedGemm"))
        return;

    for (auto _: state) {
        auto result = ops::groupedGemmSparseConv(features, weights, topo);
        torch::cuda::synchronize();
        benchmark::DoNotOptimize(result.data_ptr<float>());
    }
    state.SetItemsProcessed(state.iterations() * N);
}

// Also benchmark the existing backends at C=32/64/128 for direct comparison
static void
BM_Conv_Dense_GatherScatter_LargeC(benchmark::State &state) {
    if (!cudaIsAvailable()) {
        state.SkipWithError("CUDA not available");
        return;
    }
    torch::Device device(torch::kCUDA, 0);
    int const dim       = static_cast<int>(state.range(0));
    int64_t const C_in  = state.range(1);
    int64_t const C_out = state.range(2);

    nanovdb::Coord kernel_size(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);

    auto src_grid = makeDenseGrid(dim, device);
    auto topo     = ops::gatherScatterSparseConvTopology(*src_grid, *src_grid, kernel_size, stride);
    int64_t const N = topo.feature_total_voxels;

    torch::manual_seed(42);
    auto features = torch::randn({N, C_in}, torch::dtype(torch::kFloat32).device(device));
    auto weights =
        torch::randn({C_out, C_in, 3, 3, 3}, torch::dtype(torch::kFloat32).device(device));

    auto out = ops::gatherScatterSparseConv(features, weights, topo);
    torch::cuda::synchronize();

    for (auto _: state) {
        auto result = ops::gatherScatterSparseConv(features, weights, topo);
        torch::cuda::synchronize();
        benchmark::DoNotOptimize(result.data_ptr<float>());
    }
    state.SetItemsProcessed(state.iterations() * N);
}

static void
BM_Conv_Sphere_GatherScatter_LargeC(benchmark::State &state) {
    if (!cudaIsAvailable()) {
        state.SkipWithError("CUDA not available");
        return;
    }
    torch::Device device(torch::kCUDA, 0);
    int const R         = static_cast<int>(state.range(0));
    int64_t const C_in  = state.range(1);
    int64_t const C_out = state.range(2);

    nanovdb::Coord kernel_size(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);

    auto src_grid = makeSphereShellGrid(R, device);
    auto topo     = ops::gatherScatterSparseConvTopology(*src_grid, *src_grid, kernel_size, stride);
    int64_t const N = topo.feature_total_voxels;

    torch::manual_seed(42);
    auto features = torch::randn({N, C_in}, torch::dtype(torch::kFloat32).device(device));
    auto weights =
        torch::randn({C_out, C_in, 3, 3, 3}, torch::dtype(torch::kFloat32).device(device));

    auto out = ops::gatherScatterSparseConv(features, weights, topo);
    torch::cuda::synchronize();

    for (auto _: state) {
        auto result = ops::gatherScatterSparseConv(features, weights, topo);
        torch::cuda::synchronize();
        benchmark::DoNotOptimize(result.data_ptr<float>());
    }
    state.SetItemsProcessed(state.iterations() * N);
}

// =============================================================================
// Register benchmarks
// =============================================================================

// Dense convolution benchmarks: Args = {cube_side_length, C_in, C_out}
BENCHMARK(BM_Conv_Dense_Old) CONV_BENCH_SIZES;
BENCHMARK(BM_Conv_Dense_GatherScatter) CONV_BENCH_SIZES;
BENCHMARK(BM_Conv_Dense_GatherScatterFused) CONV_BENCH_SIZES;

// Sphere-shell convolution benchmarks: Args = {radius, C_in, C_out}
BENCHMARK(BM_Conv_Sphere_Old) SPHERE_BENCH_SIZES;
BENCHMARK(BM_Conv_Sphere_GatherScatter) SPHERE_BENCH_SIZES;
BENCHMARK(BM_Conv_Sphere_GatherScatterFused) SPHERE_BENCH_SIZES;

// CUTLASS grouped-GEMM benchmarks (C must be multiples of 32)
BENCHMARK(BM_Conv_Dense_GroupedGemm) GROUPED_GEMM_BENCH_SIZES;
BENCHMARK(BM_Conv_Sphere_GroupedGemm) GROUPED_GEMM_SPHERE_SIZES;

// GatherScatter at matching large-C sizes for direct comparison
BENCHMARK(BM_Conv_Dense_GatherScatter_LargeC) GROUPED_GEMM_BENCH_SIZES;
BENCHMARK(BM_Conv_Sphere_GatherScatter_LargeC) GROUPED_GEMM_SPHERE_SIZES;

// Topology construction benchmarks: Args = {cube_side_length}
// Only need the first arg; C_in/C_out are irrelevant for topology.
BENCHMARK(BM_Topology_Dense_Old)
    ->Args({10})
    ->Args({20})
    ->Args({40})
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Topology_Dense_GatherScatter)
    ->Args({10})
    ->Args({20})
    ->Args({40})
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);
