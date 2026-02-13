// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// convolution_benchmark.cu -- Benchmark comparing sparse convolution backends.
//
// Four backends are compared on forward sparse convolution:
//   1. Old:                dispatchSparseConvolutionKernelMap (packed neighbor map)
//   2. GatherScatter:      gatherScatterSparseConv           (dense kernel map, per-offset GEMM)
//   3. GatherScatterFused: gatherScatterSparseConvFused      (single-kernel, no intermediates)
//   4. GroupedGemm:        groupedGemmSparseConv             (compacted CSR, per-offset GEMM)
//
// Grid generation methods:
//   - Dense cube:   GridBatchImpl::dense() for peak throughput baseline
//   - Sphere shell: Procedural voxels near a spherical surface for realistic sparsity
//
// Every backend is benchmarked on both CUDA and CPU, across a range of grid
// sizes and channel counts.  All benchmarks emit Voxels and Channels counters
// so that the companion visualization script can pivot on any axis.

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
// Convolution backend and grid shape enums
// =============================================================================

enum class ConvMethod { Old, GatherScatter, GatherScatterFused, GroupedGemm };
enum class GridShape { Dense, Sphere };

[[maybe_unused]] static char const *
methodStr(ConvMethod m) {
    switch (m) {
    case ConvMethod::Old: return "Old";
    case ConvMethod::GatherScatter: return "GatherScatter";
    case ConvMethod::GatherScatterFused: return "GatherScatterFused";
    case ConvMethod::GroupedGemm: return "GroupedGemm";
    }
    return "Unknown";
}

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

// Dispatch grid creation by shape enum.
static c10::intrusive_ptr<GridBatchImpl>
createGrid(GridShape shape, int dim, torch::Device device) {
    switch (shape) {
    case GridShape::Dense: return makeDenseGrid(dim, device);
    case GridShape::Sphere: return makeSphereShellGrid(dim, device);
    }
    return makeDenseGrid(dim, device); // unreachable
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

// CUDA dense convolution: dim in {10,20,40}, C in {4,16,64,128}
#define CONV_BENCH_SIZES       \
    ->Args({10, 4, 4})         \
        ->Args({10, 16, 16})   \
        ->Args({10, 64, 64})   \
        ->Args({10, 128, 128}) \
        ->Args({20, 4, 4})     \
        ->Args({20, 16, 16})   \
        ->Args({20, 64, 64})   \
        ->Args({20, 128, 128}) \
        ->Args({40, 4, 4})     \
        ->Args({40, 16, 16})   \
        ->Args({40, 64, 64})   \
        ->Args({40, 128, 128}) \
        ->UseRealTime()        \
        ->Unit(benchmark::kMillisecond)

// CUDA sphere convolution: R in {8,16,32}, C in {4,16,64,128}
#define SPHERE_BENCH_SIZES     \
    ->Args({8, 4, 4})          \
        ->Args({8, 16, 16})    \
        ->Args({8, 64, 64})    \
        ->Args({8, 128, 128})  \
        ->Args({16, 4, 4})     \
        ->Args({16, 16, 16})   \
        ->Args({16, 64, 64})   \
        ->Args({16, 128, 128}) \
        ->Args({32, 4, 4})     \
        ->Args({32, 16, 16})   \
        ->Args({32, 64, 64})   \
        ->Args({32, 128, 128}) \
        ->UseRealTime()        \
        ->Unit(benchmark::kMillisecond)

// CPU dense convolution: dim in {5,10,15}, C in {4,16,32}
#define CPU_CONV_BENCH_SIZES \
    ->Args({5, 4, 4})        \
        ->Args({5, 16, 16})  \
        ->Args({5, 32, 32})  \
        ->Args({10, 4, 4})   \
        ->Args({10, 16, 16}) \
        ->Args({10, 32, 32}) \
        ->Args({15, 4, 4})   \
        ->Args({15, 16, 16}) \
        ->Args({15, 32, 32}) \
        ->UseRealTime()      \
        ->Unit(benchmark::kMillisecond)

// CPU sphere convolution: R in {4,8,12}, C in {4,16,32}
#define CPU_SPHERE_BENCH_SIZES \
    ->Args({4, 4, 4})          \
        ->Args({4, 16, 16})    \
        ->Args({4, 32, 32})    \
        ->Args({8, 4, 4})      \
        ->Args({8, 16, 16})    \
        ->Args({8, 32, 32})    \
        ->Args({12, 4, 4})     \
        ->Args({12, 16, 16})   \
        ->Args({12, 32, 32})   \
        ->UseRealTime()        \
        ->Unit(benchmark::kMillisecond)

// CUDA topology construction: single arg {dim_or_radius}
#define DENSE_TOPO_SIZES \
    ->Args({10})->Args({20})->Args({40})->UseRealTime()->Unit(benchmark::kMillisecond)

#define SPHERE_TOPO_SIZES \
    ->Args({8})->Args({16})->Args({32})->UseRealTime()->Unit(benchmark::kMillisecond)

// =============================================================================
// Unified convolution benchmark implementation
// =============================================================================
//
// Templated on device type so that the Old backend dispatch and CUDA syncs
// resolve at compile time.  Each ConvMethod branch handles its own topology
// construction, warmup, verification, and timed loop.

template <c10::DeviceType Dev>
static void
BM_Conv_Impl(benchmark::State &state, ConvMethod method, GridShape shape) {
    constexpr bool isCuda = (Dev == c10::DeviceType::CUDA);

    if constexpr (isCuda) {
        if (!cudaIsAvailable()) {
            state.SkipWithError("CUDA not available");
            return;
        }
    }

    torch::Device const device =
        isCuda ? torch::Device(torch::kCUDA, 0) : torch::Device(torch::kCPU);

    int const dim       = static_cast<int>(state.range(0));
    int64_t const C_in  = state.range(1);
    int64_t const C_out = state.range(2);
    int64_t const K     = 27; // 3x3x3

    nanovdb::Coord kernel_size(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);

    auto grid       = createGrid(shape, dim, device);
    int64_t const N = static_cast<int64_t>(grid->totalVoxels());

    torch::manual_seed(42);
    auto features = torch::randn({N, C_in}, torch::dtype(torch::kFloat32).device(device));
    auto weights =
        torch::randn({C_out, C_in, 3, 3, 3}, torch::dtype(torch::kFloat32).device(device));

    // ----- Old (SparseConvolutionKernelMap) -----
    if (method == ConvMethod::Old) {
        auto old_data = buildOldKernelMap(*grid, *grid, kernel_size, stride);
        auto wt       = weights.permute({2, 3, 4, 1, 0}).reshape({K, C_in, C_out}).contiguous();
        auto output   = torch::zeros({N, C_out}, torch::dtype(torch::kFloat32).device(device));

        // Warmup
        ops::dispatchSparseConvolutionKernelMap<Dev>(features,
                                                     output,
                                                     wt,
                                                     old_data.neighbor_map,
                                                     old_data.neighbor_sizes,
                                                     /*transpose=*/false,
                                                     /*middleAccel=*/true);
        if constexpr (isCuda) {
            torch::cuda::synchronize();
        }

        // Verify against GatherScatter GEMM reference
        if constexpr (isCuda) {
            auto ref = computeGemmReference(*grid, features, weights, kernel_size, stride);
            if (!verifyAgainstReference(state, output, ref, "Old"))
                return;
        }

        for (auto _: state) {
            output.zero_();
            ops::dispatchSparseConvolutionKernelMap<Dev>(features,
                                                         output,
                                                         wt,
                                                         old_data.neighbor_map,
                                                         old_data.neighbor_sizes,
                                                         /*transpose=*/false,
                                                         /*middleAccel=*/true);
            if constexpr (isCuda) {
                torch::cuda::synchronize();
            }
            benchmark::DoNotOptimize(output.data_ptr<float>());
        }
    }

    // ----- GatherScatter (dense kernel-map GEMM) -----
    else if (method == ConvMethod::GatherScatter) {
        auto topo = ops::gatherScatterSparseConvTopology(*grid, *grid, kernel_size, stride);

        auto out = ops::gatherScatterSparseConv(features, weights, topo);
        if constexpr (isCuda) {
            torch::cuda::synchronize();
        }

        // Cross-check against the fused path
        if constexpr (isCuda) {
            auto fused_ref = ops::gatherScatterSparseConvFused(
                features, weights, *grid, *grid, kernel_size, stride);
            torch::cuda::synchronize();
            if (!verifyAgainstReference(state, out, fused_ref, "GatherScatter"))
                return;
        }

        for (auto _: state) {
            auto result = ops::gatherScatterSparseConv(features, weights, topo);
            if constexpr (isCuda) {
                torch::cuda::synchronize();
            }
            benchmark::DoNotOptimize(result.data_ptr<float>());
        }
    }

    // ----- GatherScatterFused (single kernel, no intermediates) -----
    else if (method == ConvMethod::GatherScatterFused) {
        auto out =
            ops::gatherScatterSparseConvFused(features, weights, *grid, *grid, kernel_size, stride);
        if constexpr (isCuda) {
            torch::cuda::synchronize();
        }

        if constexpr (isCuda) {
            auto ref = computeGemmReference(*grid, features, weights, kernel_size, stride);
            if (!verifyAgainstReference(state, out, ref, "GatherScatterFused"))
                return;
        }

        for (auto _: state) {
            auto result = ops::gatherScatterSparseConvFused(
                features, weights, *grid, *grid, kernel_size, stride);
            if constexpr (isCuda) {
                torch::cuda::synchronize();
            }
            benchmark::DoNotOptimize(result.data_ptr<float>());
        }
    }

    // ----- GroupedGemm (compacted CSR GEMM) -----
    else {
        auto topo = ops::groupedGemmSparseConvTopology(*grid, *grid, kernel_size, stride);

        auto out = ops::groupedGemmSparseConv(features, weights, topo);
        if constexpr (isCuda) {
            torch::cuda::synchronize();
        }

        if constexpr (isCuda) {
            auto ref = computeGemmReference(*grid, features, weights, kernel_size, stride);
            if (!verifyAgainstReference(state, out, ref, "GroupedGemm"))
                return;
        }

        for (auto _: state) {
            auto result = ops::groupedGemmSparseConv(features, weights, topo);
            if constexpr (isCuda) {
                torch::cuda::synchronize();
            }
            benchmark::DoNotOptimize(result.data_ptr<float>());
        }
    }

    state.SetItemsProcessed(state.iterations() * N);
    state.counters["Voxels"]   = benchmark::Counter(static_cast<double>(N));
    state.counters["Channels"] = benchmark::Counter(static_cast<double>(C_in));
}

// =============================================================================
// Unified topology benchmark implementation
// =============================================================================
//
// Measures topology / kernel-map construction time for each backend.
// GatherScatterFused has no separate topology step and is not benchmarked here.

template <c10::DeviceType Dev>
static void
BM_Topology_Impl(benchmark::State &state, ConvMethod method, GridShape shape) {
    constexpr bool isCuda = (Dev == c10::DeviceType::CUDA);

    if constexpr (isCuda) {
        if (!cudaIsAvailable()) {
            state.SkipWithError("CUDA not available");
            return;
        }
    }

    torch::Device const device =
        isCuda ? torch::Device(torch::kCUDA, 0) : torch::Device(torch::kCPU);

    int const dim = static_cast<int>(state.range(0));

    nanovdb::Coord kernel_size(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);

    auto grid       = createGrid(shape, dim, device);
    int64_t const N = static_cast<int64_t>(grid->totalVoxels());

    if (method == ConvMethod::Old) {
        // Warmup
        auto warmup = buildOldKernelMap(*grid, *grid, kernel_size, stride);
        if constexpr (isCuda) {
            torch::cuda::synchronize();
        }

        for (auto _: state) {
            auto data = buildOldKernelMap(*grid, *grid, kernel_size, stride);
            if constexpr (isCuda) {
                torch::cuda::synchronize();
            }
            benchmark::DoNotOptimize(data.neighbor_map.data_ptr<int32_t>());
        }
    } else if (method == ConvMethod::GatherScatter) {
        auto warmup = ops::gatherScatterSparseConvTopology(*grid, *grid, kernel_size, stride);
        if constexpr (isCuda) {
            torch::cuda::synchronize();
        }

        for (auto _: state) {
            auto topo = ops::gatherScatterSparseConvTopology(*grid, *grid, kernel_size, stride);
            if constexpr (isCuda) {
                torch::cuda::synchronize();
            }
            benchmark::DoNotOptimize(topo.kernel_map.data_ptr<int32_t>());
        }
    } else if (method == ConvMethod::GroupedGemm) {
        auto warmup = ops::groupedGemmSparseConvTopology(*grid, *grid, kernel_size, stride);
        if constexpr (isCuda) {
            torch::cuda::synchronize();
        }

        for (auto _: state) {
            auto topo = ops::groupedGemmSparseConvTopology(*grid, *grid, kernel_size, stride);
            if constexpr (isCuda) {
                torch::cuda::synchronize();
            }
            benchmark::DoNotOptimize(topo.gather_indices.data_ptr<int32_t>());
        }
    } else {
        state.SkipWithError("No topology benchmark for this method");
        return;
    }

    state.SetItemsProcessed(state.iterations() * N);
    state.counters["Voxels"] = benchmark::Counter(static_cast<double>(N));
}

// =============================================================================
// Wrapper function definitions
// =============================================================================
//
// Thin wrappers keep the existing BM_Conv_<Grid>_<Backend>[_CPU] name scheme
// so that benchmark names remain stable for parsing and comparison.

// clang-format off

#define DEF_CONV_CUDA(Grid, Method, method_enum, shape_enum)                                  \
    static void BM_Conv_##Grid##_##Method(benchmark::State &state) {                          \
        BM_Conv_Impl<c10::DeviceType::CUDA>(state, ConvMethod::method_enum, GridShape::shape_enum); \
    }

#define DEF_CONV_CPU(Grid, Method, method_enum, shape_enum)                                   \
    static void BM_Conv_##Grid##_##Method##_CPU(benchmark::State &state) {                    \
        BM_Conv_Impl<c10::DeviceType::CPU>(state, ConvMethod::method_enum, GridShape::shape_enum); \
    }

#define DEF_TOPO_CUDA(Grid, Method, method_enum, shape_enum)                                  \
    static void BM_Topology_##Grid##_##Method(benchmark::State &state) {                      \
        BM_Topology_Impl<c10::DeviceType::CUDA>(state, ConvMethod::method_enum, GridShape::shape_enum); \
    }

// CUDA convolution wrappers
DEF_CONV_CUDA(Dense,  Old,                  Old,                  Dense)
DEF_CONV_CUDA(Dense,  GatherScatter,        GatherScatter,        Dense)
DEF_CONV_CUDA(Dense,  GatherScatterFused,   GatherScatterFused,   Dense)
DEF_CONV_CUDA(Dense,  GroupedGemm,          GroupedGemm,          Dense)

DEF_CONV_CUDA(Sphere, Old,                  Old,                  Sphere)
DEF_CONV_CUDA(Sphere, GatherScatter,        GatherScatter,        Sphere)
DEF_CONV_CUDA(Sphere, GatherScatterFused,   GatherScatterFused,   Sphere)
DEF_CONV_CUDA(Sphere, GroupedGemm,          GroupedGemm,          Sphere)

// CPU convolution wrappers
DEF_CONV_CPU(Dense,  Old,                  Old,                  Dense)
DEF_CONV_CPU(Dense,  GatherScatter,        GatherScatter,        Dense)
DEF_CONV_CPU(Dense,  GatherScatterFused,   GatherScatterFused,   Dense)
DEF_CONV_CPU(Dense,  GroupedGemm,          GroupedGemm,          Dense)

DEF_CONV_CPU(Sphere, Old,                  Old,                  Sphere)
DEF_CONV_CPU(Sphere, GatherScatter,        GatherScatter,        Sphere)
DEF_CONV_CPU(Sphere, GatherScatterFused,   GatherScatterFused,   Sphere)
DEF_CONV_CPU(Sphere, GroupedGemm,          GroupedGemm,          Sphere)

// CUDA topology wrappers
DEF_TOPO_CUDA(Dense,  Old,            Old,            Dense)
DEF_TOPO_CUDA(Dense,  GatherScatter,  GatherScatter,  Dense)
DEF_TOPO_CUDA(Dense,  GroupedGemm,    GroupedGemm,    Dense)

DEF_TOPO_CUDA(Sphere, Old,            Old,            Sphere)
DEF_TOPO_CUDA(Sphere, GatherScatter,  GatherScatter,  Sphere)
DEF_TOPO_CUDA(Sphere, GroupedGemm,    GroupedGemm,    Sphere)

// clang-format on

// =============================================================================
// Register benchmarks
// =============================================================================

// -- CUDA Dense convolution --
BENCHMARK(BM_Conv_Dense_Old) CONV_BENCH_SIZES;
BENCHMARK(BM_Conv_Dense_GatherScatter) CONV_BENCH_SIZES;
BENCHMARK(BM_Conv_Dense_GatherScatterFused) CONV_BENCH_SIZES;
BENCHMARK(BM_Conv_Dense_GroupedGemm) CONV_BENCH_SIZES;

// -- CUDA Sphere convolution --
BENCHMARK(BM_Conv_Sphere_Old) SPHERE_BENCH_SIZES;
BENCHMARK(BM_Conv_Sphere_GatherScatter) SPHERE_BENCH_SIZES;
BENCHMARK(BM_Conv_Sphere_GatherScatterFused) SPHERE_BENCH_SIZES;
BENCHMARK(BM_Conv_Sphere_GroupedGemm) SPHERE_BENCH_SIZES;

// -- CPU Dense convolution --
BENCHMARK(BM_Conv_Dense_Old_CPU) CPU_CONV_BENCH_SIZES;
BENCHMARK(BM_Conv_Dense_GatherScatter_CPU) CPU_CONV_BENCH_SIZES;
BENCHMARK(BM_Conv_Dense_GatherScatterFused_CPU) CPU_CONV_BENCH_SIZES;
BENCHMARK(BM_Conv_Dense_GroupedGemm_CPU) CPU_CONV_BENCH_SIZES;

// -- CPU Sphere convolution --
BENCHMARK(BM_Conv_Sphere_Old_CPU) CPU_SPHERE_BENCH_SIZES;
BENCHMARK(BM_Conv_Sphere_GatherScatter_CPU) CPU_SPHERE_BENCH_SIZES;
BENCHMARK(BM_Conv_Sphere_GatherScatterFused_CPU) CPU_SPHERE_BENCH_SIZES;
BENCHMARK(BM_Conv_Sphere_GroupedGemm_CPU) CPU_SPHERE_BENCH_SIZES;

// -- CUDA Topology construction --
BENCHMARK(BM_Topology_Dense_Old) DENSE_TOPO_SIZES;
BENCHMARK(BM_Topology_Dense_GatherScatter) DENSE_TOPO_SIZES;
BENCHMARK(BM_Topology_Dense_GroupedGemm) DENSE_TOPO_SIZES;

BENCHMARK(BM_Topology_Sphere_Old) SPHERE_TOPO_SIZES;
BENCHMARK(BM_Topology_Sphere_GatherScatter) SPHERE_TOPO_SIZES;
BENCHMARK(BM_Topology_Sphere_GroupedGemm) SPHERE_TOPO_SIZES;
