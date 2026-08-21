// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Active-voxel iteration benchmarks: legacy leaf-scan (forEachVoxelCUDA, one thread per
// 512-slot leaf position regardless of occupancy) vs VBM-backed iteration
// (forEachActiveVoxelVbmCUDA, one thread per active voxel via the register-only
// VoxelBlockManager inverse-map decode).
//
// The workload is the ActiveGridCoords kernel body (decode + 3 int32 stores), the most
// launch-bound per-active-voxel op in fVDB, across leaf occupancies from dense (100%) to a
// sparse spherical shell (~a few % of leaf slots active) — occupancy is the leaf scan's cost
// driver and the VBM decode is occupancy-independent.
//
// A separate benchmark measures the one-time VbmCache build cost (paid once per grid
// lifetime; the iteration benchmarks run against a warm cache).
//

#ifdef __NVCC__
#pragma nv_diag_suppress 177
#endif

#include <fvdb/GridBatchData.h>
#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/VbmCache.h>
#include <fvdb/detail/ops/BuildGridFromIjk.h>
#include <fvdb/detail/utils/cuda/ForEachCUDA.cuh>
#include <fvdb/detail/utils/cuda/ForEachVbmCUDA.cuh>

#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

#include <benchmark/benchmark.h>

#include <cstdint>

namespace {

using namespace fvdb;

// ============================================================================
// Grid construction at controlled leaf occupancy
// ============================================================================

// Dense cube: every voxel of a dim^3 box is active (occupancy ~100%).
torch::Tensor
denseCubeIjk(int dim) {
    auto r    = torch::arange(dim, torch::kInt32);
    auto grid = torch::meshgrid({r, r, r}, "ij");
    return torch::stack({grid[0].reshape(-1), grid[1].reshape(-1), grid[2].reshape(-1)}, 1);
}

// Random subsample of a dense cube: occupancy ~= fraction.
torch::Tensor
randomIjk(int dim, double fraction, uint64_t seed) {
    auto cube     = denseCubeIjk(dim);
    auto gen      = at::detail::createCPUGenerator(seed);
    int64_t count = int64_t(double(cube.size(0)) * fraction);
    auto perm     = torch::randperm(cube.size(0), gen, torch::kInt64).slice(0, 0, count);
    return cube.index_select(0, perm);
}

// Spherical shell of ~thickness voxels: the narrow-band case, sparsest leaf occupancy.
torch::Tensor
shellIjk(int dim, double thickness) {
    auto cube   = denseCubeIjk(dim);
    auto center = double(dim - 1) / 2.0;
    auto radius = double(dim) * 0.4;
    auto d      = (cube.to(torch::kFloat64) - center).norm(2, 1) - radius;
    return cube.index({d.abs() < thickness / 2.0});
}

c10::intrusive_ptr<GridBatchData>
makeGrid(const torch::Tensor &ijk) {
    JaggedTensor jt(ijk.to(torch::Device(torch::kCUDA, 0)));
    return fvdb::detail::ops::createNanoGridFromIJK(jt, {{1.0, 1.0, 1.0}}, {{0.0, 0.0, 0.0}});
}

double
leafOccupancy(const GridBatchData &grid) {
    return double(grid.totalVoxels()) / (double(grid.totalLeaves()) * 512.0);
}

// ============================================================================
// Workload: the ActiveGridCoords kernel body (3 int32 stores per active voxel)
// ============================================================================

struct WriteCoordsFunctor {
    int32_t *out; // [totalVoxels, 3]

    // VBM path entry point (forEachActiveVoxelVbmCUDA contract)
    __device__ void
    perActiveVoxel(nanovdb::Coord const &ijk, int64_t featureIdx) const {
        out[featureIdx * 3 + 0] = ijk[0];
        out[featureIdx * 3 + 1] = ijk[1];
        out[featureIdx * 3 + 2] = ijk[2];
    }

    // Legacy leaf-scan entry point (forEachVoxelCUDA contract)
    __device__ void
    operator()(int64_t batchIdx,
               int64_t leafIdx,
               int64_t voxelIdx,
               int64_t,
               GridBatchData::Accessor acc) const {
        auto const *grid = acc.grid(batchIdx);
        auto const &leaf = grid->tree().template getFirstNode<0>()[leafIdx];
        if (leaf.isActive(voxelIdx)) {
            auto const ijk = leaf.offsetToGlobalCoord(voxelIdx);
            perActiveVoxel(ijk, acc.voxelOffset(batchIdx) + leaf.getValue(voxelIdx) - 1);
        }
    }
};

// ============================================================================
// Benchmarks
// ============================================================================

enum class GridShape { Shell };

c10::intrusive_ptr<GridBatchData>
makeShapedGrid(GridShape shape) {
    switch (shape) {
    case GridShape::Shell: return makeGrid(shellIjk(256, 3.0));
    }
    return nullptr;
}

// Random subsample of a dense box at `percent`% occupancy (100 -> fully dense). Because the
// dense box covers whole leaves, leaf occupancy tracks the subsample fraction, letting the
// sweep locate the crossover point where the VBM path stops paying off.
c10::intrusive_ptr<GridBatchData>
makeSweepGrid(int percent) {
    constexpr int kDim = 160; // ~4.1M-voxel dense box
    if (percent >= 100) {
        return makeGrid(denseCubeIjk(kDim));
    }
    return makeGrid(randomIjk(kDim, double(percent) / 100.0, /*seed=*/percent));
}

// The leaf-scan baseline launches the header-defined forEachVoxelCUDAKernel directly (the
// forEachVoxelCUDA wrapper's optional ultra-sparse path references a kernel that is not
// exported from libfvdb, so it cannot be linked from a benchmark executable).
void
leafScanIteration(const GridBatchData &grid, WriteCoordsFunctor func) {
    constexpr int kNumThreads     = 1024;
    const int64_t VOXELS_PER_LEAF = nanovdb::OnIndexTree::LeafNodeType::NUM_VALUES;
    const int64_t numBlocks =
        (grid.totalLeaves() * VOXELS_PER_LEAF + kNumThreads - 1) / kNumThreads;
    fvdb::_private::forEachVoxelCUDAKernel<kNumThreads>
        <<<numBlocks, kNumThreads, 0, c10::cuda::getCurrentCUDAStream()>>>(
            grid.deviceAccessor(), true, 1, func);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool UseVbm>
void
runIterationBenchmark(benchmark::State &state, c10::intrusive_ptr<GridBatchData> grid) {
    auto out = torch::empty({grid->totalVoxels(), 3},
                            torch::TensorOptions().dtype(torch::kInt32).device(grid->device()));
    WriteCoordsFunctor func{out.data_ptr<int32_t>()};

    // Warmup (also builds the VBM cache so the timed loop measures iteration only).
    if constexpr (UseVbm) {
        forEachActiveVoxelVbmCUDA(*grid, func);
    } else {
        leafScanIteration(*grid, func);
    }
    c10::cuda::getCurrentCUDAStream().synchronize();

    for (auto _: state) {
        if constexpr (UseVbm) {
            forEachActiveVoxelVbmCUDA(*grid, func);
        } else {
            leafScanIteration(*grid, func);
        }
        c10::cuda::getCurrentCUDAStream().synchronize();
    }
    state.counters["voxels"]         = double(grid->totalVoxels());
    state.counters["leaf_occupancy"] = leafOccupancy(*grid);
    state.SetItemsProcessed(state.iterations() * grid->totalVoxels());
}

template <GridShape Shape, bool UseVbm>
void
BM_ActiveVoxelIteration(benchmark::State &state) {
    if (!torch::cuda::is_available()) {
        state.SkipWithError("CUDA not available");
        return;
    }
    runIterationBenchmark<UseVbm>(state, makeShapedGrid(Shape));
}

// Occupancy sweep: state.range(0) is the subsample percentage of a dense box.
template <bool UseVbm>
void
BM_ActiveVoxelIterationSweep(benchmark::State &state) {
    if (!torch::cuda::is_available()) {
        state.SkipWithError("CUDA not available");
        return;
    }
    runIterationBenchmark<UseVbm>(state, makeSweepGrid(int(state.range(0))));
}

// One-time VBM build cost (a fresh cache every iteration); state.range(0) is the subsample
// percentage of a dense box.
void
BM_VbmCacheBuild(benchmark::State &state) {
    if (!torch::cuda::is_available()) {
        state.SkipWithError("CUDA not available");
        return;
    }
    auto grid = makeSweepGrid(int(state.range(0)));
    for (auto _: state) {
        fvdb::detail::VbmCache cache;
        benchmark::DoNotOptimize(cache.get(*grid, 0));
        c10::cuda::getCurrentCUDAStream().synchronize();
    }
    state.counters["voxels"] = double(grid->totalVoxels());
}

#define OCCUPANCY_SWEEP_ARGS \
    ->Arg(100)               \
        ->Arg(95)            \
        ->Arg(90)            \
        ->Arg(85)            \
        ->Arg(80)            \
        ->Arg(70)            \
        ->Arg(60)            \
        ->Arg(50)            \
        ->Arg(40)            \
        ->Arg(30)            \
        ->Arg(20)            \
        ->Arg(10)            \
        ->Arg(5)

BENCHMARK_TEMPLATE(BM_ActiveVoxelIterationSweep, false)
    ->Name("LeafScan/occupancy_pct")
    ->Unit(benchmark::kMicrosecond) OCCUPANCY_SWEEP_ARGS;
BENCHMARK_TEMPLATE(BM_ActiveVoxelIterationSweep, true)
    ->Name("Vbm/occupancy_pct")
    ->Unit(benchmark::kMicrosecond) OCCUPANCY_SWEEP_ARGS;
BENCHMARK_TEMPLATE(BM_ActiveVoxelIteration, GridShape::Shell, false)
    ->Name("LeafScan/shell")
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_ActiveVoxelIteration, GridShape::Shell, true)
    ->Name("Vbm/shell")
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_VbmCacheBuild)
    ->Name("VbmBuild/occupancy_pct")
    ->Unit(benchmark::kMicrosecond)
    ->Arg(100)
    ->Arg(50)
    ->Arg(10);

} // namespace
