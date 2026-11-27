// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// masked_igemm_topology_benchmark.cu -- CUDA cost of maskedIGemmSparseConvTopology by how its
// neighbor-map builders walk the row grid's active voxels:
//   path 0  leaf scan
//   path 1  VoxelBlockManager (warm cache)
// Kernels of up to 32 taps default to path 1, larger kernels to path 0.
//

#ifdef __NVCC__
#pragma nv_diag_suppress 177
#endif

#include <fvdb/GridBatchData.h>
#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/ops/BuildGridForConv.h>
#include <fvdb/detail/ops/BuildGridFromIjk.h>
#include <fvdb/detail/ops/convolution/MaskedIGemm.h>

#include <torch/torch.h>

#include <benchmark/benchmark.h>

#include <cstdint>
#include <vector>

namespace {

using namespace fvdb;
using namespace fvdb::detail;

c10::intrusive_ptr<GridBatchData>
makeGrid(torch::Tensor const &ijk) {
    JaggedTensor jt(ijk);
    std::vector<nanovdb::Vec3d> voxelSizes = {{1.0, 1.0, 1.0}};
    std::vector<nanovdb::Vec3d> origins    = {{0.0, 0.0, 0.0}};
    return ops::createNanoGridFromIJK(jt, voxelSizes, origins);
}

// Voxel coordinates of a spherical shell (|x| within 1 of radius), about 25 R^2 voxels.
torch::Tensor
shellIjk(int radius) {
    auto const r = torch::arange(-radius - 1, radius + 2, torch::kInt32);
    auto yz      = torch::stack(torch::meshgrid({r, r}, "ij"), -1).reshape({-1, 2});
    auto yz2     = (yz.to(torch::kFloat64) * yz.to(torch::kFloat64)).sum(1);
    std::vector<torch::Tensor> slabs;
    for (int x = -radius - 1; x <= radius + 1; ++x) {
        auto const d   = torch::sqrt(yz2 + double(x) * double(x)) - double(radius);
        auto const sel = yz.index({d.abs() < 1.0});
        slabs.push_back(torch::cat({torch::full({sel.size(0), 1}, x, torch::kInt32), sel}, 1));
    }
    return torch::cat(slabs);
}

// Shapes: 0 = shell (~1.0M voxels), 1/2/3 = random 128^3 box at 10/50/100%,
// 4 = small shell (~90k voxels), 5 = random 64^3 box at 25% (~66k voxels).
torch::Tensor
shapeIjk(int shape) {
    if (shape == 0)
        return shellIjk(200);
    if (shape == 4)
        return shellIjk(60);
    int64_t const dim   = shape == 5 ? 64 : 128;
    int const pct       = shape == 1 ? 10 : shape == 2 ? 50 : shape == 3 ? 100 : 25;
    int64_t const total = dim * dim * dim;
    torch::manual_seed(42);
    auto const sel = torch::randperm(total, torch::kInt64).slice(0, 0, total * pct / 100);
    return torch::stack({sel / (dim * dim), (sel / dim) % dim, sel % dim}, 1).to(torch::kInt32);
}

// Args: shape, kernel size, stride, path. Same grid in and out at stride 1 (submanifold); at
// stride > 1 the output is the conv's coarse grid and the topology also holds the reverse map.
void
BM_MaskedIGemm_TopologyBuild_CUDA(benchmark::State &state) {
    if (!torch::cuda::is_available()) {
        state.SkipWithError("CUDA not available");
        return;
    }
    int const shape = int(state.range(0));
    nanovdb::Coord const ks(int(state.range(1)));
    nanovdb::Coord const stride(int(state.range(2)));
    int const path = int(state.range(3));
    if (shape < 0 || shape > 5 || path < 0 || path > 1) {
        state.SkipWithError("shape must be 0-5 and path 0-1");
        return;
    }

    auto const device = torch::Device(torch::kCUDA, 0);
    auto const input  = makeGrid(shapeIjk(shape).to(device));
    auto const output =
        stride == nanovdb::Coord(1) ? input : ops::buildGridForConv(*input, ks, stride);

    ops::setMaskedIGemmTestOverrides({.vbmIteration = path == 1});
    auto topo = ops::maskedIGemmSparseConvTopology(*input, *output, ks, stride);
    torch::cuda::synchronize();
    for (auto _: state) {
        auto t = ops::maskedIGemmSparseConvTopology(*input, *output, ks, stride);
        torch::cuda::synchronize();
        benchmark::DoNotOptimize(t.outputTotalVoxels);
    }
    ops::setMaskedIGemmTestOverrides({});

    state.SetItemsProcessed(state.iterations() * output->totalVoxels());
    state.counters["out_voxels"] = double(output->totalVoxels());
    state.counters["leaf_occupancy"] =
        double(output->totalVoxels()) /
        (double(output->totalLeaves()) * double(nanovdb::OnIndexTree::LeafNodeType::NUM_VALUES));
}

} // namespace

// clang-format off

// Submanifold k3, small and large grids
BENCHMARK(BM_MaskedIGemm_TopologyBuild_CUDA)
    ->ArgNames({"shape", "k", "s", "path"})
    ->ArgsProduct({{0, 1, 2, 3, 4, 5}, {3}, {1}, {0, 1}})
    // Strided: forward and reverse maps
    ->ArgsProduct({{0, 1, 2, 3}, {3}, {2}, {0, 1}})
    ->ArgsProduct({{0, 1, 2, 3}, {2}, {2}, {0, 1}})
    // Large kernels: compact maps from the batched probe loop
    ->ArgsProduct({{0, 1, 2}, {5, 7}, {1}, {0, 1}})
    ->UseRealTime();

// clang-format on
