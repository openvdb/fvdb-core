// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// CutlassGroupedGemm.cu -- CUTLASS-accelerated sparse 3D convolution.
//
// GPU-only. All buffers allocated via torch tensors (RAII, stream-ordered
// via PyTorch's CUDACachingAllocator). No manual cudaMalloc/cudaFree.
//
// Pipeline (forward):
//   Phase 0 -- Topology (once per grid pair):
//     Two-pass GPU builder: count kernel -> device prefix sum -> fill kernel.
//   Phase 1 -- Compute:
//     a) Gather features into contiguous [total_pairs, Cin] fp16 buffer
//     b) CUTLASS grouped GEMM (all offsets in one launch, fp16 in, fp32 accum)
//     c) Per-offset scatter-add GEMM output into fp32 accumulator (no atomics)
//     d) Cast fp32 accumulator to fp16 output
//

// ============================================================================
// Includes
// ============================================================================

#include <fvdb/detail/ops/convolution/CutlassGroupedGemm.h>

// CUTLASS -- grouped GEMM
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"

// PyTorch
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace fvdb {
namespace detail {
namespace ops {

// ============================================================================
// CUTLASS type configuration
// ============================================================================

// fp16 A * fp16 B -> fp32 C, RowMajor everything, sm80 tensor cores.
using CutlassElementA    = cutlass::half_t;
using CutlassElementB    = cutlass::half_t;
using CutlassElementC    = float;
using CutlassAccumulator = float;

using CutlassLayoutA = cutlass::layout::RowMajor;
using CutlassLayoutB = cutlass::layout::RowMajor;
using CutlassLayoutC = cutlass::layout::RowMajor;

static constexpr int kAlignmentA = 8; // 8 x half = 16 bytes
static constexpr int kAlignmentB = 8;

using CutlassEpilogueOp = cutlass::epilogue::thread::LinearCombination<
    CutlassElementC,
    128 / cutlass::sizeof_bits<CutlassElementC>::value, // 4 floats = 16 bytes
    CutlassAccumulator,
    CutlassAccumulator>;

using CutlassGemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    CutlassElementA,
    CutlassLayoutA,
    cutlass::ComplexTransform::kNone,
    kAlignmentA,
    CutlassElementB,
    CutlassLayoutB,
    cutlass::ComplexTransform::kNone,
    kAlignmentB,
    CutlassElementC,
    CutlassLayoutC,
    CutlassAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>, // threadblock tile
    cutlass::gemm::GemmShape<64, 64, 32>,   // warp tile
    cutlass::gemm::GemmShape<16, 8, 16>,    // MMA instruction
    CutlassEpilogueOp,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    4                                       // pipeline stages
    >::GemmKernel;

using CutlassGemmGrouped = cutlass::gemm::device::GemmGrouped<CutlassGemmKernel>;

// Grad-weights variant: A = ColumnMajor (transposed feat_buf), B = RowMajor.
// [Mk, Cin] RowMajor IS [Cin, Mk] ColumnMajor in memory -- zero-copy transpose.
// GemmCoord(M=Cin, N=Cout, K=Mk), A=[Cin, Mk] ColMajor, B=[Mk, Cout] RowMajor.
using CutlassGradWGemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    CutlassElementA,
    cutlass::layout::ColumnMajor, // A is transposed: [Mk, Cin] RowMajor = [Cin, Mk] ColMajor
    cutlass::ComplexTransform::kNone,
    kAlignmentA,
    CutlassElementB,
    CutlassLayoutB, // RowMajor
    cutlass::ComplexTransform::kNone,
    kAlignmentB,
    CutlassElementC,
    CutlassLayoutC, // RowMajor
    CutlassAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>, // threadblock tile
    cutlass::gemm::GemmShape<64, 64, 32>,   // warp tile
    cutlass::gemm::GemmShape<16, 8, 16>,    // MMA instruction
    CutlassEpilogueOp,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    4 // pipeline stages
    >::GemmKernel;

using CutlassGradWGemmGrouped = cutlass::gemm::device::GemmGrouped<CutlassGradWGemmKernel>;

// ============================================================================
// CUDA kernel launch helpers
// ============================================================================

static constexpr int kBlockSize = 256;

static int
gridSizeFor(int64_t total_elements) {
    return static_cast<int>(
        std::min((total_elements + kBlockSize - 1) / kBlockSize, static_cast<int64_t>(4096)));
}

// ============================================================================
// NanoVDB leaf constants
// ============================================================================

static constexpr int64_t kVoxelsPerLeaf =
    static_cast<int64_t>(nanovdb::OnIndexTree::LeafNodeType::NUM_VALUES); // 512

// ============================================================================
// Kernel: gather fp16 features into contiguous buffer
// ============================================================================
//
// dst[i, c] = src[gather_indices[i], c]
// Cin is a multiple of 32 so consecutive threads accessing consecutive c
// values produce fully coalesced loads and stores.

__global__ void
gatherHalfKernel(at::Half const *__restrict__ src,    // [NA, Cin]
                 at::Half *__restrict__ dst,          // [TP, Cin]
                 int32_t const *__restrict__ indices, // [TP]
                 int64_t total_elements,              // TP * Cin
                 int64_t Cin) {
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total_elements;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t const pos     = idx / Cin;
        int64_t const c       = idx % Cin;
        int32_t const src_row = indices[pos];
        dst[pos * Cin + c]    = src[static_cast<int64_t>(src_row) * Cin + c];
    }
}

// ============================================================================
// Kernel: scatter-add fp32 GEMM output into fp32 accumulator
// ============================================================================
//
// For a single offset k, each output voxel index appears at most once in
// scatter_indices (one probe per output per offset), so this is collision-free
// when invoked per-k sequentially.  No atomics needed.

__global__ void
scatterAddF32Kernel(float const *__restrict__ src,       // [Mk, Cout]
                    int32_t const *__restrict__ indices, // [Mk]
                    int64_t Mk,
                    int64_t Cout,
                    float *__restrict__ dst)             // [NB, Cout]
{
    int64_t const total = Mk * Cout;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t const pos     = idx / Cout;
        int64_t const c       = idx % Cout;
        int32_t const dst_row = indices[pos];
        dst[static_cast<int64_t>(dst_row) * Cout + c] += src[pos * Cout + c];
    }
}

// ============================================================================
// Topology kernel: count active (feature, output) pairs per kernel offset
// ============================================================================
//
// Grid-stride loop over output grid leaves (NanoVDB OnIndex layout).
// For each active output voxel and each kernel offset k, probes the feature
// grid via its NanoVDB tree accessor.  On hit, atomicAdd(&counts[k], 1).
//
// Probe formula depends on direction:
//   Forward:    probe = output_ijk * stride + kernel_offset
//   Transposed: probe = (output_ijk - kernel_offset) / stride  (divisibility check)
//
// Kernel offsets are computed inline from the flat index k:
//   k2 = kernel_start[2] + k % ks2
//   k1 = kernel_start[1] + (k / ks2) % ks1
//   k0 = kernel_start[0] + k / (ks1 * ks2)
// This matches the nested-loop ordering in GatherScatter.cu (k0 outermost).

__global__ void
countPairsKernel(GridBatchImpl::Accessor output_acc,
                 GridBatchImpl::Accessor feature_acc,
                 nanovdb::Coord kernel_start,
                 nanovdb::Coord kernel_size,
                 nanovdb::Coord stride,
                 bool transposed,
                 int32_t *__restrict__ counts, // [K], zeroed
                 int64_t K,
                 int64_t total) {              // output_grid.totalLeaves() * kVoxelsPerLeaf
    int64_t const ks1 = kernel_size[1];
    int64_t const ks2 = kernel_size[2];

    for (int64_t flat_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         flat_idx < total;
         flat_idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t const voxelIdx   = flat_idx % kVoxelsPerLeaf;
        int64_t const cumLeafIdx = flat_idx / kVoxelsPerLeaf;
        auto const batchIdx      = output_acc.leafBatchIndex(cumLeafIdx);
        int64_t const leafIdx    = cumLeafIdx - output_acc.leafOffset(batchIdx);

        auto const *out_grid = output_acc.grid(batchIdx);
        auto const &leaf     = out_grid->tree().getFirstNode<0>()[leafIdx];

        if (!leaf.isActive(voxelIdx))
            continue;

        auto const ijk = leaf.offsetToGlobalCoord(voxelIdx);

        // Probe feature grid for each kernel offset
        auto const *feat_grid = feature_acc.grid(batchIdx);
        auto feat_tree_acc    = feat_grid->getAccessor();

        for (int64_t k = 0; k < K; ++k) {
            int32_t const k0 = kernel_start[0] + static_cast<int32_t>(k / (ks1 * ks2));
            int32_t const k1 = kernel_start[1] + static_cast<int32_t>((k / ks2) % ks1);
            int32_t const k2 = kernel_start[2] + static_cast<int32_t>(k % ks2);

            nanovdb::Coord probe;
            if (transposed) {
                int32_t const r0 = ijk[0] - k0;
                int32_t const r1 = ijk[1] - k1;
                int32_t const r2 = ijk[2] - k2;
                if (r0 % stride[0] != 0 || r1 % stride[1] != 0 || r2 % stride[2] != 0)
                    continue;
                probe = nanovdb::Coord(r0 / stride[0], r1 / stride[1], r2 / stride[2]);
            } else {
                probe = nanovdb::Coord(
                    ijk[0] * stride[0] + k0, ijk[1] * stride[1] + k1, ijk[2] * stride[2] + k2);
            }

            if (feat_tree_acc.isActive(probe)) {
                atomicAdd(&counts[k], 1);
            }
        }
    }
}

// ============================================================================
// Topology kernel: fill gather/scatter index arrays
// ============================================================================
//
// Same traversal as countPairsKernel.  On each hit, atomicAdd on a per-k
// write head gives the position within the segment; the global position is
// offsets_dev[k] + local_pos.

__global__ void
fillPairsKernel(GridBatchImpl::Accessor output_acc,
                GridBatchImpl::Accessor feature_acc,
                nanovdb::Coord kernel_start,
                nanovdb::Coord kernel_size,
                nanovdb::Coord stride,
                bool transposed,
                const int64_t *__restrict__ offsets_dev, // [K+1]
                int32_t *__restrict__ write_head,        // [K], zeroed
                int32_t *__restrict__ gather_indices,    // [total_pairs]
                int32_t *__restrict__ scatter_indices,   // [total_pairs]
                int64_t K,
                int64_t total) { // output_grid.totalLeaves() * kVoxelsPerLeaf
    int64_t const ks1 = kernel_size[1];
    int64_t const ks2 = kernel_size[2];

    for (int64_t flat_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         flat_idx < total;
         flat_idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t const voxelIdx   = flat_idx % kVoxelsPerLeaf;
        int64_t const cumLeafIdx = flat_idx / kVoxelsPerLeaf;
        auto const batchIdx      = output_acc.leafBatchIndex(cumLeafIdx);
        int64_t const leafIdx    = cumLeafIdx - output_acc.leafOffset(batchIdx);

        auto const *out_grid = output_acc.grid(batchIdx);
        auto const &leaf     = out_grid->tree().getFirstNode<0>()[leafIdx];

        if (!leaf.isActive(voxelIdx))
            continue;

        auto const ijk         = leaf.offsetToGlobalCoord(voxelIdx);
        int64_t const out_flat = output_acc.voxelOffset(batchIdx) + leaf.getValue(voxelIdx) - 1;

        // Probe feature grid for each kernel offset
        auto const *feat_grid   = feature_acc.grid(batchIdx);
        auto feat_tree_acc      = feat_grid->getAccessor();
        int64_t const feat_base = feature_acc.voxelOffset(batchIdx);

        for (int64_t k = 0; k < K; ++k) {
            int32_t const k0 = kernel_start[0] + static_cast<int32_t>(k / (ks1 * ks2));
            int32_t const k1 = kernel_start[1] + static_cast<int32_t>((k / ks2) % ks1);
            int32_t const k2 = kernel_start[2] + static_cast<int32_t>(k % ks2);

            nanovdb::Coord probe;
            if (transposed) {
                int32_t const r0 = ijk[0] - k0;
                int32_t const r1 = ijk[1] - k1;
                int32_t const r2 = ijk[2] - k2;
                if (r0 % stride[0] != 0 || r1 % stride[1] != 0 || r2 % stride[2] != 0)
                    continue;
                probe = nanovdb::Coord(r0 / stride[0], r1 / stride[1], r2 / stride[2]);
            } else {
                probe = nanovdb::Coord(
                    ijk[0] * stride[0] + k0, ijk[1] * stride[1] + k1, ijk[2] * stride[2] + k2);
            }

            if (feat_tree_acc.isActive(probe)) {
                int32_t const feat_flat =
                    static_cast<int32_t>(feat_base + feat_tree_acc.getValue(probe) - 1);

                int64_t const pos    = offsets_dev[k] + atomicAdd(&write_head[k], 1);
                gather_indices[pos]  = feat_flat;
                scatter_indices[pos] = static_cast<int32_t>(out_flat);
            }
        }
    }
}

// ============================================================================
// GPU topology builder -- two-pass: count + fill
// ============================================================================
//
// Produces the same CSR output as groupedGemmSparseConvTopology without the
// dense O(output_voxels * kernel_volume) intermediate.

static CutlassConvTopology
buildTopologyGpu(GridBatchImpl const &feature_grid,
                 GridBatchImpl const &output_grid,
                 nanovdb::Coord kernel_size,
                 nanovdb::Coord stride,
                 bool transposed = false) {
    c10::cuda::CUDAGuard device_guard(output_grid.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(output_grid.device().index()).stream();

    auto const device = output_grid.device();

    int64_t const ks0 = kernel_size[0];
    int64_t const ks1 = kernel_size[1];
    int64_t const ks2 = kernel_size[2];
    int64_t const K   = ks0 * ks1 * ks2;

    int64_t const feature_total = feature_grid.totalVoxels();
    int64_t const output_total  = output_grid.totalVoxels();

    // Centered kernel start (matches GatherScatter.cu convention)
    nanovdb::Coord kernel_start(static_cast<int>(std::floor(-ks0 / 2.0 + 1)),
                                static_cast<int>(std::floor(-ks1 / 2.0 + 1)),
                                static_cast<int>(std::floor(-ks2 / 2.0 + 1)));

    int64_t const total = output_grid.totalLeaves() * kVoxelsPerLeaf;

    // Handle empty grids
    if (total == 0 || K == 0) {
        auto empty_i32    = torch::empty({0}, torch::dtype(torch::kInt32).device(device));
        auto offsets_host = torch::zeros({K + 1}, torch::dtype(torch::kInt64));
        return CutlassConvTopology{
            empty_i32,
            empty_i32.clone(),
            offsets_host,
            feature_total,
            output_total,
            K,
            0,
            kernel_size,
            stride,
        };
    }

    // Device accessors (trivially-copyable POD, safe for kernel pass-by-value)
    auto output_acc  = output_grid.deviceAccessor();
    auto feature_acc = feature_grid.deviceAccessor();

    // ---- Pass 1: Count pairs per kernel offset ----
    auto counts = torch::zeros({K}, torch::dtype(torch::kInt32).device(device));

    countPairsKernel<<<gridSizeFor(total), kBlockSize, 0, stream>>>(output_acc,
                                                                    feature_acc,
                                                                    kernel_start,
                                                                    kernel_size,
                                                                    stride,
                                                                    transposed,
                                                                    counts.data_ptr<int32_t>(),
                                                                    K,
                                                                    total);

    // ---- Prefix sum on device via torch::cumsum ----
    auto counts_i64  = counts.to(torch::kInt64);
    auto offsets_dev = torch::zeros({K + 1}, torch::dtype(torch::kInt64).device(device));
    offsets_dev.slice(0, 1, K + 1).copy_(torch::cumsum(counts_i64, 0));

    // Single scalar sync to learn total_pairs (needed for output buffer sizing)
    int64_t total_pairs = offsets_dev[K].item<int64_t>();

    // Host copy of offsets (needed by cutlassConv for GEMM problem setup)
    auto offsets_host = offsets_dev.cpu();

    // ---- Pass 2: Fill gather/scatter indices ----
    auto gather_indices  = torch::empty({std::max(total_pairs, int64_t{1})},
                                       torch::dtype(torch::kInt32).device(device));
    auto scatter_indices = torch::empty({std::max(total_pairs, int64_t{1})},
                                        torch::dtype(torch::kInt32).device(device));

    if (total_pairs > 0) {
        auto write_head = torch::zeros({K}, torch::dtype(torch::kInt32).device(device));

        fillPairsKernel<<<gridSizeFor(total), kBlockSize, 0, stream>>>(
            output_acc,
            feature_acc,
            kernel_start,
            kernel_size,
            stride,
            transposed,
            offsets_dev.data_ptr<int64_t>(),
            write_head.data_ptr<int32_t>(),
            gather_indices.data_ptr<int32_t>(),
            scatter_indices.data_ptr<int32_t>(),
            K,
            total);
    }

    // Trim to exact size (no-op view if total_pairs > 0, empty if 0)
    gather_indices  = gather_indices.slice(0, 0, total_pairs);
    scatter_indices = scatter_indices.slice(0, 0, total_pairs);

    return CutlassConvTopology{
        gather_indices,
        scatter_indices,
        offsets_host,
        feature_total,
        output_total,
        K,
        total_pairs,
        kernel_size,
        stride,
    };
}

// ============================================================================
// Topology builders -- direct GPU two-pass
// ============================================================================

CutlassConvTopology
cutlassConvTopology(GridBatchImpl const &feature_grid,
                    GridBatchImpl const &output_grid,
                    nanovdb::Coord kernel_size,
                    nanovdb::Coord stride) {
    return buildTopologyGpu(feature_grid, output_grid, kernel_size, stride, /*transposed=*/false);
}

CutlassConvTopology
cutlassConvTransposeTopology(GridBatchImpl const &feature_grid,
                             GridBatchImpl const &output_grid,
                             nanovdb::Coord kernel_size,
                             nanovdb::Coord stride) {
    return buildTopologyGpu(feature_grid, output_grid, kernel_size, stride, /*transposed=*/true);
}

// ============================================================================
// Forward sparse convolution
// ============================================================================

torch::Tensor
cutlassConv(torch::Tensor features, torch::Tensor weights, CutlassConvTopology const &topo) {
    // ---- Precondition checks ----
    TORCH_CHECK(features.dim() == 2, "cutlassConv: features must be 2D");
    TORCH_CHECK(features.size(0) == topo.feature_total_voxels,
                "cutlassConv: features.size(0) mismatch");
    TORCH_CHECK(features.scalar_type() == torch::kFloat16, "cutlassConv: features must be fp16");
    TORCH_CHECK(features.is_contiguous(), "cutlassConv: features must be contiguous");
    TORCH_CHECK(features.is_cuda(), "cutlassConv: features must be on CUDA");

    TORCH_CHECK(weights.dim() == 5, "cutlassConv: weights must be 5D [Cout, Cin, k0, k1, k2]");
    TORCH_CHECK(weights.scalar_type() == torch::kFloat16, "cutlassConv: weights must be fp16");
    TORCH_CHECK(features.size(1) == weights.size(1),
                "cutlassConv: Cin mismatch between features and weights");
    TORCH_CHECK(weights.size(2) == topo.kernel_size[0] && weights.size(3) == topo.kernel_size[1] &&
                    weights.size(4) == topo.kernel_size[2],
                "cutlassConv: weights spatial dims must match topology kernel_size");
    TORCH_CHECK(features.device() == weights.device(),
                "cutlassConv: features and weights must be on same device");

    int64_t const Cin  = features.size(1);
    int64_t const Cout = weights.size(0);
    TORCH_CHECK(
        Cin > 0 && Cin % 32 == 0, "cutlassConv: Cin must be a positive multiple of 32, got ", Cin);
    TORCH_CHECK(Cout > 0 && Cout % 32 == 0,
                "cutlassConv: Cout must be a positive multiple of 32, got ",
                Cout);

    // ---- Device / stream setup ----
    auto const device = features.device();
    c10::cuda::CUDAGuard guard(device);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device.index()).stream();

    int64_t const NB = topo.output_total_voxels;
    int64_t const K  = topo.kernel_volume;
    int64_t const TP = topo.total_pairs;

    // ---- Allocate output (fp16) and accumulator (fp32) ----
    // Accumulate in fp32 for precision; cast to fp16 at the end.
    auto output_f32 = torch::zeros({NB, Cout}, torch::dtype(torch::kFloat32).device(device));
    if (NB == 0 || K == 0 || TP == 0) {
        return output_f32.to(torch::kFloat16);
    }

    // ---- Reshape weights: [Cout, Cin, k0, k1, k2] -> [K, Cin, Cout] row-major ----
    auto W = weights.permute({2, 3, 4, 1, 0}).reshape({K, Cin, Cout}).contiguous();

    // ---- Phase 1: Gather features into [TP, Cin] fp16 buffer ----
    auto buf_A = torch::empty({TP, Cin}, torch::dtype(torch::kFloat16).device(device));
    {
        int64_t const total = TP * Cin;
        gatherHalfKernel<<<gridSizeFor(total), kBlockSize, 0, stream>>>(
            features.data_ptr<at::Half>(),
            buf_A.data_ptr<at::Half>(),
            topo.gather_indices.data_ptr<int32_t>(),
            total,
            Cin);
    }

    // ---- Phase 2: CUTLASS grouped GEMM ----
    // Build per-group arrays on host, copy to device as torch tensors.
    auto off_acc = topo.offsets.accessor<int64_t, 1>();

    // Count non-empty groups
    int num_groups = 0;
    for (int64_t k = 0; k < K; ++k) {
        if (off_acc[k + 1] > off_acc[k])
            ++num_groups;
    }

    // Allocate fp32 GEMM output buffer
    auto buf_C = torch::empty({TP, Cout}, torch::dtype(torch::kFloat32).device(device));

    if (num_groups > 0) {
        // Host-side per-group problem description
        // GemmCoord is layout-compatible with int[3]  (Coord<3, int> with int idx[3])
        auto h_problem_sizes = torch::empty({num_groups, 3}, torch::kInt32);
        auto h_ptr_A         = torch::empty({num_groups}, torch::kInt64);
        auto h_ptr_B         = torch::empty({num_groups}, torch::kInt64);
        auto h_ptr_C         = torch::empty({num_groups}, torch::kInt64);
        auto h_ptr_D         = torch::empty({num_groups}, torch::kInt64);
        auto h_lda           = torch::empty({num_groups}, torch::kInt64);
        auto h_ldb           = torch::empty({num_groups}, torch::kInt64);
        auto h_ldc           = torch::empty({num_groups}, torch::kInt64);
        auto h_ldd           = torch::empty({num_groups}, torch::kInt64);

        auto ps = h_problem_sizes.accessor<int32_t, 2>();
        auto pA = h_ptr_A.accessor<int64_t, 1>();
        auto pB = h_ptr_B.accessor<int64_t, 1>();
        auto pC = h_ptr_C.accessor<int64_t, 1>();
        auto pD = h_ptr_D.accessor<int64_t, 1>();
        auto la = h_lda.accessor<int64_t, 1>();
        auto lb = h_ldb.accessor<int64_t, 1>();
        auto lc = h_ldc.accessor<int64_t, 1>();
        auto ld = h_ldd.accessor<int64_t, 1>();

        uintptr_t const base_A = reinterpret_cast<uintptr_t>(buf_A.data_ptr<at::Half>());
        uintptr_t const base_B = reinterpret_cast<uintptr_t>(W.data_ptr<at::Half>());
        uintptr_t const base_C = reinterpret_cast<uintptr_t>(buf_C.data_ptr<float>());

        int g = 0;
        for (int64_t k = 0; k < K; ++k) {
            int64_t const start = off_acc[k];
            int64_t const Mk    = off_acc[k + 1] - start;
            if (Mk == 0)
                continue;

            // GemmCoord(M, N, K) = (Mk, Cout, Cin)
            ps[g][0] = static_cast<int32_t>(Mk);
            ps[g][1] = static_cast<int32_t>(Cout);
            ps[g][2] = static_cast<int32_t>(Cin);

            // Pointers into contiguous buffers (stored as uintptr_t -> int64_t)
            pA[g] = static_cast<int64_t>(base_A +
                                         static_cast<uintptr_t>(start * Cin * sizeof(at::Half)));
            pB[g] = static_cast<int64_t>(base_B +
                                         static_cast<uintptr_t>(k * Cin * Cout * sizeof(at::Half)));
            pC[g] =
                static_cast<int64_t>(base_C + static_cast<uintptr_t>(start * Cout * sizeof(float)));
            pD[g] = pC[g]; // D = C (beta=0, in-place)

            // Leading dimensions (row-major)
            la[g] = Cin;  // A [Mk, Cin]
            lb[g] = Cout; // B [Cin, Cout]
            lc[g] = Cout; // C [Mk, Cout]
            ld[g] = Cout; // D [Mk, Cout]
            ++g;
        }

        // Copy to device
        auto d_problem_sizes = h_problem_sizes.to(device).contiguous();
        auto d_ptr_A         = h_ptr_A.to(device).contiguous();
        auto d_ptr_B         = h_ptr_B.to(device).contiguous();
        auto d_ptr_C         = h_ptr_C.to(device).contiguous();
        auto d_ptr_D         = h_ptr_D.to(device).contiguous();
        auto d_lda           = h_lda.to(device).contiguous();
        auto d_ldb           = h_ldb.to(device).contiguous();
        auto d_ldc           = h_ldc.to(device).contiguous();
        auto d_ldd           = h_ldd.to(device).contiguous();

        // CUTLASS grouped GEMM
        int threadblock_count = CutlassGemmGrouped::sufficient(
            reinterpret_cast<cutlass::gemm::GemmCoord *>(h_problem_sizes.data_ptr<int32_t>()),
            num_groups);

        typename CutlassGemmGrouped::EpilogueOutputOp::Params epilogue_params(
            /*alpha=*/1.0f, /*beta=*/0.0f);

        typename CutlassGemmGrouped::Arguments args(
            reinterpret_cast<cutlass::gemm::GemmCoord *>(d_problem_sizes.data_ptr<int32_t>()),
            num_groups,
            threadblock_count,
            epilogue_params,
            reinterpret_cast<CutlassElementA **>(d_ptr_A.data_ptr<int64_t>()),
            reinterpret_cast<CutlassElementB **>(d_ptr_B.data_ptr<int64_t>()),
            reinterpret_cast<CutlassElementC **>(d_ptr_C.data_ptr<int64_t>()),
            reinterpret_cast<CutlassElementC **>(d_ptr_D.data_ptr<int64_t>()),
            d_lda.data_ptr<int64_t>(),
            d_ldb.data_ptr<int64_t>(),
            d_ldc.data_ptr<int64_t>(),
            d_ldd.data_ptr<int64_t>(),
            // host_problem_sizes for precompute scheduling
            reinterpret_cast<cutlass::gemm::GemmCoord *>(h_problem_sizes.data_ptr<int32_t>()));

        // Workspace (allocated as torch tensor for RAII)
        size_t workspace_bytes = CutlassGemmGrouped::get_workspace_size(args);
        auto workspace = torch::empty({std::max(static_cast<int64_t>(workspace_bytes), int64_t{1})},
                                      torch::dtype(torch::kByte).device(device));

        CutlassGemmGrouped gemm_op;
        cutlass::Status status = gemm_op.initialize(args, workspace.data_ptr(), stream);
        TORCH_CHECK(status == cutlass::Status::kSuccess,
                    "cutlassConv: CUTLASS initialize failed: ",
                    cutlass::cutlassGetStatusString(status));

        status = gemm_op.run(stream);
        TORCH_CHECK(status == cutlass::Status::kSuccess,
                    "cutlassConv: CUTLASS run failed: ",
                    cutlass::cutlassGetStatusString(status));
    }

    // ---- Phase 3: Per-offset scatter-add (no atomics) ----
    // Each output voxel appears at most once per offset, so per-k sequential
    // scatter into the fp32 accumulator is collision-free.
    for (int64_t k = 0; k < K; ++k) {
        int64_t const start = off_acc[k];
        int64_t const Mk    = off_acc[k + 1] - start;
        if (Mk == 0)
            continue;

        int64_t const total = Mk * Cout;
        scatterAddF32Kernel<<<gridSizeFor(total), kBlockSize, 0, stream>>>(
            buf_C.data_ptr<float>() + start * Cout,
            topo.scatter_indices.data_ptr<int32_t>() + start,
            Mk,
            Cout,
            output_f32.data_ptr<float>());
    }

    // ---- Cast fp32 accumulator to fp16 output ----
    return output_f32.to(torch::kFloat16);
}

// ============================================================================
// Backward sparse convolution
// ============================================================================
//
// Given grad_output (gradient of loss w.r.t. forward output), computes:
//   grad_features:  for each offset k, grad_buf[k] @ W_T[k] -> scatter-add
//   grad_weights:   for each offset k, feat_buf[k].T @ grad_buf[k]
//
// Uses the same topology as the forward pass (no transposed topology needed).
// Reuses gatherHalfKernel and scatterAddF32Kernel from the forward path.
//
// grad_features uses CUTLASS grouped GEMM (same kernel config as forward,
// dimensions are M=Mk, N=Cin, K=Cout instead of M=Mk, N=Cout, K=Cin).
// grad_weights uses per-offset torch::mm_out in fp32.

std::tuple<torch::Tensor, torch::Tensor>
cutlassConvBackward(torch::Tensor grad_output,
                    torch::Tensor features,
                    torch::Tensor weights,
                    CutlassConvTopology const &topo) {
    // ---- Precondition checks ----
    TORCH_CHECK(grad_output.dim() == 2, "cutlassConvBackward: grad_output must be 2D");
    TORCH_CHECK(grad_output.size(0) == topo.output_total_voxels,
                "cutlassConvBackward: grad_output.size(0) mismatch");
    TORCH_CHECK(grad_output.scalar_type() == torch::kFloat16,
                "cutlassConvBackward: grad_output must be fp16");
    TORCH_CHECK(grad_output.is_contiguous(), "cutlassConvBackward: grad_output must be contiguous");
    TORCH_CHECK(grad_output.is_cuda(), "cutlassConvBackward: grad_output must be on CUDA");

    TORCH_CHECK(features.dim() == 2, "cutlassConvBackward: features must be 2D");
    TORCH_CHECK(features.size(0) == topo.feature_total_voxels,
                "cutlassConvBackward: features.size(0) mismatch");
    TORCH_CHECK(features.scalar_type() == torch::kFloat16,
                "cutlassConvBackward: features must be fp16");
    TORCH_CHECK(features.is_contiguous(), "cutlassConvBackward: features must be contiguous");

    TORCH_CHECK(weights.dim() == 5,
                "cutlassConvBackward: weights must be 5D [Cout, Cin, k0, k1, k2]");
    TORCH_CHECK(weights.scalar_type() == torch::kFloat16,
                "cutlassConvBackward: weights must be fp16");
    TORCH_CHECK(weights.size(2) == topo.kernel_size[0] && weights.size(3) == topo.kernel_size[1] &&
                    weights.size(4) == topo.kernel_size[2],
                "cutlassConvBackward: weights spatial dims must match topology kernel_size");

    int64_t const Cin  = weights.size(1);
    int64_t const Cout = weights.size(0);
    TORCH_CHECK(features.size(1) == Cin, "cutlassConvBackward: Cin mismatch");
    TORCH_CHECK(grad_output.size(1) == Cout, "cutlassConvBackward: Cout mismatch");
    TORCH_CHECK(Cin > 0 && Cin % 32 == 0,
                "cutlassConvBackward: Cin must be a positive multiple of 32, got ", Cin);
    TORCH_CHECK(Cout > 0 && Cout % 32 == 0,
                "cutlassConvBackward: Cout must be a positive multiple of 32, got ", Cout);

    // ---- Device / stream setup ----
    auto const device = features.device();
    c10::cuda::CUDAGuard guard(device);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device.index()).stream();

    int64_t const F  = topo.feature_total_voxels;
    int64_t const K  = topo.kernel_volume;
    int64_t const TP = topo.total_pairs;

    // ---- Allocate grad outputs ----
    auto grad_features_f32 = torch::zeros({F, Cin}, torch::dtype(torch::kFloat32).device(device));
    auto grad_W_f32        = torch::zeros({K, Cin, Cout}, torch::dtype(torch::kFloat32).device(device));

    if (F == 0 || K == 0 || TP == 0) {
        auto ks           = topo.kernel_size;
        auto grad_weights = grad_W_f32.reshape({ks[0], ks[1], ks[2], Cin, Cout})
                                .permute({4, 3, 0, 1, 2})
                                .contiguous()
                                .to(torch::kFloat16);
        return {grad_features_f32.to(torch::kFloat16), grad_weights};
    }

    // ---- Reshape weights ----
    // Forward uses W [K, Cin, Cout].  Backward needs W_T [K, Cout, Cin].
    auto W = weights.permute({2, 3, 4, 1, 0}).reshape({K, Cin, Cout}).contiguous();
    auto W_T = W.permute({0, 2, 1}).contiguous(); // [K, Cout, Cin]

    auto off_acc = topo.offsets.accessor<int64_t, 1>();

    // ---- Phase 1: Gather features and grad_output into contiguous fp16 buffers ----
    auto feat_buf = torch::empty({TP, Cin}, torch::dtype(torch::kFloat16).device(device));
    auto grad_buf = torch::empty({TP, Cout}, torch::dtype(torch::kFloat16).device(device));

    {
        int64_t const total_feat = TP * Cin;
        gatherHalfKernel<<<gridSizeFor(total_feat), kBlockSize, 0, stream>>>(
            features.data_ptr<at::Half>(),
            feat_buf.data_ptr<at::Half>(),
            topo.gather_indices.data_ptr<int32_t>(),
            total_feat,
            Cin);
    }
    {
        int64_t const total_grad = TP * Cout;
        gatherHalfKernel<<<gridSizeFor(total_grad), kBlockSize, 0, stream>>>(
            grad_output.data_ptr<at::Half>(),
            grad_buf.data_ptr<at::Half>(),
            topo.scatter_indices.data_ptr<int32_t>(),
            total_grad,
            Cout);
    }

    // ---- Phase 2: CUTLASS grouped GEMM for grad_features ----
    // grad_buf[k] @ W_T[k] -> grad_feat_buf[k]
    // A = [Mk, Cout], B = [Cout, Cin] -> C = [Mk, Cin] (fp32)
    // GemmCoord(M=Mk, N=Cin, K=Cout)

    int num_groups = 0;
    for (int64_t k = 0; k < K; ++k) {
        if (off_acc[k + 1] > off_acc[k])
            ++num_groups;
    }

    auto grad_feat_buf = torch::empty({TP, Cin}, torch::dtype(torch::kFloat32).device(device));

    if (num_groups > 0) {
        auto h_problem_sizes = torch::empty({num_groups, 3}, torch::kInt32);
        auto h_ptr_A         = torch::empty({num_groups}, torch::kInt64);
        auto h_ptr_B         = torch::empty({num_groups}, torch::kInt64);
        auto h_ptr_C         = torch::empty({num_groups}, torch::kInt64);
        auto h_ptr_D         = torch::empty({num_groups}, torch::kInt64);
        auto h_lda           = torch::empty({num_groups}, torch::kInt64);
        auto h_ldb           = torch::empty({num_groups}, torch::kInt64);
        auto h_ldc           = torch::empty({num_groups}, torch::kInt64);
        auto h_ldd           = torch::empty({num_groups}, torch::kInt64);

        auto ps = h_problem_sizes.accessor<int32_t, 2>();
        auto pA = h_ptr_A.accessor<int64_t, 1>();
        auto pB = h_ptr_B.accessor<int64_t, 1>();
        auto pC = h_ptr_C.accessor<int64_t, 1>();
        auto pD = h_ptr_D.accessor<int64_t, 1>();
        auto la = h_lda.accessor<int64_t, 1>();
        auto lb = h_ldb.accessor<int64_t, 1>();
        auto lc = h_ldc.accessor<int64_t, 1>();
        auto ld = h_ldd.accessor<int64_t, 1>();

        uintptr_t const base_A = reinterpret_cast<uintptr_t>(grad_buf.data_ptr<at::Half>());
        uintptr_t const base_B = reinterpret_cast<uintptr_t>(W_T.data_ptr<at::Half>());
        uintptr_t const base_C = reinterpret_cast<uintptr_t>(grad_feat_buf.data_ptr<float>());

        int g = 0;
        for (int64_t k = 0; k < K; ++k) {
            int64_t const start = off_acc[k];
            int64_t const Mk    = off_acc[k + 1] - start;
            if (Mk == 0)
                continue;

            // GemmCoord(M, N, K_dim) = (Mk, Cin, Cout)
            ps[g][0] = static_cast<int32_t>(Mk);
            ps[g][1] = static_cast<int32_t>(Cin);
            ps[g][2] = static_cast<int32_t>(Cout);

            // A = grad_buf segment [Mk, Cout]
            pA[g] = static_cast<int64_t>(
                base_A + static_cast<uintptr_t>(start * Cout * sizeof(at::Half)));
            // B = W_T[k] [Cout, Cin]
            pB[g] = static_cast<int64_t>(
                base_B + static_cast<uintptr_t>(k * Cout * Cin * sizeof(at::Half)));
            // C = grad_feat_buf segment [Mk, Cin]
            pC[g] = static_cast<int64_t>(
                base_C + static_cast<uintptr_t>(start * Cin * sizeof(float)));
            pD[g] = pC[g];

            la[g] = Cout; // A [Mk, Cout]
            lb[g] = Cin;  // B [Cout, Cin]
            lc[g] = Cin;  // C [Mk, Cin]
            ld[g] = Cin;  // D [Mk, Cin]
            ++g;
        }

        auto d_problem_sizes = h_problem_sizes.to(device).contiguous();
        auto d_ptr_A         = h_ptr_A.to(device).contiguous();
        auto d_ptr_B         = h_ptr_B.to(device).contiguous();
        auto d_ptr_C         = h_ptr_C.to(device).contiguous();
        auto d_ptr_D         = h_ptr_D.to(device).contiguous();
        auto d_lda           = h_lda.to(device).contiguous();
        auto d_ldb           = h_ldb.to(device).contiguous();
        auto d_ldc           = h_ldc.to(device).contiguous();
        auto d_ldd           = h_ldd.to(device).contiguous();

        int threadblock_count = CutlassGemmGrouped::sufficient(
            reinterpret_cast<cutlass::gemm::GemmCoord *>(h_problem_sizes.data_ptr<int32_t>()),
            num_groups);

        typename CutlassGemmGrouped::EpilogueOutputOp::Params epilogue_params(
            /*alpha=*/1.0f, /*beta=*/0.0f);

        typename CutlassGemmGrouped::Arguments args(
            reinterpret_cast<cutlass::gemm::GemmCoord *>(d_problem_sizes.data_ptr<int32_t>()),
            num_groups,
            threadblock_count,
            epilogue_params,
            reinterpret_cast<CutlassElementA **>(d_ptr_A.data_ptr<int64_t>()),
            reinterpret_cast<CutlassElementB **>(d_ptr_B.data_ptr<int64_t>()),
            reinterpret_cast<CutlassElementC **>(d_ptr_C.data_ptr<int64_t>()),
            reinterpret_cast<CutlassElementC **>(d_ptr_D.data_ptr<int64_t>()),
            d_lda.data_ptr<int64_t>(),
            d_ldb.data_ptr<int64_t>(),
            d_ldc.data_ptr<int64_t>(),
            d_ldd.data_ptr<int64_t>(),
            reinterpret_cast<cutlass::gemm::GemmCoord *>(h_problem_sizes.data_ptr<int32_t>()));

        size_t workspace_bytes = CutlassGemmGrouped::get_workspace_size(args);
        auto workspace = torch::empty({std::max(static_cast<int64_t>(workspace_bytes), int64_t{1})},
                                      torch::dtype(torch::kByte).device(device));

        CutlassGemmGrouped gemm_op;
        cutlass::Status status = gemm_op.initialize(args, workspace.data_ptr(), stream);
        TORCH_CHECK(status == cutlass::Status::kSuccess,
                    "cutlassConvBackward: CUTLASS initialize failed: ",
                    cutlass::cutlassGetStatusString(status));

        status = gemm_op.run(stream);
        TORCH_CHECK(status == cutlass::Status::kSuccess,
                    "cutlassConvBackward: CUTLASS run failed: ",
                    cutlass::cutlassGetStatusString(status));
    }

    // ---- Phase 3: Scatter-add grad_feat_buf into grad_features (fp32) ----
    // Each feature voxel appears at most once per offset in gather_indices
    // (injective probe), so per-k sequential scatter is collision-free.
    for (int64_t k = 0; k < K; ++k) {
        int64_t const start = off_acc[k];
        int64_t const Mk    = off_acc[k + 1] - start;
        if (Mk == 0)
            continue;

        int64_t const total = Mk * Cin;
        scatterAddF32Kernel<<<gridSizeFor(total), kBlockSize, 0, stream>>>(
            grad_feat_buf.data_ptr<float>() + start * Cin,
            topo.gather_indices.data_ptr<int32_t>() + start,
            Mk,
            Cin,
            grad_features_f32.data_ptr<float>());
    }

    // ---- Phase 4: CUTLASS grouped GEMM for grad_weights ----
    // grad_W[k] = feat_buf[k].T @ grad_buf[k]
    // A = feat_buf[k] [Mk, Cin] RowMajor = [Cin, Mk] ColumnMajor (zero-copy transpose)
    // B = grad_buf[k] [Mk, Cout] RowMajor
    // C = grad_W_f32[k] [Cin, Cout] RowMajor (fp32)
    // GemmCoord(M=Cin, N=Cout, K=Mk)

    // Count non-empty groups for grad_weights
    int num_gw_groups = 0;
    for (int64_t k = 0; k < K; ++k) {
        if (off_acc[k + 1] > off_acc[k])
            ++num_gw_groups;
    }

    if (num_gw_groups > 0) {
        auto h_gw_problem_sizes = torch::empty({num_gw_groups, 3}, torch::kInt32);
        auto h_gw_ptr_A         = torch::empty({num_gw_groups}, torch::kInt64);
        auto h_gw_ptr_B         = torch::empty({num_gw_groups}, torch::kInt64);
        auto h_gw_ptr_C         = torch::empty({num_gw_groups}, torch::kInt64);
        auto h_gw_ptr_D         = torch::empty({num_gw_groups}, torch::kInt64);
        auto h_gw_lda           = torch::empty({num_gw_groups}, torch::kInt64);
        auto h_gw_ldb           = torch::empty({num_gw_groups}, torch::kInt64);
        auto h_gw_ldc           = torch::empty({num_gw_groups}, torch::kInt64);
        auto h_gw_ldd           = torch::empty({num_gw_groups}, torch::kInt64);

        auto gw_ps = h_gw_problem_sizes.accessor<int32_t, 2>();
        auto gw_pA = h_gw_ptr_A.accessor<int64_t, 1>();
        auto gw_pB = h_gw_ptr_B.accessor<int64_t, 1>();
        auto gw_pC = h_gw_ptr_C.accessor<int64_t, 1>();
        auto gw_pD = h_gw_ptr_D.accessor<int64_t, 1>();
        auto gw_la = h_gw_lda.accessor<int64_t, 1>();
        auto gw_lb = h_gw_ldb.accessor<int64_t, 1>();
        auto gw_lc = h_gw_ldc.accessor<int64_t, 1>();
        auto gw_ld = h_gw_ldd.accessor<int64_t, 1>();

        uintptr_t const gw_base_A = reinterpret_cast<uintptr_t>(feat_buf.data_ptr<at::Half>());
        uintptr_t const gw_base_B = reinterpret_cast<uintptr_t>(grad_buf.data_ptr<at::Half>());
        uintptr_t const gw_base_C = reinterpret_cast<uintptr_t>(grad_W_f32.data_ptr<float>());

        int gw_g = 0;
        for (int64_t k = 0; k < K; ++k) {
            int64_t const start = off_acc[k];
            int64_t const Mk    = off_acc[k + 1] - start;
            if (Mk == 0)
                continue;

            // GemmCoord(M, N, K_dim) = (Cin, Cout, Mk)
            gw_ps[gw_g][0] = static_cast<int32_t>(Cin);
            gw_ps[gw_g][1] = static_cast<int32_t>(Cout);
            gw_ps[gw_g][2] = static_cast<int32_t>(Mk);

            // A = feat_buf segment [Mk, Cin] RowMajor = [Cin, Mk] ColumnMajor
            gw_pA[gw_g] = static_cast<int64_t>(
                gw_base_A + static_cast<uintptr_t>(start * Cin * sizeof(at::Half)));
            // B = grad_buf segment [Mk, Cout] RowMajor
            gw_pB[gw_g] = static_cast<int64_t>(
                gw_base_B + static_cast<uintptr_t>(start * Cout * sizeof(at::Half)));
            // C = grad_W_f32[k] [Cin, Cout] RowMajor
            gw_pC[gw_g] = static_cast<int64_t>(
                gw_base_C + static_cast<uintptr_t>(k * Cin * Cout * sizeof(float)));
            gw_pD[gw_g] = gw_pC[gw_g];

            // Leading dimensions:
            // A ColumnMajor [Cin, Mk]: lda = Cin (stride between columns)
            // B RowMajor [Mk, Cout]: ldb = Cout
            // C RowMajor [Cin, Cout]: ldc = Cout
            gw_la[gw_g] = Cin;
            gw_lb[gw_g] = Cout;
            gw_lc[gw_g] = Cout;
            gw_ld[gw_g] = Cout;
            ++gw_g;
        }

        auto d_gw_problem_sizes = h_gw_problem_sizes.to(device).contiguous();
        auto d_gw_ptr_A         = h_gw_ptr_A.to(device).contiguous();
        auto d_gw_ptr_B         = h_gw_ptr_B.to(device).contiguous();
        auto d_gw_ptr_C         = h_gw_ptr_C.to(device).contiguous();
        auto d_gw_ptr_D         = h_gw_ptr_D.to(device).contiguous();
        auto d_gw_lda           = h_gw_lda.to(device).contiguous();
        auto d_gw_ldb           = h_gw_ldb.to(device).contiguous();
        auto d_gw_ldc           = h_gw_ldc.to(device).contiguous();
        auto d_gw_ldd           = h_gw_ldd.to(device).contiguous();

        int gw_threadblock_count = CutlassGradWGemmGrouped::sufficient(
            reinterpret_cast<cutlass::gemm::GemmCoord *>(h_gw_problem_sizes.data_ptr<int32_t>()),
            num_gw_groups);

        typename CutlassGradWGemmGrouped::EpilogueOutputOp::Params gw_epilogue(
            /*alpha=*/1.0f, /*beta=*/0.0f);

        typename CutlassGradWGemmGrouped::Arguments gw_args(
            reinterpret_cast<cutlass::gemm::GemmCoord *>(d_gw_problem_sizes.data_ptr<int32_t>()),
            num_gw_groups,
            gw_threadblock_count,
            gw_epilogue,
            reinterpret_cast<CutlassElementA **>(d_gw_ptr_A.data_ptr<int64_t>()),
            reinterpret_cast<CutlassElementB **>(d_gw_ptr_B.data_ptr<int64_t>()),
            reinterpret_cast<CutlassElementC **>(d_gw_ptr_C.data_ptr<int64_t>()),
            reinterpret_cast<CutlassElementC **>(d_gw_ptr_D.data_ptr<int64_t>()),
            d_gw_lda.data_ptr<int64_t>(),
            d_gw_ldb.data_ptr<int64_t>(),
            d_gw_ldc.data_ptr<int64_t>(),
            d_gw_ldd.data_ptr<int64_t>(),
            reinterpret_cast<cutlass::gemm::GemmCoord *>(
                h_gw_problem_sizes.data_ptr<int32_t>()));

        size_t gw_workspace_bytes = CutlassGradWGemmGrouped::get_workspace_size(gw_args);
        auto gw_workspace =
            torch::empty({std::max(static_cast<int64_t>(gw_workspace_bytes), int64_t{1})},
                         torch::dtype(torch::kByte).device(device));

        CutlassGradWGemmGrouped gw_gemm_op;
        cutlass::Status gw_status = gw_gemm_op.initialize(gw_args, gw_workspace.data_ptr(), stream);
        TORCH_CHECK(gw_status == cutlass::Status::kSuccess,
                    "cutlassConvBackward: CUTLASS grad_weights initialize failed: ",
                    cutlass::cutlassGetStatusString(gw_status));

        gw_status = gw_gemm_op.run(stream);
        TORCH_CHECK(gw_status == cutlass::Status::kSuccess,
                    "cutlassConvBackward: CUTLASS grad_weights run failed: ",
                    cutlass::cutlassGetStatusString(gw_status));
    }

    // ---- Reshape and cast to fp16 ----
    auto ks           = topo.kernel_size;
    auto grad_weights = grad_W_f32.reshape({ks[0], ks[1], ks[2], Cin, Cout})
                            .permute({4, 3, 0, 1, 2})
                            .contiguous()
                            .to(torch::kFloat16);

    return {grad_features_f32.to(torch::kFloat16), grad_weights};
}

// ============================================================================
// Transposed sparse convolution (forward and backward)
// ============================================================================
//
// Once the transposed topology is built, the GEMM operations are identical
// to the non-transposed case.  The topology encodes the direction; the
// compute functions are topology-agnostic.

torch::Tensor
cutlassConvTranspose(torch::Tensor features,
                     torch::Tensor weights,
                     CutlassConvTopology const &topo) {
    return cutlassConv(features, weights, topo);
}

std::tuple<torch::Tensor, torch::Tensor>
cutlassConvTransposeBackward(torch::Tensor grad_output,
                             torch::Tensor features,
                             torch::Tensor weights,
                             CutlassConvTopology const &topo) {
    return cutlassConvBackward(grad_output, features, weights, topo);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
