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
//     c) Atomic scatter-add GEMM output into fp32 accumulator (single launch)
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
    4                                       // pipeline stages
    >::GemmKernel;

using CutlassGradWGemmGrouped = cutlass::gemm::device::GemmGrouped<CutlassGradWGemmKernel>;

// Narrow-N variants: 128x64 threadblock for GEMM N-dimension <= 64.
// Avoids wasting half or more of the N-tile compute on partial tiles.
using CutlassGemmKernelNarrow = typename cutlass::gemm::kernel::DefaultGemmGrouped<
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
    cutlass::gemm::GemmShape<128, 64, 32>, // narrower N-tile
    cutlass::gemm::GemmShape<64, 32, 32>,  // warp tile
    cutlass::gemm::GemmShape<16, 8, 16>,   // MMA instruction
    CutlassEpilogueOp,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    4                                      // pipeline stages
    >::GemmKernel;

using CutlassGemmGroupedNarrow = cutlass::gemm::device::GemmGrouped<CutlassGemmKernelNarrow>;

using CutlassGradWGemmKernelNarrow = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    CutlassElementA,
    cutlass::layout::ColumnMajor,
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
    cutlass::gemm::GemmShape<128, 64, 32>,
    cutlass::gemm::GemmShape<64, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    CutlassEpilogueOp,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    4 // pipeline stages
    >::GemmKernel;

using CutlassGradWGemmGroupedNarrow =
    cutlass::gemm::device::GemmGrouped<CutlassGradWGemmKernelNarrow>;

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
// Kernel: gather fp16 features into contiguous buffer (vectorised)
// ============================================================================
//
// dst[i, :] = src[indices[i], :]
// C is a multiple of 32 (>= 32), so C/8 >= 4 and all float4 accesses are
// naturally 16-byte aligned.  Each thread loads/stores one float4 (8 halves),
// giving 8x fewer memory transactions than the scalar version.

__global__ void
gatherHalfKernel(at::Half const *__restrict__ src,    // [NA, C]
                 at::Half *__restrict__ dst,          // [TP, C]
                 int32_t const *__restrict__ indices, // [TP]
                 int64_t total_vecs,                  // TP * (C / 8)
                 int64_t C_vec) {                     // C / 8
    auto const *src4 = reinterpret_cast<float4 const *>(src);
    auto *dst4       = reinterpret_cast<float4 *>(dst);

    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total_vecs;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t const pos     = idx / C_vec;
        int64_t const v       = idx % C_vec;
        int32_t const src_row = indices[pos];
        dst4[idx]             = src4[static_cast<int64_t>(src_row) * C_vec + v];
    }
}

// ============================================================================
// Kernel: atomic scatter-add fp32 GEMM output into fp32 accumulator
// ============================================================================
//
// Processes ALL total_pairs at once using atomicAdd.  The same output voxel
// can appear in multiple kernel-offset segments, so atomics are required.
// On Ampere+ GPUs, fp32 atomicAdd is hardware-accelerated in L2 cache.
// This single-launch approach eliminates the K separate kernel launches
// that the previous per-offset sequential scatter required.

__global__ void
scatterAddF32Kernel(float const *__restrict__ src,       // [TP, C]
                    int32_t const *__restrict__ indices, // [TP]
                    int64_t TP,
                    int64_t C,
                    float *__restrict__ dst)             // [NB, C]
{
    int64_t const total = TP * C;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t const pos     = idx / C;
        int64_t const c       = idx % C;
        int32_t const dst_row = indices[pos];
        atomicAdd(&dst[static_cast<int64_t>(dst_row) * C + c], src[idx]);
    }
}

// ============================================================================
// CUTLASS grouped GEMM runner (shared by forward and backward)
// ============================================================================
//
// Handles workspace allocation, initialize, and run for any GemmGrouped type.
// d_pack points to a device buffer laid out as [8, num_groups] int64 with rows:
//   0=ptrA  1=ptrB  2=ptrC  3=ptrD  4=ldA  5=ldB  6=ldC  7=ldD

template <typename GemmGroupedT>
static void
runCutlassGroupedGemm(torch::Tensor h_problem_sizes, // [num_groups, 3] int32, host
                      int num_groups,
                      torch::Tensor d_problem_sizes, // [num_groups, 3] int32, device
                      int64_t *d_pack,               // [8 * num_groups] int64, device
                      torch::Device device,
                      cudaStream_t stream,
                      char const *label) {
    int threadblock_count = GemmGroupedT::sufficient(
        reinterpret_cast<cutlass::gemm::GemmCoord *>(h_problem_sizes.data_ptr<int32_t>()),
        num_groups);

    typename GemmGroupedT::EpilogueOutputOp::Params epilogue(1.0f, 0.0f);

    typename GemmGroupedT::Arguments args(
        reinterpret_cast<cutlass::gemm::GemmCoord *>(d_problem_sizes.data_ptr<int32_t>()),
        num_groups,
        threadblock_count,
        epilogue,
        reinterpret_cast<CutlassElementA **>(d_pack + 0 * num_groups),
        reinterpret_cast<CutlassElementB **>(d_pack + 1 * num_groups),
        reinterpret_cast<CutlassElementC **>(d_pack + 2 * num_groups),
        reinterpret_cast<CutlassElementC **>(d_pack + 3 * num_groups),
        d_pack + 4 * num_groups,
        d_pack + 5 * num_groups,
        d_pack + 6 * num_groups,
        d_pack + 7 * num_groups,
        reinterpret_cast<cutlass::gemm::GemmCoord *>(h_problem_sizes.data_ptr<int32_t>()));

    size_t workspace_bytes = GemmGroupedT::get_workspace_size(args);
    auto workspace = torch::empty({std::max(static_cast<int64_t>(workspace_bytes), int64_t{1})},
                                  torch::dtype(torch::kByte).device(device));

    GemmGroupedT gemm_op;
    cutlass::Status status = gemm_op.initialize(args, workspace.data_ptr(), stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                label,
                ": CUTLASS initialize failed: ",
                cutlass::cutlassGetStatusString(status));

    status = gemm_op.run(stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                label,
                ": CUTLASS run failed: ",
                cutlass::cutlassGetStatusString(status));
}

// ============================================================================
// Topology sweep 1: per-block pair counts (no global atomics)
// ============================================================================
//
// Grid-stride loop over output grid leaves (NanoVDB OnIndex layout).
// Per-offset hit counts are accumulated in shared-memory local_counts[K]
// using fast shared-memory atomicAdd (single-cycle, no cache coherence
// traffic).  At the end of the block, local counts are stored to
// block_counts[blockIdx.x * K + k].
//
// This eliminates the global atomicAdd contention of the previous design
// where all blocks hammered K global counters.
//
// Dynamic shared memory: K * sizeof(int32_t).

__global__ void
countPairsPerBlockKernel(GridBatchImpl::Accessor output_acc,
                         GridBatchImpl::Accessor feature_acc,
                         nanovdb::Coord kernel_start,
                         nanovdb::Coord kernel_size,
                         nanovdb::Coord stride,
                         bool transposed,
                         int32_t *__restrict__ block_counts, // [num_blocks * K]
                         int64_t K,
                         int64_t total) {
    extern __shared__ char smem_count[];
    int32_t *local_counts = reinterpret_cast<int32_t *>(smem_count);

    // Zero-initialize shared memory
    for (int64_t k = threadIdx.x; k < K; k += blockDim.x) {
        local_counts[k] = 0;
    }
    __syncthreads();

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
                atomicAdd(&local_counts[k], 1); // shared memory -- fast, block-local
            }
        }
    }

    // Store block-local counts to global memory
    __syncthreads();
    for (int64_t k = threadIdx.x; k < K; k += blockDim.x) {
        block_counts[static_cast<int64_t>(blockIdx.x) * K + k] = local_counts[k];
    }
}

// ============================================================================
// Topology sweep 2: fill gather/scatter indices (no global atomics)
// ============================================================================
//
// Same traversal as sweep 1.  Each block loads its precomputed write offsets
// from block_write_offsets[blockIdx.x * K + k] and uses a shared-memory
// running counter to assign unique positions.  No global atomics at all.
//
// Dynamic shared memory layout:
//   [0 .. K * 4)   -> local_count[K]  (int32, running write-position counter)
//   [aligned .. +K*8) -> my_offsets[K] (int64, this block's starting positions)

__global__ void
fillPairsPerBlockKernel(GridBatchImpl::Accessor output_acc,
                        GridBatchImpl::Accessor feature_acc,
                        nanovdb::Coord kernel_start,
                        nanovdb::Coord kernel_size,
                        nanovdb::Coord stride,
                        bool transposed,
                        int64_t const *__restrict__ block_write_offsets, // [num_blocks * K]
                        int32_t *__restrict__ gather_indices,            // [total_pairs]
                        int32_t *__restrict__ scatter_indices,           // [total_pairs]
                        int64_t K,
                        int64_t total) {
    extern __shared__ char smem_fill[];
    int32_t *local_count = reinterpret_cast<int32_t *>(smem_fill);
    // Align to 8 bytes for int64_t
    size_t const count_bytes_aligned =
        (static_cast<size_t>(K) * sizeof(int32_t) + 7u) & ~size_t(7u);
    int64_t *my_offsets = reinterpret_cast<int64_t *>(smem_fill + count_bytes_aligned);

    // Initialize: zero local counts, load this block's precomputed write offsets
    for (int64_t k = threadIdx.x; k < K; k += blockDim.x) {
        local_count[k] = 0;
        my_offsets[k]  = block_write_offsets[static_cast<int64_t>(blockIdx.x) * K + k];
    }
    __syncthreads();

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

                int32_t const lc     = atomicAdd(&local_count[k], 1); // shared memory
                int64_t const pos    = my_offsets[k] + lc;
                gather_indices[pos]  = feat_flat;
                scatter_indices[pos] = static_cast<int32_t>(out_flat);
            }
        }
    }
}

// ============================================================================
// GPU topology builder -- two-sweep, no global atomics
// ============================================================================
//
// Sweep 1: Each block counts hits per offset into shared memory, writes
//          per-block counts to block_counts[num_blocks, K].
// Prefix sums (torch ops): Compute block_write_offsets[num_blocks, K] giving
//          the exact global write position where each block starts per offset.
// Sweep 2: Each block re-traverses, uses shared-memory running counters +
//          precomputed offsets to write gather/scatter indices directly.
//
// Zero global atomics.  Both sweeps use the same grid-stride loop with
// identical grid dimensions, guaranteeing each block processes the same
// elements in both sweeps.

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

    // Grid dimensions must be identical for both sweeps (grid-stride determinism)
    int const num_blocks = gridSizeFor(total);

    // ---- Sweep 1: Per-block pair counts ----
    auto block_counts = torch::zeros({static_cast<int64_t>(num_blocks), K},
                                     torch::dtype(torch::kInt32).device(device));

    size_t const smem_count = static_cast<size_t>(K) * sizeof(int32_t);

    countPairsPerBlockKernel<<<num_blocks, kBlockSize, smem_count, stream>>>(
        output_acc,
        feature_acc,
        kernel_start,
        kernel_size,
        stride,
        transposed,
        block_counts.data_ptr<int32_t>(),
        K,
        total);

    // ---- Prefix sums (all via torch ops, no custom kernels) ----
    // block_counts is [num_blocks, K] int32.
    auto bc_i64 = block_counts.to(torch::kInt64); // [num_blocks, K]

    // Inclusive cumsum along blocks (dim=0) for each offset k
    auto block_incl = torch::cumsum(bc_i64, /*dim=*/0); // [num_blocks, K]

    // Per-k totals = last row of inclusive cumsum
    auto per_k_total = block_incl[-1]; // [K]

    // Global CSR offsets (exclusive cumsum of per-k totals)
    auto offsets_dev = torch::zeros({K + 1}, torch::dtype(torch::kInt64).device(device));
    offsets_dev.slice(0, 1, K + 1).copy_(torch::cumsum(per_k_total, 0));

    // Single scalar sync to learn total_pairs (needed for output buffer sizing)
    int64_t total_pairs = offsets_dev[K].item<int64_t>();

    // Host copy of offsets (needed by cutlassConv for GEMM problem setup)
    auto offsets_host = offsets_dev.cpu();

    // Exclusive cumsum along blocks: shift inclusive down by 1, prepend zeros
    auto block_excl = torch::zeros_like(block_incl); // [num_blocks, K]
    if (num_blocks > 1) {
        block_excl.slice(0, 1, num_blocks).copy_(block_incl.slice(0, 0, num_blocks - 1));
    }

    // Combined write offsets: global_offsets[k] + block-level exclusive offset
    // offsets_dev[:K] has shape [K]; unsqueeze(0) broadcasts across num_blocks rows.
    auto block_write_offsets =
        (block_excl + offsets_dev.slice(0, 0, K).unsqueeze(0)).contiguous(); // [num_blocks, K]

    // ---- Sweep 2: Fill gather/scatter indices ----
    auto gather_indices  = torch::empty({std::max(total_pairs, int64_t{1})},
                                       torch::dtype(torch::kInt32).device(device));
    auto scatter_indices = torch::empty({std::max(total_pairs, int64_t{1})},
                                        torch::dtype(torch::kInt32).device(device));

    if (total_pairs > 0) {
        size_t const count_bytes_aligned =
            (static_cast<size_t>(K) * sizeof(int32_t) + 7u) & ~size_t(7u);
        size_t const smem_fill = count_bytes_aligned + static_cast<size_t>(K) * sizeof(int64_t);

        fillPairsPerBlockKernel<<<num_blocks, kBlockSize, smem_fill, stream>>>(
            output_acc,
            feature_acc,
            kernel_start,
            kernel_size,
            stride,
            transposed,
            block_write_offsets.data_ptr<int64_t>(),
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

    // ---- Phase 1: Gather features into [TP, Cin] fp16 buffer (vectorised) ----
    auto buf_A = torch::empty({TP, Cin}, torch::dtype(torch::kFloat16).device(device));
    {
        int64_t const C_vec      = Cin / 8;
        int64_t const total_vecs = TP * C_vec;
        gatherHalfKernel<<<gridSizeFor(total_vecs), kBlockSize, 0, stream>>>(
            features.data_ptr<at::Half>(),
            buf_A.data_ptr<at::Half>(),
            topo.gather_indices.data_ptr<int32_t>(),
            total_vecs,
            C_vec);
    }

    // ---- Phase 2: CUTLASS grouped GEMM ----
    // Pack all per-group arrays into 2 host tensors, copy in 2 H2D transfers.
    auto off_acc = topo.offsets.accessor<int64_t, 1>();

    int num_groups = 0;
    for (int64_t k = 0; k < K; ++k) {
        if (off_acc[k + 1] > off_acc[k])
            ++num_groups;
    }

    auto buf_C = torch::empty({TP, Cout}, torch::dtype(torch::kFloat32).device(device));

    if (num_groups > 0) {
        // Host-side descriptors: problem_sizes [num_groups, 3] int32
        //                        packed [8, num_groups] int64  (ptrA..ptrD, ldA..ldD)
        auto h_problem_sizes = torch::empty({num_groups, 3}, torch::kInt32);
        auto h_packed        = torch::empty({8, num_groups}, torch::kInt64);

        auto ps = h_problem_sizes.accessor<int32_t, 2>();
        auto pk = h_packed.accessor<int64_t, 2>();

        uintptr_t const base_A = reinterpret_cast<uintptr_t>(buf_A.data_ptr<at::Half>());
        uintptr_t const base_B = reinterpret_cast<uintptr_t>(W.data_ptr<at::Half>());
        uintptr_t const base_C = reinterpret_cast<uintptr_t>(buf_C.data_ptr<float>());

        int g = 0;
        for (int64_t k = 0; k < K; ++k) {
            int64_t const start = off_acc[k];
            int64_t const Mk    = off_acc[k + 1] - start;
            if (Mk == 0)
                continue;

            ps[g][0] = static_cast<int32_t>(Mk);
            ps[g][1] = static_cast<int32_t>(Cout);
            ps[g][2] = static_cast<int32_t>(Cin);

            pk[0][g] = static_cast<int64_t>(base_A +
                                            static_cast<uintptr_t>(start * Cin * sizeof(at::Half)));
            pk[1][g] = static_cast<int64_t>(
                base_B + static_cast<uintptr_t>(k * Cin * Cout * sizeof(at::Half)));
            pk[2][g] =
                static_cast<int64_t>(base_C + static_cast<uintptr_t>(start * Cout * sizeof(float)));
            pk[3][g] = pk[2][g]; // D = C (beta=0, in-place)

            pk[4][g] = Cin;      // ldA: A [Mk, Cin]
            pk[5][g] = Cout;     // ldB: B [Cin, Cout]
            pk[6][g] = Cout;     // ldC: C [Mk, Cout]
            pk[7][g] = Cout;     // ldD: D [Mk, Cout]
            ++g;
        }

        // 2 H2D copies instead of 9
        auto d_problem_sizes = h_problem_sizes.to(device).contiguous();
        auto d_packed        = h_packed.to(device).contiguous();

        // Dispatch narrow tile for small N (= Cout) to avoid wasting partial tiles
        if (Cout <= 64) {
            runCutlassGroupedGemm<CutlassGemmGroupedNarrow>(h_problem_sizes,
                                                            num_groups,
                                                            d_problem_sizes,
                                                            d_packed.data_ptr<int64_t>(),
                                                            device,
                                                            stream,
                                                            "cutlassConv");
        } else {
            runCutlassGroupedGemm<CutlassGemmGrouped>(h_problem_sizes,
                                                      num_groups,
                                                      d_problem_sizes,
                                                      d_packed.data_ptr<int64_t>(),
                                                      device,
                                                      stream,
                                                      "cutlassConv");
        }
    }

    // ---- Phase 3: Scatter-add GEMM output into accumulator (single launch) ----
    {
        int64_t const total = TP * Cout;
        scatterAddF32Kernel<<<gridSizeFor(total), kBlockSize, 0, stream>>>(
            buf_C.data_ptr<float>(),
            topo.scatter_indices.data_ptr<int32_t>(),
            TP,
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
// Scatter-add uses a single atomicAdd launch (same as forward).
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
                "cutlassConvBackward: Cin must be a positive multiple of 32, got ",
                Cin);
    TORCH_CHECK(Cout > 0 && Cout % 32 == 0,
                "cutlassConvBackward: Cout must be a positive multiple of 32, got ",
                Cout);

    // ---- Device / stream setup ----
    auto const device = features.device();
    c10::cuda::CUDAGuard guard(device);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device.index()).stream();

    int64_t const F  = topo.feature_total_voxels;
    int64_t const K  = topo.kernel_volume;
    int64_t const TP = topo.total_pairs;

    // ---- Allocate grad outputs ----
    auto grad_features_f32 = torch::zeros({F, Cin}, torch::dtype(torch::kFloat32).device(device));
    auto grad_W_f32 = torch::zeros({K, Cin, Cout}, torch::dtype(torch::kFloat32).device(device));

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
    auto W   = weights.permute({2, 3, 4, 1, 0}).reshape({K, Cin, Cout}).contiguous();
    auto W_T = W.permute({0, 2, 1}).contiguous(); // [K, Cout, Cin]

    auto off_acc = topo.offsets.accessor<int64_t, 1>();

    // ---- Phase 1: Gather features and grad_output (vectorised) ----
    auto feat_buf = torch::empty({TP, Cin}, torch::dtype(torch::kFloat16).device(device));
    auto grad_buf = torch::empty({TP, Cout}, torch::dtype(torch::kFloat16).device(device));

    {
        int64_t const C_vec      = Cin / 8;
        int64_t const total_vecs = TP * C_vec;
        gatherHalfKernel<<<gridSizeFor(total_vecs), kBlockSize, 0, stream>>>(
            features.data_ptr<at::Half>(),
            feat_buf.data_ptr<at::Half>(),
            topo.gather_indices.data_ptr<int32_t>(),
            total_vecs,
            C_vec);
    }
    {
        int64_t const C_vec      = Cout / 8;
        int64_t const total_vecs = TP * C_vec;
        gatherHalfKernel<<<gridSizeFor(total_vecs), kBlockSize, 0, stream>>>(
            grad_output.data_ptr<at::Half>(),
            grad_buf.data_ptr<at::Half>(),
            topo.scatter_indices.data_ptr<int32_t>(),
            total_vecs,
            C_vec);
    }

    // ---- Count non-empty offset groups (shared by all GEMMs) ----
    int num_groups = 0;
    for (int64_t k = 0; k < K; ++k) {
        if (off_acc[k + 1] > off_acc[k])
            ++num_groups;
    }

    // ---- Phase 2: CUTLASS grouped GEMM for grad_features ----
    // grad_buf[k] @ W_T[k] -> grad_feat_buf[k]
    // GemmCoord(M=Mk, N=Cin, K=Cout)
    auto grad_feat_buf = torch::empty({TP, Cin}, torch::dtype(torch::kFloat32).device(device));

    if (num_groups > 0) {
        auto h_ps = torch::empty({num_groups, 3}, torch::kInt32);
        auto h_pk = torch::empty({8, num_groups}, torch::kInt64);
        auto ps   = h_ps.accessor<int32_t, 2>();
        auto pk   = h_pk.accessor<int64_t, 2>();

        uintptr_t const base_A = reinterpret_cast<uintptr_t>(grad_buf.data_ptr<at::Half>());
        uintptr_t const base_B = reinterpret_cast<uintptr_t>(W_T.data_ptr<at::Half>());
        uintptr_t const base_C = reinterpret_cast<uintptr_t>(grad_feat_buf.data_ptr<float>());

        int g = 0;
        for (int64_t k = 0; k < K; ++k) {
            int64_t const start = off_acc[k];
            int64_t const Mk    = off_acc[k + 1] - start;
            if (Mk == 0)
                continue;

            ps[g][0] = static_cast<int32_t>(Mk);
            ps[g][1] = static_cast<int32_t>(Cin);
            ps[g][2] = static_cast<int32_t>(Cout);

            pk[0][g] = static_cast<int64_t>(
                base_A + static_cast<uintptr_t>(start * Cout * sizeof(at::Half)));
            pk[1][g] = static_cast<int64_t>(
                base_B + static_cast<uintptr_t>(k * Cout * Cin * sizeof(at::Half)));
            pk[2][g] =
                static_cast<int64_t>(base_C + static_cast<uintptr_t>(start * Cin * sizeof(float)));
            pk[3][g] = pk[2][g];

            pk[4][g] = Cout; // ldA: A [Mk, Cout]
            pk[5][g] = Cin;  // ldB: B [Cout, Cin]
            pk[6][g] = Cin;  // ldC: C [Mk, Cin]
            pk[7][g] = Cin;  // ldD
            ++g;
        }

        auto d_ps = h_ps.to(device).contiguous();
        auto d_pk = h_pk.to(device).contiguous();

        if (Cin <= 64) {
            runCutlassGroupedGemm<CutlassGemmGroupedNarrow>(h_ps,
                                                            num_groups,
                                                            d_ps,
                                                            d_pk.data_ptr<int64_t>(),
                                                            device,
                                                            stream,
                                                            "cutlassConvBackward(grad_feat)");
        } else {
            runCutlassGroupedGemm<CutlassGemmGrouped>(h_ps,
                                                      num_groups,
                                                      d_ps,
                                                      d_pk.data_ptr<int64_t>(),
                                                      device,
                                                      stream,
                                                      "cutlassConvBackward(grad_feat)");
        }
    }

    // ---- Phase 3: Scatter-add grad_feat_buf into grad_features (single launch) ----
    {
        int64_t const total = TP * Cin;
        scatterAddF32Kernel<<<gridSizeFor(total), kBlockSize, 0, stream>>>(
            grad_feat_buf.data_ptr<float>(),
            topo.gather_indices.data_ptr<int32_t>(),
            TP,
            Cin,
            grad_features_f32.data_ptr<float>());
    }

    // ---- Phase 4: CUTLASS grouped GEMM for grad_weights ----
    // grad_W[k] = feat_buf[k].T @ grad_buf[k]
    // A = [Mk, Cin] RowMajor = [Cin, Mk] ColMajor (zero-copy transpose)
    // B = [Mk, Cout] RowMajor, C = [Cin, Cout] RowMajor (fp32)
    // GemmCoord(M=Cin, N=Cout, K=Mk)

    if (num_groups > 0) {
        auto h_gw_ps = torch::empty({num_groups, 3}, torch::kInt32);
        auto h_gw_pk = torch::empty({8, num_groups}, torch::kInt64);
        auto gw_ps   = h_gw_ps.accessor<int32_t, 2>();
        auto gw_pk   = h_gw_pk.accessor<int64_t, 2>();

        uintptr_t const gw_base_A = reinterpret_cast<uintptr_t>(feat_buf.data_ptr<at::Half>());
        uintptr_t const gw_base_B = reinterpret_cast<uintptr_t>(grad_buf.data_ptr<at::Half>());
        uintptr_t const gw_base_C = reinterpret_cast<uintptr_t>(grad_W_f32.data_ptr<float>());

        int gw_g = 0;
        for (int64_t k = 0; k < K; ++k) {
            int64_t const start = off_acc[k];
            int64_t const Mk    = off_acc[k + 1] - start;
            if (Mk == 0)
                continue;

            gw_ps[gw_g][0] = static_cast<int32_t>(Cin);
            gw_ps[gw_g][1] = static_cast<int32_t>(Cout);
            gw_ps[gw_g][2] = static_cast<int32_t>(Mk);

            gw_pk[0][gw_g] = static_cast<int64_t>(
                gw_base_A + static_cast<uintptr_t>(start * Cin * sizeof(at::Half)));
            gw_pk[1][gw_g] = static_cast<int64_t>(
                gw_base_B + static_cast<uintptr_t>(start * Cout * sizeof(at::Half)));
            gw_pk[2][gw_g] = static_cast<int64_t>(
                gw_base_C + static_cast<uintptr_t>(k * Cin * Cout * sizeof(float)));
            gw_pk[3][gw_g] = gw_pk[2][gw_g];

            // A ColumnMajor [Cin, Mk]: lda = Cin (stride between columns)
            gw_pk[4][gw_g] = Cin;
            gw_pk[5][gw_g] = Cout;
            gw_pk[6][gw_g] = Cout;
            gw_pk[7][gw_g] = Cout;
            ++gw_g;
        }

        auto d_gw_ps = h_gw_ps.to(device).contiguous();
        auto d_gw_pk = h_gw_pk.to(device).contiguous();

        if (Cout <= 64) {
            runCutlassGroupedGemm<CutlassGradWGemmGroupedNarrow>(h_gw_ps,
                                                                 num_groups,
                                                                 d_gw_ps,
                                                                 d_gw_pk.data_ptr<int64_t>(),
                                                                 device,
                                                                 stream,
                                                                 "cutlassConvBackward(grad_w)");
        } else {
            runCutlassGroupedGemm<CutlassGradWGemmGrouped>(h_gw_ps,
                                                           num_groups,
                                                           d_gw_ps,
                                                           d_gw_pk.data_ptr<int64_t>(),
                                                           device,
                                                           stream,
                                                           "cutlassConvBackward(grad_w)");
        }
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
