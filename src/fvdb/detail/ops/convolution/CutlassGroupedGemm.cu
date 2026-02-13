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
//     Delegates to groupedGemmSparseConvTopology (proven CSR compaction).
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
#include <fvdb/detail/ops/convolution/GroupedGemm.h>

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
#include <cstdint>
#include <vector>

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

// ============================================================================
// CUDA kernel launch helpers
// ============================================================================

static constexpr int kBlockSize = 256;

static int
gridSizeFor(int64_t total_elements) {
    return static_cast<int>(
        std::min((total_elements + kBlockSize - 1) / kBlockSize, static_cast<int64_t>(4096)));
}

// (Topology kernels removed -- topology is built via GroupedGemm's proven path)

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
// Topology builder -- delegates to GroupedGemm's proven topology construction
// ============================================================================
//
// Builds the CSR topology (per-offset gather/scatter index arrays) by calling
// groupedGemmSparseConvTopology, which uses the GatherScatter dense kernel_map
// internally and compacts it.  This is the same battle-tested path used by
// the GroupedGemm backend.

CutlassConvTopology
cutlassConvTopology(GridBatchImpl const &feature_grid,
                    GridBatchImpl const &output_grid,
                    nanovdb::Coord kernel_size,
                    nanovdb::Coord stride) {
    auto gg = groupedGemmSparseConvTopology(feature_grid, output_grid, kernel_size, stride);

    return CutlassConvTopology{
        gg.gather_indices,
        gg.scatter_indices,
        gg.offsets,
        gg.feature_total_voxels,
        gg.output_total_voxels,
        gg.kernel_volume,
        gg.total_pairs,
        gg.kernel_size,
        gg.stride,
    };
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

} // namespace ops
} // namespace detail
} // namespace fvdb
