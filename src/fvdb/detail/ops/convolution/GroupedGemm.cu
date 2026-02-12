// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// GroupedGemm.cu -- CUTLASS grouped-GEMM sparse convolution implementation.
//
// Three phases per convolution:
//   1. Gather:       features[gather_indices[i]] -> contiguous buffer       (1 launch)
//   2. Grouped GEMM: CUTLASS GemmGrouped over K=27 groups                  (1 launch)
//   3. Scatter-add:  result buffer -> output[scatter_indices[i]] (atomic)   (1 launch)
//
// Three CUTLASS instantiations for the three distinct matrix layouts:
//   Forward:       RowMajor A  x RowMajor    B -> RowMajor D
//   Grad-features: RowMajor A  x ColumnMajor B -> RowMajor D
//   Grad-weights:  ColumnMajor A x RowMajor  B -> RowMajor D
//

#include <fvdb/detail/ops/convolution/GatherScatter.h>
#include <fvdb/detail/ops/convolution/GroupedGemm.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>

// CUTLASS headers
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/util/device_memory.h>

#include <cstdint>
#include <tuple>
#include <vector>

namespace fvdb {
namespace detail {
namespace ops {

// =============================================================================
// CUTLASS type aliases shared by all three instantiations
// =============================================================================

using ElementA   = float;
using ElementB   = float;
using ElementC   = float;
using ElementAcc = float;

using ArchTag = cutlass::arch::Sm80;
using OpClass = cutlass::arch::OpClassTensorOp;

// Tile shapes: 64x64x32 with 4 pipeline stages uses ~64KB shared memory,
// fitting comfortably within the 100KB limit on Sm80/Sm89.  The smaller
// tiles also produce more threadblocks, improving load balance across the
// variable-M groups of a sparse convolution.
using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 32>;
using WarpShape        = cutlass::gemm::GemmShape<32, 32, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>; // TF32

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementC,
    128 / cutlass::sizeof_bits<ElementC>::value, // vector width = 4 floats
    ElementAcc,
    ElementAcc>;

using Swizzle = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;

static constexpr int kStages    = 4;
static constexpr int kAlignment = 4; // 4 floats = 16 bytes

using GroupSchedule = cutlass::gemm::kernel::GroupScheduleMode;

// =============================================================================
// Forward GEMM: RowMajor A x RowMajor B -> RowMajor D
//   A = gathered features  [Nk, C_in]   RowMajor
//   B = weights[k]         [C_in, C_out] RowMajor
//   D = result buffer      [Nk, C_out]   RowMajor
// =============================================================================

using ForwardGemmKernel =
    typename cutlass::gemm::kernel::DefaultGemmGrouped<ElementA,
                                                       cutlass::layout::RowMajor,
                                                       cutlass::ComplexTransform::kNone,
                                                       kAlignment,
                                                       ElementB,
                                                       cutlass::layout::RowMajor,
                                                       cutlass::ComplexTransform::kNone,
                                                       kAlignment,
                                                       ElementC,
                                                       cutlass::layout::RowMajor,
                                                       ElementAcc,
                                                       OpClass,
                                                       ArchTag,
                                                       ThreadblockShape,
                                                       WarpShape,
                                                       InstructionShape,
                                                       EpilogueOp,
                                                       Swizzle,
                                                       kStages,
                                                       GroupSchedule::kDeviceOnly>::GemmKernel;

using ForwardGemm = cutlass::gemm::device::GemmGrouped<ForwardGemmKernel>;

// =============================================================================
// Grad-features GEMM: RowMajor A x ColumnMajor B -> RowMajor D
//   A = gathered grad_out  [Nk, C_out]  RowMajor
//   B = weights[k]         [C_in, C_out] stored RowMajor, read as ColumnMajor = W^T
//   D = grad_feat buffer   [Nk, C_in]   RowMajor
//   GemmCoord per group: (Nk, C_in, C_out)
// =============================================================================

using GradFeatGemmKernel =
    typename cutlass::gemm::kernel::DefaultGemmGrouped<ElementA,
                                                       cutlass::layout::RowMajor,
                                                       cutlass::ComplexTransform::kNone,
                                                       kAlignment,
                                                       ElementB,
                                                       cutlass::layout::ColumnMajor,
                                                       cutlass::ComplexTransform::kNone,
                                                       kAlignment,
                                                       ElementC,
                                                       cutlass::layout::RowMajor,
                                                       ElementAcc,
                                                       OpClass,
                                                       ArchTag,
                                                       ThreadblockShape,
                                                       WarpShape,
                                                       InstructionShape,
                                                       EpilogueOp,
                                                       Swizzle,
                                                       kStages,
                                                       GroupSchedule::kDeviceOnly>::GemmKernel;

using GradFeatGemm = cutlass::gemm::device::GemmGrouped<GradFeatGemmKernel>;

// =============================================================================
// Grad-weights GEMM: ColumnMajor A x RowMajor B -> RowMajor D
//   A = gathered features  [Nk, C_in]  stored RowMajor, read as ColumnMajor = feat^T
//   B = gathered grad_out  [Nk, C_out] RowMajor
//   D = grad_W[k]          [C_in, C_out] RowMajor
//   GemmCoord per group: (C_in, C_out, Nk)
// =============================================================================

using GradWeightGemmKernel =
    typename cutlass::gemm::kernel::DefaultGemmGrouped<ElementA,
                                                       cutlass::layout::ColumnMajor,
                                                       cutlass::ComplexTransform::kNone,
                                                       kAlignment,
                                                       ElementB,
                                                       cutlass::layout::RowMajor,
                                                       cutlass::ComplexTransform::kNone,
                                                       kAlignment,
                                                       ElementC,
                                                       cutlass::layout::RowMajor,
                                                       ElementAcc,
                                                       OpClass,
                                                       ArchTag,
                                                       ThreadblockShape,
                                                       WarpShape,
                                                       InstructionShape,
                                                       EpilogueOp,
                                                       Swizzle,
                                                       kStages,
                                                       GroupSchedule::kDeviceOnly>::GemmKernel;

using GradWeightGemm = cutlass::gemm::device::GemmGrouped<GradWeightGemmKernel>;

// =============================================================================
// Gather / scatter-add CUDA kernels
// =============================================================================

// Gather: dst[i * C + c] = src[indices[i] * C + c]
__global__ void
gatherKernel(float const *__restrict__ src,
             float *__restrict__ dst,
             int32_t const *__restrict__ indices,
             int64_t total_pairs,
             int64_t C) {
    int64_t const idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_pairs * C)
        return;
    int64_t const pair    = idx / C;
    int64_t const c       = idx % C;
    int32_t const src_row = indices[pair];
    dst[pair * C + c]     = src[static_cast<int64_t>(src_row) * C + c];
}

// Scatter-add: dst[indices[i] * C + c] += src[i * C + c]
__global__ void
scatterAddKernel(float const *__restrict__ src,
                 float *__restrict__ dst,
                 int32_t const *__restrict__ indices,
                 int64_t total_pairs,
                 int64_t C) {
    int64_t const idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_pairs * C)
        return;
    int64_t const pair    = idx / C;
    int64_t const c       = idx % C;
    int32_t const dst_row = indices[pair];
    atomicAdd(&dst[static_cast<int64_t>(dst_row) * C + c], src[pair * C + c]);
}

static void
launchGather(torch::Tensor src,
             torch::Tensor dst,
             torch::Tensor indices,
             int64_t total_pairs,
             int64_t C,
             cudaStream_t stream) {
    if (total_pairs == 0)
        return;
    int64_t const total = total_pairs * C;
    int const threads   = 256;
    int const blocks    = static_cast<int>((total + threads - 1) / threads);
    gatherKernel<<<blocks, threads, 0, stream>>>(
        src.data_ptr<float>(), dst.data_ptr<float>(), indices.data_ptr<int32_t>(), total_pairs, C);
}

static void
launchScatterAdd(torch::Tensor src,
                 torch::Tensor dst,
                 torch::Tensor indices,
                 int64_t total_pairs,
                 int64_t C,
                 cudaStream_t stream) {
    if (total_pairs == 0)
        return;
    int64_t const total = total_pairs * C;
    int const threads   = 256;
    int const blocks    = static_cast<int>((total + threads - 1) / threads);
    scatterAddKernel<<<blocks, threads, 0, stream>>>(
        src.data_ptr<float>(), dst.data_ptr<float>(), indices.data_ptr<int32_t>(), total_pairs, C);
}

// =============================================================================
// CUTLASS grouped GEMM runner helper
// =============================================================================
//
// Builds the pointer / lda / problem-size arrays on host, copies to device,
// creates CUTLASS Arguments, and runs the grouped GEMM.

template <typename Gemm>
static void
runGroupedGemm(
    // Per-group data: K groups
    std::vector<cutlass::gemm::GemmCoord> const &problems,
    std::vector<float *> const &ptr_A_host,
    std::vector<float *> const &ptr_B_host,
    std::vector<float *> const &ptr_D_host,
    std::vector<int64_t> const &lda_host,
    std::vector<int64_t> const &ldb_host,
    std::vector<int64_t> const &ldd_host,
    float alpha,
    float beta,
    cudaStream_t stream) {
    int const K = static_cast<int>(problems.size());
    if (K == 0)
        return;

    // Allocate device arrays via CUTLASS utility
    cutlass::DeviceAllocation<cutlass::gemm::GemmCoord> problems_device(K);
    cutlass::DeviceAllocation<float *> ptr_A_device(K);
    cutlass::DeviceAllocation<float *> ptr_B_device(K);
    cutlass::DeviceAllocation<float *> ptr_C_device(K); // same as D for beta=0
    cutlass::DeviceAllocation<float *> ptr_D_device(K);
    cutlass::DeviceAllocation<int64_t> lda_device(K);
    cutlass::DeviceAllocation<int64_t> ldb_device(K);
    cutlass::DeviceAllocation<int64_t> ldc_device(K);
    cutlass::DeviceAllocation<int64_t> ldd_device(K);

    problems_device.copy_from_host(problems.data());
    ptr_A_device.copy_from_host(ptr_A_host.data());
    ptr_B_device.copy_from_host(ptr_B_host.data());
    ptr_C_device.copy_from_host(ptr_D_host.data()); // C == D for beta=0
    ptr_D_device.copy_from_host(ptr_D_host.data());
    lda_device.copy_from_host(lda_host.data());
    ldb_device.copy_from_host(ldb_host.data());
    ldc_device.copy_from_host(ldd_host.data()); // ldc == ldd
    ldd_device.copy_from_host(ldd_host.data());

    typename Gemm::EpilogueOutputOp::Params epilogue_params(alpha, beta);

    int threadblock_count = Gemm::sufficient(problems.data(), K);

    typename Gemm::Arguments args(
        problems_device.get(),
        K,
        threadblock_count,
        epilogue_params,
        ptr_A_device.get(),
        ptr_B_device.get(),
        ptr_C_device.get(),
        ptr_D_device.get(),
        lda_device.get(),
        ldb_device.get(),
        ldc_device.get(),
        ldd_device.get(),
        const_cast<cutlass::gemm::GemmCoord *>(problems.data()) // host copy for precompute
    );

    Gemm gemm_op;

    size_t workspace_size = gemm_op.get_workspace_size(args);
    cutlass::DeviceAllocation<uint8_t> workspace(workspace_size);

    cutlass::Status status = gemm_op.initialize(args, workspace.get(), stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "CUTLASS grouped GEMM initialize failed: ",
                cutlassGetStatusString(status));

    status = gemm_op.run(stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "CUTLASS grouped GEMM run failed: ",
                cutlassGetStatusString(status));
}

// =============================================================================
// Topology compaction
// =============================================================================

static GroupedGemmTopology
compactTopology(GatherScatterTopology const &dense_topo) {
    auto const &kmap = dense_topo.kernel_map; // [O, K] int32 on device
    int64_t const O  = dense_topo.output_total_voxels;
    int64_t const K  = dense_topo.kernel_volume;

    // Transpose to [K, O] so nonzero() iterates k-major (groups are contiguous)
    auto kmap_t = kmap.t().contiguous(); // [K, O]
    auto mask   = kmap_t != -1;          // [K, O] bool

    // Per-offset pair counts and cumulative offsets
    auto sizes = torch::sum(mask, /*dim=*/-1, /*keepdim=*/false, torch::kInt64); // [K] on device
    auto sizes_cpu = sizes.cpu().contiguous();                         // [K] int64 on host

    auto offsets = torch::zeros({K + 1}, torch::dtype(torch::kInt64)); // host
    auto off_acc = offsets.accessor<int64_t, 1>();
    auto sz_acc  = sizes_cpu.accessor<int64_t, 1>();
    for (int64_t k = 0; k < K; ++k) {
        off_acc[k + 1] = off_acc[k] + sz_acc[k];
    }
    int64_t total_pairs = off_acc[K];

    if (total_pairs == 0) {
        return GroupedGemmTopology{
            torch::empty({0}, torch::dtype(torch::kInt32).device(kmap.device())),
            torch::empty({0}, torch::dtype(torch::kInt32).device(kmap.device())),
            offsets,
            dense_topo.feature_total_voxels,
            dense_topo.output_total_voxels,
            K,
            0,
            dense_topo.kernel_size,
            dense_topo.stride,
            dense_topo.direction,
        };
    }

    // Find all active (k, o) pairs -- sorted by k because nonzero iterates row-major on [K, O]
    auto pairs = torch::nonzero(mask).contiguous(); // [total_pairs, 2] int64
    auto k_col = pairs.select(1, 0);                // [total_pairs] -- kernel offset indices
    auto o_col = pairs.select(1, 1);                // [total_pairs] -- output voxel indices

    // Look up the feature voxel index for each pair from kmap_t[k, o]
    auto flat_idx        = k_col * O + o_col;
    auto gather_indices  = kmap_t.reshape({-1}).index({flat_idx}).to(torch::kInt32).contiguous();
    auto scatter_indices = o_col.to(torch::kInt32).contiguous();

    return GroupedGemmTopology{
        gather_indices,
        scatter_indices,
        offsets,
        dense_topo.feature_total_voxels,
        dense_topo.output_total_voxels,
        K,
        total_pairs,
        dense_topo.kernel_size,
        dense_topo.stride,
        dense_topo.direction,
    };
}

// =============================================================================
// Topology builder entry points
// =============================================================================

GroupedGemmTopology
groupedGemmSparseConvTopology(GridBatchImpl const &feature_grid,
                              GridBatchImpl const &output_grid,
                              nanovdb::Coord kernel_size,
                              nanovdb::Coord stride) {
    auto dense = gatherScatterSparseConvTopology(feature_grid, output_grid, kernel_size, stride);
    return compactTopology(dense);
}

GroupedGemmTopology
groupedGemmSparseConvTransposeTopology(GridBatchImpl const &feature_grid,
                                       GridBatchImpl const &output_grid,
                                       nanovdb::Coord kernel_size,
                                       nanovdb::Coord stride) {
    auto dense =
        gatherScatterSparseConvTransposeTopology(feature_grid, output_grid, kernel_size, stride);
    return compactTopology(dense);
}

// =============================================================================
// Precondition checks
// =============================================================================

static void
checkGroupedGemmPreconditions(torch::Tensor features,
                              torch::Tensor weights,
                              GroupedGemmTopology const &topo,
                              char const *name) {
    TORCH_CHECK(features.device().is_cuda(), name, ": features must be on CUDA");
    TORCH_CHECK(features.dim() == 2, name, ": features must be 2D");
    TORCH_CHECK(features.size(0) == topo.feature_total_voxels,
                name,
                ": features.size(0)=",
                features.size(0),
                " must match feature_total_voxels=",
                topo.feature_total_voxels);
    TORCH_CHECK(features.scalar_type() == torch::kFloat32, name, ": features must be float32");
    TORCH_CHECK(features.is_contiguous(), name, ": features must be contiguous");

    TORCH_CHECK(weights.dim() == 5, name, ": weights must be 5D [C_out, C_in, k0, k1, k2]");
    TORCH_CHECK(weights.scalar_type() == torch::kFloat32, name, ": weights must be float32");
    TORCH_CHECK(features.size(1) == weights.size(1),
                name,
                ": features channels=",
                features.size(1),
                " must match weights C_in=",
                weights.size(1));

    int64_t C_in  = weights.size(1);
    int64_t C_out = weights.size(0);
    TORCH_CHECK(C_in % 32 == 0, name, ": C_in=", C_in, " must be a multiple of 32");
    TORCH_CHECK(C_out % 32 == 0, name, ": C_out=", C_out, " must be a multiple of 32");

    TORCH_CHECK(weights.size(2) == topo.kernel_size[0] && weights.size(3) == topo.kernel_size[1] &&
                    weights.size(4) == topo.kernel_size[2],
                name,
                ": weights spatial dims must match topology kernel_size");

    TORCH_CHECK(features.device() == weights.device(),
                name,
                ": features and weights must be on the same device");
}

// =============================================================================
// Forward convolution
// =============================================================================

torch::Tensor
groupedGemmSparseConv(torch::Tensor features,
                      torch::Tensor weights,
                      GroupedGemmTopology const &topo) {
    checkGroupedGemmPreconditions(features, weights, topo, "groupedGemmSparseConv");
    TORCH_CHECK(topo.direction == ConvDirection::Forward,
                "groupedGemmSparseConv requires topology with direction=Forward");

    c10::cuda::CUDAGuard device_guard(features.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    int64_t const O     = topo.output_total_voxels;
    int64_t const K     = topo.kernel_volume;
    int64_t const C_in  = weights.size(1);
    int64_t const C_out = weights.size(0);
    int64_t const TP    = topo.total_pairs;

    // Reshape weights: [C_out, C_in, k0, k1, k2] -> [K, C_in, C_out] RowMajor contiguous
    auto W = weights.permute({2, 3, 4, 1, 0}).reshape({K, C_in, C_out}).contiguous();

    auto output = torch::zeros({O, C_out}, features.options());

    if (O == 0 || K == 0 || TP == 0)
        return output;

    // Phase 1: Gather features into contiguous buffer [total_pairs, C_in]
    auto buf_A = torch::empty({TP, C_in}, features.options());
    launchGather(features, buf_A, topo.gather_indices, TP, C_in, stream);

    // Phase 2: CUTLASS grouped GEMM
    auto buf_D = torch::empty({TP, C_out}, features.options());

    auto off_acc = topo.offsets.accessor<int64_t, 1>();

    std::vector<cutlass::gemm::GemmCoord> problems(K);
    std::vector<float *> ptr_A(K), ptr_B(K), ptr_D(K);
    std::vector<int64_t> lda(K), ldb(K), ldd(K);

    float *buf_A_ptr = buf_A.data_ptr<float>();
    float *W_ptr     = W.data_ptr<float>();
    float *buf_D_ptr = buf_D.data_ptr<float>();

    for (int64_t k = 0; k < K; ++k) {
        int64_t Nk  = off_acc[k + 1] - off_acc[k];
        problems[k] = {static_cast<int>(Nk), static_cast<int>(C_out), static_cast<int>(C_in)};
        ptr_A[k]    = buf_A_ptr + off_acc[k] * C_in;
        ptr_B[k]    = W_ptr + k * C_in * C_out;
        ptr_D[k]    = buf_D_ptr + off_acc[k] * C_out;
        lda[k]      = C_in;
        ldb[k]      = C_out;
        ldd[k]      = C_out;
    }

    runGroupedGemm<ForwardGemm>(problems, ptr_A, ptr_B, ptr_D, lda, ldb, ldd, 1.0f, 0.0f, stream);

    // Phase 3: Scatter-add result into output
    launchScatterAdd(buf_D, output, topo.scatter_indices, TP, C_out, stream);

    return output;
}

// =============================================================================
// Backward convolution
// =============================================================================

std::tuple<torch::Tensor, torch::Tensor>
groupedGemmSparseConvBackward(torch::Tensor grad_output,
                              torch::Tensor features,
                              torch::Tensor weights,
                              GroupedGemmTopology const &topo) {
    checkGroupedGemmPreconditions(features, weights, topo, "groupedGemmSparseConvBackward");
    TORCH_CHECK(topo.direction == ConvDirection::Forward,
                "groupedGemmSparseConvBackward requires topology with direction=Forward");
    TORCH_CHECK(grad_output.dim() == 2 && grad_output.size(0) == topo.output_total_voxels,
                "grad_output shape mismatch");
    TORCH_CHECK(grad_output.is_contiguous(), "grad_output must be contiguous");
    TORCH_CHECK(grad_output.scalar_type() == torch::kFloat32, "grad_output must be float32");

    c10::cuda::CUDAGuard device_guard(features.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    int64_t const F     = topo.feature_total_voxels;
    int64_t const O     = topo.output_total_voxels;
    int64_t const K     = topo.kernel_volume;
    int64_t const C_in  = weights.size(1);
    int64_t const C_out = weights.size(0);
    int64_t const TP    = topo.total_pairs;

    // Reshape weights: [C_out, C_in, k0, k1, k2] -> [K, C_in, C_out]
    auto W = weights.permute({2, 3, 4, 1, 0}).reshape({K, C_in, C_out}).contiguous();

    auto grad_features = torch::zeros({F, C_in}, features.options());

    // grad_weights in flat [K, C_in, C_out] format, reshaped at end
    auto grad_W_flat = torch::zeros({K, C_in, C_out}, features.options());

    if (O == 0 || K == 0 || TP == 0) {
        auto ks           = topo.kernel_size;
        auto grad_weights = grad_W_flat.reshape({ks[0], ks[1], ks[2], C_in, C_out})
                                .permute({4, 3, 0, 1, 2})
                                .contiguous();
        return {grad_features, grad_weights};
    }

    auto off_acc = topo.offsets.accessor<int64_t, 1>();

    // Gather features and grad_output into contiguous buffers
    auto feat_buf = torch::empty({TP, C_in}, features.options());
    auto grad_buf = torch::empty({TP, C_out}, features.options());
    launchGather(features, feat_buf, topo.gather_indices, TP, C_in, stream);
    launchGather(grad_output, grad_buf, topo.scatter_indices, TP, C_out, stream);

    // --- grad_features: grad_buf @ W[k]^T -> grad_feat_buf ---
    auto grad_feat_buf = torch::empty({TP, C_in}, features.options());
    {
        std::vector<cutlass::gemm::GemmCoord> problems(K);
        std::vector<float *> ptr_A(K), ptr_B(K), ptr_D(K);
        std::vector<int64_t> lda(K), ldb(K), ldd(K);

        float *gd_ptr = grad_buf.data_ptr<float>();
        float *W_ptr  = W.data_ptr<float>();
        float *gf_ptr = grad_feat_buf.data_ptr<float>();

        for (int64_t k = 0; k < K; ++k) {
            int64_t Nk = off_acc[k + 1] - off_acc[k];
            // A=[Nk, C_out], B=W[k] as ColumnMajor[C_in, C_out] = W^T, D=[Nk, C_in]
            problems[k] = {static_cast<int>(Nk), static_cast<int>(C_in), static_cast<int>(C_out)};
            ptr_A[k]    = gd_ptr + off_acc[k] * C_out;
            ptr_B[k]    = W_ptr + k * C_in * C_out;
            ptr_D[k]    = gf_ptr + off_acc[k] * C_in;
            lda[k]      = C_out;
            ldb[k]      = C_out; // ColumnMajor ldb = number of rows = C_out? No...
            // ColumnMajor [C_in, C_out]: element (ci, co) at ci + co*C_in -> ldb = C_in
            // But we want B to represent [K_dim=C_out, N=C_in] in ColumnMajor.
            // Physical data is W[k] stored as RowMajor[C_in, C_out], so memory is:
            //   row ci, col co -> offset ci*C_out + co
            // ColumnMajor[C_out, C_in]: element (co, ci) at co + ci*C_out -> ldb = C_out
            // This reads element (K_dim_idx=co, N_idx=ci) from offset co + ci*C_out
            //   = ci*C_out + co.  Matches the RowMajor[C_in, C_out] layout!
            ldb[k] = C_out; // ldb for ColumnMajor = leading dimension = C_out
            ldd[k] = C_in;
        }

        runGroupedGemm<GradFeatGemm>(
            problems, ptr_A, ptr_B, ptr_D, lda, ldb, ldd, 1.0f, 0.0f, stream);
    }

    // Scatter-add grad_feat_buf into grad_features using gather_indices
    launchScatterAdd(grad_feat_buf, grad_features, topo.gather_indices, TP, C_in, stream);

    // --- grad_weights: feat_buf^T @ grad_buf -> grad_W[k] ---
    {
        std::vector<cutlass::gemm::GemmCoord> problems(K);
        std::vector<float *> ptr_A(K), ptr_B(K), ptr_D(K);
        std::vector<int64_t> lda(K), ldb(K), ldd(K);

        float *fb_ptr = feat_buf.data_ptr<float>();
        float *gd_ptr = grad_buf.data_ptr<float>();
        float *gw_ptr = grad_W_flat.data_ptr<float>();

        for (int64_t k = 0; k < K; ++k) {
            int64_t Nk = off_acc[k + 1] - off_acc[k];
            // A = feat_buf[Nk, C_in] RowMajor, read as ColumnMajor -> A^T[C_in, Nk]
            // B = grad_buf[Nk, C_out] RowMajor
            // D = grad_W[k][C_in, C_out] RowMajor
            // GemmCoord(M=C_in, N=C_out, K=Nk)
            problems[k] = {static_cast<int>(C_in), static_cast<int>(C_out), static_cast<int>(Nk)};
            ptr_A[k]    = fb_ptr + off_acc[k] * C_in;
            ptr_B[k]    = gd_ptr + off_acc[k] * C_out;
            ptr_D[k]    = gw_ptr + k * C_in * C_out;
            // ColumnMajor A[Nk, C_in]: element (Nk_idx, C_in_idx) at Nk_idx + C_in_idx * Nk?
            // No. Physical data is RowMajor[Nk, C_in]: element (n, ci) at n*C_in + ci.
            // ColumnMajor[M=C_in, K=Nk]: element (m=ci, k=n) at ci + n*C_in = n*C_in + ci.
            // Matches! lda = C_in (leading dimension for ColumnMajor = M = C_in)
            lda[k] = C_in;
            ldb[k] = C_out;
            ldd[k] = C_out;
        }

        runGroupedGemm<GradWeightGemm>(
            problems, ptr_A, ptr_B, ptr_D, lda, ldb, ldd, 1.0f, 0.0f, stream);
    }

    // Reshape grad_W from [K, C_in, C_out] to [C_out, C_in, k0, k1, k2]
    auto ks           = topo.kernel_size;
    auto grad_weights = grad_W_flat.reshape({ks[0], ks[1], ks[2], C_in, C_out})
                            .permute({4, 3, 0, 1, 2})
                            .contiguous();

    return {grad_features, grad_weights};
}

// =============================================================================
// Transposed convolution -- same GEMM structure, different topology
// =============================================================================

torch::Tensor
groupedGemmSparseConvTranspose(torch::Tensor features,
                               torch::Tensor weights,
                               GroupedGemmTopology const &topo) {
    checkGroupedGemmPreconditions(features, weights, topo, "groupedGemmSparseConvTranspose");
    TORCH_CHECK(topo.direction == ConvDirection::Transposed,
                "groupedGemmSparseConvTranspose requires topology with direction=Transposed");

    // The GEMM structure is identical to forward -- only the topology differs.
    // Temporarily override direction check by calling the implementation directly.
    c10::cuda::CUDAGuard device_guard(features.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    int64_t const O     = topo.output_total_voxels;
    int64_t const K     = topo.kernel_volume;
    int64_t const C_in  = weights.size(1);
    int64_t const C_out = weights.size(0);
    int64_t const TP    = topo.total_pairs;

    auto W      = weights.permute({2, 3, 4, 1, 0}).reshape({K, C_in, C_out}).contiguous();
    auto output = torch::zeros({O, C_out}, features.options());

    if (O == 0 || K == 0 || TP == 0)
        return output;

    auto buf_A = torch::empty({TP, C_in}, features.options());
    launchGather(features, buf_A, topo.gather_indices, TP, C_in, stream);

    auto buf_D   = torch::empty({TP, C_out}, features.options());
    auto off_acc = topo.offsets.accessor<int64_t, 1>();

    std::vector<cutlass::gemm::GemmCoord> problems(K);
    std::vector<float *> ptr_A(K), ptr_B(K), ptr_D(K);
    std::vector<int64_t> lda(K), ldb(K), ldd(K);

    float *buf_A_ptr = buf_A.data_ptr<float>();
    float *W_ptr     = W.data_ptr<float>();
    float *buf_D_ptr = buf_D.data_ptr<float>();

    for (int64_t k = 0; k < K; ++k) {
        int64_t Nk  = off_acc[k + 1] - off_acc[k];
        problems[k] = {static_cast<int>(Nk), static_cast<int>(C_out), static_cast<int>(C_in)};
        ptr_A[k]    = buf_A_ptr + off_acc[k] * C_in;
        ptr_B[k]    = W_ptr + k * C_in * C_out;
        ptr_D[k]    = buf_D_ptr + off_acc[k] * C_out;
        lda[k]      = C_in;
        ldb[k]      = C_out;
        ldd[k]      = C_out;
    }

    runGroupedGemm<ForwardGemm>(problems, ptr_A, ptr_B, ptr_D, lda, ldb, ldd, 1.0f, 0.0f, stream);

    launchScatterAdd(buf_D, output, topo.scatter_indices, TP, C_out, stream);

    return output;
}

std::tuple<torch::Tensor, torch::Tensor>
groupedGemmSparseConvTransposeBackward(torch::Tensor grad_output,
                                       torch::Tensor features,
                                       torch::Tensor weights,
                                       GroupedGemmTopology const &topo) {
    checkGroupedGemmPreconditions(
        features, weights, topo, "groupedGemmSparseConvTransposeBackward");
    TORCH_CHECK(topo.direction == ConvDirection::Transposed,
                "groupedGemmSparseConvTransposeBackward requires direction=Transposed");
    TORCH_CHECK(grad_output.dim() == 2 && grad_output.size(0) == topo.output_total_voxels,
                "grad_output shape mismatch");
    TORCH_CHECK(grad_output.is_contiguous(), "grad_output must be contiguous");
    TORCH_CHECK(grad_output.scalar_type() == torch::kFloat32, "grad_output must be float32");

    c10::cuda::CUDAGuard device_guard(features.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    int64_t const F     = topo.feature_total_voxels;
    int64_t const O     = topo.output_total_voxels;
    int64_t const K     = topo.kernel_volume;
    int64_t const C_in  = weights.size(1);
    int64_t const C_out = weights.size(0);
    int64_t const TP    = topo.total_pairs;

    auto W = weights.permute({2, 3, 4, 1, 0}).reshape({K, C_in, C_out}).contiguous();

    auto grad_features = torch::zeros({F, C_in}, features.options());
    auto grad_W_flat   = torch::zeros({K, C_in, C_out}, features.options());

    if (O == 0 || K == 0 || TP == 0) {
        auto ks           = topo.kernel_size;
        auto grad_weights = grad_W_flat.reshape({ks[0], ks[1], ks[2], C_in, C_out})
                                .permute({4, 3, 0, 1, 2})
                                .contiguous();
        return {grad_features, grad_weights};
    }

    auto off_acc = topo.offsets.accessor<int64_t, 1>();

    auto feat_buf = torch::empty({TP, C_in}, features.options());
    auto grad_buf = torch::empty({TP, C_out}, features.options());
    launchGather(features, feat_buf, topo.gather_indices, TP, C_in, stream);
    launchGather(grad_output, grad_buf, topo.scatter_indices, TP, C_out, stream);

    // grad_features
    auto grad_feat_buf = torch::empty({TP, C_in}, features.options());
    {
        std::vector<cutlass::gemm::GemmCoord> problems(K);
        std::vector<float *> ptr_A(K), ptr_B(K), ptr_D(K);
        std::vector<int64_t> lda_v(K), ldb_v(K), ldd_v(K);

        float *gd_ptr = grad_buf.data_ptr<float>();
        float *W_ptr  = W.data_ptr<float>();
        float *gf_ptr = grad_feat_buf.data_ptr<float>();

        for (int64_t k = 0; k < K; ++k) {
            int64_t Nk  = off_acc[k + 1] - off_acc[k];
            problems[k] = {static_cast<int>(Nk), static_cast<int>(C_in), static_cast<int>(C_out)};
            ptr_A[k]    = gd_ptr + off_acc[k] * C_out;
            ptr_B[k]    = W_ptr + k * C_in * C_out;
            ptr_D[k]    = gf_ptr + off_acc[k] * C_in;
            lda_v[k]    = C_out;
            ldb_v[k]    = C_out;
            ldd_v[k]    = C_in;
        }
        runGroupedGemm<GradFeatGemm>(
            problems, ptr_A, ptr_B, ptr_D, lda_v, ldb_v, ldd_v, 1.0f, 0.0f, stream);
    }
    launchScatterAdd(grad_feat_buf, grad_features, topo.gather_indices, TP, C_in, stream);

    // grad_weights
    {
        std::vector<cutlass::gemm::GemmCoord> problems(K);
        std::vector<float *> ptr_A(K), ptr_B(K), ptr_D(K);
        std::vector<int64_t> lda_v(K), ldb_v(K), ldd_v(K);

        float *fb_ptr = feat_buf.data_ptr<float>();
        float *gd_ptr = grad_buf.data_ptr<float>();
        float *gw_ptr = grad_W_flat.data_ptr<float>();

        for (int64_t k = 0; k < K; ++k) {
            int64_t Nk  = off_acc[k + 1] - off_acc[k];
            problems[k] = {static_cast<int>(C_in), static_cast<int>(C_out), static_cast<int>(Nk)};
            ptr_A[k]    = fb_ptr + off_acc[k] * C_in;
            ptr_B[k]    = gd_ptr + off_acc[k] * C_out;
            ptr_D[k]    = gw_ptr + k * C_in * C_out;
            lda_v[k]    = C_in;
            ldb_v[k]    = C_out;
            ldd_v[k]    = C_out;
        }
        runGroupedGemm<GradWeightGemm>(
            problems, ptr_A, ptr_B, ptr_D, lda_v, ldb_v, ldd_v, 1.0f, 0.0f, stream);
    }

    auto ks           = topo.kernel_size;
    auto grad_weights = grad_W_flat.reshape({ks[0], ks[1], ks[2], C_in, C_out})
                            .permute({4, 3, 0, 1, 2})
                            .contiguous();

    return {grad_features, grad_weights};
}

} // namespace ops
} // namespace detail
} // namespace fvdb
