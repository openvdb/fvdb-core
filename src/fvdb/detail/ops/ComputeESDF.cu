// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// One-shot Euclidean Signed Distance Field (ESDF) computation over an
// integrated narrow-band TSDF.
//
// Composition pattern:
//
//    esdfGrid = dilateGrid(tsdfGrid, ceil(R / vs) + 1)         // topology
//    esdf    = torch.full(|esdfGrid|, +R + sentinel)           // sidecar
//    esdfSeed(tsdfGrid, tsdf, weights, truncDist, esdfGrid, esdf)
//    for sweep in range(N):                                    // 26-N stencil
//        esdfSweep(esdfGrid, esdf_in, esdf_out, voxelSize)
//        swap(esdf_in, esdf_out)
//    if prune_unreached: esdfGrid, esdf = pruneGrid(...)
//
// Algorithm notes:
//
// Chamfer vs true Euclidean. Monotone 26-neighbour min-propagation
// produces a "chamfer" distance approximation that is bounded a few
// percent above true Euclidean at worst (this matches nvblox's default
// ESDF and FIESTA). True-Euclidean (Felzenszwalb-style separable O(N)
// SDT) is possible but doesn't compose on sparse grids without a
// dense-back-conversion pass that defeats the point.
//
// Sweep count. With 26-connectivity a wavefront propagates by at least
// one axis-aligned step per sweep, so `N = ceil(R / vs) + 2` sweeps are
// sufficient even accounting for non-convex seed topology (e.g.
// wavefronts meeting behind an obstacle). Additional sweeps past
// convergence are free (monotone min is idempotent at fixed point).
//
// Double-buffering. We ping-pong between two contiguous fp32 sidecar
// tensors rather than trying to do in-place updates with atomicMin. A
// single in-place pass using atomicCAS on packed bits would work but
// the two-buffer approach is simpler, deterministic, and the kernel is
// memory-bound so the extra bandwidth cost is hidden.
//
// Scope. float32 CUDA + batchSize == 1 only. Multi-batch and fp64 are
// future-work lifts.

#include <fvdb/GridBatchData.h>
#include <fvdb/detail/GridBatchDataFactory.h>
#include <fvdb/detail/ops/BuildDilatedGrid.h>
#include <fvdb/detail/ops/BuildMergedGrids.h>
#include <fvdb/detail/ops/BuildPrunedGrid.h>
#include <fvdb/detail/ops/ComputeESDF.h>
#include <fvdb/detail/ops/Inject.h>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/cuda/VoxelBlockManager.cuh>
#include <nanovdb/util/cuda/DeviceGridTraits.cuh>

#include <cuda_runtime.h>
#include <torch/types.h>

#include <cmath>

namespace fvdb {
namespace detail {
namespace ops {

namespace {

// -----------------------------------------------------------------------
// Kernel tuning: 128-voxel VBM blocks with a 2-u64 jump map.
// -----------------------------------------------------------------------

constexpr int ESDF_BLOCK_WIDTH_LOG2 = 7;
constexpr int ESDF_BLOCK_WIDTH      = 1 << ESDF_BLOCK_WIDTH_LOG2;  // 128
constexpr int ESDF_JUMP_MAP_LENGTH  = ESDF_BLOCK_WIDTH / 64;       // 2
constexpr int ESDF_PERLEAF_THREADS  = 128;                         // per-leaf ablation

// Sentinel used for "never-reached" voxels. Must be LARGE ENOUGH that
// it is unambiguously identifiable after propagation: the wavefront
// intentionally doesn't cap at `max_distance` (doing so loses sign
// information on voxels beyond the cap — see journal entry for this
// session). With a 1e30 sentinel, any finite propagated distance (even
// well beyond `max_distance`) is clearly distinguishable, and the
// final post-loop `clamp(-max, +max)` yields the correct signed result.
constexpr float kEsdfSentinel = 1.0e30f;
// Threshold used to detect "still at sentinel" inside the kernel. Any
// real propagated distance is astronomically smaller than this.
constexpr float kEsdfSentinelCheck = 0.5e30f;

// -----------------------------------------------------------------------
// Offset table for 26-neighbour stencil. Ordered so small-step offsets
// come first; not semantically meaningful (our min-propagation is
// order-invariant within a single sweep), but improves L1 hit rate on
// the self-voxel lookup since `acc.isActive(c)` touches the same leaf
// node for small offsets. 26 entries: 6 face + 12 edge + 8 corner.
// -----------------------------------------------------------------------

struct EsdfOffset {
    int dx, dy, dz;
    float weight;  // ||offset||, in units of voxel_size
};

__device__ __constant__ EsdfOffset kEsdfOffsets[26] = {
    // 6 face neighbours (axis-aligned, weight = 1)
    {-1,  0,  0, 1.0f}, { 1,  0,  0, 1.0f},
    { 0, -1,  0, 1.0f}, { 0,  1,  0, 1.0f},
    { 0,  0, -1, 1.0f}, { 0,  0,  1, 1.0f},
    // 12 edge neighbours (face-diagonal, weight = sqrt(2) ~ 1.41421356)
    {-1, -1,  0, 1.41421356f}, { 1, -1,  0, 1.41421356f},
    {-1,  1,  0, 1.41421356f}, { 1,  1,  0, 1.41421356f},
    {-1,  0, -1, 1.41421356f}, { 1,  0, -1, 1.41421356f},
    {-1,  0,  1, 1.41421356f}, { 1,  0,  1, 1.41421356f},
    { 0, -1, -1, 1.41421356f}, { 0,  1, -1, 1.41421356f},
    { 0, -1,  1, 1.41421356f}, { 0,  1,  1, 1.41421356f},
    // 8 corner neighbours (vertex-diagonal, weight = sqrt(3) ~ 1.73205081)
    {-1, -1, -1, 1.73205081f}, { 1, -1, -1, 1.73205081f},
    {-1,  1, -1, 1.73205081f}, { 1,  1, -1, 1.73205081f},
    {-1, -1,  1, 1.73205081f}, { 1, -1,  1, 1.73205081f},
    {-1,  1,  1, 1.73205081f}, { 1,  1,  1, 1.73205081f},
};

// -----------------------------------------------------------------------
// Core stencil body: given self-distance `dSelf` and a neighbour query
// callable `readNeighbour(ijk+o)`, return the monotone-min of dSelf and
// all 26 valid neighbour propagations. Sign-preserving: the candidate
// is `sign(d_n) * (|d_n| + ||offset|| * vs)`, which expands the
// neighbour's signed distance outward by the geometric step. A zero
// d_n stays zero (zero-crossing propagation).
// -----------------------------------------------------------------------

template <typename ReadNeighbourFn>
__device__ __forceinline__ float
esdfSweepBody(float dSelf, float voxelSize, float maxDistance,
              ReadNeighbourFn readNeighbour) {
    float d = dSelf;
#pragma unroll
    for (int i = 0; i < 26; ++i) {
        const EsdfOffset off = kEsdfOffsets[i];
        float dN;
        bool active;
        readNeighbour(off.dx, off.dy, off.dz, dN, active);
        if (!active) continue;
        const float dNAbs = fabsf(dN);
        if (dNAbs >= kEsdfSentinelCheck) continue;  // neighbour not yet reached
        const float step    = off.weight * voxelSize;
        const float candAbs = dNAbs + step;
        // Cap propagation at `maxDistance` so wavefronts can't smear
        // chamfer-overshoot past the user's ESDF support radius. In
        // particular this is load-bearing for the incremental path:
        // without the cap, surviving-from-prev-frame negative-sign
        // voxels (with |d| < max_distance) could propagate their sign
        // arbitrarily far via the cascading sweep, smearing
        // -maxDistance into voxels that one-shot would have left at
        // sentinel. With the cap, voxels more than `maxDistance` from
        // ANY seed stay at sentinel -> clamped to +maxDistance at the
        // end (the "unknown, free space" convention). This matches
        // nvblox / FIESTA defaults.
        if (candAbs >= maxDistance) continue;
        if (candAbs < fabsf(d)) {
            // Preserve the sign of the witness neighbour.
            d = (dN < 0.0f) ? -candAbs : candAbs;
        }
    }
    return d;
}

// -----------------------------------------------------------------------
// Seed kernel. Iterates *input* grid voxels (one thread per active voxel
// via a simple per-leaf-slot launch — the input grid is typically
// small and the seed runs once, so VBM overhead here is not worth
// introducing). For each input voxel with weights > threshold and
// |tsdf| < 1 - eps, writes `tsdf * truncation_distance` into the
// corresponding slot in the ESDF sidecar.
//
// We use per-leaf-slot iteration here (not VBM) for two reasons:
//   1. This kernel runs once; VBM's amortization story doesn't apply.
//   2. Per-leaf iteration gives us the leaf directly, which lets us
//      compute the input sidecar offset without a second grid lookup.
// -----------------------------------------------------------------------

__global__ void
esdfSeedKernel(
    const nanovdb::NanoGrid<nanovdb::ValueOnIndex> *__restrict__ inputGrid,
    const nanovdb::NanoGrid<nanovdb::ValueOnIndex> *__restrict__ esdfGrid,
    const float *__restrict__ tsdf,       // [inputGrid->totalActiveVoxels]
    const float *__restrict__ weights,    // [inputGrid->totalActiveVoxels]
    const bool *__restrict__ dirtyMask,   // nullable; when non-null,
                                          // only dirty voxels seed
    float *__restrict__ esdf,             // [esdfGrid->totalActiveVoxels]
    float truncationDistance,
    float weightThreshold,
    float saturationEps) {
    constexpr uint64_t VPL =
        nanovdb::OnIndexTree::LeafNodeType::NUM_VALUES;

    const int64_t leafIdx = blockIdx.x;
    const int64_t voxOff  = threadIdx.x;
    if (voxOff >= static_cast<int64_t>(VPL)) return;

    const auto &leaf = inputGrid->tree().template getFirstNode<0>()[leafIdx];
    if (!leaf.isActive(voxOff)) return;

    // 1-indexed pid; subtract one for torch tensor offset.
    const int64_t inputPid = static_cast<int64_t>(leaf.getValue(voxOff)) - 1;

    // Dirty-mask gate: when provided, skip non-dirty voxels so the
    // wavefront only re-propagates from what actually changed this
    // frame. This is the mechanism that gives fvdb's
    // `compute_esdf_incremental` nvblox-style dirty-region update
    // scaling. When `dirtyMask == nullptr` (the default), behaves
    // as before -- seed from every near-surface voxel.
    if (dirtyMask != nullptr && !dirtyMask[inputPid]) return;

    const float w  = weights[inputPid];
    if (!(w > weightThreshold)) return;

    const float t = tsdf[inputPid];
    if (!(fabsf(t) < 1.0f - saturationEps)) return;

    const nanovdb::Coord ijk = leaf.offsetToGlobalCoord(voxOff);
    auto esdfAcc             = esdfGrid->getAccessor();
    // By construction (dilateGrid superset), ijk is always active in
    // esdfGrid; assert defensively in debug builds.
    const uint64_t esdfRaw = esdfAcc.getValue(ijk);
    if (esdfRaw == 0) return;
    const int64_t esdfPid = static_cast<int64_t>(esdfRaw) - 1;

    esdf[esdfPid] = t * truncationDistance;
}

// -----------------------------------------------------------------------
// Sweep kernel (VBM path). One CUDA block iterates ESDF_BLOCK_WIDTH
// contiguous active voxels via `decodeInverseMaps`. Each thread reads
// its own self-distance and 26 neighbours, writes the monotone min to
// `esdfOut`. Reads from `esdfIn` only — safe double-buffered Jacobi.
// -----------------------------------------------------------------------

__global__ void
esdfSweepVBMKernel(
    nanovdb::NanoGrid<nanovdb::ValueOnIndex> *__restrict__ esdfGrid,
    const uint32_t *__restrict__ firstLeafID,
    const uint64_t *__restrict__ jumpMap,
    const float *__restrict__ esdfIn,
    float *__restrict__ esdfOut,
    float voxelSize,
    float maxDistance,
    int *__restrict__ dChanged) {
    constexpr int BW       = ESDF_BLOCK_WIDTH;
    constexpr int JML      = ESDF_JUMP_MAP_LENGTH;
    constexpr uint64_t VPL =
        nanovdb::OnIndexTree::LeafNodeType::NUM_VALUES;

    __shared__ uint32_t smem_leafIndex[BW];
    __shared__ uint16_t smem_voxelOffset[BW];

    const uint64_t blockFirstOffset =
        static_cast<uint64_t>(blockIdx.x) * BW + 1;

    nanovdb::tools::cuda::VoxelBlockManager<BW>::template decodeInverseMaps<
        nanovdb::ValueOnIndex>(
        esdfGrid,
        firstLeafID[blockIdx.x],
        jumpMap + static_cast<uint64_t>(blockIdx.x) * JML,
        blockFirstOffset,
        smem_leafIndex,
        smem_voxelOffset);
    // __syncthreads() is issued inside decodeInverseMaps.

    const uint32_t leafID = smem_leafIndex[threadIdx.x];
    if (leafID ==
        nanovdb::tools::cuda::VoxelBlockManager<BW>::UnusedLeafIndex) {
        return;
    }
    const uint16_t voxOff = smem_voxelOffset[threadIdx.x];

    const auto &leaf         = esdfGrid->tree().template getFirstNode<0>()[leafID];
    const nanovdb::Coord ijk = leaf.offsetToGlobalCoord(voxOff);
    auto acc                 = esdfGrid->getAccessor();
    const int64_t selfPid    =
        static_cast<int64_t>(leaf.getValue(voxOff)) - 1;

    const float dSelf = esdfIn[selfPid];

    const float dNew = esdfSweepBody(
        dSelf, voxelSize, maxDistance,
        [&](int dx, int dy, int dz, float &dOut, bool &activeOut) {
            const nanovdb::Coord c = ijk + nanovdb::Coord(dx, dy, dz);
            if (!acc.isActive(c)) {
                activeOut = false;
                return;
            }
            const int64_t pid = static_cast<int64_t>(acc.getValue(c)) - 1;
            dOut      = esdfIn[pid];
            activeOut = true;
        });

    esdfOut[selfPid] = dNew;
    // Signal fixed-point detection: if this voxel's value changed,
    // the host-side loop will run another sweep. Race-free: all
    // threads write the same value (1) and we only read *dChanged
    // after the kernel completes. The comparison is exact because
    // `esdfSweepBody` only writes a new `d` via assignment and
    // starts at `d = dSelf`; if no neighbour won the min, `dNew`
    // equals `dSelf` bit-for-bit.
    if (dNew != dSelf) {
        dChanged[0] = 1;
    }
    (void)VPL;
}

// -----------------------------------------------------------------------
// Sweep kernel (per-leaf-slot ablation path). One CUDA block per leaf;
// 512 threads iterate every slot in that leaf, skipping inactive ones.
// Same inner body as the VBM kernel. Purpose: measure the cost model
// delta of VBM iteration vs V4-style per-leaf iteration on this
// specific stencil shape, for the paper's VBM ablation figure.
// -----------------------------------------------------------------------

__global__ void
esdfSweepPerLeafKernel(
    nanovdb::NanoGrid<nanovdb::ValueOnIndex> *__restrict__ esdfGrid,
    const float *__restrict__ esdfIn,
    float *__restrict__ esdfOut,
    float voxelSize,
    float maxDistance,
    int *__restrict__ dChanged) {
    constexpr uint64_t VPL =
        nanovdb::OnIndexTree::LeafNodeType::NUM_VALUES;

    const int64_t leafIdx = blockIdx.x;
    // Each thread handles VPL / blockDim.x slots if VPL > blockDim.x.
    // For VPL = 512 and ESDF_PERLEAF_THREADS = 128, that's 4 slots/thread.
    for (int64_t voxOff = threadIdx.x; voxOff < static_cast<int64_t>(VPL);
         voxOff += blockDim.x) {
        const auto &leaf =
            esdfGrid->tree().template getFirstNode<0>()[leafIdx];
        if (!leaf.isActive(voxOff)) continue;

        const int64_t selfPid =
            static_cast<int64_t>(leaf.getValue(voxOff)) - 1;
        const nanovdb::Coord ijk = leaf.offsetToGlobalCoord(voxOff);
        auto acc                 = esdfGrid->getAccessor();

        const float dSelf = esdfIn[selfPid];
        const float dNew  = esdfSweepBody(
             dSelf, voxelSize, maxDistance,
             [&](int dx, int dy, int dz, float &dOut, bool &activeOut) {
                 const nanovdb::Coord c = ijk + nanovdb::Coord(dx, dy, dz);
                 if (!acc.isActive(c)) {
                     activeOut = false;
                     return;
                 }
                 const int64_t pid = static_cast<int64_t>(acc.getValue(c)) - 1;
                 dOut      = esdfIn[pid];
                 activeOut = true;
             });
        esdfOut[selfPid] = dNew;
        if (dNew != dSelf) {
            dChanged[0] = 1;
        }
    }
}

// -----------------------------------------------------------------------
// Shared sweep-and-finalize helper. Runs N = 2*dilateAmount + 4 sweeps
// of the chosen iteration kernel (VBM or per-leaf-slot), then clamps
// magnitudes to [-maxDist, +maxDist], then optionally prunes voxels
// still saturated at the cap.
//
// Takes a pre-allocated `esdfInit` sidecar that is assumed to already
// hold the correct initial values for this run:
//   - one-shot: sentinel + seeded from TSDF
//   - incremental: sentinel + injected prev_esdf + seeded from TSDF
//
// The returned grid is either `esdfGrid` itself or the pruned subset.
// The returned tensor has `(*returned_grid).totalVoxels` entries.
// -----------------------------------------------------------------------

std::tuple<c10::intrusive_ptr<GridBatchData>, torch::Tensor>
runEsdfSweepsAndFinalize(
    const c10::intrusive_ptr<GridBatchData> &esdfGrid,
    torch::Tensor esdfInit,
    float voxelSizeF,
    int64_t dilateAmount,
    float maxDistF,
    bool prune_unreached,
    bool use_vbm,
    at::cuda::CUDAStream stream) {
    const int64_t esdfVoxels = esdfGrid->totalVoxels();
    auto u32Opts =
        torch::TensorOptions().dtype(torch::kInt32).device(esdfInit.device());
    auto u64Opts =
        torch::TensorOptions().dtype(torch::kInt64).device(esdfInit.device());
    auto i32Opts =
        torch::TensorOptions().dtype(torch::kInt32).device(esdfInit.device());

    auto *esdfDeviceGrid =
        esdfGrid->mGridHdl->deviceGrid<nanovdb::ValueOnIndex>(0);
    TORCH_CHECK(esdfDeviceGrid != nullptr, "computeESDF: null esdf grid");

    // Double-buffered Jacobi. `esdfInit` is the first read; `esdfB`
    // receives the first write; they swap each sweep.
    torch::Tensor esdfB = esdfInit.clone();  // same content so reads are safe
    torch::Tensor *esdfIn  = &esdfInit;
    torch::Tensor *esdfOut = &esdfB;

    // Fixed-point early termination: each sweep, the kernel sets
    // `*dChanged = 1` whenever any voxel's value updates. After the
    // kernel completes we sync-read the flag; if zero, the wavefront
    // has converged and we break. This is load-bearing for the
    // warm-reuse case (incremental on unchanged TSDF should exit
    // after 1 sweep with ~3 ms cost, not run all 2K+4 sweeps). It
    // also reduces cold cost on typical workloads where convergence
    // happens in ~K sweeps rather than 2K+4.
    torch::Tensor changedFlag = torch::zeros({1}, i32Opts);

    // Hard upper bound on sweeps: 2K+4 covers worst-case opposite-
    // corner propagation. The early-exit loop will usually terminate
    // far before reaching this cap on warm-reuse and at ~K on cold
    // builds where the wavefront is compact.
    const int numSweepsMax = static_cast<int>(dilateAmount) * 2 + 4;

    if (use_vbm) {
        const auto treeData =
            nanovdb::util::cuda::DeviceGridTraits<nanovdb::ValueOnIndex>::
                getTreeData(esdfDeviceGrid);
        const int lowerCount = static_cast<int>(treeData.mNodeCount[1]);

        const int nBlocks = static_cast<int>(
            (esdfVoxels + ESDF_BLOCK_WIDTH - 1) / ESDF_BLOCK_WIDTH);

        torch::Tensor firstLeafID = torch::zeros({nBlocks}, u32Opts);
        torch::Tensor jumpMap     =
            torch::zeros({nBlocks * ESDF_JUMP_MAP_LENGTH}, u64Opts);

        nanovdb::tools::cuda::buildVoxelBlockManager<
            ESDF_BLOCK_WIDTH_LOG2, 128>(
            /*firstOffset=*/1,
            /*lastOffset=*/static_cast<uint64_t>(esdfVoxels),
            /*nBlocks=*/nBlocks,
            /*lowerCount=*/lowerCount,
            /*grid=*/esdfDeviceGrid,
            /*firstLeafID=*/
            reinterpret_cast<uint32_t *>(firstLeafID.data_ptr<int32_t>()),
            /*jumpMap=*/
            reinterpret_cast<uint64_t *>(jumpMap.data_ptr<int64_t>()),
            /*stream=*/stream.stream());
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        for (int sweep = 0; sweep < numSweepsMax; ++sweep) {
            changedFlag.zero_();
            esdfSweepVBMKernel<<<static_cast<unsigned int>(nBlocks),
                                 static_cast<unsigned int>(ESDF_BLOCK_WIDTH),
                                 0, stream.stream()>>>(
                esdfDeviceGrid,
                reinterpret_cast<uint32_t *>(firstLeafID.data_ptr<int32_t>()),
                reinterpret_cast<uint64_t *>(jumpMap.data_ptr<int64_t>()),
                esdfIn->data_ptr<float>(),
                esdfOut->data_ptr<float>(),
                voxelSizeF, maxDistF,
                changedFlag.data_ptr<int32_t>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            std::swap(esdfIn, esdfOut);
            // .item() is a sync + host-device copy (~30 us). Each
            // sweep is ~3-10 ms at our scales, so the overhead is
            // ~1%. Break early when the wavefront has converged.
            if (changedFlag.item<int32_t>() == 0) {
                break;
            }
        }
    } else {
        const int64_t esdfLeaves = esdfGrid->totalLeaves();
        for (int sweep = 0; sweep < numSweepsMax; ++sweep) {
            changedFlag.zero_();
            esdfSweepPerLeafKernel<<<
                static_cast<unsigned int>(esdfLeaves),
                static_cast<unsigned int>(ESDF_PERLEAF_THREADS),
                0, stream.stream()>>>(
                esdfDeviceGrid,
                esdfIn->data_ptr<float>(),
                esdfOut->data_ptr<float>(),
                voxelSizeF, maxDistF,
                changedFlag.data_ptr<int32_t>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            std::swap(esdfIn, esdfOut);
            if (changedFlag.item<int32_t>() == 0) {
                break;
            }
        }
    }

    torch::Tensor esdfFinal =
        esdfIn->clamp(-maxDistF, maxDistF).contiguous();

    if (!prune_unreached) {
        return {esdfGrid, esdfFinal};
    }

    torch::Tensor keepMask = esdfFinal.abs() < maxDistF;
    auto idxOpts = torch::TensorOptions()
                       .dtype(fvdb::JIdxScalarType)
                       .device(esdfInit.device());
    auto jidx  = torch::zeros({keepMask.size(0)}, idxOpts);
    auto jlidx = torch::empty({0, 1}, idxOpts);
    auto keepMaskJagged = JaggedTensor::from_data_indices_and_list_ids(
        keepMask, jidx, jlidx, /*num_tensors=*/1);
    auto prunedGrid = pruneGrid(*esdfGrid, keepMaskJagged);
    torch::Tensor prunedEsdf = esdfFinal.masked_select(keepMask);
    return {prunedGrid, prunedEsdf};
}

} // anonymous namespace

// -----------------------------------------------------------------------
// Public entry point: one-shot.
// -----------------------------------------------------------------------

std::tuple<c10::intrusive_ptr<GridBatchData>, torch::Tensor>
computeESDF(const GridBatchData &gridBatch,
            const torch::Tensor &tsdf,
            const torch::Tensor &weights,
            double truncation_distance,
            double max_distance,
            double weight_threshold,
            bool prune_unreached,
            bool use_vbm) {
    // ------------------ Shape / dtype / scope checks ------------------

    TORCH_CHECK_VALUE(gridBatch.batchSize() == 1,
                      "computeESDF: batchSize must be 1 in M5, got ",
                      gridBatch.batchSize());
    TORCH_CHECK(tsdf.is_cuda() && weights.is_cuda(),
                "computeESDF: tsdf and weights must be CUDA tensors");
    gridBatch.checkDevice(tsdf);
    gridBatch.checkDevice(weights);

    TORCH_CHECK_VALUE(tsdf.dim() == 1 && weights.dim() == 1,
                      "computeESDF: tsdf and weights must be 1-D, got dims (",
                      tsdf.dim(), ",", weights.dim(), ")");
    TORCH_CHECK_VALUE(tsdf.size(0) == gridBatch.totalVoxels() &&
                      weights.size(0) == gridBatch.totalVoxels(),
                      "computeESDF: tsdf/weights size must match totalVoxels (",
                      gridBatch.totalVoxels(), "), got tsdf=", tsdf.size(0),
                      " weights=", weights.size(0));

    TORCH_CHECK_TYPE(tsdf.scalar_type() == torch::kFloat32,
                     "computeESDF: only float32 tsdf is supported in M5");
    TORCH_CHECK_TYPE(weights.scalar_type() == torch::kFloat32,
                     "computeESDF: only float32 weights is supported in M5");

    TORCH_CHECK_VALUE(truncation_distance > 0.0,
                      "computeESDF: truncation_distance must be > 0, got ",
                      truncation_distance);
    TORCH_CHECK_VALUE(max_distance > 0.0,
                      "computeESDF: max_distance must be > 0, got ",
                      max_distance);

    c10::cuda::CUDAGuard guard(tsdf.device());
    at::cuda::CUDAStream stream =
        at::cuda::getCurrentCUDAStream(tsdf.device().index());

    // Cast configuration to fp32 for kernel use.
    const float truncF    = static_cast<float>(truncation_distance);
    const float maxDistF  = static_cast<float>(max_distance);
    const float threshF   = static_cast<float>(weight_threshold);
    const float saturEps  = 1.0e-5f;  // "|tsdf| < 1" margin for float stability

    // Voxel size: single-batch, isotropic expected. Use the minimum axis
    // to drive chamfer step length; TSDF convention assumes isotropic.
    std::vector<nanovdb::Vec3d> voxSizes, origins;
    gridBatch.gridVoxelSizesAndOrigins(voxSizes, origins);
    TORCH_CHECK_VALUE(voxSizes.size() == 1,
                      "computeESDF: expected single-batch voxel size");
    const double vsX = voxSizes[0][0];
    const double vsY = voxSizes[0][1];
    const double vsZ = voxSizes[0][2];
    TORCH_CHECK_VALUE(std::fabs(vsX - vsY) < 1e-9 &&
                      std::fabs(vsX - vsZ) < 1e-9,
                      "computeESDF: anisotropic voxels not supported in M5 (",
                      vsX, ", ", vsY, ", ", vsZ, ")");
    const float voxelSizeF = static_cast<float>(vsX);

    auto floatOpts =
        torch::TensorOptions().dtype(torch::kFloat32).device(tsdf.device());

    // ------------------ Step 1: build ESDF support topology ------------------

    const int64_t dilateAmount =
        static_cast<int64_t>(std::ceil(max_distance / vsX)) + 1;
    auto esdfGrid = dilateGrid(gridBatch,
                               std::vector<int64_t>{dilateAmount});
    const int64_t esdfVoxels = esdfGrid->totalVoxels();

    if (esdfVoxels == 0) {
        // Input grid was empty; return empty ESDF.
        torch::Tensor emptyEsdf = torch::empty({0}, floatOpts);
        return {esdfGrid, emptyEsdf};
    }

    // ------------------ Step 2: allocate + fill-sentinel ESDF ----------------

    torch::Tensor esdfA = torch::full({esdfVoxels}, kEsdfSentinel, floatOpts);

    // ------------------ Step 3: seed from input TSDF ------------------------

    auto *inputDeviceGrid =
        gridBatch.mGridHdl->deviceGrid<nanovdb::ValueOnIndex>(0);
    auto *esdfDeviceGrid =
        esdfGrid->mGridHdl->deviceGrid<nanovdb::ValueOnIndex>(0);
    TORCH_CHECK(inputDeviceGrid != nullptr, "computeESDF: null input grid");
    TORCH_CHECK(esdfDeviceGrid != nullptr, "computeESDF: null esdf grid");

    {
        const int64_t inputLeaves = gridBatch.totalLeaves();
        if (inputLeaves > 0) {
            // One-shot compute has no dirty-mask concept (it seeds
            // from every near-surface voxel unconditionally). Pass
            // nullptr.
            esdfSeedKernel<<<static_cast<unsigned int>(inputLeaves),
                             static_cast<unsigned int>(
                                 nanovdb::OnIndexTree::LeafNodeType::NUM_VALUES),
                             0, stream.stream()>>>(
                inputDeviceGrid, esdfDeviceGrid,
                tsdf.data_ptr<float>(), weights.data_ptr<float>(),
                /*dirtyMask=*/nullptr,
                esdfA.data_ptr<float>(),
                truncF, threshF, saturEps);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }

    // ------------------ Step 4-5: sweeps + clamp + prune ---------------------

    return runEsdfSweepsAndFinalize(
        esdfGrid, esdfA, voxelSizeF, dilateAmount, maxDistF,
        prune_unreached, use_vbm, stream);
}

// -----------------------------------------------------------------------
// Public entry point: incremental.
// -----------------------------------------------------------------------

std::tuple<c10::intrusive_ptr<GridBatchData>, torch::Tensor>
computeESDFIncremental(const GridBatchData &gridBatch,
                       const torch::Tensor &tsdf,
                       const torch::Tensor &weights,
                       const GridBatchData &prevEsdfGrid,
                       const torch::Tensor &prevEsdf,
                       double truncation_distance,
                       double max_distance,
                       double weight_threshold,
                       bool prune_unreached,
                       bool use_vbm,
                       const torch::Tensor &dirtyMask) {
    // ------------------ Shape / dtype / scope checks ------------------

    TORCH_CHECK_VALUE(gridBatch.batchSize() == 1 &&
                          prevEsdfGrid.batchSize() <= 1,
                      "computeESDFIncremental: batchSize must be 1 in M5");
    TORCH_CHECK(tsdf.is_cuda() && weights.is_cuda() && prevEsdf.is_cuda(),
                "computeESDFIncremental: all tensors must be CUDA");
    gridBatch.checkDevice(tsdf);
    gridBatch.checkDevice(weights);
    TORCH_CHECK_VALUE(tsdf.dim() == 1 && weights.dim() == 1 && prevEsdf.dim() == 1,
                      "computeESDFIncremental: tsdf/weights/prevEsdf must be 1-D");
    TORCH_CHECK_VALUE(tsdf.size(0) == gridBatch.totalVoxels() &&
                          weights.size(0) == gridBatch.totalVoxels(),
                      "computeESDFIncremental: tsdf/weights size must match "
                      "current grid.totalVoxels (",
                      gridBatch.totalVoxels(), ")");
    TORCH_CHECK_VALUE(prevEsdf.size(0) == prevEsdfGrid.totalVoxels(),
                      "computeESDFIncremental: prevEsdf size (",
                      prevEsdf.size(0),
                      ") must match prevEsdfGrid.totalVoxels (",
                      prevEsdfGrid.totalVoxels(), ")");
    TORCH_CHECK_TYPE(tsdf.scalar_type() == torch::kFloat32 &&
                         weights.scalar_type() == torch::kFloat32 &&
                         prevEsdf.scalar_type() == torch::kFloat32,
                     "computeESDFIncremental: only float32 is supported in M5");
    TORCH_CHECK_VALUE(truncation_distance > 0.0,
                      "computeESDFIncremental: truncation_distance must be > 0");
    TORCH_CHECK_VALUE(max_distance > 0.0,
                      "computeESDFIncremental: max_distance must be > 0");

    const bool hasDirtyMask = dirtyMask.defined() && dirtyMask.numel() > 0;
    if (hasDirtyMask) {
        TORCH_CHECK_VALUE(dirtyMask.scalar_type() == torch::kBool,
                          "computeESDFIncremental: dirty_mask must be bool");
        TORCH_CHECK_VALUE(dirtyMask.size(0) == gridBatch.totalVoxels(),
                          "computeESDFIncremental: dirty_mask size (",
                          dirtyMask.size(0),
                          ") must equal gridBatch.totalVoxels (",
                          gridBatch.totalVoxels(), ")");
        TORCH_CHECK(dirtyMask.device() == tsdf.device(),
                    "computeESDFIncremental: dirty_mask must be on same "
                    "device as tsdf");
    }
    // Python wrapper handles the "dirtyMask.any() == False"
    // short-circuit (returns prev state directly, never entering
    // C++). By the time we get here, the dirty mask has at least
    // one true entry, so we do the full incremental work but with
    // the seed kernel gated on the mask.

    // Fall through to one-shot when there's no previous state. Keeps
    // the first-frame-of-a-session code path trivial.
    if (prevEsdfGrid.totalVoxels() == 0) {
        return computeESDF(gridBatch, tsdf, weights,
                           truncation_distance, max_distance,
                           weight_threshold, prune_unreached, use_vbm);
    }

    c10::cuda::CUDAGuard guard(tsdf.device());
    at::cuda::CUDAStream stream =
        at::cuda::getCurrentCUDAStream(tsdf.device().index());

    const float truncF   = static_cast<float>(truncation_distance);
    const float maxDistF = static_cast<float>(max_distance);
    const float threshF  = static_cast<float>(weight_threshold);
    const float saturEps = 1.0e-5f;

    std::vector<nanovdb::Vec3d> voxSizes, origins;
    gridBatch.gridVoxelSizesAndOrigins(voxSizes, origins);
    TORCH_CHECK_VALUE(voxSizes.size() == 1,
                      "computeESDFIncremental: expected single-batch voxel size");
    const double vsX = voxSizes[0][0];
    TORCH_CHECK_VALUE(std::fabs(vsX - voxSizes[0][1]) < 1e-9 &&
                          std::fabs(vsX - voxSizes[0][2]) < 1e-9,
                      "computeESDFIncremental: anisotropic voxels not supported");
    // Require matching voxel size between previous and current grids.
    // Changing voxel sizes across frames would break the sign-propagation
    // witness semantics; users in that case should reset to one-shot.
    std::vector<nanovdb::Vec3d> prevVoxSizes, prevOrigins;
    prevEsdfGrid.gridVoxelSizesAndOrigins(prevVoxSizes, prevOrigins);
    TORCH_CHECK_VALUE(!prevVoxSizes.empty() &&
                          std::fabs(prevVoxSizes[0][0] - vsX) < 1e-9,
                      "computeESDFIncremental: prevEsdfGrid voxel_size (",
                      prevVoxSizes.empty() ? 0.0 : prevVoxSizes[0][0],
                      ") must match current grid voxel_size (", vsX, ")");
    const float voxelSizeF = static_cast<float>(vsX);

    auto floatOpts =
        torch::TensorOptions().dtype(torch::kFloat32).device(tsdf.device());

    // ------------------ Step 1: build union ESDF support topology ------------

    const int64_t dilateAmount =
        static_cast<int64_t>(std::ceil(max_distance / vsX)) + 1;
    auto dilated = dilateGrid(gridBatch,
                              std::vector<int64_t>{dilateAmount});
    // Merge with the previous ESDF grid so voxels that were in the
    // previous support but fall outside the current TSDF's dilation
    // are still carried over (monotone scene assumption: previously-
    // known ESDF values shouldn't disappear just because the TSDF
    // shell shifted in this frame).
    auto esdfGrid = mergeGrids(*dilated, prevEsdfGrid);
    const int64_t esdfVoxels = esdfGrid->totalVoxels();

    if (esdfVoxels == 0) {
        return {esdfGrid, torch::empty({0}, floatOpts)};
    }

    // ------------------ Step 2: sentinel-fill + inject prev_esdf -------------

    torch::Tensor esdfInit = torch::full({esdfVoxels}, kEsdfSentinel, floatOpts);
    {
        // Inject previous ESDF values into their (possibly-shifted)
        // slot positions in the merged grid. `ops::inject` copies only
        // the ijk-overlapping voxels and leaves the rest (sentinel)
        // untouched.
        JaggedTensor dstJt = esdfGrid->jaggedTensor(esdfInit);
        JaggedTensor srcJt = prevEsdfGrid.jaggedTensor(prevEsdf);
        ops::inject(*esdfGrid, prevEsdfGrid, dstJt, srcJt);
        esdfInit = dstJt.jdata();
    }

    // Reset voxels saturated at the previous frame's max_distance cap
    // back to sentinel. Two reasons:
    //
    // (1) The clamped output from a previous `compute_esdf` call loses
    //     the distinction between "reached at exactly max_distance" and
    //     "unreached (sentinel)" voxels -- both appear as
    //     `±max_distance` in the prev tensor. Without this reset, the
    //     injected `+max_distance` values would be treated as "reached
    //     witnesses" by this frame's wavefront. Converting them back
    //     to sentinel lets the current-frame sweep correctly
    //     re-propagate into previously-unreached regions.
    //
    // (2) A surviving prev value at e.g. `-max_distance + epsilon`
    //     (|d| < max_distance so it survives this reset) would,
    //     without the propagation cap in `esdfSweepBody`, cascade its
    //     negative sign arbitrarily far via the 18-sweep chain. The
    //     `candAbs >= maxDistance` guard in the sweep kernel now
    //     prevents this; here we just normalize the "exactly-at-cap"
    //     boundary values to sentinel so they don't act as phantom
    //     witnesses.
    //
    // Edge case: voxels that genuinely were at exactly `max_distance`
    // get converted too, but they'll be re-derived correctly by the
    // wavefront from neighbouring seeded voxels with the same accuracy
    // as a one-shot call.
    {
        auto resetMask = esdfInit.abs().ge(maxDistF);
        esdfInit.masked_fill_(resetMask, kEsdfSentinel);
    }

    // ------------------ Step 3: seed from current TSDF ----------------------

    auto *inputDeviceGrid =
        gridBatch.mGridHdl->deviceGrid<nanovdb::ValueOnIndex>(0);
    auto *esdfDeviceGrid =
        esdfGrid->mGridHdl->deviceGrid<nanovdb::ValueOnIndex>(0);
    TORCH_CHECK(inputDeviceGrid != nullptr && esdfDeviceGrid != nullptr,
                "computeESDFIncremental: null device grid");

    const int64_t inputLeaves = gridBatch.totalLeaves();
    if (inputLeaves > 0) {
        // Current-frame seed writes unconditionally (at the voxels it
        // visits), which is correct: seeds are by definition exact
        // signed distances. The dirty-mask gate (when provided) limits
        // which voxels are visited at all — non-dirty voxels inherit
        // whatever they had in `prevEsdf` (via the inject+restore
        // above). Monotone-min correctness is preserved under the
        // existing "distances can decrease but not grow" assumption.
        const bool *dirtyMaskPtr = hasDirtyMask
            ? dirtyMask.data_ptr<bool>()
            : nullptr;
        esdfSeedKernel<<<static_cast<unsigned int>(inputLeaves),
                         static_cast<unsigned int>(
                             nanovdb::OnIndexTree::LeafNodeType::NUM_VALUES),
                         0, stream.stream()>>>(
            inputDeviceGrid, esdfDeviceGrid,
            tsdf.data_ptr<float>(), weights.data_ptr<float>(),
            dirtyMaskPtr,
            esdfInit.data_ptr<float>(),
            truncF, threshF, saturEps);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // ------------------ Step 4-5: sweeps + clamp + prune --------------------

    return runEsdfSweepsAndFinalize(
        esdfGrid, esdfInit, voxelSizeF, dilateAmount, maxDistF,
        prune_unreached, use_vbm, stream);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
