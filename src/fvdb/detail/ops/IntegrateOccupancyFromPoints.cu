// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// IntegrateOccupancyFromPoints.cu
//
// Bayesian log-odds occupancy integrator for LiDAR / point-cloud
// sweeps. Sister primitive to `IntegrateTSDFFromPoints`: same shell
// allocator, same HDDA ray-walk; the only structural difference is
// the per-voxel update rule (log-odds increment instead of running
// weighted-average signed distance).
//
// Paper-framing: this is the paper's fifth application of the
// nanoVDB topology-op vocabulary. Uses:
//   - `voxelsToGrid` (via buildPointTruncationShell -> voxelsToGrid)
//   - `mergeGrids`   (to preserve previous-frame topology)
//   - `inject`       (to carry over previous log-odds values)
//   - ONE custom CUDA kernel (the ray-walk log-odds update)
//   - `torch.clamp` (for the [log_odds_min, log_odds_max] cap)
//
// No custom allocator, no custom hash table, no per-pixel projective
// integrator. Just the same sparse-substrate primitives that power
// TSDF.
//
// Pipeline:
//   P0. Build topology: union of existing grid + truncation shell of
//       new points (identical to TSDF).
//   P1. Inject previous log-odds values into the new grid; new
//       voxels default to 0 (log-odds = 0 => p = 0.5 = unknown).
//   P2. Ray-walk kernel: one thread per input point. HDDA-walks
//       active voxels along the ray; for each voxel, classifies as
//       hit / miss / unknown and atomicAdd's the appropriate
//       log-odds delta.
//   P3. Clamp to [log_odds_min, log_odds_max].

#include <fvdb/GridBatchData.h>
#include <fvdb/JaggedTensor.h>
#include <fvdb/VoxelCoordTransform.h>
#include <fvdb/detail/ops/BuildMergedGrids.h>
#include <fvdb/detail/ops/BuildPointTruncationShell.h>
#include <fvdb/detail/ops/Inject.h>
#include <fvdb/detail/ops/IntegrateOccupancyFromPoints.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/cuda/Atomics.cuh>
#include <fvdb/detail/utils/cuda/GridDim.h>
#include <fvdb/detail/utils/nanovdb/HDDAIterators.h>

#include <ATen/OpMathType.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/math/Ray.h>

#include <cmath>
#include <cuda_runtime.h>
#include <torch/types.h>

namespace fvdb::detail::ops {

namespace {

using GridT = nanovdb::ValueOnIndex;

// -------------------------------------------------------------------------
// P2: ray-walk log-odds kernel.
//
// Mirrors `rayWalkIntegrateKernel` from IntegrateTSDFFromPoints.cu
// — same HDDA-walk, same endpoint / free-band / unknown classification
// via the `sdfWorld` (range-to-surface) test — but writes log-odds
// deltas instead of accumulating weighted signed-distance sums.
//
// Per-ray update rule:
//   - For each active voxel `v` along the ray within the walk window:
//       sdfWorld = ||P - O|| - ||v - O||
//       if sdfWorld > +truncationMargin (voxel behind endpoint, free):
//           log_odds[v] += logOddsMiss   (negative -> more likely free)
//       if sdfWorld in [-truncationMargin, +truncationMargin] (hit band):
//           log_odds[v] += logOddsHit    (positive -> more likely occupied)
//       else: unknown region behind the endpoint, skip.
//
// We DO NOT clamp in the kernel; the host-side `torch::clamp_` in the
// orchestrator does the bounded update in one shot after all rays
// have been integrated. This matches the additive-log-odds Bayesian
// semantics and avoids per-write atomicMin/Max complexity.
// -------------------------------------------------------------------------

template <typename ScalarT>
__global__ void
rayWalkLogOddsKernel(
    const fvdb::BatchGridAccessor unionGridAcc,
    const fvdb::JaggedRAcc64<ScalarT, 2> pointsAcc,
    const fvdb::TorchRAcc64<ScalarT, 2> sensorOriginsAcc,
    const float truncationMargin,
    const float logOddsHit,
    const float logOddsMiss,
    fvdb::TorchRAcc64<ScalarT, 1> outLogOddsAcc) {
    using MathT = at::opmath_type<ScalarT>;
    using Vec3T = nanovdb::math::Vec3<MathT>;
    using RayT  = nanovdb::math::Ray<MathT>;

    const int64_t totalPoints = pointsAcc.elementCount();
    const int64_t pointIdx    = blockIdx.x * blockDim.x + threadIdx.x;
    if (pointIdx >= totalPoints) return;

    const fvdb::JIdxType batchIdx = pointsAcc.batchIdx(pointIdx);

    const Vec3T originWorld(
        static_cast<MathT>(sensorOriginsAcc[batchIdx][0]),
        static_cast<MathT>(sensorOriginsAcc[batchIdx][1]),
        static_cast<MathT>(sensorOriginsAcc[batchIdx][2]));
    const Vec3T endpointWorld(
        static_cast<MathT>(pointsAcc.data()[pointIdx][0]),
        static_cast<MathT>(pointsAcc.data()[pointIdx][1]),
        static_cast<MathT>(pointsAcc.data()[pointIdx][2]));
    Vec3T dirWorld         = endpointWorld - originWorld;
    const MathT rangeWorld = dirWorld.length();
    if (rangeWorld < MathT(1e-8)) return;
    dirWorld = dirWorld / rangeWorld;

    // Walk from the sensor origin through the hit band. We always
    // carve free space here — occupancy without free-space carving
    // degenerates to a "hit-set" tracker, which isn't what the
    // log-odds formulation needs.
    const MathT tWalkStart = MathT(0);
    const MathT tWalkEnd   = rangeWorld + MathT(truncationMargin);
    if (tWalkEnd <= tWalkStart) return;

    const RayT rayWorld(originWorld, dirWorld, tWalkStart, tWalkEnd);

    const VoxelCoordTransform transform =
        unionGridAcc.primalTransform(batchIdx);
    const RayT rayVox = transform.applyToRay(rayWorld);

    const nanovdb::NanoGrid<GridT> *grid = unionGridAcc.grid(batchIdx);
    auto acc                             = grid->getAccessor();
    const int64_t voxelOffsetBase = unionGridAcc.voxelOffset(batchIdx);

    fvdb::HDDAVoxelIterator<decltype(acc), MathT> it(rayVox, acc);
    while (it.isValid()) {
        const nanovdb::Coord voxIjk = it->first;
        ++it;

        // World-space "signed distance along ray to endpoint":
        // positive = voxel is on the sensor side of the endpoint
        // (free space); negative = voxel is beyond the endpoint
        // (unknown region behind the observed surface).
        const Vec3T voxPosWorld = transform.applyInv<MathT>(
            static_cast<MathT>(voxIjk[0]),
            static_cast<MathT>(voxIjk[1]),
            static_cast<MathT>(voxIjk[2]));
        const Vec3T toVox       = voxPosWorld - originWorld;
        const MathT rangeToVox  = toVox.length();
        const MathT sdfWorld    = rangeWorld - rangeToVox;

        // Classify + pick log-odds delta.
        float logOddsDelta;
        if (sdfWorld > MathT(truncationMargin)) {
            // Free space (voxel is farther from the endpoint than the
            // truncation band; sensor side).
            logOddsDelta = logOddsMiss;
        } else if (sdfWorld >= -MathT(truncationMargin)) {
            // Hit band: within +/- truncationMargin of the endpoint.
            logOddsDelta = logOddsHit;
        } else {
            // Behind the endpoint — unknown state, skip.
            continue;
        }

        const int64_t writeOffset =
            voxelOffsetBase + static_cast<int64_t>(acc.getValue(voxIjk)) - 1;
        atomAdd(&outLogOddsAcc[writeOffset], static_cast<ScalarT>(logOddsDelta));
    }
}

// -------------------------------------------------------------------------
// Host orchestrator. Callable from both single-frame and batched paths.
// -------------------------------------------------------------------------

JaggedTensor
doIntegrateOccupancyFromPoints(const float truncationMargin,
                               const JaggedTensor &points,
                               const torch::Tensor &sensorOrigins,
                               const GridBatchData &unionGrid,
                               const GridBatchData &baseGrid,
                               const JaggedTensor &logOddsIn,
                               const float logOddsHit,
                               const float logOddsMiss,
                               const float logOddsMin,
                               const float logOddsMax) {
    const c10::cuda::CUDAGuard device_guard(logOddsIn.device());

    const int64_t totalOutVoxels = unionGrid.totalVoxels();

    // P1: allocate new log-odds tensor + inject previous values onto
    // the merged grid. New voxels default to zero (log-odds = 0 =>
    // p = 0.5 = unknown), which is the standard Bayesian prior for
    // an unobserved cell.
    torch::Tensor outLogOdds =
        torch::zeros({totalOutVoxels}, logOddsIn.jdata().options());
    {
        JaggedTensor dstJt = unionGrid.jaggedTensor(outLogOdds);
        // inject(dstGrid, srcGrid, dst, src): copies ijk-overlapping
        // voxels from src into dst; leaves non-overlapping slots
        // untouched (i.e. at the zero-init value). This is the same
        // state-carry-over pattern PersistentTSDFState uses.
        ops::inject(unionGrid, baseGrid, dstJt, logOddsIn);
        outLogOdds = dstJt.jdata();
    }

    // P2: ray-walk kernel.
    AT_DISPATCH_V2(
        logOddsIn.scalar_type(),
        "integrateOccupancyFromPointsKernel",
        AT_WRAP([&] {
            const auto stream = at::cuda::getCurrentCUDAStream();
            auto outLogOddsAcc =
                outLogOdds.packed_accessor64<scalar_t, 1,
                                             torch::RestrictPtrTraits>();
            auto pointsAcc =
                points.packed_accessor64<scalar_t, 2,
                                         torch::RestrictPtrTraits>();
            auto sensorAcc =
                sensorOrigins.packed_accessor64<scalar_t, 2,
                                                torch::RestrictPtrTraits>();
            const int64_t totalPoints = points.jdata().size(0);
            if (totalPoints > 0) {
                const int64_t blocks =
                    GET_BLOCKS(totalPoints, DEFAULT_BLOCK_DIM);
                rayWalkLogOddsKernel<scalar_t>
                    <<<blocks, DEFAULT_BLOCK_DIM, 0, stream.stream()>>>(
                        unionGrid.deviceAccessor(),
                        pointsAcc,
                        sensorAcc,
                        truncationMargin,
                        logOddsHit,
                        logOddsMiss,
                        outLogOddsAcc);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
        }),
        AT_EXPAND(AT_FLOATING_TYPES),
        c10::kHalf);

    // P3: clamp. Single torch-level call, avoids a separate CUDA
    // kernel. The clamp is applied AFTER all rays have accumulated
    // so the Bayesian log-odds sum is respected even if individual
    // ray contributions would overshoot the bounds momentarily.
    outLogOdds.clamp_(logOddsMin, logOddsMax);

    return unionGrid.jaggedTensor(outLogOdds);
}

c10::intrusive_ptr<GridBatchData>
buildUnionGrid(const c10::intrusive_ptr<GridBatchData> &baseGrid,
               const JaggedTensor &points,
               double truncationMargin) {
    auto pointShell = buildPointTruncationShell(points, *baseGrid, truncationMargin);
    return mergeGrids(*baseGrid, *pointShell);
}

void
checkCommonInputs(const c10::intrusive_ptr<GridBatchData> &grid,
                  const JaggedTensor &points,
                  const torch::Tensor &sensorOrigins,
                  const JaggedTensor &logOdds,
                  double logOddsMin,
                  double logOddsMax) {
    TORCH_CHECK_VALUE(grid != nullptr, "grid must be non-null");
    TORCH_CHECK_VALUE(grid->device().is_cuda(),
                      "integrateOccupancyFromPoints requires a CUDA grid");
    TORCH_CHECK_VALUE(points.rdim() == 2 && points.rsize(-1) == 3,
                      "points must have shape [B, N, 3]");
    TORCH_CHECK_VALUE(sensorOrigins.dim() == 2 && sensorOrigins.size(1) == 3,
                      "sensorOrigins must have shape [B, 3]");
    TORCH_CHECK_VALUE(sensorOrigins.size(0) == grid->batchSize(),
                      "sensorOrigins batch size (", sensorOrigins.size(0),
                      ") must match grid batch size (", grid->batchSize(), ")");
    TORCH_CHECK_VALUE(points.num_outer_lists() == grid->batchSize(),
                      "points batch size mismatch");
    TORCH_CHECK_VALUE(logOdds.num_outer_lists() == grid->batchSize(),
                      "logOdds batch size mismatch");
    TORCH_CHECK_TYPE(logOdds.is_floating_point(),
                     "logOdds must be a floating-point dtype");
    TORCH_CHECK_TYPE(points.scalar_type() == logOdds.scalar_type(),
                     "points dtype must match logOdds dtype");
    TORCH_CHECK_TYPE(sensorOrigins.scalar_type() == logOdds.scalar_type(),
                     "sensorOrigins dtype must match logOdds dtype");
    TORCH_CHECK_VALUE(logOdds.numel() == grid->totalVoxels(),
                      "logOdds size (", logOdds.numel(),
                      ") must equal grid totalVoxels (", grid->totalVoxels(), ")");
    TORCH_CHECK_VALUE(logOddsMax > logOddsMin,
                      "logOddsMax (", logOddsMax,
                      ") must be strictly greater than logOddsMin (",
                      logOddsMin, ")");
}

} // anonymous namespace

// -------------------------------------------------------------------------
// Public entry points.
// -------------------------------------------------------------------------

std::tuple<c10::intrusive_ptr<GridBatchData>, JaggedTensor>
integrateOccupancyFromPoints(const c10::intrusive_ptr<GridBatchData> grid,
                             const double truncationMargin,
                             const JaggedTensor &points,
                             const torch::Tensor &sensorOrigins,
                             const JaggedTensor &logOdds,
                             const double logOddsHit,
                             const double logOddsMiss,
                             const double logOddsMin,
                             const double logOddsMax) {
    checkCommonInputs(grid, points, sensorOrigins, logOdds, logOddsMin, logOddsMax);

    // Empty point cloud: nothing to allocate, nothing to integrate.
    // Return the grid + log-odds unchanged. `buildPointTruncationShell`
    // doesn't handle a zero-point input cleanly (it tries to build an
    // empty grid handle which triggers a batched-handle assert); this
    // pre-check keeps the no-op case clean.
    if (points.numel() == 0) {
        return {grid, logOdds};
    }

    auto unionGrid = buildUnionGrid(grid, points, truncationMargin);
    auto newLogOdds = doIntegrateOccupancyFromPoints(
        static_cast<float>(truncationMargin),
        points, sensorOrigins,
        *unionGrid, *grid,
        logOdds,
        static_cast<float>(logOddsHit),
        static_cast<float>(logOddsMiss),
        static_cast<float>(logOddsMin),
        static_cast<float>(logOddsMax));
    return {unionGrid, newLogOdds};
}

std::tuple<c10::intrusive_ptr<GridBatchData>, JaggedTensor>
integrateOccupancyFromPointsFrames(
    const c10::intrusive_ptr<GridBatchData> grid,
    const double truncationMargin,
    const std::vector<torch::Tensor> &pointsPerFrame,
    const torch::Tensor &sensorOrigins,
    const JaggedTensor &logOdds,
    const double logOddsHit,
    const double logOddsMiss,
    const double logOddsMin,
    const double logOddsMax) {
    const int64_t N = static_cast<int64_t>(pointsPerFrame.size());
    TORCH_CHECK_VALUE(N > 0, "pointsPerFrame must have at least one frame");
    TORCH_CHECK_VALUE(
        sensorOrigins.dim() == 2 && sensorOrigins.size(0) == N &&
            sensorOrigins.size(1) == 3,
        "sensorOrigins must have shape [N=", N, ", 3]; got ",
        sensorOrigins.sizes());
    TORCH_CHECK_VALUE(grid->batchSize() == 1,
                      "integrateOccupancyFromPointsFrames supports "
                      "single-scene grids only (batchSize = 1); got ",
                      grid->batchSize());
    TORCH_CHECK_VALUE(grid->device().is_cuda(),
                      "integrateOccupancyFromPointsFrames requires a CUDA grid");

    const at::cuda::CUDAGuard device_guard(logOdds.device());

    // Running accumulator (same pattern as the LiDAR TSDF batched
    // path). Each frame builds a fresh shell, unions with accumGrid,
    // injects previous log-odds, ray-walks, and clamps. Old refs
    // drop out of scope each iteration; the caching allocator
    // reclaims memory.
    c10::intrusive_ptr<GridBatchData> accumGrid = grid;
    JaggedTensor accumLogOdds = logOdds;

    for (int64_t i = 0; i < N; ++i) {
        const torch::Tensor &ptsTensor = pointsPerFrame[i];
        TORCH_CHECK_VALUE(ptsTensor.dim() == 2 && ptsTensor.size(1) == 3,
                          "pointsPerFrame[", i, "] must be [N_i, 3]");
        TORCH_CHECK_VALUE(ptsTensor.device() == logOdds.device(),
                          "pointsPerFrame[", i,
                          "] must be on the same device as logOdds");
        TORCH_CHECK_TYPE(ptsTensor.scalar_type() == logOdds.scalar_type(),
                         "pointsPerFrame[", i,
                         "] dtype must match logOdds dtype");

        JaggedTensor ptsJagged =
            JaggedTensor(std::vector<torch::Tensor>{ptsTensor});
        torch::Tensor originI = sensorOrigins.narrow(0, i, 1).contiguous();

        auto unionGrid =
            buildUnionGrid(accumGrid, ptsJagged, truncationMargin);
        auto newLogOdds = doIntegrateOccupancyFromPoints(
            static_cast<float>(truncationMargin),
            ptsJagged, originI,
            *unionGrid, *accumGrid,
            accumLogOdds,
            static_cast<float>(logOddsHit),
            static_cast<float>(logOddsMiss),
            static_cast<float>(logOddsMin),
            static_cast<float>(logOddsMax));

        accumGrid     = unionGrid;
        accumLogOdds  = newLogOdds;
    }

    return {accumGrid, accumLogOdds};
}

} // namespace fvdb::detail::ops
