// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Native LiDAR / range-sensor TSDF integrator. Per-point thread walks
// the union grid via HDDA and updates (TSDF, weight, features) at each
// voxel within the truncation band (and optionally the free-space band)
// via lock-free atomicAdd in running-sum form.
//
// Pipeline:
//   1. Build topology: union of existing grid and truncation shell of
//      new points (via the shared `buildPointTruncationShell` primitive
//      that the depth integrator also uses).
//   2. Seed kernel: initialise (sum_w_sdf, sum_w, sum_w_feat) on the
//      union grid from the existing (tsdf, weights, features) on the
//      base grid (or zero where the voxel is new).
//   3. Ray-walk kernel: one thread per point. HDDA-walks active voxels
//      along the ray; within the truncation / free-space bands, does
//      atomicAdd updates on the three running-sum accumulators.
//   4. Normalise kernel: divides sum_w_sdf / sum_w -> tsdf,
//      sum_w_feat / sum_w -> features. sum_w stays as the per-voxel
//      weight.

#include <fvdb/GridBatchData.h>
#include <fvdb/JaggedTensor.h>
#include <fvdb/VoxelCoordTransform.h>
#include <fvdb/detail/ops/BuildMergedGrids.h>
#include <fvdb/detail/ops/BuildPointTruncationShell.h>
#include <fvdb/detail/ops/IntegrateTSDFFromPoints.h>
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
#include <c10/util/Half.h>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/math/Ray.h>

#include <cmath>
#include <cuda_runtime.h>
#include <torch/types.h>

namespace fvdb::detail::ops {

namespace {

using GridT        = nanovdb::ValueOnIndex;
using LeafNodeType = nanovdb::NanoGrid<GridT>::LeafNodeType;
constexpr uint64_t VOXELS_PER_LEAF =
    nanovdb::NanoTree<GridT>::LeafNodeType::NUM_VALUES;

// -------------------------------------------------------------------------
// M1: seed kernel.
//
// For each active voxel in the union grid, initialise the running-sum
// accumulators from the base grid's (tsdf, weights, features) if the
// voxel already exists there, otherwise zero.
//
// The output `outTsdf` and `outFeatures` tensors store SUM-OF-WEIGHTED
// values at this stage (i.e. tsdf * weight, features * weight). The
// final normalise pass divides by `outWeights` to recover the true
// running average.
// -------------------------------------------------------------------------

template <typename ScalarDataType, typename FeatureAccumT>
__global__ void
seedAccumulatorsFromBaseGridKernel(
    const fvdb::BatchGridAccessor baseGridAcc,
    const fvdb::BatchGridAccessor unionGridAcc,
    const bool hasFeatures,
    const int64_t featureDim,
    const fvdb::JaggedRAcc64<ScalarDataType, 1> tsdfAcc,
    const fvdb::JaggedRAcc64<ScalarDataType, 1> weightsAcc,
    const fvdb::JaggedRAcc64<FeatureAccumT, 2> featuresAsAccumAcc,
    fvdb::TorchRAcc64<ScalarDataType, 1> outTsdfAcc,
    fvdb::TorchRAcc64<ScalarDataType, 1> outWeightsAcc,
    fvdb::TorchRAcc64<FeatureAccumT, 2> outFeaturesAccumAcc) {
    const uint64_t problemSize =
        unionGridAcc.totalLeaves() * VOXELS_PER_LEAF;
    for (uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < problemSize;
         idx += blockDim.x * gridDim.x) {
        const int64_t cumUnionLeafIdx =
            static_cast<int64_t>(idx / VOXELS_PER_LEAF);
        const int64_t unionLeafVoxelIdx =
            static_cast<int64_t>(idx % VOXELS_PER_LEAF);
        const fvdb::JIdxType batchIdx =
            unionGridAcc.leafBatchIndex(cumUnionLeafIdx);
        const int64_t unionLeafIdx =
            cumUnionLeafIdx - unionGridAcc.leafOffset(batchIdx);

        const nanovdb::NanoGrid<GridT> *unionGrid = unionGridAcc.grid(batchIdx);
        const LeafNodeType &unionLeaf =
            unionGrid->tree().template getFirstNode<0>()[unionLeafIdx];
        const nanovdb::Coord ijk =
            unionLeaf.offsetToGlobalCoord(unionLeafVoxelIdx);

        const int64_t unionWriteOffset =
            unionGridAcc.voxelOffset(batchIdx) +
            static_cast<int64_t>(unionLeaf.getValue(unionLeafVoxelIdx)) - 1;
        if (unionWriteOffset < unionGridAcc.voxelOffset(batchIdx)) {
            continue; // inactive slot
        }

        // Check if voxel exists in base grid.
        const nanovdb::NanoGrid<GridT> *baseGrid = baseGridAcc.grid(batchIdx);
        auto baseAcc                             = baseGrid->getAccessor();
        const bool inBase                        = baseAcc.isActive(ijk);

        if (inBase) {
            const int64_t baseOffset =
                baseGridAcc.voxelOffset(batchIdx) +
                static_cast<int64_t>(baseAcc.getValue(ijk)) - 1;
            const ScalarDataType oldW = weightsAcc.data()[baseOffset];
            const ScalarDataType oldT = tsdfAcc.data()[baseOffset];
            outTsdfAcc[unionWriteOffset]    = ScalarDataType(static_cast<float>(oldT) *
                                                             static_cast<float>(oldW));
            outWeightsAcc[unionWriteOffset] = oldW;
            if (hasFeatures) {
                for (int64_t d = 0; d < featureDim; ++d) {
                    outFeaturesAccumAcc[unionWriteOffset][d] =
                        FeatureAccumT(static_cast<float>(featuresAsAccumAcc.data()[baseOffset][d]) *
                                      static_cast<float>(oldW));
                }
            }
        } else {
            outTsdfAcc[unionWriteOffset]    = ScalarDataType(0);
            outWeightsAcc[unionWriteOffset] = ScalarDataType(0);
            if (hasFeatures) {
                for (int64_t d = 0; d < featureDim; ++d) {
                    outFeaturesAccumAcc[unionWriteOffset][d] = FeatureAccumT(0);
                }
            }
        }
    }
}

// -------------------------------------------------------------------------
// M2: ray-walk kernel.
//
// One thread per input point. Walks active voxels along the ray from
// sensor origin to (point + truncation along ray) via HDDAVoxelIterator
// over the union grid. For each active voxel, computes the signed
// distance along the ray from voxel centre to endpoint and decides
// whether to update it:
//   - behind endpoint by > truncation: skip (unknown state).
//   - within [−truncation, +truncation] of endpoint: write
//     clamped tsdf_normalised, weight = 1.
//   - in front of endpoint (free space) and `carveFreeSpace`:
//     write tsdf = +1, weight = 1.
//   - free-space without carving: skip.
// Updates go via atomicAdd on the running-sum accumulators; the
// running-sum form is what makes the concurrent updates lock-free
// (see plan.md D3 and the `seedAccumulatorsFromBaseGridKernel` note).
// -------------------------------------------------------------------------

template <typename ScalarDataType, typename FeatureDataType, typename FeatureAccumT>
__global__ void
rayWalkIntegrateKernel(
    const fvdb::BatchGridAccessor unionGridAcc,
    const fvdb::JaggedRAcc64<ScalarDataType, 2> pointsAcc,
    const fvdb::TorchRAcc64<ScalarDataType, 2> sensorOriginsAcc,
    const bool hasFeatures,
    const int64_t featureDim,
    const fvdb::JaggedRAcc64<FeatureDataType, 2> pointFeaturesAcc,
    const float truncationMargin,
    const bool carveFreeSpace,
    fvdb::TorchRAcc64<ScalarDataType, 1> outTsdfAcc,
    fvdb::TorchRAcc64<ScalarDataType, 1> outWeightsAcc,
    fvdb::TorchRAcc64<FeatureAccumT, 2> outFeaturesAccumAcc) {
    using MathT = at::opmath_type<ScalarDataType>;
    using Vec3T = nanovdb::math::Vec3<MathT>;
    using RayT  = nanovdb::math::Ray<MathT>;

    const int64_t totalPoints = pointsAcc.elementCount();
    const int64_t pointIdx    = blockIdx.x * blockDim.x + threadIdx.x;
    if (pointIdx >= totalPoints) {
        return;
    }

    const fvdb::JIdxType batchIdx = pointsAcc.batchIdx(pointIdx);

    // World-space ray from sensor origin to point endpoint. We use
    // static_cast rather than functional-cast syntax (`MathT(...)`)
    // because nvcc otherwise hits a most-vexing-parse corner on some
    // versions (interprets the inner expression as a parameter-name
    // declaration inside the Vec3T constructor).
    const Vec3T originWorld(static_cast<MathT>(sensorOriginsAcc[batchIdx][0]),
                            static_cast<MathT>(sensorOriginsAcc[batchIdx][1]),
                            static_cast<MathT>(sensorOriginsAcc[batchIdx][2]));
    const Vec3T endpointWorld(static_cast<MathT>(pointsAcc.data()[pointIdx][0]),
                              static_cast<MathT>(pointsAcc.data()[pointIdx][1]),
                              static_cast<MathT>(pointsAcc.data()[pointIdx][2]));
    Vec3T dirWorld         = endpointWorld - originWorld;
    const MathT rangeWorld = dirWorld.length();
    if (rangeWorld < MathT(1e-8)) {
        return; // degenerate zero-length ray
    }
    dirWorld = dirWorld / rangeWorld;

    // Ray parametrisation (in world space):
    //   t = 0 at origin, t = rangeWorld at endpoint.
    // We walk voxels over t in [0, rangeWorld + truncationMargin] when
    // carving free space, else [rangeWorld - truncationMargin,
    // rangeWorld + truncationMargin].
    const MathT tTruncStart = rangeWorld - MathT(truncationMargin);
    const MathT tTruncEnd   = rangeWorld + MathT(truncationMargin);
    const MathT tWalkStart  = carveFreeSpace ? MathT(0) : tTruncStart;
    const MathT tWalkEnd    = tTruncEnd;
    if (tWalkEnd <= tWalkStart) {
        return; // nothing to update
    }

    const RayT rayWorld(originWorld, dirWorld, tWalkStart, tWalkEnd);

    // Transform ray to voxel-index space for HDDA.
    const VoxelCoordTransform transform =
        unionGridAcc.primalTransform(batchIdx);
    const RayT rayVox = transform.applyToRay(rayWorld);

    const nanovdb::NanoGrid<GridT> *grid = unionGridAcc.grid(batchIdx);
    auto acc                             = grid->getAccessor();
    const int64_t voxelOffsetBase = unionGridAcc.voxelOffset(batchIdx);

    // HDDAVoxelIterator walks active voxels of the sparse grid along the
    // ray, automatically skipping inactive regions. This is the sparse-
    // native "ray-walk" primitive fvdb exposes; the per-ray thread hits
    // only voxels that exist in the endpoint-shell topology (see plan.md
    // D2 — free-space carving fills topology gaps only within the
    // existing union grid, does not extend it).
    fvdb::HDDAVoxelIterator<decltype(acc), MathT> it(rayVox, acc);
    while (it.isValid()) {
        const nanovdb::Coord voxIjk = it->first;
        ++it;

        // World-space signed distance: Euclidean range-difference
        // from sensor origin, ||P - O|| - ||V - O||. Positive = voxel
        // is closer to origin than the surface point (free space);
        // negative = voxel is farther than the surface (unknown /
        // behind). This matches the VDBFusion / canonical-TSDF
        // convention.
        //
        // Using the along-ray projection (toVox · dir) would bias
        // mesh extraction outward for voxels near but not on the ray
        // — HDDA includes voxel centres that are off-ray by up to
        // sqrt(3)/2 * voxel_size, so the off-ray bias is ~1 voxel.
        // The Euclidean-range form has no such bias.
        //
        // fvdb's convention treats voxel values as stored AT integer
        // ijk coordinates (same as the existing depth integrator in
        // IntegrateTSDF.cu:204-206); no +0.5 shift.
        const Vec3T voxPosWorld = transform.applyInv<MathT>(
            static_cast<MathT>(voxIjk[0]),
            static_cast<MathT>(voxIjk[1]),
            static_cast<MathT>(voxIjk[2]));
        const Vec3T toVox       = voxPosWorld - originWorld;
        const MathT rangeToVox  = toVox.length();
        const MathT sdfWorld    = rangeWorld - rangeToVox;

        // Classify the voxel.
        MathT tsdfClamped;
        if (sdfWorld > MathT(truncationMargin)) {
            if (!carveFreeSpace) {
                continue;
            }
            tsdfClamped = MathT(1);
        } else if (sdfWorld < -MathT(truncationMargin)) {
            continue; // unknown region behind the endpoint
        } else {
            tsdfClamped = sdfWorld / MathT(truncationMargin);
        }

        // Look up the voxel's write offset. isActive was already
        // checked inside HDDA so getValue is safe.
        const int64_t writeOffset =
            voxelOffsetBase + static_cast<int64_t>(acc.getValue(voxIjk)) - 1;

        // `atomAdd` (from Atomics.cuh) is the fvdb wrapper that
        // handles both hardware-native (float / double / at::Half on
        // sm_70+) and CAS-loop-based atomic adds on all supported
        // dtypes — including the half-precision path that plain
        // `atomicAdd(c10::Half*, ...)` doesn't resolve.
        constexpr MathT kSampleWeight = MathT(1);
        atomAdd(&outTsdfAcc[writeOffset],
                static_cast<ScalarDataType>(tsdfClamped * kSampleWeight));
        atomAdd(&outWeightsAcc[writeOffset],
                static_cast<ScalarDataType>(kSampleWeight));
        if (hasFeatures) {
            for (int64_t d = 0; d < featureDim; ++d) {
                const FeatureAccumT featVal =
                    static_cast<FeatureAccumT>(pointFeaturesAcc.data()[pointIdx][d]);
                atomAdd(&outFeaturesAccumAcc[writeOffset][d],
                        static_cast<FeatureAccumT>(
                            featVal * static_cast<FeatureAccumT>(kSampleWeight)));
            }
        }
    }
}

// -------------------------------------------------------------------------
// M3: normalise kernel.
//
// After the ray-walk accumulations, outTsdf and outFeatures hold
// running sums of (tsdf * weight) and (feature * weight). Divide by
// outWeights to recover the running-average form that the public TSDF
// API contract expects. Voxels that received no updates (weights ==
// 0) are left at zero (reasonable — signals "no observation").
// -------------------------------------------------------------------------

template <typename ScalarDataType, typename FeatureDataType, typename FeatureAccumT>
__global__ void
normaliseAccumulatorsKernel(const int64_t totalVoxels,
                            const bool hasFeatures,
                            const int64_t featureDim,
                            fvdb::TorchRAcc64<ScalarDataType, 1> outTsdfAcc,
                            fvdb::TorchRAcc64<ScalarDataType, 1> outWeightsAcc,
                            const fvdb::TorchRAcc64<FeatureAccumT, 2> outFeaturesAccumAcc,
                            fvdb::TorchRAcc64<FeatureDataType, 2> outFeaturesAcc) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalVoxels) {
        return;
    }

    const float w = static_cast<float>(outWeightsAcc[idx]);
    if (w > 0.0f) {
        outTsdfAcc[idx] =
            ScalarDataType(static_cast<float>(outTsdfAcc[idx]) / w);
        if (hasFeatures) {
            for (int64_t d = 0; d < featureDim; ++d) {
                outFeaturesAcc[idx][d] =
                    FeatureDataType(static_cast<float>(outFeaturesAccumAcc[idx][d]) / w);
            }
        }
    } else {
        outTsdfAcc[idx] = ScalarDataType(0);
        if (hasFeatures) {
            for (int64_t d = 0; d < featureDim; ++d) {
                outFeaturesAcc[idx][d] = FeatureDataType(0);
            }
        }
    }
}

// -------------------------------------------------------------------------
// Host orchestrator.
//
// Given an already-merged union grid plus the new input (points +
// features), run the three-kernel pipeline above. Sequestered into a
// helper so the two public entry points (with / without features)
// share everything except input validation.
// -------------------------------------------------------------------------

#define DISPATCH_FEATURE_TYPE_LIDAR(SCALAR, FEAT_TYPE, ...)            \
    if (hasFeatures && (FEAT_TYPE) == torch::kUInt8) {                  \
        using feature_t = uint8_t;                                      \
        /* uint8 atomicAdd unsupported on-device; accumulate in fp32 */ \
        using feature_accum_t = float;                                  \
        __VA_ARGS__();                                                  \
    } else {                                                            \
        using feature_t = SCALAR;                                       \
        using feature_accum_t = SCALAR;                                 \
        __VA_ARGS__();                                                  \
    }

std::tuple<JaggedTensor, JaggedTensor, JaggedTensor>
doIntegrateFromPoints(const float truncationMargin,
                      const JaggedTensor &points,
                      const torch::Tensor &sensorOrigins,
                      const JaggedTensor &pointFeatures,
                      const GridBatchData &unionGrid,
                      const GridBatchData &baseGrid,
                      const JaggedTensor &tsdf,
                      const JaggedTensor &weights,
                      const JaggedTensor &features,
                      bool carveFreeSpace) {
    const c10::cuda::CUDAGuard device_guard(tsdf.device());

    const int64_t totalOutVoxels = unionGrid.totalVoxels();
    const int64_t featureDim     = features.rsize(-1);
    const bool hasFeatures       = featureDim > 0;

    torch::Tensor outTsdf    = torch::empty({totalOutVoxels}, tsdf.jdata().options());
    torch::Tensor outWeights = torch::empty({totalOutVoxels}, weights.jdata().options());
    // Always allocate with totalOutVoxels rows so the final
    // `unionGrid.jaggedTensor(outFeatures)` size-check passes
    // uniformly (featureDim=0 in the no-features case, matching the
    // depth integrator's convention in IntegrateTSDF.cu:841).
    torch::Tensor outFeatures = torch::empty(
        {totalOutVoxels, featureDim}, features.jdata().options());

    AT_DISPATCH_V2(
        tsdf.scalar_type(),
        "integrateTSDFFromPointsKernel",
        AT_WRAP([&] {
            DISPATCH_FEATURE_TYPE_LIDAR(scalar_t, features.scalar_type(), [&] {
                // Feature accumulator tensor (may be wider than features
                // itself when features are uint8 → accumulate in fp32).
                torch::Tensor outFeaturesAccum;
                constexpr bool accumIsSame =
                    std::is_same_v<feature_t, feature_accum_t>;
                if (hasFeatures) {
                    if constexpr (accumIsSame) {
                        outFeaturesAccum = outFeatures;
                    } else {
                        outFeaturesAccum = torch::empty(
                            {totalOutVoxels, featureDim},
                            torch::TensorOptions()
                                .dtype(c10::CppTypeToScalarType<feature_accum_t>::value)
                                .device(outFeatures.device()));
                    }
                } else {
                    outFeaturesAccum = torch::empty({0, 0},
                        torch::TensorOptions()
                            .dtype(c10::CppTypeToScalarType<feature_accum_t>::value)
                            .device(outTsdf.device()));
                }

                // Features base grid: reinterpret via the same accum
                // dtype so the seed kernel (which reads from base) can
                // use a single typed accessor. When features are uint8,
                // we promote by an explicit cast in the seed kernel.
                torch::Tensor featuresAsAccum;
                if (hasFeatures) {
                    if constexpr (accumIsSame) {
                        featuresAsAccum = features.jdata();
                    } else {
                        featuresAsAccum = features.jdata().to(
                            c10::CppTypeToScalarType<feature_accum_t>::value);
                    }
                } else {
                    featuresAsAccum = torch::empty({0, 0},
                        torch::TensorOptions()
                            .dtype(c10::CppTypeToScalarType<feature_accum_t>::value)
                            .device(outTsdf.device()));
                }

                const auto stream = at::cuda::getCurrentCUDAStream();

                // Use the JaggedTensor-valued packed_accessor64 (not
                // jdata().packed_accessor64) so the kernel receives
                // JaggedRAcc64 with batch-aware `.batchIdx(i)` access.
                auto tsdfAcc =
                    tsdf.packed_accessor64<scalar_t, 1,
                                           torch::RestrictPtrTraits>();
                auto weightsAcc =
                    weights.packed_accessor64<scalar_t, 1,
                                              torch::RestrictPtrTraits>();
                auto outTsdfAcc =
                    outTsdf.packed_accessor64<scalar_t, 1,
                                              torch::RestrictPtrTraits>();
                auto outWeightsAcc =
                    outWeights.packed_accessor64<scalar_t, 1,
                                                 torch::RestrictPtrTraits>();
                // Reinterpret features/jagged features as an accessor
                // with the accumulator's dtype; when features are
                // uint8 we already up-converted above, otherwise this
                // is the identity (accum == feature dtype).
                //
                // In the no-features case we construct a sentinel JT
                // over an empty tensor (jidx empty, jlidx empty) so
                // JaggedTensor::from_data_indices_and_list_ids' size
                // check doesn't mis-trigger against the tsdf JT's
                // `size(0) = totalVoxels` indices tensor. The kernels
                // guard with `if (hasFeatures)` before dereferencing
                // the accessor, so the contents are never read.
                torch::Tensor featuresReinterp;
                if (hasFeatures) {
                    featuresReinterp = featuresAsAccum.reshape(
                        {featuresAsAccum.size(0), featureDim});
                } else {
                    featuresReinterp = torch::empty(
                        {0, 0},
                        torch::TensorOptions()
                            .dtype(c10::CppTypeToScalarType<feature_accum_t>::value)
                            .device(outTsdf.device()));
                }
                JaggedTensor featuresAsAccumJagged;
                if (hasFeatures) {
                    featuresAsAccumJagged =
                        JaggedTensor::from_data_indices_and_list_ids(
                            featuresReinterp,
                            features.jidx(),
                            features.jlidx(),
                            features.num_outer_lists());
                } else {
                    auto idxOpts = torch::TensorOptions()
                                       .dtype(fvdb::JIdxScalarType)
                                       .device(outTsdf.device());
                    featuresAsAccumJagged =
                        JaggedTensor::from_data_indices_and_list_ids(
                            featuresReinterp,
                            torch::empty({0}, idxOpts),
                            torch::empty({0, 1}, idxOpts),
                            /*num_tensors=*/1);
                }
                auto featuresAsAccumAcc =
                    featuresAsAccumJagged.packed_accessor64<feature_accum_t, 2,
                                                            torch::RestrictPtrTraits>();
                auto outFeaturesAccumAcc =
                    outFeaturesAccum.packed_accessor64<feature_accum_t, 2,
                                                       torch::RestrictPtrTraits>();

                // Step 1: seed accumulators from the existing base grid.
                {
                    const uint64_t problemSize =
                        unionGrid.totalLeaves() * VOXELS_PER_LEAF;
                    const int64_t blocks =
                        GET_BLOCKS(problemSize, DEFAULT_BLOCK_DIM);
                    seedAccumulatorsFromBaseGridKernel<scalar_t, feature_accum_t>
                        <<<blocks, DEFAULT_BLOCK_DIM, 0, stream.stream()>>>(
                            baseGrid.deviceAccessor(),
                            unionGrid.deviceAccessor(),
                            hasFeatures,
                            featureDim,
                            tsdfAcc,
                            weightsAcc,
                            featuresAsAccumAcc,
                            outTsdfAcc,
                            outWeightsAcc,
                            outFeaturesAccumAcc);
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                }

                // Step 2: ray-walk every point and accumulate.
                auto pointsAcc =
                    points.packed_accessor64<scalar_t, 2,
                                             torch::RestrictPtrTraits>();
                auto sensorAcc =
                    sensorOrigins.packed_accessor64<scalar_t, 2,
                                                    torch::RestrictPtrTraits>();
                auto pointFeaturesAcc =
                    hasFeatures
                        ? pointFeatures
                              .packed_accessor64<feature_t, 2,
                                                 torch::RestrictPtrTraits>()
                        : pointFeatures
                              .packed_accessor64<feature_t, 2,
                                                 torch::RestrictPtrTraits>();
                const int64_t totalPoints = points.jdata().size(0);
                if (totalPoints > 0) {
                    const int64_t blocks =
                        GET_BLOCKS(totalPoints, DEFAULT_BLOCK_DIM);
                    rayWalkIntegrateKernel<scalar_t, feature_t, feature_accum_t>
                        <<<blocks, DEFAULT_BLOCK_DIM, 0, stream.stream()>>>(
                            unionGrid.deviceAccessor(),
                            pointsAcc,
                            sensorAcc,
                            hasFeatures,
                            featureDim,
                            pointFeaturesAcc,
                            truncationMargin,
                            carveFreeSpace,
                            outTsdfAcc,
                            outWeightsAcc,
                            outFeaturesAccumAcc);
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                }

                // Step 3: normalise accumulators into per-voxel TSDF / weights / features.
                {
                    auto outFeaturesAccOut =
                        hasFeatures
                            ? outFeatures.packed_accessor64<feature_t, 2,
                                                            torch::RestrictPtrTraits>()
                            : outFeatures.packed_accessor64<feature_t, 2,
                                                            torch::RestrictPtrTraits>();
                    const int64_t blocks =
                        GET_BLOCKS(totalOutVoxels, DEFAULT_BLOCK_DIM);
                    normaliseAccumulatorsKernel<scalar_t, feature_t, feature_accum_t>
                        <<<blocks, DEFAULT_BLOCK_DIM, 0, stream.stream()>>>(
                            totalOutVoxels,
                            hasFeatures,
                            featureDim,
                            outTsdfAcc,
                            outWeightsAcc,
                            outFeaturesAccumAcc,
                            outFeaturesAccOut);
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                }
            });
        }),
        AT_EXPAND(AT_FLOATING_TYPES),
        c10::kHalf);

    // outFeatures is `{totalOutVoxels, 0}` in the no-features case,
    // which passes `GridBatchData::jaggedTensor`'s size check
    // uniformly (matches the depth integrator's pattern — see
    // IntegrateTSDF.cu:866).
    return {unionGrid.jaggedTensor(outTsdf),
            unionGrid.jaggedTensor(outWeights),
            unionGrid.jaggedTensor(outFeatures)};
}

// Build the union of the base grid and the new-point truncation shell;
// reused by both public entry points.
c10::intrusive_ptr<GridBatchData>
buildUnionGrid(const c10::intrusive_ptr<GridBatchData> &baseGrid,
               const JaggedTensor &points,
               double truncationMargin) {
    auto pointShell = buildPointTruncationShell(points, *baseGrid, truncationMargin);
    return mergeGrids(*baseGrid, *pointShell);
}

// Common input validation for both public entry points.
void
checkCommonInputs(const c10::intrusive_ptr<GridBatchData> &grid,
                  const JaggedTensor &points,
                  const torch::Tensor &sensorOrigins,
                  const JaggedTensor &tsdf,
                  const JaggedTensor &weights) {
    TORCH_CHECK_VALUE(grid != nullptr, "grid must be non-null");
    TORCH_CHECK_VALUE(grid->device().is_cuda(),
                      "integrateTSDFFromPoints requires a CUDA grid");
    TORCH_CHECK_VALUE(points.rdim() == 2 && points.rsize(-1) == 3,
                      "points must have shape [B, N, 3]");
    TORCH_CHECK_VALUE(sensorOrigins.dim() == 2 && sensorOrigins.size(1) == 3,
                      "sensorOrigins must have shape [B, 3]");
    TORCH_CHECK_VALUE(sensorOrigins.size(0) == grid->batchSize(),
                      "sensorOrigins batch size (", sensorOrigins.size(0),
                      ") must match grid batch size (", grid->batchSize(), ")");
    TORCH_CHECK_VALUE(points.num_outer_lists() == grid->batchSize(),
                      "points batch size (", points.num_outer_lists(),
                      ") must match grid batch size (", grid->batchSize(), ")");
    TORCH_CHECK_VALUE(tsdf.num_outer_lists() == grid->batchSize(),
                      "tsdf batch size (", tsdf.num_outer_lists(),
                      ") must match grid batch size (", grid->batchSize(), ")");
    TORCH_CHECK_VALUE(weights.num_outer_lists() == grid->batchSize(),
                      "weights batch size must match grid batch size");
    TORCH_CHECK_TYPE(tsdf.is_floating_point(),
                     "tsdf must be a floating-point dtype");
    TORCH_CHECK_TYPE(weights.scalar_type() == tsdf.scalar_type(),
                     "weights dtype must match tsdf dtype");
    TORCH_CHECK_TYPE(points.scalar_type() == tsdf.scalar_type(),
                     "points dtype must match tsdf dtype");
    TORCH_CHECK_TYPE(sensorOrigins.scalar_type() == tsdf.scalar_type(),
                     "sensorOrigins dtype must match tsdf dtype");
    TORCH_CHECK_VALUE(tsdf.numel() == grid->totalVoxels(),
                      "tsdf size (", tsdf.numel(),
                      ") must equal grid totalVoxels (", grid->totalVoxels(), ")");
    TORCH_CHECK_VALUE(weights.numel() == grid->totalVoxels(),
                      "weights size mismatch");
}

} // anonymous namespace

// -------------------------------------------------------------------------
// Public entry points.
// -------------------------------------------------------------------------

std::tuple<c10::intrusive_ptr<GridBatchData>, JaggedTensor, JaggedTensor>
integrateTSDFFromPoints(const c10::intrusive_ptr<GridBatchData> grid,
                        const double truncationMargin,
                        const JaggedTensor &points,
                        const torch::Tensor &sensorOrigins,
                        const JaggedTensor &tsdf,
                        const JaggedTensor &weights,
                        bool carveFreeSpace) {
    checkCommonInputs(grid, points, sensorOrigins, tsdf, weights);

    auto unionGrid = buildUnionGrid(grid, points, truncationMargin);

    // Empty JaggedTensor placeholders for the features / pointFeatures
    // slots. `doIntegrateFromPoints` decides `hasFeatures` from the
    // `features.rsize(-1)` inner dimension — a `[0, 0]` JT reports
    // `rsize(-1) == 0`, so this matches the no-features branch
    // cleanly. Convention matches the depth integrator in
    // IntegrateTSDF.cu:841 (`torch::empty({0, 0}, opts)`).
    const fvdb::JaggedTensor emptyFeatures      = torch::empty({0, 0}, tsdf.jdata().options());
    const fvdb::JaggedTensor emptyPointFeatures = torch::empty({0, 0}, tsdf.jdata().options());

    auto [newTsdf, newWeights, _unusedFeatures] = doIntegrateFromPoints(
        static_cast<float>(truncationMargin),
        points,
        sensorOrigins,
        emptyPointFeatures,
        *unionGrid,
        *grid,
        tsdf,
        weights,
        emptyFeatures,
        carveFreeSpace);

    return {unionGrid, newTsdf, newWeights};
}

std::tuple<c10::intrusive_ptr<GridBatchData>, JaggedTensor, JaggedTensor, JaggedTensor>
integrateTSDFFromPointsWithFeatures(const c10::intrusive_ptr<GridBatchData> grid,
                                    const double truncationMargin,
                                    const JaggedTensor &points,
                                    const torch::Tensor &sensorOrigins,
                                    const JaggedTensor &tsdf,
                                    const JaggedTensor &features,
                                    const JaggedTensor &weights,
                                    const JaggedTensor &pointFeatures,
                                    bool carveFreeSpace) {
    checkCommonInputs(grid, points, sensorOrigins, tsdf, weights);

    TORCH_CHECK_VALUE(features.rdim() == 2,
                      "features must be 2-D [totalVoxels, featureDim]");
    TORCH_CHECK_VALUE(pointFeatures.rdim() == 2,
                      "pointFeatures must be 2-D [totalPoints, featureDim]");
    TORCH_CHECK_VALUE(features.rsize(-1) == pointFeatures.rsize(-1),
                      "features and pointFeatures must have the same featureDim");
    TORCH_CHECK_VALUE(features.numel() == grid->totalVoxels() * features.rsize(-1),
                      "features must have totalVoxels rows");
    TORCH_CHECK_VALUE(pointFeatures.num_outer_lists() == grid->batchSize(),
                      "pointFeatures batch size must match grid batch size");
    TORCH_CHECK_VALUE(pointFeatures.numel() == points.numel() / 3 * pointFeatures.rsize(-1),
                      "pointFeatures must have exactly one row per input point");
    // Matching dtype rules from the depth integrator: features must be
    // either the same fp dtype as tsdf, or uint8.
    TORCH_CHECK_TYPE(features.scalar_type() == tsdf.scalar_type() ||
                         features.scalar_type() == torch::kUInt8,
                     "features dtype must match tsdf dtype or be uint8");
    TORCH_CHECK_TYPE(pointFeatures.scalar_type() == features.scalar_type(),
                     "pointFeatures dtype must match features dtype");

    auto unionGrid = buildUnionGrid(grid, points, truncationMargin);

    auto [newTsdf, newWeights, newFeatures] = doIntegrateFromPoints(
        static_cast<float>(truncationMargin),
        points,
        sensorOrigins,
        pointFeatures,
        *unionGrid,
        *grid,
        tsdf,
        weights,
        features,
        carveFreeSpace);

    return {unionGrid, newTsdf, newWeights, newFeatures};
}

std::tuple<c10::intrusive_ptr<GridBatchData>, JaggedTensor, JaggedTensor>
integrateTSDFFromPointsFrames(const c10::intrusive_ptr<GridBatchData> grid,
                              const double truncationMargin,
                              const std::vector<torch::Tensor> &pointsPerFrame,
                              const torch::Tensor &sensorOrigins,
                              const JaggedTensor &tsdf,
                              const JaggedTensor &weights,
                              bool carveFreeSpace) {
    const int64_t N = static_cast<int64_t>(pointsPerFrame.size());
    TORCH_CHECK_VALUE(N > 0, "pointsPerFrame must have at least one frame");
    TORCH_CHECK_VALUE(
        sensorOrigins.dim() == 2 && sensorOrigins.size(0) == N &&
            sensorOrigins.size(1) == 3,
        "sensorOrigins must have shape [N=", N, ", 3]; got ",
        sensorOrigins.sizes());
    TORCH_CHECK_VALUE(grid->batchSize() == 1,
                      "integrateTSDFFromPointsFrames currently supports "
                      "single-scene grids (batchSize = 1); got batchSize = ",
                      grid->batchSize());
    TORCH_CHECK_VALUE(grid->device().is_cuda(),
                      "integrateTSDFFromPointsFrames requires a CUDA grid");

    const at::cuda::CUDAGuard device_guard(tsdf.device());

    // Per-frame profiling toggle, mirrors `integrateTSDFBatchImpl`'s
    // `FVDB_TSDF_BATCH_PROFILE=1` env var. Useful when decomposing
    // the per-frame wall clock into shell-build vs
    // grow/merge/inject vs doIntegrateFromPoints (seed + ray-walk +
    // normalize). Printing happens once per batch call on stderr.
    const bool profile_batch =
        std::getenv("FVDB_TSDF_BATCH_PROFILE") != nullptr;
    cudaEvent_t evStart{}, evEnd{};
    if (profile_batch) {
        cudaEventCreate(&evStart);
        cudaEventCreate(&evEnd);
        cudaEventRecord(evStart);
    }

    // Running accumulator: grid topology + TSDF / weights sidecars.
    // Semantically identical to the pre-refactor Python-looped pattern
    // (`for i: g,t,w = g.integrate_tsdf_from_points(trunc, pts[i],
    //   origin[i], t, w, carve)`), but keeps everything in C++
    // so we don't pay the per-frame Python dispatch + JaggedTensor
    // rewrap cost.
    //
    // We deliberately do NOT thread this through `PersistentTSDFState`
    // because the LiDAR per-frame path (`doIntegrateFromPoints`)
    // already produces fresh output tensors each frame via its
    // seed + ray-walk + normalize pipeline -- the state-holder's
    // grow-on-touch fast path can't fire here (the ray-walk
    // accumulator tensors are throwaway per-frame temporaries, not
    // persistent sidecars). Wrapping in `PersistentTSDFState` would
    // add an extra level of ref-counting without saving any work. See
    // session note `2026-04-23_stream_c_lidar.md` for the design
    // rationale.
    c10::intrusive_ptr<GridBatchData> accumGrid = grid;
    JaggedTensor accumTsdf    = tsdf;
    JaggedTensor accumWeights = weights;

    // Per-frame loop: build shell, call single-frame
    // `integrateTSDFFromPoints` logic inline, swap in new state.
    for (int64_t i = 0; i < N; ++i) {
        const torch::Tensor &ptsTensor = pointsPerFrame[i];
        TORCH_CHECK_VALUE(ptsTensor.dim() == 2 && ptsTensor.size(1) == 3,
                          "pointsPerFrame[", i, "] must be [N_i, 3]; got ",
                          ptsTensor.sizes());
        TORCH_CHECK_VALUE(ptsTensor.device() == tsdf.device(),
                          "pointsPerFrame[", i,
                          "] must be on the same device as tsdf");
        TORCH_CHECK_TYPE(ptsTensor.scalar_type() == tsdf.scalar_type(),
                         "pointsPerFrame[", i, "] dtype must match tsdf dtype");

        // Wrap the [N_i, 3] tensor as a batch-1 JaggedTensor to reuse
        // the existing buildUnionGrid + doIntegrateFromPoints helpers
        // unchanged.
        JaggedTensor ptsJagged = JaggedTensor(
            std::vector<torch::Tensor>{ptsTensor});

        // Matching slice of sensor origins. Keep as [1, 3] because
        // the existing single-frame API expects `[batchSize, 3]`
        // with batchSize = grid.batchSize() = 1.
        torch::Tensor originI =
            sensorOrigins.narrow(0, i, 1).contiguous();

        // Step 1: union grid for THIS frame's shell + current accum.
        auto unionGrid = buildUnionGrid(accumGrid, ptsJagged, truncationMargin);

        // Step 2: doIntegrateFromPoints (seed + ray-walk + normalize).
        // No features in this API (colour-features come via the
        // `*WithFeatures` variant; if we add a batched +features
        // entry point later, it plumbs features the same way as the
        // single-frame one does).
        const fvdb::JaggedTensor emptyFeatures =
            torch::empty({0, 0}, accumTsdf.jdata().options());
        const fvdb::JaggedTensor emptyPointFeatures =
            torch::empty({0, 0}, accumTsdf.jdata().options());

        auto [newTsdf, newWeights, _unusedFeatures] = doIntegrateFromPoints(
            static_cast<float>(truncationMargin),
            ptsJagged,
            originI,
            emptyPointFeatures,
            *unionGrid,
            *accumGrid,
            accumTsdf,
            accumWeights,
            emptyFeatures,
            carveFreeSpace);

        // Swap state to the new union grid + freshly-normalised
        // sidecars. Old accumGrid / accumTsdf / accumWeights refs
        // drop out of scope here and any GPU memory they held is
        // reclaimed by the caching allocator on next allocation.
        accumGrid    = unionGrid;
        accumTsdf    = newTsdf;
        accumWeights = newWeights;
    }

    if (profile_batch) {
        cudaEventRecord(evEnd);
        cudaEventSynchronize(evEnd);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, evStart, evEnd);
        std::fprintf(
            stderr,
            "[fvdb/tsdf_from_points_batch] N=%lld  incremental=%.2f ms  "
            "(%.2f ms/frame)  final_voxels=%lld  final_leaves=%lld\n",
            (long long)N, ms, ms / static_cast<float>(N),
            (long long)accumGrid->totalVoxels(),
            (long long)accumGrid->totalLeaves());
        cudaEventDestroy(evStart);
        cudaEventDestroy(evEnd);
    }

    return {accumGrid, accumTsdf, accumWeights};
}

} // namespace fvdb::detail::ops
