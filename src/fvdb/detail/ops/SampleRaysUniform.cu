// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/SampleRaysUniform.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/ForEachCPU.h>
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/cuda/Caching.cuh>
#include <fvdb/detail/utils/cuda/ForEachCUDA.cuh>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAMathCompat.h>

#include <type_traits>

namespace fvdb {
namespace detail {
namespace ops {
namespace {

template <typename ScalarType>
__hostdev__ float
_calcDt(ScalarType t, ScalarType coneAngle, ScalarType minStepSize, const ScalarType maxStepSize) {
    return nanovdb::math::Clamp(t * coneAngle, minStepSize, maxStepSize);
}

// Write-once streaming stores (`_storeStreaming` / `_storeStreamingPair`)
// for the generate kernel output live in
// <fvdb/detail/utils/cuda/Caching.cuh> and are reused across multiple ops.
// They are visible from this anonymous namespace via plain name lookup
// because they live in the enclosing `fvdb::detail::ops` namespace.

// Write one sample's (times, rayIdx) and bump the per-ray sample counter.
// CSEs `rayStartIdx + numSamples` into a single local, and routes the two
// output tensors through the streaming-store helpers above. Kept as a plain
// function template (not a lambda) because NVCC forbids extended __device__
// lambdas inside generic lambdas, which is what the launchers use.
//
// The (a, b) pair can mix types: when `ScalarType == c10::Half`, the HDDA
// iterator returns `it->t0` / `it->t1` in `at::opmath_type<Half> = float`,
// while the local `t0` / `t1` step variables stay in `ScalarType`. Rather
// than forcing callers to cast at every site, we take them as separate
// deduced types `A, B` and convert to `ScalarType` here. `ScalarType` must
// be specified explicitly at the call site because it's no longer used in a
// function parameter slot.
template <bool ReturnMidpoint,
          typename ScalarType,
          typename RayTimesAcc,
          typename JIdxAcc,
          typename A,
          typename B>
__hostdev__ __forceinline__ void
_emitSample(RayTimesAcc &outRayTimes,
            JIdxAcc &outJIdx,
            fvdb::JOffsetsType idx,
            int32_t rayIdx,
            const A &a,
            const B &b) {
    const ScalarType sa = static_cast<ScalarType>(a);
    const ScalarType sb = static_cast<ScalarType>(b);
    if constexpr (ReturnMidpoint) {
        // Cast the sum explicitly: `c10::Half + c10::Half` promotes to float
        // on some PyTorch / CUDA configurations, which would fail deduction
        // in `_storeStreaming`.
        const ScalarType mid = static_cast<ScalarType>((sa + sb) * ScalarType(0.5));
        _storeStreaming(&outRayTimes[idx][0], mid);
    } else {
        _storeStreamingPair(&outRayTimes[idx][0], sa, sb);
    }
    _storeStreaming(&outJIdx[idx], rayIdx);
}

// Counts the number of samples a ray will emit. Templated on `ConeZero`
// (coneAngle == 0) and `IncludeEndpoints` (include_end_segments) so NVCC can
// prune the dead branches and, critically, hoist `stepSize = minStepSize` out
// of the inner while-loops when ConeZero is true. That removes a Clamp+mul
// from the hot per-sample body and drops several live registers, which is the
// main lever we have on this latency-bound traversal.
template <typename ScalarType,
          bool ConeZero,
          bool IncludeEndpoints,
          template <typename T, int32_t D>
          typename JaggedAccessor,
          template <typename T, int32_t D>
          typename TensorAccessor>
__hostdev__ void
countSamplesPerRayCallback(int32_t bidx,
                           int32_t eidx,
                           const JaggedAccessor<ScalarType, 2> rayO, // [B*M, 3]
                           const JaggedAccessor<ScalarType, 2> rayD, // [B*M, 3]
                           const JaggedAccessor<ScalarType, 1> tMin, // [B*M,]
                           const JaggedAccessor<ScalarType, 1> tMax, // [B*M]
                           TensorAccessor<int32_t, 1> outRayCounts,  // [B*M]
                           BatchGridAccessor batchAccessor,
                           ScalarType minStepSize,
                           ScalarType coneAngle,
                           ScalarType eps) {
    const nanovdb::OnIndexGrid *gpuGrid = batchAccessor.grid(bidx);

    VoxelCoordTransform transform = batchAccessor.dualTransform(bidx);
    nanovdb::CoordBBox dualBbox   = batchAccessor.dualBbox(bidx);
    auto gridAcc                  = gpuGrid->getAccessor();

    const auto &rayOi                     = rayO.data()[eidx];
    const auto &rayDi                     = rayD.data()[eidx];
    const ScalarType tMini                = tMin.data()[eidx];
    const ScalarType tMaxi                = tMax.data()[eidx];
    nanovdb::math::Ray<ScalarType> rayVox = transform.applyToRay(
        rayOi[0], rayOi[1], rayOi[2], rayDi[0], rayDi[1], rayDi[2], tMini, tMaxi);

    if (!rayVox.clip(dualBbox)) {
        outRayCounts[eidx + 1] = 0;
        return;
    }

    // Count samples along ray
    int32_t numSamples = 0;

    // maxStepSize is only referenced on the non-ConeZero path; declaring it
    // at function scope keeps it visible to every `_calcDt` call in the loop
    // body without relying on constexpr (c10::Half has no constexpr ctor
    // from double in older PyTorch).
    const ScalarType maxStepSize = static_cast<ScalarType>(1e10);

    // Count samples
    ScalarType t0 = tMini;
    ScalarType stepSize;
    if constexpr (ConeZero) {
        // In the cone_angle == 0 case _calcDt collapses to minStepSize and is
        // loop-invariant; compute it once and let NVCC hoist it out.
        stepSize = minStepSize;
    } else {
        stepSize = _calcDt(t0, coneAngle, minStepSize, maxStepSize);
    }
    ScalarType t1;

    // For each contiguous segment of voxels
    for (auto it = HDDASegmentIterator<decltype(gridAcc), ScalarType>(rayVox, gridAcc);
         it.isValid();
         ++it) {
        const ScalarType deltaT = it->t1 - it->t0;
        if (deltaT < eps) {
            continue;
        }

        if constexpr (IncludeEndpoints) {
            // Step t0 consistently until it intersects the voxel (t0 is out of the voxel)
            // This MUST use the same rounding (`ceil`) as the generate-path callback below so that
            // both callbacks emit the same number of samples per ray.
            // With cone tracing (coneAngle > 0), `_calcDt` depends on `t0`, so any rounding
            // mismatch silently diverges the two paths and can OOB-write during sample generation.
            ScalarType distToVox = it->t0 - t0;
            t0 += c10::cuda::compat::ceil(distToVox / stepSize) * stepSize;
            t1 = t0 + stepSize;

            if (t0 > it->t1) {
                // A single step would take us past the end of the segment,
                // so we only record one step here.
                numSamples += 1;
                continue;
            }

            if ((t0 - it->t0) > nanovdb::math::Delta<ScalarType>::value()) {
                // There exists a gap between t0 and the start of the segment,
                // so we record it as a sample.
                numSamples += 1;
            }

            while (t1 < it->t1) {
                numSamples += 1;
                t0 = t1;
                if constexpr (!ConeZero) {
                    stepSize = _calcDt(t0, coneAngle, minStepSize, maxStepSize);
                }
                t1 += stepSize;
            }

            if ((it->t1 - t0) > nanovdb::math::Delta<ScalarType>::value()) {
                // There exists a gap between the end of the segment and t0,
                // so we record it as a sample.
                numSamples += 1;
            }
        } else {
            // Step t0 consistently until it intersects the voxel (tmid is in the voxel)
            ScalarType distToVox = it->t0 - t0;
            t0 = t0 + c10::cuda::compat::floor(distToVox / stepSize + 0.5f) * stepSize;
            if constexpr (!ConeZero) {
                stepSize = _calcDt(t0, coneAngle, minStepSize, maxStepSize);
            }
            t1 = t0 + stepSize;
            while ((t0 + t1) * 0.5 < it->t1 && (t0 + t1) * 0.5 >= it->t0) {
                numSamples += 1;
                t0 = t1;
                if constexpr (!ConeZero) {
                    stepSize = _calcDt(t0, coneAngle, minStepSize, maxStepSize);
                }
                t1 += stepSize;
            }
        }
    }
    outRayCounts[eidx + 1] = numSamples;
}

// Emits the actual samples computed by countSamplesPerRayCallback. Templated
// on ConeZero, IncludeEndpoints, and ReturnMidpoint for the same reasons as
// the count variant: prune dead branches, hoist stepSize, and drop the
// per-sample `returnMidpoint` test from the write path.
template <typename ScalarType,
          bool ConeZero,
          bool IncludeEndpoints,
          bool ReturnMidpoint,
          template <typename T, int32_t D>
          typename JaggedAccessor,
          template <typename T, int32_t D>
          typename TensorAccessor>
__hostdev__ void
generateRaySamplesCallback(int32_t bidx,
                           int32_t rayIdx,
                           const JaggedAccessor<ScalarType, 2> rayO,                // [B*M, 3]
                           const JaggedAccessor<ScalarType, 2> rayD,                // [B*M, 3]
                           const JaggedAccessor<ScalarType, 1> tMin,                // [B*M,]
                           const JaggedAccessor<ScalarType, 1> tMax,                // [B*M]
                           const TensorAccessor<fvdb::JOffsetsType, 1> outJOffsets, // [B*M, 2]
                           TensorAccessor<fvdb::JIdxType, 1> outJIdx,               // [B*M, 2]
                           TensorAccessor<fvdb::JLIdxType, 2> outJLIdx,             // [B*M, 2]
                           TensorAccessor<ScalarType, 2> outRayTimes,               // [B*M*S, 2]
                           BatchGridAccessor batchAccessor,
                           ScalarType minStepSize,
                           ScalarType coneAngle,
                           ScalarType eps) {
    const nanovdb::OnIndexGrid *gpuGrid = batchAccessor.grid(bidx);

    VoxelCoordTransform transform = batchAccessor.dualTransform(bidx);
    nanovdb::CoordBBox dualBbox   = batchAccessor.dualBbox(bidx);
    auto gridAcc                  = gpuGrid->getAccessor();

    const auto &rayOi                     = rayO.data()[rayIdx];
    const auto &rayDi                     = rayD.data()[rayIdx];
    const ScalarType tMini                = tMin.data()[rayIdx];
    const ScalarType tMaxi                = tMax.data()[rayIdx];
    nanovdb::math::Ray<ScalarType> rayVox = transform.applyToRay(
        rayOi[0], rayOi[1], rayOi[2], rayDi[0], rayDi[1], rayDi[2], tMini, tMaxi);

    if (outJLIdx.size(0) > 0) {
        const fvdb::JLIdxType batchStartIdx = rayO.offsetStart(bidx);
        outJLIdx[rayIdx][0]                 = bidx;
        outJLIdx[rayIdx][1]                 = rayIdx - batchStartIdx;
    }

    if (!rayVox.clip(dualBbox)) {
        return;
    }

    // Count samples along ray
    fvdb::JOffsetsType numSamples = 0;

    // See matching comment in countSamplesPerRayCallback.
    const ScalarType maxStepSize = static_cast<ScalarType>(1e10);

    // Track ray sample and region of space which it occupies
    ScalarType t0 = tMini;
    ScalarType stepSize;
    if constexpr (ConeZero) {
        stepSize = minStepSize;
    } else {
        stepSize = _calcDt(t0, coneAngle, minStepSize, maxStepSize);
    }
    ScalarType t1;

    const fvdb::JOffsetsType rayStartIdx = outJOffsets[rayIdx];

    // For each contiguous segment of voxels
    for (auto it = HDDASegmentIterator<decltype(gridAcc), ScalarType>(rayVox, gridAcc);
         it.isValid();
         ++it) {
        const ScalarType deltaT = it->t1 - it->t0;
        if (deltaT < eps) {
            continue;
        }

        if constexpr (IncludeEndpoints) {
            // Step t0 consistently until it intersects the voxel
            ScalarType distToVox = it->t0 - t0;
            t0 += c10::cuda::compat::ceil(distToVox / stepSize) * stepSize;
            t1 = t0 + stepSize;

            if (t0 > it->t1) {
                // A single step would take us past the end of the segment,
                // so we only record one step here.
                _emitSample<ReturnMidpoint, ScalarType>(
                    outRayTimes, outJIdx, rayStartIdx + numSamples, rayIdx, it->t0, it->t1);
                numSamples += 1;
                continue;
            }

            if ((t0 - it->t0) > nanovdb::math::Delta<ScalarType>::value()) {
                // There exists a gap between t0 and the start of the segment,
                // so we record it as a sample.
                _emitSample<ReturnMidpoint, ScalarType>(
                    outRayTimes, outJIdx, rayStartIdx + numSamples, rayIdx, it->t0, t0);
                numSamples += 1;
            }

            while (t1 < it->t1) {
                _emitSample<ReturnMidpoint, ScalarType>(
                    outRayTimes, outJIdx, rayStartIdx + numSamples, rayIdx, t0, t1);
                numSamples += 1;
                t0 = t1;
                if constexpr (!ConeZero) {
                    stepSize = _calcDt(t0, coneAngle, minStepSize, maxStepSize);
                }
                t1 += stepSize;
            }

            if ((it->t1 - t0) > nanovdb::math::Delta<ScalarType>::value()) {
                // There exists a gap between the end of the segment and t0,
                // so we record it as a sample.
                _emitSample<ReturnMidpoint, ScalarType>(
                    outRayTimes, outJIdx, rayStartIdx + numSamples, rayIdx, t0, it->t1);
                numSamples += 1;
            }
        } else {
            // Step t0 consistently until it intersects the voxel (tmid is in the voxel)
            ScalarType distToVox = it->t0 - t0;
            t0 = t0 + c10::cuda::compat::floor(distToVox / stepSize + 0.5f) * stepSize;
            if constexpr (!ConeZero) {
                stepSize = _calcDt(t0, coneAngle, minStepSize, maxStepSize);
            }
            t1 = t0 + stepSize;
            while ((t0 + t1) * 0.5 < it->t1 && (t0 + t1) * 0.5 >= it->t0) {
                _emitSample<ReturnMidpoint, ScalarType>(
                    outRayTimes, outJIdx, rayStartIdx + numSamples, rayIdx, t0, t1);
                numSamples += 1;
                t0 = t1;
                if constexpr (!ConeZero) {
                    stepSize = _calcDt(t0, coneAngle, minStepSize, maxStepSize);
                }
                t1 += stepSize;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel launchers
//
// NVCC forbids defining an extended __device__ lambda inside a *generic*
// lambda (CUDA C++ Programming Guide, "Restrictions on extended lambda
// expressions"). Plain function templates are fine, so we route all
// specialized launches through these helpers instead of wrapping them in a
// generic lambda at the call site.
// ---------------------------------------------------------------------------

template <torch::DeviceType DeviceTag,
          bool ConeZero,
          bool IncludeEndpoints,
          typename ScalarT,
          typename RayDirAcc,
          typename TMinAcc,
          typename TMaxAcc,
          typename CountsAcc,
          typename BatchAcc>
void
launchCount(const JaggedTensor &rayOrigins,
            const RayDirAcc rayDirectionsAcc,
            const TMinAcc tMinAcc,
            const TMaxAcc tMaxAcc,
            CountsAcc outCountsAcc,
            const BatchAcc batchAcc,
            ScalarT minStepSize,
            ScalarT coneAngle,
            ScalarT eps) {
    if constexpr (DeviceTag == torch::kCUDA) {
        auto cb = [=] __device__(int32_t bidx,
                                 int32_t eidx,
                                 int32_t cidx,
                                 JaggedRAcc64<ScalarT, 2> rayOriginsAcc) {
            countSamplesPerRayCallback<ScalarT,
                                       ConeZero,
                                       IncludeEndpoints,
                                       JaggedRAcc64,
                                       TorchRAcc64>(bidx,
                                                    eidx,
                                                    rayOriginsAcc,
                                                    rayDirectionsAcc,
                                                    tMinAcc,
                                                    tMaxAcc,
                                                    outCountsAcc,
                                                    batchAcc,
                                                    minStepSize,
                                                    coneAngle,
                                                    eps);
        };
        forEachJaggedElementChannelCUDA<ScalarT, 2>(1, rayOrigins, cb);
    } else {
        auto cb = [=](int32_t bidx,
                      int32_t eidx,
                      int32_t cidx,
                      JaggedAcc<ScalarT, 2> rayOriginsAcc) {
            countSamplesPerRayCallback<ScalarT, ConeZero, IncludeEndpoints, JaggedAcc, TorchAcc>(
                bidx,
                eidx,
                rayOriginsAcc,
                rayDirectionsAcc,
                tMinAcc,
                tMaxAcc,
                outCountsAcc,
                batchAcc,
                minStepSize,
                coneAngle,
                eps);
        };
        forEachJaggedElementChannelCPU<ScalarT, 2>(1, rayOrigins, cb);
    }
}

template <torch::DeviceType DeviceTag,
          bool ConeZero,
          bool IncludeEndpoints,
          bool ReturnMidpoint,
          typename ScalarT,
          typename RayDirAcc,
          typename TMinAcc,
          typename TMaxAcc,
          typename JOffsetsAcc,
          typename JIdxAcc,
          typename JLIdxAcc,
          typename RayTimesAcc,
          typename BatchAcc>
void
launchGenerate(const JaggedTensor &rayOrigins,
               const RayDirAcc rayDirectionsAcc,
               const TMinAcc tMinAcc,
               const TMaxAcc tMaxAcc,
               const JOffsetsAcc outJOffsetsAcc,
               JIdxAcc outJIdxAcc,
               JLIdxAcc outJLIdxAcc,
               RayTimesAcc outRayTimesAcc,
               const BatchAcc batchAcc,
               ScalarT minStepSize,
               ScalarT coneAngle,
               ScalarT eps) {
    if constexpr (DeviceTag == torch::kCUDA) {
        auto cb = [=] __device__(int32_t bidx,
                                 int32_t eidx,
                                 int32_t cidx,
                                 JaggedRAcc64<ScalarT, 2> rayOriginsAcc) {
            generateRaySamplesCallback<ScalarT,
                                       ConeZero,
                                       IncludeEndpoints,
                                       ReturnMidpoint,
                                       JaggedRAcc64,
                                       TorchRAcc64>(bidx,
                                                    eidx,
                                                    rayOriginsAcc,
                                                    rayDirectionsAcc,
                                                    tMinAcc,
                                                    tMaxAcc,
                                                    outJOffsetsAcc,
                                                    outJIdxAcc,
                                                    outJLIdxAcc,
                                                    outRayTimesAcc,
                                                    batchAcc,
                                                    minStepSize,
                                                    coneAngle,
                                                    eps);
        };
        forEachJaggedElementChannelCUDA<ScalarT, 2>(1, rayOrigins, cb);
    } else {
        auto cb =
            [=](int32_t bidx, int32_t eidx, int32_t cidx, JaggedAcc<ScalarT, 2> rayOriginsAcc) {
                generateRaySamplesCallback<ScalarT,
                                           ConeZero,
                                           IncludeEndpoints,
                                           ReturnMidpoint,
                                           JaggedAcc,
                                           TorchAcc>(bidx,
                                                     eidx,
                                                     rayOriginsAcc,
                                                     rayDirectionsAcc,
                                                     tMinAcc,
                                                     tMaxAcc,
                                                     outJOffsetsAcc,
                                                     outJIdxAcc,
                                                     outJLIdxAcc,
                                                     outRayTimesAcc,
                                                     batchAcc,
                                                     minStepSize,
                                                     coneAngle,
                                                     eps);
            };
        forEachJaggedElementChannelCPU<ScalarT, 2>(1, rayOrigins, cb);
    }
}

template <torch::DeviceType DeviceTag>
JaggedTensor
UniformRaySamples(const GridBatchData &batchHdl,
                  const JaggedTensor &rayOrigins,
                  const JaggedTensor &rayDirections,
                  const JaggedTensor &tMin,
                  const JaggedTensor &tMax,
                  const double minStepSize,
                  const double coneAngle,
                  const bool includeEndpointSegments,
                  const bool returnMidpoint,
                  const double eps) {
    batchHdl.checkDevice(rayOrigins);
    batchHdl.checkDevice(rayDirections);
    batchHdl.checkDevice(tMin);
    batchHdl.checkDevice(tMax);
    TORCH_CHECK_TYPE(rayOrigins.is_floating_point(), "ray_origins must have a floating point type");
    TORCH_CHECK_TYPE(rayDirections.is_floating_point(),
                     "ray_directions must have a floating point type");
    TORCH_CHECK_TYPE(tMin.is_floating_point(), "tmin must have a floating point type");
    TORCH_CHECK_TYPE(tMax.is_floating_point(), "tmax must have a floating point type");

    TORCH_CHECK_VALUE(batchHdl.batchSize() == rayOrigins.num_outer_lists(),
                      "ray_origins must have the same batch size as the grid batch");
    TORCH_CHECK_VALUE(batchHdl.batchSize() == rayDirections.num_outer_lists(),
                      "ray_directions must have the same batch size as the grid batch");
    TORCH_CHECK_VALUE(batchHdl.batchSize() == tMin.num_outer_lists(),
                      "t_min must have the same batch size as the grid batch");
    TORCH_CHECK_VALUE(batchHdl.batchSize() == tMax.num_outer_lists(),
                      "t_max must have the same batch size as the grid batch");

    TORCH_CHECK_TYPE(rayOrigins.dtype() == rayDirections.dtype(),
                     "all tensors must have the same type");
    TORCH_CHECK_TYPE(tMin.dtype() == tMin.dtype(), "all tensors must have the same type");
    TORCH_CHECK_TYPE(tMin.dtype() == rayOrigins.dtype(), "all tensors must have the same type");

    TORCH_CHECK(rayOrigins.rdim() == 2,
                std::string("Expected ray_origins to have 2 dimensions (shape (n, 3)) but got ") +
                    std::to_string(rayOrigins.rdim()) + " dimensions");
    TORCH_CHECK(
        rayDirections.rdim() == 2,
        std::string("Expected ray_directions to have 2 dimensions (shape (n, 3)) but got ") +
            std::to_string(rayDirections.rdim()) + " dimensions");
    TORCH_CHECK(tMin.rdim() == 1,
                std::string("Expected tmin to have 1 dimension (shape (n,)) but got ") +
                    std::to_string(tMin.rdim()) + " dimensions");
    TORCH_CHECK(tMax.rdim() == 1,
                std::string("Expected tmin to have 1 dimension (shape (n,)) but got ") +
                    std::to_string(tMax.rdim()) + " dimensions");
    TORCH_CHECK(rayOrigins.rsize(0) == tMin.rsize(0),
                "ray_origins and tmin must have the same size in dimension 0 but got " +
                    std::to_string(rayOrigins.rsize(0)) + " and " + std::to_string(tMin.rsize(0)));
    TORCH_CHECK(rayOrigins.rsize(0) == tMax.rsize(0),
                "ray_origins and tmin must have the same size in dimension 0 but got " +
                    std::to_string(rayOrigins.rsize(0)) + " and " + std::to_string(tMin.rsize(0)));
    TORCH_CHECK(rayOrigins.rsize(0) == rayDirections.rsize(0),
                "ray_origins and ray_directions must have the same size in dimension 0 but got " +
                    std::to_string(rayOrigins.rsize(0)) + " and " +
                    std::to_string(rayDirections.rsize(0)));
    TORCH_CHECK(minStepSize > 0.0, "minStepSize must be positive");
    TORCH_CHECK(coneAngle >= 0.0, "coneAngle must be none negitive");
    TORCH_CHECK(rayOrigins.ldim() == 1, "Invalid list dimension for ray origins.");
    TORCH_CHECK(rayDirections.ldim() == 1, "Invalid list dimension for ray directions.");

    return AT_DISPATCH_V2(
        rayOrigins.scalar_type(),
        "UniformRaySamples",
        AT_WRAP([&]() -> JaggedTensor {
            const auto optsF =
                torch::TensorOptions().dtype(rayOrigins.dtype()).device(rayOrigins.device());
            const auto optsI32 =
                torch::TensorOptions().dtype(torch::kInt32).device(rayOrigins.device());
            const auto optsJIdx =
                torch::TensorOptions().dtype(fvdb::JIdxScalarType).device(rayOrigins.device());
            const auto optsJLIdx =
                torch::TensorOptions().dtype(fvdb::JLIdxScalarType).device(rayOrigins.device());

            auto batchAcc         = gridBatchAccessor<DeviceTag>(batchHdl);
            auto rayDirectionsAcc = jaggedAccessor<DeviceTag, scalar_t, 2>(rayDirections);

            auto tMinAcc = jaggedAccessor<DeviceTag, scalar_t, 1>(tMin);
            auto tMaxAcc = jaggedAccessor<DeviceTag, scalar_t, 1>(tMax);

            // Count number of segments along each ray
            torch::Tensor rayCounts = torch::zeros({rayOrigins.rsize(0) + 1}, optsI32); // [B*M]
            auto outCountsAcc       = tensorAccessor<DeviceTag, int32_t, 1>(rayCounts);

            // Dispatch to the (ConeZero, IncludeEndpoints) specialization of
            // `launchCount` so NVCC can prune dead branches and hoist
            // stepSize out of the inner loop in `countSamplesPerRayCallback`.
            // See the callback for why. The helper itself is a plain
            // function template (not a generic lambda) because NVCC forbids
            // __device__ lambdas inside generic lambdas.
            const scalar_t castedMinStepSize = static_cast<scalar_t>(minStepSize);
            const scalar_t castedConeAngle   = static_cast<scalar_t>(coneAngle);
            const scalar_t castedEps         = static_cast<scalar_t>(eps);

            if (coneAngle == 0.0) {
                if (includeEndpointSegments) {
                    launchCount<DeviceTag, true, true, scalar_t>(rayOrigins,
                                                                 rayDirectionsAcc,
                                                                 tMinAcc,
                                                                 tMaxAcc,
                                                                 outCountsAcc,
                                                                 batchAcc,
                                                                 castedMinStepSize,
                                                                 castedConeAngle,
                                                                 castedEps);
                } else {
                    launchCount<DeviceTag, true, false, scalar_t>(rayOrigins,
                                                                  rayDirectionsAcc,
                                                                  tMinAcc,
                                                                  tMaxAcc,
                                                                  outCountsAcc,
                                                                  batchAcc,
                                                                  castedMinStepSize,
                                                                  castedConeAngle,
                                                                  castedEps);
                }
            } else {
                if (includeEndpointSegments) {
                    launchCount<DeviceTag, false, true, scalar_t>(rayOrigins,
                                                                  rayDirectionsAcc,
                                                                  tMinAcc,
                                                                  tMaxAcc,
                                                                  outCountsAcc,
                                                                  batchAcc,
                                                                  castedMinStepSize,
                                                                  castedConeAngle,
                                                                  castedEps);
                } else {
                    launchCount<DeviceTag, false, false, scalar_t>(rayOrigins,
                                                                   rayDirectionsAcc,
                                                                   tMinAcc,
                                                                   tMaxAcc,
                                                                   outCountsAcc,
                                                                   batchAcc,
                                                                   castedMinStepSize,
                                                                   castedConeAngle,
                                                                   castedEps);
                }
            }

            // Compute joffsets for the output samples
            torch::Tensor outJOffsets = rayCounts.cumsum(0, fvdb::JOffsetsScalarType); // [B*M]
            const fvdb::JOffsetsType totalSamples =
                outJOffsets[outJOffsets.size(0) - 1].item<fvdb::JOffsetsType>();

            // Allocate output JaggedTensor indexing data
            torch::Tensor outJLidx =
                torch::empty({outJOffsets.size(0) - 1, 2}, optsJLIdx);      // [total_rays, 2]
            torch::Tensor outJIdx = torch::zeros({totalSamples}, optsJIdx); // [total_intersections]

            // Allocate output tensors
            torch::Tensor outRayTimes =
                torch::zeros({totalSamples, returnMidpoint ? 1 : 2}, optsF); // [B*M*S, 2]

            // Compute output voxels and times
            auto outJOffsetsAcc = tensorAccessor<DeviceTag, fvdb::JOffsetsType, 1>(outJOffsets);
            auto outJIdxAcc     = tensorAccessor<DeviceTag, fvdb::JIdxType, 1>(outJIdx);
            auto outJLIdxAcc    = tensorAccessor<DeviceTag, fvdb::JLIdxType, 2>(outJLidx);

            auto outRayTimesAcc = tensorAccessor<DeviceTag, scalar_t, 2>(outRayTimes);

            // Same (ConeZero, IncludeEndpoints, ReturnMidpoint) specialization
            // pattern as the count pass above.
            if (coneAngle == 0.0) {
                if (includeEndpointSegments) {
                    if (returnMidpoint) {
                        launchGenerate<DeviceTag, true, true, true, scalar_t>(rayOrigins,
                                                                              rayDirectionsAcc,
                                                                              tMinAcc,
                                                                              tMaxAcc,
                                                                              outJOffsetsAcc,
                                                                              outJIdxAcc,
                                                                              outJLIdxAcc,
                                                                              outRayTimesAcc,
                                                                              batchAcc,
                                                                              castedMinStepSize,
                                                                              castedConeAngle,
                                                                              castedEps);
                    } else {
                        launchGenerate<DeviceTag, true, true, false, scalar_t>(rayOrigins,
                                                                               rayDirectionsAcc,
                                                                               tMinAcc,
                                                                               tMaxAcc,
                                                                               outJOffsetsAcc,
                                                                               outJIdxAcc,
                                                                               outJLIdxAcc,
                                                                               outRayTimesAcc,
                                                                               batchAcc,
                                                                               castedMinStepSize,
                                                                               castedConeAngle,
                                                                               castedEps);
                    }
                } else {
                    if (returnMidpoint) {
                        launchGenerate<DeviceTag, true, false, true, scalar_t>(rayOrigins,
                                                                               rayDirectionsAcc,
                                                                               tMinAcc,
                                                                               tMaxAcc,
                                                                               outJOffsetsAcc,
                                                                               outJIdxAcc,
                                                                               outJLIdxAcc,
                                                                               outRayTimesAcc,
                                                                               batchAcc,
                                                                               castedMinStepSize,
                                                                               castedConeAngle,
                                                                               castedEps);
                    } else {
                        launchGenerate<DeviceTag, true, false, false, scalar_t>(rayOrigins,
                                                                                rayDirectionsAcc,
                                                                                tMinAcc,
                                                                                tMaxAcc,
                                                                                outJOffsetsAcc,
                                                                                outJIdxAcc,
                                                                                outJLIdxAcc,
                                                                                outRayTimesAcc,
                                                                                batchAcc,
                                                                                castedMinStepSize,
                                                                                castedConeAngle,
                                                                                castedEps);
                    }
                }
            } else {
                if (includeEndpointSegments) {
                    if (returnMidpoint) {
                        launchGenerate<DeviceTag, false, true, true, scalar_t>(rayOrigins,
                                                                               rayDirectionsAcc,
                                                                               tMinAcc,
                                                                               tMaxAcc,
                                                                               outJOffsetsAcc,
                                                                               outJIdxAcc,
                                                                               outJLIdxAcc,
                                                                               outRayTimesAcc,
                                                                               batchAcc,
                                                                               castedMinStepSize,
                                                                               castedConeAngle,
                                                                               castedEps);
                    } else {
                        launchGenerate<DeviceTag, false, true, false, scalar_t>(rayOrigins,
                                                                                rayDirectionsAcc,
                                                                                tMinAcc,
                                                                                tMaxAcc,
                                                                                outJOffsetsAcc,
                                                                                outJIdxAcc,
                                                                                outJLIdxAcc,
                                                                                outRayTimesAcc,
                                                                                batchAcc,
                                                                                castedMinStepSize,
                                                                                castedConeAngle,
                                                                                castedEps);
                    }
                } else {
                    if (returnMidpoint) {
                        launchGenerate<DeviceTag, false, false, true, scalar_t>(rayOrigins,
                                                                                rayDirectionsAcc,
                                                                                tMinAcc,
                                                                                tMaxAcc,
                                                                                outJOffsetsAcc,
                                                                                outJIdxAcc,
                                                                                outJLIdxAcc,
                                                                                outRayTimesAcc,
                                                                                batchAcc,
                                                                                castedMinStepSize,
                                                                                castedConeAngle,
                                                                                castedEps);
                    } else {
                        launchGenerate<DeviceTag, false, false, false, scalar_t>(rayOrigins,
                                                                                 rayDirectionsAcc,
                                                                                 tMinAcc,
                                                                                 tMaxAcc,
                                                                                 outJOffsetsAcc,
                                                                                 outJIdxAcc,
                                                                                 outJLIdxAcc,
                                                                                 outRayTimesAcc,
                                                                                 batchAcc,
                                                                                 castedMinStepSize,
                                                                                 castedConeAngle,
                                                                                 castedEps);
                    }
                }
            }

            if (returnMidpoint) {
                outRayTimes = outRayTimes.squeeze(-1);
            }
            return JaggedTensor::from_jdata_joffsets_jidx_and_lidx_unsafe(
                outRayTimes, outJOffsets, outJIdx, outJLidx, batchHdl.batchSize());
        }),
        AT_EXPAND(AT_FLOATING_TYPES),
        c10::kHalf);
}

} // anonymous namespace

JaggedTensor
uniformRaySamples(const GridBatchData &batchHdl,
                  const JaggedTensor &rayO,
                  const JaggedTensor &rayD,
                  const JaggedTensor &tMin,
                  const JaggedTensor &tMax,
                  const double minStepSize,
                  const double coneAngle,
                  const bool includeEndSegments,
                  const bool returnMidpoint,
                  const double eps) {
    TORCH_CHECK_VALUE(
        rayO.ldim() == 1,
        "Expected ray_origins to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        rayO.ldim(),
        "list dimensions");
    TORCH_CHECK_VALUE(
        rayD.ldim() == 1,
        "Expected ray_directions to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        rayD.ldim(),
        "list dimensions");
    TORCH_CHECK_VALUE(
        tMin.ldim() == 1,
        "Expected t_min to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        tMin.ldim(),
        "list dimensions");
    TORCH_CHECK_VALUE(
        tMax.ldim() == 1,
        "Expected t_max to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        tMax.ldim(),
        "list dimensions");
    return FVDB_DISPATCH_KERNEL_DEVICE(rayO.device(), [&]() {
        return UniformRaySamples<DeviceTag>(batchHdl,
                                            rayO,
                                            rayD,
                                            tMin,
                                            tMax,
                                            minStepSize,
                                            coneAngle,
                                            includeEndSegments,
                                            returnMidpoint,
                                            eps);
    });
}

} // namespace ops
} // namespace detail
} // namespace fvdb
