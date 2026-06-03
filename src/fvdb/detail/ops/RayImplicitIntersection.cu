// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/RayImplicitIntersection.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/ForEachCPU.h>
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/cuda/Caching.cuh>
#include <fvdb/detail/utils/cuda/ForEachCUDA.cuh>

#include <ATen/OpMathType.h>
#include <c10/cuda/CUDAException.h>

namespace fvdb {
namespace detail {
namespace ops {
namespace {

constexpr int INVALID_SIGN = 10;

template <typename T>
__forceinline__ __hostdev__ int
sgn(const T &val) {
    if (val != val) {
        return INVALID_SIGN;
    }
    return (T(0) < val) - (val < T(0));
}

// Per-ray SDF zero-crossing search.
//
// Semantics match nanovdb::ZeroCrossing: the FIRST valid (non-NaN) voxel
// along the ray seeds the sign reference, and the first subsequent voxel
// with the opposite sign is reported as the hit. This naturally handles
// both rays that start outside the surface (first sample positive, hit on
// crossing into the negative band) AND rays that start inside the surface
// (first sample negative, hit on crossing back out into the positive
// band) without baking a fixed "positive = outside" convention into the
// kernel. NaN voxels are treated as gaps so the band-continuity check
// will fall back to bracket-entry time on the next valid voxel.
//
// Performance notes:
//
//  - All output writes go through `_storeStreaming` (`__stwt`, `.CS` in
//    SASS) so the write-once `outTimes` line never gets promoted into L1
//    and evicts the active-mask leaf data we are walking.
//  - `gridScalars` is read through the read-only data cache via
//    `_loadReadOnly` (`__ldg`, `.NC` in SASS) so the side-buffer SDF data
//    shares cache capacity instead of contending with the leaf accessor.
//  - Time / scalar arithmetic is done in `MathType = at::opmath_type<ScalarT>`
//    so `c10::Half` rays get fp32-precision interpolation; we cast back to
//    `ScalarT` only at the streaming-store boundary.
//  - `EpsZero` is a compile-time flag dispatched from the launcher when
//    `eps == 0`, eliding the `if (deltaT < eps) continue;` branch which is
//    the overwhelmingly common case.
//  - Iteration is over leaf voxels only via `HDDALeafVoxelIterator` (see
//    HDDAIterators.h): it yields exactly the dim==1 active voxels and skips
//    coarse active tiles in a single HDDA step. This matches the narrow-band
//    inner loop of `nanovdb::ZeroCrossing` and guarantees the per-voxel
//    `gridScalars` index below (`getValue(ijk) - 1`) is always in-bounds.
template <typename ScalarT,
          bool EpsZero,
          template <typename T, int32_t D>
          typename JaggedAccessor,
          template <typename T, int32_t D>
          typename TensorAccessor>
__hostdev__ inline void
rayImplicitCallback(int32_t bidx,
                    int32_t eidx,
                    JaggedAccessor<ScalarT, 2> raysO,
                    JaggedAccessor<ScalarT, 2> raysD,
                    JaggedAccessor<ScalarT, 1> gridScalarsJ,
                    BatchGridAccessor batchAcc,
                    TensorAccessor<ScalarT, 1> outTimes,
                    ScalarT eps) {
    using MathType = at::opmath_type<ScalarT>;

    const nanovdb::OnIndexGrid *gpuGrid = batchAcc.grid(bidx);
    const auto gridAcc                  = gpuGrid->getAccessor();

    const VoxelCoordTransform transform = batchAcc.dualTransform(bidx);
    const nanovdb::CoordBBox dualBbox   = batchAcc.dualBbox(bidx);

    const auto rayO  = raysO.data()[eidx];
    const auto rayD  = raysD.data()[eidx];
    auto gridScalars = gridScalarsJ.data();
    nanovdb::math::Ray<ScalarT> rayVox =
        transform.applyToRay(rayO[0], rayO[1], rayO[2], rayD[0], rayD[1], rayD[2]);
    if (!rayVox.clip(dualBbox)) {
        _storeStreaming(&outTimes[eidx], static_cast<ScalarT>(-1.0));
        return;
    }

    // Reference state is seeded from the FIRST valid (non-NaN) voxel
    // along the ray (see kernel docs above). `scalarSign == INVALID_SIGN`
    // is the "no reference yet" sentinel; we only flag a hit once it has
    // been replaced by a real voxel sign and the next valid voxel
    // disagrees with it.
    int scalarSign      = INVALID_SIGN;
    MathType lastScalar = MathType(0);
    MathType lastTime   = MathType(0);
    MathType lastT1     = MathType(0);
    bool found          = false;

    for (auto it = HDDALeafVoxelIterator<decltype(gridAcc), ScalarT>(rayVox, gridAcc); it.isValid();
         it++) {
        const MathType t0     = it->second.t0;
        const MathType t1     = it->second.t1;
        const MathType deltaT = t1 - t0;
        if constexpr (!EpsZero) {
            if (deltaT < static_cast<MathType>(eps)) {
                continue;
            }
        }

        const nanovdb::Coord &ijk = it->first;
        const int64_t voxelIndex  = gridAcc.getValue(ijk) - 1;
        const MathType voxelValue = static_cast<MathType>(_loadReadOnly(&gridScalars[voxelIndex]));
        const MathType voxelTime  = MathType(0.5) * (t0 + t1);
        const int voxelSign       = sgn(voxelValue);

        const bool isValid = (voxelSign != INVALID_SIGN);
        const bool hasRef  = (scalarSign != INVALID_SIGN);
        const bool isHit   = isValid && hasRef && (scalarSign != voxelSign);

        if (isHit) {
            // Sub-voxel-precise hit time when the previous sample is
            // contiguous along the ray (typical narrow-band traversal):
            // linear-interpolate between the bracketing (time, value)
            // pairs. When there is a gap between the previous valid voxel
            // and this one — either inactive voxels in the iterator or a
            // run of NaN tile values — fall back to bracket-entry time
            // to avoid interpolating across empty space (matching
            // nanovdb::ZeroCrossing precision in that case).
            const bool contiguous   = (t0 == lastT1);
            const MathType lamCont  = voxelValue / (voxelValue - lastScalar);
            const MathType timeCont = lamCont * lastTime + (MathType(1) - lamCont) * voxelTime;
            const MathType hitTime  = contiguous ? timeCont : t0;
            _storeStreaming(&outTimes[eidx], static_cast<ScalarT>(hitTime));
            found = true;
            break;
        }

        // Only update the sign-reference state from valid (non-NaN) voxels;
        // NaN voxels are treated as gaps so a subsequent valid voxel will
        // see `contiguous == false` and fall back to bracket-entry time.
        if (isValid) {
            scalarSign = voxelSign;
            lastScalar = voxelValue;
            lastTime   = voxelTime;
            lastT1     = t1;
        }
    }
    if (!found) {
        _storeStreaming(&outTimes[eidx], static_cast<ScalarT>(-1.0));
    }
}

template <torch::DeviceType DeviceTag>
JaggedTensor
RayImplicitIntersection(const GridBatchData &batchHdl,
                        const JaggedTensor &rayO,
                        const JaggedTensor &rayD,
                        const JaggedTensor &gridScalars,
                        float eps) {
    batchHdl.checkDevice(rayO);
    batchHdl.checkDevice(rayD);
    batchHdl.checkDevice(gridScalars);
    TORCH_CHECK_TYPE(rayO.is_floating_point(), "ray_origins must have a floating point type");
    TORCH_CHECK_TYPE(rayD.is_floating_point(), "ray_directions must have a floating point type");
    TORCH_CHECK_TYPE(gridScalars.is_floating_point(),
                     "gridScalars must have a floating point type");

    TORCH_CHECK_TYPE(rayO.dtype() == rayD.dtype(), "all tensors must have the same type");
    TORCH_CHECK_TYPE(rayD.dtype() == gridScalars.dtype(), "all tensors must have the same type");

    TORCH_CHECK(rayO.rdim() == 2,
                std::string("Expected ray_origins to have 2 dimensions (shape (n, 3)) but got ") +
                    std::to_string(rayO.rdim()) + " dimensions");
    TORCH_CHECK(
        rayD.rdim() == 2,
        std::string("Expected ray_directions to have 2 dimensions (shape (n, 3)) but got ") +
            std::to_string(rayD.rdim()) + " dimensions");

    TORCH_CHECK(rayD.rsize(0) == rayO.rsize(0),
                std::string("Expected ray_origins and ray_directions to have the same shape "));
    TORCH_CHECK(rayO.rsize(1) == 3, std::string("Expected ray_origins to have shape (n, 3)"));
    TORCH_CHECK(rayD.rsize(1) == 3, std::string("Expected ray_directions to have shape (n, 3)"));

    TORCH_CHECK(
        gridScalars.rdim() == 1,
        std::string("Expected grid_scalars to have 1 dimension (shape (num_voxels,)) but got ") +
            std::to_string(gridScalars.rdim()) + " dimensions");
    TORCH_CHECK(gridScalars.rsize(0) == batchHdl.totalVoxels(),
                std::string("ray_implicit_intersection iterates leaf voxels "
                            "(HDDALeafVoxelIterator) and needs exactly one scalar per active "
                            "voxel, but got ") +
                    std::to_string(gridScalars.rsize(0)) + " scalars for " +
                    std::to_string(batchHdl.totalVoxels()) +
                    " voxels. For per-active-value data that includes coarse tiles, iterate "
                    "with HDDAActiveValueIterator and branch on getDim().");

    auto optsF             = torch::TensorOptions().dtype(rayO.dtype()).device(rayO.device());
    torch::Tensor outTimes = torch::empty({rayO.rsize(0)}, optsF);

    // `eps == 0` is the dominant call site (the Python binding's default).
    // Specialise the kernel on `EpsZero` so NVCC drops the per-voxel
    // `deltaT < eps` branch and one register entirely in that case.
    const bool epsZero = (eps == 0.0f);

    AT_DISPATCH_V2(
        rayO.scalar_type(),
        "RayImplicitIntersection",
        AT_WRAP([&]() {
            auto batchAcc       = gridBatchAccessor<DeviceTag>(batchHdl);
            auto rayDAcc        = jaggedAccessor<DeviceTag, scalar_t, 2>(rayD);
            auto gridScalarsAcc = jaggedAccessor<DeviceTag, scalar_t, 1>(gridScalars);
            auto outTimesAcc    = tensorAccessor<DeviceTag, scalar_t, 1>(outTimes);

            if constexpr (DeviceTag == torch::kCUDA) {
                if (epsZero) {
                    auto cb = [=] __device__(int32_t bidx,
                                             int32_t eidx,
                                             int32_t cidx,
                                             JaggedRAcc64<scalar_t, 2> rOA) {
                        rayImplicitCallback<scalar_t,
                                            /*EpsZero=*/true,
                                            JaggedRAcc64,
                                            TorchRAcc64>(bidx,
                                                         eidx,
                                                         rOA,
                                                         rayDAcc,
                                                         gridScalarsAcc,
                                                         batchAcc,
                                                         outTimesAcc,
                                                         scalar_t(eps));
                    };
                    forEachJaggedElementChannelCUDA<scalar_t, 2>(1, rayO, cb);
                } else {
                    auto cb = [=] __device__(int32_t bidx,
                                             int32_t eidx,
                                             int32_t cidx,
                                             JaggedRAcc64<scalar_t, 2> rOA) {
                        rayImplicitCallback<scalar_t,
                                            /*EpsZero=*/false,
                                            JaggedRAcc64,
                                            TorchRAcc64>(bidx,
                                                         eidx,
                                                         rOA,
                                                         rayDAcc,
                                                         gridScalarsAcc,
                                                         batchAcc,
                                                         outTimesAcc,
                                                         scalar_t(eps));
                    };
                    forEachJaggedElementChannelCUDA<scalar_t, 2>(1, rayO, cb);
                }
            } else {
                if (epsZero) {
                    auto cb =
                        [=](int32_t bidx, int32_t eidx, int32_t cidx, JaggedAcc<scalar_t, 2> rOA) {
                            rayImplicitCallback<scalar_t,
                                                /*EpsZero=*/true,
                                                JaggedAcc,
                                                TorchAcc>(bidx,
                                                          eidx,
                                                          rOA,
                                                          rayDAcc,
                                                          gridScalarsAcc,
                                                          batchAcc,
                                                          outTimesAcc,
                                                          scalar_t(eps));
                        };
                    forEachJaggedElementChannelCPU<scalar_t, 2>(1, rayO, cb);
                } else {
                    auto cb =
                        [=](int32_t bidx, int32_t eidx, int32_t cidx, JaggedAcc<scalar_t, 2> rOA) {
                            rayImplicitCallback<scalar_t,
                                                /*EpsZero=*/false,
                                                JaggedAcc,
                                                TorchAcc>(bidx,
                                                          eidx,
                                                          rOA,
                                                          rayDAcc,
                                                          gridScalarsAcc,
                                                          batchAcc,
                                                          outTimesAcc,
                                                          scalar_t(eps));
                        };
                    forEachJaggedElementChannelCPU<scalar_t, 2>(1, rayO, cb);
                }
            }
        }),
        AT_EXPAND(AT_FLOATING_TYPES),
        c10::kHalf);

    return rayO.jagged_like(outTimes);
}

} // anonymous namespace

JaggedTensor
rayImplicitIntersection(const GridBatchData &batchHdl,
                        const JaggedTensor &rayOrigins,
                        const JaggedTensor &rayDirections,
                        const JaggedTensor &gridScalars,
                        float eps) {
    TORCH_CHECK_VALUE(
        rayOrigins.ldim() == 1,
        "Expected ray_origins to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        rayOrigins.ldim(),
        "list dimensions");
    TORCH_CHECK_VALUE(
        rayDirections.ldim() == 1,
        "Expected ray_directions to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        rayDirections.ldim(),
        "list dimensions");
    TORCH_CHECK_VALUE(
        gridScalars.ldim() == 1,
        "Expected grid_scalars to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        gridScalars.ldim(),
        "list dimensions");
    return FVDB_DISPATCH_KERNEL_DEVICE(rayOrigins.device(), [&]() {
        return RayImplicitIntersection<DeviceTag>(
            batchHdl, rayOrigins, rayDirections, gridScalars, eps);
    });
}

} // namespace ops
} // namespace detail
} // namespace fvdb
