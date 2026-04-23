// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/SampleNearest.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/ForEachCPU.h>
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/cuda/ForEachCUDA.cuh>
#include <fvdb/detail/utils/cuda/ForEachPrivateUse1.cuh>

#include <ATen/OpMathType.h>
#include <c10/cuda/CUDAException.h>

#include <cstdint>

namespace fvdb {
namespace detail {
namespace ops {

// ---------------------------------------------------------------------------
// Resolve the nearest active corner among the 8 surrounding voxels.
//
// Uses the same NanoVDB-style ijk traversal order as resolveTrilinearStencil
// to maximise ReadAccessor node-cache hits.
//
// Returns the cumulative linear index of the nearest active corner, or -1 if
// none of the 8 corners are active.
// ---------------------------------------------------------------------------
template <typename MathType, typename GridAccessorType>
__hostdev__ inline int64_t
resolveNearestCorner(const nanovdb::math::Vec3<MathType> &xyz,
                     GridAccessorType &gridAcc,
                     int64_t baseOffset) {
    nanovdb::Coord ijk = xyz.floor();
    const MathType u   = xyz[0] - MathType(ijk[0]);
    const MathType v   = xyz[1] - MathType(ijk[1]);
    const MathType w   = xyz[2] - MathType(ijk[2]);

    // Squared-distance from the query point to each of the 8 unit-cube
    // corners (di, dj, dk) where di/dj/dk are each 0 or 1.
    // dist2 = (di - u)^2 + (dj - v)^2 + (dk - w)^2
    //
    // Corner ordering matches TrilinearStencil.h for accessor-cache
    // efficiency (increment one component at a time).
    struct CornerInfo {
        int di, dj, dk;
    };
    constexpr CornerInfo corners[8] = {
        {0, 0, 0}, // corner 0  (i,   j,   k  )
        {0, 0, 1}, // corner 1  (i,   j,   k+1)
        {0, 1, 1}, // corner 2  (i,   j+1, k+1)
        {0, 1, 0}, // corner 3  (i,   j+1, k  )
        {1, 0, 0}, // corner 4  (i+1, j,   k  )
        {1, 0, 1}, // corner 5  (i+1, j,   k+1)
        {1, 1, 1}, // corner 6  (i+1, j+1, k+1)
        {1, 1, 0}, // corner 7  (i+1, j+1, k  )
    };

    int64_t bestIdx   = -1;
    MathType bestDist = MathType(4); // > max possible dist2 of 3

    // Walk corners in the same ijk-mutation order as TrilinearStencil.h.
#define FVDB_NEAREST_CORNER(CORNER)                            \
    if (gridAcc.isActive(ijk)) {                               \
        const MathType du = MathType(corners[CORNER].di) - u;  \
        const MathType dv = MathType(corners[CORNER].dj) - v;  \
        const MathType dw = MathType(corners[CORNER].dk) - w;  \
        const MathType d2 = du * du + dv * dv + dw * dw;       \
        if (d2 < bestDist) {                                   \
            bestDist = d2;                                     \
            bestIdx  = gridAcc.getValue(ijk) - 1 + baseOffset; \
        }                                                      \
    }

    FVDB_NEAREST_CORNER(0) // (i,   j,   k  )
    ijk[2] += 1;
    FVDB_NEAREST_CORNER(1) // (i,   j,   k+1)
    ijk[1] += 1;
    FVDB_NEAREST_CORNER(2) // (i,   j+1, k+1)
    ijk[2] -= 1;
    FVDB_NEAREST_CORNER(3) // (i,   j+1, k  )
    ijk[0] += 1;
    ijk[1] -= 1;
    FVDB_NEAREST_CORNER(4) // (i+1, j,   k  )
    ijk[2] += 1;
    FVDB_NEAREST_CORNER(5) // (i+1, j,   k+1)
    ijk[1] += 1;
    FVDB_NEAREST_CORNER(6) // (i+1, j+1, k+1)
    ijk[2] -= 1;
    FVDB_NEAREST_CORNER(7) // (i+1, j+1, k  )

#undef FVDB_NEAREST_CORNER

    return bestIdx;
}

// ---------------------------------------------------------------------------
// Per-point callback (scalar, all dtypes, CPU + GPU)
// ---------------------------------------------------------------------------
template <typename ScalarType,
          template <typename T, int32_t D>
          typename JaggedAccessor,
          template <typename T, int32_t D>
          typename TensorAccessor>
__hostdev__ void
sampleNearestCallback(int32_t bidx,
                      int32_t eidx,
                      int32_t /*cidx*/,
                      JaggedAccessor<ScalarType, 2> points,
                      TensorAccessor<ScalarType, 2> gridData,
                      BatchGridAccessor batchAccessor,
                      TensorAccessor<ScalarType, 2> outFeatures,
                      int64_t *outIndices,
                      int64_t numChannels) {
    using MathType = at::opmath_type<ScalarType>;

    const auto &pointsData               = points.data();
    const nanovdb::OnIndexGrid *grid     = batchAccessor.grid(bidx);
    const VoxelCoordTransform &transform = batchAccessor.primalTransform(bidx);
    const int64_t baseOffset             = batchAccessor.voxelOffset(bidx);
    auto gridAcc                         = grid->tree().getAccessor();

    const nanovdb::math::Vec3<MathType> xyz =
        transform.apply(static_cast<MathType>(pointsData[eidx][0]),
                        static_cast<MathType>(pointsData[eidx][1]),
                        static_cast<MathType>(pointsData[eidx][2]));

    const int64_t bestIdx = resolveNearestCorner(xyz, gridAcc, baseOffset);

    outIndices[eidx] = bestIdx;

    if (bestIdx < 0)
        return;

    for (int64_t c = 0; c < numChannels; ++c) {
        outFeatures[eidx][c] = gridData[bestIdx][c];
    }
}

// ---------------------------------------------------------------------------
// Per-point callback (float, vec4 fast path, GPU only)
// ---------------------------------------------------------------------------
template <template <typename T, int32_t D> typename JaggedAccessor,
          template <typename T, int32_t D>
          typename TensorAccessor>
__device__ void
sampleNearestCallbackVec4(int32_t bidx,
                          int32_t eidx,
                          int32_t /*cidx*/,
                          JaggedAccessor<float, 2> points,
                          TensorAccessor<float, 2> gridData,
                          BatchGridAccessor batchAccessor,
                          TensorAccessor<float, 2> outFeatures,
                          int64_t *outIndices,
                          int64_t numChannels) {
    const auto &pointsData               = points.data();
    const nanovdb::OnIndexGrid *grid     = batchAccessor.grid(bidx);
    const VoxelCoordTransform &transform = batchAccessor.primalTransform(bidx);
    const int64_t baseOffset             = batchAccessor.voxelOffset(bidx);
    auto gridAcc                         = grid->tree().getAccessor();

    const nanovdb::math::Vec3<float> xyz =
        transform.apply(pointsData[eidx][0], pointsData[eidx][1], pointsData[eidx][2]);

    const int64_t bestIdx = resolveNearestCorner(xyz, gridAcc, baseOffset);

    outIndices[eidx] = bestIdx;

    if (bestIdx < 0)
        return;

    const int64_t numGroups = numChannels / 4;
    for (int64_t g = 0; g < numGroups; ++g) {
        const int64_t cBase = g * 4;
        *reinterpret_cast<float4 *>(&outFeatures[eidx][cBase]) =
            *reinterpret_cast<const float4 *>(&gridData[bestIdx][cBase]);
    }
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------
template <torch::DeviceType DeviceTag, typename scalar_t>
std::vector<torch::Tensor>
SampleGridNearest(const GridBatchData &batchHdl,
                  const JaggedTensor &points,
                  const torch::Tensor &gridData) {
    auto opts = torch::TensorOptions()
                    .dtype(gridData.dtype())
                    .device(gridData.device())
                    .requires_grad(gridData.requires_grad());
    auto optsIdx = torch::TensorOptions().dtype(torch::kInt64).device(gridData.device());

    torch::Tensor gridDataReshape = featureCoalescedView(gridData).contiguous();
    torch::Tensor outFeatures =
        torch::zeros({points.rsize(0), gridDataReshape.size(1)}, opts).contiguous();
    torch::Tensor outIndices = torch::full({points.rsize(0)}, int64_t(-1), optsIdx).contiguous();
    auto outShape            = spliceShape({points.rsize(0)}, gridData, 1);

    auto batchAcc             = gridBatchAccessor<DeviceTag>(batchHdl);
    auto gridDataAcc          = tensorAccessor<DeviceTag, scalar_t, 2>(gridDataReshape);
    auto outFeaturesAcc       = tensorAccessor<DeviceTag, scalar_t, 2>(outFeatures);
    int64_t *outIndicesPtr    = outIndices.data_ptr<int64_t>();
    const int64_t numChannels = gridDataReshape.size(1);

    if constexpr (DeviceTag == torch::kCUDA || DeviceTag == torch::kPrivateUse1) {
        auto dispatchForEach = [&](const auto &cb) {
            if constexpr (DeviceTag == torch::kCUDA) {
                forEachJaggedElementChannelCUDA<scalar_t, 2>(int64_t(1), points, cb);
            } else {
                forEachJaggedElementChannelPrivateUse1<scalar_t, 2>(int64_t(1), points, cb);
            }
        };

        if constexpr (std::is_same_v<scalar_t, float>) {
            if (numChannels >= 4 && numChannels % 4 == 0 &&
                reinterpret_cast<uintptr_t>(gridDataReshape.data_ptr<float>()) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(outFeatures.data_ptr<float>()) % 16 == 0) {
                auto cb = [=] __device__(int32_t bidx,
                                         int32_t eidx,
                                         int32_t cidx,
                                         JaggedRAcc64<float, 2> pts) {
                    sampleNearestCallbackVec4<JaggedRAcc64, TorchRAcc64>(bidx,
                                                                         eidx,
                                                                         cidx,
                                                                         pts,
                                                                         gridDataAcc,
                                                                         batchAcc,
                                                                         outFeaturesAcc,
                                                                         outIndicesPtr,
                                                                         numChannels);
                };
                dispatchForEach(cb);
            } else {
                auto cb = [=] __device__(int32_t bidx,
                                         int32_t eidx,
                                         int32_t cidx,
                                         JaggedRAcc64<float, 2> pts) {
                    sampleNearestCallback<float, JaggedRAcc64, TorchRAcc64>(bidx,
                                                                            eidx,
                                                                            cidx,
                                                                            pts,
                                                                            gridDataAcc,
                                                                            batchAcc,
                                                                            outFeaturesAcc,
                                                                            outIndicesPtr,
                                                                            numChannels);
                };
                dispatchForEach(cb);
            }
        } else {
            auto cb = [=] __device__(int32_t bidx,
                                     int32_t eidx,
                                     int32_t cidx,
                                     JaggedRAcc64<scalar_t, 2> pts) {
                sampleNearestCallback<scalar_t, JaggedRAcc64, TorchRAcc64>(bidx,
                                                                           eidx,
                                                                           cidx,
                                                                           pts,
                                                                           gridDataAcc,
                                                                           batchAcc,
                                                                           outFeaturesAcc,
                                                                           outIndicesPtr,
                                                                           numChannels);
            };
            dispatchForEach(cb);
        }
    } else {
        auto cb = [=](int32_t bidx, int32_t eidx, int32_t cidx, JaggedAcc<scalar_t, 2> pts) {
            sampleNearestCallback<scalar_t, JaggedAcc, TorchAcc>(bidx,
                                                                 eidx,
                                                                 cidx,
                                                                 pts,
                                                                 gridDataAcc,
                                                                 batchAcc,
                                                                 outFeaturesAcc,
                                                                 outIndicesPtr,
                                                                 numChannels);
        };
        forEachJaggedElementChannelCPU<scalar_t, 2>(int64_t(1), points, cb);
    }

    return {outFeatures.reshape(outShape), outIndices};
}

template <torch::DeviceType DeviceTag>
std::vector<torch::Tensor>
dispatchSampleGridNearest(const GridBatchData &batchHdl,
                          const JaggedTensor &points,
                          const torch::Tensor &gridData) {
    return AT_DISPATCH_V2(
        points.scalar_type(),
        "SampleGridNearest",
        AT_WRAP([&] { return SampleGridNearest<DeviceTag, scalar_t>(batchHdl, points, gridData); }),
        AT_EXPAND(AT_FLOATING_TYPES),
        c10::kHalf);
}

template std::vector<torch::Tensor> dispatchSampleGridNearest<torch::kCPU>(const GridBatchData &,
                                                                           const JaggedTensor &,
                                                                           const torch::Tensor &);
template std::vector<torch::Tensor> dispatchSampleGridNearest<torch::kCUDA>(const GridBatchData &,
                                                                            const JaggedTensor &,
                                                                            const torch::Tensor &);
template std::vector<torch::Tensor> dispatchSampleGridNearest<torch::kPrivateUse1>(
    const GridBatchData &, const JaggedTensor &, const torch::Tensor &);

std::vector<torch::Tensor>
sampleNearest(const GridBatchData &batchHdl,
              const JaggedTensor &points,
              const torch::Tensor &gridData) {
    batchHdl.checkNonEmptyGrid();
    TORCH_CHECK_VALUE(points.device() == gridData.device(),
                      "points and data must be on the same device");
    batchHdl.checkDevice(points);
    batchHdl.checkDevice(gridData);
    points.check_valid();
    TORCH_CHECK_TYPE(points.is_floating_point(), "points must have a floating point type");
    TORCH_CHECK_TYPE(points.dtype() == gridData.dtype(), "all tensors must have the same type");
    TORCH_CHECK_VALUE(points.rdim() == 2,
                      "Expected points to have shape [B*M, 3] (wrong number of dimensions)");
    TORCH_CHECK(points.numel() > 0, "Empty tensor (points)");
    TORCH_CHECK(points.rsize(1) == 3, "points must have shape [B, M, 3] (points must be 3D)");
    TORCH_CHECK_TYPE(gridData.is_floating_point(), "data must have a floating point type");
    TORCH_CHECK_VALUE(gridData.dim() >= 2,
                      "Expected data to have shape [N, *] (at least 2 dimensions)");
    TORCH_CHECK(gridData.numel() > 0, "Empty tensor (data)");
    TORCH_CHECK(gridData.size(0) == batchHdl.totalVoxels(),
                "grid_data must have one value per voxel (shape [N, *]) (wrong first dimension)");
    return FVDB_DISPATCH_KERNEL(points.device(), [&]() {
        return dispatchSampleGridNearest<DeviceTag>(batchHdl, points, gridData);
    });
}

} // namespace ops
} // namespace detail
} // namespace fvdb
