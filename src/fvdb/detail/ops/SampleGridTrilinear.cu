// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/SampleGridTrilinear.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/ForEachCPU.h>
#include <fvdb/detail/utils/cuda/ForEachCUDA.cuh>
#include <fvdb/detail/utils/cuda/ForEachPrivateUse1.cuh>

#include <ATen/OpMathType.h>
#include <c10/cuda/CUDAException.h>

#include <cstdint>

namespace fvdb {
namespace detail {
namespace ops {

// Resolve the 8 trilinear corner indices and weights in a single pass.
// Uses NanoVDB-style coordinate traversal (increment one component at a time)
// to maximize ReadAccessor node-cache hits across successive lookups.
// Returns a bitmask indicating which corners are active in the grid.
template <typename MathType, typename GridAccessorType>
__hostdev__ inline uint8_t
resolveTrilinearStencil(const nanovdb::math::Vec3<MathType> &xyz,
                        GridAccessorType &gridAcc,
                        int64_t baseOffset,
                        int64_t (&indices)[8],
                        MathType (&weights)[8]) {
    nanovdb::Coord ijk = xyz.floor();
    const MathType u   = xyz[0] - MathType(ijk[0]);
    const MathType v   = xyz[1] - MathType(ijk[1]);
    const MathType w   = xyz[2] - MathType(ijk[2]);
    const MathType ONE = MathType(1);
    const MathType U = ONE - u, V = ONE - v, W = ONE - w;

    uint8_t activeMask = 0;

#define FVDB_RESOLVE_CORNER(CORNER, WEIGHT)                       \
    weights[CORNER] = (WEIGHT);                                   \
    if (gridAcc.isActive(ijk)) {                                  \
        activeMask |= (1 << (CORNER));                            \
        indices[CORNER] = gridAcc.getValue(ijk) - 1 + baseOffset; \
    }

    FVDB_RESOLVE_CORNER(0, U * V * W) // (i,   j,   k  )
    ijk[2] += 1;
    FVDB_RESOLVE_CORNER(1, U * V * w) // (i,   j,   k+1)
    ijk[1] += 1;
    FVDB_RESOLVE_CORNER(2, U * v * w) // (i,   j+1, k+1)
    ijk[2] -= 1;
    FVDB_RESOLVE_CORNER(3, U * v * W) // (i,   j+1, k  )
    ijk[0] += 1;
    ijk[1] -= 1;
    FVDB_RESOLVE_CORNER(4, u * V * W) // (i+1, j,   k  )
    ijk[2] += 1;
    FVDB_RESOLVE_CORNER(5, u * V * w) // (i+1, j,   k+1)
    ijk[1] += 1;
    FVDB_RESOLVE_CORNER(6, u * v * w) // (i+1, j+1, k+1)
    ijk[2] -= 1;
    FVDB_RESOLVE_CORNER(7, u * v * W) // (i+1, j+1, k  )

#undef FVDB_RESOLVE_CORNER

    return activeMask;
}

// One-thread-per-point callback: resolve the 8-corner stencil once, then iterate
// all channels with scalar loads. Works on both CPU and GPU, all scalar types.
template <typename ScalarType,
          template <typename T, int32_t D> typename JaggedAccessor,
          template <typename T, int32_t D> typename TensorAccessor>
__hostdev__ void
sampleTrilinearStencilCallback(int32_t bidx,
                               int32_t eidx,
                               int32_t /*cidx*/,
                               JaggedAccessor<ScalarType, 2> points,
                               TensorAccessor<ScalarType, 2> gridData,
                               BatchGridAccessor batchAccessor,
                               TensorAccessor<ScalarType, 2> outFeatures,
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

    int64_t indices[8] = {};
    MathType weights[8];
    const uint8_t activeMask = resolveTrilinearStencil(xyz, gridAcc, baseOffset, indices, weights);

    if (activeMask == 0)
        return;

    for (int64_t c = 0; c < numChannels; ++c) {
        MathType accum = MathType(0);
#pragma unroll
        for (int corner = 0; corner < 8; ++corner) {
            if (activeMask & (1 << corner)) {
                accum += weights[corner] * static_cast<MathType>(gridData[indices[corner]][c]);
            }
        }
        outFeatures[eidx][c] = static_cast<ScalarType>(accum);
    }
}

// One-thread-per-point callback: resolve the 8-corner stencil once, then iterate
// channels in float4 groups with explicit 128-bit loads/stores. GPU only.
template <template <typename T, int32_t D> typename JaggedAccessor,
          template <typename T, int32_t D> typename TensorAccessor>
__device__ void
sampleTrilinearStencilCallbackVec4(int32_t bidx,
                                   int32_t eidx,
                                   int32_t /*cidx*/,
                                   JaggedAccessor<float, 2> points,
                                   TensorAccessor<float, 2> gridData,
                                   BatchGridAccessor batchAccessor,
                                   TensorAccessor<float, 2> outFeatures,
                                   int64_t numChannels) {
    const auto &pointsData               = points.data();
    const nanovdb::OnIndexGrid *grid     = batchAccessor.grid(bidx);
    const VoxelCoordTransform &transform = batchAccessor.primalTransform(bidx);
    const int64_t baseOffset             = batchAccessor.voxelOffset(bidx);
    auto gridAcc                         = grid->tree().getAccessor();

    const nanovdb::math::Vec3<float> xyz =
        transform.apply(pointsData[eidx][0], pointsData[eidx][1], pointsData[eidx][2]);

    int64_t indices[8] = {};
    float weights[8];
    const uint8_t activeMask = resolveTrilinearStencil(xyz, gridAcc, baseOffset, indices, weights);

    if (activeMask == 0)
        return;

    const int64_t numGroups = numChannels / 4;
    for (int64_t g = 0; g < numGroups; ++g) {
        const int64_t cBase = g * 4;
        float4 accum        = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
        for (int corner = 0; corner < 8; ++corner) {
            if (activeMask & (1 << corner)) {
                const float wt = weights[corner];
                const float4 val =
                    *reinterpret_cast<const float4 *>(&gridData[indices[corner]][cBase]);
                accum.x += wt * val.x;
                accum.y += wt * val.y;
                accum.z += wt * val.z;
                accum.w += wt * val.w;
            }
        }
        *reinterpret_cast<float4 *>(&outFeatures[eidx][cBase]) = accum;
    }
}

template <torch::DeviceType DeviceTag, typename scalar_t>
std::vector<torch::Tensor>
SampleGridTrilinear(const GridBatchImpl &batchHdl,
                    const JaggedTensor &points,
                    const torch::Tensor &gridData) {
    auto opts = torch::TensorOptions()
                    .dtype(gridData.dtype())
                    .device(gridData.device())
                    .requires_grad(gridData.requires_grad());
    torch::Tensor gridDataReshape = featureCoalescedView(gridData).contiguous();     // [B*N, -1]
    torch::Tensor outFeatures =
        torch::zeros({points.rsize(0), gridDataReshape.size(1)}, opts).contiguous(); // [B*M, -1]
    auto outShape = spliceShape({points.rsize(0)}, gridData, 1);                     // [B*M, *]

    auto batchAcc             = gridBatchAccessor<DeviceTag>(batchHdl);
    auto gridDataAcc          = tensorAccessor<DeviceTag, scalar_t, 2>(gridDataReshape);
    auto outFeaturesAcc       = tensorAccessor<DeviceTag, scalar_t, 2>(outFeatures);
    const int64_t numChannels = gridDataReshape.size(1);

    if constexpr (DeviceTag == torch::kCUDA || DeviceTag == torch::kPrivateUse1) {
        auto dispatchForEach = [&](const auto &cb) {
            if constexpr (DeviceTag == torch::kCUDA) {
                forEachJaggedElementChannelCUDA<scalar_t, 2>(
                    DEFAULT_BLOCK_DIM, int64_t(1), points, cb);
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
                                         JaggedRAcc32<float, 2> pts) {
                    sampleTrilinearStencilCallbackVec4<JaggedRAcc32, TorchRAcc32>(
                        bidx, eidx, cidx, pts, gridDataAcc, batchAcc, outFeaturesAcc, numChannels);
                };
                dispatchForEach(cb);
            } else {
                auto cb = [=] __device__(int32_t bidx,
                                         int32_t eidx,
                                         int32_t cidx,
                                         JaggedRAcc32<float, 2> pts) {
                    sampleTrilinearStencilCallback<float, JaggedRAcc32, TorchRAcc32>(
                        bidx, eidx, cidx, pts, gridDataAcc, batchAcc, outFeaturesAcc, numChannels);
                };
                dispatchForEach(cb);
            }
        } else {
            auto cb = [=] __device__(int32_t bidx,
                                     int32_t eidx,
                                     int32_t cidx,
                                     JaggedRAcc32<scalar_t, 2> pts) {
                sampleTrilinearStencilCallback<scalar_t, JaggedRAcc32, TorchRAcc32>(
                    bidx, eidx, cidx, pts, gridDataAcc, batchAcc, outFeaturesAcc, numChannels);
            };
            dispatchForEach(cb);
        }
    } else {
        auto cb = [=](int32_t bidx, int32_t eidx, int32_t cidx, JaggedAcc<scalar_t, 2> pts) {
            sampleTrilinearStencilCallback<scalar_t, JaggedAcc, TorchAcc>(
                bidx, eidx, cidx, pts, gridDataAcc, batchAcc, outFeaturesAcc, numChannels);
        };
        forEachJaggedElementChannelCPU<scalar_t, 2>(int64_t(1), points, cb);
    }

    return {outFeatures.reshape(outShape)};
}

template <torch::DeviceType DeviceTag>
std::vector<torch::Tensor>
dispatchSampleGridTrilinear(const GridBatchImpl &batchHdl,
                            const JaggedTensor &points,
                            const torch::Tensor &gridData) {
    return AT_DISPATCH_V2(
        points.scalar_type(),
        "SampleGridTrilinear",
        AT_WRAP(
            [&] { return SampleGridTrilinear<DeviceTag, scalar_t>(batchHdl, points, gridData); }),
        AT_EXPAND(AT_FLOATING_TYPES),
        c10::kHalf);
}

template std::vector<torch::Tensor> dispatchSampleGridTrilinear<torch::kCPU>(const GridBatchImpl &,
                                                                             const JaggedTensor &,
                                                                             const torch::Tensor &);
template std::vector<torch::Tensor> dispatchSampleGridTrilinear<torch::kCUDA>(
    const GridBatchImpl &, const JaggedTensor &, const torch::Tensor &);
template std::vector<torch::Tensor> dispatchSampleGridTrilinear<torch::kPrivateUse1>(
    const GridBatchImpl &, const JaggedTensor &, const torch::Tensor &);

} // namespace ops
} // namespace detail
} // namespace fvdb
