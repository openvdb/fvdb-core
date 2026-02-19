// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/SampleGridTrilinearWithGradBackward.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/ForEachCPU.h>
#include <fvdb/detail/utils/cuda/ForEachCUDA.cuh>
#include <fvdb/detail/utils/cuda/ForEachPrivateUse1.cuh>

#include <ATen/OpMathType.h>
#include <THC/THCAtomics.cuh>
#include <c10/cuda/CUDAException.h>

#include <cstdint>

namespace fvdb {
namespace detail {
namespace ops {

template <typename MathType, typename GridAccessorType>
__hostdev__ inline uint8_t
resolveTrilinearStencilWithGrad(const nanovdb::math::Vec3<MathType> &xyz,
                                GridAccessorType &gridAcc,
                                int64_t baseOffset,
                                int64_t (&indices)[8],
                                MathType (&weights)[8],
                                MathType (&gradWeights)[8][3]) {
    nanovdb::Coord ijk = xyz.floor();
    const MathType u   = xyz[0] - MathType(ijk[0]);
    const MathType v   = xyz[1] - MathType(ijk[1]);
    const MathType w   = xyz[2] - MathType(ijk[2]);
    const MathType ONE = MathType(1);
    const MathType U = ONE - u, V = ONE - v, W = ONE - w;

    uint8_t activeMask = 0;

#define FVDB_RESOLVE_CORNER_GRAD(CORNER, WT, GU, GV, GW)          \
    weights[CORNER]        = (WT);                                \
    gradWeights[CORNER][0] = (GU);                                \
    gradWeights[CORNER][1] = (GV);                                \
    gradWeights[CORNER][2] = (GW);                                \
    if (gridAcc.isActive(ijk)) {                                  \
        activeMask |= (1 << (CORNER));                            \
        indices[CORNER] = gridAcc.getValue(ijk) - 1 + baseOffset; \
    }

    FVDB_RESOLVE_CORNER_GRAD(0, U * V * W, -V * W, -U * W, -U * V) // (i,   j,   k  )
    ijk[2] += 1;
    FVDB_RESOLVE_CORNER_GRAD(1, U * V * w, -V * w, -U * w, U * V)  // (i,   j,   k+1)
    ijk[1] += 1;
    FVDB_RESOLVE_CORNER_GRAD(2, U * v * w, -v * w, U * w, U * v)   // (i,   j+1, k+1)
    ijk[2] -= 1;
    FVDB_RESOLVE_CORNER_GRAD(3, U * v * W, -v * W, U * W, -U * v)  // (i,   j+1, k  )
    ijk[0] += 1;
    ijk[1] -= 1;
    FVDB_RESOLVE_CORNER_GRAD(4, u * V * W, V * W, -u * W, -u * V)  // (i+1, j,   k  )
    ijk[2] += 1;
    FVDB_RESOLVE_CORNER_GRAD(5, u * V * w, V * w, -u * w, u * V)   // (i+1, j,   k+1)
    ijk[1] += 1;
    FVDB_RESOLVE_CORNER_GRAD(6, u * v * w, v * w, u * w, u * v)    // (i+1, j+1, k+1)
    ijk[2] -= 1;
    FVDB_RESOLVE_CORNER_GRAD(7, u * v * W, v * W, u * W, -u * v)   // (i+1, j+1, k  )

#undef FVDB_RESOLVE_CORNER_GRAD

    return activeMask;
}

// One-thread-per-point scalar callback. Resolves stencil once, then scatters
// gradient contributions across all channels using cached indices and weights.
template <torch::DeviceType DeviceTag,
          typename ScalarType,
          template <typename T, int32_t D> typename JaggedAccessor,
          template <typename T, int32_t D> typename TensorAccessor>
__hostdev__ void
sampleTrilinearWithGradBackwardStencilCallback(int32_t bidx,
                                               int32_t eidx,
                                               int32_t /*cidx*/,
                                               JaggedAccessor<ScalarType, 2> points,
                                               TensorAccessor<ScalarType, 2> gradOutFeatures,
                                               TensorAccessor<ScalarType, 3> gradOutGradFeatures,
                                               BatchGridAccessor batchAccessor,
                                               TensorAccessor<ScalarType, 2> outGridData,
                                               int64_t numChannels) {
    using MathType = at::opmath_type<ScalarType>;

    const auto &pointsData = points.data();

    const nanovdb::OnIndexGrid *gpuGrid = batchAccessor.grid(bidx);
    auto transform                      = batchAccessor.primalTransform(bidx);
    const int64_t baseOffset            = batchAccessor.voxelOffset(bidx);
    auto gridAcc                        = gpuGrid->getAccessor();

    const nanovdb::math::Vec3<MathType> xyz =
        transform.apply(static_cast<MathType>(pointsData[eidx][0]),
                        static_cast<MathType>(pointsData[eidx][1]),
                        static_cast<MathType>(pointsData[eidx][2]));
    auto gradTransform = transform.template applyGrad<MathType>(xyz);

    int64_t indices[8] = {};
    MathType weights[8];
    MathType gradWeights[8][3];
    const uint8_t activeMask =
        resolveTrilinearStencilWithGrad(xyz, gridAcc, baseOffset, indices, weights, gradWeights);

    if (activeMask == 0)
        return;

    for (int corner = 0; corner < 8; ++corner) {
        if (!(activeMask & (1 << corner)))
            continue;
        for (int64_t c = 0; c < numChannels; ++c) {
            MathType addValue = weights[corner] * static_cast<MathType>(gradOutFeatures[eidx][c]);
#pragma unroll
            for (int dim = 0; dim < 3; ++dim) {
                addValue += gradWeights[corner][dim] *
                            static_cast<MathType>(gradOutGradFeatures[eidx][c][dim]) *
                            gradTransform[dim];
            }
            if constexpr (DeviceTag == torch::kCUDA) {
                gpuAtomicAddNoReturn(&outGridData[indices[corner]][c],
                                     static_cast<ScalarType>(addValue));
            } else {
                outGridData[indices[corner]][c] += static_cast<ScalarType>(addValue);
            }
        }
    }
}

// One-thread-per-point Vec4 callback. GPU only. Corner-outer / channel-group-inner
// ordering keeps atomic writes to the same voxel row contiguous.
template <torch::DeviceType DeviceTag,
          template <typename T, int32_t D> typename JaggedAccessor,
          template <typename T, int32_t D> typename TensorAccessor>
__device__ void
sampleTrilinearWithGradBackwardStencilCallbackVec4(int32_t bidx,
                                                   int32_t eidx,
                                                   int32_t /*cidx*/,
                                                   JaggedAccessor<float, 2> points,
                                                   TensorAccessor<float, 2> gradOutFeatures,
                                                   TensorAccessor<float, 3> gradOutGradFeatures,
                                                   BatchGridAccessor batchAccessor,
                                                   TensorAccessor<float, 2> outGridData,
                                                   int64_t numChannels) {
    const auto &pointsData = points.data();

    const nanovdb::OnIndexGrid *gpuGrid = batchAccessor.grid(bidx);
    auto transform                      = batchAccessor.primalTransform(bidx);
    const int64_t baseOffset            = batchAccessor.voxelOffset(bidx);
    auto gridAcc                        = gpuGrid->getAccessor();

    const nanovdb::math::Vec3<float> xyz =
        transform.apply(pointsData[eidx][0], pointsData[eidx][1], pointsData[eidx][2]);
    auto gradTransform = transform.template applyGrad<float>(xyz);

    int64_t indices[8] = {};
    float weights[8];
    float gradWeights[8][3];
    const uint8_t activeMask =
        resolveTrilinearStencilWithGrad(xyz, gridAcc, baseOffset, indices, weights, gradWeights);

    if (activeMask == 0)
        return;

    const float gx = gradTransform[0], gy = gradTransform[1], gz = gradTransform[2];
    const int64_t numGroups = numChannels / 4;

    for (int corner = 0; corner < 8; ++corner) {
        if (!(activeMask & (1 << corner)))
            continue;
        const float wt    = weights[corner];
        const float gw[3] = {
            gradWeights[corner][0], gradWeights[corner][1], gradWeights[corner][2]};

        for (int64_t g = 0; g < numGroups; ++g) {
            const int64_t cBase = g * 4;

            const float4 gf = *reinterpret_cast<const float4 *>(&gradOutFeatures[eidx][cBase]);

            float addValue[4];
            addValue[0] = wt * gf.x;
            addValue[1] = wt * gf.y;
            addValue[2] = wt * gf.z;
            addValue[3] = wt * gf.w;

#pragma unroll
            for (int i = 0; i < 4; ++i) {
                addValue[i] += gw[0] * gradOutGradFeatures[eidx][cBase + i][0] * gx +
                               gw[1] * gradOutGradFeatures[eidx][cBase + i][1] * gy +
                               gw[2] * gradOutGradFeatures[eidx][cBase + i][2] * gz;
            }

            if constexpr (DeviceTag == torch::kCUDA) {
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    gpuAtomicAddNoReturn(&outGridData[indices[corner]][cBase + i], addValue[i]);
            } else if constexpr (DeviceTag == torch::kPrivateUse1) {
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    atomicAdd_system(&outGridData[indices[corner]][cBase + i], addValue[i]);
            }
        }
    }
}

template <torch::DeviceType DeviceTag, typename scalar_t>
torch::Tensor
SampleGridTrilinearWithGradBackward(const GridBatchImpl &batchHdl,
                                    const JaggedTensor &points,
                                    const torch::Tensor &data,
                                    const torch::Tensor &gradOutFeatures,
                                    const torch::Tensor &gradOutGradFeatures) {
    torch::Tensor dataReshape = featureCoalescedView(data); // [N, -1]

    // FIXME handle more dimensions
    torch::Tensor outGrad = torch::zeros_like(dataReshape);          // [N, -1]
    auto outShape         = spliceShape({outGrad.size(0)}, data, 1); // [B*M, *]

    // Make gradOutFeatures contiguous for vectorized reads
    torch::Tensor gradOutFeaturesContig = gradOutFeatures.contiguous();

    auto batchAcc               = gridBatchAccessor<DeviceTag>(batchHdl);
    auto gradOutFeaturesAcc     = tensorAccessor<DeviceTag, scalar_t, 2>(gradOutFeaturesContig);
    auto gradOutGradFeaturesAcc = tensorAccessor<DeviceTag, scalar_t, 3>(gradOutGradFeatures);
    auto outGradAcc             = tensorAccessor<DeviceTag, scalar_t, 2>(outGrad);
    const int64_t numChannels   = outGrad.size(1);

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
                reinterpret_cast<uintptr_t>(gradOutFeaturesContig.data_ptr<float>()) % 16 == 0) {
                auto cb = [=] __device__(int32_t bidx,
                                         int32_t eidx,
                                         int32_t cidx,
                                         JaggedRAcc32<float, 2> pts) {
                    sampleTrilinearWithGradBackwardStencilCallbackVec4<DeviceTag,
                                                                       JaggedRAcc32,
                                                                       TorchRAcc32>(
                        bidx,
                        eidx,
                        cidx,
                        pts,
                        gradOutFeaturesAcc,
                        gradOutGradFeaturesAcc,
                        batchAcc,
                        outGradAcc,
                        numChannels);
                };
                dispatchForEach(cb);
            } else {
                auto cb = [=] __device__(int32_t bidx,
                                         int32_t eidx,
                                         int32_t cidx,
                                         JaggedRAcc32<float, 2> pts) {
                    sampleTrilinearWithGradBackwardStencilCallback<DeviceTag,
                                                                   float,
                                                                   JaggedRAcc32,
                                                                   TorchRAcc32>(
                        bidx,
                        eidx,
                        cidx,
                        pts,
                        gradOutFeaturesAcc,
                        gradOutGradFeaturesAcc,
                        batchAcc,
                        outGradAcc,
                        numChannels);
                };
                dispatchForEach(cb);
            }
        } else {
            auto cb = [=] __device__(int32_t bidx,
                                     int32_t eidx,
                                     int32_t cidx,
                                     JaggedRAcc32<scalar_t, 2> pts) {
                sampleTrilinearWithGradBackwardStencilCallback<DeviceTag,
                                                               scalar_t,
                                                               JaggedRAcc32,
                                                               TorchRAcc32>(bidx,
                                                                            eidx,
                                                                            cidx,
                                                                            pts,
                                                                            gradOutFeaturesAcc,
                                                                            gradOutGradFeaturesAcc,
                                                                            batchAcc,
                                                                            outGradAcc,
                                                                            numChannels);
            };
            dispatchForEach(cb);
        }
    } else {
        auto cb = [=](int32_t bidx, int32_t eidx, int32_t cidx, JaggedAcc<scalar_t, 2> pts) {
            sampleTrilinearWithGradBackwardStencilCallback<DeviceTag,
                                                           scalar_t,
                                                           JaggedAcc,
                                                           TorchAcc>(bidx,
                                                                     eidx,
                                                                     cidx,
                                                                     pts,
                                                                     gradOutFeaturesAcc,
                                                                     gradOutGradFeaturesAcc,
                                                                     batchAcc,
                                                                     outGradAcc,
                                                                     numChannels);
        };
        forEachJaggedElementChannelCPU<scalar_t, 2>(int64_t(1), points, cb);
    }

    return outGrad.reshape(outShape);
}

template <torch::DeviceType DeviceTag>
torch::Tensor
dispatchSampleGridTrilinearWithGradBackward<DeviceTag>(const GridBatchImpl &batchHdl,
                                                       const JaggedTensor &points,
                                                       const torch::Tensor &data,
                                                       const torch::Tensor &gradOutFeatures,
                                                       const torch::Tensor &gradOutGradFeatures) {
    return AT_DISPATCH_V2(points.scalar_type(),
                          "SampleGridTrilinearWithGradBackward",
                          AT_WRAP([&] {
                              return SampleGridTrilinearWithGradBackward<DeviceTag, scalar_t>(
                                  batchHdl, points, data, gradOutFeatures, gradOutGradFeatures);
                          }),
                          AT_EXPAND(AT_FLOATING_TYPES),
                          c10::kHalf);
}

template torch::Tensor
dispatchSampleGridTrilinearWithGradBackward<torch::kCPU>(const GridBatchImpl &,
                                                         const JaggedTensor &,
                                                         const torch::Tensor &,
                                                         const torch::Tensor &,
                                                         const torch::Tensor &);

template torch::Tensor
dispatchSampleGridTrilinearWithGradBackward<torch::kCUDA>(const GridBatchImpl &,
                                                          const JaggedTensor &,
                                                          const torch::Tensor &,
                                                          const torch::Tensor &,
                                                          const torch::Tensor &);

template torch::Tensor
dispatchSampleGridTrilinearWithGradBackward<torch::kPrivateUse1>(const GridBatchImpl &,
                                                                 const JaggedTensor &,
                                                                 const torch::Tensor &,
                                                                 const torch::Tensor &,
                                                                 const torch::Tensor &);

} // namespace ops
} // namespace detail
} // namespace fvdb
