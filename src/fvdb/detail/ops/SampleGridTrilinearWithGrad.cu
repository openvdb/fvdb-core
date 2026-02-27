// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/SampleGridTrilinearWithGrad.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/ForEachCPU.h>
#include <fvdb/detail/utils/TrilinearStencil.h>
#include <fvdb/detail/utils/cuda/ForEachCUDA.cuh>
#include <fvdb/detail/utils/cuda/ForEachPrivateUse1.cuh>

#include <ATen/OpMathType.h>
#include <c10/cuda/CUDAException.h>

#include <cstdint>

namespace fvdb {
namespace detail {
namespace ops {

// One-thread-per-point scalar callback for sample_trilinear_with_grad.
// Computes both interpolated features and spatial gradients.
template <typename ScalarType,
          template <typename T, int32_t D>
          typename JaggedAccessor,
          template <typename T, int32_t D>
          typename TensorAccessor>
__hostdev__ void
sampleTrilinearWithGradStencilCallback(int32_t bidx,
                                       int32_t eidx,
                                       int32_t /*cidx*/,
                                       JaggedAccessor<ScalarType, 2> points,
                                       TensorAccessor<ScalarType, 2> gridData,
                                       BatchGridAccessor batchAccessor,
                                       TensorAccessor<ScalarType, 2> outFeatures,
                                       TensorAccessor<ScalarType, 3> outGradFeatures,
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

    auto gradTransform = transform.template applyGrad<MathType>(xyz);

    int64_t indices[8] = {};
    MathType weights[8];
    MathType gradWeights[8][3];
    const uint8_t activeMask =
        resolveTrilinearStencilWithGrad(xyz, gridAcc, baseOffset, indices, weights, gradWeights);

    if (activeMask == 0)
        return;

    for (int64_t c = 0; c < numChannels; ++c) {
        MathType accumFeat    = MathType(0);
        MathType accumGrad[3] = {MathType(0), MathType(0), MathType(0)};
#pragma unroll
        for (int corner = 0; corner < 8; ++corner) {
            const MathType val = static_cast<MathType>(gridData[indices[corner]][c]);
            accumFeat += weights[corner] * val;
            accumGrad[0] += gradWeights[corner][0] * val;
            accumGrad[1] += gradWeights[corner][1] * val;
            accumGrad[2] += gradWeights[corner][2] * val;
        }
        outFeatures[eidx][c]        = static_cast<ScalarType>(accumFeat);
        outGradFeatures[eidx][c][0] = static_cast<ScalarType>(accumGrad[0] * gradTransform[0]);
        outGradFeatures[eidx][c][1] = static_cast<ScalarType>(accumGrad[1] * gradTransform[1]);
        outGradFeatures[eidx][c][2] = static_cast<ScalarType>(accumGrad[2] * gradTransform[2]);
    }
}

// One-thread-per-point Vec4 callback for sample_trilinear_with_grad. GPU only.
// Processes channels in float4 groups for features; gradient stores are per-channel
// since outGradFeatures has shape [M, C, 3].
template <template <typename T, int32_t D> typename JaggedAccessor,
          template <typename T, int32_t D>
          typename TensorAccessor>
__device__ void
sampleTrilinearWithGradStencilCallbackVec4(int32_t bidx,
                                           int32_t eidx,
                                           int32_t /*cidx*/,
                                           JaggedAccessor<float, 2> points,
                                           TensorAccessor<float, 2> gridData,
                                           BatchGridAccessor batchAccessor,
                                           TensorAccessor<float, 2> outFeatures,
                                           TensorAccessor<float, 3> outGradFeatures,
                                           int64_t numChannels) {
    const auto &pointsData               = points.data();
    const nanovdb::OnIndexGrid *grid     = batchAccessor.grid(bidx);
    const VoxelCoordTransform &transform = batchAccessor.primalTransform(bidx);
    const int64_t baseOffset             = batchAccessor.voxelOffset(bidx);
    auto gridAcc                         = grid->tree().getAccessor();

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
    for (int64_t g = 0; g < numGroups; ++g) {
        const int64_t cBase   = g * 4;
        float4 accumFeat      = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float accumGrad[3][4] = {};
#pragma unroll
        for (int corner = 0; corner < 8; ++corner) {
            const float wt    = weights[corner];
            const float gw[3] = {
                gradWeights[corner][0], gradWeights[corner][1], gradWeights[corner][2]};
            const float4 val = *reinterpret_cast<const float4 *>(&gridData[indices[corner]][cBase]);
            accumFeat.x += wt * val.x;
            accumFeat.y += wt * val.y;
            accumFeat.z += wt * val.z;
            accumFeat.w += wt * val.w;
#pragma unroll
            for (int d = 0; d < 3; ++d) {
                accumGrad[d][0] += gw[d] * val.x;
                accumGrad[d][1] += gw[d] * val.y;
                accumGrad[d][2] += gw[d] * val.z;
                accumGrad[d][3] += gw[d] * val.w;
            }
        }
        *reinterpret_cast<float4 *>(&outFeatures[eidx][cBase]) = accumFeat;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            outGradFeatures[eidx][cBase + i][0] = accumGrad[0][i] * gx;
            outGradFeatures[eidx][cBase + i][1] = accumGrad[1][i] * gy;
            outGradFeatures[eidx][cBase + i][2] = accumGrad[2][i] * gz;
        }
    }
}

template <torch::DeviceType DeviceTag, typename scalar_t>
std::vector<torch::Tensor>
SampleGridTrilinearWithGrad(const GridBatchImpl &batchHdl,
                            const JaggedTensor &points,
                            const torch::Tensor &gridData) {
    auto opts = torch::TensorOptions()
                    .dtype(gridData.dtype())
                    .device(gridData.device())
                    .requires_grad(gridData.requires_grad());
    torch::Tensor gridDataReshape = featureCoalescedView(gridData).contiguous();     // [B*N, -1]
    torch::Tensor outFeatures =
        torch::zeros({points.rsize(0), gridDataReshape.size(1)}, opts).contiguous(); // [B*M, -1]
    torch::Tensor outGradFeatures =
        torch::zeros({points.rsize(0), gridDataReshape.size(1), 3}, opts);           // [B*M, -1, 3]
    std::vector<int64_t> outShape     = spliceShape({points.rsize(0)}, gridData, 1); // [B*M, *]
    std::vector<int64_t> outGradShape = outShape;
    outGradShape.push_back(3);

    auto batchAcc             = gridBatchAccessor<DeviceTag>(batchHdl);
    auto gridDataAcc          = tensorAccessor<DeviceTag, scalar_t, 2>(gridDataReshape);
    auto outFeaturesAcc       = tensorAccessor<DeviceTag, scalar_t, 2>(outFeatures);
    auto outGradFeaturesAcc   = tensorAccessor<DeviceTag, scalar_t, 3>(outGradFeatures);
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
                    sampleTrilinearWithGradStencilCallbackVec4<JaggedRAcc32, TorchRAcc32>(
                        bidx,
                        eidx,
                        cidx,
                        pts,
                        gridDataAcc,
                        batchAcc,
                        outFeaturesAcc,
                        outGradFeaturesAcc,
                        numChannels);
                };
                dispatchForEach(cb);
            } else {
                auto cb = [=] __device__(int32_t bidx,
                                         int32_t eidx,
                                         int32_t cidx,
                                         JaggedRAcc32<float, 2> pts) {
                    sampleTrilinearWithGradStencilCallback<float, JaggedRAcc32, TorchRAcc32>(
                        bidx,
                        eidx,
                        cidx,
                        pts,
                        gridDataAcc,
                        batchAcc,
                        outFeaturesAcc,
                        outGradFeaturesAcc,
                        numChannels);
                };
                dispatchForEach(cb);
            }
        } else {
            auto cb = [=] __device__(int32_t bidx,
                                     int32_t eidx,
                                     int32_t cidx,
                                     JaggedRAcc32<scalar_t, 2> pts) {
                sampleTrilinearWithGradStencilCallback<scalar_t, JaggedRAcc32, TorchRAcc32>(
                    bidx,
                    eidx,
                    cidx,
                    pts,
                    gridDataAcc,
                    batchAcc,
                    outFeaturesAcc,
                    outGradFeaturesAcc,
                    numChannels);
            };
            dispatchForEach(cb);
        }
    } else {
        auto cb = [=](int32_t bidx, int32_t eidx, int32_t cidx, JaggedAcc<scalar_t, 2> pts) {
            sampleTrilinearWithGradStencilCallback<scalar_t, JaggedAcc, TorchAcc>(
                bidx,
                eidx,
                cidx,
                pts,
                gridDataAcc,
                batchAcc,
                outFeaturesAcc,
                outGradFeaturesAcc,
                numChannels);
        };
        forEachJaggedElementChannelCPU<scalar_t, 2>(int64_t(1), points, cb);
    }

    return {outFeatures.reshape(outShape), outGradFeatures.reshape(outGradShape)};
}

template <torch::DeviceType DeviceTag>
std::vector<torch::Tensor>
dispatchSampleGridTrilinearWithGrad(const GridBatchImpl &batchHdl,
                                    const JaggedTensor &points,
                                    const torch::Tensor &gridData) {
    return AT_DISPATCH_V2(points.scalar_type(),
                          "SampleGridTrilinearWithGrad",
                          AT_WRAP([&] {
                              return SampleGridTrilinearWithGrad<DeviceTag, scalar_t>(
                                  batchHdl, points, gridData);
                          }),
                          AT_EXPAND(AT_FLOATING_TYPES),
                          c10::kHalf);
}

template std::vector<torch::Tensor> dispatchSampleGridTrilinearWithGrad<torch::kCPU>(
    const GridBatchImpl &, const JaggedTensor &, const torch::Tensor &);
template std::vector<torch::Tensor> dispatchSampleGridTrilinearWithGrad<torch::kCUDA>(
    const GridBatchImpl &, const JaggedTensor &, const torch::Tensor &);
template std::vector<torch::Tensor> dispatchSampleGridTrilinearWithGrad<torch::kPrivateUse1>(
    const GridBatchImpl &, const JaggedTensor &, const torch::Tensor &);

} // namespace ops
} // namespace detail
} // namespace fvdb
