// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/SampleGridTrilinearWithGrad.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/ForEachCPU.h>
#include <fvdb/detail/utils/TrilinearInterpolationWithGradIterator.h>
#include <fvdb/detail/utils/cuda/ForEachCUDA.cuh>
#include <fvdb/detail/utils/cuda/ForEachPrivateUse1.cuh>

#include <ATen/OpMathType.h>
#include <c10/cuda/CUDAException.h>

#include <cstdint>

namespace fvdb {
namespace detail {
namespace ops {

template <typename ScalarType,
          template <typename T, int32_t D>
          typename JaggedAccessor,
          template <typename T, int32_t D>
          typename TensorAccessor>
__hostdev__ void
sampleTrilinearWithGradCallback(int32_t bidx,
                                int32_t eidx,
                                int32_t cidx,
                                JaggedAccessor<ScalarType, 2> points,
                                TensorAccessor<ScalarType, 2> gridData,
                                BatchGridAccessor batchAccessor,
                                TensorAccessor<ScalarType, 2> outFeatures,
                                TensorAccessor<ScalarType, 3> outGradFeatures) {
    using MathType = at::opmath_type<ScalarType>;

    const auto &pointsData = points.data();

    const nanovdb::OnIndexGrid *gpuGrid  = batchAccessor.grid(bidx);
    const VoxelCoordTransform &transform = batchAccessor.primalTransform(bidx);
    const int64_t baseOffset             = batchAccessor.voxelOffset(bidx);

    auto gridAcc = gpuGrid->tree().getAccessor();

    const nanovdb::math::Vec3<MathType> xyz =
        transform.apply(static_cast<MathType>(pointsData[eidx][0]),
                        static_cast<MathType>(pointsData[eidx][1]),
                        static_cast<MathType>(pointsData[eidx][2]));

    auto gradTransform = transform.template applyGrad<MathType>(xyz);

#pragma unroll
    for (auto it = TrilinearInterpolationWithGradIterator<MathType>(xyz); it.isValid(); ++it) {
        const nanovdb::math::Vec4<MathType> wXYZ = it->second;
        const nanovdb::Coord ijk                 = it->first;
        if (gridAcc.isActive(ijk)) {
            const int64_t indexIjk = gridAcc.getValue(ijk) - 1 + baseOffset;
            outFeatures[eidx][cidx] += wXYZ[0] * gridData[indexIjk][cidx];
#pragma unroll
            for (int dim = 0; dim < 3; ++dim) {
                outGradFeatures[eidx][cidx][dim] +=
                    wXYZ[dim + 1] * gridData[indexIjk][cidx] * gradTransform[dim];
            }
        }
    }
}

// Vectorized callback for float32 on GPU - processes 4 channels per thread using float4
template <template <typename T, int32_t D> typename JaggedAccessor,
          template <typename T, int32_t D>
          typename TensorAccessor>
__device__ void
sampleTrilinearWithGradCallbackVec4(int32_t bidx,
                                    int32_t eidx,
                                    int32_t cidx, // channel group index (each group = 4 channels)
                                    JaggedAccessor<float, 2> points,
                                    TensorAccessor<float, 2> gridData,
                                    BatchGridAccessor batchAccessor,
                                    TensorAccessor<float, 2> outFeatures,
                                    TensorAccessor<float, 3> outGradFeatures) {
    const int32_t cBase = cidx * 4;

    const auto &pointsData               = points.data();
    const nanovdb::OnIndexGrid *gpuGrid  = batchAccessor.grid(bidx);
    const VoxelCoordTransform &transform = batchAccessor.primalTransform(bidx);
    const int64_t baseOffset             = batchAccessor.voxelOffset(bidx);

    auto gridAcc = gpuGrid->tree().getAccessor();

    const nanovdb::math::Vec3<float> xyz =
        transform.apply(pointsData[eidx][0], pointsData[eidx][1], pointsData[eidx][2]);

    auto gradTransform = transform.template applyGrad<float>(xyz);

    // Accumulators for features and gradients (4 channels each)
    alignas(16) float accumFeat[4]  = {0.0f, 0.0f, 0.0f, 0.0f};
    alignas(16) float accumGradX[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    alignas(16) float accumGradY[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    alignas(16) float accumGradZ[4] = {0.0f, 0.0f, 0.0f, 0.0f};

#pragma unroll
    for (auto it = TrilinearInterpolationWithGradIterator<float>(xyz); it.isValid(); ++it) {
        const nanovdb::math::Vec4<float> wXYZ = it->second;
        const nanovdb::Coord ijk              = it->first;
        if (gridAcc.isActive(ijk)) {
            const int64_t indexIjk = gridAcc.getValue(ijk) - 1 + baseOffset;
            // Vectorized load
            auto gridVal = static_cast<const float *>(
                __builtin_assume_aligned(&gridData[indexIjk][cBase], 16));

            accumFeat[0] += wXYZ[0] * gridVal[0];
            accumFeat[1] += wXYZ[0] * gridVal[1];
            accumFeat[2] += wXYZ[0] * gridVal[2];
            accumFeat[3] += wXYZ[0] * gridVal[3];

            accumGradX[0] += wXYZ[1] * gridVal[0];
            accumGradX[1] += wXYZ[1] * gridVal[1];
            accumGradX[2] += wXYZ[1] * gridVal[2];
            accumGradX[3] += wXYZ[1] * gridVal[3];

            accumGradY[0] += wXYZ[2] * gridVal[0];
            accumGradY[1] += wXYZ[2] * gridVal[1];
            accumGradY[2] += wXYZ[2] * gridVal[2];
            accumGradY[3] += wXYZ[2] * gridVal[3];

            accumGradZ[0] += wXYZ[3] * gridVal[0];
            accumGradZ[1] += wXYZ[3] * gridVal[1];
            accumGradZ[2] += wXYZ[3] * gridVal[2];
            accumGradZ[3] += wXYZ[3] * gridVal[3];
        }
    }

    // Vectorized store for features
    auto outPtr = static_cast<float *>(__builtin_assume_aligned(&outFeatures[eidx][cBase], 16));
    outPtr[0]   = accumFeat[0];
    outPtr[1]   = accumFeat[1];
    outPtr[2]   = accumFeat[2];
    outPtr[3]   = accumFeat[3];

    // Store gradients (need to apply gradTransform and store to 3D tensor)
    // outGradFeatures has shape [M, C, 3], so we store each dimension separately
    outGradFeatures[eidx][cBase + 0][0] = accumGradX[0] * gradTransform[0];
    outGradFeatures[eidx][cBase + 0][1] = accumGradY[0] * gradTransform[1];
    outGradFeatures[eidx][cBase + 0][2] = accumGradZ[0] * gradTransform[2];

    outGradFeatures[eidx][cBase + 1][0] = accumGradX[1] * gradTransform[0];
    outGradFeatures[eidx][cBase + 1][1] = accumGradY[1] * gradTransform[1];
    outGradFeatures[eidx][cBase + 1][2] = accumGradZ[1] * gradTransform[2];

    outGradFeatures[eidx][cBase + 2][0] = accumGradX[2] * gradTransform[0];
    outGradFeatures[eidx][cBase + 2][1] = accumGradY[2] * gradTransform[1];
    outGradFeatures[eidx][cBase + 2][2] = accumGradZ[2] * gradTransform[2];

    outGradFeatures[eidx][cBase + 3][0] = accumGradX[3] * gradTransform[0];
    outGradFeatures[eidx][cBase + 3][1] = accumGradY[3] * gradTransform[1];
    outGradFeatures[eidx][cBase + 3][2] = accumGradZ[3] * gradTransform[2];
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
        auto dispatchForEach = [&](auto numCh, const auto &cb) {
            if constexpr (DeviceTag == torch::kCUDA) {
                forEachJaggedElementChannelCUDA<scalar_t, 2>(DEFAULT_BLOCK_DIM, numCh, points, cb);
            } else {
                forEachJaggedElementChannelPrivateUse1<scalar_t, 2>(numCh, points, cb);
            }
        };

        // Use vectorized float4 loads for float32 when channels is a multiple of 4
        // and base pointers are 16-byte aligned
        if constexpr (std::is_same_v<scalar_t, float>) {
            if (numChannels >= 4 && numChannels % 4 == 0 &&
                reinterpret_cast<uintptr_t>(gridDataReshape.data_ptr<float>()) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(outFeatures.data_ptr<float>()) % 16 == 0) {
                const auto numChannelGroups = (numChannels + 3) / 4;
                auto cb                     = [=] __device__(int32_t bidx,
                                         int32_t eidx,
                                         int32_t cidx,
                                         JaggedRAcc32<float, 2> pts) {
                    sampleTrilinearWithGradCallbackVec4<JaggedRAcc32, TorchRAcc32>(
                        bidx,
                        eidx,
                        cidx,
                        pts,
                        gridDataAcc,
                        batchAcc,
                        outFeaturesAcc,
                        outGradFeaturesAcc);
                };
                dispatchForEach(numChannelGroups, cb);
            } else {
                auto cb = [=] __device__(int32_t bidx,
                                         int32_t eidx,
                                         int32_t cidx,
                                         JaggedRAcc32<float, 2> pts) {
                    sampleTrilinearWithGradCallback<float, JaggedRAcc32, TorchRAcc32>(
                        bidx,
                        eidx,
                        cidx,
                        pts,
                        gridDataAcc,
                        batchAcc,
                        outFeaturesAcc,
                        outGradFeaturesAcc);
                };
                dispatchForEach(numChannels, cb);
            }
        } else {
            auto cb = [=] __device__(int32_t bidx,
                                     int32_t eidx,
                                     int32_t cidx,
                                     JaggedRAcc32<scalar_t, 2> pts) {
                sampleTrilinearWithGradCallback<scalar_t, JaggedRAcc32, TorchRAcc32>(
                    bidx,
                    eidx,
                    cidx,
                    pts,
                    gridDataAcc,
                    batchAcc,
                    outFeaturesAcc,
                    outGradFeaturesAcc);
            };
            dispatchForEach(numChannels, cb);
        }
    } else {
        auto cb = [=](int32_t bidx, int32_t eidx, int32_t cidx, JaggedAcc<scalar_t, 2> pts) {
            sampleTrilinearWithGradCallback<scalar_t, JaggedAcc, TorchAcc>(
                bidx, eidx, cidx, pts, gridDataAcc, batchAcc, outFeaturesAcc, outGradFeaturesAcc);
        };
        forEachJaggedElementChannelCPU<scalar_t, 2>(numChannels, points, cb);
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
