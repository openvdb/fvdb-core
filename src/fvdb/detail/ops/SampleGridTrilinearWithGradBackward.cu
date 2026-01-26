// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/SampleGridTrilinearWithGradBackward.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/ForEachCPU.h>
#include <fvdb/detail/utils/TrilinearInterpolationWithGradIterator.h>
#include <fvdb/detail/utils/cuda/ForEachCUDA.cuh>
#include <fvdb/detail/utils/cuda/ForEachPrivateUse1.cuh>

#include <ATen/OpMathType.h>
#include <THC/THCAtomics.cuh>
#include <c10/cuda/CUDAException.h>

#include <cstdint>

namespace fvdb {
namespace detail {
namespace ops {

template <torch::DeviceType DeviceTag,
          typename ScalarType,
          template <typename T, int32_t D>
          typename JaggedAccessor,
          template <typename T, int32_t D>
          typename TensorAccessor>
__hostdev__ void
sampleTrilinearWithGradBackwardCallback(int32_t bidx,
                                        int32_t eidx,
                                        int32_t cidx,
                                        JaggedAccessor<ScalarType, 2> points,
                                        TensorAccessor<ScalarType, 2> gradOutFeatures,
                                        TensorAccessor<ScalarType, 3> gradOutGradFeatures,
                                        BatchGridAccessor batchAccessor,
                                        TensorAccessor<ScalarType, 2> outGridData) {
    using MathType = at::opmath_type<ScalarType>;

    const auto &pointsData = points.data();

    const nanovdb::OnIndexGrid *gpuGrid = batchAccessor.grid(bidx);
    auto transform                      = batchAccessor.primalTransform(bidx);
    const int64_t baseOffset            = batchAccessor.voxelOffset(bidx);

    auto gridAcc = gpuGrid->getAccessor();

    const nanovdb::math::Vec3<MathType> xyz =
        transform.apply(static_cast<MathType>(pointsData[eidx][0]),
                        static_cast<MathType>(pointsData[eidx][1]),
                        static_cast<MathType>(pointsData[eidx][2]));
    auto gradTransform = transform.template applyGrad<MathType>(xyz);

#pragma unroll
    for (auto it = TrilinearInterpolationWithGradIterator<MathType>(xyz); it.isValid(); ++it) {
        if (gridAcc.isActive(it->first)) {
            const int64_t indexIjk = gridAcc.getValue(it->first) - 1 + baseOffset;
            MathType addValue      = it->second[0] * gradOutFeatures[eidx][cidx];
#pragma unroll
            for (int dim = 0; dim < 3; ++dim) {
                addValue +=
                    it->second[dim + 1] * gradOutGradFeatures[eidx][cidx][dim] * gradTransform[dim];
            }
            if constexpr (DeviceTag == torch::kCUDA) {
                gpuAtomicAddNoReturn(&outGridData[indexIjk][cidx],
                                     static_cast<ScalarType>(addValue));
            } else {
                outGridData[indexIjk][cidx] += static_cast<ScalarType>(addValue);
            }
        }
    }
}

// Vectorized callback for float32 - processes 4 channels per thread
template <torch::DeviceType DeviceTag,
          template <typename T, int32_t D>
          typename JaggedAccessor,
          template <typename T, int32_t D>
          typename TensorAccessor>
__device__ void
sampleTrilinearWithGradBackwardCallbackVec4(int32_t bidx,
                                            int32_t eidx,
                                            int32_t cidx, // channel group index
                                            JaggedAccessor<float, 2> points,
                                            TensorAccessor<float, 2> gradOutFeatures,
                                            TensorAccessor<float, 3> gradOutGradFeatures,
                                            BatchGridAccessor batchAccessor,
                                            TensorAccessor<float, 2> outGridData) {
    const int32_t cBase = cidx * 4;

    const auto &pointsData = points.data();

    const nanovdb::OnIndexGrid *gpuGrid = batchAccessor.grid(bidx);
    auto transform                      = batchAccessor.primalTransform(bidx);
    const int64_t baseOffset            = batchAccessor.voxelOffset(bidx);

    auto gridAcc = gpuGrid->getAccessor();

    const nanovdb::math::Vec3<float> xyz =
        transform.apply(pointsData[eidx][0], pointsData[eidx][1], pointsData[eidx][2]);
    auto gradTransform = transform.template applyGrad<float>(xyz);

    // Vectorized load of gradOutFeatures
    auto gradFeat =
        static_cast<const float *>(__builtin_assume_aligned(&gradOutFeatures[eidx][cBase], 16));

    // Load gradOutGradFeatures for 4 channels (each has 3 components)
    const float gradGrad0_x = gradOutGradFeatures[eidx][cBase + 0][0];
    const float gradGrad0_y = gradOutGradFeatures[eidx][cBase + 0][1];
    const float gradGrad0_z = gradOutGradFeatures[eidx][cBase + 0][2];
    const float gradGrad1_x = gradOutGradFeatures[eidx][cBase + 1][0];
    const float gradGrad1_y = gradOutGradFeatures[eidx][cBase + 1][1];
    const float gradGrad1_z = gradOutGradFeatures[eidx][cBase + 1][2];
    const float gradGrad2_x = gradOutGradFeatures[eidx][cBase + 2][0];
    const float gradGrad2_y = gradOutGradFeatures[eidx][cBase + 2][1];
    const float gradGrad2_z = gradOutGradFeatures[eidx][cBase + 2][2];
    const float gradGrad3_x = gradOutGradFeatures[eidx][cBase + 3][0];
    const float gradGrad3_y = gradOutGradFeatures[eidx][cBase + 3][1];
    const float gradGrad3_z = gradOutGradFeatures[eidx][cBase + 3][2];

#pragma unroll
    for (auto it = TrilinearInterpolationWithGradIterator<float>(xyz); it.isValid(); ++it) {
        if (gridAcc.isActive(it->first)) {
            const int64_t indexIjk                 = gridAcc.getValue(it->first) - 1 + baseOffset;
            const nanovdb::math::Vec4<float> &wXYZ = it->second;

            // Compute addValue for each of the 4 channels
            float addValue0 = wXYZ[0] * gradFeat[0] + wXYZ[1] * gradGrad0_x * gradTransform[0] +
                              wXYZ[2] * gradGrad0_y * gradTransform[1] +
                              wXYZ[3] * gradGrad0_z * gradTransform[2];

            float addValue1 = wXYZ[0] * gradFeat[1] + wXYZ[1] * gradGrad1_x * gradTransform[0] +
                              wXYZ[2] * gradGrad1_y * gradTransform[1] +
                              wXYZ[3] * gradGrad1_z * gradTransform[2];

            float addValue2 = wXYZ[0] * gradFeat[2] + wXYZ[1] * gradGrad2_x * gradTransform[0] +
                              wXYZ[2] * gradGrad2_y * gradTransform[1] +
                              wXYZ[3] * gradGrad2_z * gradTransform[2];

            float addValue3 = wXYZ[0] * gradFeat[3] + wXYZ[1] * gradGrad3_x * gradTransform[0] +
                              wXYZ[2] * gradGrad3_y * gradTransform[1] +
                              wXYZ[3] * gradGrad3_z * gradTransform[2];

            // Scalar atomic writes
            if constexpr (DeviceTag == torch::kCUDA) {
                gpuAtomicAddNoReturn(&outGridData[indexIjk][cBase + 0], addValue0);
                gpuAtomicAddNoReturn(&outGridData[indexIjk][cBase + 1], addValue1);
                gpuAtomicAddNoReturn(&outGridData[indexIjk][cBase + 2], addValue2);
                gpuAtomicAddNoReturn(&outGridData[indexIjk][cBase + 3], addValue3);
            } else if constexpr (DeviceTag == torch::kPrivateUse1) {
                atomicAdd_system(&outGridData[indexIjk][cBase + 0], addValue0);
                atomicAdd_system(&outGridData[indexIjk][cBase + 1], addValue1);
                atomicAdd_system(&outGridData[indexIjk][cBase + 2], addValue2);
                atomicAdd_system(&outGridData[indexIjk][cBase + 3], addValue3);
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
                reinterpret_cast<uintptr_t>(gradOutFeaturesContig.data_ptr<float>()) % 16 == 0) {
                const auto numChannelGroups = (numChannels + 3) / 4;
                auto cb                     = [=] __device__(int32_t bidx,
                                         int32_t eidx,
                                         int32_t cidx,
                                         JaggedRAcc32<float, 2> pts) {
                    sampleTrilinearWithGradBackwardCallbackVec4<DeviceTag,
                                                                                    JaggedRAcc32,
                                                                                    TorchRAcc32>(bidx,
                                                                             eidx,
                                                                             cidx,
                                                                             pts,
                                                                             gradOutFeaturesAcc,
                                                                             gradOutGradFeaturesAcc,
                                                                             batchAcc,
                                                                             outGradAcc);
                };
                dispatchForEach(numChannelGroups, cb);
            } else {
                auto cb = [=] __device__(int32_t bidx,
                                         int32_t eidx,
                                         int32_t cidx,
                                         JaggedRAcc32<float, 2> pts) {
                    sampleTrilinearWithGradBackwardCallback<DeviceTag,
                                                            float,
                                                            JaggedRAcc32,
                                                            TorchRAcc32>(bidx,
                                                                         eidx,
                                                                         cidx,
                                                                         pts,
                                                                         gradOutFeaturesAcc,
                                                                         gradOutGradFeaturesAcc,
                                                                         batchAcc,
                                                                         outGradAcc);
                };
                dispatchForEach(numChannels, cb);
            }
        } else {
            auto cb = [=] __device__(int32_t bidx,
                                     int32_t eidx,
                                     int32_t cidx,
                                     JaggedRAcc32<scalar_t, 2> pts) {
                sampleTrilinearWithGradBackwardCallback<DeviceTag,
                                                        scalar_t,
                                                        JaggedRAcc32,
                                                        TorchRAcc32>(bidx,
                                                                     eidx,
                                                                     cidx,
                                                                     pts,
                                                                     gradOutFeaturesAcc,
                                                                     gradOutGradFeaturesAcc,
                                                                     batchAcc,
                                                                     outGradAcc);
            };
            dispatchForEach(numChannels, cb);
        }
    } else {
        auto cb = [=](int32_t bidx, int32_t eidx, int32_t cidx, JaggedAcc<scalar_t, 2> pts) {
            sampleTrilinearWithGradBackwardCallback<DeviceTag, scalar_t, JaggedAcc, TorchAcc>(
                bidx,
                eidx,
                cidx,
                pts,
                gradOutFeaturesAcc,
                gradOutGradFeaturesAcc,
                batchAcc,
                outGradAcc);
        };
        forEachJaggedElementChannelCPU<scalar_t, 2>(numChannels, points, cb);
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
