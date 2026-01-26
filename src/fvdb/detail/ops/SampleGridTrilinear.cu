// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/SampleGridTrilinear.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/ForEachCPU.h>
#include <fvdb/detail/utils/TrilinearInterpolationIterator.h>
#include <fvdb/detail/utils/cuda/ForEachCUDA.cuh>
#include <fvdb/detail/utils/cuda/ForEachPrivateUse1.cuh>

#include <ATen/OpMathType.h>
#include <c10/cuda/CUDAException.h>

namespace fvdb {
namespace detail {
namespace ops {

template <typename ScalarType,
          template <typename T, int32_t D>
          typename JaggedAccessor,
          template <typename T, int32_t D>
          typename TensorAccessor>
__hostdev__ void
sampleTrilinearCallback(int32_t bidx,
                        int32_t eidx,
                        int32_t cidx,
                        JaggedAccessor<ScalarType, 2> points,
                        TensorAccessor<ScalarType, 2> gridData,
                        BatchGridAccessor batchAccessor,
                        TensorAccessor<ScalarType, 2> outFeatures) {
    using MathType = at::opmath_type<ScalarType>;

    const auto &pointsData               = points.data();
    const nanovdb::OnIndexGrid *gpuGrid  = batchAccessor.grid(bidx);
    const VoxelCoordTransform &transform = batchAccessor.primalTransform(bidx);
    const int64_t baseOffset             = batchAccessor.voxelOffset(bidx);

    auto gridAcc = gpuGrid->tree().getAccessor();

    const nanovdb::math::Vec3<MathType> xyz =
        transform.apply(static_cast<MathType>(pointsData[eidx][0]),
                        static_cast<MathType>(pointsData[eidx][1]),
                        static_cast<MathType>(pointsData[eidx][2]));

#pragma unroll
    for (auto it = TrilinearInterpolationIterator<MathType>(xyz); it.isValid(); ++it) {
        const MathType wTrilinear = it->second;
        const nanovdb::Coord ijk  = it->first;
        if (gridAcc.isActive(ijk)) {
            const int64_t indexIjk = gridAcc.getValue(ijk) - 1 + baseOffset;
            outFeatures[eidx][cidx] += wTrilinear * gridData[indexIjk][cidx];
        }
    }
}

// Vectorized callback for float32 - processes 4 channels per thread using float4
// cidx here represents the channel GROUP index (0, 1, 2, ...), each group has 4 channels
template <template <typename T, int32_t D> typename JaggedAccessor,
          template <typename T, int32_t D>
          typename TensorAccessor>
__device__ void
sampleTrilinearCallbackVec4(int32_t bidx,
                            int32_t eidx,
                            int32_t cidx, // channel group index (each group = 4 channels)
                            JaggedAccessor<float, 2> points,
                            TensorAccessor<float, 2> gridData,
                            BatchGridAccessor batchAccessor,
                            TensorAccessor<float, 2> outFeatures,
                            int32_t numChannels) {
    const int32_t cBase = cidx * 4; // starting channel for this group

    // Handle boundary: if this group goes past numChannels, fall back to scalar
    if (cBase + 4 > numChannels) {
        // Process remaining channels individually
        for (int32_t c = cBase; c < numChannels; ++c) {
            sampleTrilinearCallback<float, JaggedAccessor, TensorAccessor>(
                bidx, eidx, c, points, gridData, batchAccessor, outFeatures);
        }
        return;
    }

    const auto &pointsData               = points.data();
    const nanovdb::OnIndexGrid *gpuGrid  = batchAccessor.grid(bidx);
    const VoxelCoordTransform &transform = batchAccessor.primalTransform(bidx);
    const int64_t baseOffset             = batchAccessor.voxelOffset(bidx);

    auto gridAcc = gpuGrid->tree().getAccessor();

    const nanovdb::math::Vec3<float> xyz =
        transform.apply(pointsData[eidx][0], pointsData[eidx][1], pointsData[eidx][2]);

    // Accumulate in float4
    float4 accum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

#pragma unroll
    for (auto it = TrilinearInterpolationIterator<float>(xyz); it.isValid(); ++it) {
        const float wTrilinear   = it->second;
        const nanovdb::Coord ijk = it->first;
        if (gridAcc.isActive(ijk)) {
            const int64_t indexIjk = gridAcc.getValue(ijk) - 1 + baseOffset;
            // Vectorized load: load 4 consecutive floats
            const float4 gridVal = *reinterpret_cast<const float4 *>(&gridData[indexIjk][cBase]);
            accum.x += wTrilinear * gridVal.x;
            accum.y += wTrilinear * gridVal.y;
            accum.z += wTrilinear * gridVal.z;
            accum.w += wTrilinear * gridVal.w;
        }
    }

    // Vectorized store: write 4 consecutive floats
    *reinterpret_cast<float4 *>(&outFeatures[eidx][cBase]) = accum;
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
    torch::Tensor gridDataReshape = featureCoalescedView(gridData);     // [B*N, -1]
    torch::Tensor outFeatures =
        torch::zeros({points.rsize(0), gridDataReshape.size(1)}, opts); // [B*M, -1]
    auto outShape = spliceShape({points.rsize(0)}, gridData, 1);        // [B*M, *]

    auto batchAcc             = gridBatchAccessor<DeviceTag>(batchHdl);
    auto gridDataAcc          = tensorAccessor<DeviceTag, scalar_t, 2>(gridDataReshape);
    auto outFeaturesAcc       = tensorAccessor<DeviceTag, scalar_t, 2>(outFeatures);
    const int64_t numChannels = gridDataReshape.size(1);

    if constexpr (DeviceTag == torch::kCUDA || DeviceTag == torch::kPrivateUse1) {
        // Helper to dispatch to correct forEach based on device
        auto dispatchForEach = [&](auto numCh, const auto &cb) {
            if constexpr (DeviceTag == torch::kCUDA) {
                forEachJaggedElementChannelCUDA<scalar_t, 2>(DEFAULT_BLOCK_DIM, numCh, points, cb);
            } else {
                forEachJaggedElementChannelPrivateUse1<scalar_t, 2>(numCh, points, cb);
            }
        };

        // Use vectorized float4 loads for float32 when channels is a multiple of 4
        // (alignment requirement: numChannels % 4 == 0 ensures all rows are 16-byte aligned)
        if constexpr (std::is_same_v<scalar_t, float>) {
            if (numChannels >= 4 && numChannels % 4 == 0) {
                const auto numChannelGroups = (numChannels + 3) / 4;
                auto cb                     = [=] __device__(int32_t bidx,
                                         int32_t eidx,
                                         int32_t cidx,
                                         JaggedRAcc32<float, 2> pts) {
                    sampleTrilinearCallbackVec4<JaggedRAcc32, TorchRAcc32>(
                        bidx,
                        eidx,
                        cidx,
                        pts,
                        gridDataAcc,
                        batchAcc,
                        outFeaturesAcc,
                        static_cast<int32_t>(numChannels));
                };
                dispatchForEach(numChannelGroups, cb);
            } else {
                auto cb = [=] __device__(int32_t bidx,
                                         int32_t eidx,
                                         int32_t cidx,
                                         JaggedRAcc32<float, 2> pts) {
                    sampleTrilinearCallback<float, JaggedRAcc32, TorchRAcc32>(
                        bidx, eidx, cidx, pts, gridDataAcc, batchAcc, outFeaturesAcc);
                };
                dispatchForEach(numChannels, cb);
            }
        } else {
            auto cb = [=] __device__(int32_t bidx,
                                     int32_t eidx,
                                     int32_t cidx,
                                     JaggedRAcc32<scalar_t, 2> pts) {
                sampleTrilinearCallback<scalar_t, JaggedRAcc32, TorchRAcc32>(
                    bidx, eidx, cidx, pts, gridDataAcc, batchAcc, outFeaturesAcc);
            };
            dispatchForEach(numChannels, cb);
        }
    } else {
        auto cb = [=](int32_t bidx, int32_t eidx, int32_t cidx, JaggedAcc<scalar_t, 2> pts) {
            sampleTrilinearCallback<scalar_t, JaggedAcc, TorchAcc>(
                bidx, eidx, cidx, pts, gridDataAcc, batchAcc, outFeaturesAcc);
        };
        forEachJaggedElementChannelCPU<scalar_t, 2>(numChannels, points, cb);
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
