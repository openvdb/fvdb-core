// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/ForEachCPU.h>
#include <fvdb/detail/utils/TrilinearInterpolationIterator.h>
#include <fvdb/detail/utils/cuda/ForEachCUDA.cuh>
#include <fvdb/detail/utils/cuda/ForEachPrivateUse1.cuh>

#include <ATen/OpMathType.h>
#include <c10/cuda/CUDAException.h>

#include <iostream>

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
                        BatchGridAccessor<nanovdb::ValueOnIndex> batchAccessor,
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

template <c10::DeviceType DeviceTag, typename scalar_t>
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

    auto batchAcc       = gridBatchAccessor<DeviceTag, nanovdb::ValueOnIndex>(batchHdl);
    auto gridDataAcc    = tensorAccessor<DeviceTag, scalar_t, 2>(gridDataReshape);
    auto outFeaturesAcc = tensorAccessor<DeviceTag, scalar_t, 2>(outFeatures);
    if constexpr (DeviceTag == torch::kCUDA) {
        auto cb = [=]
            __device__(int32_t bidx, int32_t eidx, int32_t cidx, JaggedRAcc32<scalar_t, 2> pts) {
                sampleTrilinearCallback<scalar_t, JaggedRAcc32, TorchRAcc32>(
                    bidx, eidx, cidx, pts, gridDataAcc, batchAcc, outFeaturesAcc);
            };
        forEachJaggedElementChannelCUDA<scalar_t, 2>(DEFAULT_BLOCK_DIM, gridData.size(1), points, cb);
    } else if constexpr (DeviceTag == torch::kPrivateUse1) {
        auto cb = [=]
            __device__(int32_t bidx, int32_t eidx, int32_t cidx, JaggedRAcc32<scalar_t, 2> pts) {
                sampleTrilinearCallback<scalar_t, JaggedRAcc32, TorchRAcc32>(
                    bidx, eidx, cidx, pts, gridDataAcc, batchAcc, outFeaturesAcc);
            };
        forEachJaggedElementChannelPrivateUse1<scalar_t, 2>(gridData.size(1), points, cb);
    } else {
        auto cb = [=](int32_t bidx, int32_t eidx, int32_t cidx, JaggedAcc<scalar_t, 2> pts) {
            sampleTrilinearCallback<scalar_t, JaggedAcc, TorchAcc>(
                bidx, eidx, cidx, pts, gridDataAcc, batchAcc, outFeaturesAcc);
        };
        forEachJaggedElementChannelCPU<scalar_t, 2>(gridData.size(1), points, cb);
    }

    return {outFeatures.reshape(outShape)};
}

template <c10::DeviceType DeviceTag>
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
