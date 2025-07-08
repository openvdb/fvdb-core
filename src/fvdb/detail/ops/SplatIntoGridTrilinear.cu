// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/ForEachCPU.h>
#include <fvdb/detail/utils/TrilinearInterpolationIterator.h>
#include <fvdb/detail/utils/cuda/ForEachCUDA.cuh>
#include <fvdb/detail/utils/cuda/ForEachPrivateUse1.cuh>

#include <THC/THCAtomics.cuh>
#include <c10/cuda/CUDAException.h>

namespace fvdb {
namespace detail {
namespace ops {

template <c10::DeviceType DeviceTag,
          typename ScalarType,
          template <typename T, int32_t D>
          typename JaggedAccessor,
          template <typename T, int32_t D>
          typename TensorAccessor>
__hostdev__ void
splatIntoGridTrilinearCallback(int32_t bidx,
                               int32_t eidx,
                               int32_t cidx,
                               JaggedAccessor<ScalarType, 2> points,
                               TensorAccessor<ScalarType, 2> pointsData,
                               BatchGridAccessor batchAccessor,
                               TensorAccessor<at::opmath_type<ScalarType>, 2> outGridData) {
    using MathType = at::opmath_type<ScalarType>;

    const auto &pointCoordData = points.data();

    const nanovdb::OnIndexGrid *gpuGrid  = batchAccessor.grid(bidx);
    auto gridAcc                         = gpuGrid->getAccessor();
    const int64_t baseOffset             = batchAccessor.voxelOffset(bidx);
    const VoxelCoordTransform &transform = batchAccessor.primalTransform(bidx);

    const nanovdb::math::Vec3<MathType> xyz =
        transform.apply(static_cast<MathType>(pointCoordData[eidx][0]),
                        static_cast<MathType>(pointCoordData[eidx][1]),
                        static_cast<MathType>(pointCoordData[eidx][2]));

#pragma unroll
    for (auto it = TrilinearInterpolationIterator<MathType>(xyz); it.isValid(); ++it) {
        if (gridAcc.isActive(it->first)) {
            const int64_t indexIjk  = gridAcc.getValue(it->first) - 1 + baseOffset;
            const MathType addValue = it->second * static_cast<MathType>(pointsData[eidx][cidx]);
            if constexpr (DeviceTag == torch::kCUDA) {
                gpuAtomicAddNoReturn(&outGridData[indexIjk][cidx], addValue);
            } else if constexpr (DeviceTag == torch::kPrivateUse1) {
                atomicAdd_system(&outGridData[indexIjk][cidx], addValue);
            } else {
                outGridData[indexIjk][cidx] += addValue;
            }
        }
    }
}

template <c10::DeviceType DeviceTag, typename scalar_t>
torch::Tensor
SplatIntoGridTrilinear(const GridBatchImpl &batchHdl,
                       const JaggedTensor &points,
                       const torch::Tensor &pointsData) {
    int64_t numOutputValues = batchHdl.totalVoxels();
    auto opts               = torch::TensorOptions().dtype(points.dtype()).device(points.device());
    torch::Tensor outGridData =
        torch::zeros(spliceShape({numOutputValues}, pointsData, 1), opts); // [N, *]

    torch::Tensor pointsDataReshape  = featureCoalescedView(pointsData);   // [B*M, -1]
    torch::Tensor outGridDataReshape = featureCoalescedView(outGridData);  // [N, -1]

    torch::Tensor _outGridData;
    if (points.scalar_type() == at::kHalf) {
        _outGridData = torch::zeros_like(outGridData, outGridData.options().dtype(torch::kFloat32));
    } else {
        _outGridData = outGridData;
    }

    auto batchAcc       = gridBatchAccessor<DeviceTag>(batchHdl);
    auto pointsDataAcc  = tensorAccessor<DeviceTag, scalar_t, 2>(pointsData);
    auto outGridDataAcc = tensorAccessor<DeviceTag, at::opmath_type<scalar_t>, 2>(_outGridData);

    if constexpr (DeviceTag == torch::kCUDA) {
        auto cb = [=]
            __device__(int32_t bidx, int32_t eidx, int32_t cidx, JaggedRAcc32<scalar_t, 2> pts) {
                splatIntoGridTrilinearCallback<DeviceTag, scalar_t, JaggedRAcc32, TorchRAcc32>(
                    bidx, eidx, cidx, pts, pointsDataAcc, batchAcc, outGridDataAcc);
            };
        forEachJaggedElementChannelCUDA<scalar_t, 2>(256, pointsData.size(1), points, cb);
    } else if constexpr (DeviceTag == torch::kPrivateUse1) {
        auto cb = [=]
            __device__(int32_t bidx, int32_t eidx, int32_t cidx, JaggedRAcc32<scalar_t, 2> pts) {
                splatIntoGridTrilinearCallback<DeviceTag, scalar_t, JaggedRAcc32, TorchRAcc32>(
                    bidx, eidx, cidx, pts, pointsDataAcc, batchAcc, outGridDataAcc);
            };
        forEachJaggedElementChannelPrivateUse1<scalar_t, 2>(pointsData.size(1), points, cb);
    } else {
        auto cb = [=](int32_t bidx, int32_t eidx, int32_t cidx, JaggedAcc<scalar_t, 2> pts) {
            splatIntoGridTrilinearCallback<DeviceTag, scalar_t, JaggedAcc, TorchAcc>(
                bidx, eidx, cidx, pts, pointsDataAcc, batchAcc, outGridDataAcc);
        };
        forEachJaggedElementChannelCPU<scalar_t, 2>(pointsData.size(1), points, cb);
    }

    if (points.scalar_type() == at::kHalf) {
        outGridData.copy_(_outGridData);
    }

    return outGridData;
}

template <c10::DeviceType DeviceTag>
torch::Tensor
dispatchSplatIntoGridTrilinear<DeviceTag>(const GridBatchImpl &batchHdl,
                                          const JaggedTensor &points,
                                          const torch::Tensor &pointsData) {
    return AT_DISPATCH_V2(points.scalar_type(),
                          "SplatIntoGridTrilinear",
                          AT_WRAP([&] {
                              return SplatIntoGridTrilinear<DeviceTag, scalar_t>(
                                  batchHdl, points, pointsData);
                          }),
                          AT_EXPAND(AT_FLOATING_TYPES),
                          c10::kHalf);
}

template torch::Tensor dispatchSplatIntoGridTrilinear<torch::kCPU>(const GridBatchImpl &,
                                                                   const JaggedTensor &,
                                                                   const torch::Tensor &);

template torch::Tensor dispatchSplatIntoGridTrilinear<torch::kCUDA>(const GridBatchImpl &,
                                                                    const JaggedTensor &,
                                                                    const torch::Tensor &);

template torch::Tensor dispatchSplatIntoGridTrilinear<torch::kPrivateUse1>(const GridBatchImpl &,
                                                                           const JaggedTensor &,
                                                                           const torch::Tensor &);

} // namespace ops
} // namespace detail
} // namespace fvdb
