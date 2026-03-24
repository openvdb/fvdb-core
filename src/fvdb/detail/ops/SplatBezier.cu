// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/SplatBezier.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/BezierInterpolationIterator.h>
#include <fvdb/detail/utils/ForEachCPU.h>
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/cuda/ForEachCUDA.cuh>

#include <THC/THCAtomics.cuh>
#include <c10/cuda/CUDAException.h>

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
splatBezierCallback(int32_t bidx,
                            int32_t eidx,
                            int32_t cidx,
                            JaggedAccessor<ScalarType, 2> points,
                            TensorAccessor<ScalarType, 2> pointsData,
                            BatchGridAccessor batchAccessor,
                            TensorAccessor<at::opmath_type<ScalarType>, 2> outGridData) {
    using MathType = at::opmath_type<ScalarType>;

    const auto pointCoords               = points.data();
    const nanovdb::OnIndexGrid *gpuGrid  = batchAccessor.grid(bidx);
    auto gridAcc                         = gpuGrid->getAccessor();
    const int64_t baseOffset             = batchAccessor.voxelOffset(bidx);
    const VoxelCoordTransform &transform = batchAccessor.primalTransform(bidx);

    const nanovdb::math::Vec3<MathType> xyz =
        transform.apply(static_cast<MathType>(pointCoords[eidx][0]),
                        static_cast<MathType>(pointCoords[eidx][1]),
                        static_cast<MathType>(pointCoords[eidx][2]));

#pragma unroll
    for (auto it = BezierInterpolationIterator<MathType>(xyz); it.isValid(); ++it) {
        if (gridAcc.isActive(it->first)) {
            const int64_t indexIjk  = (int64_t)gridAcc.getValue(it->first) - 1 + baseOffset;
            const MathType addValue = it->second * static_cast<MathType>(pointsData[eidx][cidx]);
            if constexpr (DeviceTag == torch::kCUDA) {
                gpuAtomicAddNoReturn(&outGridData[indexIjk][cidx], addValue);
            } else {
                // FIXME: (@fwilliams) Make me threadsafe
                outGridData[indexIjk][cidx] += addValue;
            }
        }
    }
}

template <torch::DeviceType DeviceTag>
torch::Tensor
SplatIntoGridBezier(const GridBatchImpl &batchHdl,
                    const JaggedTensor &points,
                    const torch::Tensor &pointsData) {
    int64_t numOutputValues = batchHdl.totalVoxels();
    auto opts               = torch::TensorOptions().dtype(points.dtype()).device(points.device());
    torch::Tensor outGridData =
        torch::zeros(spliceShape({numOutputValues}, pointsData, 1), opts); // [N, *]

    torch::Tensor pointsDataReshape  = featureCoalescedView(pointsData);   // [B*M, -1]
    torch::Tensor outGridDataReshape = featureCoalescedView(outGridData);  // [N, -1]

    AT_DISPATCH_V2(
        points.scalar_type(),
        "SplatIntoGridBezier",
        AT_WRAP([&] {
            torch::Tensor _outGridData;
            if (points.scalar_type() == at::kHalf) {
                _outGridData = torch::zeros_like(outGridDataReshape,
                                                 outGridData.options().dtype(torch::kFloat32));
            } else {
                _outGridData = outGridDataReshape;
            }

            auto batchAcc      = gridBatchAccessor<DeviceTag>(batchHdl);
            auto pointsDataAcc = tensorAccessor<DeviceTag, scalar_t, 2>(pointsData);
            auto outGridDataAcc =
                tensorAccessor<DeviceTag, at::opmath_type<scalar_t>, 2>(_outGridData);

            if constexpr (DeviceTag == torch::kCUDA) {
                auto cb = [=] __device__(int32_t bidx,
                                         int32_t eidx,
                                         int32_t cidx,
                                         JaggedRAcc64<scalar_t, 2> ptsA) {
                    splatBezierCallback<DeviceTag, scalar_t, JaggedRAcc64, TorchRAcc64>(
                        bidx, eidx, cidx, ptsA, pointsDataAcc, batchAcc, outGridDataAcc);
                };
                forEachJaggedElementChannelCUDA<scalar_t, 2>(256, pointsData.size(1), points, cb);
            } else {
                auto cb =
                    [=](int32_t bidx, int32_t eidx, int32_t cidx, JaggedAcc<scalar_t, 2> ptsA) {
                        splatBezierCallback<DeviceTag, scalar_t, JaggedAcc, TorchAcc>(
                            bidx, eidx, cidx, ptsA, pointsDataAcc, batchAcc, outGridDataAcc);
                    };
                forEachJaggedElementChannelCPU<scalar_t, 2>(pointsData.size(1), points, cb);
            }

            if (points.scalar_type() == at::kHalf) {
                outGridData.copy_(_outGridData);
            }
        }),
        AT_EXPAND(AT_FLOATING_TYPES),
        c10::kHalf);

    return outGridData;
}

torch::Tensor
splatBezier(const GridBatchImpl &batchHdl,
                    const JaggedTensor &points,
                    const torch::Tensor &pointsData) {
    batchHdl.checkNonEmptyGrid();
    TORCH_CHECK_VALUE(points.device() == pointsData.device(),
                      "points and data must be on the same device");
    batchHdl.checkDevice(points);
    batchHdl.checkDevice(pointsData);
    points.check_valid();
    TORCH_CHECK_TYPE(points.is_floating_point(), "points must have a floating point type");
    TORCH_CHECK_TYPE(points.dtype() == pointsData.dtype(), "all tensors must have the same type");
    TORCH_CHECK_VALUE(points.rdim() == 2,
                      "Expected points to have shape [B*M, 3] (wrong number of dimensions)");
    TORCH_CHECK(points.numel() > 0, "Empty tensor (points)");
    TORCH_CHECK(points.rsize(1) == 3, "points must have shape [B*M, 3] (points must be 3D)");
    TORCH_CHECK_TYPE(pointsData.is_floating_point(), "point_data must have a floating point type");
    TORCH_CHECK_VALUE(pointsData.dim() >= 2,
                      "Expected data to have shape [B*M, *] (at least 3 dimensions)");
    TORCH_CHECK(pointsData.numel() > 0, "Empty tensor (data)");
    TORCH_CHECK(
        pointsData.size(0) == points.rsize(0),
        "point_data must have one value per point (shape [B*M, *]) (incorrect first dimension must match number of points)");
    return FVDB_DISPATCH_KERNEL_DEVICE(points.device(), [&]() {
        return SplatIntoGridBezier<DeviceTag>(batchHdl, points, pointsData);
    });
}

} // namespace ops
} // namespace detail
} // namespace fvdb
