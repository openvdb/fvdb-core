// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/SampleBezier.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/BezierInterpolationIterator.h>
#include <fvdb/detail/utils/ForEachCPU.h>
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/cuda/ForEachCUDA.cuh>

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
sampleBezierCallback(int32_t bidx,
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

    for (auto it = BezierInterpolationIterator<MathType>(xyz); it.isValid(); ++it) {
        const nanovdb::Coord ijk = it->first;
        const bool isActive      = gridAcc.isActive(ijk);
        const MathType wBezier   = it->second;
        const int64_t indexIjk   = gridAcc.getValue(ijk) - 1 + baseOffset;
        if (isActive) {
            outFeatures[eidx][cidx] += wBezier * gridData[indexIjk][cidx];
        }
    }
}

template <torch::DeviceType DeviceTag>
std::vector<torch::Tensor>
SampleGridBezier(const GridBatchData &batchHdl,
                 const JaggedTensor &points,
                 const torch::Tensor &gridData) {
    auto opts = torch::TensorOptions()
                    .dtype(gridData.dtype())
                    .device(gridData.device())
                    .requires_grad(gridData.requires_grad());
    torch::Tensor gridDataReshape = featureCoalescedView(gridData);     // [N, -1]
    torch::Tensor outFeatures =
        torch::zeros({points.rsize(0), gridDataReshape.size(1)}, opts); // [B*M, -1]
    auto outShape = spliceShape({points.rsize(0)}, gridData, 1);        // [B*M, *]

    AT_DISPATCH_V2(
        points.scalar_type(),
        "SampleGridBezier",
        AT_WRAP([&] {
            auto batchAcc       = gridBatchAccessor<DeviceTag>(batchHdl);
            auto gridDataAcc    = tensorAccessor<DeviceTag, scalar_t, 2>(gridDataReshape);
            auto outFeaturesAcc = tensorAccessor<DeviceTag, scalar_t, 2>(outFeatures);

            if constexpr (DeviceTag == torch::kCUDA) {
                auto cb = [=] __device__(int32_t bidx,
                                         int32_t eidx,
                                         int32_t cidx,
                                         JaggedRAcc64<scalar_t, 2> pts) {
                    sampleBezierCallback<scalar_t, JaggedRAcc64, TorchRAcc64>(
                        bidx, eidx, cidx, pts, gridDataAcc, batchAcc, outFeaturesAcc);
                };
                forEachJaggedElementChannelCUDA<scalar_t, 2>(
                    256, gridDataReshape.size(1), points, cb);
            } else {
                auto cb =
                    [=](int32_t bidx, int32_t eidx, int32_t cidx, JaggedAcc<scalar_t, 2> pts) {
                        sampleBezierCallback<scalar_t, JaggedAcc, TorchAcc>(
                            bidx, eidx, cidx, pts, gridDataAcc, batchAcc, outFeaturesAcc);
                    };
                forEachJaggedElementChannelCPU<scalar_t, 2>(gridDataReshape.size(1), points, cb);
            }
        }),
        AT_EXPAND(AT_FLOATING_TYPES),
        c10::kHalf);

    return {outFeatures.reshape(outShape)};
}

std::vector<torch::Tensor>
sampleBezier(const GridBatchData &batchHdl,
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
    return FVDB_DISPATCH_KERNEL_DEVICE(
        points.device(), [&]() { return SampleGridBezier<DeviceTag>(batchHdl, points, gridData); });
}

} // namespace ops
} // namespace detail
} // namespace fvdb
