// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/SplatTrilinear.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/ForEachCPU.h>
#include <fvdb/detail/utils/TrilinearStencil.h>
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/cuda/ForEachCUDA.cuh>
#include <fvdb/detail/utils/cuda/ForEachPrivateUse1.cuh>

#include <THC/THCAtomics.cuh>
#include <c10/cuda/CUDAException.h>

#include <cstdint>

namespace fvdb {
namespace detail {
namespace ops {

// One-thread-per-point scalar callback. Resolves stencil once, then scatters
// weighted point data across all channels using cached indices.
template <torch::DeviceType DeviceTag,
          typename ScalarType,
          template <typename T, int32_t D>
          typename JaggedAccessor,
          template <typename T, int32_t D>
          typename TensorAccessor>
__hostdev__ void
splatTrilinearStencilCallback(int32_t bidx,
                                      int32_t eidx,
                                      int32_t /*cidx*/,
                                      JaggedAccessor<ScalarType, 2> points,
                                      TensorAccessor<ScalarType, 2> pointsData,
                                      BatchGridAccessor batchAccessor,
                                      TensorAccessor<at::opmath_type<ScalarType>, 2> outGridData,
                                      int64_t numChannels) {
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

    int64_t indices[8] = {};
    MathType weights[8];
    const uint8_t activeMask = resolveTrilinearStencil(xyz, gridAcc, baseOffset, indices, weights);

    if (activeMask == 0)
        return;

    for (int corner = 0; corner < 8; ++corner) {
        if (!(activeMask & (1 << corner)))
            continue;
        const MathType wt = weights[corner];
        for (int64_t c = 0; c < numChannels; ++c) {
            const MathType addValue = wt * static_cast<MathType>(pointsData[eidx][c]);
            if constexpr (DeviceTag == torch::kCUDA) {
                gpuAtomicAddNoReturn(&outGridData[indices[corner]][c], addValue);
            } else if constexpr (DeviceTag == torch::kPrivateUse1) {
                atomicAdd_system(&outGridData[indices[corner]][c], addValue);
            } else {
                outGridData[indices[corner]][c] += addValue;
            }
        }
    }
}

// One-thread-per-point Vec4 callback. GPU only. Uses float4 reads for point data
// and scalar atomic writes per channel.
template <torch::DeviceType DeviceTag,
          template <typename T, int32_t D>
          typename JaggedAccessor,
          template <typename T, int32_t D>
          typename TensorAccessor>
__device__ void
splatTrilinearStencilCallbackVec4(int32_t bidx,
                                          int32_t eidx,
                                          int32_t /*cidx*/,
                                          JaggedAccessor<float, 2> points,
                                          TensorAccessor<float, 2> pointsData,
                                          BatchGridAccessor batchAccessor,
                                          TensorAccessor<float, 2> outGridData,
                                          int64_t numChannels) {
    const auto &pointCoordData = points.data();

    const nanovdb::OnIndexGrid *gpuGrid  = batchAccessor.grid(bidx);
    auto gridAcc                         = gpuGrid->getAccessor();
    const int64_t baseOffset             = batchAccessor.voxelOffset(bidx);
    const VoxelCoordTransform &transform = batchAccessor.primalTransform(bidx);

    const nanovdb::math::Vec3<float> xyz =
        transform.apply(pointCoordData[eidx][0], pointCoordData[eidx][1], pointCoordData[eidx][2]);

    int64_t indices[8] = {};
    float weights[8];
    const uint8_t activeMask = resolveTrilinearStencil(xyz, gridAcc, baseOffset, indices, weights);

    if (activeMask == 0)
        return;

    const int64_t numGroups = numChannels / 4;

    for (int corner = 0; corner < 8; ++corner) {
        if (!(activeMask & (1 << corner)))
            continue;
        const float wt = weights[corner];
        for (int64_t g = 0; g < numGroups; ++g) {
            const int64_t cBase = g * 4;
            const float4 pData  = *reinterpret_cast<const float4 *>(&pointsData[eidx][cBase]);

            if constexpr (DeviceTag == torch::kCUDA) {
                gpuAtomicAddNoReturn(&outGridData[indices[corner]][cBase + 0], wt * pData.x);
                gpuAtomicAddNoReturn(&outGridData[indices[corner]][cBase + 1], wt * pData.y);
                gpuAtomicAddNoReturn(&outGridData[indices[corner]][cBase + 2], wt * pData.z);
                gpuAtomicAddNoReturn(&outGridData[indices[corner]][cBase + 3], wt * pData.w);
            } else if constexpr (DeviceTag == torch::kPrivateUse1) {
                atomicAdd_system(&outGridData[indices[corner]][cBase + 0], wt * pData.x);
                atomicAdd_system(&outGridData[indices[corner]][cBase + 1], wt * pData.y);
                atomicAdd_system(&outGridData[indices[corner]][cBase + 2], wt * pData.z);
                atomicAdd_system(&outGridData[indices[corner]][cBase + 3], wt * pData.w);
            }
        }
    }
}

template <torch::DeviceType DeviceTag, typename scalar_t>
torch::Tensor
SplatIntoGridTrilinear(const GridBatchData &batchHdl,
                       const JaggedTensor &points,
                       const torch::Tensor &pointsData) {
    int64_t numOutputValues = batchHdl.totalVoxels();
    auto opts               = torch::TensorOptions().dtype(points.dtype()).device(points.device());
    torch::Tensor outGridData =
        torch::zeros(spliceShape({numOutputValues}, pointsData, 1), opts);            // [N, *]

    torch::Tensor pointsDataReshape  = featureCoalescedView(pointsData).contiguous(); // [B*M, -1]
    torch::Tensor outGridDataReshape = featureCoalescedView(outGridData);             // [N, -1]

    torch::Tensor _outGridData;
    if (points.scalar_type() == at::kHalf) {
        _outGridData = torch::zeros_like(outGridData, outGridData.options().dtype(torch::kFloat32));
    } else {
        _outGridData = outGridData;
    }

    auto batchAcc       = gridBatchAccessor<DeviceTag>(batchHdl);
    auto pointsDataAcc  = tensorAccessor<DeviceTag, scalar_t, 2>(pointsDataReshape);
    auto outGridDataAcc = tensorAccessor<DeviceTag, at::opmath_type<scalar_t>, 2>(_outGridData);
    const int64_t numChannels = pointsDataReshape.size(1);

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
                reinterpret_cast<uintptr_t>(pointsDataReshape.data_ptr<float>()) % 16 == 0) {
                auto cb = [=] __device__(int32_t bidx,
                                         int32_t eidx,
                                         int32_t cidx,
                                         JaggedRAcc64<float, 2> pts) {
                    splatTrilinearStencilCallbackVec4<DeviceTag, JaggedRAcc64, TorchRAcc64>(
                        bidx,
                        eidx,
                        cidx,
                        pts,
                        pointsDataAcc,
                        batchAcc,
                        outGridDataAcc,
                        numChannels);
                };
                dispatchForEach(cb);
            } else {
                auto cb = [=] __device__(int32_t bidx,
                                         int32_t eidx,
                                         int32_t cidx,
                                         JaggedRAcc64<float, 2> pts) {
                    splatTrilinearStencilCallback<DeviceTag,
                                                          float,
                                                          JaggedRAcc64,
                                                          TorchRAcc64>(bidx,
                                                                       eidx,
                                                                       cidx,
                                                                       pts,
                                                                       pointsDataAcc,
                                                                       batchAcc,
                                                                       outGridDataAcc,
                                                                       numChannels);
                };
                dispatchForEach(cb);
            }
        } else {
            auto cb = [=] __device__(int32_t bidx,
                                     int32_t eidx,
                                     int32_t cidx,
                                     JaggedRAcc64<scalar_t, 2> pts) {
                splatTrilinearStencilCallback<DeviceTag,
                                                      scalar_t,
                                                      JaggedRAcc64,
                                                      TorchRAcc64>(
                    bidx, eidx, cidx, pts, pointsDataAcc, batchAcc, outGridDataAcc, numChannels);
            };
            dispatchForEach(cb);
        }
    } else {
        auto cb = [=](int32_t bidx, int32_t eidx, int32_t cidx, JaggedAcc<scalar_t, 2> pts) {
            splatTrilinearStencilCallback<DeviceTag, scalar_t, JaggedAcc, TorchAcc>(
                bidx, eidx, cidx, pts, pointsDataAcc, batchAcc, outGridDataAcc, numChannels);
        };
        forEachJaggedElementChannelCPU<scalar_t, 2>(int64_t(1), points, cb);
    }

    if (points.scalar_type() == at::kHalf) {
        outGridData.copy_(_outGridData);
    }

    return outGridData;
}

template <torch::DeviceType DeviceTag>
torch::Tensor
dispatchSplatIntoGridTrilinear(const GridBatchData &batchHdl,
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

template torch::Tensor dispatchSplatIntoGridTrilinear<torch::kCPU>(const GridBatchData &,
                                                                   const JaggedTensor &,
                                                                   const torch::Tensor &);

template torch::Tensor dispatchSplatIntoGridTrilinear<torch::kCUDA>(const GridBatchData &,
                                                                    const JaggedTensor &,
                                                                    const torch::Tensor &);

template torch::Tensor dispatchSplatIntoGridTrilinear<torch::kPrivateUse1>(const GridBatchData &,
                                                                           const JaggedTensor &,
                                                                           const torch::Tensor &);

torch::Tensor
splatTrilinear(const GridBatchData &batchHdl,
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
        return dispatchSplatIntoGridTrilinear<DeviceTag>(batchHdl, points, pointsData);
    });
}

} // namespace ops
} // namespace detail
} // namespace fvdb
