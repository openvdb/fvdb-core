// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/SplatIntoGridTrilinear.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/ForEachCPU.h>
#include <fvdb/detail/utils/TrilinearInterpolationIterator.h>
#include <fvdb/detail/utils/cuda/ForEachCUDA.cuh>
#include <fvdb/detail/utils/cuda/ForEachPrivateUse1.cuh>

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

// Vectorized callback for float32 - processes 4 channels per thread
// Uses vectorized reads for pointsData, but scalar atomic writes
template <torch::DeviceType DeviceTag,
          template <typename T, int32_t D>
          typename JaggedAccessor,
          template <typename T, int32_t D>
          typename TensorAccessor>
__device__ void
splatIntoGridTrilinearCallbackVec4(int32_t bidx,
                                   int32_t eidx,
                                   int32_t cidx, // channel group index
                                   JaggedAccessor<float, 2> points,
                                   TensorAccessor<float, 2> pointsData,
                                   BatchGridAccessor batchAccessor,
                                   TensorAccessor<float, 2> outGridData) {
    const int32_t cBase = cidx * 4;

    const auto &pointCoordData = points.data();

    const nanovdb::OnIndexGrid *gpuGrid  = batchAccessor.grid(bidx);
    auto gridAcc                         = gpuGrid->getAccessor();
    const int64_t baseOffset             = batchAccessor.voxelOffset(bidx);
    const VoxelCoordTransform &transform = batchAccessor.primalTransform(bidx);

    const nanovdb::math::Vec3<float> xyz =
        transform.apply(pointCoordData[eidx][0], pointCoordData[eidx][1], pointCoordData[eidx][2]);

    // Vectorized load of 4 channels from pointsData
    const float4 pData =
        *reinterpret_cast<const float4 *>(__builtin_assume_aligned(&pointsData[eidx][cBase], 16));

#pragma unroll
    for (auto it = TrilinearInterpolationIterator<float>(xyz); it.isValid(); ++it) {
        if (gridAcc.isActive(it->first)) {
            const int64_t indexIjk = gridAcc.getValue(it->first) - 1 + baseOffset;
            const float weight     = it->second;

            if constexpr (DeviceTag == torch::kCUDA) {
                gpuAtomicAddNoReturn(&outGridData[indexIjk][cBase + 0], weight * pData.x);
                gpuAtomicAddNoReturn(&outGridData[indexIjk][cBase + 1], weight * pData.y);
                gpuAtomicAddNoReturn(&outGridData[indexIjk][cBase + 2], weight * pData.z);
                gpuAtomicAddNoReturn(&outGridData[indexIjk][cBase + 3], weight * pData.w);
            } else if constexpr (DeviceTag == torch::kPrivateUse1) {
                atomicAdd_system(&outGridData[indexIjk][cBase + 0], weight * pData.x);
                atomicAdd_system(&outGridData[indexIjk][cBase + 1], weight * pData.y);
                atomicAdd_system(&outGridData[indexIjk][cBase + 2], weight * pData.z);
                atomicAdd_system(&outGridData[indexIjk][cBase + 3], weight * pData.w);
            }
        }
    }
}

template <torch::DeviceType DeviceTag, typename scalar_t>
torch::Tensor
SplatIntoGridTrilinear(const GridBatchImpl &batchHdl,
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
                reinterpret_cast<uintptr_t>(pointsDataReshape.data_ptr<float>()) % 16 == 0) {
                const auto numChannelGroups = (numChannels + 3) / 4;
                auto cb                     = [=] __device__(int32_t bidx,
                                         int32_t eidx,
                                         int32_t cidx,
                                         JaggedRAcc32<float, 2> pts) {
                    splatIntoGridTrilinearCallbackVec4<DeviceTag, JaggedRAcc32, TorchRAcc32>(
                        bidx, eidx, cidx, pts, pointsDataAcc, batchAcc, outGridDataAcc);
                };
                dispatchForEach(numChannelGroups, cb);
            } else {
                auto cb = [=] __device__(int32_t bidx,
                                         int32_t eidx,
                                         int32_t cidx,
                                         JaggedRAcc32<float, 2> pts) {
                    splatIntoGridTrilinearCallback<DeviceTag, float, JaggedRAcc32, TorchRAcc32>(
                        bidx, eidx, cidx, pts, pointsDataAcc, batchAcc, outGridDataAcc);
                };
                dispatchForEach(numChannels, cb);
            }
        } else {
            auto cb = [=] __device__(int32_t bidx,
                                     int32_t eidx,
                                     int32_t cidx,
                                     JaggedRAcc32<scalar_t, 2> pts) {
                splatIntoGridTrilinearCallback<DeviceTag, scalar_t, JaggedRAcc32, TorchRAcc32>(
                    bidx, eidx, cidx, pts, pointsDataAcc, batchAcc, outGridDataAcc);
            };
            dispatchForEach(numChannels, cb);
        }
    } else {
        auto cb = [=](int32_t bidx, int32_t eidx, int32_t cidx, JaggedAcc<scalar_t, 2> pts) {
            splatIntoGridTrilinearCallback<DeviceTag, scalar_t, JaggedAcc, TorchAcc>(
                bidx, eidx, cidx, pts, pointsDataAcc, batchAcc, outGridDataAcc);
        };
        forEachJaggedElementChannelCPU<scalar_t, 2>(numChannels, points, cb);
    }

    if (points.scalar_type() == at::kHalf) {
        outGridData.copy_(_outGridData);
    }

    return outGridData;
}

template <torch::DeviceType DeviceTag>
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
