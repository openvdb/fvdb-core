// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/GridBatchImpl.h>
#include <fvdb/detail/ops/SerializeEncode.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/ForEachCPU.h>
#include <fvdb/detail/utils/HilbertCode.h>
#include <fvdb/detail/utils/MortonCode.h>
#include <fvdb/detail/utils/cuda/ForEachCUDA.cuh>
#include <fvdb/detail/utils/cuda/ForEachPrivateUse1.cuh>

#include <c10/cuda/CUDAException.h>

#include <cuda_runtime.h>

#include <vector>

namespace fvdb {
namespace detail {
namespace ops {

/// @brief Per-voxel callback which computes the space-filling curve code (Morton or Hilbert) for
/// each active voxel in a batch of grids
template <template <typename T, int32_t D> typename TorchAccessor>
__hostdev__ inline void
serializeEncodeVoxelCallback(int64_t batchIdx,
                             int64_t leafIdx,
                             int64_t voxelIdx,
                             GridBatchImpl::Accessor gridAccessor,
                             TorchAccessor<int64_t, 2> outMortonCodes,
                             const nanovdb::Coord &offset,
                             int order_type) {
    const nanovdb::OnIndexGrid *grid = gridAccessor.grid(batchIdx);
    const typename nanovdb::OnIndexGrid::LeafNodeType &leaf =
        grid->tree().template getFirstNode<0>()[leafIdx];
    const int64_t baseOffset = gridAccessor.voxelOffset(batchIdx);

    const nanovdb::Coord &ijk = leaf.offsetToGlobalCoord(voxelIdx);
    if (leaf.isActive(voxelIdx)) {
        const int64_t idx = baseOffset + (int64_t)leaf.getValue(voxelIdx) - 1;

        // Apply offset to coordinates
        int32_t offset_i = offset[0];
        int32_t offset_j = offset[1];
        int32_t offset_k = offset[2];

        // Compute Morton or Hilbert code with offset to ensure non-negative coordinates
        uint64_t space_filling_code;
        switch (static_cast<SpaceFillingCurveType>(order_type)) {
        case SpaceFillingCurveType::ZOrder: // Regular z-order: xyz
            space_filling_code = utils::morton_with_offset(ijk[0],
                                                           ijk[1],
                                                           ijk[2],
                                                           static_cast<uint32_t>(offset_i),
                                                           static_cast<uint32_t>(offset_j),
                                                           static_cast<uint32_t>(offset_k));
            break;
        case SpaceFillingCurveType::ZOrderTransposed: // Transposed z-order: zyx
            space_filling_code = utils::morton_with_offset(ijk[2],
                                                           ijk[1],
                                                           ijk[0],
                                                           static_cast<uint32_t>(offset_k),
                                                           static_cast<uint32_t>(offset_j),
                                                           static_cast<uint32_t>(offset_i));
            break;
        case SpaceFillingCurveType::Hilbert: // Regular Hilbert curve: xyz
            space_filling_code = utils::hilbert_with_offset(ijk[0],
                                                            ijk[1],
                                                            ijk[2],
                                                            static_cast<uint32_t>(offset_i),
                                                            static_cast<uint32_t>(offset_j),
                                                            static_cast<uint32_t>(offset_k));
            break;
        case SpaceFillingCurveType::HilbertTransposed: // Transposed Hilbert curve: zyx
            space_filling_code = utils::hilbert_with_offset(ijk[2],
                                                            ijk[1],
                                                            ijk[0],
                                                            static_cast<uint32_t>(offset_k),
                                                            static_cast<uint32_t>(offset_j),
                                                            static_cast<uint32_t>(offset_i));
            break;
        default:
            // Invalid order type - use assert for device code
            space_filling_code = 0;
            break;
        }

        outMortonCodes[idx][0] = static_cast<int64_t>(space_filling_code);
    }
}

/// @brief Get the space-filling curve codes for active voxels in a batch of grids
/// @param gridBatch The batch of grids
/// @param outMortonCodes Tensor which will contain the output space-filling curve codes
/// @param offset Offset to apply to voxel coordinates before encoding
/// @param order_type Integer representing the order type (0=z, 1=z-trans, 2=hilbert,
/// 3=hilbert-trans)
template <torch::DeviceType DeviceTag>
void
GetSerializeEncode(const GridBatchImpl &gridBatch,
                   torch::Tensor &outMortonCodes,
                   const nanovdb::Coord &offset,
                   int order_type) {
    auto outCodesAcc = tensorAccessor<DeviceTag, int64_t, 2>(outMortonCodes);

    if constexpr (DeviceTag == torch::kCUDA) {
        auto cb = [=] __device__(int64_t batchIdx,
                                 int64_t leafIdx,
                                 int64_t voxelIdx,
                                 int64_t,
                                 GridBatchImpl::Accessor gridAccessor) {
            serializeEncodeVoxelCallback<TorchRAcc32>(
                batchIdx, leafIdx, voxelIdx, gridAccessor, outCodesAcc, offset, order_type);
        };
        forEachVoxelCUDA(1024, 1, gridBatch, cb);
    } else if constexpr (DeviceTag == torch::kPrivateUse1) {
        auto cb = [=] __device__(int64_t batchIdx,
                                 int64_t leafIdx,
                                 int64_t voxelIdx,
                                 int64_t,
                                 GridBatchImpl::Accessor gridAccessor) {
            serializeEncodeVoxelCallback<TorchRAcc32>(
                batchIdx, leafIdx, voxelIdx, gridAccessor, outCodesAcc, offset, order_type);
        };
        forEachVoxelPrivateUse1(1, gridBatch, cb);
    } else {
        auto cb = [=](int64_t batchIdx,
                      int64_t leafIdx,
                      int64_t voxelIdx,
                      int64_t,
                      GridBatchImpl::Accessor gridAccessor) {
            serializeEncodeVoxelCallback<TorchAcc>(
                batchIdx, leafIdx, voxelIdx, gridAccessor, outCodesAcc, offset, order_type);
        };
        forEachVoxelCPU(1, gridBatch, cb);
    }
}

/// @brief Get the space-filling curve codes for active voxels in a batch of grids
/// @tparam DeviceTag Which device to run on
/// @param gridBatch The batch of grids to get the space-filling curve codes for
/// @param order_type The type of space-filling curve to use for encoding
/// @param offset Offset to apply to voxel coordinates before encoding
/// @return A JaggedTensor of shape [B, -1, 1] of space-filling curve codes for active voxels
template <torch::DeviceType DeviceTag>
JaggedTensor
SerializeEncode(const GridBatchImpl &gridBatch,
                SpaceFillingCurveType order_type,
                const nanovdb::Coord &offset) {
    gridBatch.checkNonEmptyGrid();
    auto opts = torch::TensorOptions().dtype(torch::kInt64).device(gridBatch.device());
    torch::Tensor outMortonCodes = torch::empty({gridBatch.totalVoxels(), 1}, opts);

    // Convert enum to integer for kernel
    const int order_type_int = static_cast<int>(order_type);

    GetSerializeEncode<DeviceTag>(gridBatch, outMortonCodes, offset, order_type_int);

    return gridBatch.jaggedTensor(outMortonCodes);
}

template <>
JaggedTensor
dispatchSerializeEncode<torch::kCUDA>(const GridBatchImpl &gridBatch,
                                      SpaceFillingCurveType order_type,
                                      const nanovdb::Coord &offset) {
    return SerializeEncode<torch::kCUDA>(gridBatch, order_type, offset);
}

template <>
JaggedTensor
dispatchSerializeEncode<torch::kCPU>(const GridBatchImpl &gridBatch,
                                     SpaceFillingCurveType order_type,
                                     const nanovdb::Coord &offset) {
    return SerializeEncode<torch::kCPU>(gridBatch, order_type, offset);
}

template <>
JaggedTensor
dispatchSerializeEncode<torch::kPrivateUse1>(const GridBatchImpl &gridBatch,
                                             SpaceFillingCurveType order_type,
                                             const nanovdb::Coord &offset) {
    return SerializeEncode<torch::kPrivateUse1>(gridBatch, order_type, offset);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
