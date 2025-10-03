// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/GridBatchImpl.h>
#include <fvdb/detail/ops/SerializeEncode.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/ForEachCPU.h>
#include <fvdb/detail/utils/cuda/ForEachCUDA.cuh>
#include <fvdb/detail/utils/cuda/ForEachPrivateUse1.cuh>
#include <fvdb/detail/utils/MortonCode.h>

#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <vector>

namespace fvdb {
namespace detail {
namespace ops {

/// @brief Per-voxel callback which computes the Morton code for each active voxel in a batch of grids
template <template <typename T, int32_t D> typename TorchAccessor>
__hostdev__ inline void
serializeEncodeVoxelCallback(int64_t batchIdx,
                             int64_t leafIdx,
                             int64_t voxelIdx,
                             GridBatchImpl::Accessor gridAccessor,
                             TorchAccessor<int64_t, 2> outMortonCodes,
                             const nanovdb::Coord* bboxMinCoords,
                             int order_type) {
    const nanovdb::OnIndexGrid *grid = gridAccessor.grid(batchIdx);
    const typename nanovdb::OnIndexGrid::LeafNodeType &leaf =
        grid->tree().template getFirstNode<0>()[leafIdx];
    const int64_t baseOffset = gridAccessor.voxelOffset(batchIdx);

    const nanovdb::Coord &ijk = leaf.offsetToGlobalCoord(voxelIdx);
    if (leaf.isActive(voxelIdx)) {
        const int64_t idx = baseOffset + (int64_t)leaf.getValue(voxelIdx) - 1;
        
        // Get bounding box minimum for this batch to use as offset
        const nanovdb::Coord &bboxMin = bboxMinCoords[batchIdx];
        int32_t offset_i = bboxMin[0];
        int32_t offset_j = bboxMin[1];
        int32_t offset_k = bboxMin[2];
        
        // Compute Morton code with offset to ensure non-negative coordinates
        uint64_t morton_code;
        if (order_type == 1) {  // ORDER_TYPE_Z_TRANS
            // Transposed z-order: swap coordinates zyx
            morton_code = utils::morton_with_offset(
                ijk[2], ijk[1], ijk[0],
                static_cast<uint32_t>(-offset_k), 
                static_cast<uint32_t>(-offset_j), 
                static_cast<uint32_t>(-offset_i)
            );
        } else {  // ORDER_TYPE_Z (default)
            // Regular z-order: xyz
            morton_code = utils::morton_with_offset(
                ijk[0], ijk[1], ijk[2],
                static_cast<uint32_t>(-offset_i), 
                static_cast<uint32_t>(-offset_j), 
                static_cast<uint32_t>(-offset_k)
            );
        }
        
        outMortonCodes[idx][0] = static_cast<int64_t>(morton_code);
    }
}

/// @brief Get the Morton codes for active voxels in a batch of grids
/// @param gridBatch The batch of grids
/// @param outMortonCodes Tensor which will contain the output Morton codes
/// @param bboxMinCoords Array of bounding box minimum coordinates for each grid
/// @param order_type Integer representing the order type (0=z, 1=z-trans, etc.)
template <torch::DeviceType DeviceTag>
void
GetSerializeEncode(const GridBatchImpl &gridBatch, 
                   torch::Tensor &outMortonCodes,
                   const nanovdb::Coord* bboxMinCoords,
                   int order_type) {
    auto outCodesAcc = tensorAccessor<DeviceTag, int64_t, 2>(outMortonCodes);

    if constexpr (DeviceTag == torch::kCUDA) {
        auto cb = [=] __device__(int64_t batchIdx,
                                 int64_t leafIdx,
                                 int64_t voxelIdx,
                                 int64_t,
                                 GridBatchImpl::Accessor gridAccessor) {
            serializeEncodeVoxelCallback<TorchRAcc32>(
                batchIdx, leafIdx, voxelIdx, gridAccessor, outCodesAcc, bboxMinCoords, order_type);
        };
        forEachVoxelCUDA(1024, 1, gridBatch, cb);
    } else if constexpr (DeviceTag == torch::kPrivateUse1) {
        auto cb = [=] __device__(int64_t batchIdx,
                                 int64_t leafIdx,
                                 int64_t voxelIdx,
                                 int64_t,
                                 GridBatchImpl::Accessor gridAccessor) {
            serializeEncodeVoxelCallback<TorchRAcc32>(
                batchIdx, leafIdx, voxelIdx, gridAccessor, outCodesAcc, bboxMinCoords, order_type);
        };
        forEachVoxelPrivateUse1(1, gridBatch, cb);
    } else {
        auto cb = [=](int64_t batchIdx,
                      int64_t leafIdx,
                      int64_t voxelIdx,
                      int64_t,
                      GridBatchImpl::Accessor gridAccessor) {
            serializeEncodeVoxelCallback<TorchAcc>(
                batchIdx, leafIdx, voxelIdx, gridAccessor, outCodesAcc, bboxMinCoords, order_type);
        };
        forEachVoxelCPU(1, gridBatch, cb);
    }
}

/// @brief Get the Morton codes for active voxels in a batch of grids
/// @tparam DeviceTag Which device to run on
/// @param gridBatch The batch of grids to get the Morton codes for
/// @param order_type The type of z-order to use ("z" for xyz, "z-trans" for zyx)
/// @return A JaggedTensor of shape [B, -1, 1] of Morton codes for active voxels
template <torch::DeviceType DeviceTag>
JaggedTensor
SerializeEncode(const GridBatchImpl &gridBatch, const std::string &order_type) {
    gridBatch.checkNonEmptyGrid();
    auto opts = torch::TensorOptions().dtype(torch::kInt64).device(gridBatch.device());
    torch::Tensor outMortonCodes = torch::empty({gridBatch.totalVoxels(), 1}, opts);
    
    // Parse order type string to integer
    int order_type_int;
    if (order_type == "z") {
        order_type_int = ORDER_TYPE_Z;  // Regular xyz z-order
    } else if (order_type == "z-trans") {
        order_type_int = ORDER_TYPE_Z_TRANS;  // Transposed zyx z-order
    } else {
        // raise an error
        throw std::runtime_error("Invalid order_type: " + order_type + ". Valid options are 'z' or 'z-trans'.");
    }
    
    // Get bounding box minimum coordinates for offset calculation
    // We need to extract the minimum coordinates from each grid's bounding box
    std::vector<nanovdb::Coord> bboxMinCoords;
    bboxMinCoords.reserve(gridBatch.batchSize());
    
    for (int64_t i = 0; i < gridBatch.batchSize(); ++i) {
        nanovdb::CoordBBox bbox = gridBatch.bboxAt(i);
        bboxMinCoords.push_back(bbox.min());
    }
    
    // For GPU execution, we need to copy the data to device
    nanovdb::Coord* bboxMinPtr = nullptr;
    if constexpr (DeviceTag == torch::kCUDA || DeviceTag == torch::kPrivateUse1) {
        // Allocate device memory and copy
        cudaMalloc(&bboxMinPtr, bboxMinCoords.size() * sizeof(nanovdb::Coord));
        cudaMemcpy(bboxMinPtr, bboxMinCoords.data(), 
                   bboxMinCoords.size() * sizeof(nanovdb::Coord), 
                   cudaMemcpyHostToDevice);
    } else {
        bboxMinPtr = bboxMinCoords.data();
    }
    
    GetSerializeEncode<DeviceTag>(gridBatch, outMortonCodes, bboxMinPtr, order_type_int);
    
    // Clean up device memory if allocated
    if constexpr (DeviceTag == torch::kCUDA || DeviceTag == torch::kPrivateUse1) {
        cudaFree(bboxMinPtr);
    }
    
    return gridBatch.jaggedTensor(outMortonCodes);
}

template <>
JaggedTensor
dispatchSerializeEncode<torch::kCUDA>(const GridBatchImpl &gridBatch, const std::string &order_type) {
    return SerializeEncode<torch::kCUDA>(gridBatch, order_type);
}

template <>
JaggedTensor
dispatchSerializeEncode<torch::kCPU>(const GridBatchImpl &gridBatch, const std::string &order_type) {
    return SerializeEncode<torch::kCPU>(gridBatch, order_type);
}

template <>
JaggedTensor
dispatchSerializeEncode<torch::kPrivateUse1>(const GridBatchImpl &gridBatch, const std::string &order_type) {
    return SerializeEncode<torch::kPrivateUse1>(gridBatch, order_type);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
