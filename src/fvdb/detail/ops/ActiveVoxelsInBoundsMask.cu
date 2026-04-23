// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/GridBatchData.h>
#include <fvdb/detail/ops/ActiveVoxelsInBoundsMask.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/ForEachCPU.h>
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/cuda/ForEachCUDA.cuh>

#include <c10/cuda/CUDAException.h>
#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

/// @brief Per-voxel callback to compute a mask of the active grid voxels in a bounding box for a
/// batch of grids
template <template <typename T, int32_t D> typename TorchAccessor>
__hostdev__ inline void
activeGridVoxelInBoundsMaskCallback(int32_t batchIdx,
                                    int32_t leafIdx,
                                    int32_t voxelIdx,
                                    GridBatchData::Accessor gridAccessor,
                                    TorchAccessor<int32_t, 3> bboxes,
                                    TorchAccessor<bool, 1> outGridBoundsMask) {
    const nanovdb::CoordBBox maskBbox(
        nanovdb::Coord(bboxes[batchIdx][0][0], bboxes[batchIdx][0][1], bboxes[batchIdx][0][2]),
        nanovdb::Coord(bboxes[batchIdx][1][0], bboxes[batchIdx][1][1], bboxes[batchIdx][1][2]));

    const nanovdb::OnIndexGrid *grid = gridAccessor.grid(batchIdx);
    const typename nanovdb::OnIndexGrid::LeafNodeType &leaf =
        grid->tree().template getFirstNode<0>()[leafIdx];
    if (maskBbox.hasOverlap(leaf.bbox())) {
        const nanovdb::Coord ijk = leaf.offsetToGlobalCoord(voxelIdx);
        if (leaf.isActive(voxelIdx) && maskBbox.isInside(ijk)) {
            const int64_t baseOffset = gridAccessor.voxelOffset(batchIdx);
            const int64_t idx        = baseOffset + (int64_t)leaf.getValue(voxelIdx) - 1;
            outGridBoundsMask[idx]   = true;
        }
    }
}

/// @brief Get a boolean mask of the active grid voxels for a batch of grids  (including disabled
/// coordinates in mutable grids)
/// @param gridBatch The batch of grids
/// @param batchBboxes The batch of bounding boxes
/// @param outGridCoords Tensor which will contain the output grid coordinates
template <torch::DeviceType DeviceTag>
void
GetActiveVoxelsInBoundsMask(const GridBatchData &gridBatch,
                            torch::Tensor &batchBboxes,
                            torch::Tensor &outGridBoundsMask) {
    auto outMaskAcc = tensorAccessor<DeviceTag, bool, 1>(outGridBoundsMask);
    auto bboxAcc    = tensorAccessor<DeviceTag, int32_t, 3>(batchBboxes);

    if constexpr (DeviceTag == torch::kCUDA) {
        auto cb = [=] __device__(int32_t batchIdx,
                                 int32_t leafIdx,
                                 int32_t voxelIdx,
                                 int32_t,
                                 GridBatchData::Accessor gridAccessor) {
            activeGridVoxelInBoundsMaskCallback<TorchRAcc64>(
                batchIdx, leafIdx, voxelIdx, gridAccessor, bboxAcc, outMaskAcc);
        };
        forEachVoxelCUDA<1024>(1, gridBatch, cb);
    } else {
        auto cb = [=](int32_t batchIdx,
                      int32_t leafIdx,
                      int32_t voxelIdx,
                      int32_t,
                      GridBatchData::Accessor gridAccessor) {
            activeGridVoxelInBoundsMaskCallback<TorchAcc>(
                batchIdx, leafIdx, voxelIdx, gridAccessor, bboxAcc, outMaskAcc);
        };
        forEachVoxelCPU(1, gridBatch, cb);
    }
}

template <torch::DeviceType DeviceTag>
JaggedTensor
ActiveVoxelsInBoundsMask(const GridBatchData &batchHdl,
                         const std::vector<nanovdb::Coord> &bboxMins,
                         const std::vector<nanovdb::Coord> &bboxMaxs) {
    batchHdl.checkNonEmptyGrid();

    // output storage
    auto opts = torch::TensorOptions().dtype(torch::kBool).device(batchHdl.device());
    torch::Tensor outGridBoundsMask = torch::zeros({batchHdl.totalVoxels()}, opts);

    torch::Tensor batchBboxes =
        torch::empty({batchHdl.batchSize(), 2, 3},
                     torch::TensorOptions().dtype(torch::kInt32).device(batchHdl.device()));

    for (int64_t batchIdx = 0; batchIdx < batchHdl.batchSize(); batchIdx++) {
        for (int32_t dimIdx = 0; dimIdx < 3; dimIdx++) {
            batchBboxes[batchIdx][0][dimIdx] = bboxMins[batchIdx][dimIdx];
            batchBboxes[batchIdx][1][dimIdx] = bboxMaxs[batchIdx][dimIdx];
        }
    }

    // create boolean mask of active voxels
    GetActiveVoxelsInBoundsMask<DeviceTag>(batchHdl, batchBboxes, outGridBoundsMask);

    return batchHdl.jaggedTensor(outGridBoundsMask);
}

JaggedTensor
activeVoxelsInBoundsMask(const GridBatchData &batchHdl,
                         const std::vector<nanovdb::Coord> &bboxMins,
                         const std::vector<nanovdb::Coord> &bboxMaxs) {
    return FVDB_DISPATCH_KERNEL_DEVICE(batchHdl.device(), [&]() {
        return ActiveVoxelsInBoundsMask<DeviceTag>(batchHdl, bboxMins, bboxMaxs);
    });
}

} // namespace ops
} // namespace detail
} // namespace fvdb
