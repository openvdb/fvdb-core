// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/utils/cuda/ForEachCUDA.cuh>

namespace fvdb {

namespace _private {

__global__ void
voxelMetaIndexCUDAKernel(fvdb::detail::GridBatchImpl::Accessor gridAccessor,
                         TorchRAcc32<int64_t, 2> metaIndex) {
    constexpr int32_t VOXELS_PER_LEAF = nanovdb::OnIndexTree::LeafNodeType::NUM_VALUES;
    const int64_t lvIdx               = ((int64_t)blockIdx.x * (int64_t)blockDim.x) + threadIdx.x;

    if (lvIdx >= gridAccessor.totalLeaves() * VOXELS_PER_LEAF) {
        return;
    }

    const int64_t cumLeafIdx   = (lvIdx / VOXELS_PER_LEAF);
    const int64_t leafVoxelIdx = lvIdx % VOXELS_PER_LEAF;

    const int64_t batchIdx = gridAccessor.leafBatchIndex(cumLeafIdx);
    const int64_t leafIdx  = cumLeafIdx - gridAccessor.leafOffset(batchIdx);

    const nanovdb::OnIndexGrid *grid = gridAccessor.grid(batchIdx);
    const typename nanovdb::OnIndexGrid::LeafNodeType &leaf =
        grid->tree().template getFirstNode<0>()[leafIdx];

    if (leaf.isActive(leafVoxelIdx)) {
        const int64_t baseOffset = gridAccessor.voxelOffset(batchIdx);
        const int64_t idx        = baseOffset + (int64_t)leaf.getValue(leafVoxelIdx) - 1;

        metaIndex[idx][0] = batchIdx;
        metaIndex[idx][1] = leafIdx;
        metaIndex[idx][2] = leafVoxelIdx;
    }
}

} // namespace _private

} // namespace fvdb
