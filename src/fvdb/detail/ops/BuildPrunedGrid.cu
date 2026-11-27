// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/BuilderResource.h>
#include <fvdb/GridBatchData.h>
#include <fvdb/GridStorage.h>
#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/GridBatchDataFactory.h>
#include <fvdb/detail/ops/BuildPrunedGrid.h>
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/cuda/StreamOrdering.h>
#include <fvdb/detail/utils/nanovdb/BatchedTopologyBuilder.cuh>

#include <nanovdb/HostBuffer.h>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/cuda/Buffer.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/GridBuilder.h>

#include <ATen/core/TensorBody.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace fvdb::detail::ops {

template <torch::DeviceType>
GridStorage dispatchPruneGrid(const GridBatchData &gridBatch, const JaggedTensor &mask);

namespace {

/// One thread per (batch leaf, mask word): the word's active voxels whose `keep` flag is set. The
/// keep flags are in batch voxel order: voxel v (1-based value index) of grid g is
/// keep[voxelStart[g] + v - 1].
__global__ void
fillKeepMasks(const nanovdb::OnIndexGrid *const *grids,
              const int64_t *leafStart, // [numGrids + 1]
              const int64_t *voxelStart,
              int32_t numGrids,
              int64_t numItems,
              const bool *keep,
              batched::LeafMaskT *keepMasks) {
    constexpr uint32_t kBitsPerWord = batched::LeafMaskT::SIZE / batched::kLeafMaskWords;
    for (int64_t item = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; item < numItems;
         item += int64_t(gridDim.x) * blockDim.x) {
        const int64_t leaf = item / batched::kLeafMaskWords;
        const uint32_t w   = uint32_t(item % batched::kLeafMaskWords);
        const int32_t g    = batched::gridOfLeaf(leafStart, numGrids, leaf);
        const batched::LeafT &src =
            grids[g]->tree().template getFirstNode<0>()[leaf - leafStart[g]];
        uint64_t active = src.valueMask().words()[w];
        uint64_t kept   = 0;
        while (active) {
            const uint32_t bit = uint32_t(__ffsll(static_cast<long long>(active)) - 1);
            active &= active - 1;
            const uint64_t value = src.getValue(w * kBitsPerWord + bit); // 1-based in grid g
            if (keep[voxelStart[g] + int64_t(value) - 1]) {
                kept |= uint64_t(1) << bit;
            }
        }
        keepMasks[leaf].words()[w] = kept;
    }
}

} // namespace

template <>
GridStorage
dispatchPruneGrid<torch::kCUDA>(const GridBatchData &gridBatch, const JaggedTensor &mask) {
    const torch::Device device = gridBatch.device();
    c10::cuda::CUDAGuard deviceGuard(device);

    // The grids are built and their storage retained on the device's current torch stream.
    const cudaStream_t stream = storageStream(device);
    const int64_t batchSize   = gridBatch.batchSize();

    // Per-leaf keep masks for the whole batch, one leaf mask (batched::LeafMaskT) per source leaf
    // in batch order, from the per-voxel mask in one launch. They are scratch for the prune pass,
    // so they go through the builders' resource (torch's active CUDA allocator), stream-ordered on
    // the pass's stream.
    const batched::BatchedTopologySource source = batched::sourceFromGridBatch(gridBatch);
    const batched::SourceLeafTables tables      = batched::sourceLeafTables(source, stream);
    BuilderBuffer<batched::LeafMaskT> keepMasks(
        stream, std::max<int64_t>(tables.numLeaves(), 1), nanovdb::cuda::noInit);
    if (tables.numLeaves() > 0) {
        std::vector<int64_t> voxelStart(batchSize);
        for (int64_t i = 0; i < batchSize; ++i) {
            voxelStart[i] = gridBatch.cumVoxelsAt(i);
        }
        const torch::Tensor maskData      = mask.jdata().contiguous();
        const torch::Tensor voxelStartDev = batched::uploadTable(voxelStart, device, stream);
        const int64_t numItems            = tables.numLeaves() * batched::kLeafMaskWords;
        fillKeepMasks<<<batched::itemBlocks(numItems), batched::kThreads, 0, stream>>>(
            tables.gridPtrs(),
            tables.leafStartPtr(),
            voxelStartDev.data_ptr<int64_t>(),
            tables.numGrids(),
            numItems,
            maskData.data_ptr<bool>(),
            keepMasks.data());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // One batched prune pass over the whole batch (issue #775, item 10). A member whose mask keeps
    // nothing becomes a valid empty grid inline.
    const std::vector<batched::TopologyPassSpec> passes{
        batched::TopologyPassSpec::prune(keepMasks.data(), tables.numLeaves())};
    return batched::batchedTopologyStorage(gridBatch, passes, stream);
}

template <>
GridStorage
dispatchPruneGrid<torch::kCPU>(const GridBatchData &gridBatch, const JaggedTensor &mask) {
    using GridT     = nanovdb::ValueOnIndex;
    using IndexTree = nanovdb::NanoTree<GridT>;

    std::vector<nanovdb::GridHandle<nanovdb::HostBuffer>> gridHandles;
    gridHandles.reserve(gridBatch.batchSize());
    for (int64_t bidx = 0; bidx < gridBatch.batchSize(); bidx += 1) {
        const nanovdb::OnIndexGrid *grid = gridBatch.hostGridPtrAt(bidx);
        if (!grid) {
            throw std::runtime_error("Failed to get pointer to nanovdb index grid");
        }
        const IndexTree &tree = grid->tree();

        using ProxyGridT       = nanovdb::tools::build::Grid<float>;
        auto proxyGrid         = std::make_shared<ProxyGridT>(-1.0f);
        auto proxyGridAccessor = proxyGrid->getWriteAccessor();

        const torch::Tensor maskI = mask.index(bidx).jdata().reshape({-1});
        const int64_t joffset     = gridBatch.cumVoxelsAt(bidx);
        const auto maskIacc       = maskI.accessor<bool, 1>();
        for (auto it = ActiveVoxelIterator<-1>(tree); it.isValid(); it++) {
            const nanovdb::Coord baseIjk = it->first;
            const auto index             = it->second;
            if (maskIacc[index]) {
                proxyGridAccessor.setValue(baseIjk, 1);
            }
        }

        proxyGridAccessor.merge();
        gridHandles.push_back(
            nanovdb::tools::createNanoGrid<ProxyGridT, GridT, nanovdb::HostBuffer>(
                *proxyGrid, 0u, false, false));
    }

    return GridStorage(gridHandles.size() == 1 ? std::move(gridHandles[0])
                                               : nanovdb::mergeGrids(gridHandles));
}

c10::intrusive_ptr<GridBatchData>
pruneGrid(const GridBatchData &gridBatch, const JaggedTensor &mask) {
    TORCH_CHECK_VALUE(mask.ldim() == 1, "Mask should be a list of tensors");
    TORCH_CHECK_VALUE(gridBatch.batchSize() == mask.num_tensors(),
                      "Cardinality of masks should match gridbatch size");
    TORCH_CHECK_VALUE(gridBatch.device() == mask.device(),
                      "GridBatch and mask should be on same device/host");
    TORCH_CHECK_VALUE(mask.rdim() == 1, "Mask must be a one-dimensional boolean tensor");
    TORCH_CHECK_VALUE(mask.scalar_type() == torch::kBool, "Mask must be a boolean tensor");
    // An empty batch prunes to an empty batch on every device, after the argument checks above
    // (matching buildPaddedGrid).
    if (gridBatch.batchSize() == 0) {
        return makeEmptyGridBatchData(gridBatch.device());
    }
    // One mask entry per voxel of each grid: the dispatches read grid b's flags at its voxel
    // offset, so a mask with the right total but a different per-grid split would prune the wrong
    // voxels, and a short one would be read past its end. On CUDA this compare waits on the device.
    TORCH_CHECK_VALUE(mask.joffsets().numel() == gridBatch.voxelOffsets().numel() &&
                          torch::equal(mask.joffsets(), gridBatch.voxelOffsets()),
                      "Mask must have one entry per voxel of each grid");
    std::vector<nanovdb::Vec3d> voxS, voxO;
    gridBatch.gridVoxelSizesAndOrigins(voxS, voxO);
    GridStorage storage = FVDB_DISPATCH_KERNEL_DEVICE(
        gridBatch.device(), [&]() { return dispatchPruneGrid<DeviceTag>(gridBatch, mask); });
    return makeGridBatchData(std::move(storage), voxS, voxO);
}

} // namespace fvdb::detail::ops
