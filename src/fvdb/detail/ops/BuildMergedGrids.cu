// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/GridStorage.h>
#include <fvdb/detail/GridBatchDataFactory.h>
#include <fvdb/detail/ops/BuildMergedGrids.h>
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/cuda/StreamOrdering.h>
#include <fvdb/detail/utils/nanovdb/BatchedTopologyBuilder.cuh>

#include <nanovdb/GridHandle.h>
#include <nanovdb/HostBuffer.h>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/GridBuilder.h>

#include <c10/cuda/CUDAGuard.h>

#include <memory>
#include <utility>
#include <vector>

namespace fvdb::detail::ops {

template <torch::DeviceType>
GridStorage dispatchMergeGrids(const GridBatchData &gridBatch1, const GridBatchData &gridBatch2);

template <>
GridStorage
dispatchMergeGrids<torch::kCUDA>(const GridBatchData &gridBatch1, const GridBatchData &gridBatch2) {
    const torch::Device device = gridBatch1.device();
    c10::cuda::CUDAGuard deviceGuard(device);
    TORCH_CHECK_VALUE(gridBatch1.device() == gridBatch2.device(),
                      "All arguments to MergeGrids must be on the same device");
    TORCH_CHECK_VALUE(gridBatch1.batchSize() == gridBatch2.batchSize(),
                      "GridBatches to merge should have the same batch size");

    // The grids are built and their storage retained on the device's current torch stream.
    const cudaStream_t stream = storageStream(device);

    // One batched Union pass over both batches (issue #775, item 5): output grid i is the union of
    // grid i of each, with grid i of the first batch's GridData header. The sources hold each
    // logical grid by byte offset, so sliced views work.
    const batched::BatchedTopologySource first  = batched::sourceFromGridBatch(gridBatch1);
    const batched::BatchedTopologySource second = batched::sourceFromGridBatch(gridBatch2);
    batched::BatchedTopologyResult result =
        batched::runBatchedTopologyPass(first, batched::TopologyPassSpec::unite(second), stream);
    return batched::storageFromResult(std::move(result), device, stream);
}

template <>
GridStorage
dispatchMergeGrids<torch::kCPU>(const GridBatchData &gridBatch1, const GridBatchData &gridBatch2) {
    using GridT     = nanovdb::ValueOnIndex;
    using IndexTree = nanovdb::NanoTree<GridT>;
    TORCH_CHECK(gridBatch1.device().is_cpu(), "All arguments to MergeGrids must be on the CPU");
    TORCH_CHECK(gridBatch2.device().is_cpu(), "All arguments to MergeGrids must be on the CPU");
    TORCH_CHECK_VALUE(gridBatch1.batchSize() == gridBatch2.batchSize(),
                      "GridBatches to merge should have the same batch size");

    std::vector<nanovdb::GridHandle<nanovdb::HostBuffer>> gridHandles;
    gridHandles.reserve(gridBatch1.batchSize());
    for (int64_t bidx = 0; bidx < gridBatch1.batchSize(); bidx += 1) {
        // Logical grids by byte offset: correct for sliced views, unlike grid<GridT>(bidx).
        const nanovdb::OnIndexGrid *grid1 = gridBatch1.hostGridPtrAt(bidx);
        const nanovdb::OnIndexGrid *grid2 = gridBatch2.hostGridPtrAt(bidx);
        TORCH_CHECK(grid1, "Failed to get pointer to nanovdb index grid (first argument to merge)");
        TORCH_CHECK(grid2,
                    "Failed to get pointer to nanovdb index grid (second argument to merge)");
        const IndexTree &tree1 = grid1->tree();
        const IndexTree &tree2 = grid2->tree();

        using ProxyGridT       = nanovdb::tools::build::Grid<float>;
        auto proxyGrid         = std::make_shared<ProxyGridT>(-1.0f);
        auto proxyGridAccessor = proxyGrid->getWriteAccessor();

        const int64_t joffset = gridBatch1.cumVoxelsAt(bidx);
        for (auto it = ActiveVoxelIterator<-1>(tree1); it.isValid(); it++) {
            proxyGridAccessor.setValue(it->first, 1);
        }
        for (auto it = ActiveVoxelIterator<-1>(tree2); it.isValid(); it++) {
            proxyGridAccessor.setValue(it->first, 1);
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
mergeGrids(const GridBatchData &gridBatch1, const GridBatchData &gridBatch2) {
    TORCH_CHECK_VALUE(gridBatch1.batchSize() == gridBatch2.batchSize(),
                      "GridBatches to merge should have same batch size");
    TORCH_CHECK_VALUE(gridBatch1.device() == gridBatch2.device(),
                      "GridBatches to merge should be on same device/host");
    if (gridBatch1.batchSize() == 0) {
        return makeEmptyGridBatchData(gridBatch1.device());
    }
    std::vector<nanovdb::Vec3d> voxS, voxO;
    gridBatch1.gridVoxelSizesAndOrigins(voxS, voxO);
    GridStorage storage = FVDB_DISPATCH_KERNEL_DEVICE(gridBatch1.device(), [&]() {
        return dispatchMergeGrids<DeviceTag>(gridBatch1, gridBatch2);
    });
    return makeGridBatchData(std::move(storage), voxS, voxO);
}

} // namespace fvdb::detail::ops
