// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/BuilderResource.h>
#include <fvdb/GridStorage.h>
#include <fvdb/detail/GridBatchDataFactory.h>
#include <fvdb/detail/ops/BuildMergedGrids.h>
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/cuda/StreamOrdering.h>
#include <fvdb/detail/utils/nanovdb/DeviceGridHandleUtils.cuh>

#include <nanovdb/HostBuffer.h>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/GridBuilder.h>
#include <nanovdb/tools/cuda/MergeGrids.cuh>
#include <nanovdb/util/MorphologyHelpers.h>

#include <c10/cuda/CUDAGuard.h>

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

    // The grids are built and their storage retained on the device's current torch stream; the
    // prototype carries that stream and the device into every allocation the builder makes.
    const cudaStream_t stream    = storageStream(device);
    const DeviceGridBuffer proto = GridStorage::deviceProto(device, stream);

    // Build one grid per batch item, then lay them end to end in one storage.
    std::vector<GridStorage> parts;
    parts.reserve(gridBatch1.batchSize());
    for (int i = 0; i < gridBatch1.batchSize(); i += 1) {
        // Logical grids by byte offset: correct for sliced views, unlike deviceGrid<>(i).
        nanovdb::OnIndexGrid *grid1 = gridBatch1.deviceGridPtrAt(i);
        TORCH_CHECK(grid1, "First Grid is null");
        nanovdb::OnIndexGrid *grid2 = gridBatch2.deviceGridPtrAt(i);
        TORCH_CHECK(grid2, "Second Grid is null");

        nanovdb::tools::cuda::MergeGrids<nanovdb::ValueOnIndex, BuilderResource> mergeOp(
            grid1, grid2, stream);
        mergeOp.setVerbose(0);

        GridStorage::DeviceHandle handle = mergeOp.getHandle(proto);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        parts.emplace_back(std::move(handle), device);
    }

    return mergeGridStorages(std::move(parts), device, stream);
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
    std::vector<nanovdb::Vec3d> voxS, voxO;
    gridBatch1.gridVoxelSizesAndOrigins(voxS, voxO);
    GridStorage storage = FVDB_DISPATCH_KERNEL_DEVICE(gridBatch1.device(), [&]() {
        return dispatchMergeGrids<DeviceTag>(gridBatch1, gridBatch2);
    });
    return makeGridBatchData(std::move(storage), voxS, voxO);
}

} // namespace fvdb::detail::ops
