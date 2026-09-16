// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/BuilderResource.h>
#include <fvdb/GridStorage.h>
#include <fvdb/detail/GridBatchDataFactory.h>
#include <fvdb/detail/ops/BuildDilatedGrid.h>
#include <fvdb/detail/ops/CloneGrid.h>
#include <fvdb/detail/ops/MakeContiguous.h>
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/cuda/StreamOrdering.h>
#include <fvdb/detail/utils/nanovdb/DeviceGridHandleUtils.cuh>

#include <nanovdb/HostBuffer.h>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/GridBuilder.h>
#include <nanovdb/tools/cuda/DilateGrid.cuh>
#include <nanovdb/util/MorphologyHelpers.h>

#include <c10/cuda/CUDAGuard.h>
#include <torch/types.h>

#include <algorithm>

namespace fvdb::detail::ops {

template <torch::DeviceType>
GridStorage dispatchDilateGrid(const GridBatchData &gridBatch,
                               const std::vector<int64_t> &dilationAmount);

template <>
GridStorage
dispatchDilateGrid<torch::kCUDA>(const GridBatchData &gridBatch,
                                 const std::vector<int64_t> &dilationAmount) {
    const torch::Device device = gridBatch.device();
    c10::cuda::CUDAGuard deviceGuard(device);

    // The grids are built and their storage retained on the device's current torch stream; the
    // prototype carries that stream and the device into every allocation the builder makes.
    const cudaStream_t stream    = storageStream(device);
    const DeviceGridBuffer proto = GridStorage::deviceProto(device, stream);

    // Build one grid per batch item, then lay them end to end in one storage. (An all-zero
    // dilation is handled upstream in dilateGrid() via cloneGrid, so the 0-branch below only
    // fires for a mixed batch.)
    std::vector<GridStorage> parts;
    parts.reserve(gridBatch.batchSize());
    for (int i = 0; i < gridBatch.batchSize(); i += 1) {
        if (dilationAmount[i] == 0) {
            // 0-dilation item in a mixed batch: clone logical grid i by byte offset (correct for
            // sliced views)
            parts.emplace_back(ops::cloneGridStorageAt(gridBatch, i));
            continue;
        }

        nanovdb::OnIndexGrid *grid = gridBatch.deviceGridPtrAt(i);
        TORCH_CHECK(grid, "Grid is null");

        // Each pass dilates the previous pass's grid; only the last one is kept.
        GridStorage::DeviceHandle handle;
        for (auto j = 0; j < dilationAmount[i]; j += 1) {
            nanovdb::tools::cuda::DilateGrid<nanovdb::ValueOnIndex, BuilderResource> dilateOp(
                grid, stream);
            dilateOp.setOperation(nanovdb::tools::morphology::NN_FACE_EDGE_VERTEX);
            dilateOp.setVerbose(0);

            handle = dilateOp.getHandle(proto);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            grid = handle.deviceGrid<nanovdb::ValueOnIndex>();
        }

        parts.emplace_back(std::move(handle), device);
    }

    return mergeGridStorages(std::move(parts), device, stream);
}

template <>
GridStorage
dispatchDilateGrid<torch::kCPU>(const GridBatchData &gridBatch,
                                const std::vector<int64_t> &dilationAmount) {
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

        const int64_t joffset = gridBatch.cumVoxelsAt(bidx);
        for (auto it = ActiveVoxelIterator<-1>(tree); it.isValid(); it++) {
            const nanovdb::Coord baseIjk = it->first;
            const int64_t dilation       = dilationAmount[bidx];
            for (auto i = -dilation; i <= dilation; i += 1) {
                for (auto j = -dilation; j <= dilation; j += 1) {
                    for (auto k = -dilation; k <= dilation; k += 1) {
                        const nanovdb::Coord fineIjk = baseIjk + nanovdb::Coord(i, j, k);
                        proxyGridAccessor.setValue(fineIjk, 1);
                    }
                }
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
dilateGrid(const GridBatchData &gridBatch, const std::vector<int64_t> &dilationAmount) {
    TORCH_CHECK_VALUE(static_cast<int64_t>(dilationAmount.size()) == gridBatch.batchSize(),
                      "dilationAmount should have same size as batch size, got ",
                      dilationAmount.size(),
                      " != ",
                      gridBatch.batchSize());
    TORCH_CHECK_VALUE(std::all_of(dilationAmount.begin(),
                                  dilationAmount.end(),
                                  [](int64_t amount) { return amount >= 0; }),
                      "dilation amount must be non-negative.");

    if (std::all_of(dilationAmount.begin(), dilationAmount.end(), [](int64_t amount) {
            return amount == 0;
        })) {
        return ops::cloneGrid(gridBatch, gridBatch.device());
    }

    std::vector<nanovdb::Vec3d> voxS, voxO;
    gridBatch.gridVoxelSizesAndOrigins(voxS, voxO);
    GridStorage storage = FVDB_DISPATCH_KERNEL_DEVICE(gridBatch.device(), [&]() {
        return dispatchDilateGrid<DeviceTag>(gridBatch, dilationAmount);
    });
    return makeGridBatchData(std::move(storage), voxS, voxO);
}

} // namespace fvdb::detail::ops
