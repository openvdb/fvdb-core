// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/TorchDeviceBuffer.h>
#include <fvdb/detail/ops/BuildDilatedGrid.h>
#include <fvdb/detail/utils/Utils.h>

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
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchDilateGrid(const GridBatchData &gridBatch, const std::vector<int64_t> &dilationAmount);

template <>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchDilateGrid<torch::kCUDA>(const GridBatchData &gridBatch,
                                 const std::vector<int64_t> &dilationAmount) {
    c10::cuda::CUDAGuard deviceGuard(gridBatch.device());

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(gridBatch.device().index());

    // This guide buffer is a hack to pass in a device with an index to the cudaCreateNanoGrid
    // function. We can't pass in a device directly but we can pass in a buffer which gets
    // passed to TorchDeviceBuffer::create. The guide buffer holds the device and effectively
    // passes it to the created buffer.
    TorchDeviceBuffer guide(0, gridBatch.device());

    // Create a grid for each batch item and store the handles
    std::vector<nanovdb::GridHandle<TorchDeviceBuffer>> handles;
    for (int i = 0; i < gridBatch.batchSize(); i += 1) {
        nanovdb::GridHandle<TorchDeviceBuffer> handle;

        if (dilationAmount[i] == 0) {
            const auto &srcHdl = gridBatch.nanoGridHandle();
            uint8_t *srcData   = srcHdl.buffer().deviceData();
            for (int g = 0; g < i; ++g) {
                srcData += srcHdl.gridSize(g);
            }
            const uint64_t gridBytes = srcHdl.gridSize(i);
            TorchDeviceBuffer dstBuf(gridBytes, gridBatch.device());
            cudaMemcpyAsync(
                dstBuf.deviceData(), srcData, gridBytes, cudaMemcpyDeviceToDevice, stream.stream());
            handle = nanovdb::GridHandle<TorchDeviceBuffer>(std::move(dstBuf));
        } else {
            nanovdb::OnIndexGrid *grid =
                gridBatch.nanoGridHandleMut().deviceGrid<nanovdb::ValueOnIndex>(i);
            TORCH_CHECK(grid, "Grid is null");

            for (auto j = 0; j < dilationAmount[i]; j += 1) {
                nanovdb::tools::cuda::DilateGrid<nanovdb::ValueOnIndex> dilateOp(grid, stream);
                dilateOp.setOperation(nanovdb::tools::morphology::NN_FACE_EDGE_VERTEX);
                dilateOp.setChecksum(nanovdb::CheckMode::Default);
                dilateOp.setVerbose(0);

                handle = dilateOp.getHandle(guide);
                C10_CUDA_KERNEL_LAUNCH_CHECK();

                grid = handle.deviceGrid<nanovdb::ValueOnIndex>();
            }
        }

        handles.push_back(std::move(handle));
    }

    if (handles.size() == 1) {
        // If there's only one handle, just return it
        return std::move(handles[0]);
    } else {
        // This copies all the handles into a single handle -- only do it if there are multiple
        // grids
        return nanovdb::cuda::mergeGridHandles(handles, &guide);
    }
}

template <>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchDilateGrid<torch::kCPU>(const GridBatchData &gridBatch,
                                const std::vector<int64_t> &dilationAmount) {
    using GridT     = nanovdb::ValueOnIndex;
    using IndexTree = nanovdb::NanoTree<GridT>;

    const nanovdb::GridHandle<TorchDeviceBuffer> &gridHdl = gridBatch.nanoGridHandle();

    std::vector<nanovdb::GridHandle<TorchDeviceBuffer>> gridHandles;
    gridHandles.reserve(gridHdl.gridCount());
    for (uint32_t bidx = 0; bidx < gridHdl.gridCount(); bidx += 1) {
        const nanovdb::OnIndexGrid *grid = gridHdl.template grid<GridT>(bidx);
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
        auto ret = nanovdb::tools::createNanoGrid<ProxyGridT, GridT, TorchDeviceBuffer>(
            *proxyGrid, 0u, false, false);
        ret.buffer().to(torch::kCPU);
        gridHandles.push_back(std::move(ret));
    }

    if (gridHandles.size() == 1) {
        return std::move(gridHandles[0]);
    } else {
        return nanovdb::mergeGrids(gridHandles);
    }
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
        return gridBatch.clone(gridBatch.device());
    }

    std::vector<nanovdb::Vec3d> voxS, voxO;
    gridBatch.gridVoxelSizesAndOrigins(voxS, voxO);
    auto hdl = FVDB_DISPATCH_KERNEL_DEVICE(gridBatch.device(), [&]() {
        return dispatchDilateGrid<DeviceTag>(gridBatch, dilationAmount);
    });
    return c10::make_intrusive<GridBatchData>(std::move(hdl), voxS, voxO);
}

} // namespace fvdb::detail::ops
