// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/GridBatchImpl.h>
#include <fvdb/detail/TorchDeviceBuffer.h>
#include <fvdb/detail/ops/Ops.h>
#include <fvdb/detail/utils/nanovdb/CreateEmptyGridHandle.h>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/GridBuilder.h>
#include <nanovdb/tools/cuda/PruneGrid.cuh>
#include <nanovdb/util/MorphologyHelpers.h>
#include <nanovdb/util/cuda/Injection.cuh>
#include <nanovdb/util/cuda/Util.h>

#include <ATen/core/TensorBody.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>

namespace fvdb::detail::ops {

template <>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchPruneGrid<torch::kCUDA>(const GridBatchImpl &gridBatch, const JaggedTensor &mask) {
    c10::cuda::CUDAGuard deviceGuard(gridBatch.device());

    TORCH_CHECK_VALUE(mask.rdim() == 1, "Mask must be a one-dimensional boolean tensor");
    TORCH_CHECK_VALUE(mask.scalar_type() == torch::kBool, "Mask must be a boolean tensor");
    TORCH_CHECK_VALUE(gridBatch.device() == mask.device(), "Grid and mask must be on same device");

    // This guide buffer is a hack to pass in a device with an index to the cudaCreateNanoGrid
    // function. We can't pass in a device directly but we can pass in a buffer which gets
    // passed to TorchDeviceBuffer::create. The guide buffer holds the device and effectively
    // passes it to the created buffer.
    TorchDeviceBuffer guide(0, gridBatch.device());

    // Create a grid for each batch item and store the handles
    std::vector<nanovdb::GridHandle<TorchDeviceBuffer>> handles;
    for (int i = 0; i < gridBatch.batchSize(); i += 1) {
        nanovdb::GridHandle<TorchDeviceBuffer> handle;
        nanovdb::OnIndexGrid *grid =
            gridBatch.nanoGridHandleMut().deviceGrid<nanovdb::ValueOnIndex>(i);
        TORCH_CHECK(grid, "Grid is null");

        const torch::Tensor maskI = mask.index(i).jdata();

        // FIXME: Handle empty case!!
        if (maskI.sum().item<int64_t>() == 0) {
            // If the mask is empty, we can just return an empty grid
            handles.push_back(std::move(createEmptyGridHandle(gridBatch.device())));
            continue;
        }

        const auto leafCount = gridBatch.numLeavesAt(i);
        TorchDeviceBuffer maskBuffer(sizeof(nanovdb::Mask<3>) * leafCount, gridBatch.device());

        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(gridBatch.device().index());
        using Op = nanovdb::util::cuda::InjectPredicateToMaskFunctor<nanovdb::ValueOnIndex, -1>;
        auto *leafMask = reinterpret_cast<nanovdb::Mask<3> *>(maskBuffer.deviceData());
        nanovdb::util::cuda::operatorKernel<Op>
            <<<leafCount, Op::MaxThreadsPerBlock, 0, stream.stream()>>>(
                grid,
                maskI.data_ptr<bool>(),
                reinterpret_cast<nanovdb::Mask<3> *>(maskBuffer.deviceData()));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        nanovdb::tools::cuda::PruneGrid<nanovdb::ValueOnIndex> pruneOp(grid, leafMask);
        pruneOp.setChecksum(nanovdb::CheckMode::Default);
        pruneOp.setVerbose(0);

        handle = pruneOp.getHandle(guide);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        grid = handle.deviceGrid<nanovdb::ValueOnIndex>();

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
dispatchPruneGrid<torch::kCPU>(const GridBatchImpl &gridBatch, const JaggedTensor &mask) {
    using GridT     = nanovdb::ValueOnIndex;
    using IndexTree = nanovdb::NanoTree<GridT>;

    TORCH_CHECK_VALUE(mask.rdim() == 1, "Mask must be a one-dimensional boolean tensor");
    TORCH_CHECK_VALUE(mask.scalar_type() == torch::kBool, "Mask must be a boolean tensor");
    TORCH_CHECK_VALUE(gridBatch.device() == mask.device(), "Grid and mask must be on same device");

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

        const torch::Tensor maskI = mask.index(bidx).jdata().squeeze();
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

} // namespace fvdb::detail::ops
