// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/GridBatchData.h>
#include <fvdb/detail/GridBatchDataFactory.h>
#include <fvdb/detail/ops/BuildCoarseGridFromFine.h>
#include <fvdb/detail/utils/VoxelSizeUtils.h>
#include <fvdb/detail/ops/BuildGridFromIjk.h>
#include <fvdb/detail/ops/CoarseIjkForFineGrid.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/Utils.h>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/GridBuilder.h>

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <torch/types.h>

namespace fvdb::detail::ops {

template <torch::DeviceType>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchBuildCoarseGridFromFine(const GridBatchData &fineGridBatch,
                                const nanovdb::Coord branchingFactor);

template <>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchBuildCoarseGridFromFine<torch::kCUDA>(const GridBatchData &fineGridBatch,
                                              const nanovdb::Coord branchingFactor) {
    JaggedTensor coords = ops::coarseIJKForFineGrid(fineGridBatch, branchingFactor);
    return ops::_createNanoGridFromIJK(coords);
}

template <>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchBuildCoarseGridFromFine<torch::kPrivateUse1>(const GridBatchData &fineGridBatch,
                                                     const nanovdb::Coord branchingFactor) {
    JaggedTensor coords = ops::coarseIJKForFineGrid(fineGridBatch, branchingFactor);
    return ops::_createNanoGridFromIJK(coords);
}

template <>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchBuildCoarseGridFromFine<torch::kCPU>(const GridBatchData &fineBatchHdl,
                                             const nanovdb::Coord branchingFactor) {
    using GridT     = nanovdb::ValueOnIndex;
    using IndexTree = nanovdb::NanoTree<GridT>;

    const nanovdb::GridHandle<TorchDeviceBuffer> &fineGridHdl = fineBatchHdl.nanoGridHandle();

    std::vector<nanovdb::GridHandle<TorchDeviceBuffer>> batchHandles;
    batchHandles.reserve(fineGridHdl.gridCount());
    for (uint32_t bidx = 0; bidx < fineGridHdl.gridCount(); bidx += 1) {
        const nanovdb::OnIndexGrid *fineGrid = fineGridHdl.template grid<GridT>(bidx);
        if (!fineGrid) {
            throw std::runtime_error("Failed to get pointer to nanovdb index grid");
        }
        const IndexTree &fineTree = fineGrid->tree();

        using ProxyGridT       = nanovdb::tools::build::Grid<float>;
        auto proxyGrid         = std::make_shared<ProxyGridT>(-1.0f);
        auto proxyGridAccessor = proxyGrid->getWriteAccessor();

        for (auto it = ActiveVoxelIterator(fineTree); it.isValid(); it++) {
            const nanovdb::Coord coarseIjk =
                (it->first.asVec3d() / branchingFactor.asVec3d()).floor();
            proxyGridAccessor.setValue(coarseIjk, 1.0f);
        }

        proxyGridAccessor.merge();
        auto ret = nanovdb::tools::createNanoGrid<ProxyGridT, GridT, TorchDeviceBuffer>(
            *proxyGrid, 0u, false, false);
        ret.buffer().to(torch::kCPU);
        batchHandles.push_back(std::move(ret));
    }

    if (batchHandles.size() == 1) {
        return std::move(batchHandles[0]);
    } else {
        return nanovdb::mergeGrids(batchHandles);
    }
}

c10::intrusive_ptr<GridBatchData>
buildCoarseGridFromFine(const GridBatchData &fineGridBatch, const nanovdb::Coord branchingFactor) {
    for (int i = 0; i < 3; i += 1) {
        TORCH_CHECK_VALUE(branchingFactor[i] > 0,
                          "coarseningFactor must be strictly positive. Got [" +
                              std::to_string(branchingFactor[0]) + ", " +
                              std::to_string(branchingFactor[1]) + ", " +
                              std::to_string(branchingFactor[2]) + "]");
    }
    std::vector<nanovdb::Vec3d> coarseVoxS, coarseVoxO;
    coarseVoxS.reserve(fineGridBatch.batchSize());
    coarseVoxO.reserve(fineGridBatch.batchSize());
    for (int64_t i = 0; i < fineGridBatch.batchSize(); ++i) {
        coarseVoxS.push_back(coarseVoxelSize(fineGridBatch.voxelSizeAt(i), branchingFactor));
        coarseVoxO.push_back(coarseVoxelOrigin(
            fineGridBatch.voxelSizeAt(i), fineGridBatch.voxelOriginAt(i), branchingFactor));
    }
    auto hdl = FVDB_DISPATCH_KERNEL(fineGridBatch.device(), [&]() {
        return dispatchBuildCoarseGridFromFine<DeviceTag>(fineGridBatch, branchingFactor);
    });
    return makeGridBatchData(std::move(hdl), coarseVoxS, coarseVoxO);
}

} // namespace fvdb::detail::ops
