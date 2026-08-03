// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/GridBatchData.h>
#include <fvdb/detail/GridBatchDataFactory.h>
#include <fvdb/detail/ops/BuildCoarseGridFromFine.h>
#include <fvdb/detail/ops/BuildGridFromIjk.h>
#include <fvdb/detail/ops/CoarseIjkForFineGrid.h>
#include <fvdb/detail/ops/MakeContiguous.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/VoxelSizeUtils.h>
#include <fvdb/detail/utils/nanovdb/CreateEmptyGridHandle.h>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/GridBuilder.h>
#include <nanovdb/tools/cuda/CoarsenGrid.cuh>

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <torch/types.h>

namespace fvdb::detail::ops {

// If `factor` is a uniform power of two (2,4,8,... in every axis), returns log2(factor) -- the
// number of unit CoarsenGrid passes needed. Returns -1 otherwise (non-uniform or non-power-of-two
// factors have no leaf-mask coarsening tool and keep the coordinate-list path). Factor 1 -> 0.
static int
uniformPowerOfTwoLog2(const nanovdb::Coord &factor) {
    if (factor[0] != factor[1] || factor[1] != factor[2] || factor[0] < 1) {
        return -1;
    }
    int v = factor[0];
    if ((v & (v - 1)) != 0) {
        return -1; // not a power of two
    }
    int log2 = 0;
    while (v > 1) {
        v >>= 1;
        log2 += 1;
    }
    return log2;
}

template <torch::DeviceType>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchBuildCoarseGridFromFine(const GridBatchData &fineGridBatch,
                                const nanovdb::Coord branchingFactor);

nanovdb::GridHandle<TorchDeviceBuffer>
coarseGridHandleFromFineCUDA(const GridBatchData &fineGridBatch,
                             const nanovdb::Coord &branchingFactor) {
    // fvdb coarsening maps fine voxel f to floor(f / factor); NanoVDB's CoarsenGrid maps f to
    // floor(f / 2) per pass (its coarsenComponent is exactly floor(n/2) for all n, and it unions
    // each 2^3 fine block). So a uniform power-of-two factor is that many CoarsenGrid passes -- no
    // coordinate list, no radix sort. Non-power-of-two / non-uniform factors keep the coord path.
    const int nPasses = uniformPowerOfTwoLog2(branchingFactor);
    if (nPasses < 0) {
        JaggedTensor coords = ops::coarseIJKForFineGrid(fineGridBatch, branchingFactor);
        return ops::_createNanoGridFromIJK(coords);
    }

    c10::cuda::CUDAGuard deviceGuard(fineGridBatch.device());
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(fineGridBatch.device().index());
    TorchDeviceBuffer guide(0, fineGridBatch.device());

    if (nPasses == 0) {
        // Coarsening factor 1 is the identity: the coarse grid == the fine grid. Compact the
        // (possibly sliced) selected grids into a fresh contiguous handle -- a byte copy with a
        // header fixup, correct for non-contiguous views and far cheaper than a coordinate rebuild.
        return ops::contiguousGridHandle(fineGridBatch);
    }

    std::vector<nanovdb::GridHandle<TorchDeviceBuffer>> handles;
    handles.reserve(fineGridBatch.batchSize());
    for (int64_t i = 0; i < fineGridBatch.batchSize(); i += 1) {
        if (fineGridBatch.numVoxelsAt(i) == 0) {
            handles.push_back(createEmptyGridHandle(fineGridBatch.device()));
            continue;
        }
        // Byte-offset accessor: correct for sliced/non-contiguous batches (see deviceGridPtrAt).
        nanovdb::OnIndexGrid *grid = fineGridBatch.deviceGridPtrAt(i);
        TORCH_CHECK(grid, "Grid is null");
        nanovdb::GridHandle<TorchDeviceBuffer> handle;
        for (int p = 0; p < nPasses; p += 1) {
            nanovdb::tools::cuda::CoarsenGrid<nanovdb::ValueOnIndex> op(grid, stream.stream());
            op.setChecksum(nanovdb::CheckMode::Default);
            op.setVerbose(0);
            handle = op.getHandle(guide);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            grid = handle.deviceGrid<nanovdb::ValueOnIndex>();
        }
        handles.push_back(std::move(handle));
    }

    return handles.size() == 1 ? std::move(handles[0])
                               : nanovdb::cuda::mergeGridHandles(handles, &guide);
}

template <>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchBuildCoarseGridFromFine<torch::kCUDA>(const GridBatchData &fineGridBatch,
                                              const nanovdb::Coord branchingFactor) {
    return coarseGridHandleFromFineCUDA(fineGridBatch, branchingFactor);
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
    for (int64_t bidx = 0; bidx < fineBatchHdl.batchSize(); bidx += 1) {
        // Byte-offset accessor: correct for sliced/non-contiguous batches (see hostGridPtrAt).
        const nanovdb::OnIndexGrid *fineGrid = fineBatchHdl.hostGridPtrAt(bidx);
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
