// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/BuilderResource.h>
#include <fvdb/GridBatchData.h>
#include <fvdb/GridStorage.h>
#include <fvdb/detail/GridBatchDataFactory.h>
#include <fvdb/detail/ops/BuildCoarseGridFromFine.h>
#include <fvdb/detail/ops/BuildGridFromIjk.h>
#include <fvdb/detail/ops/CoarseIjkForFineGrid.h>
#include <fvdb/detail/ops/MakeContiguous.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/VoxelSizeUtils.h>
#include <fvdb/detail/utils/cuda/StreamOrdering.h>
#include <fvdb/detail/utils/nanovdb/BatchedTopologyBuilder.cuh>

#include <nanovdb/HostBuffer.h>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/GridBuilder.h>

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
GridStorage dispatchBuildCoarseGridFromFine(const GridBatchData &fineGridBatch,
                                            const nanovdb::Coord branchingFactor);

GridStorage
coarseGridStorageFromFineCUDA(const GridBatchData &fineGridBatch,
                              const nanovdb::Coord &branchingFactor) {
    // fvdb coarsening maps fine voxel f to floor(f / factor); a factor-2 coarsen pass maps f to
    // floor(f / 2) (coarsenCoord is exactly floor(n/2) for all n, and each 2^3 fine block is
    // unioned). So a uniform power-of-two factor is that many batched leaf-mask coarsen passes
    // over the whole batch (BatchedTopologyBuilder) -- no coordinate list, no per-member builds.
    // Non-power-of-two / non-uniform factors keep the coordinate path.
    const int nPasses = uniformPowerOfTwoLog2(branchingFactor);
    if (nPasses < 0) {
        JaggedTensor coords = ops::coarseIJKForFineGrid(fineGridBatch, branchingFactor);
        return ops::_createNanoGridFromIJK(coords);
    }

    c10::cuda::CUDAGuard deviceGuard(fineGridBatch.device());
    const cudaStream_t stream = storageStream(fineGridBatch.device());

    if (nPasses == 0) {
        // Coarsening factor 1 is the identity: the coarse grid == the fine grid. Compact the
        // (possibly sliced) selected grids into fresh contiguous storage.
        return ops::contiguousGridStorage(fineGridBatch);
    }

    // All batch members are coarsened together, one batched pass per factor of 2: a single output
    // storage, one stream synchronization per pass, no per-member builds or storage merging
    // (issue #755). Empty members become valid empty grids inline.
    const std::vector<batched::TopologyPassSpec> passes(nPasses,
                                                        batched::TopologyPassSpec::coarsen());
    return batched::batchedTopologyStorage(fineGridBatch, passes, stream);
}

template <>
GridStorage
dispatchBuildCoarseGridFromFine<torch::kCUDA>(const GridBatchData &fineGridBatch,
                                              const nanovdb::Coord branchingFactor) {
    return coarseGridStorageFromFineCUDA(fineGridBatch, branchingFactor);
}

template <>
GridStorage
dispatchBuildCoarseGridFromFine<torch::kPrivateUse1>(const GridBatchData &fineGridBatch,
                                                     const nanovdb::Coord branchingFactor) {
    JaggedTensor coords = ops::coarseIJKForFineGrid(fineGridBatch, branchingFactor);
    return ops::_createNanoGridFromIJK(coords);
}

template <>
GridStorage
dispatchBuildCoarseGridFromFine<torch::kCPU>(const GridBatchData &fineBatchHdl,
                                             const nanovdb::Coord branchingFactor) {
    using GridT     = nanovdb::ValueOnIndex;
    using IndexTree = nanovdb::NanoTree<GridT>;

    std::vector<nanovdb::GridHandle<nanovdb::HostBuffer>> batchHandles;
    batchHandles.reserve(fineBatchHdl.batchSize());
    for (int64_t bidx = 0; bidx < fineBatchHdl.batchSize(); bidx += 1) {
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
        batchHandles.push_back(
            nanovdb::tools::createNanoGrid<ProxyGridT, GridT, nanovdb::HostBuffer>(
                *proxyGrid, 0u, false, false));
    }

    return GridStorage(batchHandles.size() == 1 ? std::move(batchHandles[0])
                                                : nanovdb::mergeGrids(batchHandles));
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
