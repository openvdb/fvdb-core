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
#include <fvdb/detail/utils/nanovdb/CreateEmptyGridStorage.h>
#include <fvdb/detail/utils/nanovdb/DeviceGridHandleUtils.cuh>

#include <nanovdb/HostBuffer.h>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/cuda/Buffer.h>
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

template <torch::DeviceType>
GridStorage dispatchPruneGrid(const GridBatchData &gridBatch, const JaggedTensor &mask);

template <>
GridStorage
dispatchPruneGrid<torch::kCUDA>(const GridBatchData &gridBatch, const JaggedTensor &mask) {
    const torch::Device device = gridBatch.device();
    c10::cuda::CUDAGuard deviceGuard(device);

    TORCH_CHECK_VALUE(mask.rdim() == 1, "Mask must be a one-dimensional boolean tensor");
    TORCH_CHECK_VALUE(mask.scalar_type() == torch::kBool, "Mask must be a boolean tensor");
    TORCH_CHECK_VALUE(gridBatch.device() == mask.device(), "Grid and mask must be on same device");

    // The grids are built and their storage retained on the device's current torch stream; the
    // prototype carries that stream and the device into every allocation the builder makes.
    const cudaStream_t stream    = storageStream(device);
    const DeviceGridBuffer proto = GridStorage::deviceProto(device, stream);

    // Build one grid per batch item, then lay them end to end in one storage.
    std::vector<GridStorage> parts;
    parts.reserve(gridBatch.batchSize());
    for (int i = 0; i < gridBatch.batchSize(); i += 1) {
        // This also keeps the grid aligned with numLeavesAt(i)/mask.index(i), which are
        // item-indexed.
        nanovdb::OnIndexGrid *grid = gridBatch.deviceGridPtrAt(i);
        TORCH_CHECK(grid, "Grid is null");

        const torch::Tensor maskI = mask.index(i).jdata();

        // FIXME: Handle empty case!!
        if (maskI.sum().item<int64_t>() == 0) {
            // If the mask is empty, we contribute a voxel-less grid
            parts.emplace_back(createEmptyGridStorage(device));
            continue;
        }

        const auto leafCount = gridBatch.numLeavesAt(i);

        // Per-leaf keep masks, one Mask<3> per source leaf. Scratch for the prune pass, so it
        // goes through the builders' resource (torch's active CUDA allocator), stream-ordered on
        // the stream the fill kernel and PruneGrid run on. Every word is written by the kernel
        // below, so the allocation is not zero-initialized.
        BuilderBuffer<nanovdb::Mask<3>> maskBuffer(stream, leafCount, nanovdb::cuda::noInit);

        using Op = nanovdb::util::cuda::InjectPredicateToMaskFunctor<nanovdb::ValueOnIndex, -1>;
        nanovdb::Mask<3> *leafMask = maskBuffer.data();
        nanovdb::util::cuda::operatorKernel<Op><<<leafCount, Op::MaxThreadsPerBlock, 0, stream>>>(
            grid, maskI.data_ptr<bool>(), leafMask);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        // PruneGrid defaults to the legacy stream 0; run it on the same stream as the mask fill
        // above, or its reads of leafMask can race the kernel that writes it.
        nanovdb::tools::cuda::PruneGrid<nanovdb::ValueOnIndex, BuilderResource> pruneOp(
            grid, leafMask, stream);
        pruneOp.setVerbose(0);

        GridStorage::DeviceHandle handle = pruneOp.getHandle(proto);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        parts.emplace_back(std::move(handle), device);
    }

    return mergeGridStorages(std::move(parts), device, stream);
}

template <>
GridStorage
dispatchPruneGrid<torch::kCPU>(const GridBatchData &gridBatch, const JaggedTensor &mask) {
    using GridT     = nanovdb::ValueOnIndex;
    using IndexTree = nanovdb::NanoTree<GridT>;

    TORCH_CHECK_VALUE(mask.rdim() == 1, "Mask must be a one-dimensional boolean tensor");
    TORCH_CHECK_VALUE(mask.scalar_type() == torch::kBool, "Mask must be a boolean tensor");
    TORCH_CHECK_VALUE(gridBatch.device() == mask.device(), "Grid and mask must be on same device");

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
    std::vector<nanovdb::Vec3d> voxS, voxO;
    gridBatch.gridVoxelSizesAndOrigins(voxS, voxO);
    GridStorage storage = FVDB_DISPATCH_KERNEL_DEVICE(
        gridBatch.device(), [&]() { return dispatchPruneGrid<DeviceTag>(gridBatch, mask); });
    return makeGridBatchData(std::move(storage), voxS, voxO);
}

} // namespace fvdb::detail::ops
