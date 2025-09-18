// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/GridBatchImpl.h>
#include <fvdb/detail/ops/BuildFineGridFromCoarse.h>
#include <fvdb/detail/ops/BuildGridFromIjk.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/cuda/ForEachCUDA.cuh>
#include <fvdb/detail/utils/cuda/ForEachPrivateUse1.cuh>
#include <fvdb/detail/utils/cuda/GridDim.h>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/GridBuilder.h>

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <torch/types.h>

#include <cub/cub.cuh>

namespace fvdb::detail::ops {

__device__ inline void
copyCoords(const fvdb::JIdxType bidx,
           const int64_t base,
           const nanovdb::Coord &ijk0,
           const nanovdb::CoordBBox &bbox,
           TorchRAcc64<int32_t, 2> outIJK,
           TorchRAcc64<fvdb::JIdxType, 1> outIJKBIdx) {
    static_assert(sizeof(nanovdb::Coord) == 3 * sizeof(int32_t));
    nanovdb::Coord ijk;
    int32_t count = 0;
    for (int di = bbox.min()[0]; di <= bbox.max()[0]; di += 1) {
        for (int dj = bbox.min()[1]; dj <= bbox.max()[1]; dj += 1) {
            for (int dk = bbox.min()[2]; dk <= bbox.max()[2]; dk += 1) {
                ijk                      = ijk0 + nanovdb::Coord(di, dj, dk);
                outIJK[base + count][0]  = ijk[0];
                outIJK[base + count][1]  = ijk[1];
                outIJK[base + count][2]  = ijk[2];
                outIJKBIdx[base + count] = bidx;
                count += 1;
            }
        }
    }
}

__device__ inline void
copyCoords(const fvdb::JIdxType bidx,
           const int64_t base,
           const nanovdb::Coord size,
           const nanovdb::Coord &ijk0,
           TorchRAcc64<int32_t, 2> outIJK,
           TorchRAcc64<fvdb::JIdxType, 1> outIJKBIdx) {
    return copyCoords(bidx,
                      base,
                      ijk0,
                      nanovdb::CoordBBox(nanovdb::Coord(0), size - nanovdb::Coord(1)),
                      outIJK,
                      outIJKBIdx);
}

__device__ void
fineIjkForCoarseGridVoxelCallback(int32_t bidx,
                                  int32_t lidx,
                                  int32_t vidx,
                                  int32_t cidx,
                                  const GridBatchImpl::Accessor batchAcc,
                                  nanovdb::Coord upsamplingFactor,
                                  TorchRAcc64<int32_t, 2> outIJKData,
                                  TorchRAcc64<fvdb::JIdxType, 1> outIJKBIdx) {
    const nanovdb::OnIndexGrid *gridPtr = batchAcc.grid(bidx);
    const typename nanovdb::OnIndexGrid::LeafNodeType &leaf =
        gridPtr->tree().template getFirstNode<0>()[lidx];
    const int64_t baseOffset     = batchAcc.voxelOffset(bidx);
    const int64_t totalPadAmount = upsamplingFactor[0] * upsamplingFactor[1] * upsamplingFactor[2];
    if (leaf.isActive(vidx)) {
        const int64_t value            = ((int64_t)leaf.getValue(vidx)) - 1;
        const int64_t index            = (baseOffset + value) * totalPadAmount;
        const nanovdb::Coord coarseIjk = leaf.offsetToGlobalCoord(vidx);
        const nanovdb::Coord fineIjk(coarseIjk[0] * upsamplingFactor[0],
                                     coarseIjk[1] * upsamplingFactor[1],
                                     coarseIjk[2] * upsamplingFactor[2]);
        copyCoords(bidx, index, upsamplingFactor, fineIjk, outIJKData, outIJKBIdx);
    }
}

__device__ void
fineIjkForCoarseGridVoxelCallback(int32_t bidx,
                                  int32_t lidx,
                                  int32_t vidx,
                                  int32_t cidx,
                                  const GridBatchImpl::Accessor batchAcc,
                                  nanovdb::Coord upsamplingFactor,
                                  TorchRAcc64<int32_t, 2> outIJKData,
                                  TorchRAcc64<fvdb::JIdxType, 1> outIJKBIdx,
                                  TorchRAcc64<bool, 1> maskData,
                                  TorchRAcc64<int64_t, 1> maskPrefixSumData) {
    const nanovdb::OnIndexGrid *gridPtr = batchAcc.grid(bidx);
    const typename nanovdb::OnIndexGrid::LeafNodeType &leaf =
        gridPtr->tree().template getFirstNode<0>()[lidx];
    const int64_t baseOffset     = batchAcc.voxelOffset(bidx);
    const int64_t totalPadAmount = upsamplingFactor[0] * upsamplingFactor[1] * upsamplingFactor[2];
    if (leaf.isActive(vidx)) {
        const int64_t value = ((int64_t)leaf.getValue(vidx)) - 1;
        if (maskData[baseOffset + value]) {
            const int64_t index = (maskPrefixSumData[baseOffset + value] - 1) * totalPadAmount;
            const nanovdb::Coord coarseIjk = leaf.offsetToGlobalCoord(vidx);
            const nanovdb::Coord fineIjk(coarseIjk[0] * upsamplingFactor[0],
                                         coarseIjk[1] * upsamplingFactor[1],
                                         coarseIjk[2] * upsamplingFactor[2]);
            copyCoords(bidx, index, upsamplingFactor, fineIjk, outIJKData, outIJKBIdx);
        }
    }
}

template <>
JaggedTensor
dispatchFineIJKForCoarseGrid<torch::kCUDA>(const GridBatchImpl &batchHdl,
                                           nanovdb::Coord upsamplingFactor,
                                           const std::optional<JaggedTensor> &mask) {
    TORCH_CHECK(batchHdl.device().is_cuda(), "GridBatchImpl must be on CUDA device");
    TORCH_CHECK(batchHdl.device().has_index(), "GridBatchImpl must have a valid index");

    const int64_t totalPadAmount = upsamplingFactor[0] * upsamplingFactor[1] * upsamplingFactor[2];

    const auto optsData = torch::TensorOptions().dtype(torch::kInt32).device(batchHdl.device());
    const auto optsBIdx = optsData.dtype(fvdb::JIdxScalarType);

    if (mask) {
        torch::Tensor maskPrefixSum = torch::cumsum(mask.value().jdata(), 0, torch::kLong);
        auto totalMaskedVoxels      = maskPrefixSum[-1].item<int64_t>();

        torch::Tensor outIJK     = torch::empty({totalMaskedVoxels * totalPadAmount, 3}, optsData);
        torch::Tensor outIJKBIdx = torch::empty({totalMaskedVoxels * totalPadAmount},
                                                optsBIdx); // TODO: Don't populate for single batch

        auto outIJKAcc = outIJK.packed_accessor64<int32_t, 2, torch::RestrictPtrTraits>();
        auto outIJKBIdxAcc =
            outIJKBIdx.packed_accessor64<fvdb::JIdxType, 1, torch::RestrictPtrTraits>();

        auto maskAcc = mask.value().jdata().packed_accessor64<bool, 1, torch::RestrictPtrTraits>();
        auto maskPrefixSumAcc =
            maskPrefixSum.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>();

        auto cb = [=] __device__(int32_t bidx,
                                 int32_t lidx,
                                 int32_t vidx,
                                 int32_t cidx,
                                 GridBatchImpl::Accessor bacc) {
            fineIjkForCoarseGridVoxelCallback(bidx,
                                              lidx,
                                              vidx,
                                              cidx,
                                              bacc,
                                              upsamplingFactor,
                                              outIJKAcc,
                                              outIJKBIdxAcc,
                                              maskAcc,
                                              maskPrefixSumAcc);
        };

        forEachVoxelCUDA(DEFAULT_BLOCK_DIM, 1, batchHdl, cb);

        at::cuda::CUDAStream stream  = at::cuda::getCurrentCUDAStream(batchHdl.device().index());
        torch::Tensor outVoxelCounts = torch::zeros_like(batchHdl.voxelOffsets());

        void *dTempStorage      = nullptr;
        size_t tempStorageBytes = 0;
        // voxelOffsets has a length equal to batchSize() + 1 such that the first element is zero
        // and the last element is equal to the size of jdata
        auto maskCounts        = outVoxelCounts.data_ptr<int64_t>() + 1;
        const auto numSegments = batchHdl.batchSize();
        // offset of the next segment is the end of the current segment
        auto beginOffsets = batchHdl.voxelOffsets().const_data_ptr<int64_t>();
        auto endOffsets   = beginOffsets + 1;
        cub::DeviceSegmentedReduce::Sum(dTempStorage,
                                        tempStorageBytes,
                                        mask.value().jdata().const_data_ptr<bool>(),
                                        maskCounts,
                                        numSegments,
                                        beginOffsets,
                                        endOffsets,
                                        stream);
        dTempStorage =
            c10::cuda::CUDACachingAllocator::raw_alloc_with_stream(tempStorageBytes, stream);
        cub::DeviceSegmentedReduce::Sum(dTempStorage,
                                        tempStorageBytes,
                                        mask.value().jdata().const_data_ptr<bool>(),
                                        maskCounts,
                                        numSegments,
                                        beginOffsets,
                                        endOffsets,
                                        stream);
        c10::cuda::CUDACachingAllocator::raw_delete(dTempStorage);

        torch::Tensor outVoxelOffsets = torch::cumsum(outVoxelCounts, 0) * totalPadAmount;
        return JaggedTensor::from_jdata_joffsets_jidx_and_lidx_unsafe(
            outIJK, outVoxelOffsets, outIJKBIdx, batchHdl.jlidx(), batchHdl.batchSize());
    } else {
        torch::Tensor outIJK = torch::empty({batchHdl.totalVoxels() * totalPadAmount, 3}, optsData);
        torch::Tensor outIJKBIdx = torch::empty({batchHdl.totalVoxels() * totalPadAmount},
                                                optsBIdx); // TODO: Don't populate for single batch

        auto outIJKAcc = outIJK.packed_accessor64<int32_t, 2, torch::RestrictPtrTraits>();
        auto outIJKBIdxAcc =
            outIJKBIdx.packed_accessor64<fvdb::JIdxType, 1, torch::RestrictPtrTraits>();

        auto cb = [=] __device__(int32_t bidx,
                                 int32_t lidx,
                                 int32_t vidx,
                                 int32_t cidx,
                                 GridBatchImpl::Accessor bacc) {
            fineIjkForCoarseGridVoxelCallback(
                bidx, lidx, vidx, cidx, bacc, upsamplingFactor, outIJKAcc, outIJKBIdxAcc);
        };

        forEachVoxelCUDA(DEFAULT_BLOCK_DIM, 1, batchHdl, cb);

        return JaggedTensor::from_data_offsets_and_list_ids(
            outIJK, batchHdl.voxelOffsets() * totalPadAmount, batchHdl.jlidx());
    }
}

template <>
JaggedTensor
dispatchFineIJKForCoarseGrid<torch::kPrivateUse1>(const GridBatchImpl &batchHdl,
                                                  nanovdb::Coord upsamplingFactor,
                                                  const std::optional<JaggedTensor> &mask) {
    TORCH_CHECK(batchHdl.device().is_privateuseone(),
                "GridBatchImpl must be on PrivateUse1 device");

    const int64_t totalPadAmount = upsamplingFactor[0] * upsamplingFactor[1] * upsamplingFactor[2];

    const auto optsData = torch::TensorOptions().dtype(torch::kInt32).device(batchHdl.device());
    const auto optsBIdx = optsData.dtype(fvdb::JIdxScalarType);

    if (mask) {
        torch::Tensor maskPrefixSum = torch::cumsum(mask.value().jdata(), 0, torch::kLong);
        auto totalMaskedVoxels      = maskPrefixSum[-1].item<int64_t>();

        torch::Tensor outIJK     = torch::empty({totalMaskedVoxels * totalPadAmount, 3}, optsData);
        torch::Tensor outIJKBIdx = torch::empty({totalMaskedVoxels * totalPadAmount},
                                                optsBIdx); // TODO: Don't populate for single batch

        auto outIJKAcc = outIJK.packed_accessor64<int32_t, 2, torch::RestrictPtrTraits>();
        auto outIJKBIdxAcc =
            outIJKBIdx.packed_accessor64<fvdb::JIdxType, 1, torch::RestrictPtrTraits>();

        auto maskAcc = mask.value().jdata().packed_accessor64<bool, 1, torch::RestrictPtrTraits>();
        auto maskPrefixSumAcc =
            maskPrefixSum.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>();

        auto cb = [=] __device__(int32_t bidx,
                                 int32_t lidx,
                                 int32_t vidx,
                                 int32_t cidx,
                                 GridBatchImpl::Accessor bacc) {
            fineIjkForCoarseGridVoxelCallback(bidx,
                                              lidx,
                                              vidx,
                                              cidx,
                                              bacc,
                                              upsamplingFactor,
                                              outIJKAcc,
                                              outIJKBIdxAcc,
                                              maskAcc,
                                              maskPrefixSumAcc);
        };

        forEachVoxelPrivateUse1(1, batchHdl, cb);

        torch::Tensor outVoxelCounts = torch::zeros_like(batchHdl.voxelOffsets());
        for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
            C10_CUDA_CHECK(cudaSetDevice(deviceId));
            cudaStream_t stream = c10::cuda::getCurrentCUDAStream(deviceId).stream();

            auto deviceNumSegments =
                (batchHdl.batchSize() + c10::cuda::device_count() - 1) / c10::cuda::device_count();
            const auto deviceOffset = deviceNumSegments * deviceId;
            deviceNumSegments = std::min(deviceNumSegments, batchHdl.batchSize() - deviceOffset);

            auto maskCounts   = outVoxelCounts.data_ptr<int64_t>() + deviceOffset + 1;
            auto beginOffsets = batchHdl.voxelOffsets().const_data_ptr<int64_t>() + deviceOffset;
            auto endOffsets   = beginOffsets + 1;

            void *dTempStorage      = nullptr;
            size_t tempStorageBytes = 0;
            cub::DeviceSegmentedReduce::Sum(dTempStorage,
                                            tempStorageBytes,
                                            mask.value().jdata().const_data_ptr<bool>(),
                                            maskCounts,
                                            deviceNumSegments,
                                            beginOffsets,
                                            endOffsets,
                                            stream);
            cudaMallocAsync(&dTempStorage, tempStorageBytes, stream);
            cub::DeviceSegmentedReduce::Sum(dTempStorage,
                                            tempStorageBytes,
                                            mask.value().jdata().const_data_ptr<bool>(),
                                            maskCounts,
                                            deviceNumSegments,
                                            beginOffsets,
                                            endOffsets,
                                            stream);
            cudaFreeAsync(dTempStorage, stream);
        }

        for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
            c10::cuda::getCurrentCUDAStream(deviceId).synchronize();
        }

        torch::Tensor outVoxelOffsets = torch::cumsum(outVoxelCounts, 0) * totalPadAmount;
        return JaggedTensor::from_jdata_joffsets_jidx_and_lidx_unsafe(
            outIJK, outVoxelOffsets, outIJKBIdx, batchHdl.jlidx(), batchHdl.batchSize());
    } else {
        torch::Tensor outIJK = torch::empty({batchHdl.totalVoxels() * totalPadAmount, 3}, optsData);
        torch::Tensor outIJKBIdx = torch::empty({batchHdl.totalVoxels() * totalPadAmount},
                                                optsBIdx); // TODO: Don't populate for single batch

        auto outIJKAcc = outIJK.packed_accessor64<int32_t, 2, torch::RestrictPtrTraits>();
        auto outIJKBIdxAcc =
            outIJKBIdx.packed_accessor64<fvdb::JIdxType, 1, torch::RestrictPtrTraits>();

        auto cb = [=] __device__(int32_t bidx,
                                 int32_t lidx,
                                 int32_t vidx,
                                 int32_t cidx,
                                 GridBatchImpl::Accessor bacc) {
            fineIjkForCoarseGridVoxelCallback(
                bidx, lidx, vidx, cidx, bacc, upsamplingFactor, outIJKAcc, outIJKBIdxAcc);
        };

        forEachVoxelPrivateUse1(1, batchHdl, cb);

        return JaggedTensor::from_data_offsets_and_list_ids(
            outIJK, batchHdl.voxelOffsets() * totalPadAmount, batchHdl.jlidx());
    }
}

template <>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchBuildFineGridFromCoarse<torch::kCUDA>(const GridBatchImpl &coarseBatchHdl,
                                              const nanovdb::Coord subdivisionFactor,
                                              const std::optional<JaggedTensor> &subdivMask) {
    JaggedTensor coords =
        dispatchFineIJKForCoarseGrid<torch::kCUDA>(coarseBatchHdl, subdivisionFactor, subdivMask);
    return ops::dispatchCreateNanoGridFromIJK<torch::kCUDA>(coords);
}

template <>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchBuildFineGridFromCoarse<torch::kPrivateUse1>(
    const GridBatchImpl &coarseBatchHdl,
    const nanovdb::Coord subdivisionFactor,
    const std::optional<JaggedTensor> &subdivMask) {
    JaggedTensor coords = dispatchFineIJKForCoarseGrid<torch::kPrivateUse1>(
        coarseBatchHdl, subdivisionFactor, subdivMask);
    return ops::dispatchCreateNanoGridFromIJK<torch::kPrivateUse1>(coords);
}

template <>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchBuildFineGridFromCoarse<torch::kCPU>(const GridBatchImpl &coarseBatchHdl,
                                             const nanovdb::Coord subdivisionFactor,
                                             const std::optional<JaggedTensor> &subdivMask) {
    using GridT = nanovdb::ValueOnIndex;
    torch::Tensor subdivMaskTensor;
    if (subdivMask.has_value()) {
        subdivMaskTensor = subdivMask.value().jdata();
    } else {
        subdivMaskTensor = torch::zeros(0, torch::TensorOptions().dtype(torch::kBool));
    }

    using IndexTree = nanovdb::NanoTree<GridT>;

    const nanovdb::GridHandle<TorchDeviceBuffer> &coarseGridHdl = coarseBatchHdl.nanoGridHandle();
    const torch::TensorAccessor<bool, 1> &subdivMaskAcc = subdivMaskTensor.accessor<bool, 1>();

    std::vector<nanovdb::GridHandle<TorchDeviceBuffer>> batchHandles;
    batchHandles.reserve(coarseGridHdl.gridCount());
    for (uint32_t bidx = 0; bidx < coarseGridHdl.gridCount(); bidx += 1) {
        const nanovdb::OnIndexGrid *coarseGrid = coarseGridHdl.template grid<GridT>(bidx);
        if (!coarseGrid) {
            throw std::runtime_error("Failed to get pointer to nanovdb index grid");
        }
        const IndexTree &coarseTree = coarseGrid->tree();

        using ProxyGridT       = nanovdb::tools::build::Grid<float>;
        auto proxyGrid         = std::make_shared<ProxyGridT>(-1.0f);
        auto proxyGridAccessor = proxyGrid->getWriteAccessor();

        const int64_t joffset = coarseBatchHdl.cumVoxelsAt(bidx);
        for (auto it = ActiveVoxelIterator<-1>(coarseTree); it.isValid(); it++) {
            const nanovdb::Coord baseIjk(it->first[0] * subdivisionFactor[0],
                                         it->first[1] * subdivisionFactor[1],
                                         it->first[2] * subdivisionFactor[2]);

            if (subdivMaskAcc.size(0) > 0 && !subdivMaskAcc[it->second + joffset]) {
                continue;
            }

            for (int i = 0; i < subdivisionFactor[0]; i += 1) {
                for (int j = 0; j < subdivisionFactor[1]; j += 1) {
                    for (int k = 0; k < subdivisionFactor[2]; k += 1) {
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
        batchHandles.push_back(std::move(ret));
    }

    if (batchHandles.size() == 1) {
        return std::move(batchHandles[0]);
    } else {
        return nanovdb::mergeGrids(batchHandles);
    }
}

} // namespace fvdb::detail::ops
