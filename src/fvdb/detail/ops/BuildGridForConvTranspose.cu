// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/GridBatchImpl.h>
#include <fvdb/detail/ops/BuildFineGridFromCoarse.h>
#include <fvdb/detail/ops/BuildGridForConvTranspose.h>
#include <fvdb/detail/ops/BuildGridFromIjk.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/cuda/ForEachCUDA.cuh>
#include <fvdb/detail/utils/cuda/GridDim.h>

#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/cuda/PointsToGrid.cuh>

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

nanovdb::GridHandle<TorchDeviceBuffer>
buildFineGridFromCoarseGridCPU(const GridBatchImpl &coarseBatchHdl,
                               const nanovdb::Coord subdivisionFactor) {
    using GridT     = nanovdb::ValueOnIndex;
    using IndexTree = nanovdb::NanoTree<GridT>;

    const nanovdb::GridHandle<TorchDeviceBuffer> &coarseGridHdl = coarseBatchHdl.nanoGridHandle();

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

        for (auto it = ActiveVoxelIterator(coarseTree); it.isValid(); it++) {
            const nanovdb::Coord baseIjk(it->first[0] * subdivisionFactor[0],
                                         it->first[1] * subdivisionFactor[1],
                                         it->first[2] * subdivisionFactor[2]);
            for (int i = 0; i < subdivisionFactor[0]; i += 1) {
                for (int j = 0; j < subdivisionFactor[1]; j += 1) {
                    for (int k = 0; k < subdivisionFactor[2]; k += 1) {
                        const nanovdb::Coord fineIjk = baseIjk + nanovdb::Coord(i, j, k);
                        proxyGridAccessor.setValue(fineIjk, 1.0f);
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

__device__ void
convTransposeIjkForGridCallback(int32_t bidx,
                                int32_t lidx,
                                int32_t vidx,
                                int32_t cidx,
                                const GridBatchImpl::Accessor batchAcc,
                                const nanovdb::Coord &kernelSize,
                                const nanovdb::Coord &stride,
                                int kernelVolume,
                                TorchRAcc32<int32_t, 2> outIJKData,
                                TorchRAcc32<fvdb::JIdxType, 1> outIJKBIdx) {
    const nanovdb::OnIndexGrid *gridPtr = batchAcc.grid(bidx);
    const typename nanovdb::OnIndexGrid::LeafNodeType &leaf =
        gridPtr->tree().template getFirstNode<0>()[lidx];
    if (!leaf.isActive(vidx))
        return;

    const nanovdb::Coord &srcIjk = leaf.offsetToGlobalCoord(vidx);
    const int64_t index          = ((int64_t)leaf.getValue(vidx)) - 1;
    const int64_t baseOffset     = batchAcc.voxelOffset(bidx);

    // Compute kernel offset bounds (same as conv)
    int lower[3], upper[3];
    for (int i = 0; i < 3; ++i) {
        if (kernelSize[i] % 2 == 0) {
            lower[i] = 0;
            upper[i] = kernelSize[i] - 1;
        } else {
            lower[i] = -(kernelSize[i] - 1) / 2;
            upper[i] = (kernelSize[i] - 1) / 2;
        }
    }

    // For ConvTranspose: dstIjk = srcIjk * stride + offset
    // Unlike conv, all positions are valid (no divisibility check needed)
    int64_t count = 0;
    for (int di = lower[0]; di <= upper[0]; di += 1) {
        for (int dj = lower[1]; dj <= upper[1]; dj += 1) {
            for (int dk = lower[2]; dk <= upper[2]; dk += 1, count += 1) {
                const nanovdb::Coord dstIjk(srcIjk[0] * stride[0] + di,
                                            srcIjk[1] * stride[1] + dj,
                                            srcIjk[2] * stride[2] + dk);

                const int64_t base  = (baseOffset + index) * kernelVolume + count;
                outIJKData[base][0] = dstIjk[0];
                outIJKData[base][1] = dstIjk[1];
                outIJKData[base][2] = dstIjk[2];
                outIJKBIdx[base]    = bidx;
            }
        }
    }
}

JaggedTensor
convTransposeIJKForGrid(const GridBatchImpl &batchHdl,
                        const nanovdb::Coord &kernelSize,
                        const nanovdb::Coord &stride) {
    TORCH_CHECK(batchHdl.device().is_cuda(), "GridBatchImpl must be on CUDA device");
    TORCH_CHECK(batchHdl.device().has_index(), "GridBatchImpl must have a valid index");

    // Special case: kernel size 1 or stride equals kernel size is pure subdivision
    if (kernelSize == nanovdb::Coord(1) || stride == kernelSize) {
        return dispatchFineIJKForCoarseGrid<torch::kCUDA>(batchHdl, stride, std::nullopt);
    }

    const int32_t kernelVolume = kernelSize.x() * kernelSize.y() * kernelSize.z();

    const torch::TensorOptions optsData =
        torch::TensorOptions().dtype(torch::kInt32).device(batchHdl.device());
    const torch::TensorOptions optsBIdx =
        torch::TensorOptions().dtype(fvdb::JIdxScalarType).device(batchHdl.device());
    torch::Tensor outIJK     = torch::empty({batchHdl.totalVoxels() * kernelVolume, 3}, optsData);
    torch::Tensor outIJKBIdx = torch::empty({batchHdl.totalVoxels() * kernelVolume}, optsBIdx);

    // For each voxel in source grid, compute possible voxels in target grid
    auto outIJKAcc = outIJK.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>();
    auto outIJKBIdxAcc =
        outIJKBIdx.packed_accessor32<fvdb::JIdxType, 1, torch::RestrictPtrTraits>();

    auto cb = [=] __device__(int32_t bidx,
                             int32_t lidx,
                             int32_t vidx,
                             int32_t cidx,
                             GridBatchImpl::Accessor bacc) {
        convTransposeIjkForGridCallback(bidx,
                                        lidx,
                                        vidx,
                                        cidx,
                                        bacc,
                                        kernelSize,
                                        stride,
                                        kernelVolume,
                                        outIJKAcc,
                                        outIJKBIdxAcc);
    };
    forEachVoxelCUDA(256, 1, batchHdl, cb);

    return JaggedTensor::from_data_indices_and_list_ids(
        outIJK, outIJKBIdx, batchHdl.jlidx(), batchHdl.batchSize());
}

template <>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchBuildGridForConvTranspose<torch::kCUDA>(const GridBatchImpl &baseGridHdl,
                                                const nanovdb::Coord &kernelSize,
                                                const nanovdb::Coord &stride) {
    JaggedTensor coords = convTransposeIJKForGrid(baseGridHdl, kernelSize, stride);
    return ops::dispatchCreateNanoGridFromIJK<torch::kCUDA>(coords);
}

template <>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchBuildGridForConvTranspose<torch::kCPU>(const GridBatchImpl &baseBatchHdl,
                                               const nanovdb::Coord &kernelSize,
                                               const nanovdb::Coord &stride) {
    using GridT = nanovdb::ValueOnIndex;

    // Special case: kernel size 1 or stride equals kernel size is pure subdivision
    if (kernelSize == nanovdb::Coord(1) || stride == kernelSize) {
        return buildFineGridFromCoarseGridCPU(baseBatchHdl, stride);
    }

    const nanovdb::GridHandle<TorchDeviceBuffer> &baseGridHdl = baseBatchHdl.nanoGridHandle();
    std::vector<nanovdb::GridHandle<TorchDeviceBuffer>> batchHandles;
    batchHandles.reserve(baseGridHdl.gridCount());

    // Compute kernel offset bounds (same as conv)
    int lower[3], upper[3];
    for (int i = 0; i < 3; i += 1) {
        if (kernelSize[i] % 2 == 0) {
            lower[i] = 0;
            upper[i] = kernelSize[i] - 1;
        } else {
            lower[i] = -(kernelSize[i] - 1) / 2;
            upper[i] = (kernelSize[i] - 1) / 2;
        }
    }

    for (uint32_t bidx = 0; bidx < baseGridHdl.gridCount(); bidx += 1) {
        const nanovdb::OnIndexGrid *baseGrid = baseGridHdl.template grid<GridT>(bidx);
        if (!baseGrid) {
            throw std::runtime_error("Failed to get pointer to nanovdb index grid");
        }

        using ProxyGridT       = nanovdb::tools::build::Grid<float>;
        auto proxyGrid         = std::make_shared<ProxyGridT>(-1.0f);
        auto proxyGridAccessor = proxyGrid->getWriteAccessor();

        for (auto it = ActiveVoxelIterator(baseGrid->tree()); it.isValid(); it++) {
            const nanovdb::Coord &ijk0 = it->first;

            // For ConvTranspose: dstIjk = srcIjk * stride + offset
            for (int di = lower[0]; di <= upper[0]; di += 1) {
                for (int dj = lower[1]; dj <= upper[1]; dj += 1) {
                    for (int dk = lower[2]; dk <= upper[2]; dk += 1) {
                        const nanovdb::Coord dstIjk(ijk0[0] * stride[0] + di,
                                                    ijk0[1] * stride[1] + dj,
                                                    ijk0[2] * stride[2] + dk);
                        proxyGridAccessor.setValue(dstIjk, 1.0f);
                    }
                }
            }
        }

        proxyGridAccessor.merge();
        auto ret = nanovdb::tools::createNanoGrid<ProxyGridT, GridT, TorchDeviceBuffer>(
            *proxyGrid, 0u, false, false);
        batchHandles.push_back(std::move(ret));
    }

    if (batchHandles.size() == 1) {
        return std::move(batchHandles[0]);
    } else {
        return nanovdb::mergeGrids(batchHandles);
    }
}

} // namespace ops
} // namespace detail
} // namespace fvdb
