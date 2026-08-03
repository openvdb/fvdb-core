// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/GridBatchData.h>
#include <fvdb/detail/GridBatchDataFactory.h>
#include <fvdb/detail/ops/BuildFineGridFromCoarse.h>
#include <fvdb/detail/ops/BuildGridForConvTranspose.h>
#include <fvdb/detail/ops/BuildGridFromIjk.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/cuda/ForEachCUDA.cuh>
#include <fvdb/detail/utils/cuda/GridDim.h>
#include <fvdb/detail/utils/nanovdb/CreateEmptyGridHandle.h>
#include <fvdb/detail/utils/nanovdb/PadGrid.cuh>

#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/cuda/DilateGrid.cuh>
#include <nanovdb/tools/cuda/PointsToGrid.cuh>
#include <nanovdb/tools/cuda/RefineGrid.cuh>
#include <nanovdb/util/MorphologyHelpers.h>

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

template <torch::DeviceType>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchBuildGridForConvTranspose(const GridBatchData &baseBatchHdl,
                                  const nanovdb::Coord &kernelSize,
                                  const nanovdb::Coord &stride);

nanovdb::GridHandle<TorchDeviceBuffer>
buildFineGridFromCoarseGridCPU(const GridBatchData &coarseBatchHdl,
                               const nanovdb::Coord subdivisionFactor) {
    using GridT     = nanovdb::ValueOnIndex;
    using IndexTree = nanovdb::NanoTree<GridT>;

    const nanovdb::GridHandle<TorchDeviceBuffer> &coarseGridHdl = coarseBatchHdl.nanoGridHandle();

    std::vector<nanovdb::GridHandle<TorchDeviceBuffer>> batchHandles;
    batchHandles.reserve(coarseGridHdl.gridCount());
    for (int64_t bidx = 0; bidx < coarseBatchHdl.batchSize(); bidx += 1) {
        // Byte-offset accessor: correct for sliced/non-contiguous batches (see hostGridPtrAt).
        const nanovdb::OnIndexGrid *coarseGrid = coarseBatchHdl.hostGridPtrAt(bidx);
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
                                const GridBatchData::Accessor batchAcc,
                                const nanovdb::Coord &kernelSize,
                                const nanovdb::Coord &stride,
                                int kernelVolume,
                                TorchRAcc64<int32_t, 2> outIJKData,
                                TorchRAcc64<fvdb::JIdxType, 1> outIJKBIdx) {
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
convTransposeIJKForGrid(const GridBatchData &batchHdl,
                        const nanovdb::Coord &kernelSize,
                        const nanovdb::Coord &stride) {
    TORCH_CHECK(batchHdl.device().is_cuda(), "GridBatchData must be on CUDA device");
    TORCH_CHECK(batchHdl.device().has_index(), "GridBatchData must have a valid index");

    // Special case: kernel size 1 or stride equals kernel size is pure subdivision
    if (kernelSize == nanovdb::Coord(1) || stride == kernelSize) {
        return fineIJKForCoarseGrid(batchHdl, stride, std::nullopt);
    }

    const int32_t kernelVolume = kernelSize.x() * kernelSize.y() * kernelSize.z();

    const torch::TensorOptions optsData =
        torch::TensorOptions().dtype(torch::kInt32).device(batchHdl.device());
    const torch::TensorOptions optsBIdx =
        torch::TensorOptions().dtype(fvdb::JIdxScalarType).device(batchHdl.device());
    torch::Tensor outIJK     = torch::empty({batchHdl.totalVoxels() * kernelVolume, 3}, optsData);
    torch::Tensor outIJKBIdx = torch::empty({batchHdl.totalVoxels() * kernelVolume}, optsBIdx);

    // For each voxel in source grid, compute possible voxels in target grid
    auto outIJKAcc = outIJK.packed_accessor64<int32_t, 2, torch::RestrictPtrTraits>();
    auto outIJKBIdxAcc =
        outIJKBIdx.packed_accessor64<fvdb::JIdxType, 1, torch::RestrictPtrTraits>();

    auto cb = [=] __device__(int32_t bidx,
                             int32_t lidx,
                             int32_t vidx,
                             int32_t cidx,
                             GridBatchData::Accessor bacc) {
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
    forEachVoxelCUDA(1, batchHdl, cb);

    return JaggedTensor::from_data_indices_and_list_ids(
        outIJK, outIJKBIdx, batchHdl.jlidx(), batchHdl.batchSize());
}

// Applies fn(grid) -> handle to each non-empty batch item, empties -> empty grid, then merges.
template <typename PerGridFn>
static nanovdb::GridHandle<TorchDeviceBuffer>
perItemGridHandle(const GridBatchData &base, const TorchDeviceBuffer &guide, PerGridFn &&fn) {
    std::vector<nanovdb::GridHandle<TorchDeviceBuffer>> handles;
    handles.reserve(base.batchSize());
    for (int64_t i = 0; i < base.batchSize(); i += 1) {
        if (base.numVoxelsAt(i) == 0) {
            handles.push_back(createEmptyGridHandle(base.device()));
            continue;
        }
        // Byte-offset accessor: correct for sliced/non-contiguous batches (see deviceGridPtrAt).
        nanovdb::OnIndexGrid *grid = base.deviceGridPtrAt(i);
        TORCH_CHECK(grid, "Grid is null");
        handles.push_back(fn(grid));
    }
    return handles.size() == 1 ? std::move(handles[0])
                               : nanovdb::cuda::mergeGridHandles(handles, &guide);
}

template <>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchBuildGridForConvTranspose<torch::kCUDA>(const GridBatchData &baseGridHdl,
                                                const nanovdb::Coord &kernelSize,
                                                const nanovdb::Coord &stride) {
    // Fast path 1: (kernel_size == 1 || stride == kernel_size) is pure subdivision by stride --
    // exactly the coordinate short circuit in convTransposeIJKForGrid -- so reuse the leaf-mask
    // subdivide builder (RefineGrid for uniform power-of-two stride, coordinate fallback
    // otherwise).
    if (kernelSize == nanovdb::Coord(1) || stride == kernelSize) {
        return fineGridHandleFromCoarseCUDA(baseGridHdl, stride, std::nullopt);
    }

    const bool uniformKernel = (kernelSize[0] == kernelSize[1] && kernelSize[1] == kernelSize[2]);

    c10::cuda::CUDAGuard deviceGuard(baseGridHdl.device());
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(baseGridHdl.device().index());
    TorchDeviceBuffer guide(0, baseGridHdl.device());

    // Fast path 2: stride 1 with a uniform kernel builds S (+) window (dstIjk = srcIjk + offset),
    // identical to the forward conv: odd kernel k -> (k-1)/2 symmetric dilations, even kernel k ->
    // (k-1) positive pad passes.
    if (stride == nanovdb::Coord(1) && uniformKernel && kernelSize[0] > 1) {
        const int k = kernelSize[0];
        return perItemGridHandle(baseGridHdl, guide, [&](nanovdb::OnIndexGrid *grid) {
            nanovdb::GridHandle<TorchDeviceBuffer> handle;
            if (k % 2 == 1) {
                for (int p = 0; p < (k - 1) / 2; p += 1) {
                    nanovdb::tools::cuda::DilateGrid<nanovdb::ValueOnIndex> op(grid,
                                                                               stream.stream());
                    op.setOperation(nanovdb::tools::morphology::NN_FACE_EDGE_VERTEX);
                    op.setChecksum(nanovdb::CheckMode::Default);
                    op.setVerbose(0);
                    handle = op.getHandle(guide);
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                    grid = handle.deviceGrid<nanovdb::ValueOnIndex>();
                }
            } else {
                for (int p = 0; p < k - 1; p += 1) {
                    morphology::PadGrid<nanovdb::ValueOnIndex> op(
                        grid, /*positiveOctant=*/true, stream.stream());
                    op.setChecksum(nanovdb::CheckMode::Default);
                    handle = op.getHandle(guide);
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                    grid = handle.deviceGrid<nanovdb::ValueOnIndex>();
                }
            }
            return handle;
        });
    }

    // Fast path 3: stride 2, kernel 3 (the classic upsampling conv-transpose). The output is
    // 2S (+) [-1,1]^3 (dstIjk = 2*srcIjk + offset, offset in [-1,1]^3). RefineGrid gives
    // 2S (+) {0,1}^3, and one negative pad pass adds (+) {-1,0}^3, composing to (+) [-1,1]^3.
    if (stride == nanovdb::Coord(2) && uniformKernel && kernelSize[0] == 3) {
        return perItemGridHandle(baseGridHdl, guide, [&](nanovdb::OnIndexGrid *grid) {
            nanovdb::tools::cuda::RefineGrid<nanovdb::ValueOnIndex> refineOp(grid, stream.stream());
            refineOp.setChecksum(nanovdb::CheckMode::Default);
            refineOp.setVerbose(0);
            nanovdb::GridHandle<TorchDeviceBuffer> refined = refineOp.getHandle(guide);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            morphology::PadGrid<nanovdb::ValueOnIndex> padOp(
                refined.deviceGrid<nanovdb::ValueOnIndex>(),
                /*positiveOctant=*/false,
                stream.stream());
            padOp.setChecksum(nanovdb::CheckMode::Default);
            nanovdb::GridHandle<TorchDeviceBuffer> handle = padOp.getHandle(guide);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return handle;
        });
    }

    // Fallback: general coordinate-list path.
    JaggedTensor coords = convTransposeIJKForGrid(baseGridHdl, kernelSize, stride);
    return ops::_createNanoGridFromIJK(coords);
}

template <>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchBuildGridForConvTranspose<torch::kCPU>(const GridBatchData &baseBatchHdl,
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

    for (int64_t bidx = 0; bidx < baseBatchHdl.batchSize(); bidx += 1) {
        // Byte-offset accessor: correct for sliced/non-contiguous batches (see hostGridPtrAt).
        const nanovdb::OnIndexGrid *baseGrid = baseBatchHdl.hostGridPtrAt(bidx);
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

c10::intrusive_ptr<GridBatchData>
buildGridForConvTranspose(const GridBatchData &baseBatchHdl,
                          const nanovdb::Coord &kernelSize,
                          const nanovdb::Coord &stride) {
    TORCH_CHECK_VALUE(nanovdb::Coord(0) < kernelSize, "kernel_size must be strictly positive.");
    TORCH_CHECK_VALUE(nanovdb::Coord(0) < stride, "stride must be strictly positive.");
    std::vector<nanovdb::Vec3d> voxS, voxO;
    baseBatchHdl.gridVoxelSizesAndOrigins(voxS, voxO);
    auto hdl = FVDB_DISPATCH_KERNEL_DEVICE(baseBatchHdl.device(), [&]() {
        return dispatchBuildGridForConvTranspose<DeviceTag>(baseBatchHdl, kernelSize, stride);
    });
    return makeGridBatchData(std::move(hdl), voxS, voxO);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
