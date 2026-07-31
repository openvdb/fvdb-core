// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/GridBatchData.h>
#include <fvdb/TorchDeviceBuffer.h>
#include <fvdb/detail/GridBatchDataFactory.h>
#include <fvdb/detail/ops/BuildGridFromIjk.h>
#include <fvdb/detail/ops/BuildPaddedGrid.h>
#include <fvdb/detail/ops/PopulateGridMetadata.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/cuda/ForEachCUDA.cuh>
#include <fvdb/detail/utils/cuda/ForEachPrivateUse1.cuh>
#include <fvdb/detail/utils/cuda/GridDim.h>
#include <fvdb/detail/utils/nanovdb/CreateEmptyGridHandle.h>
#include <fvdb/detail/utils/nanovdb/PadGrid.cuh>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/cuda/PruneGrid.cuh>
#include <nanovdb/util/cuda/DeviceGridTraits.cuh>
#include <nanovdb/util/cuda/Util.h>

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMathCompat.h>

namespace fvdb {
namespace detail {
namespace ops {

template <torch::DeviceType>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchBuildPaddedGrid(const GridBatchData &baseBatchHdl, int bmin, int bmax, bool excludeBorder);

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
ijkForGridVoxelCallback(int32_t bidx,
                        int32_t lidx,
                        int32_t vidx,
                        int32_t cidx,
                        const GridBatchData::Accessor batchAcc,
                        const nanovdb::CoordBBox bbox,
                        TorchRAcc64<int32_t, 2> outIJKData,
                        TorchRAcc64<fvdb::JIdxType, 1> outIJKBIdx) {
    const int32_t totalPadAmount = static_cast<int32_t>(bbox.volume());

    const nanovdb::OnIndexGrid *gridPtr = batchAcc.grid(bidx);
    const int64_t totalVoxels           = gridPtr->activeVoxelCount();
    const typename nanovdb::OnIndexGrid::LeafNodeType &leaf =
        gridPtr->tree().template getFirstNode<0>()[lidx];
    const int64_t baseOffset = batchAcc.voxelOffset(bidx);

    if (leaf.isActive(vidx)) {
        const int64_t value       = ((int64_t)leaf.getValue(vidx)) - 1;
        const int64_t base        = (baseOffset + value) * totalPadAmount;
        const nanovdb::Coord ijk0 = leaf.offsetToGlobalCoord(vidx);
        copyCoords(bidx, base, ijk0, bbox, outIJKData, outIJKBIdx);
    }
}

template <torch::DeviceType DeviceTag>
JaggedTensor
paddedIJKForGrid(const GridBatchData &batchHdl, const nanovdb::CoordBBox &bbox) {
    TORCH_CHECK(batchHdl.device().is_cuda() || batchHdl.device().is_privateuseone(),
                "GridBatchData must be on CUDA or PrivateUse1 device");
    TORCH_CHECK(batchHdl.device().has_index(), "GridBatchData must have a valid index");

    const int32_t totalPadAmount = static_cast<int32_t>(bbox.volume());

    const torch::TensorOptions optsData =
        torch::TensorOptions().dtype(torch::kInt32).device(batchHdl.device());
    const torch::TensorOptions optsBIdx =
        torch::TensorOptions().dtype(fvdb::JIdxScalarType).device(batchHdl.device());
    torch::Tensor outIJK     = torch::empty({batchHdl.totalVoxels() * totalPadAmount, 3}, optsData);
    torch::Tensor outIJKBIdx = torch::empty({batchHdl.totalVoxels() * totalPadAmount},
                                            optsBIdx); // TODO: Don't populate for single batch

    auto outIJKAcc = outIJK.packed_accessor64<int32_t, 2, torch::RestrictPtrTraits>();
    auto outIJKBIdxAcc =
        outIJKBIdx.packed_accessor64<fvdb::JIdxType, 1, torch::RestrictPtrTraits>();

    auto cb = [=] __device__(int32_t bidx,
                             int32_t lidx,
                             int32_t vidx,
                             int32_t cidx,
                             GridBatchData::Accessor bacc) {
        ijkForGridVoxelCallback(bidx, lidx, vidx, cidx, bacc, bbox, outIJKAcc, outIJKBIdxAcc);
    };

    if constexpr (DeviceTag == torch::kCUDA) {
        forEachVoxelCUDA(1, batchHdl, cb);
    } else if constexpr (DeviceTag == torch::kPrivateUse1) {
        forEachVoxelPrivateUse1(1, batchHdl, cb);
    }

    return JaggedTensor::from_data_offsets_and_list_ids(
        outIJK, batchHdl.voxelOffsets() * totalPadAmount, batchHdl.jlidx());
}

nanovdb::GridHandle<TorchDeviceBuffer>
buildPaddedGridFromGridWithoutBorderCPU(const GridBatchData &baseBatchHdl, int BMIN, int BMAX) {
    using GridT = nanovdb::ValueOnIndex;

    TORCH_CHECK(BMIN <= BMAX, "BMIN must be less than BMAX");

    const nanovdb::GridHandle<TorchDeviceBuffer> &baseGridHdl = baseBatchHdl.nanoGridHandle();

    std::vector<nanovdb::GridHandle<TorchDeviceBuffer>> batchHandles;
    batchHandles.reserve(baseGridHdl.gridCount());
    for (uint32_t bidx = 0; bidx < baseGridHdl.gridCount(); bidx += 1) {
        const nanovdb::OnIndexGrid *baseGrid = baseGridHdl.template grid<GridT>(bidx);
        if (!baseGrid) {
            throw std::runtime_error("Failed to get pointer to nanovdb index grid");
        }
        auto baseGridAccessor = baseGrid->getAccessor();

        using ProxyGridT       = nanovdb::tools::build::Grid<float>;
        auto proxyGrid         = std::make_shared<ProxyGridT>(-1.0f);
        auto proxyGridAccessor = proxyGrid->getWriteAccessor();

        for (auto it = ActiveVoxelIterator(baseGrid->tree()); it.isValid(); it++) {
            nanovdb::Coord ijk0 = it->first;
            bool active         = true;
            for (int di = BMIN; di <= BMAX && active; di += 1) {
                for (int dj = BMIN; dj <= BMAX && active; dj += 1) {
                    for (int dk = BMIN; dk <= BMAX && active; dk += 1) {
                        const nanovdb::Coord ijk = ijk0 + nanovdb::Coord(di, dj, dk);
                        if (ijk != ijk0) {
                            active = active && baseGridAccessor.isActive(
                                                   ijk); // if any surrounding is off, turn it off.
                        }
                    }
                }
            }
            if (active) {
                proxyGridAccessor.setValue(ijk0, 1.0f);
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

nanovdb::GridHandle<TorchDeviceBuffer>
buildPaddedGridFromGridCPU(const GridBatchData &baseBatchHdl, int BMIN, int BMAX) {
    using GridT = nanovdb::ValueOnIndex;

    TORCH_CHECK(BMIN <= BMAX, "BMIN must be less than BMAX");

    const nanovdb::GridHandle<TorchDeviceBuffer> &baseGridHdl = baseBatchHdl.nanoGridHandle();

    std::vector<nanovdb::GridHandle<TorchDeviceBuffer>> batchHandles;
    batchHandles.reserve(baseGridHdl.gridCount());
    for (uint32_t bidx = 0; bidx < baseGridHdl.gridCount(); bidx += 1) {
        const nanovdb::OnIndexGrid *baseGrid = baseGridHdl.template grid<GridT>(bidx);
        if (!baseGrid) {
            throw std::runtime_error("Failed to get pointer to nanovdb index grid");
        }

        using ProxyGridT       = nanovdb::tools::build::Grid<float>;
        auto proxyGrid         = std::make_shared<ProxyGridT>(-1.0f);
        auto proxyGridAccessor = proxyGrid->getWriteAccessor();

        for (auto it = ActiveVoxelIterator(baseGrid->tree()); it.isValid(); it++) {
            nanovdb::Coord ijk0 = it->first;
            for (int di = BMIN; di <= BMAX; di += 1) {
                for (int dj = BMIN; dj <= BMAX; dj += 1) {
                    for (int dk = BMIN; dk <= BMAX; dk += 1) {
                        const nanovdb::Coord ijk = ijk0 + nanovdb::Coord(di, dj, dk);
                        proxyGridAccessor.setValue(ijk, 1.0f);
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

// One unit padding pass (Minkowski sum by {0,1}^3 if positive, else {-1,0}^3), building the
// padded topology directly from the source leaf masks via TopologyBuilder (no coordinate list).
static nanovdb::GridHandle<TorchDeviceBuffer>
padOncePass(nanovdb::OnIndexGrid *grid,
            bool positive,
            const TorchDeviceBuffer &guide,
            cudaStream_t stream) {
    fvdb::detail::morphology::PadGrid<nanovdb::ValueOnIndex> op(grid, positive, stream);
    op.setChecksum(nanovdb::CheckMode::Default);
    auto handle = op.getHandle(guide);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return handle;
}

// One unit erosion pass (exclude-border padding). Computes a per-leaf "keep" mask sidecar
// (voxel survives iff its whole octant neighborhood is active) then prunes the source grid to
// it via PruneGrid. The keep mask is a subset of the source, so the result is exact.
static nanovdb::GridHandle<TorchDeviceBuffer>
erodeOncePass(nanovdb::OnIndexGrid *grid,
              uint32_t leafCount,
              bool positive,
              const torch::Device &device,
              const TorchDeviceBuffer &guide,
              cudaStream_t stream) {
    // Allocate the per-leaf keep-mask sidecar as a torch CUDA tensor so its emptiness can be
    // tested reliably with a torch reduction below (a raw device buffer viewed via from_blob does
    // not reduce correctly).
    const int64_t maskBytes = static_cast<int64_t>(sizeof(nanovdb::Mask<3>)) * leafCount;
    torch::Tensor keepTensor =
        torch::empty({maskBytes}, torch::TensorOptions().dtype(torch::kUInt8).device(device));
    auto *keepMasks = reinterpret_cast<nanovdb::Mask<3> *>(keepTensor.data_ptr());
    if (positive) {
        nanovdb::util::cuda::lambdaKernel<<<(leafCount + 127) / 128, 128, 0, stream>>>(
            leafCount,
            fvdb::detail::morphology::ErodeKeepMaskFunctor<nanovdb::ValueOnIndex, true>(),
            grid,
            keepMasks);
    } else {
        nanovdb::util::cuda::lambdaKernel<<<(leafCount + 127) / 128, 128, 0, stream>>>(
            leafCount,
            fvdb::detail::morphology::ErodeKeepMaskFunctor<nanovdb::ValueOnIndex, false>(),
            grid,
            keepMasks);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // PruneGrid (via TopologyBuilder) dereferences a null d_upperOffsets when the result has no
    // nodes, so it cannot build an empty grid. Detect an all-empty keep mask (erosion removed
    // everything) and return an explicit empty grid instead -- same guard as BuildPrunedGrid.cu.
    if (!keepTensor.any().item<bool>()) {
        return createEmptyGridHandle(device);
    }

    nanovdb::tools::cuda::PruneGrid<nanovdb::ValueOnIndex> pruneOp(grid, keepMasks, stream);
    pruneOp.setChecksum(nanovdb::CheckMode::Default);
    pruneOp.setVerbose(0);
    auto handle = pruneOp.getHandle(guide);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return handle;
}

template <>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchBuildPaddedGrid<torch::kCUDA>(const GridBatchData &baseBatchHdl,
                                      int bmin,
                                      int bmax,
                                      bool excludeBorder) {
    c10::cuda::CUDAGuard deviceGuard(baseBatchHdl.device());
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(baseBatchHdl.device().index());

    // This guide buffer is a hack to pass a device (with index) into the TopologyBuilder buffer
    // allocation (see BuildDilatedGrid.cu). The created grid buffers inherit the guide's
    // device.
    TorchDeviceBuffer guide(0, baseBatchHdl.device());

    // Pad by [bmin, bmax]^3 = (bmax positive unit passes) followed by (-bmin negative unit
    // passes). Minkowski sums / erosions by boxes compose, so the order is immaterial.
    // dual_grid (0, 1) is exactly one positive pass.
    const int numPositive = bmax;
    const int numNegative = -bmin;
    const int totalPasses = numPositive + numNegative;

    // Identity case (bmin == bmax == 0): no morphology passes run, so return a copy of the whole
    // source handle (all batch items, empty ones included). The tail then applies the dual
    // transform swap.
    if (totalPasses == 0) {
        return baseBatchHdl.nanoGridHandle().copy<TorchDeviceBuffer>(guide);
    }

    std::vector<nanovdb::GridHandle<TorchDeviceBuffer>> handles;
    handles.reserve(baseBatchHdl.batchSize());
    for (int64_t i = 0; i < baseBatchHdl.batchSize(); ++i) {
        if (baseBatchHdl.numVoxelsAt(i) == 0) {
            handles.push_back(createEmptyGridHandle(baseBatchHdl.device()));
            continue;
        }

        nanovdb::OnIndexGrid *grid = baseBatchHdl.mGridHdl->deviceGrid<nanovdb::ValueOnIndex>(i);
        TORCH_CHECK(grid, "Grid is null");

        nanovdb::GridHandle<TorchDeviceBuffer> handle;
        bool haveHandle = false;
        for (int p = 0; p < totalPasses; ++p) {
            const bool positive = (p < numPositive);
            if (!excludeBorder) {
                handle = padOncePass(grid, positive, guide, stream.stream());
            } else {
                const uint32_t leafCount =
                    haveHandle
                        ? nanovdb::util::cuda::DeviceGridTraits<nanovdb::ValueOnIndex>::getTreeData(
                              grid)
                              .mNodeCount[0]
                        : baseBatchHdl.numLeavesAt(i);
                if (leafCount == 0) {
                    break; // already eroded to empty; further erosion is a no-op
                }
                handle = erodeOncePass(
                    grid, leafCount, positive, baseBatchHdl.device(), guide, stream.stream());
            }
            haveHandle = true;
            grid       = handle.deviceGrid<nanovdb::ValueOnIndex>();
        }

        handles.push_back(std::move(handle));
    }

    if (handles.size() == 1) {
        return std::move(handles[0]);
    }
    return nanovdb::cuda::mergeGridHandles(handles, &guide);
}

template <>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchBuildPaddedGrid<torch::kPrivateUse1>(const GridBatchData &baseBatchHdl,
                                             int bmin,
                                             int bmax,
                                             bool excludeBorder) {
    // Multi-GPU / PrivateUse1 keeps the coordinate-list path (TopologyBuilder-based morphology
    // is single-device). The exclude-border variant was never supported here (its old code path
    // called a CUDA-only helper that asserts is_cuda), so reject it explicitly rather than
    // crash obscurely.
    TORCH_CHECK(!excludeBorder,
                "dual_grid/build_padded_grid with exclude_border=True is not supported on "
                "PrivateUse1 (multi-GPU) devices");
    nanovdb::CoordBBox bbox = nanovdb::CoordBBox::createCube(bmin, bmax);
    JaggedTensor coords     = paddedIJKForGrid<torch::kPrivateUse1>(baseBatchHdl, bbox);
    return ops::_createNanoGridFromIJK(coords);
}

template <>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchBuildPaddedGrid<torch::kCPU>(const GridBatchData &baseBatchHdl,
                                     int bmin,
                                     int bmax,
                                     bool excludeBorder) {
    if (excludeBorder) {
        return buildPaddedGridFromGridWithoutBorderCPU(baseBatchHdl, bmin, bmax);
    } else {
        return buildPaddedGridFromGridCPU(baseBatchHdl, bmin, bmax);
    }
}

c10::intrusive_ptr<GridBatchData>
buildPaddedGrid(const GridBatchData &baseBatchHdl, int bmin, int bmax, bool excludeBorder) {
    // The structuring element [bmin, bmax]^3 must contain the origin so the padded grid is a
    // superset of the primal (and the erosion a subset); a box excluding the origin would be a
    // translation, which this op does not model. This also lets the CUDA path decompose the box
    // into unit positive/negative octant passes.
    TORCH_CHECK_VALUE(bmin <= 0 && bmax >= 0,
                      "buildPaddedGrid requires bmin <= 0 <= bmax, got bmin=",
                      bmin,
                      ", bmax=",
                      bmax);
    std::vector<nanovdb::Vec3d> voxS, voxO;
    baseBatchHdl.gridVoxelSizesAndOrigins(voxS, voxO);
    auto hdl = FVDB_DISPATCH_KERNEL(baseBatchHdl.device(), [&]() {
        return dispatchBuildPaddedGrid<DeviceTag>(baseBatchHdl, bmin, bmax, excludeBorder);
    });

    const int64_t bs           = hdl.gridCount();
    const torch::Device device = hdl.buffer().device();

    GridBatchData::GridMetadata *hostMeta   = nullptr;
    GridBatchData::GridMetadata *deviceMeta = nullptr;
    if (device.is_cpu() || device.is_cuda()) {
        hostMeta = allocateHostGridMetadata(bs);
        if (device.is_cuda()) {
            deviceMeta = allocateDeviceGridMetadata(device, bs);
        }
    } else if (device.is_privateuseone()) {
        deviceMeta = allocateUnifiedMemoryGridMetadata(bs);
        hostMeta   = deviceMeta;
    }

    torch::Tensor batchOffsets;
    GridBatchData::GridBatchMetadata batchMeta;
    ops::populateGridMetadata(hdl, voxS, voxO, batchOffsets, hostMeta, deviceMeta, &batchMeta);
    batchMeta.mIsContiguous = true;

    for (int64_t i = 0; i < bs; i++) {
        hostMeta[i].mDualTransform   = baseBatchHdl.mHostGridMetadata[i].mPrimalTransform;
        hostMeta[i].mPrimalTransform = baseBatchHdl.mHostGridMetadata[i].mDualTransform;
        hostMeta[i].mVoxelSize       = baseBatchHdl.mHostGridMetadata[i].mVoxelSize;
    }
    syncMetadataToDevice(hostMeta, deviceMeta, bs, device, true);

    const torch::Tensor listIndices =
        torch::empty({0, 1}, torch::TensorOptions().dtype(fvdb::JLIdxScalarType).device(device));
    std::vector<torch::Tensor> leafBatchIdxs;
    leafBatchIdxs.reserve(bs);
    for (int64_t i = 0; i < bs; i += 1) {
        leafBatchIdxs.push_back(
            torch::full({hostMeta[i].mNumLeaves},
                        static_cast<fvdb::JIdxType>(i),
                        torch::TensorOptions().dtype(fvdb::JIdxScalarType).device(device)));
    }
    torch::Tensor leafBatchIndices = torch::cat(leafBatchIdxs, 0);

    auto gridHdlPtr = std::make_shared<nanovdb::GridHandle<TorchDeviceBuffer>>(std::move(hdl));

    return c10::make_intrusive<GridBatchData>(std::move(gridHdlPtr),
                                              hostMeta,
                                              deviceMeta,
                                              bs,
                                              std::move(batchMeta),
                                              std::move(leafBatchIndices),
                                              std::move(batchOffsets),
                                              std::move(listIndices));
}

} // namespace ops
} // namespace detail
} // namespace fvdb
