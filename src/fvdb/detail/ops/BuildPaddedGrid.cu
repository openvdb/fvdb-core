// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/BuilderResource.h>
#include <fvdb/GridBatchData.h>
#include <fvdb/GridStorage.h>
#include <fvdb/detail/GridBatchDataFactory.h>
#include <fvdb/detail/ops/BuildGridFromIjk.h>
#include <fvdb/detail/ops/BuildPaddedGrid.h>
#include <fvdb/detail/ops/MakeContiguous.h>
#include <fvdb/detail/ops/PopulateGridMetadata.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/cuda/ForEachCUDA.cuh>
#include <fvdb/detail/utils/cuda/ForEachPrivateUse1.cuh>
#include <fvdb/detail/utils/cuda/GridDim.h>
#include <fvdb/detail/utils/cuda/StreamOrdering.h>
#include <fvdb/detail/utils/nanovdb/DeviceGridHandleUtils.cuh>
#include <fvdb/detail/utils/nanovdb/PadGrid.cuh>

#include <nanovdb/HostBuffer.h>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/cuda/PruneGrid.cuh>
#include <nanovdb/util/cuda/DeviceGridTraits.cuh>
#include <nanovdb/util/cuda/Util.h>

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMathCompat.h>

#include <optional>

namespace fvdb {
namespace detail {
namespace ops {

template <torch::DeviceType>
GridStorage
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

GridStorage
buildPaddedGridFromGridWithoutBorderCPU(const GridBatchData &baseBatchHdl, int BMIN, int BMAX) {
    using GridT = nanovdb::ValueOnIndex;

    TORCH_CHECK(BMIN <= BMAX, "BMIN must be less than BMAX");

    std::vector<nanovdb::GridHandle<nanovdb::HostBuffer>> batchHandles;
    batchHandles.reserve(baseBatchHdl.batchSize());
    for (int64_t i = 0; i < baseBatchHdl.batchSize(); i += 1) {
        // View-aware byte-offset accessor: the i-th *logical* grid (correct for sliced/
        // non-contiguous batches, unlike grid<GridT>(i) which indexes physically).
        const nanovdb::OnIndexGrid *baseGrid = baseBatchHdl.hostGridPtrAt(i);
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
        batchHandles.push_back(
            nanovdb::tools::createNanoGrid<ProxyGridT, GridT, nanovdb::HostBuffer>(
                *proxyGrid, 0u, false, false));
    }

    return GridStorage(batchHandles.size() == 1 ? std::move(batchHandles[0])
                                                : nanovdb::mergeGrids(batchHandles));
}

GridStorage
buildPaddedGridFromGridCPU(const GridBatchData &baseBatchHdl, int BMIN, int BMAX) {
    using GridT = nanovdb::ValueOnIndex;

    TORCH_CHECK(BMIN <= BMAX, "BMIN must be less than BMAX");

    std::vector<nanovdb::GridHandle<nanovdb::HostBuffer>> batchHandles;
    batchHandles.reserve(baseBatchHdl.batchSize());
    for (int64_t i = 0; i < baseBatchHdl.batchSize(); i += 1) {
        // View-aware byte-offset accessor: the i-th *logical* grid (correct for sliced/
        // non-contiguous batches, unlike grid<GridT>(i) which indexes physically).
        const nanovdb::OnIndexGrid *baseGrid = baseBatchHdl.hostGridPtrAt(i);
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
        batchHandles.push_back(
            nanovdb::tools::createNanoGrid<ProxyGridT, GridT, nanovdb::HostBuffer>(
                *proxyGrid, 0u, false, false));
    }

    return GridStorage(batchHandles.size() == 1 ? std::move(batchHandles[0])
                                                : nanovdb::mergeGrids(batchHandles));
}

// One unit padding pass (Minkowski sum by {0,1}^3 if positive, else {-1,0}^3), building the
// padded topology directly from the source leaf masks via TopologyBuilder (no coordinate list).
// The result is allocated from `proto`'s resource on `stream`, which must be `proto`'s retained
// stream.
static GridStorage::DeviceHandle
padOncePass(nanovdb::OnIndexGrid *grid,
            bool positive,
            const DeviceGridBuffer &proto,
            cudaStream_t stream) {
    fvdb::detail::morphology::PadGrid<nanovdb::ValueOnIndex, BuilderResource> op(
        grid, positive, stream);
    auto handle = op.getHandle(proto);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return handle;
}

// One unit erosion pass (exclude-border padding). Computes a per-leaf "keep" mask sidecar
// (voxel survives iff its whole octant neighborhood is active) then prunes the source grid to
// it via PruneGrid. The keep mask is a subset of the source, so the result is exact. Returns
// nullopt when the erosion removes every voxel; the caller substitutes an empty grid.
static std::optional<GridStorage::DeviceHandle>
erodeOncePass(nanovdb::OnIndexGrid *grid,
              uint32_t leafCount,
              bool positive,
              const torch::Device &device,
              const DeviceGridBuffer &proto,
              cudaStream_t stream) {
    // Allocate the per-leaf keep-mask sidecar as a torch CUDA tensor so its emptiness can be
    // tested reliably with a torch reduction below (a raw device buffer viewed via from_blob does
    // not reduce correctly).
    // Use a uint64 tensor (not uint8) so its data pointer is guaranteed 8-byte aligned for
    // Mask<3> -- which is 8 uint64_t words -- since the kernels dereference it as Mask<3>*.
    // (A uint8 tensor's data_ptr alignment isn't guaranteed by the tensor API, even though
    // torch's allocators over-align in practice.)
    const int64_t maskWords =
        static_cast<int64_t>(sizeof(nanovdb::Mask<3>) / sizeof(uint64_t)) * leafCount;
    torch::Tensor keepTensor =
        torch::empty({maskWords}, torch::TensorOptions().dtype(torch::kUInt64).device(device));
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
    // everything) and report it so the caller emits an explicit empty grid instead -- same guard
    // as BuildPrunedGrid.cu.
    // Reduce via (!= 0).any(): any() alone (an `or` reduction) is not implemented for uint64 on
    // CUDA, but the `!=` yields a bool tensor whose reduction is.
    if (!(keepTensor != 0).any().item<bool>()) {
        return std::nullopt;
    }

    nanovdb::tools::cuda::PruneGrid<nanovdb::ValueOnIndex, BuilderResource> pruneOp(
        grid, keepMasks, stream);
    pruneOp.setVerbose(0);
    auto handle = pruneOp.getHandle(proto);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return handle;
}

template <>
GridStorage
dispatchBuildPaddedGrid<torch::kCUDA>(const GridBatchData &baseBatchHdl,
                                      int bmin,
                                      int bmax,
                                      bool excludeBorder) {
    const torch::Device device = baseBatchHdl.device();
    c10::cuda::CUDAGuard deviceGuard(device);
    // The grids are built and their storage retained on the device's current torch stream; the
    // prototype carries that stream and the device into every allocation the builders make.
    const cudaStream_t stream    = storageStream(device);
    const DeviceGridBuffer proto = GridStorage::deviceProto(device, stream);

    // Pad by [bmin, bmax]^3 = (bmax positive unit passes) followed by (-bmin negative unit
    // passes). Minkowski sums / erosions by boxes compose, so the order is immaterial.
    // dual_grid (0, 1) is exactly one positive pass.
    const int numPositive = bmax;
    const int numNegative = -bmin;
    const int totalPasses = numPositive + numNegative;

    // Identity case (bmin == bmax == 0): no morphology passes run, so the result is a copy of the
    // selected grids: a byte copy with header fix-up, no radix sort and no joffsets().cpu() sync.
    // (contiguousGridStorage copies by logical grid, so a sliced / non-contiguous batch, whose
    // storage holds MORE grids than batchSize(), comes out right too.) This is rare and degenerate
    // -- only build_padded_grid(0, 0) reaches it; dual_grid, being (0, 1), never does. The tail
    // then applies the transform fix-up (dual swap or verbatim copy, per `dualTransform`).
    if (totalPasses == 0) {
        return ops::contiguousGridStorage(baseBatchHdl);
    }

    // Build one grid per batch item, then lay them end to end in one storage.
    GridStorageParts parts(device, stream, baseBatchHdl.batchSize());
    for (int64_t i = 0; i < baseBatchHdl.batchSize(); ++i) {
        if (baseBatchHdl.numVoxelsAt(i) == 0) {
            parts.addEmpty();
            continue;
        }

        // View-aware byte-offset accessor: the i-th *logical* grid (correct for sliced/
        // non-contiguous batches, unlike gridStorage().deviceGridAt(i) which indexes physically).
        nanovdb::OnIndexGrid *grid = baseBatchHdl.deviceGridPtrAt(i);

        // The passes chain on the device handle; only the final one is wrapped as storage.
        // `handle` is empty (default-constructed) until the first pass produces a grid.
        GridStorage::DeviceHandle handle;
        bool haveHandle    = false;
        bool erodedToEmpty = false;
        for (int p = 0; p < totalPasses; ++p) {
            const bool positive = (p < numPositive);
            if (!excludeBorder) {
                handle = padOncePass(grid, positive, proto, stream);
            } else {
                const uint32_t leafCount =
                    haveHandle
                        ? nanovdb::util::cuda::DeviceGridTraits<nanovdb::ValueOnIndex>::getTreeData(
                              grid)
                              .mNodeCount[0]
                        : baseBatchHdl.numLeavesAt(i);
                if (leafCount == 0) {
                    // No leaves to erode -- (defensively) a leaf-free tile grid, or a prior pass
                    // left a grid without leaves. Without a handle yet, the item is empty.
                    if (!haveHandle) {
                        erodedToEmpty = true;
                    }
                    break;
                }
                auto eroded = erodeOncePass(grid, leafCount, positive, device, proto, stream);
                if (!eroded) {
                    // Erosion removed every voxel; further passes cannot bring anything back.
                    erodedToEmpty = true;
                    break;
                }
                handle = std::move(*eroded);
            }
            haveHandle = true;
            grid       = handle.deviceGrid<nanovdb::ValueOnIndex>();
        }

        if (erodedToEmpty) {
            parts.addEmpty();
        } else {
            parts.add(GridStorage(std::move(handle), device));
        }
    }

    return parts.merge();
}

template <>
GridStorage
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
GridStorage
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
buildPaddedGrid(
    const GridBatchData &baseBatchHdl, int bmin, int bmax, bool excludeBorder, bool dualTransform) {
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
    GridStorage built = FVDB_DISPATCH_KERNEL(baseBatchHdl.device(), [&]() {
        return dispatchBuildPaddedGrid<DeviceTag>(baseBatchHdl, bmin, bmax, excludeBorder);
    });

    auto storage               = std::make_shared<GridStorage>(std::move(built));
    const int64_t bs           = storage->gridCount();
    const torch::Device device = storage->device();

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
    ops::populateGridMetadata(*storage, voxS, voxO, batchOffsets, hostMeta, deviceMeta, &batchMeta);
    batchMeta.mIsContiguous = true;

    // Fix up the per-grid transforms. populateGridMetadata already recomputed primal/dual
    // transforms from the source's (voxelSize, origin), so here we just carry the source's stored
    // transforms over verbatim -- either swapped (dual) or as-is (plain pad).
    for (int64_t i = 0; i < bs; i++) {
        const auto &srcMeta = baseBatchHdl.mHostGridMetadata[i];
        if (dualTransform) {
            // dual_grid: result voxels sit at the *corners* of the source voxels, so the source's
            // dual (corner-aligned) transform becomes the result's primal (center-aligned)
            // transform, and vice versa. This shifts the origin by half a voxel.
            hostMeta[i].mPrimalTransform = srcMeta.mDualTransform;
            hostMeta[i].mDualTransform   = srcMeta.mPrimalTransform;
        } else {
            // Plain padded grid: same lattice as the source, so keep its transforms unchanged.
            hostMeta[i].mPrimalTransform = srcMeta.mPrimalTransform;
            hostMeta[i].mDualTransform   = srcMeta.mDualTransform;
        }
        hostMeta[i].mVoxelSize = srcMeta.mVoxelSize;
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

    return c10::make_intrusive<GridBatchData>(std::move(storage),
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
