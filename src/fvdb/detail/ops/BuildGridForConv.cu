// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/GridBatchData.h>
#include <fvdb/detail/GridBatchDataFactory.h>
#include <fvdb/detail/ops/BuildCoarseGridFromFine.h>
#include <fvdb/detail/ops/BuildGridForConv.h>
#include <fvdb/detail/ops/BuildGridFromIjk.h>
#include <fvdb/detail/ops/CoarseIjkForFineGrid.h>
#include <fvdb/detail/ops/convolution/ConvolutionGeometry.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/cuda/ForEachCUDA.cuh>
#include <fvdb/detail/utils/nanovdb/CreateEmptyGridHandle.h>
#include <fvdb/detail/utils/nanovdb/PadGrid.cuh>

#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/cuda/DilateGrid.cuh>
#include <nanovdb/util/MorphologyHelpers.h>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/types.h>

#include <algorithm>
#include <limits>

namespace fvdb {
namespace detail {
namespace ops {

namespace {

// Set by the last dispatchBuildGridForConv call on this thread. Read it on the same thread
// before any subsequent build; this diagnostic intentionally has last-call-wins semantics.
thread_local BuildGridForConvResourceStats gLastBuildGridForConvResourceStats;

int64_t
checkedAddInt64(int64_t lhs, int64_t rhs, const char *description) {
    TORCH_CHECK_VALUE(lhs >= 0 && rhs >= 0, description, " must be nonnegative");
    TORCH_CHECK_VALUE(
        rhs <= std::numeric_limits<int64_t>::max() - lhs, description, " overflows int64");
    return lhs + rhs;
}

int64_t
checkedMultiplyInt64(int64_t lhs, int64_t rhs, const char *description) {
    TORCH_CHECK_VALUE(lhs >= 0 && rhs >= 0, description, " must be nonnegative");
    TORCH_CHECK_VALUE(lhs == 0 || rhs <= std::numeric_limits<int64_t>::max() / lhs,
                      description,
                      " overflows int64");
    return lhs * rhs;
}

uint64_t
checkedBytes(int64_t count, uint64_t bytesPerElement, const char *description) {
    TORCH_CHECK_VALUE(count >= 0, description, " count must be nonnegative");
    const uint64_t unsignedCount = static_cast<uint64_t>(count);
    TORCH_CHECK_VALUE(unsignedCount == 0 ||
                          bytesPerElement <= std::numeric_limits<uint64_t>::max() / unsignedCount,
                      description,
                      " byte count overflows uint64");
    return unsignedCount * bytesPerElement;
}

uint64_t
checkedAdd(uint64_t lhs, uint64_t rhs, const char *description) {
    TORCH_CHECK_VALUE(rhs <= std::numeric_limits<uint64_t>::max() - lhs,
                      description,
                      " byte count overflows uint64");
    return lhs + rhs;
}

bool
isDirectProjection(ConvolutionGeometry const &geometry) {
    return geometry.kernelSize() == geometry.stride();
}

bool
isUnshiftedDirectProjection(ConvolutionGeometry const &geometry) {
    return isDirectProjection(geometry) && geometry.paddingBefore() == nanovdb::Coord(0);
}

BuildGridForConvResourceStats
directProjectionStats(int64_t inputVoxelCount,
                      int64_t kernelVolume,
                      bool usesCoordinateStaging = true) {
    BuildGridForConvResourceStats stats;
    stats.inputVoxelCount        = inputVoxelCount;
    stats.kernelVolume           = kernelVolume;
    stats.validEmissionCount     = inputVoxelCount;
    stats.emissionRequestedBytes = usesCoordinateStaging
                                       ? checkedBytes(inputVoxelCount,
                                                      3 * sizeof(int32_t) + sizeof(fvdb::JIdxType),
                                                      "direct projection staging")
                                       : 0;
    stats.peakRequestedBytes     = stats.emissionRequestedBytes;
    stats.usedDirectProjection   = true;
    return stats;
}

BuildGridForConvResourceStats
morphologyStats(int64_t inputVoxelCount, int64_t kernelVolume) {
    BuildGridForConvResourceStats stats;
    stats.inputVoxelCount = inputVoxelCount;
    stats.kernelVolume    = kernelVolume;
    stats.validEmissionCount =
        checkedMultiplyInt64(inputVoxelCount, kernelVolume, "morphology emission count");
    return stats;
}

bool
isUniformKernel(ConvolutionGeometry const &geometry) {
    nanovdb::Coord const &kernelSize = geometry.kernelSize();
    return kernelSize[0] == kernelSize[1] && kernelSize[1] == kernelSize[2];
}

bool
supportsLeafMaskDirectProjection(ConvolutionGeometry const &geometry) {
    return isUnshiftedDirectProjection(geometry) && isUniformKernel(geometry);
}

BuildGridForConvResourceStats
countThenFillStats(int64_t inputVoxelCount, int64_t kernelVolume, int64_t validEmissionCount) {
    BuildGridForConvResourceStats stats;
    stats.inputVoxelCount    = inputVoxelCount;
    stats.kernelVolume       = kernelVolume;
    stats.validEmissionCount = validEmissionCount;
    stats.countRequestedBytes =
        checkedBytes(inputVoxelCount, sizeof(int32_t), "forward count staging");
    stats.prefixRequestedBytes =
        checkedBytes(inputVoxelCount + 1, sizeof(int64_t), "forward prefix staging");
    stats.emissionRequestedBytes = checkedBytes(validEmissionCount,
                                                3 * sizeof(int32_t) + sizeof(fvdb::JIdxType),
                                                "forward emission staging");
    stats.peakRequestedBytes     = std::max(checkedAdd(stats.countRequestedBytes,
                                                   stats.prefixRequestedBytes,
                                                   "forward count/prefix staging"),
                                        checkedAdd(stats.prefixRequestedBytes,
                                                   stats.emissionRequestedBytes,
                                                   "forward prefix/emission staging"));
    return stats;
}

void
checkForwardInputAndKernel(int64_t inputVoxelCount, ConvolutionGeometry const &geometry) {
    TORCH_CHECK_VALUE(inputVoxelCount >= 0, "input voxel count must be nonnegative");
    TORCH_CHECK_VALUE(inputVoxelCount < std::numeric_limits<int64_t>::max(),
                      "input voxel count is too large for a prefix array");
    TORCH_CHECK_VALUE(geometry.kernelVolume() <= std::numeric_limits<int32_t>::max(),
                      "kernel volume exceeds int32 count capacity");
    (void)checkedBytes(inputVoxelCount, sizeof(int32_t), "forward count staging");
    (void)checkedBytes(inputVoxelCount + 1, sizeof(int64_t), "forward prefix staging");
}

} // namespace

BuildGridForConvResourceStats
lastBuildGridForConvResourceStats() {
    return gLastBuildGridForConvResourceStats;
}

template <torch::DeviceType>
nanovdb::GridHandle<TorchDeviceBuffer> dispatchBuildGridForConv(const GridBatchData &baseBatchHdl,
                                                                const nanovdb::Coord &kernelSize,
                                                                const nanovdb::Coord &stride);

nanovdb::GridHandle<TorchDeviceBuffer>
buildCoarseGridFromFineGridCPU(const GridBatchData &fineBatchHdl,
                               const nanovdb::Coord branchingFactor) {
    using GridT     = nanovdb::ValueOnIndex;
    using IndexTree = nanovdb::NanoTree<GridT>;

    std::vector<nanovdb::GridHandle<TorchDeviceBuffer>> batchHandles;
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
        auto ret = nanovdb::tools::createNanoGrid<ProxyGridT, GridT, TorchDeviceBuffer>(
            *proxyGrid, 0u, false, false);
        ret.buffer().to(torch::kCPU);
        batchHandles.push_back(std::move(ret));
    }
    return batchHandles.size() == 1 ? std::move(batchHandles[0])
                                    : nanovdb::mergeGrids(batchHandles);
}

__device__ void
countConvIJKForGridCallback(int32_t bidx,
                            int32_t lidx,
                            int32_t vidx,
                            int32_t,
                            GridBatchData::Accessor batchAcc,
                            ConvolutionGeometry geometry,
                            TorchRAcc64<int32_t, 1> counts) {
    const nanovdb::OnIndexGrid *gridPtr = batchAcc.grid(bidx);
    const typename nanovdb::OnIndexGrid::LeafNodeType &leaf =
        gridPtr->tree().template getFirstNode<0>()[lidx];
    if (!leaf.isActive(vidx)) {
        return;
    }

    const nanovdb::Coord srcIjk = leaf.offsetToGlobalCoord(vidx);
    const int64_t sourceIndex =
        batchAcc.voxelOffset(bidx) + static_cast<int64_t>(leaf.getValue(vidx)) - 1;
    int32_t count = 0;
    for (int64_t tapIndex = 0; tapIndex < geometry.kernelVolume(); ++tapIndex) {
        nanovdb::Coord coarse;
        if (geometry.coarseFromFine(srcIjk, geometry.tapCoord(tapIndex), coarse)) {
            count += 1;
        }
    }
    counts[sourceIndex] = count;
}

__device__ void
fillConvIJKForGridCallback(int32_t bidx,
                           int32_t lidx,
                           int32_t vidx,
                           int32_t,
                           GridBatchData::Accessor batchAcc,
                           ConvolutionGeometry geometry,
                           TorchRAcc64<int64_t, 1> prefix,
                           TorchRAcc64<int32_t, 2> outIJK,
                           TorchRAcc64<fvdb::JIdxType, 1> outIJKBIdx) {
    const nanovdb::OnIndexGrid *gridPtr = batchAcc.grid(bidx);
    const typename nanovdb::OnIndexGrid::LeafNodeType &leaf =
        gridPtr->tree().template getFirstNode<0>()[lidx];
    if (!leaf.isActive(vidx)) {
        return;
    }

    const nanovdb::Coord srcIjk = leaf.offsetToGlobalCoord(vidx);
    const int64_t sourceIndex =
        batchAcc.voxelOffset(bidx) + static_cast<int64_t>(leaf.getValue(vidx)) - 1;
    int64_t writeIndex = prefix[sourceIndex];
    for (int64_t tapIndex = 0; tapIndex < geometry.kernelVolume(); ++tapIndex) {
        nanovdb::Coord coarse;
        if (!geometry.coarseFromFine(srcIjk, geometry.tapCoord(tapIndex), coarse)) {
            continue;
        }
        outIJK[writeIndex][0]  = coarse[0];
        outIJK[writeIndex][1]  = coarse[1];
        outIJK[writeIndex][2]  = coarse[2];
        outIJKBIdx[writeIndex] = bidx;
        writeIndex += 1;
    }
}

__device__ void
directConvIJKForGridCallback(int32_t bidx,
                             int32_t lidx,
                             int32_t vidx,
                             int32_t,
                             GridBatchData::Accessor batchAcc,
                             ConvolutionGeometry geometry,
                             TorchRAcc64<int32_t, 2> outIJK,
                             TorchRAcc64<fvdb::JIdxType, 1> outIJKBIdx) {
    const nanovdb::OnIndexGrid *gridPtr = batchAcc.grid(bidx);
    const typename nanovdb::OnIndexGrid::LeafNodeType &leaf =
        gridPtr->tree().template getFirstNode<0>()[lidx];
    if (!leaf.isActive(vidx)) {
        return;
    }

    const nanovdb::Coord srcIjk = leaf.offsetToGlobalCoord(vidx);
    const int64_t sourceIndex =
        batchAcc.voxelOffset(bidx) + static_cast<int64_t>(leaf.getValue(vidx)) - 1;
    const nanovdb::Coord &paddingBefore = geometry.paddingBefore();
    const nanovdb::Coord &stride        = geometry.stride();
    outIJK[sourceIndex][0]              = static_cast<int32_t>(ConvolutionGeometry::floorDiv(
        static_cast<int64_t>(srcIjk[0]) + paddingBefore[0], stride[0]));
    outIJK[sourceIndex][1]              = static_cast<int32_t>(ConvolutionGeometry::floorDiv(
        static_cast<int64_t>(srcIjk[1]) + paddingBefore[1], stride[1]));
    outIJK[sourceIndex][2]              = static_cast<int32_t>(ConvolutionGeometry::floorDiv(
        static_cast<int64_t>(srcIjk[2]) + paddingBefore[2], stride[2]));
    outIJKBIdx[sourceIndex]             = bidx;
}

JaggedTensor
directConvIJKForGrid(const GridBatchData &batchHdl, ConvolutionGeometry const &geometry) {
    const int64_t inputVoxelCount = batchHdl.totalVoxels();
    gLastBuildGridForConvResourceStats =
        directProjectionStats(inputVoxelCount, geometry.kernelVolume());

    const auto dataOptions = torch::TensorOptions().dtype(torch::kInt32).device(batchHdl.device());
    const auto batchOptions =
        torch::TensorOptions().dtype(fvdb::JIdxScalarType).device(batchHdl.device());
    torch::Tensor outIJK     = torch::empty({inputVoxelCount, 3}, dataOptions);
    torch::Tensor outIJKBIdx = torch::empty({inputVoxelCount}, batchOptions);
    auto outIJKAcc           = outIJK.packed_accessor64<int32_t, 2, torch::RestrictPtrTraits>();
    auto outIJKBIdxAcc =
        outIJKBIdx.packed_accessor64<fvdb::JIdxType, 1, torch::RestrictPtrTraits>();
    auto callback = [=] __device__(int32_t bidx,
                                   int32_t lidx,
                                   int32_t vidx,
                                   int32_t cidx,
                                   GridBatchData::Accessor batchAcc) {
        directConvIJKForGridCallback(
            bidx, lidx, vidx, cidx, batchAcc, geometry, outIJKAcc, outIJKBIdxAcc);
    };
    forEachVoxelCUDA(1, batchHdl, callback);
    return JaggedTensor::from_data_indices_and_list_ids(
        outIJK, outIJKBIdx, batchHdl.jlidx(), batchHdl.batchSize());
}

JaggedTensor
countThenFillConvIJKForGrid(const GridBatchData &batchHdl, ConvolutionGeometry const &geometry) {
    const int64_t inputVoxelCount = batchHdl.totalVoxels();
    const auto countOptions = torch::TensorOptions().dtype(torch::kInt32).device(batchHdl.device());
    torch::Tensor counts    = torch::empty({inputVoxelCount}, countOptions);
    auto countAcc           = counts.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>();
    auto countCallback      = [=] __device__(int32_t bidx,
                                        int32_t lidx,
                                        int32_t vidx,
                                        int32_t cidx,
                                        GridBatchData::Accessor batchAcc) {
        countConvIJKForGridCallback(bidx, lidx, vidx, cidx, batchAcc, geometry, countAcc);
    };
    forEachVoxelCUDA(1, batchHdl, countCallback);

    torch::Tensor prefix =
        torch::zeros({inputVoxelCount + 1},
                     torch::TensorOptions().dtype(torch::kInt64).device(batchHdl.device()));
    prefix.slice(0, 1, inputVoxelCount + 1).copy_(torch::cumsum(counts, 0, torch::kInt64));
    const int64_t validEmissionCount = prefix[-1].item<int64_t>();
    TORCH_CHECK_VALUE(validEmissionCount >= 0, "forward valid emission count must be nonnegative");
    gLastBuildGridForConvResourceStats =
        countThenFillStats(inputVoxelCount, geometry.kernelVolume(), validEmissionCount);

    counts = torch::Tensor();

    const auto dataOptions = torch::TensorOptions().dtype(torch::kInt32).device(batchHdl.device());
    const auto batchOptions =
        torch::TensorOptions().dtype(fvdb::JIdxScalarType).device(batchHdl.device());
    torch::Tensor outIJK     = torch::empty({validEmissionCount, 3}, dataOptions);
    torch::Tensor outIJKBIdx = torch::empty({validEmissionCount}, batchOptions);
    auto prefixAcc           = prefix.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>();
    auto outIJKAcc           = outIJK.packed_accessor64<int32_t, 2, torch::RestrictPtrTraits>();
    auto outIJKBIdxAcc =
        outIJKBIdx.packed_accessor64<fvdb::JIdxType, 1, torch::RestrictPtrTraits>();
    auto fillCallback = [=] __device__(int32_t bidx,
                                       int32_t lidx,
                                       int32_t vidx,
                                       int32_t cidx,
                                       GridBatchData::Accessor batchAcc) {
        fillConvIJKForGridCallback(
            bidx, lidx, vidx, cidx, batchAcc, geometry, prefixAcc, outIJKAcc, outIJKBIdxAcc);
    };
    forEachVoxelCUDA(1, batchHdl, fillCallback);

    prefix = torch::Tensor();
    return JaggedTensor::from_data_indices_and_list_ids(
        outIJK, outIJKBIdx, batchHdl.jlidx(), batchHdl.batchSize());
}

// Applies fn(grid) -> handle to each logical non-empty batch item, preserves empties, then merges.
template <typename PerGridFn>
nanovdb::GridHandle<TorchDeviceBuffer>
perItemGridHandle(const GridBatchData &base, const TorchDeviceBuffer &guide, PerGridFn &&fn) {
    std::vector<nanovdb::GridHandle<TorchDeviceBuffer>> handles;
    handles.reserve(base.batchSize());
    for (int64_t i = 0; i < base.batchSize(); i += 1) {
        if (base.numVoxelsAt(i) == 0) {
            handles.push_back(createEmptyGridHandle(base.device()));
            continue;
        }

        nanovdb::OnIndexGrid *grid = base.deviceGridPtrAt(i);
        TORCH_CHECK(grid, "Grid is null");
        handles.push_back(fn(grid));
    }
    return handles.size() == 1 ? std::move(handles[0])
                               : nanovdb::cuda::mergeGridHandles(handles, &guide);
}

template <>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchBuildGridForConv<torch::kCUDA>(const GridBatchData &baseGridHdl,
                                       const nanovdb::Coord &kernelSize,
                                       const nanovdb::Coord &stride) {
    ConvolutionGeometry const geometry(kernelSize, stride);
    checkForwardInputAndKernel(baseGridHdl.totalVoxels(), geometry);

    // NanoVDB can perform the unshifted K=S={1,2} projection directly on leaf masks.
    // Larger K=S geometries have a nonzero canonical phase and use the exact quotient below.
    if (supportsLeafMaskDirectProjection(geometry)) {
        gLastBuildGridForConvResourceStats =
            directProjectionStats(baseGridHdl.totalVoxels(), geometry.kernelVolume(), false);
        return coarseGridHandleFromFineCUDA(baseGridHdl, geometry.stride());
    }
    if (isUnshiftedDirectProjection(geometry)) {
        gLastBuildGridForConvResourceStats =
            directProjectionStats(baseGridHdl.totalVoxels(), geometry.kernelVolume());
        return ops::_createNanoGridFromIJK(coarseIJKForFineGrid(baseGridHdl, geometry.stride()));
    }

    // At stride one the canonical forward support is source (+)
    // [-paddingAfter, paddingBefore]^3. Realize that box with NanoVDB morphology.
    if (geometry.stride() == nanovdb::Coord(1) && isUniformKernel(geometry) &&
        geometry.kernelSize()[0] > 1) {
        gLastBuildGridForConvResourceStats =
            morphologyStats(baseGridHdl.totalVoxels(), geometry.kernelVolume());
        c10::cuda::CUDAGuard deviceGuard(baseGridHdl.device());
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(baseGridHdl.device().index());
        TorchDeviceBuffer guide(0, baseGridHdl.device());
        const int k = geometry.kernelSize()[0];

        return perItemGridHandle(baseGridHdl, guide, [&](nanovdb::OnIndexGrid *grid) {
            nanovdb::GridHandle<TorchDeviceBuffer> handle;
            if (k % 2 == 1) {
                for (int p = 0; p < geometry.paddingBefore()[0]; p += 1) {
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
                for (int p = 0; p < geometry.paddingAfter()[0]; p += 1) {
                    morphology::PadGrid<nanovdb::ValueOnIndex> op(
                        grid, /*positiveOctant=*/false, stream.stream());
                    op.setChecksum(nanovdb::CheckMode::Default);
                    handle = op.getHandle(guide);
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                    grid = handle.deviceGrid<nanovdb::ValueOnIndex>();
                }
                for (int p = 0; p < geometry.paddingBefore()[0]; p += 1) {
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

    // Shifted K=S uses one exact quotient per input. Other geometries use exact-M count/fill.
    JaggedTensor coords = isDirectProjection(geometry)
                              ? directConvIJKForGrid(baseGridHdl, geometry)
                              : countThenFillConvIJKForGrid(baseGridHdl, geometry);
    return ops::_createNanoGridFromIJK(coords);
}

template <>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchBuildGridForConv<torch::kCPU>(const GridBatchData &baseBatchHdl,
                                      const nanovdb::Coord &kernelSize,
                                      const nanovdb::Coord &stride) {
    using GridT = nanovdb::ValueOnIndex;
    ConvolutionGeometry const geometry(kernelSize, stride);
    checkForwardInputAndKernel(baseBatchHdl.totalVoxels(), geometry);
    if (isUnshiftedDirectProjection(geometry)) {
        gLastBuildGridForConvResourceStats =
            directProjectionStats(baseBatchHdl.totalVoxels(), geometry.kernelVolume());
        return buildCoarseGridFromFineGridCPU(baseBatchHdl, geometry.stride());
    }

    std::vector<nanovdb::GridHandle<TorchDeviceBuffer>> batchHandles;
    batchHandles.reserve(baseBatchHdl.batchSize());
    const bool directProjection = isDirectProjection(geometry);
    int64_t validEmissionCount  = 0;
    for (int64_t bidx = 0; bidx < baseBatchHdl.batchSize(); bidx += 1) {
        const nanovdb::OnIndexGrid *baseGrid = baseBatchHdl.hostGridPtrAt(bidx);
        TORCH_CHECK(baseGrid != nullptr, "Failed to get pointer to nanovdb index grid");
        using ProxyGridT       = nanovdb::tools::build::Grid<float>;
        auto proxyGrid         = std::make_shared<ProxyGridT>(-1.0f);
        auto proxyGridAccessor = proxyGrid->getWriteAccessor();
        for (auto it = ActiveVoxelIterator(baseGrid->tree()); it.isValid(); it++) {
            const nanovdb::Coord fine = it->first;
            if (directProjection) {
                const nanovdb::Coord &paddingBefore = geometry.paddingBefore();
                const nanovdb::Coord &strideValue   = geometry.stride();
                proxyGridAccessor.setValue(
                    nanovdb::Coord(
                        static_cast<int32_t>(ConvolutionGeometry::floorDiv(
                            static_cast<int64_t>(fine[0]) + paddingBefore[0], strideValue[0])),
                        static_cast<int32_t>(ConvolutionGeometry::floorDiv(
                            static_cast<int64_t>(fine[1]) + paddingBefore[1], strideValue[1])),
                        static_cast<int32_t>(ConvolutionGeometry::floorDiv(
                            static_cast<int64_t>(fine[2]) + paddingBefore[2], strideValue[2]))),
                    1.0f);
                continue;
            }
            for (int64_t tapIndex = 0; tapIndex < geometry.kernelVolume(); ++tapIndex) {
                nanovdb::Coord coarse;
                if (geometry.coarseFromFine(fine, geometry.tapCoord(tapIndex), coarse)) {
                    proxyGridAccessor.setValue(coarse, 1.0f);
                    validEmissionCount =
                        checkedAddInt64(validEmissionCount, 1, "CPU emission count");
                }
            }
        }
        proxyGridAccessor.merge();
        batchHandles.push_back(nanovdb::tools::createNanoGrid<ProxyGridT, GridT, TorchDeviceBuffer>(
            *proxyGrid, 0u, false, false));
    }
    gLastBuildGridForConvResourceStats =
        directProjection
            ? directProjectionStats(baseBatchHdl.totalVoxels(), geometry.kernelVolume())
            : countThenFillStats(
                  baseBatchHdl.totalVoxels(), geometry.kernelVolume(), validEmissionCount);
    return batchHandles.size() == 1 ? std::move(batchHandles[0])
                                    : nanovdb::mergeGrids(batchHandles);
}

c10::intrusive_ptr<GridBatchData>
buildGridForConv(const GridBatchData &baseBatchHdl,
                 const nanovdb::Coord &kernelSize,
                 const nanovdb::Coord &stride) {
    ConvolutionGeometry const geometry(kernelSize, stride);
    std::vector<nanovdb::Vec3d> voxS, voxO;
    baseBatchHdl.gridVoxelSizesAndOrigins(voxS, voxO);
    for (auto &voxelSize: voxS) {
        for (int axis = 0; axis < 3; ++axis) {
            voxelSize[axis] *= geometry.stride()[axis];
        }
    }
    auto hdl = FVDB_DISPATCH_KERNEL_DEVICE(baseBatchHdl.device(), [&]() {
        return dispatchBuildGridForConv<DeviceTag>(
            baseBatchHdl, geometry.kernelSize(), geometry.stride());
    });
    return makeGridBatchData(std::move(hdl), voxS, voxO);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
