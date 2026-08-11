// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/GridBatchData.h>
#include <fvdb/detail/GridBatchDataFactory.h>
#include <fvdb/detail/ops/BuildFineGridFromCoarse.h>
#include <fvdb/detail/ops/BuildGridForConvTranspose.h>
#include <fvdb/detail/ops/BuildGridFromIjk.h>
#include <fvdb/detail/ops/convolution/ConvolutionGeometry.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/cuda/ForEachCUDA.cuh>
#include <fvdb/detail/utils/nanovdb/CreateEmptyGridHandle.h>
#include <fvdb/detail/utils/nanovdb/PadGrid.cuh>

#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/cuda/DilateGrid.cuh>
#include <nanovdb/tools/cuda/RefineGrid.cuh>
#include <nanovdb/util/MorphologyHelpers.h>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <torch/types.h>

#include <limits>

namespace fvdb {
namespace detail {
namespace ops {

namespace {

int64_t
checkedMultiply(int64_t lhs, int64_t rhs, const char *description) {
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

bool
isUnshiftedSubdivision(ConvolutionGeometry const &geometry) {
    return geometry.kernelSize() == geometry.stride() &&
           geometry.paddingBefore() == nanovdb::Coord(0);
}

bool
isUniformKernel(ConvolutionGeometry const &geometry) {
    nanovdb::Coord const &kernelSize = geometry.kernelSize();
    return kernelSize[0] == kernelSize[1] && kernelSize[1] == kernelSize[2];
}

bool
supportsLeafMaskSubdivision(ConvolutionGeometry const &geometry) {
    return isUnshiftedSubdivision(geometry) && isUniformKernel(geometry);
}

uint64_t
checkTransposeInputAndKernel(int64_t inputVoxelCount, ConvolutionGeometry const &geometry) {
    TORCH_CHECK_VALUE(inputVoxelCount >= 0, "input voxel count must be nonnegative");
    const int64_t emissionCount = checkedMultiply(
        inputVoxelCount, geometry.kernelVolume(), "transposed-convolution emission count");
    return checkedBytes(emissionCount,
                        3 * sizeof(int32_t) + sizeof(fvdb::JIdxType),
                        "transposed-convolution emission staging");
}

} // namespace

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

    std::vector<nanovdb::GridHandle<TorchDeviceBuffer>> batchHandles;
    batchHandles.reserve(coarseBatchHdl.batchSize());
    for (int64_t bidx = 0; bidx < coarseBatchHdl.batchSize(); bidx += 1) {
        const nanovdb::OnIndexGrid *coarseGrid = coarseBatchHdl.hostGridPtrAt(bidx);
        TORCH_CHECK(coarseGrid != nullptr, "Failed to get pointer to nanovdb index grid");
        const IndexTree &coarseTree = coarseGrid->tree();
        using ProxyGridT            = nanovdb::tools::build::Grid<float>;
        auto proxyGrid              = std::make_shared<ProxyGridT>(-1.0f);
        auto proxyGridAccessor      = proxyGrid->getWriteAccessor();
        for (auto it = ActiveVoxelIterator(coarseTree); it.isValid(); it++) {
            const nanovdb::Coord baseIjk(it->first[0] * subdivisionFactor[0],
                                         it->first[1] * subdivisionFactor[1],
                                         it->first[2] * subdivisionFactor[2]);
            for (int i = 0; i < subdivisionFactor[0]; i += 1) {
                for (int j = 0; j < subdivisionFactor[1]; j += 1) {
                    for (int k = 0; k < subdivisionFactor[2]; k += 1) {
                        proxyGridAccessor.setValue(baseIjk + nanovdb::Coord(i, j, k), 1.0f);
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
    return batchHandles.size() == 1 ? std::move(batchHandles[0])
                                    : nanovdb::mergeGrids(batchHandles);
}

__device__ void
convTransposeIJKForGridCallback(int32_t bidx,
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
    const nanovdb::Coord coarse = leaf.offsetToGlobalCoord(vidx);
    const int64_t sourceIndex =
        batchAcc.voxelOffset(bidx) + static_cast<int64_t>(leaf.getValue(vidx)) - 1;
    const int64_t base = sourceIndex * geometry.kernelVolume();
    for (int64_t tapIndex = 0; tapIndex < geometry.kernelVolume(); ++tapIndex) {
        const nanovdb::Coord fine   = geometry.fineFromCoarse(coarse, geometry.tapCoord(tapIndex));
        outIJK[base + tapIndex][0]  = fine[0];
        outIJK[base + tapIndex][1]  = fine[1];
        outIJK[base + tapIndex][2]  = fine[2];
        outIJKBIdx[base + tapIndex] = bidx;
    }
}

JaggedTensor
convTransposeIJKForGrid(const GridBatchData &batchHdl, ConvolutionGeometry const &geometry) {
    const int64_t inputVoxelCount = batchHdl.totalVoxels();
    const int64_t emissionCount   = checkedMultiply(
        inputVoxelCount, geometry.kernelVolume(), "transposed-convolution emission count");
    const auto dataOptions = torch::TensorOptions().dtype(torch::kInt32).device(batchHdl.device());
    const auto batchOptions =
        torch::TensorOptions().dtype(fvdb::JIdxScalarType).device(batchHdl.device());
    torch::Tensor outIJK     = torch::empty({emissionCount, 3}, dataOptions);
    torch::Tensor outIJKBIdx = torch::empty({emissionCount}, batchOptions);
    auto outIJKAcc           = outIJK.packed_accessor64<int32_t, 2, torch::RestrictPtrTraits>();
    auto outIJKBIdxAcc =
        outIJKBIdx.packed_accessor64<fvdb::JIdxType, 1, torch::RestrictPtrTraits>();
    auto callback = [=] __device__(int32_t bidx,
                                   int32_t lidx,
                                   int32_t vidx,
                                   int32_t cidx,
                                   GridBatchData::Accessor batchAcc) {
        convTransposeIJKForGridCallback(
            bidx, lidx, vidx, cidx, batchAcc, geometry, outIJKAcc, outIJKBIdxAcc);
    };
    forEachVoxelCUDA(1, batchHdl, callback);
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
    ConvolutionGeometry const geometry(kernelSize, stride);

    // NanoVDB realizes the unshifted K=S={1,2} subdivision directly on leaf masks.
    // Shifted K=S geometries retain the canonical -paddingBefore phase in the fallback below.
    if (supportsLeafMaskSubdivision(geometry)) {
        return fineGridHandleFromCoarseCUDA(baseGridHdl, geometry.stride(), std::nullopt);
    }

    c10::cuda::CUDAGuard deviceGuard(baseGridHdl.device());
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(baseGridHdl.device().index());
    TorchDeviceBuffer guide(0, baseGridHdl.device());

    // At stride one the canonical transpose support is source (+)
    // [-paddingBefore, paddingAfter]^3. Realize that box with NanoVDB morphology.
    if (geometry.stride() == nanovdb::Coord(1) && isUniformKernel(geometry) &&
        geometry.kernelSize()[0] > 1) {
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
                for (int p = 0; p < geometry.paddingBefore()[0]; p += 1) {
                    morphology::PadGrid<nanovdb::ValueOnIndex> op(
                        grid, /*positiveOctant=*/false, stream.stream());
                    op.setChecksum(nanovdb::CheckMode::Default);
                    handle = op.getHandle(guide);
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                    grid = handle.deviceGrid<nanovdb::ValueOnIndex>();
                }
                for (int p = 0; p < geometry.paddingAfter()[0]; p += 1) {
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
    if (geometry.stride() == nanovdb::Coord(2) && isUniformKernel(geometry) &&
        geometry.kernelSize()[0] == 3) {
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

    if (isUnshiftedSubdivision(geometry)) {
        return fineGridHandleFromCoarseCUDA(baseGridHdl, geometry.stride(), std::nullopt);
    }

    // Coordinate fallback: preserve exact phase and let the CUDA allocator decide whether the
    // checked request is serviceable. Only this path advertises coordinate-staging context.
    const uint64_t stagingBytes = checkTransposeInputAndKernel(baseGridHdl.totalVoxels(), geometry);
    try {
        return ops::_createNanoGridFromIJK(convTransposeIJKForGrid(baseGridHdl, geometry));
    } catch (const c10::OutOfMemoryError &error) {
        TORCH_CHECK_WITH(
            OutOfMemoryError,
            false,
            "Generative transposed-convolution topology construction ran out of CUDA memory. "
            "Coordinate staging alone requires ",
            stagingBytes,
            " bytes for ",
            baseGridHdl.totalVoxels(),
            " input voxels * ",
            geometry.kernelVolume(),
            " kernel taps. Reduce the input or kernel size, provide an explicit target grid for "
            "restricted transposed convolution, or release CUDA memory before retrying. "
            "Original allocator error: ",
            error.what_without_backtrace());
    }
}

template <>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchBuildGridForConvTranspose<torch::kCPU>(const GridBatchData &baseBatchHdl,
                                               const nanovdb::Coord &kernelSize,
                                               const nanovdb::Coord &stride) {
    using GridT = nanovdb::ValueOnIndex;
    ConvolutionGeometry const geometry(kernelSize, stride);
    checkTransposeInputAndKernel(baseBatchHdl.totalVoxels(), geometry);
    if (isUnshiftedSubdivision(geometry)) {
        return buildFineGridFromCoarseGridCPU(baseBatchHdl, geometry.stride());
    }

    std::vector<nanovdb::GridHandle<TorchDeviceBuffer>> batchHandles;
    batchHandles.reserve(baseBatchHdl.batchSize());
    for (int64_t bidx = 0; bidx < baseBatchHdl.batchSize(); bidx += 1) {
        const nanovdb::OnIndexGrid *baseGrid = baseBatchHdl.hostGridPtrAt(bidx);
        TORCH_CHECK(baseGrid != nullptr, "Failed to get pointer to nanovdb index grid");
        using ProxyGridT       = nanovdb::tools::build::Grid<float>;
        auto proxyGrid         = std::make_shared<ProxyGridT>(-1.0f);
        auto proxyGridAccessor = proxyGrid->getWriteAccessor();
        for (auto it = ActiveVoxelIterator(baseGrid->tree()); it.isValid(); it++) {
            const nanovdb::Coord coarse = it->first;
            for (int64_t tapIndex = 0; tapIndex < geometry.kernelVolume(); ++tapIndex) {
                proxyGridAccessor.setValue(
                    geometry.fineFromCoarse(coarse, geometry.tapCoord(tapIndex)), 1.0f);
            }
        }
        proxyGridAccessor.merge();
        batchHandles.push_back(nanovdb::tools::createNanoGrid<ProxyGridT, GridT, TorchDeviceBuffer>(
            *proxyGrid, 0u, false, false));
    }
    return batchHandles.size() == 1 ? std::move(batchHandles[0])
                                    : nanovdb::mergeGrids(batchHandles);
}

c10::intrusive_ptr<GridBatchData>
buildGridForConvTranspose(const GridBatchData &baseBatchHdl,
                          const nanovdb::Coord &kernelSize,
                          const nanovdb::Coord &stride) {
    ConvolutionGeometry const geometry(kernelSize, stride);
    std::vector<nanovdb::Vec3d> voxS, voxO;
    baseBatchHdl.gridVoxelSizesAndOrigins(voxS, voxO);
    for (auto &voxelSize: voxS) {
        for (int axis = 0; axis < 3; ++axis) {
            voxelSize[axis] /= geometry.stride()[axis];
        }
    }
    auto hdl = FVDB_DISPATCH_KERNEL_DEVICE(baseBatchHdl.device(), [&]() {
        return dispatchBuildGridForConvTranspose<DeviceTag>(
            baseBatchHdl, geometry.kernelSize(), geometry.stride());
    });
    return makeGridBatchData(std::move(hdl), voxS, voxO);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
