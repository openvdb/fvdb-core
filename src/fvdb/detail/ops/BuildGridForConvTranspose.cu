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

#include <nanovdb/tools/CreateNanoGrid.h>

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/types.h>

#include <algorithm>
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

uint64_t
inactiveCudaCacheBytes(c10::CachingDeviceAllocator::DeviceStats const &stats) {
    constexpr auto aggregate = static_cast<size_t>(c10::CachingAllocator::StatType::AGGREGATE);
    const int64_t reserved   = stats.reserved_bytes[aggregate].current;
    const int64_t active     = stats.active_bytes[aggregate].current;
    const int64_t fragmented = stats.inactive_split_bytes[aggregate].current;
    if (reserved <= 0 || active >= reserved) {
        return 0;
    }

    // Do not count inactive split blocks as generally reusable: their aggregate byte count can
    // overstate what is available for either of the two large staging allocations.
    const int64_t inactive = reserved - std::max<int64_t>(active, 0);
    return static_cast<uint64_t>(inactive - std::min(std::max<int64_t>(fragmented, 0), inactive));
}

uint64_t
cudaReservedBytes(c10::CachingDeviceAllocator::DeviceStats const &stats) {
    constexpr auto aggregate = static_cast<size_t>(c10::CachingAllocator::StatType::AGGREGATE);
    return static_cast<uint64_t>(std::max<int64_t>(stats.reserved_bytes[aggregate].current, 0));
}

void
checkCudaTransposeStaging(const torch::Device &device,
                          int64_t inputVoxelCount,
                          int64_t kernelVolume,
                          uint64_t requestedBytes) {
    if (requestedBytes == 0) {
        return;
    }

    const c10::DeviceIndex deviceIndex = device.index();
    const at::cuda::CUDAGuard deviceGuard(device);
    size_t driverFreeBytes = 0;
    size_t totalBytes      = 0;
    C10_CUDA_CHECK(cudaMemGetInfo(&driverFreeBytes, &totalBytes));
    auto *allocator           = c10::cuda::CUDACachingAllocator::get();
    const auto allocatorStats = c10::cuda::CUDACachingAllocator::getDeviceStats(deviceIndex);
    const uint64_t inactiveCacheBytes  = inactiveCudaCacheBytes(allocatorStats);
    const uint64_t reservedBytes       = cudaReservedBytes(allocatorStats);
    const double memoryFraction        = allocator->getMemoryFraction(deviceIndex);
    const uint64_t allocatorLimitBytes = static_cast<uint64_t>(
        std::min<long double>(totalBytes, static_cast<long double>(totalBytes) * memoryFraction));
    const uint64_t reservationAllowanceBytes =
        allocatorLimitBytes > reservedBytes ? allocatorLimitBytes - reservedBytes : 0;
    const uint64_t newReservationBytes =
        std::min<uint64_t>(driverFreeBytes, reservationAllowanceBytes);
    const uint64_t allocatorAvailableBytes =
        inactiveCacheBytes > totalBytes - std::min<uint64_t>(totalBytes, newReservationBytes)
            ? totalBytes
            : inactiveCacheBytes + newReservationBytes;

    // Preserve enough space for allocator rounding, the output NanoVDB, and ordinary live
    // application tensors. The cap avoids withholding an excessive fraction on large devices.
    constexpr uint64_t minimumHeadroomBytes = UINT64_C(64) * 1024 * 1024;
    constexpr uint64_t maximumHeadroomBytes = UINT64_C(1024) * 1024 * 1024;
    const uint64_t desiredHeadroomBytes =
        std::min(maximumHeadroomBytes, std::max(minimumHeadroomBytes, allocatorLimitBytes / 20));
    const uint64_t headroomBytes = std::min(desiredHeadroomBytes, allocatorLimitBytes / 2);
    const uint64_t safeAvailableBytes =
        allocatorAvailableBytes > headroomBytes ? allocatorAvailableBytes - headroomBytes : 0;

    TORCH_CHECK(
        requestedBytes <= safeAvailableBytes,
        "Generative transposed convolution would stage ",
        requestedBytes,
        " bytes for ",
        inputVoxelCount,
        " input voxels * ",
        kernelVolume,
        " kernel taps, but CUDA device ",
        static_cast<int>(deviceIndex),
        " has only ",
        safeAvailableBytes,
        " safely available bytes (",
        driverFreeBytes,
        " driver-free; ",
        inactiveCacheBytes,
        " reusable PyTorch cache + ",
        newReservationBytes,
        " new-allocation allowance under PyTorch memory fraction ",
        memoryFraction,
        " - ",
        headroomBytes,
        " reserved headroom). Reduce the input or kernel size, provide an explicit target grid "
        "for restricted transposed convolution, or release CUDA memory before retrying.");
}

bool
isUnshiftedSubdivision(ConvolutionGeometry const &geometry) {
    return geometry.kernelSize() == geometry.stride() &&
           geometry.paddingBefore() == nanovdb::Coord(0);
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

    const auto &coarseGridHdl = coarseBatchHdl.nanoGridHandle();
    std::vector<nanovdb::GridHandle<TorchDeviceBuffer>> batchHandles;
    batchHandles.reserve(coarseGridHdl.gridCount());
    for (uint32_t bidx = 0; bidx < coarseGridHdl.gridCount(); bidx += 1) {
        const nanovdb::OnIndexGrid *coarseGrid = coarseGridHdl.template grid<GridT>(bidx);
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

template <>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchBuildGridForConvTranspose<torch::kCUDA>(const GridBatchData &baseGridHdl,
                                                const nanovdb::Coord &kernelSize,
                                                const nanovdb::Coord &stride) {
    ConvolutionGeometry const geometry(kernelSize, stride);
    const uint64_t stagingBytes = checkTransposeInputAndKernel(baseGridHdl.totalVoxels(), geometry);
    checkCudaTransposeStaging(
        baseGridHdl.device(), baseGridHdl.totalVoxels(), geometry.kernelVolume(), stagingBytes);
    if (isUnshiftedSubdivision(geometry)) {
        return ops::_createNanoGridFromIJK(
            fineIJKForCoarseGrid(baseGridHdl, geometry.stride(), std::nullopt));
    }
    return ops::_createNanoGridFromIJK(convTransposeIJKForGrid(baseGridHdl, geometry));
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

    const auto &baseGridHdl = baseBatchHdl.nanoGridHandle();
    std::vector<nanovdb::GridHandle<TorchDeviceBuffer>> batchHandles;
    batchHandles.reserve(baseGridHdl.gridCount());
    for (uint32_t bidx = 0; bidx < baseGridHdl.gridCount(); bidx += 1) {
        const nanovdb::OnIndexGrid *baseGrid = baseGridHdl.template grid<GridT>(bidx);
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
