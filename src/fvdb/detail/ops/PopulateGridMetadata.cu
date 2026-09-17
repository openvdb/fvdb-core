// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/GridBatchData.h>
#include <fvdb/detail/ops/PopulateGridMetadata.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/cuda/RAIIRawDeviceBuffer.h>

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMathCompat.h>

constexpr int NUM_THREADS = 1;

namespace fvdb {
namespace detail {
namespace ops {

template <template <typename T, int> typename TensorAccessorT>
__hostdev__ void
populateGridMetadataKernel(uint32_t numGrids,
                           const nanovdb::OnIndexGrid *grids,
                           const nanovdb::Vec3d *voxelSizes,
                           const nanovdb::Vec3d *voxelOrigins,
                           TensorAccessorT<fvdb::JOffsetsType, 1> gridOffsets,
                           GridBatchData::GridMetadata *perGridMetadata,
                           GridBatchData::GridBatchMetadata *batchMetadata) {
    // The records may be raw allocations (device memory, or the unified memory PrivateUse1 shares
    // between host and device), so every field a reader checks is written here, the version
    // stamps included.
    batchMetadata->version       = GridBatchData::GridBatchMetadata::kVersion;
    batchMetadata->mMaxVoxels    = 0;
    batchMetadata->mMaxLeafCount = 0;

    nanovdb::Coord bbMin = nanovdb::Coord::max();
    nanovdb::Coord bbMax = nanovdb::Coord::min();

    nanovdb::OnIndexGrid *currentGrid = (nanovdb::OnIndexGrid *)&grids[0];
    uint32_t i                        = 0;
    uint64_t byteCount                = 0;

    perGridMetadata[i].mCumVoxels = 0;
    perGridMetadata[i].mCumBytes  = 0;
    perGridMetadata[i].mCumLeaves = 0;

    gridOffsets[i] = 0;
    while (i < numGrids - 1) {
        byteCount                 = currentGrid->gridSize();
        const uint32_t leafCount  = currentGrid->tree().nodeCount(0);
        const uint64_t voxelCount = currentGrid->tree().activeVoxelCount();

        GridBatchData::GridMetadata &metaCur  = perGridMetadata[i];
        GridBatchData::GridMetadata &metaNext = perGridMetadata[i + 1];

        metaCur.version = GridBatchData::GridMetadata::kVersion;
        metaCur.setTransform(voxelSizes[i], voxelOrigins[i]);
        metaCur.mNumVoxels = voxelCount;
        metaCur.mNumBytes  = byteCount;
        metaCur.mNumLeaves = leafCount;
        metaCur.mBBox      = currentGrid->tree().bbox();

        metaNext.mCumVoxels = metaCur.mCumVoxels + voxelCount;
        metaNext.mCumBytes  = metaCur.mCumBytes + byteCount;
        metaNext.mCumLeaves = metaCur.mCumLeaves + leafCount;

        gridOffsets[i + 1] = metaCur.mCumVoxels + metaCur.mNumVoxels;

        // number of voxels exceeds maximum indexable value
        assert(voxelCount <= std::numeric_limits<int64_t>::max());
        batchMetadata->mMaxVoxels =
            max(batchMetadata->mMaxVoxels, static_cast<int64_t>(voxelCount));
        batchMetadata->mMaxLeafCount = max(batchMetadata->mMaxLeafCount, leafCount);

        bbMin       = bbMin.minComponent(currentGrid->tree().bbox().min());
        bbMax       = bbMax.maxComponent(currentGrid->tree().bbox().max());
        currentGrid = (nanovdb::OnIndexGrid *)(((uint8_t *)currentGrid) + byteCount);
        i += 1;
    }

    perGridMetadata[i].version = GridBatchData::GridMetadata::kVersion;
    perGridMetadata[i].setTransform(voxelSizes[i], voxelOrigins[i]);
    perGridMetadata[i].mNumVoxels = currentGrid->tree().activeVoxelCount();
    perGridMetadata[i].mNumBytes  = currentGrid->gridSize();
    perGridMetadata[i].mNumLeaves = currentGrid->tree().nodeCount(0);
    perGridMetadata[i].mBBox      = currentGrid->tree().bbox();

    gridOffsets[i + 1] = perGridMetadata[i].mCumVoxels + perGridMetadata[i].mNumVoxels;

    batchMetadata->mMaxVoxels    = max(batchMetadata->mMaxVoxels, perGridMetadata[i].mNumVoxels);
    batchMetadata->mMaxLeafCount = max(batchMetadata->mMaxLeafCount, perGridMetadata[i].mNumLeaves);

    // number of voxels exceeds maximum indexable value
    assert(perGridMetadata[i].mCumVoxels + perGridMetadata[i].mNumVoxels <=
           std::numeric_limits<int64_t>::max());
    batchMetadata->mTotalVoxels = perGridMetadata[i].mCumVoxels + perGridMetadata[i].mNumVoxels;

    // number of grid leaf nodes exceeds maximum indexable value
    assert(perGridMetadata[i].mCumLeaves + perGridMetadata[i].mNumLeaves <=
           std::numeric_limits<int64_t>::max());
    batchMetadata->mTotalLeaves = perGridMetadata[i].mCumLeaves + perGridMetadata[i].mNumLeaves;

    bbMin                     = bbMin.minComponent(currentGrid->tree().bbox().min());
    bbMax                     = bbMax.maxComponent(currentGrid->tree().bbox().max());
    batchMetadata->mTotalBBox = nanovdb::CoordBBox(bbMin, bbMax);
}

template <template <typename T, int I> typename TensorAccessorT>
__global__ __launch_bounds__(NUM_THREADS) void
populateGridMetadataCUDA(uint32_t numGrids,
                         const nanovdb::OnIndexGrid *grids,
                         const nanovdb::Vec3d *voxelSizes,
                         const nanovdb::Vec3d *voxelOrigins,
                         TensorAccessorT<fvdb::JOffsetsType, 1> outBatchOffsets,
                         GridBatchData::GridMetadata *perGridMetadata,
                         GridBatchData::GridBatchMetadata *batchMetadata) {
    populateGridMetadataKernel<TensorAccessorT>(
        numGrids, grids, voxelSizes, voxelOrigins, outBatchOffsets, perGridMetadata, batchMetadata);
}

template <torch::DeviceType>
void dispatchPopulateGridMetadata(const GridStorage &storage,
                                  const std::vector<nanovdb::Vec3d> &voxelSizes,
                                  const std::vector<nanovdb::Vec3d> &voxelOrigins,
                                  torch::Tensor &outBatchOffsets,
                                  GridBatchData::GridMetadata *outPerGridMetadataHost,
                                  GridBatchData::GridMetadata *outPerGridMetadataDevice,
                                  GridBatchData::GridBatchMetadata *outBatchMetadataHost,
                                  GridBatchData::GridBatchMetadata *outBatchMetadataDevice);

template <>
void
dispatchPopulateGridMetadata<torch::kCUDA>(
    const GridStorage &storage,
    const std::vector<nanovdb::Vec3d> &voxelSizes,
    const std::vector<nanovdb::Vec3d> &voxelOrigins,
    torch::Tensor &outBatchOffsets,
    GridBatchData::GridMetadata *outPerGridMetadataHost,
    GridBatchData::GridMetadata *outPerGridMetadataDevice,
    GridBatchData::GridBatchMetadata *outBatchMetadataHost,
    GridBatchData::GridBatchMetadata *outBatchMetadataDevice) {
    c10::cuda::CUDAGuard deviceGuard(storage.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(storage.device().index()).stream();

    // Copy sizes and origins to device buffers
    RAIIRawDeviceBuffer<nanovdb::Vec3d> deviceVoxSizes(voxelSizes.size(), storage.device());
    deviceVoxSizes.setData((nanovdb::Vec3d *)voxelSizes.data(), true /* blocking */);
    const nanovdb::Vec3d *deviceVoxSizesPtr = deviceVoxSizes.devicePtr;

    RAIIRawDeviceBuffer<nanovdb::Vec3d> deviceVoxOrigins(voxelOrigins.size(), storage.device());
    deviceVoxOrigins.setData((nanovdb::Vec3d *)voxelOrigins.data(), true /* blocking */);
    const nanovdb::Vec3d *deviceVoxOriginsPtr = deviceVoxOrigins.devicePtr;

    outBatchOffsets = torch::empty(
        {(fvdb::JOffsetsType)(voxelOrigins.size() + 1)},
        torch::TensorOptions().dtype(fvdb::JOffsetsScalarType).device(storage.device()));

    // Read metadata into device buffers
    TORCH_CHECK(storage.deviceBytes() != nullptr, "GridStorage is empty");
    const nanovdb::OnIndexGrid *grids = (const nanovdb::OnIndexGrid *)storage.deviceBytes();

    const size_t metaDataByteSize = sizeof(GridBatchData::GridMetadata) * storage.gridCount();
    populateGridMetadataCUDA<TorchRAcc64><<<1, NUM_THREADS, 0, stream>>>(
        storage.gridCount(),
        grids,
        (const nanovdb::Vec3d *)deviceVoxSizesPtr,
        (const nanovdb::Vec3d *)deviceVoxOriginsPtr,
        outBatchOffsets.packed_accessor64<fvdb::JOffsetsType, 1, torch::RestrictPtrTraits>(),
        outPerGridMetadataDevice,
        outBatchMetadataDevice);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Read back on the kernel's stream and wait for it. A synchronous cudaMemcpy would run on the
    // legacy default stream, which torch's non-blocking streams do not synchronize with, so under
    // torch.cuda.stream(s) it could read the records before the kernel had written them.
    C10_CUDA_CHECK(cudaMemcpyAsync(outPerGridMetadataHost,
                                   outPerGridMetadataDevice,
                                   metaDataByteSize,
                                   cudaMemcpyDeviceToHost,
                                   stream));
    C10_CUDA_CHECK(cudaMemcpyAsync(outBatchMetadataHost,
                                   outBatchMetadataDevice,
                                   sizeof(GridBatchData::GridBatchMetadata),
                                   cudaMemcpyDeviceToHost,
                                   stream));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}

template <>
void
dispatchPopulateGridMetadata<torch::kCPU>(
    const GridStorage &storage,
    const std::vector<nanovdb::Vec3d> &voxelSizes,
    const std::vector<nanovdb::Vec3d> &voxelOrigins,
    torch::Tensor &outBatchOffsets,
    GridBatchData::GridMetadata *outPerGridMetadataHost,
    GridBatchData::GridMetadata *outPerGridMetadataDevice,
    GridBatchData::GridBatchMetadata *outBatchMetadataHost,
    GridBatchData::GridBatchMetadata *outBatchMetadataDevice) {
    outBatchOffsets = torch::empty(
        {(fvdb::JOffsetsType)(voxelOrigins.size() + 1)},
        torch::TensorOptions().dtype(fvdb::JOffsetsScalarType).device(storage.device()));
    TORCH_CHECK(storage.hostBytes() != nullptr, "GridStorage is empty");
    const nanovdb::OnIndexGrid *grids = (const nanovdb::OnIndexGrid *)storage.hostBytes();
    populateGridMetadataKernel<TorchAcc>(storage.gridCount(),
                                         grids,
                                         voxelSizes.data(),
                                         voxelOrigins.data(),
                                         outBatchOffsets.accessor<fvdb::JOffsetsType, 1>(),
                                         outPerGridMetadataHost,
                                         outBatchMetadataHost);
}

template <>
void
dispatchPopulateGridMetadata<torch::kPrivateUse1>(
    const GridStorage &storage,
    const std::vector<nanovdb::Vec3d> &voxelSizes,
    const std::vector<nanovdb::Vec3d> &voxelOrigins,
    torch::Tensor &outBatchOffsets,
    GridBatchData::GridMetadata *outPerGridMetadataHost,
    GridBatchData::GridMetadata *outPerGridMetadataDevice,
    GridBatchData::GridBatchMetadata *outBatchMetadataHost,
    GridBatchData::GridBatchMetadata *outBatchMetadataDevice) {
    outBatchOffsets = torch::empty(
        {(fvdb::JOffsetsType)(voxelOrigins.size() + 1)},
        torch::TensorOptions().dtype(fvdb::JOffsetsScalarType).device(storage.device()));
    TORCH_CHECK(storage.hostBytes() != nullptr, "GridStorage is empty");
    const nanovdb::OnIndexGrid *grids = (const nanovdb::OnIndexGrid *)storage.hostBytes();
    populateGridMetadataKernel<TorchAcc>(storage.gridCount(),
                                         grids,
                                         voxelSizes.data(),
                                         voxelOrigins.data(),
                                         outBatchOffsets.accessor<fvdb::JOffsetsType, 1>(),
                                         outPerGridMetadataHost,
                                         outBatchMetadataHost);
}

void
populateGridMetadata(const GridStorage &storage,
                     const std::vector<nanovdb::Vec3d> &voxelSizes,
                     const std::vector<nanovdb::Vec3d> &voxelOrigins,
                     torch::Tensor &outBatchOffsets,
                     GridBatchData::GridMetadata *outPerGridMetadataHost,
                     GridBatchData::GridMetadata *outPerGridMetadataDevice,
                     GridBatchData::GridBatchMetadata *outBatchMetadataHost) {
    const torch::Device device = storage.device();
    FVDB_DISPATCH_KERNEL(device, [&]() {
        GridBatchData::GridBatchMetadata *deviceBatchMetadataPtr = nullptr;
        if constexpr (DeviceTag == torch::kCUDA) {
            c10::cuda::CUDAGuard deviceGuard(device);
            auto wrapper = c10::cuda::getCurrentCUDAStream(device.index());
            auto data    = c10::cuda::CUDACachingAllocator::raw_alloc_with_stream(
                sizeof(GridBatchData::GridBatchMetadata), wrapper.stream());
            deviceBatchMetadataPtr = static_cast<GridBatchData::GridBatchMetadata *>(data);
        }
        dispatchPopulateGridMetadata<DeviceTag>(storage,
                                                voxelSizes,
                                                voxelOrigins,
                                                outBatchOffsets,
                                                outPerGridMetadataHost,
                                                outPerGridMetadataDevice,
                                                outBatchMetadataHost,
                                                deviceBatchMetadataPtr);
        if constexpr (DeviceTag == torch::kCUDA) {
            c10::cuda::CUDAGuard deviceGuard(device);
            c10::cuda::CUDACachingAllocator::raw_delete(deviceBatchMetadataPtr);
        }
    });
}

} // namespace ops
} // namespace detail
} // namespace fvdb
