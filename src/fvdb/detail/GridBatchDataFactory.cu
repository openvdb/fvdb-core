// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/GridBatchDataFactory.h>
#include <fvdb/detail/ops/PopulateGridMetadata.h>
#include <fvdb/detail/utils/nanovdb/CreateEmptyGridStorage.h>

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

__global__ void
computeBatchOffsetsFromMetadata(
    uint32_t numGrids,
    fvdb::GridBatchData::GridMetadata *perGridMetadata,
    torch::PackedTensorAccessor64<fvdb::JOffsetsType, 1, torch::RestrictPtrTraits>
        outBatchOffsets) {
    if (numGrids == 0) {
        return;
    }
    outBatchOffsets[0] = 0;
    for (uint32_t i = 1; i < (numGrids + 1); i += 1) {
        outBatchOffsets[i] = outBatchOffsets[i - 1] + perGridMetadata[i - 1].mNumVoxels;
    }
}

} // namespace

namespace fvdb {
namespace detail {

GridBatchData::GridMetadata *
allocateHostGridMetadata(int64_t batchSize) {
    using GridMetadata = GridBatchData::GridMetadata;
    TORCH_CHECK(batchSize > 0, "Batch size must be greater than 0");
    GridMetadata *ret =
        reinterpret_cast<GridMetadata *>(std::malloc(sizeof(GridMetadata) * batchSize));
    for (auto i = 0; i < batchSize; ++i) {
        ret[i] = GridMetadata();
    }
    return ret;
}

void
freeHostGridMetadata(GridBatchData::GridMetadata *ptr) {
    std::free(ptr);
}

GridBatchData::GridMetadata *
allocateDeviceGridMetadata(torch::Device device, int64_t batchSize) {
    using GridMetadata = GridBatchData::GridMetadata;
    TORCH_CHECK(batchSize > 0, "Batch size must be greater than 0");

    c10::cuda::CUDAGuard deviceGuard(device);
    auto wrapper                  = c10::cuda::getCurrentCUDAStream(device.index());
    const size_t metadataByteSize = sizeof(GridMetadata) * batchSize;
    auto data =
        c10::cuda::CUDACachingAllocator::raw_alloc_with_stream(metadataByteSize, wrapper.stream());
    return static_cast<GridMetadata *>(data);
}

void
freeDeviceGridMetadata(torch::Device device, GridBatchData::GridMetadata *ptr) {
    c10::cuda::CUDAGuard deviceGuard(device);
    c10::cuda::CUDACachingAllocator::raw_delete(ptr);
}

GridBatchData::GridMetadata *
allocateUnifiedMemoryGridMetadata(int64_t batchSize) {
    using GridMetadata = GridBatchData::GridMetadata;
    TORCH_CHECK(batchSize > 0, "Batch size must be greater than 0");

    auto allocator = c10::GetAllocator(c10::DeviceType::PrivateUse1);
    auto data      = allocator->raw_allocate(sizeof(GridMetadata) * batchSize);
    return static_cast<GridMetadata *>(data);
}

void
freeUnifiedMemoryGridMetadata(GridBatchData::GridMetadata *ptr) {
    auto allocator = c10::GetAllocator(c10::DeviceType::PrivateUse1);
    allocator->raw_deallocate(ptr);
}

void
syncMetadataToDevice(GridBatchData::GridMetadata *hostMeta,
                     GridBatchData::GridMetadata *deviceMeta,
                     int64_t batchSize,
                     torch::Device device,
                     bool blocking) {
    if (!device.is_cuda()) {
        return;
    }
    if (!hostMeta || !deviceMeta) {
        return;
    }
    c10::cuda::CUDAGuard deviceGuard(device);
    c10::cuda::CUDAStream wrapper = c10::cuda::getCurrentCUDAStream(device.index());
    const size_t metadataByteSize = sizeof(GridBatchData::GridMetadata) * batchSize;
    C10_CUDA_CHECK(cudaMemcpyAsync(
        deviceMeta, hostMeta, metadataByteSize, cudaMemcpyHostToDevice, wrapper.stream()));
    if (blocking) {
        C10_CUDA_CHECK(cudaStreamSynchronize(wrapper.stream()));
    }
}

torch::Tensor
computeBatchOffsets(GridBatchData::GridMetadata *hostMeta,
                    GridBatchData::GridMetadata *deviceMeta,
                    int64_t batchSize,
                    torch::Device device) {
    torch::Tensor offsets = torch::empty(
        {batchSize + 1}, torch::TensorOptions().dtype(fvdb::JOffsetsScalarType).device(device));
    if (device.is_cuda()) {
        const c10::cuda::CUDAGuard device_guard(device);
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device.index()).stream();
        computeBatchOffsetsFromMetadata<<<1, 1, 0, stream>>>(
            batchSize,
            deviceMeta,
            offsets.packed_accessor64<fvdb::JOffsetsType, 1, torch::RestrictPtrTraits>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        auto outBatchOffsets = offsets.accessor<fvdb::JOffsetsType, 1>();
        outBatchOffsets[0]   = 0;
        for (int64_t i = 1; i < (batchSize + 1); i += 1) {
            outBatchOffsets[i] = outBatchOffsets[i - 1] + hostMeta[i - 1].mNumVoxels;
        }
    }
    return offsets;
}

c10::intrusive_ptr<GridBatchData>
makeGridBatchData(GridStorage &&storage,
                  const std::vector<nanovdb::Vec3d> &voxelSizes,
                  const std::vector<nanovdb::Vec3d> &voxelOrigins) {
    TORCH_CHECK(!storage.isEmpty(), "Cannot create a GridBatchData from empty grid storage");
    for (std::size_t i = 0; i < voxelSizes.size(); i += 1) {
        TORCH_CHECK_VALUE(voxelSizes[i][0] > 0 && voxelSizes[i][1] > 0 && voxelSizes[i][2] > 0,
                          "Voxel size must be greater than 0");
    }
    TORCH_CHECK(voxelSizes.size() == storage.gridCount(),
                "voxelSizes array does not have the same size as the number of grids, got ",
                voxelSizes.size(),
                " expected ",
                storage.gridCount());
    TORCH_CHECK(voxelOrigins.size() == storage.gridCount(),
                "Voxel origins must be the same size as the number of grids");
    TORCH_CHECK(storage.gridType(0) == nanovdb::GridType::OnIndex,
                "GridBatchData only supports ValueOnIndex grids");

    const int64_t batchSize    = storage.gridCount();
    const torch::Device device = storage.device();

    GridBatchData::GridMetadata *hostMeta   = nullptr;
    GridBatchData::GridMetadata *deviceMeta = nullptr;

    if (device.is_cpu() || device.is_cuda()) {
        hostMeta = allocateHostGridMetadata(batchSize);
        if (device.is_cuda()) {
            deviceMeta = allocateDeviceGridMetadata(device, batchSize);
        }
    } else if (device.is_privateuseone()) {
        deviceMeta = allocateUnifiedMemoryGridMetadata(batchSize);
        hostMeta   = deviceMeta;
    } else {
        TORCH_CHECK(false, "Only CPU, CUDA, and PrivateUse1 devices are supported");
    }

    torch::Tensor batchOffsets;
    GridBatchData::GridBatchMetadata batchMeta;
    ops::populateGridMetadata(
        storage, voxelSizes, voxelOrigins, batchOffsets, hostMeta, deviceMeta, &batchMeta);
    batchMeta.mIsContiguous = true;

    const torch::Tensor listIndices =
        torch::empty({0, 1}, torch::TensorOptions().dtype(fvdb::JLIdxScalarType).device(device));
    TORCH_CHECK(listIndices.numel() == 0 || listIndices.size(0) == (batchOffsets.size(0) - 1),
                "Invalid list indices when building grid");

    // One repeat_interleave instead of a torch::full + torch::cat per member: the per-member
    // version costs B+1 kernel dispatches on every grid construction (issue #755). Leaf counts
    // are already host-side in hostMeta.
    torch::Tensor leafCounts =
        torch::empty({batchSize}, torch::TensorOptions().dtype(torch::kInt64));
    auto leafCountsAcc = leafCounts.accessor<int64_t, 1>();
    for (int64_t i = 0; i < batchSize; i += 1) {
        leafCountsAcc[i] = hostMeta[i].mNumLeaves;
    }
    torch::Tensor leafBatchIndices = torch::repeat_interleave(
        torch::arange(batchSize, torch::TensorOptions().dtype(fvdb::JIdxScalarType).device(device)),
        leafCounts.to(device));

    return c10::make_intrusive<GridBatchData>(std::make_shared<GridStorage>(std::move(storage)),
                                              hostMeta,
                                              deviceMeta,
                                              batchSize,
                                              std::move(batchMeta),
                                              std::move(leafBatchIndices),
                                              std::move(batchOffsets),
                                              std::move(listIndices));
}

c10::intrusive_ptr<GridBatchData>
makeEmptyGridBatchData(const torch::Device &device) {
    c10::DeviceGuard guard(device);
    auto deviceTensorOptions = torch::TensorOptions().device(device);

    torch::Tensor leafBatchIndices =
        torch::empty({0}, deviceTensorOptions.dtype(fvdb::JIdxScalarType));
    torch::Tensor batchOffsets =
        torch::zeros({1}, deviceTensorOptions.dtype(fvdb::JOffsetsScalarType));
    torch::Tensor listIndices =
        torch::empty({0, 1}, deviceTensorOptions.dtype(fvdb::JLIdxScalarType));

    GridBatchData::GridBatchMetadata batchMeta;
    batchMeta.mIsContiguous = true;

    return c10::make_intrusive<GridBatchData>(
        std::make_shared<GridStorage>(createEmptyGridStorage(device)),
        nullptr,
        nullptr,
        0,
        std::move(batchMeta),
        std::move(leafBatchIndices),
        std::move(batchOffsets),
        std::move(listIndices));
}

c10::intrusive_ptr<GridBatchData>
makeEmptyGridBatchData(const torch::Device &device,
                       const nanovdb::Vec3d &voxelSize,
                       const nanovdb::Vec3d &origin) {
    return makeGridBatchData(createEmptyGridStorage(device), {voxelSize}, {origin});
}

c10::intrusive_ptr<GridBatchData>
makeEmptyGridBatchData(const torch::Device &device,
                       const std::vector<nanovdb::Vec3d> &voxelSizes,
                       const std::vector<nanovdb::Vec3d> &origins) {
    return makeGridBatchData(
        createEmptyGridStorage(device, voxelSizes.size()), voxelSizes, origins);
}

} // namespace detail
} // namespace fvdb
