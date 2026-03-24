// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/GridBatchImpl.h>
#include <fvdb/detail/utils/nanovdb/CreateEmptyGridHandle.h>

// Ops headers for dispatch functions
#include <fvdb/detail/ops/ActiveGridCoords.h>
#include <fvdb/detail/ops/ActiveVoxelsInBoundsMask.h>
#include <fvdb/detail/ops/BuildCoarseGridFromFine.h>
#include <fvdb/detail/ops/BuildDenseGrid.h>
#include <fvdb/detail/ops/BuildDilatedGrid.h>
#include <fvdb/detail/ops/BuildFineGridFromCoarse.h>
#include <fvdb/detail/ops/BuildGridForConv.h>
#include <fvdb/detail/ops/BuildGridForConvTranspose.h>
#include <fvdb/detail/ops/BuildGridFromIjk.h>
#include <fvdb/detail/ops/BuildGridFromMesh.h>
#include <fvdb/detail/ops/BuildGridFromNearestVoxelsToPoints.h>
#include <fvdb/detail/ops/BuildGridFromPoints.h>
#include <fvdb/detail/ops/BuildMergedGrids.h>
#include <fvdb/detail/ops/BuildPaddedGrid.h>
#include <fvdb/detail/ops/BuildPrunedGrid.h>
#include <fvdb/detail/ops/JIdxForGrid.h>
#include <fvdb/detail/ops/PopulateGridMetadata.h>

#include <nanovdb/cuda/GridHandle.cuh>
#include <nanovdb/tools/CreateNanoGrid.h>

#include <c10/core/DeviceType.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>

namespace {

__global__ void
computeBatchOffsetsFromMetadata(
    uint32_t numGrids,
    fvdb::detail::GridBatchImpl::GridMetadata *perGridMetadata,
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

__global__ void
updateGridCountAndZeroChecksum(nanovdb::GridData *d_data, uint32_t gridIndex, uint32_t gridCount) {
    NANOVDB_ASSERT(gridIndex < gridCount);
    if (d_data->mGridIndex != gridIndex || d_data->mGridCount != gridCount) {
        d_data->mGridIndex = gridIndex;
        d_data->mGridCount = gridCount;
    }
    d_data->mChecksum.disable();
}

// We use a helper with std::malloc rather than just using new because it makes
// clangd not crash on this file.
fvdb::detail::GridBatchImpl::GridMetadata *
allocateHostGridMetadata(int64_t batchSize) {
    using GridMetadata = fvdb::detail::GridBatchImpl::GridMetadata;
    TORCH_CHECK(batchSize > 0, "Batch size must be greater than 0");
    GridMetadata *ret =
        reinterpret_cast<GridMetadata *>(std::malloc(sizeof(GridMetadata) * batchSize));
    for (auto i = 0; i < batchSize; ++i) {
        ret[i] = GridMetadata();
    }
    return ret;
}

void
freeHostGridMetadata(fvdb::detail::GridBatchImpl::GridMetadata *hostGridMetadata) {
    std::free(hostGridMetadata);
}

fvdb::detail::GridBatchImpl::GridMetadata *
allocateDeviceGridMetadata(torch::Device device, int64_t batchSize) {
    using GridMetadata = fvdb::detail::GridBatchImpl::GridMetadata;
    TORCH_CHECK(batchSize > 0, "Batch size must be greater than 0");

    c10::cuda::CUDAGuard deviceGuard(device);
    auto wrapper                  = c10::cuda::getCurrentCUDAStream(device.index());
    const size_t metadataByteSize = sizeof(GridMetadata) * batchSize;
    auto data =
        c10::cuda::CUDACachingAllocator::raw_alloc_with_stream(metadataByteSize, wrapper.stream());
    return static_cast<GridMetadata *>(data);
}

void
freeDeviceGridMetadata(torch::Device device,
                       fvdb::detail::GridBatchImpl::GridMetadata *deviceGridMetadata) {
    c10::cuda::CUDAGuard deviceGuard(device);
    c10::cuda::CUDACachingAllocator::raw_delete(deviceGridMetadata);
}

fvdb::detail::GridBatchImpl::GridMetadata *
allocateUnifiedMemoryGridMetadata(int64_t batchSize) {
    using GridMetadata = fvdb::detail::GridBatchImpl::GridMetadata;
    TORCH_CHECK(batchSize > 0, "Batch size must be greater than 0");

    auto allocator = c10::GetAllocator(c10::DeviceType::PrivateUse1);
    auto data      = allocator->raw_allocate(sizeof(GridMetadata) * batchSize);
    return static_cast<GridMetadata *>(data);
}

void
freeUnifiedMemoryGridMetadata(
    fvdb::detail::GridBatchImpl::GridMetadata *unifiedMemoryGridMetadata) {
    auto allocator = c10::GetAllocator(c10::DeviceType::PrivateUse1);
    allocator->raw_deallocate(unifiedMemoryGridMetadata);
}

} // namespace

namespace fvdb {
namespace detail {

GridBatchImpl::GridBatchImpl(const torch::Device &device) {
    c10::DeviceGuard guard(device);
    auto deviceTensorOptions = torch::TensorOptions().device(device);
    // TODO (Francis): No list-of-lists support for now, so we just assign an empty list of indices
    mLeafBatchIndices = torch::empty({0}, deviceTensorOptions.dtype(fvdb::JIdxScalarType));
    mBatchOffsets     = torch::zeros({1}, deviceTensorOptions.dtype(fvdb::JOffsetsScalarType));
    mListIndices      = torch::empty({0, 1}, deviceTensorOptions.dtype(fvdb::JLIdxScalarType));

    auto gridHdl = createEmptyGridHandle(device);
    mGridHdl     = std::make_shared<nanovdb::GridHandle<TorchDeviceBuffer>>(std::move(gridHdl));

    mBatchMetadata.mIsContiguous = true;
}

GridBatchImpl::GridBatchImpl(const torch::Device &device,
                             const nanovdb::Vec3d &voxelSize,
                             const nanovdb::Vec3d &origin)
    : GridBatchImpl(createEmptyGridHandle(device), {voxelSize}, {origin}) {}

GridBatchImpl::GridBatchImpl(const torch::Device &device,
                             const std::vector<nanovdb::Vec3d> &voxelSizes,
                             const std::vector<nanovdb::Vec3d> &origins)
    : GridBatchImpl(createEmptyGridHandle(device, voxelSizes.size()), voxelSizes, origins) {}

GridBatchImpl::GridBatchImpl(nanovdb::GridHandle<TorchDeviceBuffer> &&gridHdl,
                             const std::vector<nanovdb::Vec3d> &voxelSizes,
                             const std::vector<nanovdb::Vec3d> &voxelOrigins) {
    TORCH_CHECK(!gridHdl.buffer().isEmpty(),
                "Cannot create a batched grid handle from an empty grid handle");
    for (std::size_t i = 0; i < voxelSizes.size(); i += 1) {
        TORCH_CHECK_VALUE(voxelSizes[i][0] > 0 && voxelSizes[i][1] > 0 && voxelSizes[i][2] > 0,
                          "Voxel size must be greater than 0");
    }
    // TODO (Francis): No list-of-lists support for now, so we just pass an empty list of indices
    const torch::Tensor lidx = torch::empty(
        {0, 1},
        torch::TensorOptions().dtype(fvdb::JLIdxScalarType).device(gridHdl.buffer().device()));
    setGrid(std::move(gridHdl), lidx, voxelSizes, voxelOrigins, false /* blocking */);
    mBatchMetadata.mIsContiguous = true;
};

GridBatchImpl::GridBatchImpl(nanovdb::GridHandle<TorchDeviceBuffer> &&gridHdl,
                             const nanovdb::Vec3d &globalVoxelSize,
                             const nanovdb::Vec3d &globalVoxelOrigin) {
    TORCH_CHECK(!gridHdl.buffer().isEmpty(),
                "Cannot create a batched grid handle from an empty grid handle");
    TORCH_CHECK_VALUE(globalVoxelSize[0] > 0 && globalVoxelSize[1] > 0 && globalVoxelSize[2] > 0,
                      "Voxel size must be greater than 0");
    std::vector<nanovdb::Vec3d> voxelSizes, voxelOrigins;
    for (size_t i = 0; i < gridHdl.gridCount(); ++i) {
        voxelSizes.push_back(globalVoxelSize);
        voxelOrigins.push_back(globalVoxelOrigin);
    }
    // TODO (Francis): No list-of-lists support for now, so we just pass an empty list of indices
    const torch::Tensor lidx = torch::empty(
        {0, 1},
        torch::TensorOptions().dtype(fvdb::JLIdxScalarType).device(gridHdl.buffer().device()));
    setGrid(std::move(gridHdl), lidx, voxelSizes, voxelOrigins, false /* blocking */);
    mBatchMetadata.mIsContiguous = true;
};

GridBatchImpl::~GridBatchImpl() {
    const torch::Device device = mGridHdl->buffer().device();
    if (device.is_cpu() || device.is_cuda()) {
        freeHostGridMetadata(mHostGridMetadata);
        mHostGridMetadata = nullptr;
        if (device.is_cuda()) {
            freeDeviceGridMetadata(device, mDeviceGridMetadata);
            mDeviceGridMetadata = nullptr;
        }
    } else if (device.is_privateuseone()) {
        freeUnifiedMemoryGridMetadata(mDeviceGridMetadata);
        mHostGridMetadata   = nullptr;
        mDeviceGridMetadata = nullptr;
    } else {
        TORCH_CHECK(false, "Only CPU, CUDA, and PrivateUse1 devices are supported");
    }
};

torch::Tensor
GridBatchImpl::numLeavesPerGridTensor() const {
    torch::Tensor retTorch =
        torch::empty({batchSize()}, torch::TensorOptions().dtype(torch::kInt64));
    auto acc = retTorch.accessor<int64_t, 1>();

    for (int64_t bi = 0; bi < batchSize(); bi += 1) {
        acc[bi] = numLeavesAt(bi);
    }
    return retTorch;
}

torch::Tensor
GridBatchImpl::numVoxelsPerGridTensor() const {
    torch::Tensor retTorch =
        torch::empty({batchSize()}, torch::TensorOptions().dtype(torch::kInt64));
    auto acc = retTorch.accessor<int64_t, 1>();

    for (int64_t bi = 0; bi < batchSize(); bi += 1) {
        acc[bi] = numVoxelsAt(bi);
    }
    return retTorch;
}

torch::Tensor
GridBatchImpl::cumVoxelsPerGridTensor() const {
    torch::Tensor retTorch =
        torch::empty({batchSize()}, torch::TensorOptions().dtype(torch::kInt64));
    auto acc = retTorch.accessor<int64_t, 1>();

    for (int64_t bi = 0; bi < batchSize(); bi += 1) {
        acc[bi] = cumVoxelsAt(bi);
    }
    return retTorch;
}

torch::Tensor
GridBatchImpl::numBytesPerGridTensor() const {
    torch::Tensor retTorch =
        torch::empty({batchSize()}, torch::TensorOptions().dtype(torch::kInt64));
    auto acc = retTorch.accessor<int64_t, 1>();

    for (int64_t bi = 0; bi < batchSize(); bi += 1) {
        acc[bi] = numBytesAt(bi);
    }
    return retTorch;
}

const torch::Tensor
GridBatchImpl::voxelOriginAtTensor(int64_t bi) const {
    bi                = negativeToPositiveIndexWithRangecheck(bi);
    const auto origin = mHostGridMetadata[bi].voxelOrigin();

    auto ret = torch::empty({3}, torch::TensorOptions().dtype(torch::kFloat64));
    ret[0]   = origin[0];
    ret[1]   = origin[1];
    ret[2]   = origin[2];
    return ret;
}

const torch::Tensor
GridBatchImpl::voxelSizeAtTensor(int64_t bi) const {
    bi       = negativeToPositiveIndexWithRangecheck(bi);
    auto ret = torch::empty({3}, torch::TensorOptions().dtype(torch::kFloat64));
    ret[0]   = mHostGridMetadata[bi].mVoxelSize[0];
    ret[1]   = mHostGridMetadata[bi].mVoxelSize[1];
    ret[2]   = mHostGridMetadata[bi].mVoxelSize[2];
    return ret;
}

const torch::Tensor
GridBatchImpl::voxelSizesTensor() const {
    torch::Tensor retTorch =
        torch::empty({batchSize(), 3}, torch::TensorOptions().dtype(torch::kFloat64));
    for (int64_t bi = 0; bi < batchSize(); bi += 1) {
        const auto voxSize = voxelSizeAt(bi);
        retTorch[bi][0]    = voxSize[0];
        retTorch[bi][1]    = voxSize[1];
        retTorch[bi][2]    = voxSize[2];
    }
    return retTorch;
}

const torch::Tensor
GridBatchImpl::voxelOriginsTensor() const {
    torch::Tensor retTorch =
        torch::empty({batchSize(), 3}, torch::TensorOptions().dtype(torch::kFloat64));
    for (int64_t bi = 0; bi < batchSize(); bi += 1) {
        const auto voxOrigin = voxelOriginAt(bi);
        retTorch[bi][0]      = voxOrigin[0];
        retTorch[bi][1]      = voxOrigin[1];
        retTorch[bi][2]      = voxOrigin[2];
    }
    return retTorch;
}

const torch::Tensor
GridBatchImpl::bboxAtTensor(int64_t bi) const {
    torch::Tensor ret = torch::zeros({2, 3}, torch::TensorOptions().dtype(torch::kInt32));
    const nanovdb::CoordBBox &bbox = this->bboxAt(bi);
    ret[0][0]                      = bbox.min()[0];
    ret[0][1]                      = bbox.min()[1];
    ret[0][2]                      = bbox.min()[2];
    ret[1][0]                      = bbox.max()[0];
    ret[1][1]                      = bbox.max()[1];
    ret[1][2]                      = bbox.max()[2];
    return ret;
}

const torch::Tensor
GridBatchImpl::bboxPerGridTensor() const {
    const int64_t bs  = batchSize();
    torch::Tensor ret = torch::zeros({bs, 2, 3}, torch::TensorOptions().dtype(torch::kInt32));
    for (int64_t i = 0; i < bs; ++i) {
        const nanovdb::CoordBBox &bbox = this->bboxAt(i);
        ret[i][0][0]                   = bbox.min()[0];
        ret[i][0][1]                   = bbox.min()[1];
        ret[i][0][2]                   = bbox.min()[2];
        ret[i][1][0]                   = bbox.max()[0];
        ret[i][1][1]                   = bbox.max()[1];
        ret[i][1][2]                   = bbox.max()[2];
    }
    return ret;
}

const torch::Tensor
GridBatchImpl::dualBBoxAtTensor(int64_t bi) const {
    torch::Tensor ret = torch::zeros({2, 3}, torch::TensorOptions().dtype(torch::kInt32));
    const nanovdb::CoordBBox &bbox = this->dualBBoxAt(bi);
    ret[0][0]                      = bbox.min()[0];
    ret[0][1]                      = bbox.min()[1];
    ret[0][2]                      = bbox.min()[2];
    ret[1][0]                      = bbox.max()[0];
    ret[1][1]                      = bbox.max()[1];
    ret[1][2]                      = bbox.max()[2];
    return ret;
}

const torch::Tensor
GridBatchImpl::dualBBoxPerGridTensor() const {
    const int64_t bs  = batchSize();
    torch::Tensor ret = torch::zeros({bs, 2, 3}, torch::TensorOptions().dtype(torch::kInt32));
    for (int64_t i = 0; i < bs; ++i) {
        const nanovdb::CoordBBox &bbox = this->dualBBoxAt(i);
        ret[i][0][0]                   = bbox.min()[0];
        ret[i][0][1]                   = bbox.min()[1];
        ret[i][0][2]                   = bbox.min()[2];
        ret[i][1][0]                   = bbox.max()[0];
        ret[i][1][1]                   = bbox.max()[1];
        ret[i][1][2]                   = bbox.max()[2];
    }
    return ret;
}

const torch::Tensor
GridBatchImpl::totalBBoxTensor() const {
    const nanovdb::CoordBBox &bbox = this->totalBBox();
    return torch::tensor({{bbox.min()[0], bbox.min()[1], bbox.min()[2]},
                          {bbox.max()[0], bbox.max()[1], bbox.max()[2]}},
                         torch::TensorOptions().dtype(torch::kInt32));
}

const std::vector<VoxelCoordTransform>
GridBatchImpl::primalTransforms() const {
    std::vector<detail::VoxelCoordTransform> transforms;
    transforms.reserve(batchSize());
    for (int64_t bi = 0; bi < batchSize(); ++bi) {
        transforms.push_back(primalTransformAt(bi));
    }
    return transforms;
}

const std::vector<VoxelCoordTransform>
GridBatchImpl::dualTransforms() const {
    std::vector<detail::VoxelCoordTransform> transforms;
    transforms.reserve(batchSize());
    for (int64_t bi = 0; bi < batchSize(); ++bi) {
        transforms.push_back(dualTransformAt(bi));
    }
    return transforms;
}

torch::Tensor
GridBatchImpl::worldToGridMatrixAt(int64_t bi) const {
    bi = negativeToPositiveIndexWithRangecheck(bi);

    torch::Tensor xformMat =
        torch::eye(4, torch::TensorOptions().device(device()).dtype(torch::kDouble));
    const VoxelCoordTransform &transform = primalTransformAt(bi);
    const nanovdb::Vec3d &scale          = transform.scale<double>();
    const nanovdb::Vec3d &translate      = transform.translate<double>();

    xformMat[0][0] = scale[0];
    xformMat[1][1] = scale[1];
    xformMat[2][2] = scale[2];

    xformMat[3][0] = translate[0];
    xformMat[3][1] = translate[1];
    xformMat[3][2] = translate[2];

    return xformMat;
}

void
GridBatchImpl::recomputeBatchOffsets() {
    mBatchOffsets = torch::empty(
        {batchSize() + 1}, torch::TensorOptions().dtype(fvdb::JOffsetsScalarType).device(device()));
    if (device().is_cuda()) {
        computeBatchOffsetsFromMetadata<<<1, 1>>>(
            batchSize(),
            mDeviceGridMetadata,
            mBatchOffsets.packed_accessor64<fvdb::JOffsetsType, 1, torch::RestrictPtrTraits>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        auto outBatchOffsets = mBatchOffsets.accessor<fvdb::JOffsetsType, 1>();
        outBatchOffsets[0]   = 0;
        for (int64_t i = 1; i < (batchSize() + 1); i += 1) {
            outBatchOffsets[i] = outBatchOffsets[i - 1] + mHostGridMetadata[i - 1].mNumVoxels;
        }
    }
}

template <typename Indexable>
c10::intrusive_ptr<GridBatchImpl>
GridBatchImpl::indexInternal(const Indexable &idx, int64_t size) const {
    if (size == 0) {
        return c10::make_intrusive<GridBatchImpl>(device());
    }
    TORCH_CHECK(size >= 0,
                "Indexing with negative size is not supported (this should never happen)");
    c10::intrusive_ptr<GridBatchImpl> ret = c10::make_intrusive<GridBatchImpl>();
    ret->mGridHdl                         = mGridHdl;

    int64_t cumVoxels     = 0;
    int64_t cumLeaves     = 0;
    int64_t maxVoxels     = 0;
    uint32_t maxLeafCount = 0;
    int64_t count         = 0;
    nanovdb::CoordBBox totalBbox;

    std::vector<torch::Tensor> leafBatchIdxs;

    // If the grid we're creating a view over is contiguous, we inherit that
    bool isContiguous      = mBatchMetadata.mIsContiguous;
    ret->mHostGridMetadata = allocateHostGridMetadata(size);
    ret->mBatchSize        = size;
    for (int64_t i = 0; i < size; i += 1) {
        int64_t bi = idx[i];
        bi         = negativeToPositiveIndexWithRangecheck(bi);

        // If indices are not contiguous or the grid we're viewing is not contiguous, then we're
        // no longer contiguous
        isContiguous = isContiguous && (bi == count);

        const uint32_t numLeaves       = mHostGridMetadata[bi].mNumLeaves;
        const int64_t numVoxels        = mHostGridMetadata[bi].mNumVoxels;
        const nanovdb::CoordBBox &bbox = mHostGridMetadata[bi].mBBox;

        ret->mHostGridMetadata[count]            = mHostGridMetadata[bi];
        ret->mHostGridMetadata[count].mCumLeaves = cumLeaves;
        ret->mHostGridMetadata[count].mCumVoxels = cumVoxels;

        if (count == 0) {
            totalBbox = bbox;
        } else {
            totalBbox.expand(bbox);
        }
        cumLeaves += numLeaves;
        cumVoxels += numVoxels;
        maxVoxels    = std::max(maxVoxels, numVoxels);
        maxLeafCount = std::max(maxLeafCount, numLeaves);
        leafBatchIdxs.push_back(
            torch::full({numLeaves},
                        torch::Scalar(count),
                        torch::TensorOptions().dtype(fvdb::JIdxScalarType).device(device())));
        count += 1;
    }

    // If all the indices were contiguous and the grid we're viewing is contiguous, then we're
    // contiguous
    ret->mBatchMetadata.mIsContiguous = isContiguous && (count == batchSize());
    ret->mBatchMetadata.mTotalLeaves  = cumLeaves;
    ret->mBatchMetadata.mTotalVoxels  = cumVoxels;
    ret->mBatchMetadata.mMaxVoxels    = maxVoxels;
    ret->mBatchMetadata.mMaxLeafCount = maxLeafCount;
    ret->mBatchMetadata.mTotalBBox    = totalBbox;

    if (leafBatchIdxs.size() > 0) {
        ret->mLeafBatchIndices = torch::cat(leafBatchIdxs, 0);
    }

    ret->syncMetadataToDeviceIfCUDA(false);
    ret->recomputeBatchOffsets();

    if (mListIndices.size(0) > 0) {
        TORCH_CHECK(false, "Nested lists of GridBatches are not supported yet");
    } else {
        ret->mListIndices = mListIndices;
    }
    return ret;
}

torch::Tensor
GridBatchImpl::gridToWorldMatrixAt(int64_t bi) const {
    bi = negativeToPositiveIndexWithRangecheck(bi);
    return at::linalg_inv(worldToGridMatrixAt(bi));
}

torch::Tensor
GridBatchImpl::gridToWorldMatrixPerGrid() const {
    std::vector<torch::Tensor> retTorch;
    for (int64_t bi = 0; bi < batchSize(); ++bi) {
        retTorch.emplace_back(gridToWorldMatrixAt(bi));
    }
    return torch::stack(retTorch, 0);
}

torch::Tensor
GridBatchImpl::worldToGridMatrixPerGrid() const {
    c10::DeviceGuard guard(device());
    std::vector<torch::Tensor> retTorch;
    for (int64_t bi = 0; bi < batchSize(); ++bi) {
        retTorch.emplace_back(worldToGridMatrixAt(bi));
    }
    return torch::stack(retTorch, 0);
}

c10::intrusive_ptr<GridBatchImpl>
GridBatchImpl::clone(const torch::Device &device, bool blocking) const {
    // If you're cloning an empty grid, just create a new empty grid on the right device and return
    // it
    if (batchSize() == 0) {
        return c10::make_intrusive<GridBatchImpl>(device);
    }

    // The guide buffer is a hack to perform the correct copy (i.e. host -> device / device -> host
    // etc...) The guide carries the desired target device to the copy. The reason we do this is to
    // conform with the nanovdb which can only accept a buffer as an extra argument.
    TorchDeviceBuffer guide(0, device);

    // Make a copy of this gridHandle on the same device as the guide buffer
    nanovdb::GridHandle<TorchDeviceBuffer> clonedHdl = mGridHdl->copy<TorchDeviceBuffer>(guide);

    // Copy the voxel sizes and origins for this grid
    std::vector<nanovdb::Vec3d> voxelSizes, voxelOrigins;
    gridVoxelSizesAndOrigins(voxelSizes, voxelOrigins);

    // Build a GridBatchImpl from the cloned grid handle and voxel sizes/origins
    // FIXME: (@fwilliams) This makes an extra copy or non contiguous grids
    return GridBatchImpl::contiguous(
        c10::make_intrusive<GridBatchImpl>(std::move(clonedHdl), voxelSizes, voxelOrigins));
}

void
GridBatchImpl::syncMetadataToDeviceIfCUDA(bool blocking) {
    if (!device().is_cuda()) {
        return;
    }

    // There is something to sync and we're on a cuda device
    // Global device guards as we operate on this.
    c10::cuda::CUDAGuard deviceGuard(device());
    c10::cuda::CUDAStream wrapper = c10::cuda::getCurrentCUDAStream(device().index());

    // There are no grids in the batch so we need to free the device metadata if it exists
    if (!mHostGridMetadata && mDeviceGridMetadata) {
        freeDeviceGridMetadata(device(), mDeviceGridMetadata);
        mDeviceGridMetadata = nullptr;
        return;
    }

    if (!mDeviceGridMetadata) { // Allocate the CUDA memory if it hasn't been allocated already
        mDeviceGridMetadata = allocateDeviceGridMetadata(device(), batchSize());
    }

    // Copy host grid metadata to device grid metadata
    const size_t metadataByteSize = sizeof(GridMetadata) * batchSize();
    C10_CUDA_CHECK(cudaMemcpyAsync(mDeviceGridMetadata,
                                   mHostGridMetadata,
                                   metadataByteSize,
                                   cudaMemcpyHostToDevice,
                                   wrapper.stream()));
    // Block if you asked for it
    if (blocking) {
        C10_CUDA_CHECK(cudaStreamSynchronize(wrapper.stream()));
    }
}

namespace {
__global__ void
setGlobalPrimalTransformKernel(GridBatchImpl::GridMetadata *metadata,
                               VoxelCoordTransform transform) {
    unsigned int i               = threadIdx.x;
    metadata[i].mPrimalTransform = transform;
}

__global__ void
setGlobalDualTransformKernel(GridBatchImpl::GridMetadata *metadata, VoxelCoordTransform transform) {
    unsigned int i             = threadIdx.x;
    metadata[i].mDualTransform = transform;
}

__global__ void
setGlobalVoxelSizeKernel(GridBatchImpl::GridMetadata *metadata, nanovdb::Vec3d voxelSize) {
    unsigned int i = threadIdx.x;
    metadata[i].setTransform(voxelSize, metadata[i].voxelOrigin());
}

__global__ void
setGlobalVoxelOriginKernel(GridBatchImpl::GridMetadata *metadata, nanovdb::Vec3d voxelOrigin) {
    unsigned int i = threadIdx.x;
    metadata[i].setTransform(metadata[i].mVoxelSize, voxelOrigin);
}

__global__ void
setGlobalVoxelSizeAndOriginKernel(GridBatchImpl::GridMetadata *metadata,
                                  nanovdb::Vec3d voxelSize,
                                  nanovdb::Vec3d voxelOrigin) {
    unsigned int i = threadIdx.x;
    metadata[i].setTransform(voxelSize, voxelOrigin);
}

__global__ void
setPrimalTransformFromDualGridKernel(GridBatchImpl::GridMetadata *metadata,
                                     const GridBatchImpl::GridMetadata *dualMetadata) {
    unsigned int i               = threadIdx.x;
    metadata[i].mDualTransform   = dualMetadata[i].mPrimalTransform;
    metadata[i].mPrimalTransform = dualMetadata[i].mDualTransform;
    metadata[i].mVoxelSize       = dualMetadata[i].mVoxelSize;
}

__hostdev__ nanovdb::Vec3d
fineVoxelSize(const nanovdb::Vec3d &voxelSize, const nanovdb::Coord &subdivFactor) {
    return voxelSize / subdivFactor.asVec3d();
}

__hostdev__ nanovdb::Vec3d
fineVoxelOrigin(const nanovdb::Vec3d &voxelSize,
                const nanovdb::Vec3d &voxelOrigin,
                const nanovdb::Coord &subdivFactor) {
    return voxelOrigin - (subdivFactor.asVec3d() - nanovdb::Vec3d(1.0)) *
                             (voxelSize / subdivFactor.asVec3d()) * 0.5;
}

__hostdev__ nanovdb::Vec3d
coarseVoxelSize(const nanovdb::Vec3d &voxelSize, const nanovdb::Coord &coarseningFactor) {
    return coarseningFactor.asVec3d() * voxelSize;
}

__hostdev__ nanovdb::Vec3d
coarseVoxelOrigin(const nanovdb::Vec3d &voxelSize,
                  const nanovdb::Vec3d &voxelOrigin,
                  const nanovdb::Coord &coarseningFactor) {
    return (coarseningFactor.asVec3d() - nanovdb::Vec3d(1.0)) * voxelSize * 0.5 + voxelOrigin;
}

__global__ void
setFineTransformFromCoarseGridKernel(GridBatchImpl::GridMetadata *metadata,
                                     const GridBatchImpl::GridMetadata *coarseMetadata,
                                     nanovdb::Coord subdivisionFactor) {
    unsigned int i = threadIdx.x;
    metadata[i].setTransform(fineVoxelSize(coarseMetadata[i].mVoxelSize, subdivisionFactor),
                             fineVoxelOrigin(coarseMetadata[i].mVoxelSize,
                                             coarseMetadata[i].voxelOrigin(),
                                             subdivisionFactor));
}

__global__ void
setCoarseTransformFromFineGridKernel(GridBatchImpl::GridMetadata *metadata,
                                     const GridBatchImpl::GridMetadata *fineMetadata,
                                     nanovdb::Coord coarseningFactor) {
    unsigned int i = threadIdx.x;
    metadata[i].setTransform(coarseVoxelSize(fineMetadata[i].mVoxelSize, coarseningFactor),
                             coarseVoxelOrigin(fineMetadata[i].mVoxelSize,
                                               fineMetadata[i].voxelOrigin(),
                                               coarseningFactor));
}

} // namespace

void
GridBatchImpl::setGlobalPrimalTransform(const VoxelCoordTransform &transform) {
    TORCH_CHECK(batchSize() > 0, "Cannot set global voxel size on an empty batch of grids");

    for (int64_t i = 0; i < batchSize(); i++) {
        mHostGridMetadata[i].mPrimalTransform = transform;
    }

    if (device().is_cuda() && batchSize()) {
        TORCH_CHECK(mDeviceGridMetadata);
        c10::cuda::CUDAGuard deviceGuard(device());

        auto wrapper = at::cuda::getCurrentCUDAStream(device().index());
        setGlobalPrimalTransformKernel<<<1, batchSize(), 0, wrapper.stream()>>>(mDeviceGridMetadata,
                                                                                transform);
    }
}

void
GridBatchImpl::setGlobalDualTransform(const VoxelCoordTransform &transform) {
    TORCH_CHECK(batchSize() > 0, "Cannot set global voxel size on an empty batch of grids");

    for (int64_t i = 0; i < batchSize(); i++) {
        mHostGridMetadata[i].mDualTransform = transform;
    }

    if (device().is_cuda() && batchSize()) {
        TORCH_CHECK(mDeviceGridMetadata);
        c10::cuda::CUDAGuard deviceGuard(device());

        auto wrapper = at::cuda::getCurrentCUDAStream(device().index());
        setGlobalDualTransformKernel<<<1, batchSize(), 0, wrapper.stream()>>>(mDeviceGridMetadata,
                                                                              transform);
    }
}

void
GridBatchImpl::setGlobalVoxelSize(const nanovdb::Vec3d &voxelSize) {
    c10::DeviceGuard guard(device());
    TORCH_CHECK(batchSize() > 0, "Cannot set global voxel size on an empty batch of grids");

    for (int64_t i = 0; i < batchSize(); i++) {
        mHostGridMetadata[i].setTransform(voxelSize, mHostGridMetadata[i].voxelOrigin());
    }

    if (device().is_cuda() && batchSize()) {
        TORCH_CHECK(mDeviceGridMetadata);
        c10::cuda::CUDAGuard deviceGuard(device());

        auto wrapper = at::cuda::getCurrentCUDAStream(device().index());
        setGlobalVoxelSizeKernel<<<1, batchSize(), 0, wrapper.stream()>>>(mDeviceGridMetadata,
                                                                          voxelSize);
    }
}

void
GridBatchImpl::setGlobalVoxelOrigin(const nanovdb::Vec3d &voxelOrigin) {
    c10::DeviceGuard guard(device());
    TORCH_CHECK(batchSize() > 0, "Cannot set global voxel origin on an empty batch of grids");

    for (int64_t i = 0; i < batchSize(); i++) {
        mHostGridMetadata[i].setTransform(mHostGridMetadata[i].mVoxelSize, voxelOrigin);
    }

    if (device().is_cuda() && batchSize()) {
        TORCH_CHECK(mDeviceGridMetadata);
        c10::cuda::CUDAGuard deviceGuard(device());

        auto wrapper = at::cuda::getCurrentCUDAStream(device().index());
        setGlobalVoxelOriginKernel<<<1, batchSize(), 0, wrapper.stream()>>>(mDeviceGridMetadata,
                                                                            voxelOrigin);
    }
}

void
GridBatchImpl::setGlobalVoxelSizeAndOrigin(const nanovdb::Vec3d &voxelSize,
                                           const nanovdb::Vec3d &voxelOrigin) {
    TORCH_CHECK(batchSize() > 0,
                "Cannot set global voxel size and origin on an empty batch of grids");

    for (int64_t i = 0; i < batchSize(); i++) {
        mHostGridMetadata[i].setTransform(voxelSize, voxelOrigin);
    }

    if (device().is_cuda() && batchSize()) {
        TORCH_CHECK(mDeviceGridMetadata);
        c10::cuda::CUDAGuard deviceGuard(device());

        auto wrapper = at::cuda::getCurrentCUDAStream(device().index());
        setGlobalVoxelSizeAndOriginKernel<<<1, batchSize(), 0, wrapper.stream()>>>(
            mDeviceGridMetadata, voxelSize, voxelOrigin);
    }
}

void
GridBatchImpl::setFineTransformFromCoarseGrid(const GridBatchImpl &coarseBatch,
                                              nanovdb::Coord subdivisionFactor) {
    TORCH_CHECK(batchSize() > 0,
                "Cannot set global voxel size and origin on an empty batch of grids");

    TORCH_CHECK(coarseBatch.batchSize() == batchSize(),
                "Coarse grid batch size must match fine grid batch size");
    TORCH_CHECK(subdivisionFactor[0] > 0 && subdivisionFactor[1] > 0 && subdivisionFactor[2] > 0,
                "Subdivision factor must be greater than 0");

    for (int64_t i = 0; i < batchSize(); i++) {
        mHostGridMetadata[i].setTransform(
            fineVoxelSize(coarseBatch.voxelSizeAt(i), subdivisionFactor),
            fineVoxelOrigin(
                coarseBatch.voxelSizeAt(i), coarseBatch.voxelOriginAt(i), subdivisionFactor));
    }

    if (device().is_cuda() && batchSize()) {
        TORCH_CHECK(mDeviceGridMetadata);
        TORCH_CHECK(coarseBatch.mDeviceGridMetadata);
        c10::cuda::CUDAGuard deviceGuard(device());

        auto wrapper = at::cuda::getCurrentCUDAStream(device().index());
        setFineTransformFromCoarseGridKernel<<<1, batchSize(), 0, wrapper.stream()>>>(
            mDeviceGridMetadata, coarseBatch.mDeviceGridMetadata, subdivisionFactor);
    }
}

void
GridBatchImpl::setCoarseTransformFromFineGrid(const GridBatchImpl &fineBatch,
                                              nanovdb::Coord coarseningFactor) {
    TORCH_CHECK(batchSize() > 0,
                "Cannot set global voxel size and origin on an empty batch of grids");

    TORCH_CHECK(fineBatch.batchSize() == batchSize(),
                "Fine grid batch size must match coarse grid batch size");
    TORCH_CHECK(coarseningFactor[0] > 0 && coarseningFactor[1] > 0 && coarseningFactor[2] > 0,
                "Coarsening factor must be greater than 0");

    for (int64_t i = 0; i < batchSize(); i++) {
        mHostGridMetadata[i].setTransform(
            coarseVoxelSize(fineBatch.voxelSizeAt(i), coarseningFactor),
            coarseVoxelOrigin(
                fineBatch.voxelSizeAt(i), fineBatch.voxelOriginAt(i), coarseningFactor));
    }

    if (device().is_cuda() && batchSize()) {
        TORCH_CHECK(mDeviceGridMetadata);
        TORCH_CHECK(fineBatch.mDeviceGridMetadata);
        c10::cuda::CUDAGuard deviceGuard(device());

        auto wrapper = at::cuda::getCurrentCUDAStream(device().index());
        setCoarseTransformFromFineGridKernel<<<1, batchSize(), 0, wrapper.stream()>>>(
            mDeviceGridMetadata, fineBatch.mDeviceGridMetadata, coarseningFactor);
    }
}

void
GridBatchImpl::setPrimalTransformFromDualGrid(const GridBatchImpl &dualBatch) {
    TORCH_CHECK(batchSize() > 0,
                "Cannot set global voxel size and origin on an empty batch of grids");

    TORCH_CHECK(dualBatch.batchSize() == batchSize(),
                "Dual grid batch size must match primal grid batch size");

    for (int64_t i = 0; i < batchSize(); i++) {
        mHostGridMetadata[i].mDualTransform   = dualBatch.mHostGridMetadata[i].mPrimalTransform;
        mHostGridMetadata[i].mPrimalTransform = dualBatch.mHostGridMetadata[i].mDualTransform;
        mHostGridMetadata[i].mVoxelSize       = dualBatch.mHostGridMetadata[i].mVoxelSize;
    }

    if (device().is_cuda() && batchSize()) {
        TORCH_CHECK(mDeviceGridMetadata);
        c10::cuda::CUDAGuard deviceGuard(device());

        auto wrapper = at::cuda::getCurrentCUDAStream(device().index());
        setPrimalTransformFromDualGridKernel<<<1, batchSize(), 0, wrapper.stream()>>>(
            mDeviceGridMetadata, dualBatch.mDeviceGridMetadata);
    }
}

void
GridBatchImpl::setGrid(nanovdb::GridHandle<TorchDeviceBuffer> &&gridHdl,
                       const torch::Tensor listIndices,
                       const std::vector<nanovdb::Vec3d> &voxelSizes,
                       const std::vector<nanovdb::Vec3d> &voxelOrigins,
                       bool blocking) {
    TORCH_CHECK(!gridHdl.buffer().isEmpty(), "Empty grid handle");
    TORCH_CHECK(voxelSizes.size() == gridHdl.gridCount(),
                "voxelSizes array does not have the same size as the number of grids, got ",
                voxelSizes.size(),
                " expected ",
                gridHdl.gridCount());
    TORCH_CHECK(voxelOrigins.size() == gridHdl.gridCount(),
                "Voxel origins must be the same size as the number of grids");
    TORCH_CHECK(gridHdl.gridType(0) == nanovdb::GridType::OnIndex,
                "GridBatchImpl only supports ValueOnIndex grids");

    // Reallocate GridMetadata
    mBatchSize = gridHdl.gridCount();

    const torch::Device device = gridHdl.buffer().device();
    if (device.is_cpu() || device.is_cuda()) {
        freeHostGridMetadata(mHostGridMetadata);
        mHostGridMetadata = allocateHostGridMetadata(gridHdl.gridCount());

        if (device.is_cuda()) {
            freeDeviceGridMetadata(device, mDeviceGridMetadata);
            mDeviceGridMetadata = allocateDeviceGridMetadata(device, mBatchSize);
        } else {
            TORCH_CHECK(!mDeviceGridMetadata, "mDeviceGridMetadata should be null for CPU devices");
        }
    } else if (device.is_privateuseone()) {
        freeUnifiedMemoryGridMetadata(mDeviceGridMetadata);
        mDeviceGridMetadata = allocateUnifiedMemoryGridMetadata(mBatchSize);
        mHostGridMetadata   = mDeviceGridMetadata;
    } else {
        TORCH_CHECK(false, "Only CPU, CUDA, and PrivateUse1 devices are supported");
    }

    ops::populateGridMetadata(gridHdl,
                              voxelSizes,
                              voxelOrigins,
                              mBatchOffsets,
                              mHostGridMetadata,
                              mDeviceGridMetadata,
                              &mBatchMetadata);
    TORCH_CHECK(listIndices.numel() == 0 || listIndices.size(0) == (mBatchOffsets.size(0) - 1),
                "Invalid list indices when building grid");
    mListIndices = listIndices;

    // FIXME: This is slow
    // Populate batch offsets for each leaf node
    {
        std::vector<torch::Tensor> leafBatchIdxs;
        leafBatchIdxs.reserve(gridHdl.gridCount());
        for (uint32_t i = 0; i < gridHdl.gridCount(); i += 1) {
            leafBatchIdxs.push_back(
                torch::full({mHostGridMetadata[i].mNumLeaves},
                            static_cast<fvdb::JIdxType>(i),
                            torch::TensorOptions().dtype(fvdb::JIdxScalarType).device(device)));
        }
        mLeafBatchIndices = torch::cat(leafBatchIdxs, 0);
    }

    // Replace the grid handle with the new one
    mGridHdl = std::make_shared<nanovdb::GridHandle<TorchDeviceBuffer>>(std::move(gridHdl));
}

c10::intrusive_ptr<GridBatchImpl>
GridBatchImpl::index(int64_t bi) const {
    c10::DeviceGuard guard(device());
    bi = negativeToPositiveIndexWithRangecheck(bi);

    return index(bi, bi + 1, 1);
}

c10::intrusive_ptr<GridBatchImpl>
GridBatchImpl::index(const torch::Tensor &indices) const {
    c10::DeviceGuard guard(device());
    TORCH_CHECK_INDEX(indices.dim() == 1, "indices must be a 1D tensor");
    TORCH_CHECK_INDEX(!indices.is_floating_point(), "indices must be an integer tensor");

    torch::Tensor numericIndices;
    if (indices.scalar_type() == torch::kBool) {
        TORCH_CHECK_INDEX(indices.dim() == 1, "bool indices must be a 1D tensor");
        TORCH_CHECK_INDEX(
            indices.numel() == batchSize(),
            "bool indices must have the same number of entries as grids in the batch");
        numericIndices = torch::arange(
            batchSize(), torch::TensorOptions().dtype(torch::kInt64).device(indices.device()));
        numericIndices = numericIndices.masked_select(indices);
    } else {
        numericIndices = indices;
    }

    torch::Tensor indicesCpu = numericIndices.to(torch::kCPU).to(torch::kInt64);
    auto indicesAccessor     = indicesCpu.accessor<int64_t, 1>();
    return indexInternal(indicesAccessor, indicesAccessor.size(0));
}

c10::intrusive_ptr<GridBatchImpl>
GridBatchImpl::index(const std::vector<int64_t> &indices) const {
    c10::DeviceGuard guard(device());
    return indexInternal(indices, indices.size());
}

c10::intrusive_ptr<GridBatchImpl>
GridBatchImpl::index(const std::vector<bool> &indices) const {
    c10::DeviceGuard guard(device());
    std::vector<int64_t> indicesInt;
    indicesInt.reserve(indices.size());
    for (size_t i = 0; i < indices.size(); i += 1) {
        if (indices[i]) {
            indicesInt.push_back(i);
        }
    }

    return indexInternal(indicesInt, indicesInt.size());
}

c10::intrusive_ptr<GridBatchImpl>
GridBatchImpl::index(ssize_t start, ssize_t stop, ssize_t step) const {
    c10::DeviceGuard guard(device());
    struct RangeAccessor {
        ssize_t mStart;
        ssize_t mStop;
        ssize_t mStep;
        ssize_t mLen;

        RangeAccessor(ssize_t start, ssize_t stop, ssize_t step, ssize_t batchSize)
            : mStart(start), mStop(stop), mStep(step) {
            TORCH_CHECK_INDEX(step != 0, "slice step cannot be zero");
            TORCH_CHECK_INDEX(0 <= start && start <= batchSize, "slice index out of range");
            TORCH_CHECK_INDEX(-1 <= stop && stop <= batchSize, "slice index out of range");

            if (stop <= start && step > 0) {
                mLen = 0;
            } else if (stop > start && step > 0) {
                mLen = (mStop - mStart + mStep - 1) / mStep;
            } else if (stop <= start && step < 0) {
                mLen = (mStart - mStop - mStep - 1) / -mStep;
            } else {
                TORCH_CHECK_INDEX(false,
                                  "Invalid slice start=",
                                  start,
                                  ", stop=",
                                  stop,
                                  ", step=",
                                  step,
                                  " for batch size ",
                                  batchSize);
            }
        }
        size_t
        operator[](size_t i) const {
            return mStart + i * mStep;
        }
    };

    auto acc = RangeAccessor(start, stop, step, batchSize());
    return indexInternal(acc, acc.mLen);
}

c10::intrusive_ptr<GridBatchImpl>
GridBatchImpl::concatenate(const std::vector<c10::intrusive_ptr<GridBatchImpl>> &elements) {
    TORCH_CHECK_VALUE(elements.size() > 0, "Must provide at least one grid for concatenate!")

    torch::Device device = elements[0]->device();

    std::vector<std::shared_ptr<nanovdb::GridHandle<TorchDeviceBuffer>>> handles;
    std::vector<std::vector<int64_t>> byteSizes;
    std::vector<std::vector<int64_t>> readByteOffsets;
    std::vector<std::vector<int64_t>> writeByteOffsets;
    int64_t totalByteSize = 0;
    int64_t totalGrids    = 0;
    handles.reserve(elements.size());
    byteSizes.reserve(elements.size());
    readByteOffsets.reserve(elements.size());
    writeByteOffsets.reserve(elements.size());

    std::vector<nanovdb::Vec3d> voxelSizes, voxelOrigins;

    for (size_t i = 0; i < elements.size(); i += 1) {
        TORCH_CHECK(elements[i]->device() == device,
                    "All grid batches must be on the same device!");

        // Empty grids don't contribute to the concatenation
        if (elements[i]->batchSize() == 0) {
            continue;
        }

        handles.push_back(elements[i]->mGridHdl);

        readByteOffsets.push_back(std::vector<int64_t>());
        writeByteOffsets.push_back(std::vector<int64_t>());
        byteSizes.push_back(std::vector<int64_t>());
        readByteOffsets.back().reserve(elements[i]->batchSize());
        writeByteOffsets.back().reserve(elements[i]->batchSize());
        byteSizes.back().reserve(elements[i]->batchSize());

        totalGrids += elements[i]->batchSize();

        for (int64_t j = 0; j < elements[i]->batchSize(); j += 1) {
            voxelSizes.push_back(elements[i]->voxelSizeAt(j));
            voxelOrigins.push_back(elements[i]->voxelOriginAt(j));

            readByteOffsets.back().push_back(
                elements[i]->cumBytesAt(j)); // Where to start reading from in the current grid
            byteSizes.back().push_back(elements[i]->numBytesAt(j)); // How many bytes to read
            writeByteOffsets.back().push_back(
                totalByteSize); // Where to start writing to in the concatenated grid
            totalByteSize += elements[i]->numBytesAt(j);
        }
    }
    if (handles.size() == 0) {
        return c10::make_intrusive<GridBatchImpl>(device);
    }

    TorchDeviceBuffer buffer(totalByteSize, device);

    int count         = 0;
    int nonEmptyCount = 0;
    if (device.is_cpu()) {
        for (size_t i = 0; i < elements.size(); i += 1) {
            if (elements[i]->batchSize() == 0) {
                continue;
            }

            for (int64_t j = 0; j < elements[i]->batchSize(); j += 1) {
                const int64_t readOffset  = readByteOffsets[nonEmptyCount][j];
                const int64_t writeOffset = writeByteOffsets[nonEmptyCount][j];
                const int64_t numBytes    = byteSizes[nonEmptyCount][j];

                nanovdb::GridData *dst =
                    reinterpret_cast<nanovdb::GridData *>(buffer.data() + writeOffset);
                const uint8_t *src = elements[i]->mGridHdl->buffer().data() + readOffset;
                memcpy((void *)dst, (void *)src, numBytes);
                nanovdb::tools::updateGridCount(dst, count++, totalGrids);
            }
            nonEmptyCount += 1;
        }
    } else {
        TORCH_CHECK(device.has_index(), "Device must have an index for CUDA operations");
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(device.index());

        for (size_t i = 0; i < elements.size(); i += 1) {
            if (elements[i]->batchSize() == 0) {
                continue;
            }

            for (int64_t j = 0; j < elements[i]->batchSize(); j += 1) {
                const int64_t readOffset  = readByteOffsets[nonEmptyCount][j];
                const int64_t writeOffset = writeByteOffsets[nonEmptyCount][j];
                const int64_t numBytes    = byteSizes[nonEmptyCount][j];

                c10::cuda::CUDAGuard deviceGuard(device.index());
                nanovdb::GridData *dst =
                    reinterpret_cast<nanovdb::GridData *>(buffer.deviceData() + writeOffset);
                const uint8_t *src = elements[i]->mGridHdl->buffer().deviceData() + readOffset;
                cudaMemcpyAsync((uint8_t *)dst, src, numBytes, cudaMemcpyDeviceToDevice, stream);

                updateGridCountAndZeroChecksum<<<1, 1, 0, stream>>>(dst, count++, totalGrids);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
            nonEmptyCount += 1;
        }
    }
    nanovdb::GridHandle<TorchDeviceBuffer> gridHdl =
        nanovdb::GridHandle<TorchDeviceBuffer>(std::move(buffer));
    return c10::make_intrusive<GridBatchImpl>(std::move(gridHdl), voxelSizes, voxelOrigins);
}

c10::intrusive_ptr<GridBatchImpl>
GridBatchImpl::contiguous(c10::intrusive_ptr<GridBatchImpl> input) {
    c10::DeviceGuard guard(input->device());
    if (input->isContiguous()) {
        return input;
    }

    const int64_t totalGrids = input->batchSize();

    int64_t totalByteSize = 0;
    for (int64_t i = 0; i < input->batchSize(); i += 1) {
        totalByteSize += input->numBytesAt(i);
    }

    TorchDeviceBuffer buffer(totalByteSize, input->device());

    int64_t writeOffset = 0;
    std::vector<nanovdb::Vec3d> voxelSizes, voxelOrigins;
    voxelSizes.reserve(input->batchSize());
    voxelOrigins.reserve(input->batchSize());

    if (input->device().is_cpu()) {
        for (int64_t i = 0; i < input->batchSize(); i += 1) {
            voxelSizes.push_back(input->voxelSizeAt(i));
            voxelOrigins.push_back(input->voxelOriginAt(i));

            nanovdb::GridData *dst =
                reinterpret_cast<nanovdb::GridData *>(buffer.data() + writeOffset);
            const uint8_t *src = input->nanoGridHandle().buffer().data() + input->cumBytesAt(i);
            memcpy((void *)dst, (void *)src, input->numBytesAt(i));
            nanovdb::tools::updateGridCount(dst, i, totalGrids);
            writeOffset += input->numBytesAt(i);
        }

    } else {
        TORCH_CHECK(input->device().has_index(), "Device must have an index for CUDA operations");
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(input->device().index());

        for (int64_t i = 0; i < input->batchSize(); i += 1) {
            voxelSizes.push_back(input->voxelSizeAt(i));
            voxelOrigins.push_back(input->voxelOriginAt(i));

            c10::cuda::CUDAGuard deviceGuard(input->device().index());
            nanovdb::GridData *dst =
                reinterpret_cast<nanovdb::GridData *>(buffer.deviceData() + writeOffset);
            const uint8_t *src =
                input->nanoGridHandle().buffer().deviceData() + input->cumBytesAt(i);
            cudaMemcpyAsync(
                (uint8_t *)dst, src, input->numBytesAt(i), cudaMemcpyDeviceToDevice, stream);

            updateGridCountAndZeroChecksum<<<1, 1, 0, stream>>>(dst, i, totalGrids);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            writeOffset += input->numBytesAt(i);
        }
    }

    return c10::make_intrusive<GridBatchImpl>(
        nanovdb::GridHandle<TorchDeviceBuffer>(std::move(buffer)), voxelSizes, voxelOrigins);
}

JaggedTensor
GridBatchImpl::jaggedTensor(const torch::Tensor &data) const {
    checkDevice(data);
    TORCH_CHECK(data.dim() >= 1, "Data have more than one dimensions");
    TORCH_CHECK(data.size(0) == totalVoxels(), "Data size mismatch");
    return JaggedTensor::from_data_offsets_and_list_ids(data, voxelOffsets(), jlidx());
}

torch::Tensor
GridBatchImpl::jidx() const {
    return ops::jIdxForGrid(*this);
}

torch::Tensor
GridBatchImpl::jlidx() const {
    return mListIndices;
}

torch::Tensor
GridBatchImpl::voxelOffsets() const {
    return mBatchOffsets;
}

torch::Tensor
GridBatchImpl::serialize() const {
    c10::DeviceGuard guard(device());
    return serializeV0();
}

c10::intrusive_ptr<GridBatchImpl>
GridBatchImpl::deserialize(const torch::Tensor &serialized) {
    c10::DeviceGuard guard(serialized.device());
    return deserializeV0(serialized);
}

torch::Tensor
GridBatchImpl::serializeV0() const {
    c10::intrusive_ptr<GridBatchImpl> self =
        c10::intrusive_ptr<GridBatchImpl>::reclaim_copy((GridBatchImpl *)this);
    if (!device().is_cpu()) {
        self = clone(torch::kCPU, true);
    }

    int64_t numGrids   = self->nanoGridHandle().gridCount();
    int64_t hdlBufSize = self->nanoGridHandle().buffer().size();

    struct V01Header {
        uint64_t magic   = 0x0F0F0F0F0F0F0F0F;
        uint64_t version = 0;
        uint64_t numGrids;
        uint64_t totalBytes;
    } header;

    const int64_t headerSize =
        sizeof(V01Header) + numGrids * sizeof(GridMetadata) + sizeof(GridBatchMetadata);
    const int64_t totalByteSize = headerSize + hdlBufSize;

    header.totalBytes = totalByteSize;
    header.numGrids   = numGrids;

    torch::Tensor ret = torch::empty({totalByteSize}, torch::kInt8);
    int8_t *retPtr    = ret.data_ptr<int8_t>();

    memcpy(retPtr, &header, sizeof(V01Header));
    retPtr += sizeof(V01Header);

    memcpy(retPtr, &self->mBatchMetadata, sizeof(GridBatchMetadata));
    retPtr += sizeof(GridBatchMetadata);

    memcpy(retPtr, self->mHostGridMetadata, numGrids * sizeof(GridMetadata));
    retPtr += numGrids * sizeof(GridMetadata);

    memcpy(retPtr, self->nanoGridHandle().buffer().data(), hdlBufSize);
    retPtr += hdlBufSize;

    TORCH_CHECK(retPtr == (ret.data_ptr<int8_t>() + totalByteSize),
                "Something went wrong with serialization");

    return ret;
}

c10::intrusive_ptr<GridBatchImpl>
GridBatchImpl::deserializeV0(const torch::Tensor &serialized) {
    struct V01Header {
        uint64_t magic   = 0x0F0F0F0F0F0F0F0F;
        uint64_t version = 0;
        uint64_t numGrids;
        uint64_t totalBytes;
    };

    TORCH_CHECK(serialized.scalar_type() == torch::kInt8, "Serialized data must be of type int8");
    TORCH_CHECK(serialized.numel() >= static_cast<int64_t>(sizeof(V01Header)),
                "Serialized data is too small to be a valid grid handle");

    const int8_t *serializedPtr = serialized.data_ptr<int8_t>();

    const V01Header *header = reinterpret_cast<const V01Header *>(serializedPtr);
    TORCH_CHECK(header->magic == 0x0F0F0F0F0F0F0F0F,
                "Serialized data is not a valid grid handle. Bad magic.");
    TORCH_CHECK(header->version == 0, "Serialized data is not a valid grid handle. Bad version.");
    TORCH_CHECK(static_cast<uint64_t>(serialized.numel()) == header->totalBytes,
                "Serialized data is not a valid grid handle. Bad total bytes.");

    const uint64_t numGrids = header->numGrids;

    const GridBatchMetadata *batchMetadata =
        reinterpret_cast<const GridBatchMetadata *>(serializedPtr + sizeof(V01Header));
    TORCH_CHECK(batchMetadata->version == 1,
                "Serialized data is not a valid grid handle. Bad batch metadata version.");

    const GridMetadata *gridMetadata = reinterpret_cast<const GridMetadata *>(
        serializedPtr + sizeof(V01Header) + sizeof(GridBatchMetadata));
    for (uint64_t i = 0; i < numGrids; i += 1) {
        TORCH_CHECK(gridMetadata[i].version == 1,
                    "Serialized data is not a valid grid handle. Bad grid metadata version.");
    }
    const int8_t *gridBuffer = serializedPtr + sizeof(V01Header) + sizeof(GridBatchMetadata) +
                               numGrids * sizeof(GridMetadata);

    const uint64_t sizeofMetadata =
        sizeof(V01Header) + sizeof(GridBatchMetadata) + numGrids * sizeof(GridMetadata);
    const uint64_t sizeofGrid = header->totalBytes - sizeofMetadata;

    auto buf = TorchDeviceBuffer(sizeofGrid, torch::kCPU);
    memcpy(buf.data(), gridBuffer, sizeofGrid);

    nanovdb::GridHandle gridHdl = nanovdb::GridHandle<TorchDeviceBuffer>(std::move(buf));

    std::vector<nanovdb::Vec3d> voxelSizes, voxelOrigins;
    voxelSizes.reserve(numGrids);
    voxelOrigins.reserve(numGrids);
    for (uint64_t i = 0; i < numGrids; i += 1) {
        voxelSizes.emplace_back(gridMetadata[i].mVoxelSize);
        voxelOrigins.emplace_back(gridMetadata[i].voxelOrigin());
    }

    return c10::make_intrusive<GridBatchImpl>(std::move(gridHdl), voxelSizes, voxelOrigins);
}

template c10::intrusive_ptr<GridBatchImpl>
GridBatchImpl::indexInternal(const torch::TensorAccessor<int64_t, 1> &idx, int64_t size) const;
template c10::intrusive_ptr<GridBatchImpl>
GridBatchImpl::indexInternal(const std::vector<int64_t> &idx, int64_t size) const;

c10 ::intrusive_ptr<GridBatchImpl>
GridBatchImpl::createFromEmpty(const torch::Device &device,
                               const nanovdb::Vec3d &voxelSize,
                               const nanovdb::Vec3d &origin) {
    return c10::make_intrusive<GridBatchImpl>(device, voxelSize, origin);
}

c10::intrusive_ptr<GridBatchImpl>
GridBatchImpl::createFromIjk(const JaggedTensor &ijk,
                             const std::vector<nanovdb::Vec3d> &voxelSizes,
                             const std::vector<nanovdb::Vec3d> &origins) {
    return detail::ops::createNanoGridFromIJK(ijk, voxelSizes, origins);
}

c10::intrusive_ptr<GridBatchImpl>
GridBatchImpl::createFromPoints(const JaggedTensor &points,
                                const std::vector<nanovdb::Vec3d> &voxelSizes,
                                const std::vector<nanovdb::Vec3d> &origins) {
    return detail::ops::buildGridFromPoints(points, voxelSizes, origins);
}

c10::intrusive_ptr<GridBatchImpl>
GridBatchImpl::createFromMesh(const JaggedTensor &meshVertices,
                              const JaggedTensor &meshFaces,
                              const std::vector<nanovdb::Vec3d> &voxelSizes,
                              const std::vector<nanovdb::Vec3d> &origins) {
    return detail::ops::buildGridFromMesh(meshVertices, meshFaces, voxelSizes, origins);
}

c10::intrusive_ptr<GridBatchImpl>
GridBatchImpl::dense(const int64_t numGrids,
                     const torch::Device &device,
                     const nanovdb::Coord &denseDims,
                     const nanovdb::Coord &ijkMin,
                     const std::vector<nanovdb::Vec3d> &voxelSizes,
                     const std::vector<nanovdb::Vec3d> &origins,
                     std::optional<torch::Tensor> mask) {
    return detail::ops::createNanoGridFromDense(
        numGrids, ijkMin, denseDims, device, mask, voxelSizes, origins);
}

c10::intrusive_ptr<GridBatchImpl>
GridBatchImpl::createFromNeighborVoxelsToPoints(const JaggedTensor &points,
                                                const std::vector<nanovdb::Vec3d> &voxelSizes,
                                                const std::vector<nanovdb::Vec3d> &origins) {
    return detail::ops::buildGridFromNearestVoxelsToPoints(points, voxelSizes, origins);
}

c10::intrusive_ptr<GridBatchImpl>
GridBatchImpl::coarsen(const nanovdb::Coord coarseningFactor) {
    if (batchSize() == 0) {
        return c10::make_intrusive<detail::GridBatchImpl>(device());
    }
    return detail::ops::buildCoarseGridFromFine(*this, coarseningFactor);
}

c10::intrusive_ptr<GridBatchImpl>
GridBatchImpl::upsample(const nanovdb::Coord upsampleFactor,
                        const std::optional<JaggedTensor> mask) {
    if (batchSize() == 0) {
        return c10::make_intrusive<detail::GridBatchImpl>(device());
    }
    return detail::ops::buildFineGridFromCoarse(*this, upsampleFactor, mask);
}

c10::intrusive_ptr<GridBatchImpl>
GridBatchImpl::dual(const bool excludeBorder) {
    if (batchSize() == 0) {
        return c10::make_intrusive<detail::GridBatchImpl>(device());
    }
    return detail::ops::buildPaddedGrid(*this, 0, 1, excludeBorder);
}

std::tuple<c10::intrusive_ptr<GridBatchImpl>, JaggedTensor>
GridBatchImpl::clipWithMask(const std::vector<nanovdb::Coord> &bboxMins,
                            const std::vector<nanovdb::Coord> &bboxMaxs) {
    JaggedTensor activeVoxelMask =
        fvdb::detail::ops::activeVoxelsInBoundsMask(*this, bboxMins, bboxMaxs);

    JaggedTensor activeVoxelCoords = fvdb::detail::ops::activeGridCoords(*this);

    // active voxel coords masked by the voxels in bounds
    JaggedTensor activeVoxelMaskCoords = activeVoxelCoords.rmask(activeVoxelMask.jdata());

    std::vector<nanovdb::Vec3d> voxS, voxO;
    gridVoxelSizesAndOrigins(voxS, voxO);
    if (batchSize() == 0) {
        return std::make_tuple(c10::make_intrusive<detail::GridBatchImpl>(device()),
                               activeVoxelMask);
    } else {
        auto clippedGridPtr = GridBatchImpl::createFromIjk(activeVoxelMaskCoords, voxS, voxO);
        return std::make_tuple(clippedGridPtr, activeVoxelMask);
    }
}

c10::intrusive_ptr<GridBatchImpl>
GridBatchImpl::clip(const std::vector<nanovdb::Coord> &ijkMin,
                    const std::vector<nanovdb::Coord> &ijkMax) {
    auto [clippedGridPtr, activeVoxelMask] = clipWithMask(ijkMin, ijkMax);
    return clippedGridPtr;
}

std::pair<JaggedTensor, c10::intrusive_ptr<GridBatchImpl>>
GridBatchImpl::clipFeaturesWithMask(const JaggedTensor &features,
                                    const std::vector<nanovdb::Coord> &ijkMin,
                                    const std::vector<nanovdb::Coord> &ijkMax) {
    TORCH_CHECK_VALUE(features.ldim() == 1,
                      "Expected features to have 1 list dimension, i.e. be a single list of "
                      "coordinate values, but got",
                      features.ldim(),
                      "list dimensions");
    checkDevice(features);
    TORCH_CHECK(features.rsize(0) == totalVoxels(), "Value count of features does not match grid");
    TORCH_CHECK(features.num_outer_lists() == batchSize(),
                "Batch size of features does not match grid.");
    TORCH_CHECK(torch::equal(features.joffsets(), voxelOffsets()),
                "Offsets of features does not match grid.");

    auto [clippedGridPtr, activeVoxelMask] = clipWithMask(ijkMin, ijkMax);
    JaggedTensor clippedFeatures           = features.rmask(activeVoxelMask.jdata());
    return {clippedFeatures, clippedGridPtr};
}

c10::intrusive_ptr<GridBatchImpl>
GridBatchImpl::dilate(const int64_t dilationAmount) {
    std::vector<int64_t> dilationAmountVec(batchSize(), dilationAmount);
    return dilate(dilationAmountVec);
}

c10::intrusive_ptr<GridBatchImpl>
GridBatchImpl::dilate(const std::vector<int64_t> dilationAmount) {
    if (batchSize() == 0) {
        return c10::make_intrusive<detail::GridBatchImpl>(device());
    }
    return detail::ops::dilateGrid(*this, dilationAmount);
}

c10::intrusive_ptr<GridBatchImpl>
GridBatchImpl::merge(c10::intrusive_ptr<GridBatchImpl> other) {
    if (batchSize() == 0) {
        return c10::make_intrusive<detail::GridBatchImpl>(device());
    }
    return detail::ops::mergeGrids(*this, *other);
}

c10::intrusive_ptr<GridBatchImpl>
GridBatchImpl::prune(const JaggedTensor &mask) {
    if (batchSize() == 0) {
        return c10::make_intrusive<detail::GridBatchImpl>(device());
    }
    return detail::ops::pruneGrid(*this, mask);
}

c10::intrusive_ptr<GridBatchImpl>
GridBatchImpl::convolutionOutput(const nanovdb::Coord kernelSize, const nanovdb::Coord stride) {
    if (batchSize() == 0) {
        return c10::make_intrusive<detail::GridBatchImpl>(device());
    }
    return detail::ops::buildGridForConv(*this, kernelSize, stride);
}

c10::intrusive_ptr<GridBatchImpl>
GridBatchImpl::convolutionTransposeOutput(const nanovdb::Coord kernelSize,
                                          const nanovdb::Coord stride) {
    if (batchSize() == 0) {
        return c10::make_intrusive<detail::GridBatchImpl>(device());
    }
    return detail::ops::buildGridForConvTranspose(*this, kernelSize, stride);
}

} // namespace detail
} // namespace fvdb
