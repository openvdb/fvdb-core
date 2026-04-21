// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/GridBatchData.h>
#include <fvdb/detail/GridBatchDataFactory.h>
#include <fvdb/detail/ops/JIdxForGrid.h>

namespace fvdb {

// -----------------------------------------------------------------------
// Methods that dereference mGridHdl (require complete TorchDeviceBuffer)
// -----------------------------------------------------------------------

const nanovdb::GridHandle<TorchDeviceBuffer> &
GridBatchData::nanoGridHandle() const {
    return *mGridHdl;
}

const c10::Device
GridBatchData::device() const {
    return mGridHdl->buffer().device();
}

bool
GridBatchData::isEmpty() const {
    return mGridHdl->buffer().isEmpty();
}

GridBatchData::Accessor
GridBatchData::hostAccessor() const {
    TORCH_CHECK(!isEmpty(), "Cannot access empty grid");
    TORCH_CHECK(mGridHdl->template grid<nanovdb::ValueOnIndex>(),
                "Failed to get host grid pointer");
    return Accessor(mHostGridMetadata,
                    mGridHdl->template grid<nanovdb::ValueOnIndex>(),
                    mLeafBatchIndices.data_ptr<fvdb::JIdxType>(),
                    mBatchMetadata.mTotalVoxels,
                    mBatchMetadata.mTotalLeaves,
                    mBatchMetadata.mMaxVoxels,
                    mBatchMetadata.mMaxLeafCount,
                    mBatchSize);
}

GridBatchData::Accessor
GridBatchData::deviceAccessor() const {
    TORCH_CHECK(!isEmpty(), "Cannot access empty grid");
    TORCH_CHECK(device().is_cuda() || device().is_privateuseone(),
                "Cannot access device accessor without a CUDA or PrivateUse1 device");
    TORCH_CHECK(mGridHdl->template deviceGrid<nanovdb::ValueOnIndex>(),
                "Failed to get device grid pointer");
    return Accessor(mDeviceGridMetadata,
                    mGridHdl->template deviceGrid<nanovdb::ValueOnIndex>(),
                    mLeafBatchIndices.data_ptr<fvdb::JIdxType>(),
                    mBatchMetadata.mTotalVoxels,
                    mBatchMetadata.mTotalLeaves,
                    mBatchMetadata.mMaxVoxels,
                    mBatchMetadata.mMaxLeafCount,
                    mBatchSize);
}

void
GridBatchData::checkDevice(const torch::Tensor &t) const {
    torch::Device hdlDevice = mGridHdl->buffer().device();
    TORCH_CHECK(hdlDevice == t.device(),
                "All tensors must be on the same device (" + hdlDevice.str() +
                    ") as index grid but got " + t.device().str());
}

void
GridBatchData::checkDevice(const JaggedTensor &t) const {
    torch::Device hdlDevice = mGridHdl->buffer().device();
    TORCH_CHECK(hdlDevice == t.device(),
                "All tensors must be on the same device (" + hdlDevice.str() +
                    ") as index grid but got " + t.device().str());
}

GridBatchData::~GridBatchData() {
    if (!mGridHdl) {
        return;
    }
    const torch::Device dev = mGridHdl->buffer().device();
    if (dev.is_cpu() || dev.is_cuda()) {
        fvdb::detail::freeHostGridMetadata(mHostGridMetadata);
        mHostGridMetadata = nullptr;
        if (dev.is_cuda()) {
            fvdb::detail::freeDeviceGridMetadata(dev, mDeviceGridMetadata);
            mDeviceGridMetadata = nullptr;
        }
    } else if (dev.is_privateuseone()) {
        fvdb::detail::freeUnifiedMemoryGridMetadata(mDeviceGridMetadata);
        mHostGridMetadata   = nullptr;
        mDeviceGridMetadata = nullptr;
    } else {
        TORCH_CHECK(false, "Only CPU, CUDA, and PrivateUse1 devices are supported");
    }
}

torch::Tensor
GridBatchData::numLeavesPerGridTensor() const {
    torch::Tensor retTorch =
        torch::empty({batchSize()}, torch::TensorOptions().dtype(torch::kInt64));
    auto acc = retTorch.accessor<int64_t, 1>();
    for (int64_t bi = 0; bi < batchSize(); bi += 1) {
        acc[bi] = numLeavesAt(bi);
    }
    return retTorch;
}

torch::Tensor
GridBatchData::numVoxelsPerGridTensor() const {
    torch::Tensor retTorch =
        torch::empty({batchSize()}, torch::TensorOptions().dtype(torch::kInt64));
    auto acc = retTorch.accessor<int64_t, 1>();
    for (int64_t bi = 0; bi < batchSize(); bi += 1) {
        acc[bi] = numVoxelsAt(bi);
    }
    return retTorch;
}

torch::Tensor
GridBatchData::cumVoxelsPerGridTensor() const {
    torch::Tensor retTorch =
        torch::empty({batchSize()}, torch::TensorOptions().dtype(torch::kInt64));
    auto acc = retTorch.accessor<int64_t, 1>();
    for (int64_t bi = 0; bi < batchSize(); bi += 1) {
        acc[bi] = cumVoxelsAt(bi);
    }
    return retTorch;
}

torch::Tensor
GridBatchData::numBytesPerGridTensor() const {
    torch::Tensor retTorch =
        torch::empty({batchSize()}, torch::TensorOptions().dtype(torch::kInt64));
    auto acc = retTorch.accessor<int64_t, 1>();
    for (int64_t bi = 0; bi < batchSize(); bi += 1) {
        acc[bi] = numBytesAt(bi);
    }
    return retTorch;
}

const torch::Tensor
GridBatchData::voxelOriginAtTensor(int64_t bi) const {
    bi                = negativeToPositiveIndexWithRangecheck(bi);
    const auto origin = mHostGridMetadata[bi].voxelOrigin();

    auto ret = torch::empty({3}, torch::TensorOptions().dtype(torch::kFloat64));
    ret[0]   = origin[0];
    ret[1]   = origin[1];
    ret[2]   = origin[2];
    return ret;
}

const torch::Tensor
GridBatchData::voxelSizeAtTensor(int64_t bi) const {
    bi       = negativeToPositiveIndexWithRangecheck(bi);
    auto ret = torch::empty({3}, torch::TensorOptions().dtype(torch::kFloat64));
    ret[0]   = mHostGridMetadata[bi].mVoxelSize[0];
    ret[1]   = mHostGridMetadata[bi].mVoxelSize[1];
    ret[2]   = mHostGridMetadata[bi].mVoxelSize[2];
    return ret;
}

const torch::Tensor
GridBatchData::voxelSizesTensor() const {
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
GridBatchData::voxelOriginsTensor() const {
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
GridBatchData::bboxAtTensor(int64_t bi) const {
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
GridBatchData::bboxPerGridTensor() const {
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
GridBatchData::dualBBoxAtTensor(int64_t bi) const {
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
GridBatchData::dualBBoxPerGridTensor() const {
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
GridBatchData::totalBBoxTensor() const {
    const nanovdb::CoordBBox &bbox = this->totalBBox();
    return torch::tensor({{bbox.min()[0], bbox.min()[1], bbox.min()[2]},
                          {bbox.max()[0], bbox.max()[1], bbox.max()[2]}},
                         torch::TensorOptions().dtype(torch::kInt32));
}

const std::vector<VoxelCoordTransform>
GridBatchData::primalTransforms() const {
    std::vector<VoxelCoordTransform> transforms;
    transforms.reserve(batchSize());
    for (int64_t bi = 0; bi < batchSize(); ++bi) {
        transforms.push_back(primalTransformAt(bi));
    }
    return transforms;
}

const std::vector<VoxelCoordTransform>
GridBatchData::dualTransforms() const {
    std::vector<VoxelCoordTransform> transforms;
    transforms.reserve(batchSize());
    for (int64_t bi = 0; bi < batchSize(); ++bi) {
        transforms.push_back(dualTransformAt(bi));
    }
    return transforms;
}

torch::Tensor
GridBatchData::worldToGridMatrixAt(int64_t bi) const {
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

torch::Tensor
GridBatchData::gridToWorldMatrixAt(int64_t bi) const {
    bi = negativeToPositiveIndexWithRangecheck(bi);
    return at::linalg_inv(worldToGridMatrixAt(bi));
}

torch::Tensor
GridBatchData::gridToWorldMatrixPerGrid() const {
    std::vector<torch::Tensor> retTorch;
    for (int64_t bi = 0; bi < batchSize(); ++bi) {
        retTorch.emplace_back(gridToWorldMatrixAt(bi));
    }
    return torch::stack(retTorch, 0);
}

torch::Tensor
GridBatchData::worldToGridMatrixPerGrid() const {
    c10::DeviceGuard guard(device());
    std::vector<torch::Tensor> retTorch;
    for (int64_t bi = 0; bi < batchSize(); ++bi) {
        retTorch.emplace_back(worldToGridMatrixAt(bi));
    }
    return torch::stack(retTorch, 0);
}

JaggedTensor
GridBatchData::jaggedTensor(const torch::Tensor &data) const {
    checkDevice(data);
    TORCH_CHECK(data.dim() >= 1, "Data have more than one dimensions");
    TORCH_CHECK(data.size(0) == totalVoxels(), "Data size mismatch");
    return JaggedTensor::from_data_offsets_and_list_ids(data, voxelOffsets(), jlidx());
}

torch::Tensor
GridBatchData::jidx() const {
    return fvdb::detail::ops::jIdxForGrid(*this);
}

torch::Tensor
GridBatchData::jlidx() const {
    return mListIndices;
}

} // namespace fvdb
