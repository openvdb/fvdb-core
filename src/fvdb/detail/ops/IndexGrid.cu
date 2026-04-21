// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/GridBatchDataFactory.h>
#include <fvdb/detail/ops/IndexGrid.h>

#include <c10/cuda/CUDAGuard.h>

namespace {

template <typename Indexable>
c10::intrusive_ptr<fvdb::GridBatchData>
indexGridInternal(const fvdb::GridBatchData &grid, const Indexable &idx, int64_t size) {
    using namespace fvdb::detail;

    if (size == 0) {
        return makeEmptyGridBatchData(grid.device());
    }
    TORCH_CHECK(size >= 0,
                "Indexing with negative size is not supported (this should never happen)");

    fvdb::GridBatchData::GridMetadata *hostMeta   = allocateHostGridMetadata(size);
    fvdb::GridBatchData::GridMetadata *deviceMeta = nullptr;

    const torch::Device device = grid.device();
    if (device.is_cuda()) {
        deviceMeta = allocateDeviceGridMetadata(device, size);
    } else if (device.is_privateuseone()) {
        deviceMeta = allocateUnifiedMemoryGridMetadata(size);
        hostMeta   = deviceMeta;
    }

    int64_t cumVoxels     = 0;
    int64_t cumLeaves     = 0;
    int64_t maxVoxels     = 0;
    uint32_t maxLeafCount = 0;
    int64_t count         = 0;
    nanovdb::CoordBBox totalBbox;

    std::vector<torch::Tensor> leafBatchIdxs;

    bool isContiguous = grid.mBatchMetadata.mIsContiguous;
    for (int64_t i = 0; i < size; i += 1) {
        int64_t bi = idx[i];
        bi         = grid.negativeToPositiveIndexWithRangecheck(bi);

        isContiguous = isContiguous && (bi == count);

        const uint32_t numLeaves       = grid.mHostGridMetadata[bi].mNumLeaves;
        const int64_t numVoxels        = grid.mHostGridMetadata[bi].mNumVoxels;
        const nanovdb::CoordBBox &bbox = grid.mHostGridMetadata[bi].mBBox;

        hostMeta[count]            = grid.mHostGridMetadata[bi];
        hostMeta[count].mCumLeaves = cumLeaves;
        hostMeta[count].mCumVoxels = cumVoxels;

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
                        torch::TensorOptions().dtype(fvdb::JIdxScalarType).device(device)));
        count += 1;
    }

    fvdb::GridBatchData::GridBatchMetadata batchMeta;
    batchMeta.mIsContiguous = isContiguous && (count == grid.batchSize());
    batchMeta.mTotalLeaves  = cumLeaves;
    batchMeta.mTotalVoxels  = cumVoxels;
    batchMeta.mMaxVoxels    = maxVoxels;
    batchMeta.mMaxLeafCount = maxLeafCount;
    batchMeta.mTotalBBox    = totalBbox;

    torch::Tensor leafBatchIndices;
    if (leafBatchIdxs.size() > 0) {
        leafBatchIndices = torch::cat(leafBatchIdxs, 0);
    } else {
        leafBatchIndices =
            torch::empty({0}, torch::TensorOptions().dtype(fvdb::JIdxScalarType).device(device));
    }

    syncMetadataToDevice(hostMeta, deviceMeta, size, device, false);
    torch::Tensor batchOffsets = computeBatchOffsets(hostMeta, deviceMeta, size, device);

    torch::Tensor listIndices;
    if (grid.mListIndices.size(0) > 0) {
        TORCH_CHECK(false, "Nested lists of GridBatches are not supported yet");
    } else {
        listIndices = grid.mListIndices;
    }

    return c10::make_intrusive<fvdb::GridBatchData>(grid.mGridHdl,
                                                    hostMeta,
                                                    deviceMeta,
                                                    size,
                                                    std::move(batchMeta),
                                                    std::move(leafBatchIndices),
                                                    std::move(batchOffsets),
                                                    std::move(listIndices));
}

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

} // namespace

namespace fvdb {
namespace detail {
namespace ops {

c10::intrusive_ptr<GridBatchData>
indexGrid(const GridBatchData &grid, int64_t bi) {
    c10::DeviceGuard guard(grid.device());
    bi = grid.negativeToPositiveIndexWithRangecheck(bi);
    return indexGrid(grid, bi, bi + 1, 1);
}

c10::intrusive_ptr<GridBatchData>
indexGrid(const GridBatchData &grid, ssize_t start, ssize_t stop, ssize_t step) {
    c10::DeviceGuard guard(grid.device());
    auto acc = RangeAccessor(start, stop, step, grid.batchSize());
    return indexGridInternal(grid, acc, acc.mLen);
}

c10::intrusive_ptr<GridBatchData>
indexGrid(const GridBatchData &grid, const torch::Tensor &indices) {
    c10::DeviceGuard guard(grid.device());
    TORCH_CHECK_INDEX(indices.dim() == 1, "indices must be a 1D tensor");
    TORCH_CHECK_INDEX(!indices.is_floating_point(), "indices must be an integer tensor");

    torch::Tensor numericIndices;
    if (indices.scalar_type() == torch::kBool) {
        TORCH_CHECK_INDEX(indices.dim() == 1, "bool indices must be a 1D tensor");
        TORCH_CHECK_INDEX(
            indices.numel() == grid.batchSize(),
            "bool indices must have the same number of entries as grids in the batch");
        numericIndices = torch::arange(
            grid.batchSize(), torch::TensorOptions().dtype(torch::kInt64).device(indices.device()));
        numericIndices = numericIndices.masked_select(indices);
    } else {
        numericIndices = indices;
    }

    torch::Tensor indicesCpu = numericIndices.to(torch::kCPU).to(torch::kInt64);
    auto indicesAccessor     = indicesCpu.accessor<int64_t, 1>();
    return indexGridInternal(grid, indicesAccessor, indicesAccessor.size(0));
}

c10::intrusive_ptr<GridBatchData>
indexGrid(const GridBatchData &grid, const std::vector<int64_t> &indices) {
    c10::DeviceGuard guard(grid.device());
    return indexGridInternal(grid, indices, indices.size());
}

c10::intrusive_ptr<GridBatchData>
indexGrid(const GridBatchData &grid, const std::vector<bool> &indices) {
    c10::DeviceGuard guard(grid.device());
    std::vector<int64_t> indicesInt;
    indicesInt.reserve(indices.size());
    for (size_t i = 0; i < indices.size(); i += 1) {
        if (indices[i]) {
            indicesInt.push_back(i);
        }
    }
    return indexGridInternal(grid, indicesInt, indicesInt.size());
}

} // namespace ops
} // namespace detail
} // namespace fvdb
