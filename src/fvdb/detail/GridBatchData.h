// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_GRIDBATCHDATA_H
#define FVDB_DETAIL_GRIDBATCHDATA_H

#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/TorchDeviceBuffer.h>
#include <fvdb/detail/VoxelCoordTransform.h>

#include <nanovdb/GridHandle.h>
#include <nanovdb/NanoVDB.h>

#include <ATen/core/TensorBody.h>
#include <torch/types.h>

#include <vector>

#if !defined(__CUDACC__) && !defined(__restrict__)
#define __restrict__
#endif

namespace fvdb {
namespace detail {

struct GridBatchData : public torch::CustomClassHolder {
    static constexpr int64_t MAX_GRIDS_PER_BATCH = 1024; // Maximum number of grids in a batch

    // Metadata about a single grid in the batch
    struct GridMetadata {
        uint32_t version = 1;   // Version of this struct

        int64_t mCumLeaves = 0; // Cumulative number of leaf nodes in the batch up to this grid
        int64_t mCumVoxels = 0; // Cumulative number of voxels in the batch up to this grid
        uint64_t mCumBytes = 0; // Cumulative number of bytes in the buffer of grids up to this grid
        VoxelCoordTransform mPrimalTransform; // Primal Transform of this grid (i.e. transform which
                                              // aligns origin with voxel center)
        VoxelCoordTransform mDualTransform;   // Dual Transform of this grid (i.e. transform which
                                              // aligns origin with voxel corner)
        nanovdb::Vec3d mVoxelSize;            // Size of a single voxel in world space
        uint32_t mNumLeaves;                  // Number of leaf nodes in this grid
        int64_t mNumVoxels;                   // Number of voxels in this grid
        uint64_t mNumBytes;                   // Number of bytes in the buffer of this grid
        nanovdb::CoordBBox mBBox;             // Bounding box of this grid

        __hostdev__ nanovdb::Vec3d
        voxelOrigin() const {
            return mPrimalTransform.applyInv<double>(0., 0., 0.);
        }

        __hostdev__ void
        setTransform(const nanovdb::Vec3d &voxSize, const nanovdb::Vec3d &voxOrigin) {
            mVoxelSize = voxSize;
            voxelTransformForSizeAndOrigin(voxSize, voxOrigin, mPrimalTransform, mDualTransform);
        }
    };

    // Metadata about the whole batch
    struct GridBatchMetadata {
        uint32_t version = 1; // Version of this struct

        // Total number of leaf nodes across all grids
        int64_t mTotalLeaves = 0;

        // Total number of voxels across all grids
        int64_t mTotalVoxels = 0;

        // Maximum number of voxels in any grid. Used to set thread count
        int64_t mMaxVoxels = 0;

        // Maximum number of leaf nodes in any grid. Used to set thread count
        uint32_t mMaxLeafCount = 0;

        // Bounding box enclosing all the grids in the batch
        nanovdb::CoordBBox mTotalBBox;

        // Is this grid contiguous
        bool mIsContiguous = true;
    };

    // -----------------------------------------------------------------------
    // Data fields (all public, immutable after construction)
    // -----------------------------------------------------------------------
    GridMetadata *mHostGridMetadata{nullptr};   // CPU only
    GridMetadata *mDeviceGridMetadata{nullptr}; // CUDA only
    int64_t mBatchSize{0};
    GridBatchMetadata mBatchMetadata;           // Metadata about the whole batch
    std::shared_ptr<nanovdb::GridHandle<TorchDeviceBuffer>> mGridHdl; // NanoVDB grid handle
    torch::Tensor mLeafBatchIndices; // Indices of leaf nodes in the batch shape = [total_leafs]
    torch::Tensor mBatchOffsets;     // Batch indices for grid
    torch::Tensor mListIndices;      // List indices for grid (same as JaggedTensor)

    // -----------------------------------------------------------------------
    // Single constructor: bundles pre-computed fields (takes ownership of
    // metadata pointers). All computation happens outside, in factory
    // functions, before this constructor is called.
    // -----------------------------------------------------------------------
    GridBatchData(std::shared_ptr<nanovdb::GridHandle<TorchDeviceBuffer>> gridHdl,
                  GridMetadata *hostGridMetadata,
                  GridMetadata *deviceGridMetadata,
                  int64_t batchSize,
                  GridBatchMetadata batchMetadata,
                  torch::Tensor leafBatchIndices,
                  torch::Tensor batchOffsets,
                  torch::Tensor listIndices)
        : mHostGridMetadata(hostGridMetadata), mDeviceGridMetadata(deviceGridMetadata),
          mBatchSize(batchSize), mBatchMetadata(std::move(batchMetadata)),
          mGridHdl(std::move(gridHdl)), mLeafBatchIndices(std::move(leafBatchIndices)),
          mBatchOffsets(std::move(batchOffsets)), mListIndices(std::move(listIndices)) {}

    ~GridBatchData();

    GridBatchData &operator=(GridBatchData &&other) = delete;
    GridBatchData(GridBatchData &&other)            = delete;
    GridBatchData(GridBatchData &other)             = delete;
    GridBatchData &operator=(GridBatchData &other)  = delete;

    // -----------------------------------------------------------------------
    // Accessor (lightweight view for host/device kernels)
    // -----------------------------------------------------------------------
    class Accessor {
        const GridBatchData::GridMetadata *__restrict__ mMetadata = nullptr;
        const nanovdb::OnIndexGrid *__restrict__ mGridPtr         = nullptr;
        fvdb::JIdxType *__restrict__ mLeafBatchIndices            = nullptr;
        int64_t mTotalVoxels                                      = 0;
        int64_t mTotalLeaves                                      = 0;
        int64_t mMaxVoxels                                        = 0;
        uint32_t mMaxLeafCount                                    = 0;
        int64_t mGridCount                                        = 0;

        __hostdev__ inline int64_t
        negativeToPositiveIndexWithRangecheck(int64_t bi) const {
            if (bi < 0) {
                bi += batchSize();
            }
            assert(bi >= 0 && bi < batchSize());
            return static_cast<int64_t>(bi);
        }

      public:
        Accessor(const GridBatchData::GridMetadata *metadata,
                 const nanovdb::OnIndexGrid *gridPtr,
                 fvdb::JIdxType *leafBatchIndices,
                 int64_t totalVoxels,
                 int64_t totalLeaves,
                 int64_t maxVoxels,
                 uint32_t maxLeafCount,
                 int64_t gridCount)
            : mMetadata(metadata), mGridPtr(gridPtr), mLeafBatchIndices(leafBatchIndices),
              mTotalVoxels(totalVoxels), mTotalLeaves(totalLeaves), mMaxVoxels(maxVoxels),
              mMaxLeafCount(maxLeafCount), mGridCount(gridCount) {}

        __hostdev__ const nanovdb::OnIndexGrid *
        grid(int64_t bi) const {
            bi = negativeToPositiveIndexWithRangecheck(bi);
            return reinterpret_cast<const nanovdb::OnIndexGrid *>(
                reinterpret_cast<const char *>(mGridPtr) + mMetadata[bi].mCumBytes);
        }

        __hostdev__ nanovdb::CoordBBox
        bbox(int64_t bi) const {
            bi = negativeToPositiveIndexWithRangecheck(bi);
            return grid(bi)->tree().bbox();
        }

        __hostdev__ nanovdb::CoordBBox
        dualBbox(int64_t bi) const {
            bi = negativeToPositiveIndexWithRangecheck(bi);
            nanovdb::CoordBBox dualBbox(bbox(bi));
            dualBbox.mCoord[1] += nanovdb::Coord(1, 1, 1);
            return dualBbox;
        }

        __hostdev__ int64_t
        batchSize() const {
            return mGridCount;
        }

        __hostdev__ int64_t
        voxelOffset(int64_t bi) const {
            bi = negativeToPositiveIndexWithRangecheck(bi);
            return mMetadata[bi].mCumVoxels;
        }

        __hostdev__ int64_t
        leafOffset(int64_t bi) const {
            bi = negativeToPositiveIndexWithRangecheck(bi);
            return mMetadata[bi].mCumLeaves;
        }

        __hostdev__ int64_t
        maxVoxels() const {
            return mMaxVoxels;
        }

        __hostdev__ uint32_t
        maxLeafCount() const {
            return mMaxLeafCount;
        }

        __hostdev__ int64_t
        totalVoxels() const {
            return mTotalVoxels;
        }

        __hostdev__ int64_t
        totalLeaves() const {
            return mTotalLeaves;
        }

        __hostdev__ const VoxelCoordTransform &
        primalTransform(int64_t bi) const {
            bi = negativeToPositiveIndexWithRangecheck(bi);
            return mMetadata[bi].mPrimalTransform;
        }

        __hostdev__ const VoxelCoordTransform &
        dualTransform(int64_t bi) const {
            bi = negativeToPositiveIndexWithRangecheck(bi);
            return mMetadata[bi].mDualTransform;
        }

        __hostdev__ fvdb::JIdxType
        leafBatchIndex(int64_t leaf_idx) const {
            return mLeafBatchIndices[leaf_idx];
        }
    };

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------
    Accessor
    hostAccessor() const {
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

    Accessor
    deviceAccessor() const {
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

    // -----------------------------------------------------------------------
    // Scalar getters
    // -----------------------------------------------------------------------
    int64_t
    totalLeaves() const {
        return mBatchMetadata.mTotalLeaves;
    }

    int64_t
    totalVoxels() const {
        return mBatchMetadata.mTotalVoxels;
    }

    int64_t
    maxVoxelsPerGrid() const {
        return mBatchMetadata.mMaxVoxels;
    }

    int64_t
    maxLeavesPerGrid() const {
        return static_cast<int64_t>(mBatchMetadata.mMaxLeafCount);
    }

    int64_t
    batchSize() const {
        TORCH_CHECK(mBatchSize <= MAX_GRIDS_PER_BATCH,
                    "Cannot have more than ",
                    MAX_GRIDS_PER_BATCH,
                    " grids in a batch");
        return mBatchSize;
    }

    uint64_t
    totalBytes() const {
        uint64_t sum = 0;
        for (int64_t i = 0; i < mBatchSize; ++i) {
            sum += mHostGridMetadata[i].mNumBytes;
        }
        return sum;
    }

    const nanovdb::GridHandle<TorchDeviceBuffer> &
    nanoGridHandle() const {
        return *mGridHdl;
    }

    const c10::Device
    device() const {
        return mGridHdl->buffer().device();
    }

    bool
    isEmpty() const {
        return mGridHdl->buffer().isEmpty();
    }

    bool
    isContiguous() const {
        return mBatchMetadata.mIsContiguous;
    }

    // -----------------------------------------------------------------------
    // Per-grid scalar getters
    // -----------------------------------------------------------------------
    inline int64_t
    negativeToPositiveIndexWithRangecheck(int64_t bi) const {
        if (bi < 0) {
            bi += batchSize();
        }
        TORCH_CHECK_INDEX(bi >= 0 && bi < batchSize(),
                          "Batch index ",
                          bi,
                          " is out of range for grid batch of size " + std::to_string(batchSize()));
        return static_cast<int64_t>(bi);
    }

    uint32_t
    numLeavesAt(int64_t bi) const {
        bi = negativeToPositiveIndexWithRangecheck(bi);
        return mHostGridMetadata[bi].mNumLeaves;
    }

    int64_t
    numVoxelsAt(int64_t bi) const {
        bi = negativeToPositiveIndexWithRangecheck(bi);
        return mHostGridMetadata[bi].mNumVoxels;
    }

    int64_t
    cumVoxelsAt(int64_t bi) const {
        bi = negativeToPositiveIndexWithRangecheck(bi);
        return mHostGridMetadata[bi].mCumVoxels;
    }

    uint64_t
    numBytesAt(int64_t bi) const {
        bi = negativeToPositiveIndexWithRangecheck(bi);
        return mHostGridMetadata[bi].mNumBytes;
    }

    uint64_t
    cumBytesAt(int64_t bi) const {
        bi = negativeToPositiveIndexWithRangecheck(bi);
        return mHostGridMetadata[bi].mCumBytes;
    }

    const VoxelCoordTransform &
    primalTransformAt(int64_t bi) const {
        bi = negativeToPositiveIndexWithRangecheck(bi);
        return mHostGridMetadata[bi].mPrimalTransform;
    }

    const VoxelCoordTransform &
    dualTransformAt(int64_t bi) const {
        bi = negativeToPositiveIndexWithRangecheck(bi);
        return mHostGridMetadata[bi].mDualTransform;
    }

    const nanovdb::CoordBBox &
    totalBBox() const {
        return mBatchMetadata.mTotalBBox;
    }

    const nanovdb::CoordBBox &
    bboxAt(int64_t bi) const {
        checkNonEmptyGrid();
        bi = negativeToPositiveIndexWithRangecheck(bi);
        return mHostGridMetadata[bi].mBBox;
    }

    const nanovdb::CoordBBox
    dualBBoxAt(int64_t bi) const {
        bi                          = negativeToPositiveIndexWithRangecheck(bi);
        nanovdb::CoordBBox dualBbox = bboxAt(bi);
        dualBbox.mCoord[1] += nanovdb::Coord(1, 1, 1);
        return dualBbox;
    }

    const nanovdb::Vec3d &
    voxelSizeAt(int64_t bi) const {
        bi = negativeToPositiveIndexWithRangecheck(bi);
        return mHostGridMetadata[bi].mVoxelSize;
    }

    const nanovdb::Vec3d
    voxelOriginAt(int64_t bi) const {
        bi = negativeToPositiveIndexWithRangecheck(bi);
        return mHostGridMetadata[bi].voxelOrigin();
    }

    void
    gridVoxelSizesAndOrigins(std::vector<nanovdb::Vec3d> &outVoxelSizes,
                             std::vector<nanovdb::Vec3d> &outVoxelOrigins) const {
        outVoxelSizes.clear();
        outVoxelOrigins.clear();
        for (int64_t i = 0; i < batchSize(); ++i) {
            outVoxelSizes.emplace_back(mHostGridMetadata[i].mVoxelSize);
            outVoxelOrigins.emplace_back(mHostGridMetadata[i].voxelOrigin());
        }
    }

    // -----------------------------------------------------------------------
    // Tensor getters (declared here, defined in GridBatchData.cu)
    // -----------------------------------------------------------------------
    torch::Tensor numLeavesPerGridTensor() const;
    torch::Tensor numVoxelsPerGridTensor() const;
    torch::Tensor cumVoxelsPerGridTensor() const;
    torch::Tensor numBytesPerGridTensor() const;
    const torch::Tensor voxelOriginAtTensor(int64_t bi) const;
    const torch::Tensor voxelSizeAtTensor(int64_t bi) const;
    const torch::Tensor voxelSizesTensor() const;
    const torch::Tensor voxelOriginsTensor() const;
    const torch::Tensor bboxAtTensor(int64_t bi) const;
    const torch::Tensor bboxPerGridTensor() const;
    const torch::Tensor dualBBoxAtTensor(int64_t bi) const;
    const torch::Tensor dualBBoxPerGridTensor() const;
    const torch::Tensor totalBBoxTensor() const;
    const std::vector<VoxelCoordTransform> primalTransforms() const;
    const std::vector<VoxelCoordTransform> dualTransforms() const;
    torch::Tensor worldToGridMatrixAt(int64_t bi) const;
    torch::Tensor gridToWorldMatrixAt(int64_t bi) const;
    torch::Tensor gridToWorldMatrixPerGrid() const;
    torch::Tensor worldToGridMatrixPerGrid() const;

    torch::Tensor
    voxelOffsets() const {
        return mBatchOffsets;
    }

    torch::Tensor jlidx() const;
    torch::Tensor jidx() const;

    // -----------------------------------------------------------------------
    // Validation
    // -----------------------------------------------------------------------
    void
    checkNonEmptyGrid() const {
        TORCH_CHECK(!isEmpty(), "Empty grid");
    }

    void
    checkDevice(const torch::Tensor &t) const {
        torch::Device hdlDevice = mGridHdl->buffer().device();
        TORCH_CHECK(hdlDevice == t.device(),
                    "All tensors must be on the same device (" + hdlDevice.str() +
                        ") as index grid but got " + t.device().str());
    }

    void
    checkDevice(const JaggedTensor &t) const {
        torch::Device hdlDevice = mGridHdl->buffer().device();
        TORCH_CHECK(hdlDevice == t.device(),
                    "All tensors must be on the same device (" + hdlDevice.str() +
                        ") as index grid but got " + t.device().str());
    }

    void
    checkDevice(const std::optional<torch::Tensor> t) const {
        if (t.has_value()) {
            checkDevice(t.value());
        }
    }

    void
    checkDevice(const std::optional<JaggedTensor> t) const {
        if (t.has_value()) {
            checkDevice(t.value());
        }
    }

    // -----------------------------------------------------------------------
    // Utility
    // -----------------------------------------------------------------------
    JaggedTensor jaggedTensor(const torch::Tensor &data) const;
};

using BatchGridAccessor = typename GridBatchData::Accessor;

} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_GRIDBATCHDATA_H
