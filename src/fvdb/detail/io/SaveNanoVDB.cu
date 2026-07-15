// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/io/SaveNanoVDB.h>
#include <fvdb/detail/utils/Utils.h>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/cuda/DeviceBuffer.h>
#include <nanovdb/io/IO.h>
#include <nanovdb/tools/GridChecksum.h>
#include <nanovdb/tools/cuda/IndexToGrid.cuh>

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/types.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <type_traits>

namespace fvdb {
namespace detail {
namespace io {

namespace {

/// @brief Copy a std::string to a char buffer with a fixed size and throw an exception if the
/// string is too long.
void
setFixedSizeStringBuf(char *targetBuf,
                      size_t maxSize,
                      const std::string &sourceString,
                      const std::string &bufName = "String") {
    std::memset(targetBuf, 0, maxSize);
    TORCH_CHECK_VALUE(sourceString.size() < maxSize,
                      bufName + " exceeds maximum character length of " + std::to_string(maxSize) +
                          ".");
    std::strncpy(targetBuf, sourceString.c_str(), maxSize);
}

/// @brief Validate that @p data's per-batch row counts (`joffsets[bi+1] - joffsets[bi]`) match
/// `gridBatchData.numVoxelsAt(bi)`. The save helpers index per-batch slices via the GridBatch's
/// cumulative voxel offsets, so this invariant is required for correctness.
void
checkPerBatchVoxelCounts(const GridBatchData &gridBatchData, const JaggedTensor &data) {
    torch::Tensor jOffsetsHost = data.joffsets().cpu().contiguous();
    auto joff                  = jOffsetsHost.accessor<JOffsetsType, 1>();
    TORCH_CHECK_VALUE(joff.size(0) == gridBatchData.batchSize() + 1,
                      "Jagged tensor joffsets length does not match grid batch size: expected ",
                      gridBatchData.batchSize() + 1,
                      " but got ",
                      joff.size(0));
    for (int64_t bi = 0; bi < gridBatchData.batchSize(); ++bi) {
        const int64_t expected = gridBatchData.numVoxelsAt(bi);
        const int64_t actual   = static_cast<int64_t>(joff[bi + 1] - joff[bi]);
        TORCH_CHECK_VALUE(expected == actual,
                          "Invalid number of voxels in jagged tensor at index ",
                          bi,
                          ": expected ",
                          expected,
                          " but got ",
                          actual);
    }
}

/// @brief Patch the GridData of a freshly-built NanoVDB grid (already laid out in @p gridBuf) with
/// fvdb-specific transform/name information and append a single fvdb_jdata blind metadata entry
/// followed by @p shapeInts shape ints describing the round-trip tensor shape.
///
/// Layout produced (offset 0 -> end):
///   [grid bytes (origGridBytes)]
///   [GridBlindMetaData header (sizeof(GridBlindMetaData))]
///   [shape ints (rdim+1) * sizeof(int64_t), zero-padded up to 32B]
///
/// @param gridBuf Pointer to the start of the grid in the host buffer (must hold @p totalBytes).
/// @param origGridBytes Size of the grid portion before appending fvdb blind metadata.
/// @param totalBytes Total bytes including blind data.
/// @param voxelSize World-space voxel size to write into mMap / mVoxelSize.
/// @param voxelOrigin World-space voxel origin to write into mMap.
/// @param name Grid name (checked against MaxNameSize).
/// @param shapeInts (rdim+1) int64s describing the per-batch tensor shape: [rdim, dim0, dim1, ...].
void
patchGridWithBlindShape(uint8_t *gridBuf,
                        const uint64_t origGridBytes,
                        const uint64_t totalBytes,
                        const nanovdb::Vec3d &voxelSize,
                        const nanovdb::Vec3d &voxelOrigin,
                        const std::string &name,
                        const std::vector<int64_t> &shapeInts) {
    const uint64_t paddedBlindDataBytes =
        nanovdb::math::AlignUp<32UL>(sizeof(int64_t) * shapeInts.size());
    TORCH_CHECK(totalBytes ==
                    origGridBytes + sizeof(nanovdb::GridBlindMetaData) + paddedBlindDataBytes,
                "Internal error: inconsistent buffer sizes for blind data layout.");

    nanovdb::GridData *gd    = reinterpret_cast<nanovdb::GridData *>(gridBuf);
    gd->mGridSize            = totalBytes;
    gd->mGridIndex           = 0;
    gd->mGridCount           = 1;
    gd->mBlindMetadataCount  = 1;
    gd->mBlindMetadataOffset = static_cast<int64_t>(origGridBytes);

    const double sx           = voxelSize[0];
    const double sy           = voxelSize[1];
    const double sz           = voxelSize[2];
    gd->mVoxelSize            = {sx, sy, sz};
    const double mat[3][3]    = {{sx, 0.0, 0.0}, {0.0, sy, 0.0}, {0.0, 0.0, sz}};
    const double invMat[3][3] = {{1.0 / sx, 0.0, 0.0}, {0.0, 1.0 / sy, 0.0}, {0.0, 0.0, 1.0 / sz}};
    nanovdb::Vec3d trans      = {voxelOrigin[0], voxelOrigin[1], voxelOrigin[2]};
    gd->mMap.set(mat, invMat, trans);

    setFixedSizeStringBuf(gd->mGridName, nanovdb::GridData::MaxNameSize, name, "Grid name " + name);

    nanovdb::GridBlindMetaData *bmd =
        reinterpret_cast<nanovdb::GridBlindMetaData *>(gridBuf + origGridBytes);
    bmd->mDataOffset = static_cast<int64_t>(sizeof(nanovdb::GridBlindMetaData));
    bmd->mValueCount = static_cast<uint64_t>(shapeInts.size());
    bmd->mValueSize  = static_cast<uint32_t>(sizeof(int64_t));
    bmd->mSemantic   = nanovdb::GridBlindDataSemantic::Unknown;
    bmd->mDataClass  = nanovdb::GridBlindDataClass::Unknown;
    bmd->mDataType   = nanovdb::GridType::Unknown;
    setFixedSizeStringBuf(
        bmd->mName, nanovdb::GridBlindMetaData::MaxNameSize, "fvdb_jdata", "blind metadata name");
    TORCH_CHECK(bmd->isValid(), "Invalid blind metadata");

    int64_t *shapeWriteHead =
        reinterpret_cast<int64_t *>(gridBuf + origGridBytes + sizeof(nanovdb::GridBlindMetaData));
    for (size_t i = 0; i < shapeInts.size(); ++i) {
        shapeWriteHead[i] = shapeInts[i];
    }
    const uint64_t writtenBytes = sizeof(int64_t) * shapeInts.size();
    if (paddedBlindDataBytes > writtenBytes) {
        std::memset(reinterpret_cast<uint8_t *>(shapeWriteHead) + writtenBytes,
                    0,
                    paddedBlindDataBytes - writtenBytes);
    }
}

/// @brief Patch a freshly-copied index grid's `GridData` into an fvdb ONINDEX tensor grid that
/// carries a single blind-metadata slot: sets grid class/type, the blind-metadata offset/count,
/// the world transform (voxel size + origin), the grid name, and the total grid size. Leaves
/// `mGridIndex`/`mGridCount` untouched so multi-grid handles keep their per-grid indexing.
///
/// @param gridData Pointer to the grid's `GridData` (start of the copied grid).
/// @param gridBytes Size of the copied grid, i.e. the offset of the appended blind metadata.
/// @param paddedPayloadBytes 32B-aligned size reserved for the blind payload after its header.
/// @param voxelSize World-space voxel size to write into mMap / mVoxelSize.
/// @param voxelOrigin World-space voxel origin (center of voxel [0,0,0]) to write into mMap.
/// @param name Grid name (checked against MaxNameSize); empty string clears the name.
void
patchOnIndexTensorGridData(nanovdb::GridData *gridData,
                           const uint64_t gridBytes,
                           const uint64_t paddedPayloadBytes,
                           const nanovdb::Vec3d &voxelSize,
                           const nanovdb::Vec3d &voxelOrigin,
                           const std::string &name) {
    gridData->mGridClass           = nanovdb::GridClass::TensorGrid;
    gridData->mGridType            = nanovdb::GridType::OnIndex;
    gridData->mBlindMetadataCount  = 1;
    gridData->mBlindMetadataOffset = static_cast<int64_t>(gridBytes);
    gridData->mGridSize = gridBytes + sizeof(nanovdb::GridBlindMetaData) + paddedPayloadBytes;

    const double scaleX            = voxelSize[0];
    const double scaleY            = voxelSize[1];
    const double scaleZ            = voxelSize[2];
    gridData->mVoxelSize           = {scaleX, scaleY, scaleZ};
    const double scaleMatrix[3][3] = {{scaleX, 0.0, 0.0}, {0.0, scaleY, 0.0}, {0.0, 0.0, scaleZ}};
    const double inverseScaleMatrix[3][3] = {
        {1.0 / scaleX, 0.0, 0.0}, {0.0, 1.0 / scaleY, 0.0}, {0.0, 0.0, 1.0 / scaleZ}};
    nanovdb::Vec3d translation = {voxelOrigin[0], voxelOrigin[1], voxelOrigin[2]};
    gridData->mMap.set(scaleMatrix, inverseScaleMatrix, translation);

    setFixedSizeStringBuf(
        gridData->mGridName, nanovdb::GridData::MaxNameSize, name, "Grid name " + name);
}

/// @brief Assemble a host buffer of ONINDEX tensor grids, one per grid in @p gridBatchData, each
/// carrying a single blind-metadata slot. This is the shared spine of `saveIndexGridWithBlindData`
/// (shape-prefixed jdata) and `toNVDBWithBlindFloat` (raw float32 values): it copies each source
/// grid (from host or device), patches its `GridData`, then hands off to @p writeBlind to fill the
/// per-grid blind descriptor and payload.
///
/// @param gridBatchData Source grids.
/// @param names Optional per-grid names (empty, or one per grid); written into each grid's name.
/// @param paddedPayloadBytes Per-grid 32B-aligned blind payload size; must match what @p writeBlind
///        writes for that grid, and is what the grid's `mGridSize` reserves.
/// @param writeBlind Callback `(batchIdx, blindMeta, payload)` invoked once per grid: fills the
///        `GridBlindMetaData` header @p blindMeta and writes the blind payload starting at
///        @p payload (the byte just past the header). It must not write more than
///        `paddedPayloadBytes[batchIdx]`.
/// @return A `GridHandle<HostBuffer>` owning the assembled multi-grid buffer.
template <typename WriteBlindFn>
nanovdb::GridHandle<nanovdb::HostBuffer>
assembleOnIndexBlindBuffer(const GridBatchData &gridBatchData,
                           const std::vector<std::string> &names,
                           const std::vector<uint64_t> &paddedPayloadBytes,
                           WriteBlindFn &&writeBlind) {
    const nanovdb::GridHandle<TorchDeviceBuffer> &nanoGridHdl = gridBatchData.nanoGridHandle();
    const bool isCuda = nanoGridHdl.buffer().device().is_cuda();

    uint64_t totalPayload = 0;
    for (const uint64_t payloadSize: paddedPayloadBytes) {
        totalPayload += payloadSize;
    }

    // Grids (already 32B aligned) + one blind-metadata header per grid + padded payloads.
    const size_t allocSize = nanoGridHdl.buffer().size() +
                             sizeof(nanovdb::GridBlindMetaData) * gridBatchData.batchSize() +
                             totalPayload;
    nanovdb::HostBuffer writeBuf(allocSize);

    uint8_t *writeHead = static_cast<uint8_t *>(writeBuf.data());
    uint8_t *readHead  = static_cast<uint8_t *>(isCuda ? nanoGridHdl.buffer().deviceData()
                                                       : nanoGridHdl.buffer().data());

    for (int64_t batchIdx = 0; batchIdx < gridBatchData.batchSize(); ++batchIdx) {
        // Copy this batch's index grid to the buffer. D2H copies into pageable host memory are
        // synchronous w.r.t. the host, so the immediate host-side patches below are safe; the
        // final stream sync is belt-and-suspenders.
        const size_t gridBytes = nanoGridHdl.gridSize(batchIdx);
        if (isCuda) {
            c10::cuda::CUDAGuard deviceGuard(gridBatchData.device());
            at::cuda::CUDAStream stream =
                at::cuda::getCurrentCUDAStream(gridBatchData.device().index());
            cudaMemcpyAsync((void *)writeHead, (void *)readHead, gridBytes, cudaMemcpyDeviceToHost,
                            stream.stream());
        } else {
            std::memcpy((void *)writeHead, (void *)readHead, gridBytes);
        }

        const std::string name = names.empty() ? std::string() : names[batchIdx];
        patchOnIndexTensorGridData(reinterpret_cast<nanovdb::GridData *>(writeHead), gridBytes,
                                   paddedPayloadBytes[batchIdx], gridBatchData.voxelSizeAt(batchIdx),
                                   gridBatchData.voxelOriginAt(batchIdx), name);

        readHead  += gridBytes;
        writeHead += gridBytes;

        nanovdb::GridBlindMetaData *blindMeta =
            reinterpret_cast<nanovdb::GridBlindMetaData *>(writeHead);
        writeBlind(batchIdx, blindMeta, writeHead + sizeof(nanovdb::GridBlindMetaData));
        TORCH_CHECK(blindMeta->isValid(), "Invalid blind metadata");
        writeHead += sizeof(nanovdb::GridBlindMetaData) + paddedPayloadBytes[batchIdx];
    }

    if (isCuda) {
        at::cuda::CUDAStream stream =
            at::cuda::getCurrentCUDAStream(gridBatchData.device().index());
        cudaStreamSynchronize(stream.stream());
    }

    return nanovdb::GridHandle<nanovdb::HostBuffer>(std::move(writeBuf));
}

/// @brief Host-only port of `nanovdb::tools::cuda::indexToGrid` for a single
/// `nanovdb::NanoGrid<nanovdb::ValueOnIndex>`.
///
/// Mirrors the layout and per-node operations of `IndexToGrid.cuh`: copies the grid/tree/root
/// headers, fixes child offsets where the destination node sizes differ from the source, and
/// fills every internal-node value slot and leaf voxel slot via `srcValues[srcIndex]`. The
/// kernel's per-thread/per-block work becomes a sequential loop on the host, so CPU-resident
/// grids avoid CUDA context setup, CUDA allocations, and host/device copies.
///
/// @tparam DstBuildT Destination NanoVDB build type (e.g. `float`, `nanovdb::Vec3f`).
/// @param srcGrid Host pointer to the source `NanoGrid<ValueOnIndex>`.
/// @param srcValues Host pointer to a flat array of per-voxel values, indexed by the source
///        grid's per-voxel index. `srcValues[0]` is the background slot; `srcValues[1..N]` are
///        the active voxel values (matches the convention used by `indexToGrid`).
/// @return A new `GridHandle<HostBuffer>` wrapping the freshly built typed grid.
template <typename DstBuildT>
nanovdb::GridHandle<nanovdb::HostBuffer>
indexToGridHost(const nanovdb::NanoGrid<nanovdb::ValueOnIndex> *srcGrid,
                const typename nanovdb::BuildToValueMap<DstBuildT>::type *srcValues) {
    using SrcBuildT = nanovdb::ValueOnIndex;

    using DstGridT  = nanovdb::NanoGrid<DstBuildT>;
    using DstTreeT  = nanovdb::NanoTree<DstBuildT>;
    using DstRootT  = nanovdb::NanoRoot<DstBuildT>;
    using DstUpperT = nanovdb::NanoUpper<DstBuildT>;
    using DstLowerT = nanovdb::NanoLower<DstBuildT>;
    using DstLeafT  = nanovdb::NanoLeaf<DstBuildT>;

    using SrcRootT  = nanovdb::NanoRoot<SrcBuildT>;
    using SrcUpperT = nanovdb::NanoUpper<SrcBuildT>;
    using SrcLowerT = nanovdb::NanoLower<SrcBuildT>;
    using SrcLeafT  = nanovdb::NanoLeaf<SrcBuildT>;

    using SrcValueT = typename nanovdb::BuildToValueMap<DstBuildT>::type;
    using DstStatsT = typename nanovdb::NanoRoot<DstBuildT>::FloatType;

    static_assert(!nanovdb::BuildTraits<DstBuildT>::is_special,
                  "indexToGridHost: destination build type must not be a special type "
                  "(Fp16, ValueIndex, Boolean, etc.)");

    const auto &srcTree         = srcGrid->tree();
    const uint32_t nodeCount[4] = {
        srcTree.nodeCount(0),
        srcTree.nodeCount(1),
        srcTree.nodeCount(2),
        static_cast<uint32_t>(srcTree.root().tileCount()),
    };

    const uint64_t offGrid   = 0;
    const uint64_t offTree   = DstGridT::memUsage();
    const uint64_t offRoot   = offTree + DstTreeT::memUsage();
    const uint64_t offNode2  = offRoot + DstRootT::memUsage(nodeCount[3]);
    const uint64_t offNode1  = offNode2 + DstUpperT::memUsage() * nodeCount[2];
    const uint64_t offNode0  = offNode1 + DstLowerT::memUsage() * nodeCount[1];
    const uint64_t totalSize = offNode0 + DstLeafT::DataType::memUsage() * nodeCount[0];

    nanovdb::HostBuffer buffer(totalSize);
    uint8_t *dstPtr = static_cast<uint8_t *>(buffer.data());

    DstGridT *dstGrid             = reinterpret_cast<DstGridT *>(dstPtr + offGrid);
    DstTreeT *dstTree             = reinterpret_cast<DstTreeT *>(dstPtr + offTree);
    DstRootT *dstRoot             = reinterpret_cast<DstRootT *>(dstPtr + offRoot);
    DstUpperT *const dstUpperBase = reinterpret_cast<DstUpperT *>(dstPtr + offNode2);
    DstLowerT *const dstLowerBase = reinterpret_cast<DstLowerT *>(dstPtr + offNode1);
    DstLeafT *const dstLeafBase   = reinterpret_cast<DstLeafT *>(dstPtr + offNode0);

    *dstGrid->data()   = *srcGrid->data();
    dstGrid->mGridType = nanovdb::toGridType<DstBuildT>();
    dstGrid->mData1    = 0u;

    *dstTree->data() = *srcTree.data();
    dstTree->setRoot(dstRoot);
    dstTree->setFirstNode(dstUpperBase);
    dstTree->setFirstNode(dstLowerBase);
    dstTree->setFirstNode(dstLeafBase);

    const SrcRootT &srcRoot = srcTree.root();
    dstRoot->mBBox          = srcRoot.mBBox;
    dstRoot->mTableSize     = srcRoot.mTableSize;
    dstRoot->mBackground    = srcValues[srcRoot.mBackground];
    if (srcGrid->hasMinMax()) {
        dstRoot->mMinimum = srcValues[srcRoot.mMinimum];
        dstRoot->mMaximum = srcValues[srcRoot.mMaximum];
    }
    if constexpr (std::is_same_v<SrcValueT, DstStatsT>) {
        if (srcGrid->hasAverage())
            dstRoot->mAverage = srcValues[srcRoot.mAverage];
        if (srcGrid->hasStdDeviation())
            dstRoot->mStdDevi = srcValues[srcRoot.mStdDevi];
    }

    const uint32_t tileCount = nodeCount[3];
    const uint64_t srcRootChildBase =
        sizeof(SrcRootT) + tileCount * sizeof(typename SrcRootT::Tile);
    const uint64_t dstRootChildBase =
        sizeof(DstRootT) + tileCount * sizeof(typename DstRootT::Tile);
    for (uint32_t tileID = 0; tileID < tileCount; ++tileID) {
        const auto &srcTile = *srcRoot.tile(tileID);
        auto &dstTile       = *dstRoot->tile(tileID);
        dstTile.key         = srcTile.key;
        if (srcTile.child) {
            const uint64_t childID =
                (srcTile.child - srcRootChildBase) / sizeof(typename SrcRootT::ChildNodeType);
            dstTile.child = dstRootChildBase + childID * sizeof(typename DstRootT::ChildNodeType);
            dstTile.value = srcValues[0]; // background slot
            dstTile.state = false;
        } else {
            dstTile.child = 0;
            dstTile.value = srcValues[srcTile.value];
            dstTile.state = srcTile.state;
        }
    }

    for (uint32_t nodeID = 0; nodeID < nodeCount[2]; ++nodeID) {
        const SrcUpperT *srcNode = srcTree.template getFirstNode<2>() + nodeID;
        DstUpperT *dstNode       = dstUpperBase + nodeID;

        dstNode->mBBox      = srcNode->mBBox;
        dstNode->mFlags     = srcNode->mFlags;
        dstNode->mValueMask = srcNode->mValueMask;
        dstNode->mChildMask = srcNode->mChildMask;
        if (srcGrid->hasMinMax()) {
            dstNode->mMinimum = srcValues[srcNode->mMinimum];
            dstNode->mMaximum = srcValues[srcNode->mMaximum];
        }
        if constexpr (std::is_same_v<SrcValueT, DstStatsT>) {
            if (srcGrid->hasAverage())
                dstNode->mAverage = srcValues[srcNode->mAverage];
            if (srcGrid->hasStdDeviation())
                dstNode->mStdDevi = srcValues[srcNode->mStdDevi];
        }
        constexpr uint32_t SIZE = SrcUpperT::SIZE;
        for (uint32_t i = 0; i < SIZE; ++i) {
            if (srcNode->mChildMask.isOn(i)) {
                if constexpr (sizeof(SrcUpperT) == sizeof(DstUpperT) &&
                              sizeof(SrcLowerT) == sizeof(DstLowerT)) {
                    dstNode->mTable[i].child = srcNode->mTable[i].child;
                } else {
                    const uint64_t nodeSkip = nodeCount[2] - nodeID;
                    const uint64_t srcOff   = sizeof(SrcUpperT) * nodeSkip;
                    const uint64_t dstOff   = sizeof(DstUpperT) * nodeSkip;
                    const uint64_t childID =
                        (srcNode->mTable[i].child - srcOff) / sizeof(SrcLowerT);
                    dstNode->mTable[i].child = dstOff + childID * sizeof(DstLowerT);
                }
            } else {
                dstNode->mTable[i].value = srcValues[srcNode->mTable[i].value];
            }
        }
    }

    for (uint32_t nodeID = 0; nodeID < nodeCount[1]; ++nodeID) {
        const SrcLowerT *srcNode = srcTree.template getFirstNode<1>() + nodeID;
        DstLowerT *dstNode       = dstLowerBase + nodeID;

        dstNode->mBBox      = srcNode->mBBox;
        dstNode->mFlags     = srcNode->mFlags;
        dstNode->mValueMask = srcNode->mValueMask;
        dstNode->mChildMask = srcNode->mChildMask;
        if (srcGrid->hasMinMax()) {
            dstNode->mMinimum = srcValues[srcNode->mMinimum];
            dstNode->mMaximum = srcValues[srcNode->mMaximum];
        }
        if constexpr (std::is_same_v<SrcValueT, DstStatsT>) {
            if (srcGrid->hasAverage())
                dstNode->mAverage = srcValues[srcNode->mAverage];
            if (srcGrid->hasStdDeviation())
                dstNode->mStdDevi = srcValues[srcNode->mStdDevi];
        }
        constexpr uint32_t SIZE = SrcLowerT::SIZE;
        for (uint32_t i = 0; i < SIZE; ++i) {
            if (srcNode->mChildMask.isOn(i)) {
                if constexpr (sizeof(SrcLowerT) == sizeof(DstLowerT) &&
                              sizeof(SrcLeafT) == sizeof(DstLeafT)) {
                    dstNode->mTable[i].child = srcNode->mTable[i].child;
                } else {
                    const uint64_t nodeSkip = nodeCount[1] - nodeID;
                    const uint64_t srcOff   = sizeof(SrcLowerT) * nodeSkip;
                    const uint64_t dstOff   = sizeof(DstLowerT) * nodeSkip;
                    const uint64_t childID = (srcNode->mTable[i].child - srcOff) / sizeof(SrcLeafT);
                    dstNode->mTable[i].child = dstOff + childID * sizeof(DstLeafT);
                }
            } else {
                dstNode->mTable[i].value = srcValues[srcNode->mTable[i].value];
            }
        }
    }

    for (uint32_t leafID = 0; leafID < nodeCount[0]; ++leafID) {
        const SrcLeafT *srcLeaf = srcTree.template getFirstNode<0>() + leafID;
        DstLeafT *dstLeaf       = dstLeafBase + leafID;

        dstLeaf->mBBoxMin = srcLeaf->mBBoxMin;
        for (int i = 0; i < 3; ++i)
            dstLeaf->mBBoxDif[i] = srcLeaf->mBBoxDif[i];
        dstLeaf->mFlags     = srcLeaf->mFlags;
        dstLeaf->mValueMask = srcLeaf->mValueMask;
        if (srcGrid->hasMinMax()) {
            dstLeaf->mMinimum = srcValues[srcLeaf->getMin()];
            dstLeaf->mMaximum = srcValues[srcLeaf->getMax()];
        }
        if constexpr (std::is_same_v<SrcValueT, DstStatsT>) {
            if (srcGrid->hasAverage())
                dstLeaf->mAverage = srcValues[srcLeaf->getAvg()];
            if (srcGrid->hasStdDeviation())
                dstLeaf->mStdDevi = srcValues[srcLeaf->getDev()];
        }
        constexpr uint32_t LEAF_SIZE = 512u;
        for (uint32_t i = 0; i < LEAF_SIZE; ++i) {
            dstLeaf->mValues[i] = srcValues[srcLeaf->getValue(i)];
        }
    }

    nanovdb::tools::updateChecksum(dstGrid);

    return nanovdb::GridHandle<nanovdb::HostBuffer>(std::move(buffer));
}

} // namespace

/// @brief Host-only build path for `fvdbToNanovdbGridWithValues`.
///
/// The public save/toNVDB entry points require data and GridBatchData to live on the same device,
/// so this path is used for CPU-resident grid+data pairs and avoids the H2D upload, kernel launch,
/// and D2H copy that the CUDA path would otherwise incur.
template <typename OutBuildT, typename TorchScalarT>
nanovdb::GridHandle<nanovdb::HostBuffer>
fvdbToNanovdbGridWithValuesHost(const GridBatchData &gridBatchData,
                                const JaggedTensor &data,
                                const std::vector<std::string> &names) {
    using HostGridHandle = nanovdb::GridHandle<nanovdb::HostBuffer>;
    using ValueT         = typename nanovdb::BuildToValueMap<OutBuildT>::type;

    JaggedTensor cpuData = data;
    if (cpuData.is_cuda()) {
        cpuData = cpuData.cpu();
    }
    if (!cpuData.is_contiguous()) {
        cpuData = cpuData.contiguous();
    }

    const int64_t rdim = cpuData.rdim();
    std::vector<int64_t> tailSizes;
    tailSizes.reserve(rdim > 0 ? static_cast<size_t>(rdim - 1) : 0);
    for (int64_t di = 1; di < rdim; ++di) {
        tailSizes.push_back(cpuData.rsize(di));
    }
    const uint64_t shapeBytes           = sizeof(int64_t) * (rdim + 1);
    const uint64_t paddedBlindDataBytes = nanovdb::math::AlignUp<32UL>(shapeBytes);
    const uint64_t blindOverhead        = sizeof(nanovdb::GridBlindMetaData) + paddedBlindDataBytes;

    const uint8_t *hSrcBufferStart =
        static_cast<const uint8_t *>(gridBatchData.nanoGridHandle().buffer().data());
    const ValueT *hDataValuesBase = reinterpret_cast<const ValueT *>(cpuData.jdata().data_ptr());

    // Per-batch values buffer of size `numVoxels + 1`: slot [0] is the inactive/background value
    // (zero), slots [1..N] are the user data slice. Reused across batches if the new size fits.
    std::vector<ValueT> valueBuf;

    std::vector<HostGridHandle> buffers;
    buffers.reserve(gridBatchData.batchSize());

    for (int64_t bi = 0; bi < gridBatchData.batchSize(); ++bi) {
        const std::string name = names.size() > 0 ? names[bi] : "";
        TORCH_CHECK_VALUE(name.size() < nanovdb::GridData::MaxNameSize,
                          "Grid name " + name + " exceeds maximum character length of " +
                              std::to_string(nanovdb::GridData::MaxNameSize) + ".");

        const int64_t numVoxelsBi = gridBatchData.numVoxelsAt(bi);
        const int64_t cumVoxelsBi = gridBatchData.cumVoxelsAt(bi);
        TORCH_CHECK_VALUE(
            numVoxelsBi >= 0, "Invalid number of voxels at grid index ", bi, ": ", numVoxelsBi);

        const auto *hSrcGrid = reinterpret_cast<const nanovdb::NanoGrid<nanovdb::ValueOnIndex> *>(
            hSrcBufferStart + gridBatchData.cumBytesAt(bi));

        const size_t valueBufElems = static_cast<size_t>(numVoxelsBi) + 1u;
        if (valueBuf.size() < valueBufElems) {
            valueBuf.resize(valueBufElems);
        }
        valueBuf[0] = ValueT{}; // background / inactive slot
        if (numVoxelsBi > 0) {
            std::memcpy(valueBuf.data() + 1,
                        hDataValuesBase + cumVoxelsBi,
                        static_cast<size_t>(numVoxelsBi) * sizeof(ValueT));
        }

        HostGridHandle gh = indexToGridHost<OutBuildT>(hSrcGrid, valueBuf.data());

        const uint64_t origGridBytes = gh.buffer().size();
        const uint64_t totalBytes    = origGridBytes + blindOverhead;

        nanovdb::HostBuffer outBuf(totalBytes);
        std::memcpy(outBuf.data(), gh.buffer().data(), origGridBytes);

        std::vector<int64_t> shapeInts;
        shapeInts.reserve(static_cast<size_t>(rdim + 1));
        shapeInts.push_back(static_cast<int64_t>(rdim));
        shapeInts.push_back(numVoxelsBi);
        for (int64_t s: tailSizes) {
            shapeInts.push_back(s);
        }
        TORCH_CHECK(static_cast<int64_t>(shapeInts.size()) == rdim + 1,
                    "Internal error: blind data shape int count mismatch.");

        patchGridWithBlindShape(static_cast<uint8_t *>(outBuf.data()),
                                origGridBytes,
                                totalBytes,
                                gridBatchData.voxelSizeAt(bi),
                                gridBatchData.voxelOriginAt(bi),
                                name,
                                shapeInts);

        buffers.emplace_back(std::move(outBuf));
    }

    if (buffers.size() == 1) {
        return std::move(buffers[0]);
    }
    return nanovdb::mergeGrids(buffers);
}

/// @brief Helper function to build a NanoVDB grid (e.g. nanovdb::FloatGrid, Vec3fGrid, ...) from
/// an existing fvdb ValueOnIndex grid batch and a JaggedTensor of per-voxel values.
///
/// CPU-resident inputs use the host port above; CUDA-resident inputs use
/// `nanovdb::tools::cuda::indexToGrid`. The returned host-side grid handle has an `fvdb_jdata`
/// blind metadata entry recording the original tensor shape so the values can be loaded back with
/// the same shape they were saved with.
///
/// @tparam OutBuildT The destination NanoVDB build type (e.g. `float`, `nanovdb::Vec3f`, ...).
/// @tparam TorchScalarT The scalar type of the input JaggedTensor data
/// (e.g. `float`, `int32_t`, `c10::Half`).
template <typename OutBuildT, typename TorchScalarT>
nanovdb::GridHandle<nanovdb::HostBuffer>
fvdbToNanovdbGridWithValues(const GridBatchData &gridBatchData,
                            const JaggedTensor &data,
                            const std::vector<std::string> &names) {
    TORCH_CHECK(
        names.size() == 0 || names.size() == (size_t)gridBatchData.batchSize(),
        "Invalid parameter for names, must be empty or a list of the same length as the batch size. Got " +
            std::to_string(names.size()) + " names for batch size " +
            std::to_string(gridBatchData.batchSize()));

    // CPU-resident grids take a pure-host path that mirrors `indexToGrid` without any CUDA
    // context init, H2D upload, kernel launch, or D2H copy. This avoids regressing CPU-only
    // workloads (the GPU path otherwise pays full upload+download costs even when the input
    // never leaves the host).
    if (!gridBatchData.device().is_cuda()) {
        return fvdbToNanovdbGridWithValuesHost<OutBuildT, TorchScalarT>(gridBatchData, data, names);
    }

    using HostGridHandle   = nanovdb::GridHandle<nanovdb::HostBuffer>;
    using DeviceGridHandle = nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>;
    using ValueT           = typename nanovdb::BuildToValueMap<OutBuildT>::type;

    // Hoist tensor shape info out of the per-batch loop. The data tensor has shape
    // [totalVoxels, *tail]; per-batch, dim 0 is `numVoxelsAt(bi)` and the tail is identical.
    const int64_t rdim = data.rdim();
    std::vector<int64_t> tailSizes;
    tailSizes.reserve(rdim > 0 ? static_cast<size_t>(rdim - 1) : 0);
    for (int64_t di = 1; di < rdim; ++di) {
        tailSizes.push_back(data.rsize(di));
    }
    const uint64_t shapeBytes           = sizeof(int64_t) * (rdim + 1);
    const uint64_t paddedBlindDataBytes = nanovdb::math::AlignUp<32UL>(shapeBytes);
    const uint64_t blindOverhead        = sizeof(nanovdb::GridBlindMetaData) + paddedBlindDataBytes;

    // Make sure the data is CUDA-resident and contiguous, so its memory layout
    // matches the contiguous flat array of `BuildToValueMap<OutBuildT>::type` that indexToGrid
    // expects (e.g. (N,3) float -> N consecutive Vec3f via reinterpret-cast).
    JaggedTensor cudaData = data;
    if (!cudaData.is_cuda()) {
        cudaData = cudaData.cuda();
    }
    if (!cudaData.is_contiguous()) {
        cudaData = cudaData.contiguous();
    }

    // Determine the device pointer to the source index grid buffer. CPU-resident grids normally
    // return through the host path above; the upload branch is kept as a defensive fallback if
    // this helper is reused without that dispatch.
    nanovdb::cuda::DeviceBuffer tmpDevBuf; // empty unless we need to upload
    const torch::Device gridDevice = gridBatchData.device();
    const torch::Device cudaDevice = gridDevice.is_cuda()
                                         ? gridDevice
                                         : torch::Device(torch::kCUDA, c10::cuda::current_device());
    c10::cuda::CUDAGuard deviceGuard(cudaDevice);
    const at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(cudaDevice.index());

    const uint8_t *dSrcBufferStart = nullptr;
    if (gridDevice.is_cuda()) {
        dSrcBufferStart = gridBatchData.nanoGridHandle().buffer().deviceData();
    } else {
        const uint64_t srcBufferSize = gridBatchData.nanoGridHandle().buffer().size();
        const uint8_t *srcHostData =
            static_cast<const uint8_t *>(gridBatchData.nanoGridHandle().buffer().data());
        tmpDevBuf = nanovdb::cuda::DeviceBuffer(srcBufferSize, cudaDevice.index(), stream.stream());
        cudaCheck(cudaMemcpyAsync(tmpDevBuf.deviceData(),
                                  srcHostData,
                                  srcBufferSize,
                                  cudaMemcpyHostToDevice,
                                  stream.stream()));
        dSrcBufferStart = static_cast<const uint8_t *>(tmpDevBuf.deviceData());
    }

    const ValueT *dDataValuesBase = reinterpret_cast<const ValueT *>(cudaData.jdata().data_ptr());

    // indexToGrid reads `srcValues[srcLeaf.getValue(i)]` for every (active and inactive) voxel
    // slot in each leaf of the destination grid. For a ValueOnIndex source grid, getValue(i)
    // returns 0 for inactive voxels (background) and 1..N for active voxels, where N is the
    // grid's active voxel count. fvdb's user data is a flat array of length N (data[0] is the
    // first active voxel's value), so we cannot pass it directly: we need a per-batch buffer
    // whose [0] slot is a background zero and whose [1..N] slots match the user data slice.
    //
    // We therefore allocate a single (N+1)-sized device buffer per batch, memset its [0] slot
    // to zero, and D2D-copy the data slice into [1..N]. All allocations and copies are queued
    // on the same stream as the indexToGrid kernels so the GPU can run them back-to-back.

    std::vector<DeviceGridHandle> deviceHandles;
    std::vector<nanovdb::cuda::DeviceBuffer> perBatchValueBufs;
    std::vector<nanovdb::HostBuffer> hostBuffers;
    std::vector<uint64_t> origGridBytesPerBi;
    deviceHandles.reserve(gridBatchData.batchSize());
    perBatchValueBufs.reserve(gridBatchData.batchSize());
    hostBuffers.reserve(gridBatchData.batchSize());
    origGridBytesPerBi.reserve(gridBatchData.batchSize());

    for (int64_t bi = 0; bi < gridBatchData.batchSize(); ++bi) {
        const std::string name = names.size() > 0 ? names[bi] : "";
        TORCH_CHECK_VALUE(name.size() < nanovdb::GridData::MaxNameSize,
                          "Grid name " + name + " exceeds maximum character length of " +
                              std::to_string(nanovdb::GridData::MaxNameSize) + ".");

        const int64_t numVoxelsBi = gridBatchData.numVoxelsAt(bi);
        const int64_t cumVoxelsBi = gridBatchData.cumVoxelsAt(bi);
        TORCH_CHECK_VALUE(
            numVoxelsBi >= 0, "Invalid number of voxels at grid index ", bi, ": ", numVoxelsBi);

        const auto *dSrcGrid = reinterpret_cast<const nanovdb::NanoGrid<nanovdb::ValueOnIndex> *>(
            dSrcBufferStart + gridBatchData.cumBytesAt(bi));

        const uint64_t valueBufElems = static_cast<uint64_t>(numVoxelsBi) + 1u;
        nanovdb::cuda::DeviceBuffer valueBuf(
            valueBufElems * sizeof(ValueT), cudaDevice.index(), stream.stream());
        ValueT *dValuesBufBase = static_cast<ValueT *>(valueBuf.deviceData());
        cudaCheck(cudaMemsetAsync(dValuesBufBase, 0, sizeof(ValueT), stream.stream()));
        if (numVoxelsBi > 0) {
            cudaCheck(cudaMemcpyAsync(dValuesBufBase + 1,
                                      dDataValuesBase + cumVoxelsBi,
                                      numVoxelsBi * sizeof(ValueT),
                                      cudaMemcpyDeviceToDevice,
                                      stream.stream()));
        }

        DeviceGridHandle dh = nanovdb::tools::cuda::indexToGrid<OutBuildT>(
            dSrcGrid, dValuesBufBase, nanovdb::cuda::DeviceBuffer(), stream.stream());

        const uint64_t origGridBytes = dh.buffer().size();
        const uint64_t totalBytes    = origGridBytes + blindOverhead;
        origGridBytesPerBi.push_back(origGridBytes);
        hostBuffers.emplace_back(totalBytes);
        cudaCheck(cudaMemcpyAsync(hostBuffers.back().data(),
                                  dh.buffer().deviceData(),
                                  origGridBytes,
                                  cudaMemcpyDeviceToHost,
                                  stream.stream()));
        deviceHandles.emplace_back(std::move(dh));
        perBatchValueBufs.emplace_back(std::move(valueBuf));
    }

    // Wait for all kernels and D2H copies to land before patching host-side metadata.
    cudaCheck(cudaStreamSynchronize(stream.stream()));
    deviceHandles.clear();     // free GPU buffers as soon as possible
    perBatchValueBufs.clear(); // ditto

    // Second pass: patch host-side GridData (mGridSize, mMap, name) and append fvdb_jdata blind
    // data, then wrap the buffer in a HostBuffer-backed GridHandle.
    std::vector<HostGridHandle> buffers;
    buffers.reserve(gridBatchData.batchSize());
    for (int64_t bi = 0; bi < gridBatchData.batchSize(); ++bi) {
        const std::string name = names.size() > 0 ? names[bi] : "";
        std::vector<int64_t> shapeInts;
        shapeInts.reserve(static_cast<size_t>(rdim + 1));
        shapeInts.push_back(static_cast<int64_t>(rdim));
        shapeInts.push_back(gridBatchData.numVoxelsAt(bi));
        for (int64_t s: tailSizes) {
            shapeInts.push_back(s);
        }
        TORCH_CHECK(static_cast<int64_t>(shapeInts.size()) == rdim + 1,
                    "Internal error: blind data shape int count mismatch.");

        const uint64_t origGridBytes = origGridBytesPerBi[bi];
        const uint64_t totalBytes    = origGridBytes + blindOverhead;
        nanovdb::HostBuffer &outBuf  = hostBuffers[bi];
        patchGridWithBlindShape(static_cast<uint8_t *>(outBuf.data()),
                                origGridBytes,
                                totalBytes,
                                gridBatchData.voxelSizeAt(bi),
                                gridBatchData.voxelOriginAt(bi),
                                name,
                                shapeInts);

        buffers.emplace_back(std::move(outBuf));
    }

    if (buffers.size() == 1) {
        return std::move(buffers[0]);
    }
    return nanovdb::mergeGrids(buffers);
}

nanovdb::GridHandle<nanovdb::HostBuffer>
maybeConvertToStandardNanovdbGrid(const GridBatchData &gridBatchData,
                                  const JaggedTensor &data,
                                  const std::vector<std::string> &names) {
    // Get a squeezed view of the tensor so we can save data with singleton dimensions
    // (e.g. shape (N, 1, 3) can get saved as a Vec3f grid)
    torch::Tensor jdataSqueezed = data.jdata().squeeze();
    if (jdataSqueezed.numel() == 1 &&
        jdataSqueezed.dim() == 0) { // Make sure we have at least 1 dimension
        jdataSqueezed = jdataSqueezed.unsqueeze(0);
        TORCH_CHECK(jdataSqueezed.ndimension() == 1,
                    "Internal error: Invalid jdata shape when saving grid.");
    }
    // Note: c10::Half is intentionally NOT routed through the standard grid path. NanoVDB's
    // Fp16 leaf has a quantized layout and is rejected by the indexToGrid implementations.
    // Half-precision data is preserved on round-trip via `saveIndexGridWithBlindData`, which
    // stores the raw c10::Half bytes plus an `fvdb_jdataHalf` blind metadata tag.
    if (data.dtype() == torch::kFloat32) {
        if (jdataSqueezed.dim() == 1 || (jdataSqueezed.dim() == 2 && jdataSqueezed.size(1) == 1)) {
            using GridT = float;
            return fvdbToNanovdbGridWithValues<GridT, float>(gridBatchData, data, names);
        } else if (jdataSqueezed.dim() == 2 && jdataSqueezed.size(1) == 3) {
            using GridT = nanovdb::Vec3f;
            return fvdbToNanovdbGridWithValues<GridT, float>(gridBatchData, data, names);
        } else if (jdataSqueezed.dim() == 2 && jdataSqueezed.size(1) == 4) {
            using GridT = nanovdb::Vec4f;
            return fvdbToNanovdbGridWithValues<GridT, float>(gridBatchData, data, names);
        }
    } else if (data.dtype() == torch::kFloat64) {
        if (jdataSqueezed.dim() == 1 || (jdataSqueezed.dim() == 2 && jdataSqueezed.size(1) == 1)) {
            using GridT = double;
            return fvdbToNanovdbGridWithValues<GridT, double>(gridBatchData, data, names);
        } else if (jdataSqueezed.dim() == 2 && jdataSqueezed.size(1) == 3) {
            using GridT = nanovdb::Vec3d;
            return fvdbToNanovdbGridWithValues<GridT, double>(gridBatchData, data, names);
        } else if (jdataSqueezed.dim() == 2 && jdataSqueezed.size(1) == 4) {
            using GridT = nanovdb::Vec4d;
            return fvdbToNanovdbGridWithValues<GridT, double>(gridBatchData, data, names);
        }
    } else if (data.dtype() == torch::kInt32) {
        if (jdataSqueezed.dim() == 1 || (jdataSqueezed.dim() == 2 && jdataSqueezed.size(1) == 1)) {
            using GridT = int32_t;
            return fvdbToNanovdbGridWithValues<GridT, int32_t>(gridBatchData, data, names);
        } else if (jdataSqueezed.dim() == 2 && jdataSqueezed.size(1) == 3) {
            using GridT = nanovdb::Vec3i;
            return fvdbToNanovdbGridWithValues<GridT, int32_t>(gridBatchData, data, names);
        }
    } else if (data.dtype() == torch::kInt64) {
        if (jdataSqueezed.dim() == 1 || (jdataSqueezed.dim() == 2 && jdataSqueezed.size(1) == 1)) {
            using GridT = int64_t;
            return fvdbToNanovdbGridWithValues<GridT, int64_t>(gridBatchData, data, names);
        }
    } else if (data.dtype() == torch::kUInt8) {
        if (jdataSqueezed.dim() == 2 && jdataSqueezed.size(1) == 4) {
            using GridT = nanovdb::math::Rgba8;
            return fvdbToNanovdbGridWithValues<GridT, uint8_t>(gridBatchData, data, names);
        }
    }

    return nanovdb::GridHandle<nanovdb::HostBuffer>();
}

bool
maybeSaveStandardNanovdbGrid(const std::string &path,
                             const GridBatchData &gridBatchData,
                             const JaggedTensor &data,
                             const std::vector<std::string> &names,
                             nanovdb::io::Codec codec,
                             bool verbose) {
    nanovdb::GridHandle<nanovdb::HostBuffer> gridHandle =
        maybeConvertToStandardNanovdbGrid(gridBatchData, data, names);
    if (gridHandle.isEmpty()) {
        return false;
    }

    nanovdb::io::writeGrid(path, gridHandle, codec, verbose);
    return true;
}

nanovdb::GridHandle<nanovdb::HostBuffer>
getIndexGrid(const GridBatchData &gridBatchData, const std::vector<std::string> &names = {}) {
    const nanovdb::GridHandle<TorchDeviceBuffer> &nanoGridHdl = gridBatchData.nanoGridHandle();

    // Allocate memory and get pointer to host grid buffer
    nanovdb::HostBuffer writeBuf(nanoGridHdl.buffer().size());
    void *writeHead = writeBuf.data();

    // Get pointer to grid read from (possibly on the device)
    const bool isCuda = nanoGridHdl.buffer().device().is_cuda();
    void *readHead    = isCuda ? nanoGridHdl.buffer().deviceData() : nanoGridHdl.buffer().data();
    const size_t sourceGridByteSize = nanoGridHdl.buffer().size();

    if (isCuda) {
        c10::cuda::CUDAGuard deviceGuard(gridBatchData.device());
        cudaCheck(cudaMemcpy(writeHead, readHead, sourceGridByteSize, cudaMemcpyDeviceToHost));
    } else {
        std::memcpy(writeHead, readHead, sourceGridByteSize);
    }

    nanovdb::GridHandle<nanovdb::HostBuffer> retHandle =
        nanovdb::GridHandle<nanovdb::HostBuffer>(std::move(writeBuf));

    // Write voxelSize and origin information to the output buffer
    for (int64_t bi = 0; bi < gridBatchData.batchSize(); bi += 1) {
        nanovdb::GridData *retGridData = (nanovdb::GridData *)(retHandle.gridData(bi));
        const nanovdb::Vec3d &vs       = gridBatchData.voxelSizeAt(bi);
        const nanovdb::Vec3d vo        = gridBatchData.voxelOriginAt(bi);
        retGridData->mVoxelSize        = {vs[0], vs[1], vs[2]};
        retGridData->mMap              = nanovdb::Map(vs[0], {vo[0], vo[1], vo[2]});
    }

    // If you passed in grid names, write them to the output buffer
    if (names.size() > 0) {
        for (int64_t bi = 0; bi < gridBatchData.batchSize(); bi += 1) {
            const std::string name = names.size() > 0 ? names[bi] : "";
            TORCH_CHECK_VALUE(name.size() < nanovdb::GridData::MaxNameSize,
                              "Grid name " + name + " exceeds maximum character length of " +
                                  std::to_string(nanovdb::GridData::MaxNameSize) + ".");
            nanovdb::GridData *retGridData = (nanovdb::GridData *)(retHandle.gridData(bi));
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstringop-truncation"
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#pragma GCC diagnostic ignored "-Warray-bounds"
            strncpy(retGridData->mGridName, names[bi].c_str(), nanovdb::GridData::MaxNameSize);
#pragma GCC diagnostic pop
        }
    }

    // Return the copied grid handle.
    return retHandle;
}

void
saveIndexGrid(const std::string &path,
              const GridBatchData &gridBatchData,
              const std::vector<std::string> &names,
              nanovdb::io::Codec codec,
              bool verbose) {
    // If you don't pass in data, then we just write the grid
    nanovdb::GridHandle<nanovdb::HostBuffer> writeHandle = getIndexGrid(gridBatchData, names);

    // Save the grid to disk
    nanovdb::io::writeGrid(path, writeHandle, codec, verbose);
}

void
saveIndexGridWithBlindData(const std::string &path,
                           const GridBatchData &gridBatchData,
                           const JaggedTensor &data,
                           const std::vector<std::string> &names,
                           nanovdb::io::Codec codec,
                           bool verbose) {
    // Make a CPU-resident contiguous copy of the data jagged tensor when needed.
    JaggedTensor cpuData = data.cpu().contiguous();
    TORCH_CHECK(cpuData.is_contiguous(), "Jagged tensor must be contiguous");

    // Hoist tensor shape info: rank and tail (everything past dim 0) are constant across batch
    // entries. Per-batch dim 0 is `numVoxelsAt(batchIdx)`.
    const int64_t rank = cpuData.rdim();
    std::vector<int64_t> tailSizes;
    tailSizes.reserve(rank > 0 ? static_cast<size_t>(rank - 1) : 0);
    for (int64_t dimIdx = 1; dimIdx < rank; ++dimIdx) {
        tailSizes.push_back(cpuData.rsize(dimIdx));
    }
    const int64_t elementSize = cpuData.jdata().element_size();
    int64_t tailElems         = 1;
    for (int64_t tailSize: tailSizes) {
        tailElems *= tailSize;
    }

    // Blind payload per grid: [rank+1 shape int64s][jdata bytes], padded to 32B.
    std::vector<uint64_t> paddedPayloadBytes;
    paddedPayloadBytes.reserve(gridBatchData.batchSize());
    for (int64_t batchIdx = 0; batchIdx < gridBatchData.batchSize(); ++batchIdx) {
        const int64_t jdataBytes     = gridBatchData.numVoxelsAt(batchIdx) * tailElems * elementSize;
        const uint64_t blindDataSize = jdataBytes + sizeof(int64_t) * (rank + 1);
        paddedPayloadBytes.push_back(nanovdb::math::AlignUp<32UL>(blindDataSize));
    }

    const std::string fvdbBlindName = "fvdb_jdata" + TorchScalarTypeToStr(cpuData.scalar_type());
    const uint8_t *jdataBase        = static_cast<const uint8_t *>(cpuData.jdata().data_ptr());

    nanovdb::GridHandle<nanovdb::HostBuffer> writeHandle = assembleOnIndexBlindBuffer(
        gridBatchData, names, paddedPayloadBytes,
        [&](int64_t batchIdx, nanovdb::GridBlindMetaData *blindMeta, uint8_t *payload) {
            const int64_t numVoxels = gridBatchData.numVoxelsAt(batchIdx);

            blindMeta->mDataOffset = int64_t(sizeof(nanovdb::GridBlindMetaData));
            blindMeta->mValueCount = paddedPayloadBytes[batchIdx]; // Number of bytes
            blindMeta->mValueSize  = 1;                            // 1 byte per value
            blindMeta->mSemantic   = nanovdb::GridBlindDataSemantic::Unknown;
            blindMeta->mDataClass  = nanovdb::GridBlindDataClass::Unknown;
            blindMeta->mDataType   = nanovdb::GridType::Unknown;
            setFixedSizeStringBuf(blindMeta->mName, nanovdb::GridBlindMetaData::MaxNameSize,
                                  fvdbBlindName, "blind metadata name");

            // Shape prefix so the tensor can be reloaded with its original shape. The hoisted
            // tail sizes are reused; only dim 0 (numVoxels) varies per batch.
            int64_t *shapeHead = reinterpret_cast<int64_t *>(payload);
            *shapeHead++       = static_cast<int64_t>(rank);
            *shapeHead++       = numVoxels;
            for (int64_t tailSize: tailSizes) {
                *shapeHead++ = tailSize;
            }

            // jdata bytes for this grid, sliced via the cumulative voxel offset.
            const int64_t jdataSize = numVoxels * tailElems * elementSize;
            const uint8_t *srcBytes =
                jdataBase + gridBatchData.cumVoxelsAt(batchIdx) * tailElems * elementSize;
            std::memcpy(reinterpret_cast<uint8_t *>(shapeHead), srcBytes, jdataSize);
        });

    // Write the grid to disk
    nanovdb::io::writeGrid(path, writeHandle, codec, verbose);
}

nanovdb::GridHandle<nanovdb::HostBuffer>
toNVDB(const GridBatchData &gridBatchData,
       const std::optional<JaggedTensor> &maybeData,
       const std::vector<std::string> &names) {
    // Validate optional names.
    if (!names.empty()) {
        TORCH_CHECK_VALUE(
            names.size() == (size_t)gridBatchData.batchSize(),
            "Invalid parameter for names, must be empty or a list of the same length as the batch size. Got " +
                std::to_string(names.size()) + " names for batch size " +
                std::to_string(gridBatchData.batchSize()));
    }

    if (maybeData.has_value()) {
        const JaggedTensor &data = maybeData.value();
        TORCH_CHECK_VALUE(data.jdata().ndimension() >= 1,
                          "Invalid jagged data shape in to_nanovdb");
        TORCH_CHECK_VALUE(gridBatchData.totalVoxels() == data.jdata().size(0),
                          "Invalid jagged data shape in to_nanovdb. Must match number of voxels");
        TORCH_CHECK_VALUE(gridBatchData.device() == data.device(),
                          "Device should match between grid batch and data");
        checkPerBatchVoxelCounts(gridBatchData, data);
        return maybeConvertToStandardNanovdbGrid(gridBatchData, data, names);
    } else {
        return getIndexGrid(gridBatchData, names);
    }
}

void
saveNVDB(const std::string &path,
         const GridBatchData &gridBatchData,
         const std::optional<JaggedTensor> &maybeData,
         const std::vector<std::string> &names,
         bool compressed,
         bool verbose) {
    // Which Codec to use for saving
    nanovdb::io::Codec codec = compressed ? nanovdb::io::Codec::BLOSC : nanovdb::io::Codec::NONE;

    // Validate optional names.
    if (!names.empty()) {
        TORCH_CHECK_VALUE(
            names.size() == (size_t)gridBatchData.batchSize(),
            "Invalid parameter for names, must be empty or a list of the same length as the batch size. Got " +
                std::to_string(names.size()) + " names for batch size " +
                std::to_string(gridBatchData.batchSize()));
    }

    if (!maybeData.has_value()) {
        saveIndexGrid(path, gridBatchData, names, codec, verbose);
        return;
    }
    const JaggedTensor &data = maybeData.value();

    TORCH_CHECK_VALUE(data.jdata().ndimension() >= 1, "Invalid jagged data shape in save_nvdb");
    TORCH_CHECK_VALUE(gridBatchData.totalVoxels() == data.jdata().size(0),
                      "Invalid jagged data shape in save_nvdb. Must match number of voxels");
    TORCH_CHECK_VALUE(gridBatchData.device() == data.device(),
                      "Device should match between grid batch and data");
    checkPerBatchVoxelCounts(gridBatchData, data);

    // Heuristically determine if we can use a standard NanoVDB grid (e.g. Vec3f, float, Vec3i,
    // etc.) to store the data. If so, save such a grid; otherwise save an index grid with custom
    // blind data.
    if (maybeSaveStandardNanovdbGrid(path, gridBatchData, data, names, codec, verbose)) {
        return;
    } else {
        // If we didn't manage to save a standard NanoVDB grid, just save a tensor grid with blind
        // data
        saveIndexGridWithBlindData(path, gridBatchData, data, names, codec, verbose);
    }
}

nanovdb::GridHandle<nanovdb::HostBuffer>
toNVDBWithBlindFloat(const GridBatchData &gridBatchData, const JaggedTensor &floatValues) {
    TORCH_CHECK_VALUE(floatValues.jdata().scalar_type() == torch::kFloat32,
                      "toNVDBWithBlindFloat: floatValues must be float32, got ",
                      floatValues.jdata().scalar_type());
    TORCH_CHECK_VALUE(floatValues.jdata().ndimension() == 1,
                      "toNVDBWithBlindFloat: floatValues must be 1D (one float per voxel), got ",
                      floatValues.jdata().ndimension(), "D");
    TORCH_CHECK_VALUE(gridBatchData.totalVoxels() == floatValues.jdata().size(0),
                      "toNVDBWithBlindFloat: floatValues length ", floatValues.jdata().size(0),
                      " must match total voxel count ", gridBatchData.totalVoxels());
    checkPerBatchVoxelCounts(gridBatchData, floatValues);

    // Move float values to CPU (contiguous) for the host-side memcopy below.
    JaggedTensor cpuFloats = floatValues.cpu().contiguous();
    TORCH_CHECK(cpuFloats.is_contiguous(), "Internal error: float values must be contiguous");

    // Blind payload per grid: raw float32 values (one per voxel), padded to 32B.
    std::vector<uint64_t> paddedPayloadBytes;
    paddedPayloadBytes.reserve(gridBatchData.batchSize());
    for (int64_t batchIdx = 0; batchIdx < gridBatchData.batchSize(); ++batchIdx) {
        paddedPayloadBytes.push_back(
            nanovdb::math::AlignUp<32UL>(gridBatchData.numVoxelsAt(batchIdx) * sizeof(float)));
    }

    const float *floatBase = static_cast<const float *>(cpuFloats.jdata().data_ptr());

    return assembleOnIndexBlindBuffer(
        gridBatchData, /*names=*/{}, paddedPayloadBytes,
        [&](int64_t batchIdx, nanovdb::GridBlindMetaData *blindMeta, uint8_t *payload) {
            const int64_t numVoxels = gridBatchData.numVoxelsAt(batchIdx);

            // float32 values at data_offset from this header. The nanovdb-editor surface shader
            // reads: val_addr + val_index * 4, interpreting the result as a float.
            blindMeta->mDataOffset = int64_t(sizeof(nanovdb::GridBlindMetaData));
            blindMeta->mValueCount = numVoxels;
            blindMeta->mValueSize  = sizeof(float);
            blindMeta->mSemantic   = nanovdb::GridBlindDataSemantic::Unknown;
            blindMeta->mDataClass  = nanovdb::GridBlindDataClass::Unknown;
            blindMeta->mDataType   = nanovdb::GridType::Float;
            setFixedSizeStringBuf(blindMeta->mName, nanovdb::GridBlindMetaData::MaxNameSize,
                                  "fvdb_sdf_float32", "blind metadata name");

            // Raw float values in active-voxel order (val_index == flat active index).
            const float *srcFloats = floatBase + gridBatchData.cumVoxelsAt(batchIdx);
            std::memcpy(payload, srcFloats, numVoxels * sizeof(float));
        });
}

} // namespace io
} // namespace detail
} // namespace fvdb
