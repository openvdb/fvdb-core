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
/// @param origGridBytes Size of the grid portion (output of indexToGrid).
/// @param totalBytes Total bytes including blind data.
/// @param voxelSize World-space voxel size to write into mMap / mVoxelSize.
/// @param voxelOrigin World-space voxel origin to write into mMap.
/// @param name Grid name (truncated/checked against MaxNameSize).
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

/// @brief Host-only port of `nanovdb::tools::cuda::indexToGrid` for a single
/// `nanovdb::NanoGrid<nanovdb::ValueOnIndex>`.
///
/// Mirrors the layout and per-node operations of `IndexToGrid.cuh`: copies the grid/tree/root
/// headers, fixes child offsets where the destination node sizes differ from the source, and
/// fills every internal-node value slot and leaf voxel slot via `srcValues[srcIndex]`. The
/// kernel's per-thread/per-block work becomes a sequential loop on the host, so this avoids
/// any CUDA context, allocation, or memcpy when the source grid is already CPU-resident.
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

/// @brief Host-only build path for `fvdbToNanovdbGridWithValues`. Used when both the source grid
/// and the user data are CPU-resident: avoids all CUDA context init, H2D upload, kernel launch,
/// and D2H copy that the GPU path incurs.
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
/// an existing fvdb ValueOnIndex grid batch and a JaggedTensor of per-voxel values, using the
/// device-side `nanovdb::tools::cuda::indexToGrid` kernel. The returned host-side grid handle has
/// an `fvdb_jdata` blind metadata entry recording the original tensor shape so the values can be
/// loaded back with the same shape they were saved with.
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

    // Make sure the data is on the current CUDA device and contiguous, so its memory layout
    // matches the contiguous flat array of `BuildToValueMap<OutBuildT>::type` that indexToGrid
    // expects (e.g. (N,3) float -> N consecutive Vec3f via reinterpret-cast).
    JaggedTensor cudaData = data;
    if (!cudaData.is_cuda()) {
        cudaData = cudaData.cuda();
    }
    if (!cudaData.is_contiguous()) {
        cudaData = cudaData.contiguous();
    }

    // Determine the device pointer to the source index grid buffer. If the grid batch is on CPU,
    // upload the entire grid bytes to a temporary device buffer once (rather than per-batch) and
    // use that.
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
    // Get a squeezed view of the tensor so we can save data with redundant dimensions
    // (e.g. shape (N, 1, 3) can get saved as a Vec3f grid)
    torch::Tensor jdataSqueezed = data.jdata().squeeze();
    if (jdataSqueezed.numel() == 1 &&
        jdataSqueezed.dim() == 0) { // Make sure we have at least 1 dimension
        jdataSqueezed = jdataSqueezed.unsqueeze(0);
        TORCH_CHECK(jdataSqueezed.ndimension() == 1,
                    "Internal error: Invalid jdata shape when saving grid.");
    }
    // Note: c10::Half is intentionally NOT routed through the standard grid path. NanoVDB's
    // device-side `indexToGrid<Fp16>` does not compile (the Fp16 leaf has a quantized layout
    // and is rejected by the kernel's static_assert). Half-precision data is preserved on
    // round-trip via `saveIndexGridWithBlindData`, which stores the raw c10::Half bytes plus
    // an `fvdb_jdataHalf` blind metadata tag.
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

    // Issue #6: collapse the async-then-immediate-sync pattern. The previous code launched a
    // single cudaMemcpyAsync and then immediately synchronized, which is equivalent to a plain
    // cudaMemcpy with extra noise. Use a synchronous copy under the device guard for clarity.
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

    // Build a grid handle
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
    const nanovdb::GridHandle<TorchDeviceBuffer> &nanoGridHdl = gridBatchData.nanoGridHandle();

    // Make a (possible) cpu copy of the data jagged tensor
    JaggedTensor cpuData = data.cpu().contiguous();
    TORCH_CHECK(cpuData.is_contiguous(), "Jagged tensor must be contiguous");

    // Hoist tensor shape info (issue #4): rdim and tail (everything past dim 0) are constant
    // across batch entries. Per-batch dim 0 is `numVoxelsAt(bi)`.
    const int64_t rdim = cpuData.rdim();
    std::vector<int64_t> tailSizes;
    tailSizes.reserve(rdim > 0 ? static_cast<size_t>(rdim - 1) : 0);
    for (int64_t di = 1; di < rdim; ++di) {
        tailSizes.push_back(cpuData.rsize(di));
    }
    const int64_t elementSize = cpuData.jdata().element_size();
    int64_t tailElems         = 1;
    for (int64_t s: tailSizes) {
        tailElems *= s;
    }

    // Compute blind data sizes padded to be 32 byte aligned
    std::vector<uint64_t> blindDataPadding;     // Size of each blind data padded to 32 bytes
    std::vector<uint64_t> paddedBlindDataSizes; // The amount of padding added to each blind data to
                                                // achieve 32 byte alignment
    blindDataPadding.reserve(gridBatchData.batchSize());
    paddedBlindDataSizes.reserve(gridBatchData.batchSize());
    uint64_t totalBlindDataSize = 0;
    for (int64_t bi = 0; bi < gridBatchData.batchSize(); bi += 1) {
        const int64_t numVoxelsBi            = gridBatchData.numVoxelsAt(bi);
        const int64_t jdataBytesBi           = numVoxelsBi * tailElems * elementSize;
        const uint64_t blindDataSizeBi       = jdataBytesBi + sizeof(int64_t) * (rdim + 1);
        const uint64_t paddedBlindDataSizeBi = nanovdb::math::AlignUp<32UL>(blindDataSizeBi);
        blindDataPadding.push_back(paddedBlindDataSizeBi - blindDataSizeBi);
        paddedBlindDataSizes.push_back(paddedBlindDataSizeBi);
        totalBlindDataSize += paddedBlindDataSizeBi;
    }

    // Allocate a big enough buffer to allocate the index grid and blind data
    const size_t allocSize = nanoGridHdl.buffer().size() +   // Grids (32B aligned)
                             sizeof(nanovdb::GridBlindMetaData) *
                                 gridBatchData.batchSize() + // Blind metadata (32B aligned)
                             totalBlindDataSize;             // Blind data (32B aligned)
    nanovdb::HostBuffer writeBuf(allocSize);

    // Get pointer to read (possibly on the device) and write pointers
    const bool isCuda  = nanoGridHdl.buffer().device().is_cuda();
    uint8_t *writeHead = static_cast<uint8_t *>(writeBuf.data());
    uint8_t *readHead  = static_cast<uint8_t *>(isCuda ? nanoGridHdl.buffer().deviceData()
                                                      : nanoGridHdl.buffer().data());

    // Pointer to the start of jdata for slicing per-batch values without allocating fresh
    // JaggedTensor objects each iteration.
    const uint8_t *jdataBase = static_cast<const uint8_t *>(cpuData.jdata().data_ptr());

    // Copy each grid and each entry in the jagged tensor
    for (int64_t bi = 0; bi < gridBatchData.batchSize(); bi += 1) {
        // Copy the full bi^th index grid to the buffer
        const size_t sourceGridByteSize = nanoGridHdl.gridSize(bi);
        if (isCuda) {
            c10::cuda::CUDAGuard deviceGuard(gridBatchData.device());
            at::cuda::CUDAStream defaultStream =
                at::cuda::getCurrentCUDAStream(gridBatchData.device().index());
            cudaMemcpyAsync((void *)writeHead,
                            (void *)readHead,
                            sourceGridByteSize,
                            cudaMemcpyDeviceToHost,
                            defaultStream.stream());
        } else {
            memcpy((void *)writeHead, (void *)readHead, sourceGridByteSize);
        }
        // Update the metadata for the copied grid in the buffer to be a tensor grid with blind data
        nanovdb::GridData *writeGridData    = reinterpret_cast<nanovdb::GridData *>(writeHead);
        writeGridData->mGridClass           = nanovdb::GridClass::TensorGrid;
        writeGridData->mGridType            = nanovdb::GridType::OnIndex;
        writeGridData->mBlindMetadataCount  = 1;
        writeGridData->mBlindMetadataOffset = sourceGridByteSize;
        const std::string name              = names.size() > 0 ? names[bi] : "";
        setFixedSizeStringBuf(
            writeGridData->mGridName, nanovdb::GridData::MaxNameSize, name, "Grid name " + name);
        writeGridData->mGridSize =
            sourceGridByteSize + sizeof(nanovdb::GridBlindMetaData) + paddedBlindDataSizes[bi];

        // Write voxelSize and origin
        const nanovdb::Vec3d &vs  = gridBatchData.voxelSizeAt(bi);
        const nanovdb::Vec3d vo   = gridBatchData.voxelOriginAt(bi);
        const double sx           = vs[0];
        const double sy           = vs[1];
        const double sz           = vs[2];
        writeGridData->mVoxelSize = {sx, sy, sz};
        const double mat[3][3]    = {{sx, 0.0, 0.0},        // row 0
                                     {0.0, sy, 0.0},        // row 1
                                     {0.0, 0.0, sz}};       // row 2
        const double invMat[3][3] = {{1.0 / sx, 0.0, 0.0},  // row 0
                                     {0.0, 1.0 / sy, 0.0},  // row 1
                                     {0.0, 0.0, 1.0 / sz}}; // row 2
        nanovdb::Vec3d trans      = {vo[0], vo[1], vo[2]};
        writeGridData->mMap.set(mat, invMat, trans);

        readHead += sourceGridByteSize;
        writeHead += sourceGridByteSize;

        // Write out blind metadata to the end of the grid
        nanovdb::GridBlindMetaData *blindMetadata =
            reinterpret_cast<nanovdb::GridBlindMetaData *>(writeHead);
        blindMetadata->mDataOffset = int64_t(sizeof(nanovdb::GridBlindMetaData));
        blindMetadata->mValueCount = paddedBlindDataSizes[bi]; // Number of bytes
        blindMetadata->mValueSize  = 1;                        // 1 byte per value
        blindMetadata->mSemantic   = nanovdb::GridBlindDataSemantic::Unknown;
        blindMetadata->mDataClass  = nanovdb::GridBlindDataClass::Unknown;
        blindMetadata->mDataType   = nanovdb::GridType::Unknown;
        const std::string fvdbBlindName =
            "fvdb_jdata" + TorchScalarTypeToStr(cpuData.scalar_type());
        setFixedSizeStringBuf(blindMetadata->mName,
                              nanovdb::GridBlindMetaData::MaxNameSize,
                              fvdbBlindName,
                              "blind metadata name");
        TORCH_CHECK(blindMetadata->isValid(), "Invalid blind metadata");
        writeHead += sizeof(nanovdb::GridBlindMetaData);

        // Write the shape of the bi^th jdata tensor so we can load it with the same shape it was
        // saved with. The hoisted tail sizes are reused; only dim 0 (numVoxelsBi) varies per batch.
        const int64_t numVoxelsBi               = gridBatchData.numVoxelsAt(bi);
        *reinterpret_cast<int64_t *>(writeHead) = static_cast<int64_t>(rdim);
        writeHead += sizeof(int64_t);
        *reinterpret_cast<int64_t *>(writeHead) = numVoxelsBi;
        writeHead += sizeof(int64_t);
        for (int64_t s: tailSizes) {
            *reinterpret_cast<int64_t *>(writeHead) = s;
            writeHead += sizeof(int64_t);
        }

        // Copy the bi^th jdata tensor as blind data to the buffer
        const int64_t jdataSize = numVoxelsBi * tailElems * elementSize;
        const uint8_t *srcBytes =
            jdataBase + gridBatchData.cumVoxelsAt(bi) * tailElems * elementSize;
        memcpy((void *)writeHead, (const void *)srcBytes, jdataSize);
        writeHead += jdataSize;
        writeHead += blindDataPadding[bi]; // Add padding to make sure we're 32 byte aligned
    }

    // Synchronize cuda stream if we just did a bunch of GPU -> CPU transfers
    if (isCuda) {
        at::cuda::CUDAStream defaultStream =
            at::cuda::getCurrentCUDAStream(gridBatchData.device().index());
        cudaStreamSynchronize(defaultStream.stream());
    }

    // Write the grid to disk
    nanovdb::GridHandle<nanovdb::HostBuffer> writeHandle(std::move(writeBuf));
    nanovdb::io::writeGrid(path, writeHandle, codec, verbose);
}

nanovdb::GridHandle<nanovdb::HostBuffer>
toNVDB(const GridBatchData &gridBatchData,
       const std::optional<JaggedTensor> &maybeData,
       const std::vector<std::string> &names) {
    // Get optional names
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

    // Get optional names
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

    // Heuristically determine if we can use a standard nanovdb grid (e.g. vec3f, float, vec3i,
    // etc...) to store the data If so, we save such a grid -- otherwise we save an index grid with
    // custom blind data
    if (maybeSaveStandardNanovdbGrid(path, gridBatchData, data, names, codec, verbose)) {
        return;
    } else {
        // If we didn't manage to save a standard nanovdb grid, just save a tensor grid with blind
        // data
        saveIndexGridWithBlindData(path, gridBatchData, data, names, codec, verbose);
    }
}

} // namespace io
} // namespace detail
} // namespace fvdb
