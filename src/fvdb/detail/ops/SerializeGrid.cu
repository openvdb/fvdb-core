// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/GridBatchDataFactory.h>
#include <fvdb/detail/ops/MakeContiguous.h>
#include <fvdb/detail/ops/SerializeGrid.h>

#include <nanovdb/HostBuffer.h>

#include <optional>

namespace fvdb {
namespace detail {
namespace ops {

namespace {

struct V01Header {
    uint64_t magic   = 0x0F0F0F0F0F0F0F0F;
    uint64_t version = 0;
    uint64_t numGrids;
    uint64_t totalBytes;
};

} // namespace

torch::Tensor
serializeGrid(const GridBatchData &grid) {
    c10::DeviceGuard guard(grid.device());

    // The bytes to write are the batch's logical grids on the host. A contiguous CPU batch has
    // them already; a contiguous device batch is one copy of its storage; anything else is
    // compacted straight onto the host in one pass. The per-grid metadata written below is the
    // batch's own: only its voxel sizes and origins are read back, and deserialization recomputes
    // the rest from the grids.
    std::optional<GridStorage> compacted;
    if (grid.batchSize() > 0 && !(grid.device().is_cpu() && grid.isContiguous())) {
        compacted = grid.isContiguous() ? grid.gridStorage().to(torch::kCPU)
                                        : contiguousGridStorage(grid, torch::kCPU);
    }
    const GridStorage &storage = compacted ? *compacted : grid.gridStorage();
    const int64_t numGrids     = grid.batchSize();
    // An empty batch's storage holds a sentinel grid with no metadata record behind it; it
    // serializes as zero grids and zero grid bytes.
    const int64_t hdlBufSize = static_cast<int64_t>(grid.totalBytes());
    const int64_t headerSize = sizeof(V01Header) + numGrids * sizeof(GridBatchData::GridMetadata) +
                               sizeof(GridBatchData::GridBatchMetadata);
    const int64_t totalByteSize = headerSize + hdlBufSize;

    V01Header header;
    header.totalBytes = totalByteSize;
    header.numGrids   = numGrids;

    torch::Tensor ret = torch::empty({totalByteSize}, torch::kInt8);
    int8_t *retPtr    = ret.data_ptr<int8_t>();

    memcpy(retPtr, &header, sizeof(V01Header));
    retPtr += sizeof(V01Header);

    memcpy(retPtr, &grid.mBatchMetadata, sizeof(GridBatchData::GridBatchMetadata));
    retPtr += sizeof(GridBatchData::GridBatchMetadata);

    if (numGrids > 0) {
        memcpy(retPtr, grid.mHostGridMetadata, numGrids * sizeof(GridBatchData::GridMetadata));
        retPtr += numGrids * sizeof(GridBatchData::GridMetadata);
        memcpy(retPtr, storage.hostBytes(), hdlBufSize);
    }
    retPtr += hdlBufSize;

    TORCH_CHECK(retPtr == (ret.data_ptr<int8_t>() + totalByteSize),
                "Something went wrong with serialization");

    return ret;
}

c10::intrusive_ptr<GridBatchData>
deserializeGrid(const torch::Tensor &serialized) {
    c10::DeviceGuard guard(serialized.device());

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

    const GridBatchData::GridBatchMetadata *batchMetadata =
        reinterpret_cast<const GridBatchData::GridBatchMetadata *>(serializedPtr +
                                                                   sizeof(V01Header));
    TORCH_CHECK(batchMetadata->version == GridBatchData::GridBatchMetadata::kVersion,
                "Serialized data is not a valid grid handle. Bad batch metadata version.");

    const GridBatchData::GridMetadata *gridMetadata =
        reinterpret_cast<const GridBatchData::GridMetadata *>(
            serializedPtr + sizeof(V01Header) + sizeof(GridBatchData::GridBatchMetadata));
    for (uint64_t i = 0; i < numGrids; i += 1) {
        TORCH_CHECK(gridMetadata[i].version == GridBatchData::GridMetadata::kVersion,
                    "Serialized data is not a valid grid handle. Bad grid metadata version.");
    }
    const int8_t *gridBuffer = serializedPtr + sizeof(V01Header) +
                               sizeof(GridBatchData::GridBatchMetadata) +
                               numGrids * sizeof(GridBatchData::GridMetadata);

    const uint64_t sizeofMetadata = sizeof(V01Header) + sizeof(GridBatchData::GridBatchMetadata) +
                                    numGrids * sizeof(GridBatchData::GridMetadata);
    const uint64_t sizeofGrid = header->totalBytes - sizeofMetadata;
    if (numGrids == 0) {
        TORCH_CHECK(sizeofGrid == 0,
                    "Serialized data is not a valid grid handle. Empty batch with grid bytes.");
        return makeEmptyGridBatchData(torch::kCPU);
    }

    nanovdb::HostBuffer buf(sizeofGrid);
    memcpy(buf.data(), gridBuffer, sizeofGrid);

    // Parsing the chain validates what was read back before anything trusts it.
    GridStorage storage(GridStorage::HostHandle(std::move(buf)));

    std::vector<nanovdb::Vec3d> voxelSizes, voxelOrigins;
    voxelSizes.reserve(numGrids);
    voxelOrigins.reserve(numGrids);
    for (uint64_t i = 0; i < numGrids; i += 1) {
        voxelSizes.emplace_back(gridMetadata[i].mVoxelSize);
        voxelOrigins.emplace_back(gridMetadata[i].voxelOrigin());
    }

    return makeGridBatchData(std::move(storage), voxelSizes, voxelOrigins);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
