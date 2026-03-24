// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/GridBatchDataFactory.h>
#include <fvdb/detail/ops/CloneGrid.h>
#include <fvdb/detail/ops/SerializeGrid.h>

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

    const GridBatchData *self = &grid;
    c10::intrusive_ptr<GridBatchData> cpuClone;
    if (!grid.device().is_cpu()) {
        cpuClone = cloneGrid(grid, torch::kCPU, true);
        self     = cpuClone.get();
    }

    int64_t numGrids   = self->nanoGridHandle().gridCount();
    int64_t hdlBufSize = self->nanoGridHandle().buffer().size();

    const int64_t headerSize =
        sizeof(V01Header) + numGrids * sizeof(GridBatchData::GridMetadata) +
        sizeof(GridBatchData::GridBatchMetadata);
    const int64_t totalByteSize = headerSize + hdlBufSize;

    V01Header header;
    header.totalBytes = totalByteSize;
    header.numGrids   = numGrids;

    torch::Tensor ret = torch::empty({totalByteSize}, torch::kInt8);
    int8_t *retPtr    = ret.data_ptr<int8_t>();

    memcpy(retPtr, &header, sizeof(V01Header));
    retPtr += sizeof(V01Header);

    memcpy(retPtr, &self->mBatchMetadata, sizeof(GridBatchData::GridBatchMetadata));
    retPtr += sizeof(GridBatchData::GridBatchMetadata);

    memcpy(retPtr, self->mHostGridMetadata, numGrids * sizeof(GridBatchData::GridMetadata));
    retPtr += numGrids * sizeof(GridBatchData::GridMetadata);

    memcpy(retPtr, self->nanoGridHandle().buffer().data(), hdlBufSize);
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
        reinterpret_cast<const GridBatchData::GridBatchMetadata *>(
            serializedPtr + sizeof(V01Header));
    TORCH_CHECK(batchMetadata->version == 1,
                "Serialized data is not a valid grid handle. Bad batch metadata version.");

    const GridBatchData::GridMetadata *gridMetadata =
        reinterpret_cast<const GridBatchData::GridMetadata *>(
            serializedPtr + sizeof(V01Header) + sizeof(GridBatchData::GridBatchMetadata));
    for (uint64_t i = 0; i < numGrids; i += 1) {
        TORCH_CHECK(gridMetadata[i].version == 1,
                    "Serialized data is not a valid grid handle. Bad grid metadata version.");
    }
    const int8_t *gridBuffer = serializedPtr + sizeof(V01Header) +
                               sizeof(GridBatchData::GridBatchMetadata) +
                               numGrids * sizeof(GridBatchData::GridMetadata);

    const uint64_t sizeofMetadata =
        sizeof(V01Header) + sizeof(GridBatchData::GridBatchMetadata) +
        numGrids * sizeof(GridBatchData::GridMetadata);
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

    return makeGridBatchData(std::move(gridHdl), voxelSizes, voxelOrigins);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
