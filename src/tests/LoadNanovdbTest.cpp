// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <fvdb/FVDB.h>
#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/io/SaveNanoVDB.h>

#include <nanovdb/NanoVDB.h>

#include <torch/types.h>

#include <gtest/gtest.h>

#include <cstring>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace {

constexpr const char *kGridName = "tensor-grid-with-leading-blind-name";

void
copyFixedString(char *target, const size_t maxSize, const std::string &source) {
    TORCH_CHECK_VALUE(source.size() < maxSize, "String is too long for fixed NanoVDB buffer.");
    std::memset(target, 0, maxSize);
    std::memcpy(target, source.data(), source.size());
}

nanovdb::GridHandle<nanovdb::HostBuffer>
prependGridNameBlindData(nanovdb::GridHandle<nanovdb::HostBuffer> &sourceHandle,
                         const std::string &gridName) {
    constexpr uint64_t metadataBytes = sizeof(nanovdb::GridBlindMetaData);

    const nanovdb::GridData *sourceGridData = sourceHandle.gridData(0);
    TORCH_CHECK(sourceGridData != nullptr, "Expected a valid source grid.");
    TORCH_CHECK(sourceGridData->mBlindMetadataCount == 1,
                "Test fixture expects exactly one FVDB blind metadata entry.");

    const uint8_t *sourceBytes  = static_cast<const uint8_t *>(sourceHandle.buffer().data());
    const uint64_t oldGridBytes = sourceGridData->mGridSize;
    const uint64_t blindMetadataOffset =
        static_cast<uint64_t>(sourceGridData->mBlindMetadataOffset);
    const nanovdb::GridBlindMetaData *oldBlindMetadata =
        reinterpret_cast<const nanovdb::GridBlindMetaData *>(sourceBytes + blindMetadataOffset);
    const uint64_t oldBlindDataOffset =
        blindMetadataOffset + static_cast<uint64_t>(oldBlindMetadata->mDataOffset);
    const uint64_t oldBlindDataBytes = oldGridBytes - oldBlindDataOffset;

    const uint64_t gridNameBytes       = gridName.size() + 1;
    const uint64_t paddedGridNameBytes = nanovdb::math::AlignUp<32UL>(gridNameBytes);
    const uint64_t newGridBytes        = oldGridBytes + metadataBytes + paddedGridNameBytes;

    nanovdb::HostBuffer outBuffer(newGridBytes);
    uint8_t *outBytes = static_cast<uint8_t *>(outBuffer.data());
    std::memset(outBytes, 0, newGridBytes);

    std::memcpy(outBytes, sourceBytes, blindMetadataOffset);

    nanovdb::GridData *outGridData    = reinterpret_cast<nanovdb::GridData *>(outBytes);
    outGridData->mGridSize            = newGridBytes;
    outGridData->mBlindMetadataCount  = 2;
    outGridData->mBlindMetadataOffset = static_cast<int64_t>(blindMetadataOffset);

    nanovdb::GridBlindMetaData *gridNameMetadata =
        reinterpret_cast<nanovdb::GridBlindMetaData *>(outBytes + blindMetadataOffset);
    gridNameMetadata->mDataOffset = static_cast<int64_t>(2 * metadataBytes);
    gridNameMetadata->mValueCount = paddedGridNameBytes;
    gridNameMetadata->mValueSize  = 1;
    gridNameMetadata->mSemantic   = nanovdb::GridBlindDataSemantic::Unknown;
    gridNameMetadata->mDataClass  = nanovdb::GridBlindDataClass::GridName;
    gridNameMetadata->mDataType   = nanovdb::GridType::Unknown;
    copyFixedString(gridNameMetadata->mName, nanovdb::GridBlindMetaData::MaxNameSize, "grid_name");

    nanovdb::GridBlindMetaData *fvdbMetadata = reinterpret_cast<nanovdb::GridBlindMetaData *>(
        outBytes + blindMetadataOffset + metadataBytes);
    *fvdbMetadata = *oldBlindMetadata;
    fvdbMetadata->mDataOffset =
        static_cast<int64_t>(oldBlindMetadata->mDataOffset + paddedGridNameBytes);

    uint8_t *gridNameData = outBytes + blindMetadataOffset + 2 * metadataBytes;
    std::memcpy(gridNameData, gridName.c_str(), gridNameBytes);

    uint8_t *fvdbBlindData = gridNameData + paddedGridNameBytes;
    std::memcpy(fvdbBlindData, sourceBytes + oldBlindDataOffset, oldBlindDataBytes);

    return nanovdb::GridHandle<nanovdb::HostBuffer>(std::move(outBuffer));
}

nanovdb::GridHandle<nanovdb::HostBuffer>
makeTensorGridBlindDataHandle(const fvdb::GridBatchData &gridBatchData,
                              const torch::Tensor &sourceData,
                              const std::string &gridName) {
    TORCH_CHECK(gridBatchData.batchSize() == 1, "Test fixture expects a single source grid.");
    TORCH_CHECK(gridBatchData.device().is_cpu(), "Test fixture expects a CPU source grid.");
    TORCH_CHECK(sourceData.device().is_cpu(), "Test fixture expects CPU tensor data.");
    TORCH_CHECK(sourceData.scalar_type() == torch::kFloat32,
                "Test fixture expects float32 tensor data.");
    TORCH_CHECK(sourceData.dim() >= 1, "Test fixture expects at least one tensor dimension.");
    TORCH_CHECK(gridBatchData.numVoxelsAt(0) == sourceData.size(0),
                "Test fixture expects one tensor row per voxel.");

    torch::Tensor contiguousData = sourceData.contiguous();

    const nanovdb::GridData *sourceGridData = gridBatchData.nanoGridHandle().gridData(0);
    TORCH_CHECK(sourceGridData != nullptr, "Expected a valid source grid.");

    constexpr uint64_t metadataBytes = sizeof(nanovdb::GridBlindMetaData);
    const uint64_t sourceGridBytes   = sourceGridData->mGridSize;
    const uint64_t shapeBytes = sizeof(int64_t) * static_cast<uint64_t>(contiguousData.dim() + 1);
    const uint64_t tensorBytes =
        static_cast<uint64_t>(contiguousData.numel() * contiguousData.element_size());
    const uint64_t blindDataBytes       = shapeBytes + tensorBytes;
    const uint64_t paddedBlindDataBytes = nanovdb::math::AlignUp<32UL>(blindDataBytes);
    const uint64_t totalBytes           = sourceGridBytes + metadataBytes + paddedBlindDataBytes;

    nanovdb::HostBuffer outBuffer(totalBytes);
    uint8_t *outBytes = static_cast<uint8_t *>(outBuffer.data());
    std::memset(outBytes, 0, totalBytes);
    std::memcpy(outBytes, gridBatchData.nanoGridHandle().buffer().data(), sourceGridBytes);

    nanovdb::GridData *outGridData    = reinterpret_cast<nanovdb::GridData *>(outBytes);
    outGridData->mGridSize            = totalBytes;
    outGridData->mGridClass           = nanovdb::GridClass::TensorGrid;
    outGridData->mGridType            = nanovdb::GridType::OnIndex;
    outGridData->mBlindMetadataCount  = 1;
    outGridData->mBlindMetadataOffset = static_cast<int64_t>(sourceGridBytes);
    copyFixedString(outGridData->mGridName, nanovdb::GridData::MaxNameSize, gridName);

    nanovdb::GridBlindMetaData *blindMetadata =
        reinterpret_cast<nanovdb::GridBlindMetaData *>(outBytes + sourceGridBytes);
    blindMetadata->mDataOffset = static_cast<int64_t>(metadataBytes);
    blindMetadata->mValueCount = paddedBlindDataBytes;
    blindMetadata->mValueSize  = 1;
    blindMetadata->mSemantic   = nanovdb::GridBlindDataSemantic::Unknown;
    blindMetadata->mDataClass  = nanovdb::GridBlindDataClass::Unknown;
    blindMetadata->mDataType   = nanovdb::GridType::Unknown;
    copyFixedString(
        blindMetadata->mName, nanovdb::GridBlindMetaData::MaxNameSize, "fvdb_jdatafloat32");

    uint8_t *writeHead                      = outBytes + sourceGridBytes + metadataBytes;
    *reinterpret_cast<int64_t *>(writeHead) = contiguousData.dim();
    writeHead += sizeof(int64_t);
    for (int64_t di = 0; di < contiguousData.dim(); ++di) {
        *reinterpret_cast<int64_t *>(writeHead) = contiguousData.size(di);
        writeHead += sizeof(int64_t);
    }
    std::memcpy(writeHead, contiguousData.data_ptr(), tensorBytes);

    return nanovdb::GridHandle<nanovdb::HostBuffer>(std::move(outBuffer));
}

c10::intrusive_ptr<fvdb::GridBatchData>
makeTestGrid() {
    torch::Tensor ijk = torch::tensor({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
                                      torch::TensorOptions().dtype(torch::kInt32));
    fvdb::JaggedTensor jaggedIJK(std::vector<torch::Tensor>{ijk});
    return fvdb::gridbatch_from_ijk(
        jaggedIJK, {nanovdb::Vec3d(1.0, 1.0, 1.0)}, {nanovdb::Vec3d(0.0, 0.0, 0.0)});
}

void
expectRoundTripWithLeadingGridNameBlindData(nanovdb::GridHandle<nanovdb::HostBuffer> &sourceHandle,
                                            const torch::Tensor &sourceData) {
    nanovdb::GridHandle<nanovdb::HostBuffer> patchedHandle =
        prependGridNameBlindData(sourceHandle, kGridName);

    const nanovdb::GridData *gridData = patchedHandle.gridData(0);
    ASSERT_EQ(gridData->mBlindMetadataCount, 2u);
    const auto *loadedGrid = patchedHandle.grid<nanovdb::ValueOnIndex>(0);
    if (loadedGrid != nullptr) {
        ASSERT_EQ(loadedGrid->blindMetaData(0).mDataClass, nanovdb::GridBlindDataClass::GridName);
        ASSERT_EQ(std::string(loadedGrid->blindMetaData(1).mName).rfind("fvdb_jdata", 0), 0u);
    }

    auto loaded                                 = fvdb::from_nanovdb(patchedHandle);
    const fvdb::JaggedTensor &loadedData        = std::get<1>(loaded);
    const std::vector<std::string> &loadedNames = std::get<2>(loaded);

    ASSERT_EQ(loadedNames.size(), 1u);
    EXPECT_EQ(loadedNames[0], kGridName);
    EXPECT_TRUE(torch::equal(loadedData.jdata(), sourceData));
}

} // namespace

TEST(LoadNanovdb, TensorGridBlindDataCanFollowGridNameBlindData) {
    torch::Tensor sourceData =
        torch::arange(16, torch::TensorOptions().dtype(torch::kFloat32)).reshape({4, 2, 2});
    auto grid = makeTestGrid();
    nanovdb::GridHandle<nanovdb::HostBuffer> sourceHandle =
        makeTensorGridBlindDataHandle(*grid, sourceData, kGridName);
    expectRoundTripWithLeadingGridNameBlindData(sourceHandle, sourceData);
}

TEST(LoadNanovdb, TensorGridShapeBlindDataCanFollowGridNameBlindData) {
    torch::Tensor sourceData =
        torch::arange(12, torch::TensorOptions().dtype(torch::kFloat32)).reshape({4, 3});
    auto grid = makeTestGrid();
    fvdb::JaggedTensor jaggedData(std::vector<torch::Tensor>{sourceData});
    nanovdb::GridHandle<nanovdb::HostBuffer> sourceHandle =
        fvdb::to_nanovdb(*grid, std::optional<fvdb::JaggedTensor>(jaggedData), {kGridName});
    expectRoundTripWithLeadingGridNameBlindData(sourceHandle, sourceData);
}

TEST(SaveNanoVDB, BlindFloatPayloadMatchesValueOnIndexIndexSpace) {
    auto grid = makeTestGrid();
    torch::Tensor sourceData =
        torch::tensor({-3.5f, -1.25f, 2.0f, 7.0f}, torch::TensorOptions().dtype(torch::kFloat32));
    fvdb::JaggedTensor jaggedData(std::vector<torch::Tensor>{sourceData});

    nanovdb::GridHandle<nanovdb::HostBuffer> handle =
        fvdb::detail::io::toNVDBWithBlindFloat(*grid, jaggedData);

    const auto *savedGrid = handle.grid<nanovdb::ValueOnIndex>(0);
    ASSERT_NE(savedGrid, nullptr);
    ASSERT_EQ(savedGrid->blindDataCount(), 1u);

    const nanovdb::GridBlindMetaData &metadata = savedGrid->blindMetaData(0);
    EXPECT_EQ(std::string(metadata.mName), "fvdb_sdf_float32");
    EXPECT_EQ(metadata.mDataType, nanovdb::GridType::Float);
    EXPECT_EQ(metadata.mValueSize, sizeof(float));

    // ValueOnIndex reserves index 0 for background and active voxel values use
    // indices 1..N, so the blind payload must live in that same index space.
    ASSERT_EQ(metadata.mValueCount, static_cast<uint64_t>(sourceData.numel() + 1));

    const float *payload = static_cast<const float *>(metadata.blindData());
    ASSERT_NE(payload, nullptr);
    EXPECT_FLOAT_EQ(payload[0], 0.0f);
    for (int64_t i = 0; i < sourceData.numel(); ++i) {
        EXPECT_FLOAT_EQ(payload[i + 1], sourceData[i].item<float>());
    }
}
