// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <fvdb/FVDB.h>
#include <fvdb/JaggedTensor.h>

#include <nanovdb/NanoVDB.h>

#include <torch/types.h>

#include <gtest/gtest.h>

#include <cstring>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace {

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

c10::intrusive_ptr<fvdb::GridBatchData>
makeTestGrid() {
    torch::Tensor ijk = torch::tensor({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
                                      torch::TensorOptions().dtype(torch::kInt32));
    fvdb::JaggedTensor jaggedIJK(std::vector<torch::Tensor>{ijk});
    return fvdb::gridbatch_from_ijk(
        jaggedIJK, {nanovdb::Vec3d(1.0, 1.0, 1.0)}, {nanovdb::Vec3d(0.0, 0.0, 0.0)});
}

void
expectRoundTripWithLeadingGridNameBlindData(const torch::Tensor &sourceData) {
    constexpr const char *gridName = "tensor-grid-with-leading-blind-name";

    auto grid = makeTestGrid();
    fvdb::JaggedTensor jaggedData(std::vector<torch::Tensor>{sourceData});
    nanovdb::GridHandle<nanovdb::HostBuffer> sourceHandle =
        fvdb::to_nanovdb(*grid, std::optional<fvdb::JaggedTensor>(jaggedData), {gridName});
    nanovdb::GridHandle<nanovdb::HostBuffer> patchedHandle =
        prependGridNameBlindData(sourceHandle, gridName);

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
    EXPECT_EQ(loadedNames[0], gridName);
    EXPECT_TRUE(torch::equal(loadedData.jdata(), sourceData));
}

} // namespace

TEST(LoadNanovdb, TensorGridBlindDataCanFollowGridNameBlindData) {
    torch::Tensor sourceData =
        torch::arange(16, torch::TensorOptions().dtype(torch::kFloat32)).reshape({4, 2, 2});
    expectRoundTripWithLeadingGridNameBlindData(sourceData);
}

TEST(LoadNanovdb, TensorGridShapeBlindDataCanFollowGridNameBlindData) {
    torch::Tensor sourceData =
        torch::arange(12, torch::TensorOptions().dtype(torch::kFloat32)).reshape({4, 3});
    expectRoundTripWithLeadingGridNameBlindData(sourceData);
}
