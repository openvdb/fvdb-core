// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <fvdb/detail/utils/cuda/GradientReduction.h>
#include <fvdb/detail/utils/cuda/LocalGradient.h>

#include <ATen/ops/from_blob.h>
#include <c10/cuda/CUDAGuard.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using ReductionCase = std::tuple<int64_t, torch::ScalarType, bool>;

class GradientReductionTest : public ::testing::TestWithParam<ReductionCase> {};

TEST(GradientReductionValidationTest, RejectsNonzeroStorageOffset) {
    const int deviceCount = c10::cuda::device_count();
    if (deviceCount == 0) {
        GTEST_SKIP() << "CUDA is required for gradient reduction tests";
    }

    const c10::cuda::CUDAGuard deviceGuard(0);
    const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0);
    // This view has enough capacity for reduction, but does not start at its storage base.
    const auto gradient = torch::zeros({deviceCount + 1}, options).narrow(0, 1, deviceCount);
    const std::vector<torch::Tensor> localGradients(deviceCount, gradient);
    try {
        fvdb::detail::reduceLocalGradients(localGradients);
        FAIL() << "Expected a nonzero storage offset to be rejected";
    } catch (const c10::Error &error) {
        EXPECT_NE(std::string(error.what_without_backtrace()).find("beginning of its storage"),
                  std::string::npos);
    }
}

TEST(GradientReductionValidationTest, RejectsIncorrectStorageSize) {
    const int deviceCount = c10::cuda::device_count();
    if (deviceCount == 0) {
        GTEST_SKIP() << "CUDA is required for gradient reduction tests";
    }

    const c10::cuda::CUDAGuard deviceGuard(0);
    const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0);
    for (const int64_t storageElements: {int64_t{1}, int64_t{deviceCount} + 1}) {
        if (storageElements == deviceCount) {
            continue; // A one-element allocation needs no padding on a single device.
        }
        SCOPED_TRACE(storageElements);
        const auto gradient = torch::zeros({storageElements}, options).narrow(0, 0, 1);
        const std::vector<torch::Tensor> localGradients(deviceCount, gradient);
        try {
            fvdb::detail::reduceLocalGradients(localGradients);
            FAIL() << "Expected incorrect padded storage size to be rejected";
        } catch (const c10::Error &error) {
            EXPECT_NE(std::string(error.what_without_backtrace())
                          .find("storage must match the padded reduction size"),
                      std::string::npos);
        }
    }
}

TEST_P(GradientReductionTest, MatchesIndependentSumAndPreservesLogicalShape) {
    const int deviceCount = c10::cuda::device_count();
    if (deviceCount == 0) {
        GTEST_SKIP() << "CUDA is required for gradient reduction tests";
    }
    const auto [numElements, dtype, useNonDefaultStreams] = GetParam();
    const c10::cuda::CUDAGuard deviceGuard(0);
    std::vector<c10::cuda::CUDAStream> streams;
    for (const auto deviceId: c10::irange(deviceCount)) {
        streams.emplace_back(useNonDefaultStreams ? c10::cuda::getStreamFromPool(false, deviceId)
                                                  : c10::cuda::getDefaultCUDAStream(deviceId));
    }
    const c10::cuda::CUDAMultiStreamGuard streamGuard(streams);
    const auto options = torch::TensorOptions().dtype(dtype).device(torch::kCPU);
    // A noncontiguous shape/dtype template must still produce contiguous local gradient buffers.
    const auto shape = torch::zeros({}, options).expand({1, numElements});
    auto expected    = torch::zeros({numElements}, options);
    std::vector<torch::Tensor> localGradients;
    const int64_t shardSize = fvdb::detail::localGradientShardSize(numElements, deviceCount);
    const int64_t paddedNumElements = shardSize * deviceCount;
    for (const auto deviceId: c10::irange(deviceCount)) {
        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        auto gradient = fvdb::detail::makeLocalGradient(shape, deviceId, streams[deviceId]);
        ASSERT_EQ(gradient.sizes(), shape.sizes());
        ASSERT_EQ(gradient.scalar_type(), dtype);
        ASSERT_TRUE(gradient.is_contiguous());
        ASSERT_EQ(gradient.storage().nbytes(), paddedNumElements * gradient.element_size());
        EXPECT_EQ(
            gradient.as_strided({paddedNumElements}, {1}).cpu().count_nonzero().item<int64_t>(), 0);
        // Exact small integers, different on each device, catch omissions and duplicated shards.
        auto input = torch::arange(numElements, options).remainder(17) - 8 + 3 * deviceId;
        expected += input;
        gradient.view({-1}).copy_(input);
        localGradients.emplace_back(std::move(gradient));
    }

    fvdb::detail::reduceLocalGradients(localGradients);
    for (const auto deviceId: c10::irange(deviceCount)) {
        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        C10_CUDA_CHECK(cudaStreamSynchronize(streams[deviceId]));
    }

    // Reassemble the public logical shards independently of the reduction's padded views.
    auto actual = torch::empty_like(expected);
    for (const auto deviceId: c10::irange(deviceCount)) {
        const int64_t begin = std::min<int64_t>(deviceId * shardSize, numElements);
        const int64_t end   = std::min<int64_t>((deviceId + 1) * shardSize, numElements);
        ASSERT_EQ(localGradients[deviceId].sizes(), shape.sizes());
        actual.slice(0, begin, end)
            .copy_(localGradients[deviceId].view({-1}).slice(0, begin, end).cpu());
        const auto padding = localGradients[deviceId]
                                 .as_strided({paddedNumElements}, {1})
                                 .slice(0, numElements, paddedNumElements)
                                 .cpu();
        EXPECT_EQ(padding.count_nonzero().item<int64_t>(), 0);
    }
    EXPECT_TRUE(torch::equal(actual, expected));
}

TEST_P(GradientReductionTest, CameraShardsMatchIndependentSum) {
    const int deviceCount = c10::cuda::device_count();
    if (deviceCount == 0) {
        GTEST_SKIP() << "CUDA is required for gradient reduction tests";
    }
    const auto [numGaussians, dtype, useNonDefaultStreams] = GetParam();
    const c10::cuda::CUDAGuard deviceGuard(0);
    std::vector<c10::cuda::CUDAStream> streams;
    fvdb::detail::DeviceLists deviceLists(deviceCount);
    std::vector<c10::DeviceIndex> allDevices;
    for (const auto deviceId: c10::irange(deviceCount)) {
        streams.emplace_back(useNonDefaultStreams ? c10::cuda::getStreamFromPool(false, deviceId)
                                                  : c10::cuda::getDefaultCUDAStream(deviceId));
        deviceLists[deviceId].push_back(deviceId);
        allDevices.push_back(deviceId);
    }
    // Mix one camera per GPU with an all-device reduction and a camera with no contributors.
    // On three or more GPUs, also exercise deferred summation of point-to-point contributions.
    deviceLists.push_back(allDevices);
    deviceLists.emplace_back();
    if (deviceCount > 2) {
        deviceLists.push_back({allDevices.front(), allDevices.back()});
    }
    const c10::cuda::CUDAMultiStreamGuard streamGuard(streams);
    const auto options        = torch::TensorOptions().dtype(dtype).device(torch::kCPU);
    const int64_t numCameras  = deviceLists.size();
    const int64_t numChannels = 3;
    const auto shape  = torch::zeros({}, options).expand({numCameras, numGaussians, numChannels});
    auto expected     = torch::zeros({numCameras, numGaussians, numChannels}, options);
    const auto values = torch::arange(numGaussians * numChannels, options)
                            .remainder(17)
                            .view({numGaussians, numChannels});
    std::vector<torch::Tensor> localGradients;
    for (const auto deviceId: c10::irange(deviceCount)) {
        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        auto gradient = fvdb::detail::makeLocalGradient<false>(shape, deviceId, streams[deviceId]);
        auto input    = torch::zeros_like(expected);
        for (const auto cameraId: c10::irange(numCameras)) {
            const auto &sources = deviceLists[cameraId];
            if (std::find(sources.begin(), sources.end(), deviceId) != sources.end()) {
                // Distinct exact values catch swapped cameras, missing peers, and duplicate sums.
                input.select(0, cameraId).copy_(values - 8 + 3 * deviceId + 5 * cameraId);
            }
        }
        expected += input;
        gradient.copy_(input);
        localGradients.emplace_back(std::move(gradient));
    }

    fvdb::detail::reduceLocalGradients<false>(localGradients, deviceLists);
    for (const auto deviceId: c10::irange(deviceCount)) {
        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        C10_CUDA_CHECK(cudaStreamSynchronize(streams[deviceId]));
    }

    // Reassemble each camera's Gaussian shards without using the production copy helper.
    const int64_t shardSize = fvdb::detail::localGradientShardSize(numGaussians, deviceCount);
    const int64_t paddedNumGaussians = shardSize * deviceCount;
    auto actual                      = torch::empty_like(expected);
    for (const auto deviceId: c10::irange(deviceCount)) {
        const auto &gradient = localGradients[deviceId];
        const int64_t begin  = std::min<int64_t>(deviceId * shardSize, numGaussians);
        const int64_t end    = std::min<int64_t>((deviceId + 1) * shardSize, numGaussians);
        ASSERT_EQ(gradient.sizes(), shape.sizes());
        ASSERT_EQ(gradient.scalar_type(), dtype);
        actual.slice(1, begin, end).copy_(gradient.slice(1, begin, end).cpu());
        const auto padding = gradient
                                 .as_strided({numCameras, paddedNumGaussians, numChannels},
                                             {paddedNumGaussians * numChannels, numChannels, 1})
                                 .slice(1, numGaussians, paddedNumGaussians)
                                 .cpu();
        EXPECT_EQ(padding.count_nonzero().item<int64_t>(), 0);
    }
    EXPECT_TRUE(torch::equal(actual, expected));
}

TEST(CompactCameraGradientTest, ReducesAndPublishesOnlyOwnedShards) {
    const int deviceCount = c10::cuda::device_count();
    if (deviceCount == 0) {
        GTEST_SKIP() << "CUDA is required for gradient reduction tests";
    }
    const c10::cuda::CUDAGuard deviceGuard(0);
    std::vector<c10::cuda::CUDAStream> streams;
    for (const auto deviceId: c10::irange(deviceCount)) {
        streams.emplace_back(c10::cuda::getStreamFromPool(false, deviceId));
    }
    const c10::cuda::CUDAMultiStreamGuard streamGuard(streams);
    const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    using fvdb::detail::DeviceLists;
    using fvdb::detail::makeDeviceLists;
    const c10::DeviceIndex lastDevice      = deviceCount - 1;
    const std::vector<DeviceLists> layouts = {
        makeDeviceLists(1, 16 * deviceCount, 4, deviceCount), // One camera shared by every GPU.
        makeDeviceLists(2, 16 * deviceCount, 4, deviceCount), // Two cameras, including 2C/4GPU.
        makeDeviceLists(2, 15, 3, deviceCount),               // A rank can straddle cameras.
        makeDeviceLists(deviceCount, 16 * deviceCount, 4, deviceCount), // One camera per GPU.
        DeviceLists{{lastDevice}, {}}, // Receive-only ranks and a camera with no contributors.
        DeviceLists{{}}};              // Every rank has an empty logical accumulation tensor.

    for (const auto &deviceLists: layouts) {
        const int64_t numCameras = deviceLists.size();
        for (const int64_t numGaussians: {int64_t{0}, int64_t{1}, int64_t{17}}) {
            SCOPED_TRACE(numCameras);
            SCOPED_TRACE(numGaussians);
            constexpr int64_t numChannels = 3;
            const int64_t shardSize =
                fvdb::detail::localGradientShardSize(numGaussians, deviceCount);
            const auto shape =
                torch::zeros({}, options).expand({numCameras, numGaussians, numChannels});
            auto expected = torch::zeros({numCameras, numGaussians, numChannels}, options);
            // A managed CUDA output exercises the production copy helper from every device,
            // including ranks whose logical accumulation tensor is empty but whose tail is not.
            // Initialize it before queuing any GPU work for this case.
            C10_CUDA_CHECK(cudaSetDevice(0));
            void *outputData = nullptr;
            C10_CUDA_CHECK(cudaMallocManaged(
                &outputData, std::max<int64_t>(expected.numel(), 1) * sizeof(float)));
            auto output = at::from_blob(
                outputData,
                expected.sizes(),
                [](void *data) { C10_CUDA_CHECK(cudaFree(data)); },
                options.device(torch::kCUDA, 0));
            std::fill_n(static_cast<float *>(outputData),
                        expected.numel(),
                        std::numeric_limits<float>::quiet_NaN());
            const auto values = torch::arange(numGaussians * numChannels, options)
                                    .remainder(17)
                                    .view({numGaussians, numChannels});
            std::vector<int64_t> cameraOffsets(deviceCount, 0);
            std::vector<torch::Tensor> localGradients(deviceCount);
            for (const auto deviceId: c10::irange(deviceCount)) {
                C10_CUDA_CHECK(cudaSetDevice(deviceId));
                int64_t cameraCount = 0;
                auto input          = torch::zeros_like(expected);
                for (const auto cameraId: c10::irange(numCameras)) {
                    const auto &sources = deviceLists[cameraId];
                    if (std::find(sources.begin(), sources.end(), deviceId) == sources.end()) {
                        continue;
                    }
                    if (cameraCount == 0) {
                        cameraOffsets[deviceId] = cameraId;
                    }
                    cameraCount = cameraId + 1 - cameraOffsets[deviceId];
                    input.select(0, cameraId).copy_(values - 8 + 3 * deviceId + 5 * cameraId);
                }
                expected += input;
                auto gradient = fvdb::detail::makeLocalGradient<false>(
                    shape, deviceId, streams[deviceId], cameraCount);
                const int64_t begin = std::min<int64_t>(deviceId * shardSize, numGaussians);
                const int64_t end   = std::min<int64_t>((deviceId + 1) * shardSize, numGaussians);
                const int64_t storageElements = (cameraCount * shardSize * deviceCount +
                                                 (numCameras - cameraCount) * (end - begin)) *
                                                numChannels;
                ASSERT_EQ(gradient.size(0), cameraCount);
                ASSERT_EQ(gradient.storage().nbytes(), storageElements * sizeof(float));
                gradient.copy_(input.narrow(0, cameraOffsets[deviceId], cameraCount));
                localGradients[deviceId] = std::move(gradient);
            }

            fvdb::detail::reduceLocalGradients<false>(localGradients, deviceLists, cameraOffsets);
            fvdb::detail::copyLocalGradients<false>(localGradients, output, cameraOffsets);
            // Exercise stream-ordered release after copies have been queued.
            localGradients.clear();
            for (const auto deviceId: c10::irange(deviceCount)) {
                C10_CUDA_CHECK(cudaSetDevice(deviceId));
                C10_CUDA_CHECK(cudaStreamSynchronize(streams[deviceId]));
            }
            C10_CUDA_CHECK(cudaSetDevice(0));
            EXPECT_TRUE(torch::equal(output.cpu(), expected));
        }
    }
}

INSTANTIATE_TEST_SUITE_P(ShapesAndStreams,
                         GradientReductionTest,
                         ::testing::Combine(::testing::Values(int64_t{0},
                                                              int64_t{1},
                                                              int64_t{2},
                                                              int64_t{3},
                                                              int64_t{7},
                                                              int64_t{17},
                                                              int64_t{1023},
                                                              int64_t{1024},
                                                              int64_t{1025}),
                                            ::testing::Values(torch::kFloat32, torch::kFloat64),
                                            ::testing::Bool()));

TEST(DeviceListsTest, UsesActualAlignedTileRanges) {
    using fvdb::detail::DeviceLists;
    using fvdb::detail::makeDeviceLists;
    EXPECT_EQ(makeDeviceLists(1, 24, 4, 2), (DeviceLists{{0, 1}}));
    EXPECT_EQ(makeDeviceLists(2, 24, 4, 2), (DeviceLists{{0}, {1}}));
    EXPECT_EQ(makeDeviceLists(2, 24, 4, 4), (DeviceLists{{0, 1}, {2, 3}}));
    // A device's tile rows can straddle a camera boundary.
    EXPECT_EQ(makeDeviceLists(2, 15, 3, 4), (DeviceLists{{0, 1}, {1, 2, 3}}));
    EXPECT_EQ(makeDeviceLists(3, 10, 5, 2), (DeviceLists{{0}, {0, 1}, {1}}));
    // Empty raster ranks, empty images, and one device all use the same partition rule.
    EXPECT_EQ(makeDeviceLists(2, 1, 1, 4), (DeviceLists{{0}, {1}}));
    EXPECT_EQ(makeDeviceLists(2, 0, 1, 4), (DeviceLists{{}, {}}));
    EXPECT_EQ(makeDeviceLists(3, 24, 4, 1), (DeviceLists{{0}, {0}, {0}}));
    // Sparse membership stays conservative, but devices with no active tiles are excluded.
    EXPECT_EQ(makeDeviceLists(2, 24, 1, 4, 2), (DeviceLists{{0, 1}, {0, 1}}));
}

} // namespace
