// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <fvdb/detail/utils/cuda/GradientReduction.h>
#include <fvdb/detail/utils/cuda/LocalGradient.h>

#include <c10/cuda/CUDAGuard.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using ReductionCase = std::tuple<int64_t, torch::ScalarType, bool>;

class GradientReductionTest : public ::testing::TestWithParam<ReductionCase> {};

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
    const int64_t shardSize         = (numElements + deviceCount - 1) / deviceCount;
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

    fvdb::detail::reduceGradientShards(localGradients);
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

} // namespace
