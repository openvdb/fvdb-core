// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <fvdb/detail/utils/cuda/LocalGradient.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/cuda.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <string>

TEST(LocalGradientTest, ShardSizeChecksPaddingOverflow) {
    const auto maxElements = std::numeric_limits<int64_t>::max();
    EXPECT_EQ(fvdb::detail::localGradientShardSize(maxElements, 1), maxElements);
    EXPECT_EQ(fvdb::detail::localGradientShardSize(maxElements - 1, 2), maxElements / 2);
    EXPECT_THROW(fvdb::detail::localGradientShardSize(maxElements, 2), c10::Error);
    EXPECT_THROW(fvdb::detail::localGradientShardSize(1, 0), c10::Error);
}

TEST(LocalGradientTest, AllocationByteCountOverflowThrowsBeforeAllocation) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA is required for LocalGradient tests";
    }

    const c10::cuda::CUDAGuard deviceGuard(0);
    const auto stream = c10::cuda::getCurrentCUDAStream(0).stream();
    const auto scalar =
        torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    // Exercise both signed and unsigned byte-count overflow using only one stored scalar.
    for (const int64_t numElements: {int64_t{1} << 61, int64_t{1} << 62}) {
        SCOPED_TRACE(numElements);
        const auto expanded = scalar.expand({numElements});
        try {
            fvdb::detail::makeLocalGradient(expanded, 0, stream);
            FAIL() << "Expected allocation byte-count overflow to be rejected";
        } catch (const c10::Error &error) {
            EXPECT_NE(std::string(error.what_without_backtrace()).find("allocation size overflows"),
                      std::string::npos);
        }
    }
}

TEST(LocalGradientTest, OversizedAllocationThrows) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA is required for LocalGradient tests";
    }

    const c10::cuda::CUDAGuard deviceGuard(0);
    const auto stream = c10::cuda::getCurrentCUDAStream(0).stream();
    const auto scalar =
        torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    // Request an impossibly large CUDA allocation while storing only one scalar on the CPU.
    const auto expanded = scalar.expand({int64_t{1} << 60});
    ASSERT_EQ(expanded.storage().nbytes(), scalar.element_size());

    EXPECT_THROW(fvdb::detail::makeLocalGradient(expanded, 0, stream), c10::Error);
}
