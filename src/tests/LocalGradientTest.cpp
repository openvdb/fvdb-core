// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <fvdb/detail/utils/cuda/LocalGradient.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/cuda.h>

#include <gtest/gtest.h>

#include <cstdint>

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
