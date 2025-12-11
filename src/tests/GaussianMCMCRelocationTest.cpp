// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include "utils/Tensor.h"

#include <fvdb/detail/ops/gsplat/GaussianMCMCRelocation.h>

#include <torch/torch.h>

#include <gtest/gtest.h>

#include <cmath>
#include <tuple>

namespace {

int64_t
binomial(int n, int k) {
    if (k < 0 || k > n)
        return 0;
    if (k == 0 || k == n)
        return 1;
    int64_t res = 1;
    for (int i = 1; i <= k; ++i) {
        res = res * (n - (k - i));
        res /= i;
    }
    return res;
}

torch::Tensor
buildBinomialCoeffsCPU(int nMax) {
    auto coeffs = torch::zeros({nMax, nMax}, torch::TensorOptions().dtype(torch::kFloat32));
    for (int row = 0; row < nMax; ++row) {
        for (int k = 0; k <= row; ++k) {
            coeffs[row][k] = static_cast<float>(binomial(row, k));
        }
    }
    return coeffs;
}

std::tuple<torch::Tensor, torch::Tensor>
referenceRelocation(const torch::Tensor &logScales,
                    const torch::Tensor &logitOpacities,
                    const torch::Tensor &ratios,
                    const torch::Tensor &binomialCoeffsCPU) {
    // Convert to probability/linear space to mirror the kernel's internal math.
    auto opacities    = torch::sigmoid(logitOpacities);
    auto scales       = torch::exp(logScales);
    auto opacitiesNew = torch::empty_like(opacities);
    auto scalesNew    = torch::empty_like(scales);

    const auto N = opacities.size(0);
    for (int64_t idx = 0; idx < N; ++idx) {
        const int32_t nIdx = ratios[idx].item<int32_t>();
        const float opacity =
            opacities[idx].item<float>(); // CPU tensor so item<float> is fine for test sizes
        const float opacityNew = 1.0f - std::pow(1.0f - opacity, 1.0f / static_cast<float>(nIdx));
        opacitiesNew[idx]      = opacityNew;

        float denomSum = 0.0f;
        for (int32_t i = 1; i <= nIdx; ++i) {
            for (int32_t k = 0; k <= (i - 1); ++k) {
                const float binomCoeff = binomialCoeffsCPU[(i - 1)][k].item<float>();
                const float sign       = (k % 2 == 0) ? 1.0f : -1.0f;
                denomSum += binomCoeff * sign * std::pow(opacityNew, static_cast<float>(k + 1)) /
                            std::sqrt(static_cast<float>(k + 1));
            }
        }

        const float coeff = opacity / denomSum;
        scalesNew[idx]    = coeff * scales[idx];
    }

    // Convert back to logit/log for comparison with kernel outputs.
    auto logitOpacitiesNew = torch::log(opacitiesNew) - torch::log1p(-opacitiesNew);
    auto logScalesNew      = torch::log(scalesNew);
    return {logitOpacitiesNew, logScalesNew};
}

class GaussianRelocationTest : public ::testing::Test {
  protected:
    void
    SetUp() override {
        if (!torch::cuda::is_available()) {
            GTEST_SKIP() << "CUDA is required for GaussianRelocation tests";
        }
        torch::manual_seed(0);
    }

    void
    TestRelocation(const torch::Tensor &logScales,
                   const torch::Tensor &logitOpacities,
                   const torch::Tensor &ratios) {
        const int nMax               = logScales.size(0);
        auto const binomialCoeffsCPU = buildBinomialCoeffsCPU(nMax);
        auto const binomialCoeffs    = binomialCoeffsCPU.to(logScales.device());

        const auto [gpuLogitOpacitiesNew, gpuLogScalesNew] =
            fvdb::detail::ops::dispatchGaussianRelocation<torch::kCUDA>(
                logScales, logitOpacities, ratios, binomialCoeffs, nMax);

        const auto [refLogitNew, refLogScalesNew] = referenceRelocation(
            logScales.cpu(), logitOpacities.cpu(), ratios.cpu(), binomialCoeffsCPU);

        EXPECT_TRUE(torch::allclose(gpuLogitOpacitiesNew.cpu(), refLogitNew, 1e-6, 1e-6));
        EXPECT_TRUE(torch::allclose(gpuLogScalesNew.cpu(), refLogScalesNew, 1e-6, 1e-6));
    }
};

TEST_F(GaussianRelocationTest, ComputesExpectedValues) {
    // Focus on low/dead opacities; include one mildly higher entry to ensure mixed behavior.
    auto opacities =
        torch::tensor({1e-6f, 5e-4f, 1e-2f, 0.2f}, fvdb::test::tensorOpts<float>(torch::kCUDA));
    auto scales = torch::tensor(
        {{1.0f, 0.8f, 1.2f}, {0.5f, 0.25f, 0.125f}, {1.5f, 0.6f, 0.9f}, {0.8f, 1.1f, 0.7f}},
        fvdb::test::tensorOpts<float>(torch::kCUDA));
    auto logScales      = torch::log(scales);
    auto logitOpacities = torch::log(opacities) - torch::log1p(-opacities);
    auto ratios         = torch::tensor({1, 2, 3, 4},
                                torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32));

    TestRelocation(logScales, logitOpacities, ratios);
}

TEST_F(GaussianRelocationTest, HandlesEdgeOpacitiesAndRatios) {
    auto opacities =
        torch::tensor({1e-7f, 1e-5f, 5e-4f, 1e-2f}, fvdb::test::tensorOpts<float>(torch::kCUDA));
    auto scales = torch::tensor(
        {{1.0f, 1.0f, 1.0f}, {0.4f, 0.3f, 0.2f}, {1.8f, 0.6f, 0.9f}, {0.9f, 1.1f, 0.7f}},
        fvdb::test::tensorOpts<float>(torch::kCUDA));
    auto logScales      = torch::log(scales);
    auto logitOpacities = torch::log(opacities) - torch::log1p(-opacities);
    auto ratios         = torch::tensor({1, 4, 2, 3},
                                torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32));

    TestRelocation(logScales, logitOpacities, ratios);
}

TEST_F(GaussianRelocationTest, ValidatesInputs) {
    const int nMax         = 2;
    auto binomialCoeffsCPU = buildBinomialCoeffsCPU(nMax);

    auto opacities =
        torch::tensor({0.25f, 0.5f}, fvdb::test::tensorOpts<float>(torch::kCUDA)).contiguous();
    auto scales = torch::tensor({{1.0f, 2.0f, 3.0f}, {0.5f, 1.0f, 2.0f}},
                                fvdb::test::tensorOpts<float>(torch::kCUDA))
                      .contiguous();
    auto logScales      = torch::log(scales);
    auto logitOpacities = torch::log(opacities) - torch::log1p(-opacities);
    auto ratios =
        torch::tensor({1, 2}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32))
            .contiguous();
    auto binomialCoeffs = binomialCoeffsCPU.to(torch::kCUDA);

    // binomialCoeffs on CPU
    EXPECT_THROW(fvdb::detail::ops::dispatchGaussianRelocation<torch::kCUDA>(
                     logScales, logitOpacities, ratios, binomialCoeffsCPU, nMax),
                 c10::Error);

    // binomialCoeffs wrong shape
    auto badBinomShape = binomialCoeffs.slice(/*dim=*/0, 0, nMax - 1);
    EXPECT_THROW(fvdb::detail::ops::dispatchGaussianRelocation<torch::kCUDA>(
                     logScales, logitOpacities, ratios, badBinomShape, nMax),
                 c10::Error);

    // ratios wrong dtype
    auto ratiosLong = ratios.to(torch::kInt64);
    EXPECT_THROW(fvdb::detail::ops::dispatchGaussianRelocation<torch::kCUDA>(
                     logScales, logitOpacities, ratiosLong, binomialCoeffs, nMax),
                 c10::Error);

    // opacities on CPU
    EXPECT_THROW(fvdb::detail::ops::dispatchGaussianRelocation<torch::kCUDA>(
                     logScales.cpu(), logitOpacities.cpu(), ratios, binomialCoeffs, nMax),
                 c10::Error);

    // scales wrong shape
    auto logScalesBad = logScales.view({2, 3, 1});
    EXPECT_THROW(fvdb::detail::ops::dispatchGaussianRelocation<torch::kCUDA>(
                     logScalesBad, logitOpacities, ratios, binomialCoeffs, nMax),
                 c10::Error);
}

TEST_F(GaussianRelocationTest, CpuNotImplemented) {
    const int nMax         = 2;
    auto binomialCoeffsCPU = buildBinomialCoeffsCPU(nMax);

    auto opacities      = torch::tensor({0.25f, 0.5f}, fvdb::test::tensorOpts<float>(torch::kCPU));
    auto scales         = torch::tensor({{1.0f, 2.0f, 3.0f}, {0.5f, 1.0f, 2.0f}},
                                fvdb::test::tensorOpts<float>(torch::kCPU));
    auto logScales      = torch::log(scales);
    auto logitOpacities = torch::log(opacities) - torch::log1p(-opacities);
    auto ratios =
        torch::tensor({1, 2}, torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32));

    EXPECT_THROW(fvdb::detail::ops::dispatchGaussianRelocation<torch::kCPU>(
                     logScales, logitOpacities, ratios, binomialCoeffsCPU, nMax),
                 c10::Error);
}

} // namespace
