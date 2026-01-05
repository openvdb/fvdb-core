// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include "utils/Tensor.h"

#include <fvdb/detail/ops/gsplat/GaussianMCMCAddNoise.h>

#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <torch/torch.h>

#include <gtest/gtest.h>

#include <cmath>

namespace {

// Match kernel logistic parameters (k = 100, x0 = 0.995).
torch::Tensor
logisticTensor(const torch::Tensor &x) {
    return 1.0f / (1.0f + torch::exp(-100.0f * (x - 0.995f)));
}

class GaussianMCMCAddNoiseTest : public ::testing::Test {
  protected:
    void
    SetUp() override {
        if (!torch::cuda::is_available()) {
            GTEST_SKIP() << "CUDA is required for GaussianMCMCAddNoise tests";
        }
        torch::manual_seed(0);
    }

    torch::TensorOptions
    floatOpts() const {
        return fvdb::test::tensorOpts<float>(torch::kCUDA);
    }

    // Save the current CUDA RNG state so we can reproduce the baseNoise that
    // dispatchGaussianMCMCAddNoise draws internally.
    torch::Tensor
    saveCudaGeneratorState() {
        auto gen = at::cuda::detail::getDefaultCUDAGenerator();
        return gen.get_state();
    }

    void
    restoreCudaGeneratorState(const torch::Tensor &state) {
        auto gen = at::cuda::detail::getDefaultCUDAGenerator();
        gen.set_state(state);
    }
};

TEST_F(GaussianMCMCAddNoiseTest, AppliesNoiseWithDeterministicBaseNoise) {
    auto means = torch::tensor({{0.0f, 0.0f, 0.0f}, {1.0f, 2.0f, 3.0f}}, floatOpts()).contiguous();
    const auto logScales      = torch::zeros({2, 3}, floatOpts()).contiguous(); // unit covariance
    const auto opacities      = torch::tensor({0.25f, 0.6f}, floatOpts());
    const auto logitOpacities = torch::log(opacities) - torch::log1p(-opacities);
    const auto quats =
        torch::tensor({{1.0f, 0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f}}, floatOpts())
            .contiguous();
    constexpr float noiseScale = 0.4f;

    const auto rngState = saveCudaGeneratorState();
    auto meansBaseline  = means.clone();

    fvdb::detail::ops::dispatchGaussianMCMCAddNoise<torch::kCUDA>(
        means, logScales, logitOpacities, quats, noiseScale, 0.005, 100);

    restoreCudaGeneratorState(rngState);
    const auto baseNoise = torch::randn_like(meansBaseline);

    // Expected delta on CPU: gate * noiseScale * baseNoise, then scaled by covariance diag.
    auto opacityCpu     = opacities.cpu();
    auto gate           = logisticTensor(torch::ones_like(opacityCpu) - opacityCpu); // [N]
    auto delta          = baseNoise.cpu() * gate.unsqueeze(1) * noiseScale;          // [N,3]
    const auto expected = meansBaseline.cpu() + delta;

    EXPECT_TRUE(torch::allclose(means.cpu(), expected, 1e-5, 1e-6));
}

TEST_F(GaussianMCMCAddNoiseTest, RespectsAnisotropicScales) {
    auto means = torch::zeros({1, 3}, floatOpts()).contiguous();
    const auto scales =
        torch::tensor({std::log(2.0f), std::log(1.0f), std::log(0.5f)}, floatOpts());
    const auto logScales      = scales.view({1, 3}).contiguous();
    const auto opacities      = torch::tensor({0.3f}, floatOpts());
    const auto logitOpacities = torch::log(opacities) - torch::log1p(-opacities);
    const auto quats          = torch::tensor({{1.0f, 0.0f, 0.0f, 0.0f}}, floatOpts()).contiguous();
    constexpr float noiseScale = 1.0f;

    const auto rngState = saveCudaGeneratorState();

    fvdb::detail::ops::dispatchGaussianMCMCAddNoise<torch::kCUDA>(
        means, logScales, logitOpacities, quats, noiseScale, 0.005, 100);

    restoreCudaGeneratorState(rngState);
    const auto baseNoise = torch::randn_like(means);

    auto gate = logisticTensor(torch::ones({1}, torch::kFloat32) - opacities.cpu()); // scalar
    const auto covarDiag = torch::pow(torch::exp(logScales.cpu()), 2);               // [1,3]
    const auto expected  = (baseNoise.cpu() * gate.unsqueeze(1) * noiseScale) * covarDiag +
                          torch::zeros_like(baseNoise.cpu());

    // With identity rotation, covariance is diagonal; check elementwise scaling.
    EXPECT_TRUE(torch::allclose(means.cpu(), expected, 1e-5, 1e-6));
}

TEST_F(GaussianMCMCAddNoiseTest, HighOpacitySuppressesNoise) {
    auto means                = torch::zeros({2, 3}, floatOpts()).contiguous();
    const auto logScales      = torch::zeros({2, 3}, floatOpts()).contiguous();
    const auto logitOpacities = torch::full({2}, 10.0f, floatOpts()); // opacity ~ 1
    const auto quats =
        torch::tensor({{1.0f, 0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f}}, floatOpts())
            .contiguous();
    constexpr float noiseScale = 1.0f;

    fvdb::detail::ops::dispatchGaussianMCMCAddNoise<torch::kCUDA>(
        means, logScales, logitOpacities, quats, noiseScale, 0.005, 100);

    // Gate approaches zero when opacity ~1; expect negligible movement.
    const auto maxAbs = torch::abs(means).max().item<float>();
    EXPECT_LT(maxAbs, 1e-5f);
}

TEST_F(GaussianMCMCAddNoiseTest, ZeroNoiseScaleNoOp) {
    auto means                = torch::rand({3, 3}, floatOpts()).contiguous();
    const auto origMeans      = means.clone();
    const auto logScales      = torch::zeros({3, 3}, floatOpts()).contiguous();
    const auto opacities      = torch::tensor({0.2f, 0.5f, 0.8f}, floatOpts());
    const auto logitOpacities = torch::log(opacities) - torch::log1p(-opacities);
    const auto quats          = torch::tensor(
        {{1.0f, 0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f}},
        floatOpts());

    fvdb::detail::ops::dispatchGaussianMCMCAddNoise<torch::kCUDA>(
        means, logScales, logitOpacities, quats, /*noiseScale=*/0.0f, 0.005, 100);

    EXPECT_TRUE(torch::allclose(means, origMeans));
}

TEST_F(GaussianMCMCAddNoiseTest, CpuAndPrivateUseNotImplemented) {
    auto means                = torch::zeros({1, 3}, fvdb::test::tensorOpts<float>(torch::kCPU));
    const auto logScales      = torch::zeros({1, 3}, fvdb::test::tensorOpts<float>(torch::kCPU));
    const auto logitOpacities = torch::zeros({1}, fvdb::test::tensorOpts<float>(torch::kCPU));
    const auto quats =
        torch::tensor({{1.0f, 0.0f, 0.0f, 0.0f}}, fvdb::test::tensorOpts<float>(torch::kCPU));

    EXPECT_THROW((fvdb::detail::ops::dispatchGaussianMCMCAddNoise<torch::kCPU>(
                     means, logScales, logitOpacities, quats, 1.0f, 0.005, 100)),
                 c10::Error);

    auto meansCuda          = means.cuda();
    auto logScalesCuda      = logScales.cuda();
    auto logitOpacitiesCuda = logitOpacities.cuda();
    auto quatsCuda          = quats.cuda();
    EXPECT_THROW((fvdb::detail::ops::dispatchGaussianMCMCAddNoise<torch::kPrivateUse1>(
                     meansCuda, logScalesCuda, logitOpacitiesCuda, quatsCuda, 1.0f, 0.005, 100)),
                 c10::Error);
}

} // namespace
