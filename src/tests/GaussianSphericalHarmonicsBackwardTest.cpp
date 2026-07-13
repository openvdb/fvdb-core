// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include "utils/Tensor.h"

#include <fvdb/detail/ops/gsplat/EvaluateSphericalHarmonicsBackward.h>

#include <torch/types.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdlib>

struct TestParams {
    float azimuth;
    float elevation;
    int64_t shDegreeToUse;
    int64_t numGaussians;
    int64_t numChannels;
    int64_t numCameras;
    bool setZeroRadii;
    bool noRadii;
};

struct SphericalHarmonincsBackwardTestFixture : public ::testing::TestWithParam<TestParams> {
    void
    SetUp() override {
        TestParams testParams = GetParam();
        float azimuth         = testParams.azimuth;
        float elevation       = testParams.elevation;
        shDegreeToUse         = testParams.shDegreeToUse;
        numGaussians          = testParams.numGaussians;
        numChannels           = testParams.numChannels;
        numCameras            = testParams.numCameras;
        setZeroRadii          = testParams.setZeroRadii;
        noRadii               = testParams.noRadii;

        const auto floatOptsCUDA = fvdb::test::tensorOpts<float>(torch::kCUDA);
        const auto intOptsCUDA   = fvdb::test::tensorOpts<int>(torch::kCUDA);

        const auto cosAzimuth = torch::cos(torch::full({numGaussians, 1}, azimuth, floatOptsCUDA));
        const auto sinAzimuth = torch::sin(torch::full({numGaussians, 1}, azimuth, floatOptsCUDA));
        const auto cosElevation =
            torch::cos(torch::full({numGaussians, 1}, elevation, floatOptsCUDA));
        const auto sinElevation =
            torch::sin(torch::full({numGaussians, 1}, elevation, floatOptsCUDA));

        means = torch::cat({cosAzimuth * cosElevation, sinAzimuth * cosElevation, sinElevation},
                           1); // [N, 3]
        worldToCamMatrices = torch::eye(4, floatOptsCUDA).unsqueeze(0).repeat({numCameras, 1, 1});
        cameraIds          = torch::empty({0}, intOptsCUDA);
        gaussianIds        = torch::empty({0}, intOptsCUDA);

        sh0Coeffs = torch::full({numGaussians, 1, numChannels}, 1.0f, floatOptsCUDA);

        K         = (shDegreeToUse + 1) * (shDegreeToUse + 1);
        shNCoeffs = torch::full({numGaussians, K - 1, numChannels}, 1.0f, floatOptsCUDA);

        radii = torch::full({numCameras, numGaussians, 2}, 1, intOptsCUDA);

        dLossDRenderQuantities =
            torch::full({numCameras, numGaussians, numChannels}, 1.0f, floatOptsCUDA);

        if (noRadii) {
            radii = torch::Tensor();
        }
        if (setZeroRadii) {
            setHalfOfRadiiToZero();
        }
    }

    void
    setHalfOfRadiiToZero() {
        radii         = radii.cpu();
        auto radiiAcc = radii.accessor<int, 3>();
        for (int64_t i = 0; i < radii.size(0); ++i) {
            for (int64_t j = 0; j < radii.size(1); ++j) {
                radiiAcc[i][j][0] = j % 2;
                radiiAcc[i][j][1] = j % 2;
            }
        }
        const auto intOptsCUDA = fvdb::test::tensorOpts<int>(torch::kCUDA);
        radii                  = radii.to(intOptsCUDA);
    }

    void
    checkSh(const int64_t numCameras,
            const int64_t numGaussians,
            const int64_t numChannels,
            const int64_t shDegreeToUse,
            const bool setZeroRadii = false) {
        {
            auto [dLossDSh0Coeffs, dLossDShNCoeffs, dLossDMeans, dLossDWorldToCamMatrices] =
                fvdb::detail::ops::evaluateSphericalHarmonicsBwd(shDegreeToUse,
                                                                 numCameras,
                                                                 numGaussians,
                                                                 means,
                                                                 worldToCamMatrices,
                                                                 cameraIds,
                                                                 gaussianIds,
                                                                 shNCoeffs,
                                                                 dLossDRenderQuantities,
                                                                 radii,
                                                                 true,
                                                                 true);

            if (setZeroRadii) {
                const auto dLdSh0Slice = dLossDSh0Coeffs.index({torch::indexing::Slice(0, -1, 2),
                                                                torch::indexing::Slice(),
                                                                torch::indexing::Slice()});
                const auto dLdShNSlice = dLossDShNCoeffs.index({torch::indexing::Slice(0, -1, 2),
                                                                torch::indexing::Slice(),
                                                                torch::indexing::Slice()});
                const auto dLdMeansSlice =
                    dLossDMeans.index({torch::indexing::Slice(0, -1, 2), torch::indexing::Slice()});

                EXPECT_TRUE(torch::allclose(dLdSh0Slice, torch::zeros_like(dLdSh0Slice)));
                EXPECT_TRUE(torch::allclose(dLdShNSlice, torch::zeros_like(dLdShNSlice)));
                EXPECT_TRUE(torch::allclose(dLdMeansSlice, torch::zeros_like(dLdMeansSlice)));
            }
            EXPECT_TRUE(dLossDSh0Coeffs.sizes() == sh0Coeffs.sizes());
            EXPECT_TRUE(dLossDShNCoeffs.sizes() == shNCoeffs.sizes());
            EXPECT_TRUE(dLossDMeans.sizes() == means.sizes());
            EXPECT_TRUE(dLossDWorldToCamMatrices.sizes() == worldToCamMatrices.sizes());
        }

        // We don't return geometry gradients if you don't ask for them
        {
            auto [dLossDSh0Coeffs, dLossDShNCoeffs, dLossDMeans, dLossDWorldToCamMatrices] =
                fvdb::detail::ops::evaluateSphericalHarmonicsBwd(shDegreeToUse,
                                                                 numCameras,
                                                                 numGaussians,
                                                                 means,
                                                                 worldToCamMatrices,
                                                                 cameraIds,
                                                                 gaussianIds,
                                                                 shNCoeffs,
                                                                 dLossDRenderQuantities,
                                                                 radii,
                                                                 false,
                                                                 false);
            if (setZeroRadii) {
                const auto dLdSh0Slice = dLossDSh0Coeffs.index({torch::indexing::Slice(0, -1, 2),
                                                                torch::indexing::Slice(),
                                                                torch::indexing::Slice()});
                const auto dLdShNSlice = dLossDShNCoeffs.index({torch::indexing::Slice(0, -1, 2),
                                                                torch::indexing::Slice(),
                                                                torch::indexing::Slice()});

                EXPECT_TRUE(torch::allclose(dLdSh0Slice, torch::zeros_like(dLdSh0Slice)));
                EXPECT_TRUE(torch::allclose(dLdShNSlice, torch::zeros_like(dLdShNSlice)));
            }
            EXPECT_TRUE(dLossDSh0Coeffs.sizes() == sh0Coeffs.sizes());
            EXPECT_TRUE(dLossDShNCoeffs.sizes() == shNCoeffs.sizes());
            EXPECT_FALSE(dLossDMeans.defined());
            EXPECT_FALSE(dLossDWorldToCamMatrices.defined());
        }
    }

    void
    checkOnlySh0(const int64_t numCameras,
                 const int64_t numGaussians,
                 const int64_t numChannels,
                 const int64_t shDegreeToUse,
                 bool setZeroRadii = false) {
        const auto floatOptsCUDA = fvdb::test::tensorOpts<float>(torch::kCUDA);

        if (setZeroRadii) {
            setHalfOfRadiiToZero();
        }
        torch::Tensor expectedDLossDSh0Coeffs =
            torch::full({numGaussians, 1, numChannels}, 0.282095f * numCameras, floatOptsCUDA);
        if (setZeroRadii) {
            expectedDLossDSh0Coeffs.index_put_({torch::indexing::Slice(0, -1, 2),
                                                torch::indexing::Slice(),
                                                torch::indexing::Slice()},
                                               0.0f);
        }

        const auto expectedSh0Sizes = std::vector({numGaussians, int64_t(1), numChannels});

        {
            auto [dLossDSh0Coeffs, dLossDShNCoeffs, dLossDMeans, dLossDWorldToCamMatrices] =
                fvdb::detail::ops::evaluateSphericalHarmonicsBwd(shDegreeToUse,
                                                                 numCameras,
                                                                 numGaussians,
                                                                 means,
                                                                 worldToCamMatrices,
                                                                 cameraIds,
                                                                 gaussianIds,
                                                                 shNCoeffs,
                                                                 dLossDRenderQuantities,
                                                                 radii,
                                                                 true,
                                                                 true);
            EXPECT_TRUE(dLossDSh0Coeffs.sizes() == expectedSh0Sizes);
            EXPECT_FALSE(dLossDShNCoeffs.defined());
            EXPECT_FALSE(dLossDMeans.defined());
            EXPECT_FALSE(dLossDWorldToCamMatrices.defined());
            EXPECT_TRUE(torch::allclose(dLossDSh0Coeffs, expectedDLossDSh0Coeffs));
        }

        // You can pass in an empty tensor for shNCoeffs and we return an empty tensor for the
        // gradient of shN, means, and worldToCamMatrices
        {
            shNCoeffs = torch::Tensor();
            auto [dLossDSh0Coeffs, dLossDShNCoeffs, dLossDMeans, dLossDWorldToCamMatrices] =
                fvdb::detail::ops::evaluateSphericalHarmonicsBwd(shDegreeToUse,
                                                                 numCameras,
                                                                 numGaussians,
                                                                 means,
                                                                 worldToCamMatrices,
                                                                 cameraIds,
                                                                 gaussianIds,
                                                                 shNCoeffs,
                                                                 dLossDRenderQuantities,
                                                                 radii,
                                                                 true,
                                                                 true);
            EXPECT_TRUE(dLossDSh0Coeffs.sizes() == expectedSh0Sizes);
            EXPECT_FALSE(dLossDShNCoeffs.defined());
            EXPECT_FALSE(dLossDMeans.defined());
            EXPECT_FALSE(dLossDWorldToCamMatrices.defined());
            EXPECT_TRUE(torch::allclose(dLossDSh0Coeffs, expectedDLossDSh0Coeffs));
        }

        // You can pass in empty tensors for shNCoeffs, means, and worldToCamMatrices and we return
        // empty tensors for their gradients
        {
            shNCoeffs          = torch::Tensor();
            means              = torch::Tensor();
            worldToCamMatrices = torch::Tensor();
            auto [dLossDSh0Coeffs, dLossDShNCoeffs, dLossDMeans, dLossDWorldToCamMatrices] =
                fvdb::detail::ops::evaluateSphericalHarmonicsBwd(shDegreeToUse,
                                                                 numCameras,
                                                                 numGaussians,
                                                                 means,
                                                                 worldToCamMatrices,
                                                                 cameraIds,
                                                                 gaussianIds,
                                                                 shNCoeffs,
                                                                 dLossDRenderQuantities,
                                                                 radii,
                                                                 true,
                                                                 true);
            EXPECT_TRUE(dLossDSh0Coeffs.sizes() == expectedSh0Sizes);
            EXPECT_FALSE(dLossDShNCoeffs.defined());
            EXPECT_FALSE(dLossDMeans.defined());
            EXPECT_FALSE(dLossDWorldToCamMatrices.defined());
            EXPECT_TRUE(torch::allclose(dLossDSh0Coeffs, expectedDLossDSh0Coeffs));
        }
    }

    torch::Tensor sh0Coeffs;
    torch::Tensor shNCoeffs;
    torch::Tensor means;
    torch::Tensor worldToCamMatrices;
    torch::Tensor cameraIds;
    torch::Tensor gaussianIds;
    torch::Tensor radii;
    torch::Tensor dLossDRenderQuantities;
    int64_t K;

    int64_t numCameras;
    int64_t numChannels;
    int64_t numGaussians;
    int64_t shDegreeToUse;
    bool setZeroRadii;
    bool noRadii;
};

TEST_P(SphericalHarmonincsBackwardTestFixture, TestShBackward) {
    if (shDegreeToUse == 0) {
        checkOnlySh0(numCameras, numGaussians, numChannels, shDegreeToUse, setZeroRadii);
    } else {
        checkSh(numCameras, numGaussians, numChannels, shDegreeToUse, setZeroRadii);
    }
}

TEST(GaussianSphericalHarmonicsBackwardTest, TestPackedModeScattersGeometryGradients) {
    const auto floatOptsCUDA = fvdb::test::tensorOpts<float>(torch::kCUDA);
    const auto intOptsCUDA   = fvdb::test::tensorOpts<int>(torch::kCUDA);

    const auto means =
        torch::tensor({{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}}, floatOptsCUDA);
    auto worldToCamMatrices = torch::eye(4, floatOptsCUDA).unsqueeze(0).repeat({3, 1, 1});
    worldToCamMatrices.index_put_({0, 0, 3}, 0.25f);
    worldToCamMatrices.index_put_({2, 1, 3}, -0.5f);

    const auto cameraIds              = torch::tensor({2, 0}, intOptsCUDA);
    const auto gaussianIds            = torch::tensor({2, 0}, intOptsCUDA);
    const auto shNCoeffs              = torch::ones({2, 3, 1}, floatOptsCUDA);
    const auto dLossDRenderQuantities = torch::ones({1, 2, 1}, floatOptsCUDA);
    const auto radii                  = torch::ones({1, 2, 2}, intOptsCUDA);

    auto [dLossDSh0Coeffs, dLossDShNCoeffs, dLossDMeans, dLossDWorldToCamMatrices] =
        fvdb::detail::ops::evaluateSphericalHarmonicsBwd(1,
                                                         1,
                                                         2,
                                                         means,
                                                         worldToCamMatrices,
                                                         cameraIds,
                                                         gaussianIds,
                                                         shNCoeffs,
                                                         dLossDRenderQuantities,
                                                         radii,
                                                         true,
                                                         true);

    EXPECT_TRUE(dLossDSh0Coeffs.sizes() == std::vector<int64_t>({2, 1, 1}));
    EXPECT_TRUE(dLossDShNCoeffs.sizes() == shNCoeffs.sizes());
    EXPECT_TRUE(dLossDMeans.sizes() == means.sizes());
    EXPECT_TRUE(dLossDWorldToCamMatrices.sizes() == worldToCamMatrices.sizes());
    EXPECT_TRUE(torch::allclose(dLossDMeans.index({1}), torch::zeros({3}, floatOptsCUDA)));
    EXPECT_TRUE(
        torch::allclose(dLossDWorldToCamMatrices.index({1}), torch::zeros({4, 4}, floatOptsCUDA)));
}

#undef DEBUG_BENCHMARK
#ifdef DEBUG_BENCHMARK
TEST_F(SphericalHarmonincsBackwardTestFixture, BenchmarkSh0) {
    const float azimuth         = 0.0f;
    const float elevation       = 0.0f;
    const int64_t shDegreeToUse = 0;
    const int64_t numGaussians  = 6128356;
    const int64_t numChannels   = 3;
    const int64_t numCameras    = 4;

    initInputs(azimuth, elevation, shDegreeToUse, numCameras, numGaussians, numChannels);

    const auto floatOptsCUDA = fvdb::test::tensorOpts<float>(torch::kCUDA);

    for (int i = 0; i < 10; i += 1) {
        torch::cuda::synchronize();
        auto [dLossDSh0Coeffs, dLossDShNCoeffs, dLossDMeans, dLossDWorldToCamMatrices] =
            fvdb::detail::ops::evaluateSphericalHarmonicsBwd(shDegreeToUse,
                                                             numCameras,
                                                             numGaussians,
                                                             means,
                                                             worldToCamMatrices,
                                                             cameraIds,
                                                             gaussianIds,
                                                             shNCoeffs,
                                                             dLossDRenderQuantities,
                                                             radii,
                                                             false,
                                                             false);
        torch::cuda::synchronize();
    }

    const int totalIters = 1000;
    int64_t totalTime    = 0;
    for (int i = 0; i < totalIters; i += 1) {
        torch::cuda::synchronize();
        auto start = std::chrono::high_resolution_clock::now();
        auto [dLossDSh0Coeffs, dLossDShNCoeffs, dLossDMeans, dLossDWorldToCamMatrices] =
            fvdb::detail::ops::evaluateSphericalHarmonicsBwd(shDegreeToUse,
                                                             numCameras,
                                                             numGaussians,
                                                             means,
                                                             worldToCamMatrices,
                                                             cameraIds,
                                                             gaussianIds,
                                                             shNCoeffs,
                                                             dLossDRenderQuantities,
                                                             radii,
                                                             false,
                                                             false);
        torch::cuda::synchronize();
        auto end      = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        totalTime += duration.count();
    }

    std::cerr << "Avg for deg-0 Spherical Harmonics Backward with no geometry grad (over "
              << totalIters << " iters): " << (double(totalTime) / double(totalIters)) << " ms"
              << std::endl;
}

TEST_F(SphericalHarmonincsBackwardTestFixture, BenchmarkSh0WithGeometryGrad) {
    const float azimuth         = 0.0f;
    const float elevation       = 0.0f;
    const int64_t shDegreeToUse = 0;
    const int64_t numGaussians  = 6128356;
    const int64_t numChannels   = 3;
    const int64_t numCameras    = 4;

    initInputs(azimuth, elevation, shDegreeToUse, numCameras, numGaussians, numChannels);

    const auto floatOptsCUDA = fvdb::test::tensorOpts<float>(torch::kCUDA);

    for (int i = 0; i < 10; i += 1) {
        torch::cuda::synchronize();
        auto [dLossDSh0Coeffs, dLossDShNCoeffs, dLossDMeans, dLossDWorldToCamMatrices] =
            fvdb::detail::ops::evaluateSphericalHarmonicsBwd(shDegreeToUse,
                                                             numCameras,
                                                             numGaussians,
                                                             means,
                                                             worldToCamMatrices,
                                                             cameraIds,
                                                             gaussianIds,
                                                             shNCoeffs,
                                                             dLossDRenderQuantities,
                                                             radii,
                                                             false,
                                                             false);
        torch::cuda::synchronize();
    }

    const int totalIters = 1000;
    int64_t totalTime    = 0;
    for (int i = 0; i < totalIters; i += 1) {
        torch::cuda::synchronize();
        auto start = std::chrono::high_resolution_clock::now();
        auto [dLossDSh0Coeffs, dLossDShNCoeffs, dLossDMeans, dLossDWorldToCamMatrices] =
            fvdb::detail::ops::evaluateSphericalHarmonicsBwd(shDegreeToUse,
                                                             numCameras,
                                                             numGaussians,
                                                             means,
                                                             worldToCamMatrices,
                                                             cameraIds,
                                                             gaussianIds,
                                                             shNCoeffs,
                                                             dLossDRenderQuantities,
                                                             radii,
                                                             true,
                                                             true);
        torch::cuda::synchronize();
        auto end      = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        totalTime += duration.count();
    }

    std::cerr << "Avg for deg-0 Spherical Harmonics Backward with geometry grad (over "
              << totalIters << " iters): " << (double(totalTime) / double(totalIters)) << " ms"
              << std::endl;
}

TEST_F(SphericalHarmonincsBackwardTestFixture, BenchmarkShNWithGeometryGrad) {
    const float azimuth         = 0.0f;
    const float elevation       = 0.0f;
    const int64_t shDegreeToUse = 4;
    const int64_t numGaussians  = 6128356;
    const int64_t numChannels   = 3;
    const int64_t numCameras    = 4;

    initInputs(azimuth, elevation, shDegreeToUse, numCameras, numGaussians, numChannels);

    const auto floatOptsCUDA = fvdb::test::tensorOpts<float>(torch::kCUDA);

    for (int i = 0; i < 10; i += 1) {
        torch::cuda::synchronize();
        auto [dLossDSh0Coeffs, dLossDShNCoeffs, dLossDMeans, dLossDWorldToCamMatrices] =
            fvdb::detail::ops::evaluateSphericalHarmonicsBwd(shDegreeToUse,
                                                             numCameras,
                                                             numGaussians,
                                                             means,
                                                             worldToCamMatrices,
                                                             cameraIds,
                                                             gaussianIds,
                                                             shNCoeffs,
                                                             dLossDRenderQuantities,
                                                             radii,
                                                             false,
                                                             false);
        torch::cuda::synchronize();
    }

    const int totalIters = 1000;
    int64_t totalTime    = 0;
    for (int i = 0; i < totalIters; i += 1) {
        torch::cuda::synchronize();
        auto start = std::chrono::high_resolution_clock::now();
        auto [dLossDSh0Coeffs, dLossDShNCoeffs, dLossDMeans, dLossDWorldToCamMatrices] =
            fvdb::detail::ops::evaluateSphericalHarmonicsBwd(shDegreeToUse,
                                                             numCameras,
                                                             numGaussians,
                                                             means,
                                                             worldToCamMatrices,
                                                             cameraIds,
                                                             gaussianIds,
                                                             shNCoeffs,
                                                             dLossDRenderQuantities,
                                                             radii,
                                                             true,
                                                             true);
        torch::cuda::synchronize();
        auto end      = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        totalTime += duration.count();
    }

    std::cerr << "Avg for deg-N Spherical Harmonics Backward with geometry grad (over "
              << totalIters << " iters): " << (double(totalTime) / double(totalIters)) << " ms"
              << std::endl;
}

TEST_F(SphericalHarmonincsBackwardTestFixture, BenchmarkShNWithoutGeometryGrad) {
    const float azimuth         = 0.0f;
    const float elevation       = 0.0f;
    const int64_t shDegreeToUse = 4;
    const int64_t numGaussians  = 6128356;
    const int64_t numChannels   = 3;
    const int64_t numCameras    = 4;

    initInputs(azimuth, elevation, shDegreeToUse, numCameras, numGaussians, numChannels);

    const auto floatOptsCUDA = fvdb::test::tensorOpts<float>(torch::kCUDA);

    for (int i = 0; i < 10; i += 1) {
        torch::cuda::synchronize();
        auto [dLossDSh0Coeffs, dLossDShNCoeffs, dLossDMeans, dLossDWorldToCamMatrices] =
            fvdb::detail::ops::evaluateSphericalHarmonicsBwd(shDegreeToUse,
                                                             numCameras,
                                                             numGaussians,
                                                             means,
                                                             worldToCamMatrices,
                                                             cameraIds,
                                                             gaussianIds,
                                                             shNCoeffs,
                                                             dLossDRenderQuantities,
                                                             radii,
                                                             false,
                                                             false);
        torch::cuda::synchronize();
    }

    const int totalIters = 1000;
    int64_t totalTime    = 0;
    for (int i = 0; i < totalIters; i += 1) {
        torch::cuda::synchronize();
        auto start = std::chrono::high_resolution_clock::now();
        auto [dLossDSh0Coeffs, dLossDShNCoeffs, dLossDMeans, dLossDWorldToCamMatrices] =
            fvdb::detail::ops::evaluateSphericalHarmonicsBwd(shDegreeToUse,
                                                             numCameras,
                                                             numGaussians,
                                                             means,
                                                             worldToCamMatrices,
                                                             cameraIds,
                                                             gaussianIds,
                                                             shNCoeffs,
                                                             dLossDRenderQuantities,
                                                             radii,
                                                             false,
                                                             false);
        torch::cuda::synchronize();
        auto end      = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        totalTime += duration.count();
    }

    std::cerr << "Avg for deg-N Spherical Harmonics Backward with no geometry grad (over "
              << totalIters << " iters): " << (double(totalTime) / double(totalIters)) << " ms"
              << std::endl;
}
#endif

INSTANTIATE_TEST_SUITE_P(ShBackwardTests,
                         SphericalHarmonincsBackwardTestFixture,
                         ::testing::Values(TestParams{0.0f, 0.0f, 0, 0, 3, 1, false, false},
                                           TestParams{0.0f, 0.0f, 0, 1, 3, 1, false, false},
                                           TestParams{0.0f, 0.0f, 0, 10, 3, 1, false, false},
                                           TestParams{0.0f, 0.0f, 0, 10, 3, 1, false, true},
                                           TestParams{0.0f, 0.0f, 0, 10, 3, 1, true, false},

                                           TestParams{0.0f, 0.0f, 0, 0, 8, 1, false, false},
                                           TestParams{0.0f, 0.0f, 0, 1, 8, 1, false, false},
                                           TestParams{0.0f, 0.0f, 0, 10, 8, 1, false, false},
                                           TestParams{0.0f, 0.0f, 0, 10, 8, 1, false, true},
                                           TestParams{0.0f, 0.0f, 0, 10, 8, 1, true, false},

                                           TestParams{0.0f, 0.0f, 0, 0, 3, 2, false, false},
                                           TestParams{0.0f, 0.0f, 0, 1, 3, 2, false, false},
                                           TestParams{0.0f, 0.0f, 0, 10, 3, 2, false, false},
                                           TestParams{0.0f, 0.0f, 0, 10, 3, 2, false, true},
                                           TestParams{0.0f, 0.0f, 0, 10, 3, 2, true, false},

                                           TestParams{0.0f, 0.0f, 0, 0, 7, 2, false, false},
                                           TestParams{0.0f, 0.0f, 0, 1, 7, 2, false, false},
                                           TestParams{0.0f, 0.0f, 0, 10, 7, 2, false, false},
                                           TestParams{0.0f, 0.0f, 0, 10, 7, 2, false, true},
                                           TestParams{0.0f, 0.0f, 0, 10, 7, 2, true, false},

                                           TestParams{0.0f, 0.0f, 4, 0, 3, 1, false, false},
                                           TestParams{0.0f, 0.0f, 4, 1, 3, 1, false, false},
                                           TestParams{0.0f, 0.0f, 4, 10, 3, 1, false, false},
                                           TestParams{0.0f, 0.0f, 4, 10, 3, 1, false, true},
                                           TestParams{0.0f, 0.0f, 4, 10, 3, 1, true, false},

                                           TestParams{0.0f, 0.0f, 4, 0, 8, 1, false, false},
                                           TestParams{0.0f, 0.0f, 4, 1, 8, 1, false, false},
                                           TestParams{0.0f, 0.0f, 4, 10, 8, 1, false, false},
                                           TestParams{0.0f, 0.0f, 4, 10, 8, 1, false, true},
                                           TestParams{0.0f, 0.0f, 4, 10, 8, 1, true, false},

                                           TestParams{0.0f, 0.0f, 4, 0, 3, 2, false, false},
                                           TestParams{0.0f, 0.0f, 4, 1, 3, 2, false, false},
                                           TestParams{0.0f, 0.0f, 4, 10, 3, 2, false, false},
                                           TestParams{0.0f, 0.0f, 4, 10, 3, 2, false, true},
                                           TestParams{0.0f, 0.0f, 4, 10, 3, 2, true, false},

                                           TestParams{0.0f, 0.0f, 4, 0, 7, 2, false, false},
                                           TestParams{0.0f, 0.0f, 4, 1, 7, 2, false, false},
                                           TestParams{0.0f, 0.0f, 4, 10, 7, 2, false, false},
                                           TestParams{0.0f, 0.0f, 4, 10, 7, 2, false, true},
                                           TestParams{0.0f, 0.0f, 4, 10, 7, 2, true, false}));
