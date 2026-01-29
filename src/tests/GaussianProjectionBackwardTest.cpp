// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include "utils/Tensor.h"

#include <fvdb/detail/ops/gsplat/GaussianProjectionBackward.h>
#include <fvdb/detail/ops/gsplat/GaussianProjectionForward.h>

#include <torch/script.h>
#include <torch/types.h>

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

#include <cstdlib>

#ifndef FVDB_EXTERNAL_TEST_DATA_PATH
#error "FVDB_EXTERNAL_TEST_DATA_PATH must be defined"
#endif

struct GaussianProjectionBackwardTestFixture : public ::testing::Test {
    // return rtol and atol for the current device
    static std::pair<double, double>
    tolerances() {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, dev);
        // Loosen tolerances on Blackwell (sm_100+) where atomic/reduction order differs slightly
        if (prop.major >= 10) {
            return {5e-4, 6e-3};
        }
        return {1e-4, 1e-3};
    }
    void
    loadInputData(const std::string insPath) {
        const auto dataPath =
            std::string(FVDB_EXTERNAL_TEST_DATA_PATH) + std::string("/unit_tests/gsplat");
        const auto inputsPath = dataPath + std::string("/") + insPath;

        auto inputs = fvdb::test::loadTensors(inputsPath, inputNames);
        means       = inputs[0].cuda();
        quats       = inputs[1].cuda();
        scales      = inputs[2].cuda();
        viewmats    = inputs[3].cuda();
        Ks          = inputs[4].cuda();
        if (inputNames.size() > 5) {
            compensations = inputs[5].cuda();
            radii         = inputs[6].cuda();
            conics        = inputs[7].cuda();
        }

        imageWidth   = 647;
        imageHeight  = 420;
        imageOriginW = 0;
        imageOriginH = 0;
        eps2d        = 0.3;
    }

    void
    storeData(const std::string outsPath, const std::vector<torch::Tensor> &outputData) {
        const auto dataPath =
            std::string(FVDB_EXTERNAL_TEST_DATA_PATH) + std::string("/unit_tests/gsplat");
        const auto outputPath = dataPath + std::string("/") + outsPath;

        fvdb::test::storeTensors(outputPath, outputData, outputNames);
    }

    void
    loadTestData(const std::string insPath, const std::string outsPath) {
        // Set the random seed for reproducibility.
        torch::manual_seed(0);

        loadInputData(insPath);

        const auto dataPath =
            std::string(FVDB_EXTERNAL_TEST_DATA_PATH) + std::string("/unit_tests/gsplat");
        const auto expectedOutputsPath = dataPath + std::string("/") + outsPath;

        auto expectedOutputs = fvdb::test::loadTensors(expectedOutputsPath, outputNames);
        expectedDLossDMeans  = expectedOutputs[0].cuda();
        // expectedDLossDCovars      = expectedOutputs[1].cuda();
        expectedDLossDQuats                      = expectedOutputs[1].cuda();
        expectedDLossDScales                     = expectedOutputs[2].cuda();
        expectedDLossDCamToWorlds                = expectedOutputs[3].cuda();
        expectedNormalizeddLossdMeans2dNormAccum = expectedOutputs[4].cuda();
        expectedNormalizedMaxRadiiAccum          = expectedOutputs[5].cuda();
        expectedGradientStepCounts               = expectedOutputs[6].cuda();
    }

    void
    moveToDevice(const torch::Device &device) {
        means               = means.to(device);
        quats               = quats.to(device);
        scales              = scales.to(device);
        viewmats            = viewmats.to(device);
        Ks                  = Ks.to(device);
        compensations       = compensations.to(device);
        radii               = radii.to(device);
        conics              = conics.to(device);
        expectedDLossDMeans = expectedDLossDMeans.to(device);
        // expectedDLossDCovars      = expectedDLossDCovars.to(device);
        expectedDLossDQuats       = expectedDLossDQuats.to(device);
        expectedDLossDScales      = expectedDLossDScales.to(device);
        expectedDLossDCamToWorlds = expectedDLossDCamToWorlds.to(device);
        expectedNormalizeddLossdMeans2dNormAccum =
            expectedNormalizeddLossdMeans2dNormAccum.to(device);
        expectedNormalizedMaxRadiiAccum = expectedNormalizedMaxRadiiAccum.to(device);
        expectedGradientStepCounts      = expectedGradientStepCounts.to(device);
    }

    std::vector<std::string> inputNames = {
        "means",
        "quats",
        "scales",
        "viewmats",
        "Ks",    // projection inputs
        "compensations",
        "radii",
        "conics" // disable if reading projection inputs and writing backwards outputs
    };

    const std::vector<std::string> outputNames = {"dLossDMeans",
                                                  "dLossDQuats",
                                                  "dLossDScales",
                                                  "dLossDCamToWorlds",
                                                  "normalizeddLossdMeans2dNormAccum",
                                                  "normalizedMaxRadiiAccum",
                                                  "gradientStepCounts"};

    // Input tensors
    torch::Tensor means;         // [C, N, 3] or [nnz, 3]
    torch::Tensor quats;         // [C, N, 4] or [nnz, 4]
    torch::Tensor scales;        // [C, N, 3] or [nnz, 3]
    torch::Tensor viewmats;      // [C, 16] or [nnz, 16]
    torch::Tensor Ks;            // [C, 9] or [nnz, 9]
    torch::Tensor compensations; // [C, N] or [nnz]
    torch::Tensor radii;         // [C, N] or [nnz]
    torch::Tensor conics;        // [C, N, 3] or [nnz, 3]

    // Expected output tensors
    torch::Tensor expectedDLossDMeans; // [C, N, 3] or [nnz, 3]
    // torch::Tensor expectedDLossDCovars;      // [C, N, 9] or [nnz, 9]
    torch::Tensor expectedDLossDQuats;                      // [C, N, 4] or [nnz, 4]
    torch::Tensor expectedDLossDScales;                     // [C, N, 3] or [nnz, 3]
    torch::Tensor expectedDLossDCamToWorlds;                // [C, 16] or [nnz, 16]
    torch::Tensor expectedNormalizeddLossdMeans2dNormAccum; // [C]
    torch::Tensor expectedNormalizedMaxRadiiAccum;          // [C]
    torch::Tensor expectedGradientStepCounts;               // [C]

    // Parameters
    uint32_t imageWidth;
    uint32_t imageHeight;
    uint32_t imageOriginW;
    uint32_t imageOriginH;
    float eps2d;
};

// This is a helper function to generate the output data for the test cases.
// Only enable this test when you want to update the output data.
TEST_F(GaussianProjectionBackwardTestFixture, DISABLED_GenerateOutputData) {
    // Load test data using our helper method
    auto inputFilename      = std::string("projection_forward_inputs.pt");
    auto originalInputNames = inputNames;
    // remove last 3 input names
    inputNames.pop_back();
    inputNames.pop_back();
    inputNames.pop_back();
    loadInputData(inputFilename);

    {
        // Perspective projection
        const auto [radii_proj, means2d_proj, depths_proj, conics_proj, compensations_proj] =
            fvdb::detail::ops::dispatchGaussianProjectionForward<torch::kCUDA>(means,
                                                                               quats,
                                                                               scales,
                                                                               viewmats,
                                                                               Ks,
                                                                               imageWidth,
                                                                               imageHeight,
                                                                               0.3,
                                                                               1e-2,
                                                                               1e10,
                                                                               0,
                                                                               true,
                                                                               false);

        const auto C = radii_proj.size(0);
        const auto N = radii_proj.size(1);

        // store the input data for the backward pass
        auto backwardInputFilename = std::string("projection_persp_backward_inputs.pt");
        storeData(
            backwardInputFilename,
            {means, quats, scales, viewmats, Ks, compensations_proj, radii_proj, conics_proj});

        // run backwards pass and store outputs
        torch::Tensor dLossDMeans2d       = torch::full({C, N, 2}, 0.1, torch::kCUDA);
        torch::Tensor dLossDDepths        = torch::full({C, N}, 0.2, torch::kCUDA);
        torch::Tensor dLossDConics        = torch::full({C, N, 3}, 0.3, torch::kCUDA);
        torch::Tensor dLossDCompensations = torch::full({C, N}, 0.4, torch::kCUDA);

        torch::TensorOptions options = torch::kCUDA;
        auto outNormalizeddLossdMeans2dNormAccum =
            torch::zeros({N}, options.dtype(torch::kFloat32));
        auto outNormalizedMaxRadiiAccum = torch::zeros({N}, options.dtype(torch::kInt32));
        auto outGradientStepCounts      = torch::zeros({N}, options.dtype(torch::kInt32));

        const auto [dLossDMeans, dLossDCovars, dLossDQuats, dLossDScales, dLossDCamToWorlds] =
            fvdb::detail::ops::dispatchGaussianProjectionBackward<torch::kCUDA>(
                means,
                quats,
                scales,
                viewmats,
                Ks,
                compensations_proj,
                imageWidth,
                imageHeight,
                eps2d,
                radii_proj,
                conics_proj,
                dLossDMeans2d,
                dLossDDepths,
                dLossDConics,
                dLossDCompensations,
                true,
                false,
                outNormalizeddLossdMeans2dNormAccum,
                outNormalizedMaxRadiiAccum,
                outGradientStepCounts);

        std::vector<torch::Tensor> outputData = {
            dLossDMeans,
            // dLossDCovars, Currently dLossDCovars is not output, not exposed, see
            // dispatchGaussianProjectionBackward
            dLossDQuats,
            dLossDScales,
            dLossDCamToWorlds,
            outNormalizeddLossdMeans2dNormAccum,
            outNormalizedMaxRadiiAccum,
            outGradientStepCounts};
        auto outputFilename = std::string("projection_persp_backward_outputs.pt");

        storeData(outputFilename, outputData);
    }

    {
        // Orthographic projection
        const auto [radii_proj, means2d_proj, depths_proj, conics_proj, compensations_proj] =
            fvdb::detail::ops::dispatchGaussianProjectionForward<torch::kCUDA>(means,
                                                                               quats,
                                                                               scales,
                                                                               viewmats,
                                                                               Ks,
                                                                               imageWidth,
                                                                               imageHeight,
                                                                               0.3,
                                                                               1e-2,
                                                                               1e10,
                                                                               0,
                                                                               true,
                                                                               true);

        const auto C = radii_proj.size(0);
        const auto N = radii_proj.size(1);

        // store the input data for the backward pass
        auto backwardInputFilename = std::string("projection_ortho_backward_inputs.pt");
        storeData(
            backwardInputFilename,
            {means, quats, scales, viewmats, Ks, compensations_proj, radii_proj, conics_proj});

        // run backwards pass and store outputs
        torch::Tensor dLossDMeans2d       = torch::full({C, N, 2}, 0.1, torch::kCUDA);
        torch::Tensor dLossDDepths        = torch::full({C, N}, 0.2, torch::kCUDA);
        torch::Tensor dLossDConics        = torch::full({C, N, 3}, 0.3, torch::kCUDA);
        torch::Tensor dLossDCompensations = torch::full({C, N}, 0.4, torch::kCUDA);

        torch::TensorOptions options = torch::kCUDA;
        auto outNormalizeddLossdMeans2dNormAccum =
            torch::zeros({N}, options.dtype(torch::kFloat32));
        auto outNormalizedMaxRadiiAccum = torch::zeros({N}, options.dtype(torch::kInt32));
        auto outGradientStepCounts      = torch::zeros({N}, options.dtype(torch::kInt32));

        const auto [dLossDMeans, dLossDCovars, dLossDQuats, dLossDScales, dLossDCamToWorlds] =
            fvdb::detail::ops::dispatchGaussianProjectionBackward<torch::kCUDA>(
                means,
                quats,
                scales,
                viewmats,
                Ks,
                compensations_proj,
                imageWidth,
                imageHeight,
                eps2d,
                radii_proj,
                conics_proj,
                dLossDMeans2d,
                dLossDDepths,
                dLossDConics,
                dLossDCompensations,
                true,
                true,
                outNormalizeddLossdMeans2dNormAccum,
                outNormalizedMaxRadiiAccum,
                outGradientStepCounts);

        std::vector<torch::Tensor> outputData = {
            dLossDMeans,
            // dLossDCovars, Currently dLossDCovars is not output, not exposed, see
            // dispatchGaussianProjectionBackward
            dLossDQuats,
            dLossDScales,
            dLossDCamToWorlds,
            outNormalizeddLossdMeans2dNormAccum,
            outNormalizedMaxRadiiAccum,
            outGradientStepCounts};
        auto outputFilename = std::string("projection_ortho_backward_outputs.pt");

        storeData(outputFilename, outputData);
    }

    // restore input names
    inputNames = originalInputNames;
}

TEST_F(GaussianProjectionBackwardTestFixture, TestPerspectiveProjection) {
    loadTestData("projection_persp_backward_inputs.pt", "projection_persp_backward_outputs.pt");

    const auto C                      = radii.size(0);
    const auto N                      = radii.size(1);
    torch::Tensor dLossDMeans2d       = torch::full({C, N, 2}, 0.1, torch::kCUDA);
    torch::Tensor dLossDDepths        = torch::full({C, N}, 0.2, torch::kCUDA);
    torch::Tensor dLossDConics        = torch::full({C, N, 3}, 0.3, torch::kCUDA);
    torch::Tensor dLossDCompensations = torch::full({C, N}, 0.4, torch::kCUDA);

    torch::TensorOptions options             = torch::kCUDA;
    auto outNormalizeddLossdMeans2dNormAccum = torch::zeros({N}, options.dtype(torch::kFloat32));
    auto outNormalizedMaxRadiiAccum          = torch::zeros({N}, options.dtype(torch::kInt32));
    auto outGradientStepCounts               = torch::zeros({N}, options.dtype(torch::kInt32));

    const auto [dLossDMeans, dLossDCovars, dLossDQuats, dLossDScales, dLossDCamToWorlds] =
        fvdb::detail::ops::dispatchGaussianProjectionBackward<torch::kCUDA>(
            means,
            quats,
            torch::log(scales),
            viewmats,
            Ks,
            compensations,
            imageWidth,
            imageHeight,
            eps2d,
            radii,
            conics,
            dLossDMeans2d,
            dLossDDepths,
            dLossDConics,
            dLossDCompensations,
            true,
            false,
            outNormalizeddLossdMeans2dNormAccum,
            outNormalizedMaxRadiiAccum,
            outGradientStepCounts);

    auto [rtol, atol] = tolerances();
#if 0
    std::cerr << "Max diff between dLossDMeans "
              << (dLossDMeans - expectedDLossDMeans).abs().max().item<float>() << "\n";
    std::cerr << "Range of dLossDMeans " << dLossDMeans.min().item<float>() << " to "
              << dLossDMeans.max().item<float>() << "\n";
    std::cerr << "Max diff between dLossDQuats "
              << (dLossDQuats - expectedDLossDQuats).abs().max().item<float>() << "\n";
    std::cerr << "Range of dLossDQuats " << dLossDQuats.min().item<float>() << " to "
              << dLossDQuats.max().item<float>() << "\n";
    std::cerr << "Max diff between dLossDScales "
              << (dLossDScales - expectedDLossDScales).abs().max().item<float>() << "\n";
    std::cerr << "Range of dLossDScales " << dLossDScales.min().item<float>() << " to "
              << dLossDScales.max().item<float>() << "\n";
    std::cerr << "Max diff between dLossDCamToWorlds "
              << (dLossDCamToWorlds - expectedDLossDCamToWorlds).abs().max().item<float>() << "\n";
    std::cerr << "Max diff between outNormalizeddLossdMeans2dNormAccum "
              << (outNormalizeddLossdMeans2dNormAccum - expectedNormalizeddLossdMeans2dNormAccum)
                     .abs()
                     .max()
                     .item<float>()
              << "\n";
    std::cerr
        << "Max diff between outNormalizedMaxRadiiAccum "
        << (outNormalizedMaxRadiiAccum - expectedNormalizedMaxRadiiAccum).abs().max().item<float>()
        << "\n";
    std::cerr << "Max diff between outGradientStepCounts "
              << (outGradientStepCounts - expectedGradientStepCounts).abs().max().item<float>()
              << "\n";
#endif
    EXPECT_TRUE(torch::allclose(dLossDMeans, expectedDLossDMeans, rtol, atol));
    EXPECT_TRUE(torch::allclose(dLossDQuats, expectedDLossDQuats, rtol, atol));
    EXPECT_TRUE(torch::allclose(dLossDScales, expectedDLossDScales, rtol, atol));
    EXPECT_TRUE(torch::allclose(dLossDCamToWorlds, expectedDLossDCamToWorlds, rtol, atol));
    EXPECT_TRUE(torch::allclose(
        outNormalizeddLossdMeans2dNormAccum, expectedNormalizeddLossdMeans2dNormAccum, rtol, atol));
    EXPECT_TRUE(torch::allclose(outNormalizedMaxRadiiAccum, expectedNormalizedMaxRadiiAccum));
    EXPECT_TRUE(torch::allclose(outGradientStepCounts, expectedGradientStepCounts));
}

TEST_F(GaussianProjectionBackwardTestFixture, TestOrthographicProjection) {
    loadTestData("projection_ortho_backward_inputs.pt", "projection_ortho_backward_outputs.pt");

    const auto C                      = radii.size(0);
    const auto N                      = radii.size(1);
    torch::Tensor dLossDMeans2d       = torch::full({C, N, 2}, 0.1, torch::kCUDA);
    torch::Tensor dLossDDepths        = torch::full({C, N}, 0.2, torch::kCUDA);
    torch::Tensor dLossDConics        = torch::full({C, N, 3}, 0.3, torch::kCUDA);
    torch::Tensor dLossDCompensations = torch::full({C, N}, 0.4, torch::kCUDA);

    torch::TensorOptions options             = torch::kCUDA;
    auto outNormalizeddLossdMeans2dNormAccum = torch::zeros({N}, options.dtype(torch::kFloat32));
    auto outNormalizedMaxRadiiAccum          = torch::zeros({N}, options.dtype(torch::kInt32));
    auto outGradientStepCounts               = torch::zeros({N}, options.dtype(torch::kInt32));

    const auto [dLossDMeans, dLossDCovars, dLossDQuats, dLossDScales, dLossDCamToWorlds] =
        fvdb::detail::ops::dispatchGaussianProjectionBackward<torch::kCUDA>(
            means,
            quats,
            torch::log(scales),
            viewmats,
            Ks,
            compensations,
            imageWidth,
            imageHeight,
            eps2d,
            radii,
            conics,
            dLossDMeans2d,
            dLossDDepths,
            dLossDConics,
            dLossDCompensations,
            true,
            true,
            outNormalizeddLossdMeans2dNormAccum,
            outNormalizedMaxRadiiAccum,
            outGradientStepCounts);

    auto tol2 = tolerances();
#if 0
    std::cerr << "Max diff between dLossDMeans "
              << (dLossDMeans - expectedDLossDMeans).abs().max().item<float>() << "\n";
    std::cerr << "Range of dLossDMeans " << dLossDMeans.min().item<float>() << " to "
              << dLossDMeans.max().item<float>() << "\n";
    std::cerr << "Max diff between dLossDQuats "
              << (dLossDQuats - expectedDLossDQuats).abs().max().item<float>() << "\n";
    std::cerr << "Range of dLossDQuats " << dLossDQuats.min().item<float>() << " to "
              << dLossDQuats.max().item<float>() << "\n";
    std::cerr << "Max diff between dLossDScales "
              << (dLossDScales - expectedDLossDScales).abs().max().item<float>() << "\n";
    std::cerr << "Range of dLossDScales " << dLossDScales.min().item<float>() << " to "
              << dLossDScales.max().item<float>() << "\n";
    std::cerr << "Max diff between dLossDCamToWorlds "
              << (dLossDCamToWorlds - expectedDLossDCamToWorlds).abs().max().item<float>() << "\n";
    std::cerr << "Max diff between outNormalizeddLossdMeans2dNormAccum "
              << (outNormalizeddLossdMeans2dNormAccum - expectedNormalizeddLossdMeans2dNormAccum)
                     .abs()
                     .max()
                     .item<float>()
              << "\n";
    std::cerr
        << "Max diff between outNormalizedMaxRadiiAccum "
        << (outNormalizedMaxRadiiAccum - expectedNormalizedMaxRadiiAccum).abs().max().item<float>()
        << "\n";
    std::cerr << "Max diff between outGradientStepCounts "
              << (outGradientStepCounts - expectedGradientStepCounts).abs().max().item<float>()
              << "\n";
#endif
    double rtol = tol2.first;
    double atol = tol2.second;
    EXPECT_TRUE(torch::allclose(dLossDMeans, expectedDLossDMeans, rtol, atol));
    // EXPECT_TRUE(torch::allclose(dLossDCovars, expectedDLossDCovars, rtol, atol));
    // We use fairly large atol here since the original data was generated using scales not
    // log(scales) so there is some numerical drift.
    // The range of dLossDQuats is around [-20, 20] so atol=0.005 is reasonable
    // The range of dLossDScales is around [-2, 0.1] so atol=0.05 is very reasonable
    EXPECT_TRUE(torch::allclose(dLossDQuats, expectedDLossDQuats, rtol, 0.005));
    EXPECT_TRUE(torch::allclose(dLossDScales, expectedDLossDScales, rtol, 0.05));
    EXPECT_TRUE(torch::allclose(dLossDCamToWorlds, expectedDLossDCamToWorlds, rtol, atol));
    EXPECT_TRUE(torch::allclose(
        outNormalizeddLossdMeans2dNormAccum, expectedNormalizeddLossdMeans2dNormAccum, rtol, atol));
    EXPECT_TRUE(torch::allclose(outNormalizedMaxRadiiAccum, expectedNormalizedMaxRadiiAccum));
    EXPECT_TRUE(torch::allclose(outGradientStepCounts, expectedGradientStepCounts));
}
