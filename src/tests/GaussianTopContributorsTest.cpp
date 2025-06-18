// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include "utils/Tensor.h"

#include <fvdb/detail/ops/Ops.h>

#include <torch/script.h>
#include <torch/torch.h>

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>
#include <vector>

#ifndef FVDB_EXTERNAL_TEST_DATA_PATH
#error "FVDB_EXTERNAL_TEST_DATA_PATH must be defined"
#endif

struct GaussianTopContributorsTestFixture : public ::testing::Test {
    void
    loadInputData(const std::string insPath) {
        const std::string dataPath =
            std::string(FVDB_EXTERNAL_TEST_DATA_PATH) + std::string("/unit_tests/gsplat");
        const std::string inputsPath = dataPath + std::string("/") + insPath;

        std::vector<torch::Tensor> inputs = fvdb::test::loadTensors(inputsPath, inputNames);
        means2d                           = inputs[0].cuda();
        conics                            = inputs[1].cuda();
        opacities                         = inputs[2].cuda();
        tileOffsets                       = inputs[3].cuda();
        tileGaussianIds                   = inputs[4].cuda();
        imageDims                         = inputs[5];

        imageWidth      = imageDims[0].item<int32_t>();
        imageHeight     = imageDims[1].item<int32_t>();
        imageOriginW    = 0;
        imageOriginH    = 0;
        tileSize        = 16;
        numDepthSamples = 16;
    }

    void
    storeData(const std::string outsPath, const std::vector<torch::Tensor> &outputData) {
        const std::string dataPath =
            std::string(FVDB_EXTERNAL_TEST_DATA_PATH) + std::string("/unit_tests/gsplat");
        const std::string outputPath = dataPath + std::string("/") + outsPath;

        fvdb::test::storeTensors(outputPath, outputData, outputNames);
    }

    void
    loadTestData(const std::string insPath, const std::string outsPath = "") {
        // Set the random seed for reproducibility.
        torch::manual_seed(0);

        loadInputData(insPath);

        const std::string dataPath =
            std::string(FVDB_EXTERNAL_TEST_DATA_PATH) + std::string("/unit_tests/gsplat");
        if (!outsPath.empty()) {
            const std::string expectedOutputsPath = dataPath + std::string("/") + outsPath;

            std::vector<torch::Tensor> expectedOutputs =
                fvdb::test::loadTensors(expectedOutputsPath, outputNames);
            expectedIds     = expectedOutputs[0].cuda();
            expectedWeights = expectedOutputs[1].cuda();
        }
    }

    /// @brief Concatenate channels in a color tensor
    /// @param tensor The tensor to concatenate channels in
    /// @param numChannels The number of channels to concatenate
    /// @return The concatenated tensor
    torch::Tensor
    catChannelsToDim(const torch::Tensor &tensor, int numChannels) {
        const int64_t lastDim = tensor.dim() - 1;
        TORCH_CHECK(lastDim >= 0, "tensor must have at least one dimension");
        TORCH_CHECK(numChannels >= tensor.size(lastDim),
                    "numChannels must be at least as large as the last dimension of tensor");

        if (numChannels == tensor.size(lastDim)) {
            return tensor;
        }

        std::vector<torch::Tensor> toConcat;
        toConcat.push_back(tensor);

        const auto extraChannels = numChannels - tensor.size(lastDim);
        if (extraChannels > 0) {
            std::vector<int64_t> extraShape = tensor.sizes().vec();
            extraShape[lastDim]             = extraChannels;
            torch::Tensor extraTensor       = torch::zeros(extraShape, tensor.options());
            toConcat.push_back(extraTensor);
        }

        return torch::cat(toConcat, lastDim);
    }

    void
    moveToDevice(const torch::Device &device) {
        means2d         = means2d.to(device);
        conics          = conics.to(device);
        opacities       = opacities.to(device);
        tileOffsets     = tileOffsets.to(device);
        tileGaussianIds = tileGaussianIds.to(device);
        imageDims       = imageDims.to(device);
        if (expectedIds.defined()) {
            expectedIds = expectedIds.to(device);
        }
        if (expectedWeights.defined()) {
            expectedWeights = expectedWeights.to(device);
        }
    }

    const std::vector<std::string> inputNames = {
        "means2d", "conics", "opacities", "tile_offsets", "tile_gaussian_ids", "image_dims"};
    const std::vector<std::string> outputNames = {"ids", "weights"};

    // Input tensors
    torch::Tensor means2d;         // [C, N, 2] or [nnz, 2]
    torch::Tensor conics;          // [C, N, 3] or [nnz, 3]
    torch::Tensor opacities;       // [C, N] or [nnz]
    torch::Tensor tileOffsets;     // [C, tileHeight, tileWidth]
    torch::Tensor tileGaussianIds; // [nIsects]
    torch::Tensor imageDims;       // [2]

    // Expected output tensors
    torch::Tensor expectedIds;     // [C, imageHeight, imageWidth, D]
    torch::Tensor expectedWeights; // [C, imageHeight, imageWidth, 1]

    // Parameters
    uint32_t imageWidth;
    uint32_t imageHeight;
    uint32_t imageOriginW;
    uint32_t imageOriginH;
    uint32_t tileSize;
    uint32_t numDepthSamples;
};

TEST_F(GaussianTopContributorsTestFixture, TestBasicInputsAndOutputs) {
    loadTestData("gaussian_top_contributors_1point_input.pt");

    fvdb::detail::ops::RenderSettings settings;
    settings.imageWidth      = imageWidth;
    settings.imageHeight     = imageHeight;
    settings.tileSize        = tileSize;
    settings.numDepthSamples = numDepthSamples;

    const auto [outIds, outWeights] =
        fvdb::detail::ops::dispatchGaussianRasterizeTopContributingGaussianIds<torch::kCUDA>(
            means2d, conics, opacities, tileOffsets, tileGaussianIds, settings);

    const int h                 = imageHeight;
    const int w                 = imageWidth;
    const int numGaussianLayers = 5;

    auto centerIdsSlice =
        outIds.index({0,
                      h / 2 - 1,
                      w / 2 - 1,
                      torch::indexing::Slice(torch::indexing::None, numGaussianLayers)});

    auto expectedRange = torch::arange(
        numGaussianLayers, torch::TensorOptions().device(outIds.device()).dtype(outIds.dtype()));

    // Test the IDs appear in the correct order
    EXPECT_TRUE(torch::equal(centerIdsSlice, expectedRange));

    // Test that remaining slots are filled with -1
    auto remainingIdsSlice =
        outIds.index({0,
                      h / 2 - 1,
                      w / 2 - 1,
                      torch::indexing::Slice(numGaussianLayers, torch::indexing::None)});

    int64_t remainingSize     = outIds.size(3) - numGaussianLayers;
    auto expectedNegativeOnes = torch::full(
        {remainingSize}, -1, torch::TensorOptions().device(outIds.device()).dtype(torch::kInt32));

    EXPECT_TRUE(torch::equal(remainingIdsSlice, expectedNegativeOnes));

    // Test expected weights calculation
    const int numSamples = numDepthSamples;
    auto expectedWeights = torch::zeros(
        numSamples, torch::TensorOptions().device(outWeights.device()).dtype(outWeights.dtype()));
    float accumulatedTransparency = 1.0f;

    // Get the first opacity value as a scalar
    float opacityVal = opacities.flatten()[0].item<float>();

    for (int i = 0; i < numGaussianLayers; ++i) {
        expectedWeights[i] = accumulatedTransparency * opacityVal;
        accumulatedTransparency *= (1.0f - opacityVal);
    }

    // Extract weights at center pixel: weights[0][h // 2 - 1][w // 2 - 1]
    auto centerWeights = outWeights.index({0, h / 2 - 1, w / 2 - 1});

    EXPECT_TRUE(torch::allclose(centerWeights, expectedWeights));
}

TEST_F(GaussianTopContributorsTestFixture, CPUThrows) {
    loadTestData("gaussian_top_contributors_1point_input.pt");
    moveToDevice(torch::kCPU);

    fvdb::detail::ops::RenderSettings settings;
    settings.imageWidth      = imageWidth;
    settings.imageHeight     = imageHeight;
    settings.tileSize        = tileSize;
    settings.numDepthSamples = numDepthSamples;

    EXPECT_THROW(
        fvdb::detail::ops::dispatchGaussianRasterizeTopContributingGaussianIds<torch::kCPU>(
            means2d, conics, opacities, tileOffsets, tileGaussianIds, settings),
        c10::NotImplementedError);
}
