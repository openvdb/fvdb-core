// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include "utils/Tensor.h"

#include <fvdb/detail/ops/gsplat/GaussianRasterizeContributingGaussianIds.h>
#include <fvdb/detail/ops/gsplat/GaussianRasterizeNumContributingGaussians.h>
#include <fvdb/detail/ops/gsplat/GaussianSplatSparse.h>

#include <torch/script.h>
#include <torch/types.h>

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>
#include <vector>

#ifndef FVDB_EXTERNAL_TEST_DATA_PATH
#error "FVDB_EXTERNAL_TEST_DATA_PATH must be defined"
#endif

struct GaussianRasterizeContributingGaussianIdsTestFixture : public ::testing::Test {
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

        imageWidth   = imageDims[0].item<int32_t>();
        imageHeight  = imageDims[1].item<int32_t>();
        imageOriginW = 0;
        imageOriginH = 0;
        tileSize     = 16;
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
};

TEST_F(GaussianRasterizeContributingGaussianIdsTestFixture, TestBasicInputsAndOutputs) {
    loadTestData("gaussian_top_contributors_1point_input.pt");

    fvdb::detail::ops::RenderSettings settings;
    settings.imageWidth  = imageWidth;
    settings.imageHeight = imageHeight;
    settings.tileSize    = tileSize;

    // First compute the number of contributing Gaussians
    const auto [numContributingGaussians, alphas] =
        fvdb::detail::ops::dispatchGaussianRasterizeNumContributingGaussians<torch::kCUDA>(
            means2d, conics, opacities, tileOffsets, tileGaussianIds, settings);

    // Then compute the IDs and weights
    const auto [outIds, outWeights] =
        fvdb::detail::ops::dispatchGaussianRasterizeContributingGaussianIds<torch::kCUDA>(
            means2d,
            conics,
            opacities,
            tileOffsets,
            tileGaussianIds,
            numContributingGaussians,
            settings);

    const int h                 = imageHeight;
    const int w                 = imageWidth;
    const int numGaussianLayers = 5;

    // Get the data for camera 0
    // outIds and outWeights are nested JaggedTensors: list of cameras, each containing list of
    // pixels
    auto idsUnbind     = outIds.unbind2();
    auto weightsUnbind = outWeights.unbind2();

    // Get camera 0's list of pixels (each element is a 1D tensor of IDs/weights)
    auto camera0Ids     = idsUnbind[0];
    auto camera0Weights = weightsUnbind[0];

    // Calculate the pixel index for the center pixel
    int centerPixelIdx = (h / 2 - 1) * w + (w / 2 - 1);

    // Get the IDs and weights for the center pixel
    auto centerPixelIds     = camera0Ids[centerPixelIdx];
    auto centerPixelWeights = camera0Weights[centerPixelIdx];

    // Slice out the first numGaussianLayers IDs
    auto centerIdsSlice =
        centerPixelIds.index({torch::indexing::Slice(torch::indexing::None, numGaussianLayers)});

    auto expectedRange = torch::arange(
        numGaussianLayers,
        torch::TensorOptions().device(centerPixelIds.device()).dtype(centerPixelIds.dtype()));

    // Test the IDs appear in the correct order
    EXPECT_TRUE(torch::equal(centerIdsSlice, expectedRange));

    // Test expected weights calculation
    // Create expectedWeights with the same size as the actual returned weights (numGaussianLayers)
    auto expectedWeights          = torch::zeros(numGaussianLayers,
                                        torch::TensorOptions()
                                            .device(centerPixelWeights.device())
                                            .dtype(centerPixelWeights.dtype()));
    float accumulatedTransparency = 1.0f;

    // Get the first opacity value as a scalar
    float opacityVal = opacities.flatten()[0].item<float>();

    for (int i = 0; i < numGaussianLayers; ++i) {
        expectedWeights[i] = accumulatedTransparency * opacityVal;
        accumulatedTransparency *= (1.0f - opacityVal);
    }

    // The returned weights should match our expected weights
    EXPECT_TRUE(torch::allclose(centerPixelWeights, expectedWeights));
}

TEST_F(GaussianRasterizeContributingGaussianIdsTestFixture, TestBasicInputsAndOutputsSparse) {
    loadTestData("gaussian_top_contributors_1point_input.pt");

    fvdb::detail::ops::RenderSettings settings;
    settings.imageWidth  = imageWidth;
    settings.imageHeight = imageHeight;
    settings.tileSize    = tileSize;

    // First compute the number of contributing Gaussians for dense rendering
    const auto [numContributingGaussians, alphas] =
        fvdb::detail::ops::dispatchGaussianRasterizeNumContributingGaussians<torch::kCUDA>(
            means2d, conics, opacities, tileOffsets, tileGaussianIds, settings);

    // Then compute the IDs and weights for dense rendering
    const auto [outIds, outWeights] =
        fvdb::detail::ops::dispatchGaussianRasterizeContributingGaussianIds<torch::kCUDA>(
            means2d,
            conics,
            opacities,
            tileOffsets,
            tileGaussianIds,
            numContributingGaussians,
            settings);

    const int h = imageHeight;
    const int w = imageWidth;

    // Create a JaggedTensor for pixels to render (center pixel only)
    const auto pixelsToRenderTensor = torch::tensor({{h / 2 - 1, w / 2 - 1}}).cuda();
    fvdb::JaggedTensor pixelsToRender({pixelsToRenderTensor.unsqueeze(0)});

    auto [activeTiles, activeTileMask, tilePixelMask, tilePixelCumsum, pixelMap] =
        fvdb::detail::ops::computeSparseInfo(
            tileSize, tileOffsets.size(2), tileOffsets.size(1), pixelsToRenderTensor);

    // Compute num contributing gaussians for sparse rendering
    const auto [numContributingGaussiansSparse, alphasSparse] =
        fvdb::detail::ops::dispatchGaussianSparseRasterizeNumContributingGaussians<torch::kCUDA>(
            means2d,
            conics,
            opacities,
            tileOffsets,
            tileGaussianIds,
            pixelsToRender,
            activeTiles,
            tilePixelMask,
            tilePixelCumsum,
            pixelMap,
            settings);

    // Run the same scene with sparse sampling of only the center pixel
    const auto [outIdsSparse, outWeightsSparse] =
        fvdb::detail::ops::dispatchGaussianSparseRasterizeContributingGaussianIds<torch::kCUDA>(
            means2d,
            conics,
            opacities,
            tileOffsets,
            tileGaussianIds,
            pixelsToRender,
            activeTiles,
            tilePixelMask,
            tilePixelCumsum,
            pixelMap,
            numContributingGaussiansSparse,
            settings);

    const int numGaussianLayers = 5;

    // Get the data for camera 0
    auto idsUnbind     = outIds.unbind2();
    auto weightsUnbind = outWeights.unbind2();

    auto camera0Ids     = idsUnbind[0];
    auto camera0Weights = weightsUnbind[0];

    // Calculate the pixel index for the center pixel
    int centerPixelIdx = (h / 2 - 1) * w + (w / 2 - 1);

    // Get the IDs and weights for the center pixel
    auto centerPixelIds     = camera0Ids[centerPixelIdx];
    auto centerPixelWeights = camera0Weights[centerPixelIdx];

    auto centerIdsSlice =
        centerPixelIds.index({torch::indexing::Slice(torch::indexing::None, numGaussianLayers)});

    // Get the IDs from sparse rendering
    // For sparse rendering with 1 pixel, camera 0 should have 1 tensor
    auto sparseIdsUnbind     = outIdsSparse.unbind2();
    auto sparseWeightsUnbind = outWeightsSparse.unbind2();

    auto sparseCamera0Ids     = sparseIdsUnbind[0];
    auto sparseCamera0Weights = sparseWeightsUnbind[0];

    // The sparse rendering only rendered the center pixel, so it's at index 0
    auto sparsePixelIds     = sparseCamera0Ids[0];
    auto sparsePixelWeights = sparseCamera0Weights[0];

    auto outIdsSparseSlice = sparsePixelIds.index({torch::indexing::Slice(0, numGaussianLayers)});

    EXPECT_TRUE(torch::equal(outIdsSparseSlice, centerIdsSlice));

    // Test expected weights calculation
    {
        // Create expectedWeights with the same size as actual returned weights (numGaussianLayers)
        auto expectedWeights          = torch::zeros(numGaussianLayers,
                                            torch::TensorOptions()
                                                .device(centerPixelWeights.device())
                                                .dtype(centerPixelWeights.dtype()));
        float accumulatedTransparency = 1.0f;

        // Get the first opacity value as a scalar
        float opacityVal = opacities.flatten()[0].item<float>();

        for (int i = 0; i < numGaussianLayers; ++i) {
            expectedWeights[i] = accumulatedTransparency * opacityVal;
            accumulatedTransparency *= (1.0f - opacityVal);
        }

        EXPECT_TRUE(torch::equal(sparsePixelWeights, expectedWeights));
    }

    // Compare sparse weights to dense weights at center pixel
    EXPECT_TRUE(torch::allclose(sparsePixelWeights, centerPixelWeights));
}

TEST_F(GaussianRasterizeContributingGaussianIdsTestFixture, CPUThrows) {
    loadTestData("gaussian_top_contributors_1point_input.pt");
    moveToDevice(torch::kCPU);

    fvdb::detail::ops::RenderSettings settings;
    settings.imageWidth  = imageWidth;
    settings.imageHeight = imageHeight;
    settings.tileSize    = tileSize;

    EXPECT_THROW(fvdb::detail::ops::dispatchGaussianRasterizeNumContributingGaussians<torch::kCPU>(
                     means2d, conics, opacities, tileOffsets, tileGaussianIds, settings),
                 c10::NotImplementedError);
}
