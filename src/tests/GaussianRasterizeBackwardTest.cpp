// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include "utils/Tensor.h"

#include <fvdb/detail/ops/gsplat/GaussianRasterizeBackward.h>
#include <fvdb/detail/ops/gsplat/GaussianRasterizeForward.h>
#include <fvdb/detail/ops/gsplat/GaussianSplatSparse.h>

#include <torch/all.h>

#include <gtest/gtest.h>

#include <cstdlib>

#ifndef FVDB_EXTERNAL_TEST_DATA_PATH
#error "FVDB_EXTERNAL_TEST_DATA_PATH must be defined"
#endif

// Helper class to reduce test code repetition
class GaussianTestHelper {
  public:
    struct GaussianParams {
        torch::Tensor means2d, conics, colors, opacities;
    };

    struct TileStructure {
        torch::Tensor tileOffsets, tileGaussianIds;
        int32_t numTilesW, numTilesH;
    };

    struct SparseSetup {
        fvdb::JaggedTensor pixelsToRender;
        torch::Tensor activeTiles, tilePixelMask, tilePixelCumsum, pixelMap;
    };

    struct SparseForwardResults {
        fvdb::JaggedTensor colors, alphas, lastIds;
    };

    struct DenseForwardResults {
        torch::Tensor colors, alphas, lastIds;
    };

    struct BackwardResults {
        torch::Tensor dLossDMeans2dAbs, dLossDMeans2d, dLossDConics, dLossDColors, dLossDOpacities;
    };

    static GaussianParams
    createTestGaussians(int numCameras,
                        int numGaussians,
                        int channels,
                        const std::vector<std::pair<float, float>> &positions = {}) {
        auto means2d = torch::zeros({numCameras, numGaussians, 2},
                                    torch::dtype(torch::kFloat32).device(torch::kCUDA));

        // Set positions if provided, otherwise use defaults
        // Apply same positions to all cameras
        for (int cam = 0; cam < numCameras; cam++) {
            if (!positions.empty()) {
                for (int g = 0; g < std::min(numGaussians, (int)positions.size()); g++) {
                    means2d[cam][g][0] = positions[g].first;  // x
                    means2d[cam][g][1] = positions[g].second; // y
                }
            } else {
                // Default positions
                for (int g = 0; g < numGaussians; g++) {
                    means2d[cam][g][0] = 16.0f + g * 8.0f;
                    means2d[cam][g][1] = 16.0f + g * 8.0f;
                }
            }
        }

        auto conics = torch::ones({numCameras, numGaussians, 3},
                                  torch::dtype(torch::kFloat32).device(torch::kCUDA));
        conics.select(-1, 1).fill_(0.0f); // Set xy component to 0

        auto colors = torch::rand({numCameras, numGaussians, channels},
                                  torch::dtype(torch::kFloat32).device(torch::kCUDA));

        auto opacities = torch::ones({numCameras, numGaussians},
                                     torch::dtype(torch::kFloat32).device(torch::kCUDA));

        return {means2d, conics, colors, opacities};
    }

    // Returns a vector of (x, y) integer pixel positions by rounding the float gaussian positions.
    static auto
    getGaussianPositionsFromPixels(const std::vector<std::pair<int, int>> &positions) {
        std::vector<std::pair<float, float>> gaussianPositions(positions.size());
        std::transform(
            positions.begin(), positions.end(), gaussianPositions.begin(), [](const auto &pos) {
                return std::make_pair(static_cast<float>(pos.first),
                                      static_cast<float>(pos.second));
            });
        return gaussianPositions;
    };

    static TileStructure
    createTileStructure(int imageWidth,
                        int imageHeight,
                        int tileSize,
                        const std::vector<std::pair<float, float>> &gaussianPositions,
                        int numCameras = 1) {
        const int32_t numTilesW = (imageWidth + tileSize - 1) / tileSize;
        const int32_t numTilesH = (imageHeight + tileSize - 1) / tileSize;

        auto tileOffsets = torch::zeros({numCameras, numTilesH, numTilesW},
                                        torch::dtype(torch::kInt32).device(torch::kCUDA));

        std::vector<int32_t> tileGaussianIds;
        int32_t currentOffset = 0;

        // Create tile intersections for each camera
        for (int cameraIdx = 0; cameraIdx < numCameras; ++cameraIdx) {
            for (int h = 0; h < numTilesH; h++) {
                for (int w = 0; w < numTilesW; w++) {
                    tileOffsets[cameraIdx][h][w] = currentOffset;

                    // Add gaussians that intersect this tile
                    for (int g = 0; g < (int)gaussianPositions.size(); g++) {
                        int32_t gaussianTileRow = (int)gaussianPositions[g].second / tileSize;
                        int32_t gaussianTileCol = (int)gaussianPositions[g].first / tileSize;

                        if (h == gaussianTileRow && w == gaussianTileCol) {
                            // Each camera references the same gaussian indices (0 to
                            // numGaussians-1)
                            tileGaussianIds.push_back(g);
                            currentOffset++;
                        }
                    }
                }
            }
        }

        auto tileGaussianIdsTensor =
            torch::tensor(tileGaussianIds, torch::dtype(torch::kInt32).device(torch::kCUDA));

        return {tileOffsets, tileGaussianIdsTensor, numTilesW, numTilesH};
    }

    static SparseSetup
    createSparseSetup(int tileSize,
                      int numTilesW,
                      int numTilesH,
                      const std::vector<std::pair<int, int>> &pixelPositions,
                      int numCameras = 1) {
        std::vector<torch::Tensor> sparsePixelCoords;

        for (int camera = 0; camera < numCameras; camera++) {
            auto pixelCoords = torch::zeros({(int)pixelPositions.size(), 2},
                                            torch::dtype(torch::kInt32).device(torch::kCUDA));

            for (int i = 0; i < (int)pixelPositions.size(); i++) {
                pixelCoords[i][0] = pixelPositions[i].second; // row (y)
                pixelCoords[i][1] = pixelPositions[i].first;  // col (x)
            }
            sparsePixelCoords.push_back(pixelCoords);
        }

        auto pixelsToRender = fvdb::JaggedTensor(sparsePixelCoords);

        auto [activeTiles, activeTileMask, tilePixelMask, tilePixelCumsum, pixelMap] =
            fvdb::detail::ops::computeSparseInfo(tileSize, numTilesW, numTilesH, pixelsToRender);

        return {pixelsToRender, activeTiles, tilePixelMask, tilePixelCumsum, pixelMap};
    }

    static DenseForwardResults
    runForwardDense(const GaussianParams &gaussians,
                    const TileStructure &tiles,
                    int imageWidth,
                    int imageHeight,
                    int tileSize) {
        auto [colors, alphas, lastIds] =
            fvdb::detail::ops::dispatchGaussianRasterizeForward<torch::kCUDA>(
                gaussians.means2d,
                gaussians.conics,
                gaussians.colors,
                gaussians.opacities,
                imageWidth,
                imageHeight,
                0,
                0,
                tileSize,
                tiles.tileOffsets,
                tiles.tileGaussianIds);
        return {colors, alphas, lastIds};
    }

    static SparseForwardResults
    runForwardSparse(const GaussianParams &gaussians,
                     const TileStructure &tiles,
                     const SparseSetup &sparse,
                     int imageWidth,
                     int imageHeight,
                     int tileSize) {
        auto [colors, alphas, lastIds] =
            fvdb::detail::ops::dispatchGaussianSparseRasterizeForward<torch::kCUDA>(
                sparse.pixelsToRender,
                gaussians.means2d,
                gaussians.conics,
                gaussians.colors,
                gaussians.opacities,
                imageWidth,
                imageHeight,
                0,
                0,
                tileSize,
                tiles.tileOffsets,
                tiles.tileGaussianIds,
                sparse.activeTiles,
                sparse.tilePixelMask,
                sparse.tilePixelCumsum,
                sparse.pixelMap);
        return {colors, alphas, lastIds};
    }

    static torch::Tensor
    createMaskedGradients(const torch::Tensor &templateTensor,
                          const std::vector<std::pair<int, int>> &pixelPositions,
                          float value = 1.0f) {
        auto gradients       = torch::zeros_like(templateTensor);
        const int numCameras = templateTensor.size(0);

        // Apply the same pixel positions to all cameras
        for (int camIdx = 0; camIdx < numCameras; camIdx++) {
            for (const auto &pos: pixelPositions) {
                if (templateTensor.dim() == 4) { // [C, H, W, channels]
                    gradients[camIdx][pos.second][pos.first] = value;
                } else {                         // [C, H, W]
                    gradients[camIdx][pos.second][pos.first] = value;
                }
            }
        }
        return gradients;
    }

    static DenseForwardResults
    createMaskedForwardResults(const DenseForwardResults &denseResults,
                               const fvdb::JaggedTensor &pixelsToRender,
                               int imageWidth,
                               int imageHeight) {
        const int numCameras = denseResults.colors.size(0);

        // Create masked dense results (zeros everywhere, actual values at sparse pixels)
        auto maskedColors  = torch::zeros_like(denseResults.colors);
        auto maskedAlphas  = torch::zeros_like(denseResults.alphas);
        auto maskedLastIds = torch::full_like(denseResults.lastIds, -1); // background = -1

        // Copy actual forward results at sparse pixel positions
        for (int camIdx = 0; camIdx < numCameras; ++camIdx) {
            const auto coords   = pixelsToRender.index(camIdx).jdata();
            const int numPixels = coords.size(0);

            for (int pixelIdx = 0; pixelIdx < numPixels; ++pixelIdx) {
                const int32_t row = coords[pixelIdx][0].item<int32_t>();
                const int32_t col = coords[pixelIdx][1].item<int32_t>();

                // Check if coordinates are within image bounds
                if (row >= 0 && row < imageHeight && col >= 0 && col < imageWidth) {
                    maskedColors[camIdx][row][col]  = denseResults.colors[camIdx][row][col];
                    maskedAlphas[camIdx][row][col]  = denseResults.alphas[camIdx][row][col];
                    maskedLastIds[camIdx][row][col] = denseResults.lastIds[camIdx][row][col];
                }
            }
        }

        return {maskedColors, maskedAlphas, maskedLastIds};
    }

    static void
    compareSparseVsDense(const torch::Tensor &sparse,
                         const torch::Tensor &dense,
                         const std::string &name,
                         float rtol                   = 1e-4f,
                         float atol                   = 1e-6f,
                         const std::string &extraInfo = "") {
        std::string fullName = name;
        if (!extraInfo.empty()) {
            fullName += " (" + extraInfo + ")";
        }

        EXPECT_TRUE(torch::allclose(sparse, dense, rtol, atol)) << fullName << " mismatch";
    }

    static BackwardResults
    runBackwardDense(const GaussianParams &gaussians,
                     const TileStructure &tiles,
                     const DenseForwardResults &forwardResults,
                     const torch::Tensor &gradColors,
                     const torch::Tensor &gradAlphas,
                     int imageWidth,
                     int imageHeight,
                     int tileSize,
                     int64_t numSharedChannelsOverride = -1) {
        auto [dLossDMeans2dAbs, dLossDMeans2d, dLossDConics, dLossDColors, dLossDOpacities] =
            fvdb::detail::ops::dispatchGaussianRasterizeBackward<torch::kCUDA>(
                gaussians.means2d,
                gaussians.conics,
                gaussians.colors,
                gaussians.opacities,
                imageWidth,
                imageHeight,
                0, // imageOriginW
                0, // imageOriginH
                tileSize,
                tiles.tileOffsets,
                tiles.tileGaussianIds,
                forwardResults.alphas,
                forwardResults.lastIds,
                gradColors,
                gradAlphas,
                false,
                numSharedChannelsOverride);
        return {dLossDMeans2dAbs, dLossDMeans2d, dLossDConics, dLossDColors, dLossDOpacities};
    }

    static BackwardResults
    runBackwardSparse(const GaussianParams &gaussians,
                      const TileStructure &tiles,
                      const SparseSetup &sparse,
                      const SparseForwardResults &forwardResults,
                      const fvdb::JaggedTensor &gradColors,
                      const fvdb::JaggedTensor &gradAlphas,
                      int imageWidth,
                      int imageHeight,
                      int tileSize,
                      int64_t numSharedChannelsOverride = -1) {
        auto [dLossDMeans2dAbs, dLossDMeans2d, dLossDConics, dLossDColors, dLossDOpacities] =
            fvdb::detail::ops::dispatchGaussianSparseRasterizeBackward<torch::kCUDA>(
                sparse.pixelsToRender,
                gaussians.means2d,
                gaussians.conics,
                gaussians.colors,
                gaussians.opacities,
                imageWidth,
                imageHeight,
                0, // imageOriginW
                0, // imageOriginH
                tileSize,
                tiles.tileOffsets,
                tiles.tileGaussianIds,
                forwardResults.alphas,
                forwardResults.lastIds,
                gradColors,
                gradAlphas,
                sparse.activeTiles,
                sparse.tilePixelMask,
                sparse.tilePixelCumsum,
                sparse.pixelMap,
                false,
                numSharedChannelsOverride);
        return {dLossDMeans2dAbs, dLossDMeans2d, dLossDConics, dLossDColors, dLossDOpacities};
    }
};

struct GaussianRasterizeTestFixture : public ::testing::Test {
    void
    loadTestData(const std::string insPath, const std::string outsPath) {
        // Set the random seed for reproducibility.
        torch::manual_seed(0);

        const std::string dataPath =
            std::string(FVDB_EXTERNAL_TEST_DATA_PATH) + std::string("/unit_tests/gsplat");
        const std::string inputsPath          = dataPath + std::string("/") + insPath;
        const std::string expectedOutputsPath = dataPath + std::string("/") + outsPath;

        auto inputs = fvdb::test::loadTensors(inputsPath, inputNames);

        means2d                 = inputs[0].cuda();
        conics                  = inputs[1].cuda();
        colors                  = inputs[2].cuda();
        opacities               = inputs[3].cuda();
        tileOffsets             = inputs[4].cuda();
        tileGaussianIds         = inputs[5].cuda();
        renderedAlphas          = inputs[6].cuda();
        lastGaussianIdsPerPixel = inputs[7].cuda();
        dLossDRenderedColors    = inputs[8].cuda();
        dLossDRenderedAlphas    = inputs[9].cuda();

        auto expectedOutputs    = fvdb::test::loadTensors(expectedOutputsPath, outputNames);
        expectedDLossDMeans2d   = expectedOutputs[0].cuda();
        expectedDLossDConics    = expectedOutputs[1].cuda();
        expectedDLossDColors    = expectedOutputs[2].cuda();
        expectedDLossDOpacities = expectedOutputs[3].cuda();

        imageWidth   = 1297;
        imageHeight  = 840;
        imageOriginW = 0;
        imageOriginH = 0;
        tileSize     = 16;
    }

    void
    SetUp() override {}

    torch::Tensor
    catChannelsToDim(const torch::Tensor &input, int64_t numOutChannels) {
        using namespace at::indexing;

        const int64_t numInChannels = input.size(-1);

        EXPECT_TRUE(numInChannels <= numOutChannels);

        if (numInChannels == numOutChannels) {
            return input;
        }

        std::vector<torch::Tensor> tensorsToCat;
        for (int i = 0; i < numOutChannels / numInChannels; i += 1) {
            tensorsToCat.push_back(input);
        }
        tensorsToCat.push_back(input.index({Ellipsis, Slice(0, numOutChannels % numInChannels)}));
        torch::Tensor ret = torch::cat(tensorsToCat, -1);
        return ret;
    }

    void
    moveToDevice(torch::Device device) {
        means2d                 = means2d.to(device);
        conics                  = conics.to(device);
        colors                  = colors.to(device);
        opacities               = opacities.to(device);
        tileOffsets             = tileOffsets.to(device);
        tileGaussianIds         = tileGaussianIds.to(device);
        renderedAlphas          = renderedAlphas.to(device);
        lastGaussianIdsPerPixel = lastGaussianIdsPerPixel.to(device);
        dLossDRenderedColors    = dLossDRenderedColors.to(device);
        dLossDRenderedAlphas    = dLossDRenderedAlphas.to(device);
    }

    const std::vector<std::string> inputNames  = {"means2d",
                                                  "conics",
                                                  "colors",
                                                  "opacities",
                                                  "tile_offsets",
                                                  "tile_gaussian_ids",
                                                  "rendered_alphas",
                                                  "last_gaussian_ids_per_pixel",
                                                  "d_loss_d_rendered_colors",
                                                  "d_loss_d_rendered_alphas"};
    const std::vector<std::string> outputNames = {
        "d_loss_d_means2d", "d_loss_d_conics", "d_loss_d_colors", "d_loss_d_opacities"};

    torch::Tensor means2d;
    torch::Tensor conics;
    torch::Tensor colors;
    torch::Tensor opacities;
    torch::Tensor tileOffsets;
    torch::Tensor tileGaussianIds;
    torch::Tensor renderedAlphas;
    torch::Tensor lastGaussianIdsPerPixel;
    torch::Tensor dLossDRenderedColors;
    torch::Tensor dLossDRenderedAlphas;

    torch::Tensor expectedDLossDMeans2d;
    torch::Tensor expectedDLossDConics;
    torch::Tensor expectedDLossDColors;
    torch::Tensor expectedDLossDOpacities;

    int32_t imageWidth   = 1297;
    int32_t imageHeight  = 840;
    int32_t imageOriginW = 0;
    int32_t imageOriginH = 0;
    int32_t tileSize     = 16;

    // Helper function to generate sparse pixel coordinates for testing
    fvdb::JaggedTensor
    generateSparsePixelCoords(int numCameras, int maxPixelsPerCamera) {
        // Create a list of tensors, one for each camera
        std::vector<torch::Tensor> pixelCoordsList;
        for (int i = 0; i < numCameras; i++) {
            // Generate random number of pixels for this camera (up to maxPixelsPerCamera)
            int numPixels = torch::randint(1, maxPixelsPerCamera + 1, {1}).item<int>();

            // Generate random pixel coordinates within image bounds
            auto const xCoords = torch::randint(0, this->imageWidth, {numPixels});
            auto const yCoords = torch::randint(0, this->imageHeight, {numPixels});

            // Stack x and y coordinates to form 2D pixel coordinates
            auto const pixelCoords = torch::stack({yCoords, xCoords}, 1);

            // Note even with sorted=false torch::unique_dim returns a sorted tensor
            auto const [unique_coords, unused_1, unused_2] =
                torch::unique_dim(pixelCoords, 0, false);

            pixelCoordsList.push_back(unique_coords);
        }

        // Create JaggedTensor from the list of pixel coordinate tensors
        return fvdb::JaggedTensor(pixelCoordsList);
    }
};

TEST_F(GaussianRasterizeTestFixture, TestBasicInputsAndOutputs) {
    loadTestData("rasterize_backward_inputs.pt", "rasterize_backward_outputs.pt");

    // Create helper structures from test fixture data
    GaussianTestHelper::GaussianParams gaussians = {means2d, conics, colors, opacities};
    GaussianTestHelper::TileStructure tiles      = {
        tileOffsets, tileGaussianIds, (int32_t)tileOffsets.size(2), (int32_t)tileOffsets.size(1)};
    // Create forward results structure (colors not needed for backward-only test)
    auto placeholderColors =
        torch::zeros_like(renderedAlphas.unsqueeze(-1).expand({1, -1, -1, -1, 3}));
    GaussianTestHelper::DenseForwardResults forwardResults = {
        placeholderColors, renderedAlphas, lastGaussianIdsPerPixel};

    auto backwardResults = GaussianTestHelper::runBackwardDense(gaussians,
                                                                tiles,
                                                                forwardResults,
                                                                dLossDRenderedColors,
                                                                dLossDRenderedAlphas,
                                                                imageWidth,
                                                                imageHeight,
                                                                tileSize);

    EXPECT_TRUE(torch::allclose(backwardResults.dLossDMeans2d, expectedDLossDMeans2d));

    // This is a big sum of products in parallel that is pretty ill conditioned and so we
    // only expect about 1 digit of accuracy.
    EXPECT_TRUE(torch::allclose(backwardResults.dLossDConics, expectedDLossDConics, 1e-1 /*rtol*/));

    EXPECT_TRUE(torch::allclose(backwardResults.dLossDColors, expectedDLossDColors));
    EXPECT_TRUE(torch::allclose(backwardResults.dLossDOpacities, expectedDLossDOpacities, 1e-4));
}

TEST_F(GaussianRasterizeTestFixture, TestConcatenatedChannels) {
    loadTestData("rasterize_backward_inputs.pt", "rasterize_backward_outputs_64.pt");

    colors               = catChannelsToDim(colors, 64);
    dLossDRenderedColors = catChannelsToDim(dLossDRenderedColors, 64);
    expectedDLossDColors = catChannelsToDim(expectedDLossDColors, 64);

    // Create helper structures from test fixture data
    GaussianTestHelper::GaussianParams gaussians = {means2d, conics, colors, opacities};
    GaussianTestHelper::TileStructure tiles      = {
        tileOffsets, tileGaussianIds, (int32_t)tileOffsets.size(2), (int32_t)tileOffsets.size(1)};
    // Create forward results structure (colors not needed for backward-only test)
    auto placeholderColors =
        torch::zeros_like(renderedAlphas.unsqueeze(-1).expand({1, -1, -1, -1, 64}));
    GaussianTestHelper::DenseForwardResults forwardResults = {
        placeholderColors, renderedAlphas, lastGaussianIdsPerPixel};

    auto backwardResults = GaussianTestHelper::runBackwardDense(gaussians,
                                                                tiles,
                                                                forwardResults,
                                                                dLossDRenderedColors,
                                                                dLossDRenderedAlphas,
                                                                imageWidth,
                                                                imageHeight,
                                                                tileSize);

    EXPECT_TRUE(torch::allclose(backwardResults.dLossDMeans2d, expectedDLossDMeans2d));

    // This is a big sum of products in parallel that is pretty ill conditioned and so we
    // only expect about 1 digit of accuracy.
    EXPECT_TRUE(torch::allclose(backwardResults.dLossDConics, expectedDLossDConics, 1e-1 /*rtol*/));

    EXPECT_TRUE(torch::allclose(backwardResults.dLossDColors, expectedDLossDColors));
    EXPECT_TRUE(torch::allclose(backwardResults.dLossDOpacities, expectedDLossDOpacities, 1e-4));
}

TEST_F(GaussianRasterizeTestFixture, TestConcatenatedChunkedChannelsWithUnusedChannels) {
    loadTestData("rasterize_backward_inputs.pt", "rasterize_backward_outputs_47.pt");

    colors               = catChannelsToDim(colors, 47);
    dLossDRenderedColors = catChannelsToDim(dLossDRenderedColors, 47);

    // Create helper structures from test fixture data
    GaussianTestHelper::GaussianParams gaussians = {means2d, conics, colors, opacities};
    GaussianTestHelper::TileStructure tiles      = {
        tileOffsets, tileGaussianIds, (int32_t)tileOffsets.size(2), (int32_t)tileOffsets.size(1)};
    // Create forward results structure (colors not needed for backward-only test)
    auto placeholderColors =
        torch::zeros_like(renderedAlphas.unsqueeze(-1).expand({1, -1, -1, -1, 47}));
    GaussianTestHelper::DenseForwardResults forwardResults = {
        placeholderColors, renderedAlphas, lastGaussianIdsPerPixel};

    auto backwardResults = GaussianTestHelper::runBackwardDense(gaussians,
                                                                tiles,
                                                                forwardResults,
                                                                dLossDRenderedColors,
                                                                dLossDRenderedAlphas,
                                                                imageWidth,
                                                                imageHeight,
                                                                tileSize,
                                                                32);

    EXPECT_TRUE(torch::allclose(backwardResults.dLossDMeans2d, expectedDLossDMeans2d));
    EXPECT_TRUE(torch::allclose(backwardResults.dLossDColors, expectedDLossDColors));
    EXPECT_TRUE(torch::allclose(backwardResults.dLossDOpacities, expectedDLossDOpacities, 1e-4));
}

TEST_F(GaussianRasterizeTestFixture, TestChunkedChannels) {
    loadTestData("rasterize_backward_inputs.pt", "rasterize_backward_outputs_64.pt");

    colors               = catChannelsToDim(colors, 64);
    dLossDRenderedColors = catChannelsToDim(dLossDRenderedColors, 64);
    expectedDLossDColors = catChannelsToDim(expectedDLossDColors, 64);

    // Create helper structures from test fixture data
    GaussianTestHelper::GaussianParams gaussians = {means2d, conics, colors, opacities};
    GaussianTestHelper::TileStructure tiles      = {
        tileOffsets, tileGaussianIds, (int32_t)tileOffsets.size(2), (int32_t)tileOffsets.size(1)};
    // Create forward results structure (colors not needed for backward-only test)
    auto placeholderColors =
        torch::zeros_like(renderedAlphas.unsqueeze(-1).expand({1, -1, -1, -1, 64}));
    GaussianTestHelper::DenseForwardResults forwardResults = {
        placeholderColors, renderedAlphas, lastGaussianIdsPerPixel};

    auto backwardResults = GaussianTestHelper::runBackwardDense(gaussians,
                                                                tiles,
                                                                forwardResults,
                                                                dLossDRenderedColors,
                                                                dLossDRenderedAlphas,
                                                                imageWidth,
                                                                imageHeight,
                                                                tileSize,
                                                                32);

    EXPECT_TRUE(torch::allclose(backwardResults.dLossDMeans2d, expectedDLossDMeans2d));
    EXPECT_TRUE(torch::allclose(backwardResults.dLossDColors, expectedDLossDColors));
    EXPECT_TRUE(torch::allclose(backwardResults.dLossDOpacities, expectedDLossDOpacities, 1e-4));
}

TEST_F(GaussianRasterizeTestFixture, CPUThrows) {
    loadTestData("rasterize_backward_inputs.pt", "rasterize_backward_outputs.pt");
    moveToDevice(torch::kCPU);
    EXPECT_THROW(
        fvdb::detail::ops::dispatchGaussianRasterizeBackward<torch::kCPU>(means2d,
                                                                          conics,
                                                                          colors,
                                                                          opacities,
                                                                          imageWidth,
                                                                          imageHeight,
                                                                          imageOriginW,
                                                                          imageOriginH,
                                                                          tileSize,
                                                                          tileOffsets,
                                                                          tileGaussianIds,
                                                                          renderedAlphas,
                                                                          lastGaussianIdsPerPixel,
                                                                          dLossDRenderedColors,
                                                                          dLossDRenderedAlphas,
                                                                          false),
        c10::NotImplementedError);
}

TEST_F(GaussianRasterizeTestFixture, TestSparseBackwardRasterization) {
    loadTestData("rasterize_backward_inputs.pt", "rasterize_backward_outputs.pt");

    const int numCameras = means2d.size(0);

    // Generate sparse pixel coordinates to render
    auto const pixelsToRender = generateSparsePixelCoords(numCameras, 100).cuda();

    // Compute sparse info for the pixels to render
    auto [activeTiles, activeTileMask, tilePixelMask, tilePixelCumsum, pixelMap] =
        fvdb::detail::ops::computeSparseInfo(
            tileSize, tileOffsets.size(2), tileOffsets.size(1), pixelsToRender);

    // Step 1: Run forward dense on the same scene to get dense rendered output
    // First create the gaussian params and tile structure for the helpers
    GaussianTestHelper::GaussianParams gaussians = {means2d, conics, colors, opacities};
    GaussianTestHelper::TileStructure tiles      = {
        tileOffsets, tileGaussianIds, (int32_t)tileOffsets.size(2), (int32_t)tileOffsets.size(1)};

    auto denseForwardResults =
        GaussianTestHelper::runForwardDense(gaussians, tiles, imageWidth, imageHeight, tileSize);

    // Step 2: Black out all pixels not in the PixelsToRender input using helper
    auto maskedResults = GaussianTestHelper::createMaskedForwardResults(
        denseForwardResults, pixelsToRender, imageWidth, imageHeight);

    // Create masked gradients to use for both dense and sparse backward passes
    // This ensures we're comparing like with like
    auto placeholderLastIds = torch::zeros_like(
        dLossDRenderedAlphas.squeeze(-1)); // placeholder tensor for reusing helper
    auto maskedGradResults = GaussianTestHelper::createMaskedForwardResults(
        {dLossDRenderedColors, dLossDRenderedAlphas, placeholderLastIds},
        pixelsToRender,
        imageWidth,
        imageHeight);

    // Step 3: Run backward dense on this modified output to get expected gradients
    auto expectedBackwardResults = GaussianTestHelper::runBackwardDense(gaussians,
                                                                        tiles,
                                                                        maskedResults,
                                                                        maskedGradResults.colors,
                                                                        maskedGradResults.alphas,
                                                                        imageWidth,
                                                                        imageHeight,
                                                                        tileSize);

    // Step 4: Run forward and backward sparse and compare
    // Create sparse setup for the helper
    GaussianTestHelper::SparseSetup sparse = {
        pixelsToRender, activeTiles, tilePixelMask, tilePixelCumsum, pixelMap};

    auto sparseForwardResults = GaussianTestHelper::runForwardSparse(
        gaussians, tiles, sparse, imageWidth, imageHeight, tileSize);

    // Create sparse gradient inputs by extracting from the masked gradients

    // Extract sparse gradient inputs from the masked gradients
    std::vector<torch::Tensor> sparseGradColorsList;
    std::vector<torch::Tensor> sparseGradAlphasList;

    for (int camIdx = 0; camIdx < numCameras; ++camIdx) {
        const auto coords     = pixelsToRender.index(camIdx).jdata();
        const int numPixels   = coords.size(0);
        const int numChannels = colors.size(-1);

        // Extract gradients from masked case at sparse pixel locations
        auto gradColors = torch::zeros({numPixels, numChannels},
                                       torch::dtype(torch::kFloat32).device(torch::kCUDA));
        auto gradAlphas =
            torch::zeros({numPixels, 1}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

        for (int pixelIdx = 0; pixelIdx < numPixels; ++pixelIdx) {
            const int32_t row = coords[pixelIdx][0].item<int32_t>();
            const int32_t col = coords[pixelIdx][1].item<int32_t>();

            // Check if coordinates are within image bounds
            if (row >= 0 && row < imageHeight && col >= 0 && col < imageWidth) {
                gradColors[pixelIdx] = maskedGradResults.colors[camIdx][row][col];
                gradAlphas[pixelIdx] = maskedGradResults.alphas[camIdx][row][col];
            }
        }

        sparseGradColorsList.push_back(gradColors);
        sparseGradAlphasList.push_back(gradAlphas);
    }

    auto sparseGradColors = fvdb::JaggedTensor(sparseGradColorsList);
    auto sparseGradAlphas = fvdb::JaggedTensor(sparseGradAlphasList);

    auto sparseBackwardResults = GaussianTestHelper::runBackwardSparse(gaussians,
                                                                       tiles,
                                                                       sparse,
                                                                       sparseForwardResults,
                                                                       sparseGradColors,
                                                                       sparseGradAlphas,
                                                                       imageWidth,
                                                                       imageHeight,
                                                                       tileSize);

    EXPECT_TRUE(torch::allclose(
        sparseBackwardResults.dLossDMeans2d, expectedBackwardResults.dLossDMeans2d, 1e-4, 1e-6));
    EXPECT_TRUE(torch::allclose(
        sparseBackwardResults.dLossDConics, expectedBackwardResults.dLossDConics, 1e-1, 1e-4));
    EXPECT_TRUE(torch::allclose(
        sparseBackwardResults.dLossDColors, expectedBackwardResults.dLossDColors, 1e-4, 1e-6));
    EXPECT_TRUE(torch::allclose(sparseBackwardResults.dLossDOpacities,
                                expectedBackwardResults.dLossDOpacities,
                                1e-4,
                                1e-6));

    // For now, just ensure we can get expected gradients from the masked dense approach
    // This validates our test methodology before implementing sparse backward
    EXPECT_GT(torch::norm(expectedBackwardResults.dLossDMeans2d).item<float>(), 0.0f);
    EXPECT_GT(torch::norm(expectedBackwardResults.dLossDConics).item<float>(), 0.0f);
    EXPECT_GT(torch::norm(expectedBackwardResults.dLossDColors).item<float>(), 0.0f);
    EXPECT_GT(torch::norm(expectedBackwardResults.dLossDOpacities).item<float>(), 0.0f);
}

// Helper function to test sparse backward with controlled gaussians (parameterized by camera count)
void
testSparseBackwardWithControlledGaussians(int numCameras) {
    const int numGaussians = 3;
    const int channels     = 3;

    // Override image size for this test to be smaller and more controlled
    const int32_t testImageWidth  = 64;
    const int32_t testImageHeight = 64;
    const int32_t testTileSize    = 16;

    // Define Pixel Positions
    std::vector<std::pair<int, int>> pixelPositions = {{32, 32}, {16, 16}, {48, 48}};
    auto gaussianPositions = GaussianTestHelper::getGaussianPositionsFromPixels(pixelPositions);

    auto gaussians = GaussianTestHelper::createTestGaussians(
        numCameras, numGaussians, channels, gaussianPositions);

    // Override with specific test colors for all cameras
    gaussians.colors = torch::tensor({{{1.0f, 0.0f, 0.0f},   // Red
                                       {0.0f, 1.0f, 0.0f},   // Green
                                       {0.0f, 0.0f, 1.0f}}}, // Blue
                                     torch::dtype(torch::kFloat32).device(torch::kCUDA))
                           .repeat({numCameras, 1, 1});

    // Create tile structure using helper
    auto tiles = GaussianTestHelper::createTileStructure(
        testImageWidth, testImageHeight, testTileSize, gaussianPositions, numCameras);

    // Step 1: Define sparse pixels we want to render based on gaussian positions

    auto sparse = GaussianTestHelper::createSparseSetup(
        testTileSize, tiles.numTilesW, tiles.numTilesH, pixelPositions, numCameras);

    // Step 2: Run forward dense
    auto denseResults = GaussianTestHelper::runForwardDense(
        gaussians, tiles, testImageWidth, testImageHeight, testTileSize);

    // Step 3: Create gradient inputs using helper and mask dense results to sparse pixels only
    auto dLossDColors =
        GaussianTestHelper::createMaskedGradients(denseResults.colors, pixelPositions, 1.0f);
    auto dLossDAlphas =
        GaussianTestHelper::createMaskedGradients(denseResults.alphas, pixelPositions, 1.0f);

    // Create masked forward results using helper
    // Convert pixel positions to JaggedTensor
    auto pixelCoords = torch::zeros({(int)pixelPositions.size(), 2},
                                    torch::dtype(torch::kInt32).device(torch::kCUDA));
    for (int i = 0; i < (int)pixelPositions.size(); i++) {
        pixelCoords[i][0] = pixelPositions[i].second; // row (y)
        pixelCoords[i][1] = pixelPositions[i].first;  // col (x)
    }
    std::vector<int64_t> lsizes(numCameras, pixelPositions.size());
    auto pixelsToRender = fvdb::JaggedTensor(lsizes, pixelCoords.repeat({numCameras, 1}));

    auto maskedDenseResults = GaussianTestHelper::createMaskedForwardResults(
        denseResults, pixelsToRender, testImageWidth, testImageHeight);

    // Step 5: Run backward dense on masked results (this gives expected gradients)
    auto expectedBackwardResults = GaussianTestHelper::runBackwardDense(gaussians,
                                                                        tiles,
                                                                        maskedDenseResults,
                                                                        dLossDColors,
                                                                        dLossDAlphas,
                                                                        testImageWidth,
                                                                        testImageHeight,
                                                                        testTileSize);

    // Step 6: Run forward and backward sparse
    auto sparseResults = GaussianTestHelper::runForwardSparse(
        gaussians, tiles, sparse, testImageWidth, testImageHeight, testTileSize);

    // Create sparse gradient inputs by extracting from dense gradients at sparse pixel locations
    // (convert from dense tensor format to jagged tensor format required by sparse backward)
    std::vector<torch::Tensor> sparseGradColorsList;
    std::vector<torch::Tensor> sparseGradAlphasList;

    for (int camIdx = 0; camIdx < numCameras; ++camIdx) {
        const auto coords   = sparse.pixelsToRender.index(camIdx).jdata();
        const int numPixels = coords.size(0);

        auto gradColors =
            torch::zeros({numPixels, channels}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        auto gradAlphas =
            torch::zeros({numPixels, 1}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

        for (int pixelIdx = 0; pixelIdx < numPixels; ++pixelIdx) {
            const int32_t row = coords[pixelIdx][0].item<int32_t>();
            const int32_t col = coords[pixelIdx][1].item<int32_t>();

            if (row >= 0 && row < testImageHeight && col >= 0 && col < testImageWidth) {
                gradColors[pixelIdx] = dLossDColors[camIdx][row][col];
                gradAlphas[pixelIdx] = dLossDAlphas[camIdx][row][col];
            }
        }

        sparseGradColorsList.push_back(gradColors);
        sparseGradAlphasList.push_back(gradAlphas);
    }

    auto sparseGradColors = fvdb::JaggedTensor(sparseGradColorsList);
    auto sparseGradAlphas = fvdb::JaggedTensor(sparseGradAlphasList);

    auto sparseBackwardResults = GaussianTestHelper::runBackwardSparse(gaussians,
                                                                       tiles,
                                                                       sparse,
                                                                       sparseResults,
                                                                       sparseGradColors,
                                                                       sparseGradAlphas,
                                                                       testImageWidth,
                                                                       testImageHeight,
                                                                       testTileSize);

    // Step 7: Compare sparse vs expected dense results
    std::string extraInfo = std::to_string(numCameras) + " camera(s)";
    GaussianTestHelper::compareSparseVsDense(sparseBackwardResults.dLossDMeans2d,
                                             expectedBackwardResults.dLossDMeans2d,
                                             "means2d",
                                             1e-4f,
                                             1e-6f,
                                             extraInfo);
    GaussianTestHelper::compareSparseVsDense(sparseBackwardResults.dLossDConics,
                                             expectedBackwardResults.dLossDConics,
                                             "conics",
                                             1e-1f,
                                             1e-4f,
                                             extraInfo);
    GaussianTestHelper::compareSparseVsDense(sparseBackwardResults.dLossDColors,
                                             expectedBackwardResults.dLossDColors,
                                             "colors",
                                             1e-4f,
                                             1e-6f,
                                             extraInfo);
    GaussianTestHelper::compareSparseVsDense(sparseBackwardResults.dLossDOpacities,
                                             expectedBackwardResults.dLossDOpacities,
                                             "opacities",
                                             1e-4f,
                                             1e-6f,
                                             extraInfo);
}

TEST_F(GaussianRasterizeTestFixture, TestSparseBackwardWithControlledGaussians) {
    // Test single camera
    testSparseBackwardWithControlledGaussians(1);
}

TEST_F(GaussianRasterizeTestFixture, TestSparseBackwardWithControlledGaussiansMultiCamera) {
    // Test multiple cameras
    testSparseBackwardWithControlledGaussians(3);
}

// Helper function to test sparse backward with multiple channels, parameterized by number of
// cameras
void
testSparseBackwardMultiChannel(int numCameras) {
    // Test sparse backward with different channel counts to validate chunked vs non-chunked paths
    const int numGaussians = 2;

    const int32_t testImageWidth  = 32;
    const int32_t testImageHeight = 32;
    const int32_t testTileSize    = 16;

    // Test different channel counts that exercise different code paths
    // These match the explicitly supported channel counts in the template instantiations
    // Using smaller subset to avoid override parameter complications
    std::vector<int> channelCounts = {1, 2, 3, 4, 5, 8, 9, 16, 17, 32, 33, 64, 65, 128, 129};

    for (int channels: channelCounts) {
        std::vector<std::pair<int, int>> pixelPositions = {{16, 16}, {24, 24}};
        auto gaussianPositions = GaussianTestHelper::getGaussianPositionsFromPixels(pixelPositions);

        auto gaussians = GaussianTestHelper::createTestGaussians(
            numCameras, numGaussians, channels, gaussianPositions);
        auto tiles = GaussianTestHelper::createTileStructure(
            testImageWidth, testImageHeight, testTileSize, gaussianPositions, numCameras);

        // Run forward dense
        auto denseResults = GaussianTestHelper::runForwardDense(
            gaussians, tiles, testImageWidth, testImageHeight, testTileSize);

        // Set up sparse pixels at the gaussian centers
        auto sparse = GaussianTestHelper::createSparseSetup(
            testTileSize, tiles.numTilesW, tiles.numTilesH, pixelPositions, numCameras);

        // Run forward sparse
        auto sparseResults = GaussianTestHelper::runForwardSparse(
            gaussians, tiles, sparse, testImageWidth, testImageHeight, testTileSize);

        // Create gradient inputs using helper
        auto dLossDColors =
            GaussianTestHelper::createMaskedGradients(denseResults.colors, pixelPositions, 1.0f);
        auto dLossDAlphas =
            GaussianTestHelper::createMaskedGradients(denseResults.alphas, pixelPositions, 1.0f);

        // Create sparse gradient inputs
        std::vector<torch::Tensor> sparseDLossDColorsList;
        std::vector<torch::Tensor> sparseDLossDAlphasList;

        for (int cam = 0; cam < numCameras; cam++) {
            auto sparseDLossDColorsData =
                torch::ones({2, channels}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
            auto sparseDLossDAlphasData =
                torch::ones({2, 1}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

            sparseDLossDColorsList.push_back(sparseDLossDColorsData);
            sparseDLossDAlphasList.push_back(sparseDLossDAlphasData);
        }

        auto sparseDLossDColorsJagged = fvdb::JaggedTensor(sparseDLossDColorsList);
        auto sparseDLossDAlphasJagged = fvdb::JaggedTensor(sparseDLossDAlphasList);

        // Test different numSharedChannelsOverride values for channels that support chunking
        std::vector<int64_t> overrideValues(1, -1); // Default
        overrideValues.reserve(3);
        if (channels > 32) {
            overrideValues.push_back(32);
        }
        if (channels > 16) {
            overrideValues.push_back(16);
        }

        for (int64_t override: overrideValues) {
            // Run backward dense with same override parameter
            auto denseBkwdResults = GaussianTestHelper::runBackwardDense(gaussians,
                                                                         tiles,
                                                                         denseResults,
                                                                         dLossDColors,
                                                                         dLossDAlphas,
                                                                         testImageWidth,
                                                                         testImageHeight,
                                                                         testTileSize,
                                                                         override);

            // Run backward sparse
            auto sparseBkwdResults = GaussianTestHelper::runBackwardSparse(gaussians,
                                                                           tiles,
                                                                           sparse,
                                                                           sparseResults,
                                                                           sparseDLossDColorsJagged,
                                                                           sparseDLossDAlphasJagged,
                                                                           testImageWidth,
                                                                           testImageHeight,
                                                                           testTileSize,
                                                                           override);

            // Compare results using helper
            std::string extraInfo = std::to_string(numCameras) + " cameras, " +
                                    std::to_string(channels) +
                                    " channels, override=" + std::to_string(override);
            GaussianTestHelper::compareSparseVsDense(sparseBkwdResults.dLossDMeans2d,
                                                     denseBkwdResults.dLossDMeans2d,
                                                     "means2d",
                                                     1e-4f,
                                                     1e-6f,
                                                     extraInfo);
            GaussianTestHelper::compareSparseVsDense(sparseBkwdResults.dLossDConics,
                                                     denseBkwdResults.dLossDConics,
                                                     "conics",
                                                     1e-1f,
                                                     1e-4f,
                                                     extraInfo);
            GaussianTestHelper::compareSparseVsDense(sparseBkwdResults.dLossDColors,
                                                     denseBkwdResults.dLossDColors,
                                                     "colors",
                                                     1e-4f,
                                                     1e-6f,
                                                     extraInfo);
            GaussianTestHelper::compareSparseVsDense(sparseBkwdResults.dLossDOpacities,
                                                     denseBkwdResults.dLossDOpacities,
                                                     "opacities",
                                                     1e-4f,
                                                     1e-6f,
                                                     extraInfo);
        }
    }
}

TEST_F(GaussianRasterizeTestFixture, TestSparseBackwardMultiChannelSingleCamera) {
    testSparseBackwardMultiChannel(1);
}

TEST_F(GaussianRasterizeTestFixture, TestSparseBackwardMultiChannelMultiCamera) {
    testSparseBackwardMultiChannel(3);
}

// NOTE: This is called as a backward pass so the forward pass will handle most of the error
//       checking. We just need to test that the backward pass doesn't throw.
// TODO: Test empty inputs
// TODO: Test error inputs
// TODO: Test with backgrounds
