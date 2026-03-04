// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include "utils/ImageUtils.h"
#include "utils/Tensor.h"

#include <fvdb/detail/ops/gsplat/GaussianRasterizeForward.h>
#include <fvdb/detail/ops/gsplat/GaussianSplatSparse.h>
#include <fvdb/detail/ops/gsplat/GaussianTileIntersection.h>

#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/script.h>
#include <torch/types.h>

#include <gtest/gtest.h>

#if defined(__linux__)
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <cstdlib>
#include <string>
#include <vector>

#ifndef FVDB_EXTERNAL_TEST_DATA_PATH
#error "FVDB_EXTERNAL_TEST_DATA_PATH must be defined"
#endif

namespace {
constexpr const char *kMaskedEdgeTileChildEnv = "FVDB_GSPLAT_MASKED_EDGE_TILE_CHILD";
} // namespace

struct GaussianRasterizeForwardTestFixture : public ::testing::Test {
    void
    loadInputData(const std::string insPath) {
        const std::string dataPath =
            std::string(FVDB_EXTERNAL_TEST_DATA_PATH) + std::string("/unit_tests/gsplat");
        const std::string inputsPath = dataPath + std::string("/") + insPath;

        std::vector<torch::Tensor> inputs = fvdb::test::loadTensors(inputsPath, inputNames);
        means2d                           = inputs[0].cuda();
        conics                            = inputs[1].cuda();
        colors                            = inputs[2].cuda();
        opacities                         = inputs[3].cuda();
        tileOffsets                       = inputs[4].cuda();
        tileGaussianIds                   = inputs[5].cuda();
        imageDims                         = inputs[6].cuda();

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
    loadTestData(const std::string insPath, const std::string outsPath) {
        // Set the random seed for reproducibility.
        torch::manual_seed(0);

        loadInputData(insPath);

        const std::string dataPath =
            std::string(FVDB_EXTERNAL_TEST_DATA_PATH) + std::string("/unit_tests/gsplat");
        const std::string expectedOutputsPath = dataPath + std::string("/") + outsPath;

        std::vector<torch::Tensor> expectedOutputs =
            fvdb::test::loadTensors(expectedOutputsPath, outputNames);

        imageHeight  = expectedOutputs[0].size(1);
        imageWidth   = expectedOutputs[0].size(2);
        imageOriginW = 0;
        imageOriginH = 0;
        tileSize     = 16;

        expectedRenderedColors = expectedOutputs[0].cuda();
        expectedRenderedAlphas = expectedOutputs[1].cuda();
        expectedLastIds        = expectedOutputs[2].cuda();
    }

    fvdb::JaggedTensor
    generateSparsePixelCoords(int numCameras, int maxPixelsPerCamera) {
        // Create a list of tensors, one for each camera
        std::vector<torch::Tensor> pixelCoordsList;
        for (int i = 0; i < numCameras; i++) {
            // Generate random number of pixels for this camera (up to maxPixelsPerCamera)
            int numPixels = torch::randint(1, maxPixelsPerCamera + 1, {1}).item<int>();

            // Generate random pixel coordinates within image bounds
            auto const xCoords = torch::randint(0, imageWidth, {numPixels});
            auto const yCoords = torch::randint(0, imageHeight, {numPixels});

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

    // @brief Compares sparse pixel data with corresponding pixels in dense images
    // @param sparsePixelCoords JaggedTensor containing 2D pixel coordinates [camera][pixel_id, 2]
    // @param sparseImageData JaggedTensor containing data for sparse pixels [camera][pixel_id, D]
    // @param denseImageData Tensor containing dense image data [C, image_height, image_width, D]
    // @return True if all sparse pixel values match their corresponding dense image values
    static bool
    compareSparseWithDensePixels(const fvdb::JaggedTensor &sparsePixelCoords,
                                 const fvdb::JaggedTensor &sparseImageFeatures,
                                 const fvdb::JaggedTensor &sparseImageAlphas,
                                 const fvdb::JaggedTensor &sparseImageLastIds,
                                 const torch::Tensor &denseImageFeatures,
                                 const torch::Tensor &denseImageAlphas,
                                 const torch::Tensor &denseImageLastIds) {
        TORCH_CHECK(
            sparsePixelCoords.num_outer_lists() == sparseImageFeatures.num_outer_lists(),
            "Number of cameras must match between sparsePixelCoords and sparseImageFeatures ",
            std::to_string(sparsePixelCoords.num_outer_lists()) +
                " != " + std::to_string(sparseImageFeatures.num_outer_lists()));
        TORCH_CHECK(sparsePixelCoords.num_outer_lists() == denseImageFeatures.size(0),
                    "Number of cameras must match between sparse tensors and denseImageData ",
                    std::to_string(sparsePixelCoords.num_outer_lists()) +
                        " != " + std::to_string(denseImageFeatures.size(0)));
        TORCH_CHECK(sparsePixelCoords.num_outer_lists() == sparseImageAlphas.num_outer_lists(),
                    "Number of cameras must match between sparsePixelCoords and sparseImageAlphas ",
                    std::to_string(sparsePixelCoords.num_outer_lists()) +
                        " != " + std::to_string(sparseImageAlphas.num_outer_lists()));
        TORCH_CHECK(
            sparsePixelCoords.num_outer_lists() == sparseImageLastIds.num_outer_lists(),
            "Number of cameras must match between sparsePixelCoords and sparseImageLastIds ",
            std::to_string(sparsePixelCoords.num_outer_lists()) +
                " != " + std::to_string(sparseImageLastIds.num_outer_lists()));

        const int numCameras = sparsePixelCoords.num_outer_lists();
        const int channels   = denseImageFeatures.size(3);

        for (int camIdx = 0; camIdx < numCameras; ++camIdx) {
            const auto sparseFeatures = sparseImageFeatures.index(camIdx).jdata();
            const auto sparseAlphas   = sparseImageAlphas.index(camIdx).jdata();
            const auto sparseLastIds  = sparseImageLastIds.index(camIdx).jdata();
            const auto coords         = sparsePixelCoords.index(camIdx).jdata();

            TORCH_CHECK(coords.size(0) == sparseFeatures.size(0),
                        "Number of pixels must match between coordinates and data for camera ",
                        std::to_string(camIdx) + ": " + std::to_string(coords.size(0)) +
                            " != " + std::to_string(sparseFeatures.size(0)));
            TORCH_CHECK(coords.size(0) == sparseAlphas.size(0),
                        "Number of pixels must match between coordinates and alphas for camera ",
                        std::to_string(camIdx) + ": " + std::to_string(coords.size(0)) +
                            " != " + std::to_string(sparseAlphas.size(0)));
            TORCH_CHECK(coords.size(0) == sparseLastIds.size(0),
                        "Number of pixels must match between coordinates and lastIds for camera ",
                        std::to_string(camIdx) + ": " + std::to_string(coords.size(0)) +
                            " != " + std::to_string(sparseLastIds.size(0)));
            TORCH_CHECK(coords.size(1) == 2,
                        "Pixel coordinates must have shape [numPixels, 2] for camera ",
                        camIdx);
            TORCH_CHECK(sparseFeatures.size(1) == channels,
                        "Sparse features must have shape [numPixels, channels] for camera ",
                        camIdx);
            TORCH_CHECK(sparseAlphas.size(1) == 1,
                        "Sparse alphas must have shape [numPixels, 1] for camera ",
                        camIdx);

            const int numPixels = coords.size(0);

            for (int pixelIdx = 0; pixelIdx < numPixels; ++pixelIdx) {
                const int row = coords[pixelIdx][0].item<int>();
                const int col = coords[pixelIdx][1].item<int>();

                // Check if coordinates are within image bounds
                if (row < 0 || row >= denseImageFeatures.size(1) || col < 0 ||
                    col >= denseImageFeatures.size(2)) {
                    std::cout << "Pixel out of bounds: (" << row << "," << col << ")" << std::endl;
                    return false;
                }

                // Extract the corresponding pixel from the dense image
                torch::Tensor denseFeature  = denseImageFeatures[camIdx][row][col];
                torch::Tensor denseAlpha    = denseImageAlphas[camIdx][row][col];
                torch::Tensor denseLastId   = denseImageLastIds[camIdx][row][col];
                torch::Tensor sparseFeature = sparseFeatures[pixelIdx];
                torch::Tensor sparseAlpha   = sparseAlphas[pixelIdx];
                torch::Tensor sparseLastId  = sparseLastIds[pixelIdx];

                // Compare values
                if (!torch::allclose(denseFeature, sparseFeature)) {
                    std::cout << "Dense feature " << pixelIdx << " at (" << row << "," << col
                              << "): " << denseFeature << std::endl;
                    std::cout << "Sparse feature " << pixelIdx << ": " << sparseFeature
                              << std::endl;
                    return false;
                }
                if (!torch::allclose(denseAlpha, sparseAlpha)) {
                    std::cout << "Dense alpha " << pixelIdx << " at (" << row << "," << col
                              << "): " << denseAlpha << std::endl;
                    std::cout << "Sparse alpha " << pixelIdx << ": " << sparseAlpha << std::endl;
                    return false;
                }
                if (!torch::equal(denseLastId, sparseLastId)) {
                    std::cout << "Dense lastId " << pixelIdx << " at (" << row << "," << col
                              << "): " << denseLastId << std::endl;
                    std::cout << "Sparse lastId " << pixelIdx << ": " << sparseLastId << std::endl;
                    return false;
                }
            }
        }

        return true;
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
        means2d                = means2d.to(device);
        conics                 = conics.to(device);
        colors                 = colors.to(device);
        opacities              = opacities.to(device);
        tileOffsets            = tileOffsets.to(device);
        tileGaussianIds        = tileGaussianIds.to(device);
        imageDims              = imageDims.to(device);
        expectedRenderedColors = expectedRenderedColors.to(device);
        expectedRenderedAlphas = expectedRenderedAlphas.to(device);
        expectedLastIds        = expectedLastIds.to(device);
    }

    const std::vector<std::string> inputNames  = {"means2d",
                                                  "conics",
                                                  "colors",
                                                  "opacities",
                                                  "tile_offsets",
                                                  "tile_gaussian_ids",
                                                  "image_dims"};
    const std::vector<std::string> outputNames = {"rendered_colors", "rendered_alphas", "last_ids"};

    // Input tensors
    torch::Tensor means2d;             // [C, N, 2] or [nnz, 2]
    torch::Tensor conics;              // [C, N, 3] or [nnz, 3]
    torch::Tensor colors;              // [C, N, D] or [nnz, D]
    torch::Tensor opacities;           // [C, N] or [nnz]
    torch::Tensor tileOffsets;         // [C, tileHeight, tileWidth]
    torch::Tensor tileGaussianIds;     // [nIsects]
    torch::Tensor imageDims;           // [2]

    fvdb::JaggedTensor pixelsToRender; // [C, maxPixelsPerCamera, 2]

    // Expected output tensors
    torch::Tensor expectedRenderedColors; // [C, imageHeight, imageWidth, D]
    torch::Tensor expectedRenderedAlphas; // [C, imageHeight, imageWidth, 1]
    torch::Tensor expectedLastIds;        // [C, imageHeight, imageWidth]

    // Parameters
    uint32_t imageWidth;
    uint32_t imageHeight;
    uint32_t imageOriginW;
    uint32_t imageOriginH;
    uint32_t tileSize;
};

TEST(GaussianRasterizeForwardMaskedEdgeTile, Child) {
#if !defined(__linux__)
    GTEST_SKIP() << "This regression test is Linux-only.";
#else
    const char *isChild = std::getenv(kMaskedEdgeTileChildEnv);
    if (!(isChild && std::string(isChild) == "1")) {
        GTEST_SKIP() << "Not running child path.";
    }

    if (c10::cuda::device_count() <= 0) {
        GTEST_SKIP() << "CUDA not available.";
    }

    const at::cuda::CUDAGuard device_guard(0);

    constexpr int64_t C = 1;

    constexpr uint32_t imageWidth  = 17;
    constexpr uint32_t imageHeight = 17;
    constexpr uint32_t tileSize    = 16;
    constexpr uint32_t tileExtentH = (imageHeight + tileSize - 1) / tileSize; // 2
    constexpr uint32_t tileExtentW = (imageWidth + tileSize - 1) / tileSize;  // 2

    auto fopts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);

    // Single Gaussian that intersects the bottom-right edge tile.
    const auto means2d   = torch::tensor({{{16.5f, 16.5f}}}, fopts);                  // [C,N,2]
    const auto conics    = torch::tensor({{{1.0f, 0.0f, 1.0f}}}, fopts);              // [C,N,3]
    const auto features  = torch::tensor({{{0.4f, 0.5f, -0.6f}}}, fopts);             // [C,N,D]
    const auto opacities = torch::tensor({{0.9f}}, fopts);                            // [C,N]
    const auto radii     = torch::tensor(
        {{1}}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32)); // [C,N]
    const auto depths      = torch::tensor({{1.0f}}, fopts);                          // [C,N]
    const auto backgrounds = torch::tensor({{0.1f, -0.2f, 0.3f}}, fopts);             // [C,D]

    auto masks     = torch::ones({C, (int64_t)tileExtentH, (int64_t)tileExtentW},
                             torch::TensorOptions().device(torch::kCUDA).dtype(torch::kBool));
    masks[0][1][1] = false; // mask out bottom-right edge tile

    auto [tileOffsets, tileGaussianIds] =
        fvdb::detail::ops::dispatchGaussianTileIntersection<torch::kCUDA>(
            means2d, radii, depths, at::nullopt, (uint32_t)C, tileSize, tileExtentH, tileExtentW);

    auto [outFeatures, outAlphas, outLastIds] =
        fvdb::detail::ops::dispatchGaussianRasterizeForward<torch::kCUDA>(means2d,
                                                                          conics,
                                                                          features,
                                                                          opacities,
                                                                          imageWidth,
                                                                          imageHeight,
                                                                          0,
                                                                          0,
                                                                          tileSize,
                                                                          tileOffsets,
                                                                          tileGaussianIds,
                                                                          backgrounds,
                                                                          masks);

    (void)outLastIds;

    // Ensure the kernel completed (this would hang if there is a deadlock).
    C10_CUDA_CHECK(cudaDeviceSynchronize());

    // The only in-bounds pixel in the bottom-right edge tile is (16,16). It should be filled
    // with background and alpha=0.
    const auto outFeaturesCpu = outFeatures.cpu();
    const auto outAlphasCpu   = outAlphas.cpu();

    EXPECT_EQ(outAlphasCpu[0][16][16][0].item<float>(), 0.0f);
    EXPECT_EQ(outFeaturesCpu[0][16][16][0].item<float>(), 0.1f);
    EXPECT_EQ(outFeaturesCpu[0][16][16][1].item<float>(), -0.2f);
    EXPECT_EQ(outFeaturesCpu[0][16][16][2].item<float>(), 0.3f);
#endif
}

TEST(GaussianRasterizeForwardMaskedEdgeTile, NoDeadlock) {
#if !defined(__linux__)
    GTEST_SKIP() << "This regression test is Linux-only.";
#else
    const char *isChild = std::getenv(kMaskedEdgeTileChildEnv);
    if (isChild && std::string(isChild) == "1") {
        GTEST_SKIP() << "Running child path.";
    }

    pid_t pid = fork();
    ASSERT_GE(pid, 0) << "fork() failed";

    if (pid == 0) {
        // Child: exec the same test binary but run only the child test.
        setenv(kMaskedEdgeTileChildEnv, "1", 1);
        execl("/proc/self/exe",
              "/proc/self/exe",
              "--gtest_filter=GaussianRasterizeForwardMaskedEdgeTile.Child",
              "--gtest_color=no",
              (char *)nullptr);
        _exit(127);
    }

    int status = 0;
    bool exited{false};
    constexpr int kTimeoutMs = 20000;
    for (int elapsed = 0; elapsed < kTimeoutMs; elapsed += 50) {
        const pid_t r = waitpid(pid, &status, WNOHANG);
        if (r == pid) {
            exited = true;
            break;
        }
        usleep(50 * 1000);
    }

    if (!exited) {
        kill(pid, SIGKILL);
        (void)waitpid(pid, &status, 0);
        FAIL() << "Deadlock detected: child process timed out.";
    }

    ASSERT_TRUE(WIFEXITED(status)) << "Child did not exit cleanly.";
    ASSERT_EQ(WEXITSTATUS(status), 0) << "Child test failed.";
#endif
}

// This is a helper function to generate the output data for the test cases.
// Only enable this test when you want to update the output data.
TEST_F(GaussianRasterizeForwardTestFixture, DISABLED_GenerateOutputData) {
    // Load test data using our helper method
    loadInputData("rasterize_forward_inputs.pt");

    // Test with 3 channels
    {
        const auto [renderedColors, renderedAlphas, lastIds] =
            fvdb::detail::ops::dispatchGaussianRasterizeForward<torch::kCUDA>(means2d,
                                                                              conics,
                                                                              colors,
                                                                              opacities,
                                                                              imageWidth,
                                                                              imageHeight,
                                                                              imageOriginW,
                                                                              imageOriginH,
                                                                              tileSize,
                                                                              tileOffsets,
                                                                              tileGaussianIds);

        std::vector<torch::Tensor> outputData = {renderedColors, renderedAlphas, lastIds};

        auto outputFilename = std::string("rasterize_forward_outputs.pt");

        storeData(outputFilename, outputData);
    }

    // Test with 64 channels
    {
        auto colors_64 = catChannelsToDim(colors, 64);

        const auto [renderedColors, renderedAlphas, lastIds] =
            fvdb::detail::ops::dispatchGaussianRasterizeForward<torch::kCUDA>(means2d,
                                                                              conics,
                                                                              colors_64,
                                                                              opacities,
                                                                              imageWidth / 2,
                                                                              imageHeight / 2,
                                                                              imageOriginW,
                                                                              imageOriginH,
                                                                              tileSize,
                                                                              tileOffsets,
                                                                              tileGaussianIds);

        std::vector<torch::Tensor> outputData = {renderedColors, renderedAlphas, lastIds};

        auto outputFilename = std::string("rasterize_forward_outputs_64.pt");
        storeData(outputFilename, outputData);
    }
}

TEST_F(GaussianRasterizeForwardTestFixture, TestBasicInputsAndOutputs) {
    loadTestData("rasterize_forward_inputs.pt", "rasterize_forward_outputs.pt");

    const auto [outColors, outAlphas, outLastIds] =
        fvdb::detail::ops::dispatchGaussianRasterizeForward<torch::kCUDA>(means2d,
                                                                          conics,
                                                                          colors,
                                                                          opacities,
                                                                          imageWidth,
                                                                          imageHeight,
                                                                          imageOriginW,
                                                                          imageOriginH,
                                                                          tileSize,
                                                                          tileOffsets,
                                                                          tileGaussianIds);

    EXPECT_TRUE(torch::allclose(outColors, expectedRenderedColors));
    EXPECT_TRUE(torch::allclose(outAlphas, expectedRenderedAlphas));
    EXPECT_TRUE(torch::equal(outLastIds, expectedLastIds));
}

TEST_F(GaussianRasterizeForwardTestFixture, TestConcatenatedChannels) {
    loadTestData("rasterize_forward_inputs.pt", "rasterize_forward_outputs_64.pt");

    colors                 = catChannelsToDim(colors, 64);
    expectedRenderedColors = catChannelsToDim(expectedRenderedColors, 64);

    const auto [outColors, outAlphas, outLastIds] =
        fvdb::detail::ops::dispatchGaussianRasterizeForward<torch::kCUDA>(means2d,
                                                                          conics,
                                                                          colors,
                                                                          opacities,
                                                                          imageWidth,
                                                                          imageHeight,
                                                                          imageOriginW,
                                                                          imageOriginH,
                                                                          tileSize,
                                                                          tileOffsets,
                                                                          tileGaussianIds);

    EXPECT_TRUE(torch::allclose(outColors, expectedRenderedColors));
    EXPECT_TRUE(torch::allclose(outAlphas, expectedRenderedAlphas));
    EXPECT_TRUE(torch::equal(outLastIds, expectedLastIds));
}

// Compares the output of multi-camera rasterization with the output of sequential single-camera
// rasterization.
TEST_F(GaussianRasterizeForwardTestFixture, TestMultipleCameras) {
    // the output here is not used in this test.
    loadTestData("rasterize_forward_inputs_3cams.pt", "rasterize_forward_outputs.pt");

    // run all 3 cameras at once
    const auto [outColorsAll, outAlphasAll, outLastIdsAll] =
        fvdb::detail::ops::dispatchGaussianRasterizeForward<torch::kCUDA>(means2d,
                                                                          conics,
                                                                          colors,
                                                                          opacities,
                                                                          imageWidth,
                                                                          imageHeight,
                                                                          imageOriginW,
                                                                          imageOriginH,
                                                                          tileSize,
                                                                          tileOffsets,
                                                                          tileGaussianIds);

    // rasterize each camera individually
    std::vector<torch::Tensor> outColorsList;
    std::vector<torch::Tensor> outAlphasList;
    std::vector<torch::Tensor> outLastIdsList;
    for (int i = 0; i < means2d.size(0); i++) {
        // extract the ith camera data from each tensor and add a leading dim of 1
        auto means2d_1cam     = means2d[i].unsqueeze(0);
        auto conics_1cam      = conics[i].unsqueeze(0);
        auto colors_1cam      = colors[i].unsqueeze(0);
        auto opacities_1cam   = opacities[i].unsqueeze(0);
        auto tileOffsets_1cam = tileOffsets[i].unsqueeze(0);

        auto numCameras = means2d.size(0);

        // find the start and end of the ith camera in tileGaussianIds
        auto start = tileOffsets[i][0][0].item<int64_t>();
        auto end   = i == numCameras - 1 ? tileGaussianIds.numel()
                                         : tileOffsets[i + 1][0][0].item<int64_t>();

        // slice out this camera's tileGaussianIds and adjust to 0-based
        auto tileGaussianIds_1cam =
            tileGaussianIds.index({torch::indexing::Slice(start, end)}) - i * means2d.size(1);

        // Adjust the tileOffsets to be 0-based
        tileOffsets_1cam = tileOffsets_1cam - start;

        // Kernel receives adjusted offsets and 0-based IDs for this camera
        auto [outColors, outAlphas, outLastIds] =
            fvdb::detail::ops::dispatchGaussianRasterizeForward<torch::kCUDA>(means2d_1cam,
                                                                              conics_1cam,
                                                                              colors_1cam,
                                                                              opacities_1cam,
                                                                              imageWidth,
                                                                              imageHeight,
                                                                              imageOriginW,
                                                                              imageOriginH,
                                                                              tileSize,
                                                                              tileOffsets_1cam,
                                                                              tileGaussianIds_1cam);

        // add start offset back to non-background pixels
        outLastIds = outLastIds + start;
        // mask out the background pixels
        auto background_mask = (outLastIdsAll[i].unsqueeze(0) == -1);
        outLastIds.masked_fill_(background_mask, -1);

        outColorsList.push_back(outColors);
        outAlphasList.push_back(outAlphas);
        outLastIdsList.push_back(outLastIds); // Store the raw output index

// Uncomment to dump binarized lastIds images for comparison
#if 0
        // binarize the outLastIds tensor, stack it to a 3-channel image and write a PNG
        auto binarizedLastIds = (outLastIds[0] == -1);

        auto binarizedLastIds_3ch =
            torch::stack({ binarizedLastIds, binarizedLastIds, binarizedLastIds }, -1)
                .to(torch::kFloat32);
        auto alpha = torch::ones_like(outAlphas[0]);
        // print shape of binarizedLastIds_3ch
        std::cout << "binarizedLastIds_3ch shape: " << binarizedLastIds_3ch.sizes() << std::endl;
        fvdb::test::writePNG(binarizedLastIds_3ch.cpu(), alpha.cpu(),
                             "test_output_camera_" + std::to_string(i) + "_binarized.png");

        // also write the i-th 3-camera image binarized
        auto binarizedLastIdsAll = (outLastIdsAll[i] == -1);
        auto binarizedLastIdsAll_3ch =
            torch::stack({ binarizedLastIdsAll, binarizedLastIdsAll, binarizedLastIdsAll }, -1)
                .to(torch::kFloat32);
        std::cout << "binarizedLastIdsAll_3ch shape: " << binarizedLastIdsAll_3ch.sizes()
                  << std::endl;
        fvdb::test::writePNG(binarizedLastIdsAll_3ch.cpu(), alpha.cpu(),
                             "test_output_camera_" + std::to_string(i) + "_binarized_all.png");

        // compute a difference image between the two
        auto diff = binarizedLastIds_3ch - binarizedLastIdsAll_3ch;
        std::cout << "diff shape: " << diff.sizes() << std::endl;
        fvdb::test::writePNG(diff.cpu(), alpha.cpu(),
                             "test_output_camera_" + std::to_string(i) + "_diff.png");
#endif

// Uncomment to dump color images for comparison
#if 0
        // write out the ith camera's image
        fvdb::test::writePNG(outColors[0].cpu(), alpha.cpu(), // outAlphas[0].cpu(),
                             "test_output_camera_" + std::to_string(i) + ".png");
        // write out the three images from the single-pass rasterization
        fvdb::test::writePNG(outColorsAll[i].cpu(), alpha.cpu(), // outAlphasAll[i].cpu(),
                             "test_output_camera_" + std::to_string(i) + "_all.png");
#endif
    }

    auto combinedColors  = torch::cat(outColorsList, 0);
    auto combinedAlphas  = torch::cat(outAlphasList, 0);
    auto combinedLastIds = torch::cat(outLastIdsList, 0);

    EXPECT_TRUE(torch::allclose(combinedColors, outColorsAll));
    EXPECT_TRUE(torch::allclose(combinedAlphas, outAlphasAll));
    EXPECT_TRUE(torch::equal(combinedLastIds, outLastIdsAll));
}

TEST_F(GaussianRasterizeForwardTestFixture, TestMultipleCamerasWithBackgrounds) {
    loadTestData("rasterize_forward_inputs_3cams.pt", "rasterize_forward_outputs.pt");

    const int numCameras  = means2d.size(0);
    const int numChannels = colors.size(-1);

    // Create different background colors for each camera
    // Camera 0: Red [1, 0, 0]
    // Camera 1: Green [0, 1, 0]
    // Camera 2: Blue [0, 0, 1]
    torch::Tensor backgrounds = torch::zeros({numCameras, numChannels}, colors.options());
    backgrounds[0][0]         = 1.0f; // Red
    if (numCameras > 1)
        backgrounds[1][1] = 1.0f;     // Green
    if (numCameras > 2)
        backgrounds[2][2] = 1.0f;     // Blue

    // Render without background
    const auto [outColorsNoBackground, outAlphasNoBackground, outLastIdsNoBackground] =
        fvdb::detail::ops::dispatchGaussianRasterizeForward<torch::kCUDA>(means2d,
                                                                          conics,
                                                                          colors,
                                                                          opacities,
                                                                          imageWidth,
                                                                          imageHeight,
                                                                          imageOriginW,
                                                                          imageOriginH,
                                                                          tileSize,
                                                                          tileOffsets,
                                                                          tileGaussianIds);

    // Render with different background per camera
    const auto [outColorsWithBackground, outAlphasWithBackground, outLastIdsWithBackground] =
        fvdb::detail::ops::dispatchGaussianRasterizeForward<torch::kCUDA>(means2d,
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
                                                                          backgrounds);

    // Alphas and last IDs should be identical regardless of background
    EXPECT_TRUE(torch::allclose(outAlphasNoBackground, outAlphasWithBackground));
    EXPECT_TRUE(torch::equal(outLastIdsNoBackground, outLastIdsWithBackground));

    // Verify that we have at least some non-opaque pixels to actually test background blending
    auto nonOpaquePixels = (outAlphasNoBackground < 0.99f).sum();
    EXPECT_GT(nonOpaquePixels.item<int64_t>(), 0)
        << "Test requires some non-opaque pixels to validate background blending";

    // Compute expected colors with background blending and compare per camera
    torch::Tensor expectedColorsWithBackground = torch::zeros_like(outColorsNoBackground);
    for (int c = 0; c < numCameras; c++) {
        auto alpha     = outAlphasNoBackground[c];                 // [H, W, 1]
        auto baseColor = outColorsNoBackground[c];                 // [H, W, D]
        auto bg        = backgrounds[c].view({1, 1, numChannels}); // [1, 1, D]

        // Expected: renderedColor + (1 - alpha) * background
        expectedColorsWithBackground[c] = baseColor + (1.0f - alpha) * bg;
    }

    EXPECT_TRUE(torch::allclose(outColorsWithBackground, expectedColorsWithBackground, 1e-5, 1e-5));
}

TEST_F(GaussianRasterizeForwardTestFixture, TestSparseRasterization) {
    loadTestData("rasterize_forward_inputs.pt", "rasterize_forward_outputs.pt");

    const int numCameras = means2d.size(0);

    auto const pixelsToRender = generateSparsePixelCoords(numCameras, 100).cuda();

    auto [activeTiles, activeTileMask, tilePixelMask, tilePixelCumsum, pixelMap] =
        fvdb::detail::ops::computeSparseInfo(
            tileSize, tileOffsets.size(2), tileOffsets.size(1), pixelsToRender);

    const auto [outColorsSparse, outAlphasSparse, outLastIdsSparse] =
        fvdb::detail::ops::dispatchGaussianSparseRasterizeForward<torch::kCUDA>(pixelsToRender,
                                                                                means2d,
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
                                                                                activeTiles,
                                                                                tilePixelMask,
                                                                                tilePixelCumsum,
                                                                                pixelMap);

    EXPECT_TRUE(compareSparseWithDensePixels(pixelsToRender,
                                             outColorsSparse,
                                             outAlphasSparse,
                                             outLastIdsSparse,
                                             expectedRenderedColors,
                                             expectedRenderedAlphas,
                                             expectedLastIds));
}

TEST_F(GaussianRasterizeForwardTestFixture, TestSparseRasterizationConcatenatedChannels) {
    loadTestData("rasterize_forward_inputs.pt", "rasterize_forward_outputs_64.pt");

    colors                 = catChannelsToDim(colors, 64);
    expectedRenderedColors = catChannelsToDim(expectedRenderedColors, 64);

    const int numCameras = means2d.size(0);

    auto const pixelsToRender = generateSparsePixelCoords(numCameras, 100).cuda();

    auto [activeTiles, activeTileMask, tilePixelMask, tilePixelCumsum, pixelMap] =
        fvdb::detail::ops::computeSparseInfo(
            tileSize, tileOffsets.size(2), tileOffsets.size(1), pixelsToRender);

    const auto [outColorsSparse, outAlphasSparse, outLastIdsSparse] =
        fvdb::detail::ops::dispatchGaussianSparseRasterizeForward<torch::kCUDA>(pixelsToRender,
                                                                                means2d,
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
                                                                                activeTiles,
                                                                                tilePixelMask,
                                                                                tilePixelCumsum,
                                                                                pixelMap);

    EXPECT_TRUE(compareSparseWithDensePixels(pixelsToRender,
                                             outColorsSparse,
                                             outAlphasSparse,
                                             outLastIdsSparse,
                                             expectedRenderedColors,
                                             expectedRenderedAlphas,
                                             expectedLastIds));
}

TEST_F(GaussianRasterizeForwardTestFixture, TestSparseRasterizationMultipleCameras) {
    // the output here is not used in this test.
    loadTestData("rasterize_forward_inputs_3cams.pt", "rasterize_forward_outputs.pt");

    const int numCameras = means2d.size(0);

    auto const pixelsToRender = generateSparsePixelCoords(numCameras, 100).cuda();

    auto [activeTiles, activeTileMask, tilePixelMask, tilePixelCumsum, pixelMap] =
        fvdb::detail::ops::computeSparseInfo(
            tileSize, tileOffsets.size(2), tileOffsets.size(1), pixelsToRender);

    // run all 3 cameras at once
    const auto [outColorsAll, outAlphasAll, outLastIdsAll] =
        fvdb::detail::ops::dispatchGaussianRasterizeForward<torch::kCUDA>(means2d,
                                                                          conics,
                                                                          colors,
                                                                          opacities,
                                                                          imageWidth,
                                                                          imageHeight,
                                                                          imageOriginW,
                                                                          imageOriginH,
                                                                          tileSize,
                                                                          tileOffsets,
                                                                          tileGaussianIds);

    const auto [outColorsSparse, outAlphasSparse, outLastIdsSparse] =
        fvdb::detail::ops::dispatchGaussianSparseRasterizeForward<torch::kCUDA>(pixelsToRender,
                                                                                means2d,
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
                                                                                activeTiles,
                                                                                tilePixelMask,
                                                                                tilePixelCumsum,
                                                                                pixelMap);

    EXPECT_TRUE(compareSparseWithDensePixels(pixelsToRender,
                                             outColorsSparse,
                                             outAlphasSparse,
                                             outLastIdsSparse,
                                             outColorsAll,
                                             outAlphasAll,
                                             outLastIdsAll));
}

TEST_F(GaussianRasterizeForwardTestFixture, TestSparseRasterizationMultipleCamerasWithBackgrounds) {
    loadTestData("rasterize_forward_inputs_3cams.pt", "rasterize_forward_outputs.pt");

    const int numCameras  = means2d.size(0);
    const int numChannels = colors.size(-1);

    // Create different background colors for each camera
    // Camera 0: Red [1, 0, 0]
    // Camera 1: Green [0, 1, 0]
    // Camera 2: Blue [0, 0, 1]
    torch::Tensor backgrounds = torch::zeros({numCameras, numChannels}, colors.options());
    backgrounds[0][0]         = 1.0f; // Red
    if (numCameras > 1)
        backgrounds[1][1] = 1.0f;     // Green
    if (numCameras > 2)
        backgrounds[2][2] = 1.0f;     // Blue

    auto const pixelsToRender = generateSparsePixelCoords(numCameras, 100).cuda();

    auto [activeTiles, activeTileMask, tilePixelMask, tilePixelCumsum, pixelMap] =
        fvdb::detail::ops::computeSparseInfo(
            tileSize, tileOffsets.size(2), tileOffsets.size(1), pixelsToRender);

    // Render sparse without background
    const auto [outColorsSparseNoBackground,
                outAlphasSparseNoBackground,
                outLastIdsSparseNoBackground] =
        fvdb::detail::ops::dispatchGaussianSparseRasterizeForward<torch::kCUDA>(pixelsToRender,
                                                                                means2d,
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
                                                                                activeTiles,
                                                                                tilePixelMask,
                                                                                tilePixelCumsum,
                                                                                pixelMap);

    // Render sparse with different background per camera
    const auto [outColorsSparseWithBackground,
                outAlphasSparseWithBackground,
                outLastIdsSparseWithBackground] =
        fvdb::detail::ops::dispatchGaussianSparseRasterizeForward<torch::kCUDA>(pixelsToRender,
                                                                                means2d,
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
                                                                                activeTiles,
                                                                                tilePixelMask,
                                                                                tilePixelCumsum,
                                                                                pixelMap,
                                                                                backgrounds);

    // Alphas and last IDs should be identical regardless of background
    for (int c = 0; c < numCameras; c++) {
        EXPECT_TRUE(torch::allclose(outAlphasSparseNoBackground.index(c).jdata(),
                                    outAlphasSparseWithBackground.index(c).jdata()));
        EXPECT_TRUE(torch::equal(outLastIdsSparseNoBackground.index(c).jdata(),
                                 outLastIdsSparseWithBackground.index(c).jdata()));
    }

    // Verify that we have at least some non-opaque pixels
    bool hasTransparentPixels = false;
    for (int c = 0; c < numCameras; c++) {
        auto alphas          = outAlphasSparseNoBackground.index(c).jdata();
        auto nonOpaquePixels = (alphas < 0.99f).sum();
        if (nonOpaquePixels.item<int64_t>() > 0) {
            hasTransparentPixels = true;
            break;
        }
    }
    EXPECT_TRUE(hasTransparentPixels)
        << "Test requires some non-opaque pixels to validate background blending";

    // Compute expected colors with background blending and compare per camera
    for (int c = 0; c < numCameras; c++) {
        auto alpha     = outAlphasSparseNoBackground.index(c).jdata(); // [N, 1]
        auto baseColor = outColorsSparseNoBackground.index(c).jdata(); // [N, D]
        auto bg        = backgrounds[c].view({1, numChannels});        // [1, D]

        // Expected: renderedColor + (1 - alpha) * background
        auto expectedColors = baseColor + (1.0f - alpha) * bg;
        auto actualColors   = outColorsSparseWithBackground.index(c).jdata();

        EXPECT_TRUE(torch::allclose(actualColors, expectedColors, 1e-5, 1e-5))
            << "Background blending mismatch for camera " << c;
    }
}

// Test packed mode rasterization with multiple cameras.
// This verifies that when means2d has shape [nnz, 2] (packed) instead of [C, N, 2] (non-packed),
// the rasterization produces the same results as non-packed mode.
// This specifically tests the fix for deriving numCameras from tileOffsets instead of means2d.
TEST_F(GaussianRasterizeForwardTestFixture, TestPackedModeMultipleCameras) {
    loadTestData("rasterize_forward_inputs_3cams.pt", "rasterize_forward_outputs.pt");

    const int numCameras         = means2d.size(0);
    const int numGaussiansPerCam = means2d.size(1);
    const int totalGaussians     = numCameras * numGaussiansPerCam;

    ASSERT_GT(numCameras, 1) << "This test requires multiple cameras";

    // Step 1: Run non-packed rasterization to get expected results
    const auto [expectedColors, expectedAlphas, expectedLastIds] =
        fvdb::detail::ops::dispatchGaussianRasterizeForward<torch::kCUDA>(means2d,
                                                                          conics,
                                                                          colors,
                                                                          opacities,
                                                                          imageWidth,
                                                                          imageHeight,
                                                                          imageOriginW,
                                                                          imageOriginH,
                                                                          tileSize,
                                                                          tileOffsets,
                                                                          tileGaussianIds);

    // Step 2: Reshape tensors to packed format [nnz, D]
    // The test data's tileGaussianIds already contains global indices (0 to C*N-1).
    // In non-packed mode, the kernel converts these to [cid][gid] internally.
    // In packed mode, the kernel uses them directly as indices into the packed tensor.
    auto means2dPacked   = means2d.reshape({totalGaussians, 2});
    auto conicsPacked    = conics.reshape({totalGaussians, 3});
    auto colorsPacked    = colors.reshape({totalGaussians, colors.size(-1)});
    auto opacitiesPacked = opacities.reshape({totalGaussians});

    // Step 3: Run packed rasterization with same tileOffsets and tileGaussianIds
    const auto [outColorsPacked, outAlphasPacked, outLastIdsPacked] =
        fvdb::detail::ops::dispatchGaussianRasterizeForward<torch::kCUDA>(means2dPacked,
                                                                          conicsPacked,
                                                                          colorsPacked,
                                                                          opacitiesPacked,
                                                                          imageWidth,
                                                                          imageHeight,
                                                                          imageOriginW,
                                                                          imageOriginH,
                                                                          tileSize,
                                                                          tileOffsets,
                                                                          tileGaussianIds);

    // Step 4: Compare results
    // The output shapes should match: [C, H, W, D] for colors, [C, H, W, 1] for alphas
    EXPECT_EQ(outColorsPacked.sizes(), expectedColors.sizes())
        << "Packed output colors shape mismatch";
    EXPECT_EQ(outAlphasPacked.sizes(), expectedAlphas.sizes())
        << "Packed output alphas shape mismatch";
    EXPECT_EQ(outLastIdsPacked.sizes(), expectedLastIds.sizes())
        << "Packed output lastIds shape mismatch";

    // The rendered colors and alphas should match (allowing for small numerical differences)
    EXPECT_TRUE(torch::allclose(outColorsPacked, expectedColors, 1e-4, 1e-4))
        << "Packed mode colors don't match non-packed mode";
    EXPECT_TRUE(torch::allclose(outAlphasPacked, expectedAlphas, 1e-4, 1e-4))
        << "Packed mode alphas don't match non-packed mode";

    // lastIds should be identical since both modes use the same global indices
    EXPECT_TRUE(torch::equal(outLastIdsPacked, expectedLastIds))
        << "Packed mode lastIds don't match non-packed mode";
}

// Test packed mode with sparse rasterization and multiple cameras
TEST_F(GaussianRasterizeForwardTestFixture, TestPackedModeSparseMultipleCameras) {
    loadTestData("rasterize_forward_inputs_3cams.pt", "rasterize_forward_outputs.pt");

    const int numCameras         = means2d.size(0);
    const int numGaussiansPerCam = means2d.size(1);
    const int totalGaussians     = numCameras * numGaussiansPerCam;

    ASSERT_GT(numCameras, 1) << "This test requires multiple cameras";

    // Generate sparse pixel coordinates to render
    auto const pixelsToRender = generateSparsePixelCoords(numCameras, 100).cuda();

    // Compute sparse info from pixels
    auto [activeTiles, activeTileMask, tilePixelMask, tilePixelCumsum, pixelMap] =
        fvdb::detail::ops::computeSparseInfo(
            tileSize, tileOffsets.size(2), tileOffsets.size(1), pixelsToRender);

    // Step 1: Run non-packed sparse rasterization to get expected results
    const auto [expectedColorsSparse, expectedAlphasSparse, expectedLastIdsSparse] =
        fvdb::detail::ops::dispatchGaussianSparseRasterizeForward<torch::kCUDA>(pixelsToRender,
                                                                                means2d,
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
                                                                                activeTiles,
                                                                                tilePixelMask,
                                                                                tilePixelCumsum,
                                                                                pixelMap);

    // Step 2: Reshape tensors to packed format [nnz, D]
    // The test data's tileGaussianIds already contains global indices (0 to C*N-1).
    // In non-packed mode, the kernel converts these to [cid][gid] internally.
    // In packed mode, the kernel uses them directly as indices into the packed tensor.
    auto means2dPacked   = means2d.reshape({totalGaussians, 2});
    auto conicsPacked    = conics.reshape({totalGaussians, 3});
    auto colorsPacked    = colors.reshape({totalGaussians, colors.size(-1)});
    auto opacitiesPacked = opacities.reshape({totalGaussians});

    // Step 3: Run packed sparse rasterization with same sparse info and same gaussian IDs
    const auto [outColorsPacked, outAlphasPacked, outLastIdsPacked] =
        fvdb::detail::ops::dispatchGaussianSparseRasterizeForward<torch::kCUDA>(pixelsToRender,
                                                                                means2dPacked,
                                                                                conicsPacked,
                                                                                colorsPacked,
                                                                                opacitiesPacked,
                                                                                imageWidth,
                                                                                imageHeight,
                                                                                imageOriginW,
                                                                                imageOriginH,
                                                                                tileSize,
                                                                                tileOffsets,
                                                                                tileGaussianIds,
                                                                                activeTiles,
                                                                                tilePixelMask,
                                                                                tilePixelCumsum,
                                                                                pixelMap);

    // Step 4: Compare results
    EXPECT_EQ(outColorsPacked.num_outer_lists(), expectedColorsSparse.num_outer_lists())
        << "Packed sparse output has wrong number of cameras";

    for (int c = 0; c < numCameras; ++c) {
        auto expectedColors = expectedColorsSparse.index(c).jdata();
        auto actualColors   = outColorsPacked.index(c).jdata();
        auto expectedAlphas = expectedAlphasSparse.index(c).jdata();
        auto actualAlphas   = outAlphasPacked.index(c).jdata();

        EXPECT_EQ(actualColors.sizes(), expectedColors.sizes())
            << "Packed sparse colors shape mismatch for camera " << c;
        EXPECT_EQ(actualAlphas.sizes(), expectedAlphas.sizes())
            << "Packed sparse alphas shape mismatch for camera " << c;

        EXPECT_TRUE(torch::allclose(actualColors, expectedColors, 1e-4, 1e-4))
            << "Packed sparse mode colors don't match for camera " << c;
        EXPECT_TRUE(torch::allclose(actualAlphas, expectedAlphas, 1e-4, 1e-4))
            << "Packed sparse mode alphas don't match for camera " << c;
    }
}

TEST_F(GaussianRasterizeForwardTestFixture, CPUThrows) {
    loadTestData("rasterize_forward_inputs.pt", "rasterize_forward_outputs.pt");
    moveToDevice(torch::kCPU);
    EXPECT_THROW(fvdb::detail::ops::dispatchGaussianRasterizeForward<torch::kCPU>(means2d,
                                                                                  conics,
                                                                                  colors,
                                                                                  opacities,
                                                                                  imageWidth,
                                                                                  imageHeight,
                                                                                  imageOriginW,
                                                                                  imageOriginH,
                                                                                  tileSize,
                                                                                  tileOffsets,
                                                                                  tileGaussianIds),
                 c10::NotImplementedError);
}
