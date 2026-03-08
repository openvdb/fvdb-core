// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <fvdb/GaussianSplat3d.h>

#include <c10/util/Exception.h>
#include <torch/cuda.h>
#include <torch/script.h>

#include <gtest/gtest.h>

#include <optional>
#include <string>

namespace {

using CameraModel      = fvdb::GaussianSplat3d::CameraModel;
using ProjectionMethod = fvdb::GaussianSplat3d::ProjectionMethod;

template <typename Fn>
void
expectTorchErrorContains(Fn &&fn, const std::string &messageSubstring) {
    try {
        fn();
        FAIL() << "Expected c10::Error containing: " << messageSubstring;
    } catch (const c10::Error &e) {
        EXPECT_NE(std::string(e.what()).find(messageSubstring), std::string::npos)
            << "Actual error was: " << e.what();
    }
}

struct GaussianSplat3dCameraApiTest : public ::testing::Test {
    void
    SetUp() override {
        torch::manual_seed(0);
        if (!torch::cuda::is_available()) {
            GTEST_SKIP() << "CUDA is not available; skipping GaussianSplat3d camera API tests.";
        }
    }

    static constexpr int64_t kImageWidth  = 32;
    static constexpr int64_t kImageHeight = 24;
    static constexpr float kNearPlane     = 0.05f;
    static constexpr float kFarPlane      = 20.0f;

    static fvdb::GaussianSplat3d
    makeSimpleGaussianSplat() {
        auto opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);

        const torch::Tensor means =
            torch::tensor({{0.18f, -0.12f, 2.8f}, {-0.08f, 0.10f, 3.4f}}, opts);
        const torch::Tensor quats =
            torch::tensor({{1.0f, 0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f}}, opts);
        const torch::Tensor logScales =
            torch::log(torch::tensor({{0.06f, 0.05f, 0.04f}, {0.05f, 0.07f, 0.06f}}, opts));
        const torch::Tensor logitOpacities = torch::tensor({2.2f, 1.8f}, opts);
        const torch::Tensor sh0 =
            torch::tensor({{{0.7f, 0.1f, -0.2f}}, {{-0.3f, 0.5f, 0.4f}}}, opts);
        const torch::Tensor shN = torch::empty({2, 0, 3}, opts);

        return fvdb::GaussianSplat3d(
            means, quats, logScales, logitOpacities, sh0, shN, false, false, false);
    }

    static torch::Tensor
    makeWorldToCameraMatrices(const int64_t C) {
        auto worldToCamera =
            torch::eye(4, torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32))
                .unsqueeze(0)
                .repeat({C, 1, 1});
        auto acc = worldToCamera.accessor<float, 3>();
        for (int64_t c = 0; c < C; ++c) {
            acc[c][0][3] = 0.03f * static_cast<float>(c);
            acc[c][1][3] = -0.02f * static_cast<float>(c);
        }
        return worldToCamera.cuda().contiguous();
    }

    static torch::Tensor
    makeProjectionMatrices(const int64_t C, const CameraModel cameraModel) {
        auto projectionMatrices = torch::zeros(
            {C, 3, 3}, torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32));
        auto acc = projectionMatrices.accessor<float, 3>();
        for (int64_t c = 0; c < C; ++c) {
            const float fx =
                cameraModel == CameraModel::ORTHOGRAPHIC ? 9.0f + 0.5f * c : 18.0f + 1.5f * c;
            const float fy =
                cameraModel == CameraModel::ORTHOGRAPHIC ? 8.5f + 0.5f * c : 17.0f + 1.25f * c;
            acc[c][0][0] = fx;
            acc[c][1][1] = fy;
            acc[c][0][2] = (static_cast<float>(kImageWidth) - 1.0f) / 2.0f + 0.3f * c;
            acc[c][1][2] = (static_cast<float>(kImageHeight) - 1.0f) / 2.0f - 0.2f * c;
            acc[c][2][2] = 1.0f;
        }
        return projectionMatrices.cuda().contiguous();
    }

    static torch::Tensor
    makeDistortionCoeffs(const int64_t C) {
        auto distortionCoeffs = torch::zeros(
            {C, 12}, torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32));
        auto acc = distortionCoeffs.accessor<float, 2>();
        for (int64_t c = 0; c < C; ++c) {
            const float s = static_cast<float>(c + 1);
            acc[c][0]     = 0.02f * s;
            acc[c][1]     = -0.004f * s;
            acc[c][2]     = 0.001f * s;
            acc[c][6]     = 0.0015f * s;
            acc[c][7]     = -0.0012f * s;
        }
        return distortionCoeffs.cuda().contiguous();
    }
};

TEST_F(GaussianSplat3dCameraApiTest, ProjectionMethodAutoResolvesByCameraModel) {
    auto gs           = makeSimpleGaussianSplat();
    auto worldToCam   = makeWorldToCameraMatrices(1);
    auto pinholeProj  = makeProjectionMatrices(1, CameraModel::PINHOLE);
    auto orthoProj    = makeProjectionMatrices(1, CameraModel::ORTHOGRAPHIC);
    auto distortion   = makeDistortionCoeffs(1);
    auto pinholeState = gs.projectGaussiansForImages(worldToCam,
                                                     pinholeProj,
                                                     kImageWidth,
                                                     kImageHeight,
                                                     kNearPlane,
                                                     kFarPlane,
                                                     CameraModel::PINHOLE,
                                                     ProjectionMethod::AUTO,
                                                     std::nullopt,
                                                     0);
    auto orthoState   = gs.projectGaussiansForImages(worldToCam,
                                                   orthoProj,
                                                   kImageWidth,
                                                   kImageHeight,
                                                   kNearPlane,
                                                   kFarPlane,
                                                   CameraModel::ORTHOGRAPHIC,
                                                   ProjectionMethod::AUTO,
                                                   std::nullopt,
                                                   0);
    auto utState      = gs.projectGaussiansForImages(worldToCam,
                                                pinholeProj,
                                                kImageWidth,
                                                kImageHeight,
                                                kNearPlane,
                                                kFarPlane,
                                                CameraModel::PINHOLE,
                                                ProjectionMethod::UNSCENTED,
                                                std::nullopt,
                                                0);
    auto opencvState  = gs.projectGaussiansForImages(worldToCam,
                                                    pinholeProj,
                                                    kImageWidth,
                                                    kImageHeight,
                                                    kNearPlane,
                                                    kFarPlane,
                                                    CameraModel::OPENCV_RADTAN_5,
                                                    ProjectionMethod::AUTO,
                                                    distortion,
                                                    0);

    EXPECT_EQ(pinholeState.cameraModel(), CameraModel::PINHOLE);
    EXPECT_EQ(pinholeState.projectionMethod(), ProjectionMethod::ANALYTIC);
    EXPECT_EQ(orthoState.cameraModel(), CameraModel::ORTHOGRAPHIC);
    EXPECT_EQ(orthoState.projectionMethod(), ProjectionMethod::ANALYTIC);
    EXPECT_EQ(utState.cameraModel(), CameraModel::PINHOLE);
    EXPECT_EQ(utState.projectionMethod(), ProjectionMethod::UNSCENTED);
    EXPECT_EQ(opencvState.cameraModel(), CameraModel::OPENCV_RADTAN_5);
    EXPECT_EQ(opencvState.projectionMethod(), ProjectionMethod::UNSCENTED);
}

TEST_F(GaussianSplat3dCameraApiTest, CameraApiValidationRejectsInvalidArguments) {
    auto gs         = makeSimpleGaussianSplat();
    auto worldToCam = makeWorldToCameraMatrices(1);
    auto projection = makeProjectionMatrices(1, CameraModel::PINHOLE);
    auto distortion = makeDistortionCoeffs(1);

    expectTorchErrorContains(
        [&]() {
            gs.projectGaussiansForImages(worldToCam,
                                         projection,
                                         kImageWidth,
                                         kImageHeight,
                                         kNearPlane,
                                         kFarPlane,
                                         CameraModel::OPENCV_RADTAN_5,
                                         ProjectionMethod::AUTO,
                                         std::nullopt,
                                         0);
        },
        "distortionCoeffs must be provided");

    expectTorchErrorContains(
        [&]() {
            gs.projectGaussiansForImages(
                worldToCam,
                projection,
                kImageWidth,
                kImageHeight,
                kNearPlane,
                kFarPlane,
                CameraModel::OPENCV_RADTAN_5,
                ProjectionMethod::AUTO,
                distortion.index({torch::indexing::Slice(), torch::indexing::Slice(0, 5)}),
                0);
        },
        "distortionCoeffs must have shape");

    expectTorchErrorContains(
        [&]() {
            gs.renderImages(worldToCam,
                            projection,
                            kImageWidth,
                            kImageHeight,
                            kNearPlane,
                            kFarPlane,
                            CameraModel::PINHOLE,
                            ProjectionMethod::AUTO,
                            distortion,
                            0);
        },
        "distortionCoeffs must be None");

    expectTorchErrorContains(
        [&]() {
            gs.renderImagesFromWorld(worldToCam,
                                     projection,
                                     kImageWidth,
                                     kImageHeight,
                                     kNearPlane,
                                     kFarPlane,
                                     CameraModel::OPENCV_RADTAN_5,
                                     ProjectionMethod::ANALYTIC,
                                     distortion,
                                     0);
        },
        "ProjectionMethod::UNSCENTED or AUTO");

    expectTorchErrorContains(
        [&]() {
            gs.projectGaussiansForImages(worldToCam,
                                         projection.transpose(1, 2),
                                         kImageWidth,
                                         kImageHeight,
                                         kNearPlane,
                                         kFarPlane,
                                         CameraModel::PINHOLE,
                                         ProjectionMethod::AUTO,
                                         std::nullopt,
                                         0);
        },
        "projectionMatrices must be contiguous");

    expectTorchErrorContains(
        [&]() {
            gs.projectGaussiansForImages(worldToCam.transpose(1, 2),
                                         projection,
                                         kImageWidth,
                                         kImageHeight,
                                         kNearPlane,
                                         kFarPlane,
                                         CameraModel::PINHOLE,
                                         ProjectionMethod::AUTO,
                                         std::nullopt,
                                         0);
        },
        "worldToCameraMatrices must be contiguous");
}

TEST_F(GaussianSplat3dCameraApiTest, ProjectedRenderMatchesDenseProjectedApis) {
    auto gs = makeSimpleGaussianSplat();

    for (const CameraModel cameraModel:
         {CameraModel::PINHOLE, CameraModel::ORTHOGRAPHIC, CameraModel::OPENCV_RADTAN_5}) {
        SCOPED_TRACE(static_cast<int>(cameraModel));
        const auto worldToCam = makeWorldToCameraMatrices(1);
        const auto projection = makeProjectionMatrices(1, cameraModel);
        const auto distortion = cameraModel == CameraModel::OPENCV_RADTAN_5
                                    ? std::optional<torch::Tensor>(makeDistortionCoeffs(1))
                                    : std::nullopt;

        const auto projectedImages = gs.projectGaussiansForImages(worldToCam,
                                                                  projection,
                                                                  kImageWidth,
                                                                  kImageHeight,
                                                                  kNearPlane,
                                                                  kFarPlane,
                                                                  cameraModel,
                                                                  ProjectionMethod::AUTO,
                                                                  distortion,
                                                                  0);
        const auto [imagesFromProjected, imageAlphasFromProjected] =
            gs.renderFromProjectedGaussians(projectedImages);
        const auto [imagesFromDense, imageAlphasFromDense] = gs.renderImages(worldToCam,
                                                                             projection,
                                                                             kImageWidth,
                                                                             kImageHeight,
                                                                             kNearPlane,
                                                                             kFarPlane,
                                                                             cameraModel,
                                                                             ProjectionMethod::AUTO,
                                                                             distortion,
                                                                             0);

        EXPECT_TRUE(torch::allclose(imagesFromProjected, imagesFromDense, 1e-6, 1e-6));
        EXPECT_TRUE(torch::allclose(imageAlphasFromProjected, imageAlphasFromDense, 1e-6, 1e-6));

        const auto projectedDepths = gs.projectGaussiansForDepths(worldToCam,
                                                                  projection,
                                                                  kImageWidth,
                                                                  kImageHeight,
                                                                  kNearPlane,
                                                                  kFarPlane,
                                                                  cameraModel,
                                                                  ProjectionMethod::AUTO,
                                                                  distortion);
        const auto [depthsFromProjected, depthAlphasFromProjected] =
            gs.renderFromProjectedGaussians(projectedDepths);
        const auto [depthsFromDense, depthAlphasFromDense] = gs.renderDepths(worldToCam,
                                                                             projection,
                                                                             kImageWidth,
                                                                             kImageHeight,
                                                                             kNearPlane,
                                                                             kFarPlane,
                                                                             cameraModel,
                                                                             ProjectionMethod::AUTO,
                                                                             distortion);

        EXPECT_TRUE(torch::allclose(depthsFromProjected, depthsFromDense, 1e-6, 1e-6));
        EXPECT_TRUE(torch::allclose(depthAlphasFromProjected, depthAlphasFromDense, 1e-6, 1e-6));

        const auto projectedRgbd = gs.projectGaussiansForImagesAndDepths(worldToCam,
                                                                         projection,
                                                                         kImageWidth,
                                                                         kImageHeight,
                                                                         kNearPlane,
                                                                         kFarPlane,
                                                                         cameraModel,
                                                                         ProjectionMethod::AUTO,
                                                                         distortion,
                                                                         0);
        const auto [rgbdFromProjected, rgbdAlphasFromProjected] =
            gs.renderFromProjectedGaussians(projectedRgbd);
        const auto [rgbdFromDense, rgbdAlphasFromDense] =
            gs.renderImagesAndDepths(worldToCam,
                                     projection,
                                     kImageWidth,
                                     kImageHeight,
                                     kNearPlane,
                                     kFarPlane,
                                     cameraModel,
                                     ProjectionMethod::AUTO,
                                     distortion,
                                     0);

        EXPECT_TRUE(torch::allclose(rgbdFromProjected, rgbdFromDense, 1e-6, 1e-6));
        EXPECT_TRUE(torch::allclose(rgbdAlphasFromProjected, rgbdAlphasFromDense, 1e-6, 1e-6));
    }
}

TEST_F(GaussianSplat3dCameraApiTest, WorldSpaceRenderVariantsShareAlphaAndPacking) {
    auto gs = makeSimpleGaussianSplat();

    for (const CameraModel cameraModel:
         {CameraModel::PINHOLE, CameraModel::ORTHOGRAPHIC, CameraModel::OPENCV_RADTAN_5}) {
        SCOPED_TRACE(static_cast<int>(cameraModel));
        const auto worldToCam = makeWorldToCameraMatrices(1);
        const auto projection = makeProjectionMatrices(1, cameraModel);
        const auto distortion = cameraModel == CameraModel::OPENCV_RADTAN_5
                                    ? std::optional<torch::Tensor>(makeDistortionCoeffs(1))
                                    : std::nullopt;

        const auto [images, imageAlphas] = gs.renderImagesFromWorld(worldToCam,
                                                                    projection,
                                                                    kImageWidth,
                                                                    kImageHeight,
                                                                    kNearPlane,
                                                                    kFarPlane,
                                                                    cameraModel,
                                                                    ProjectionMethod::AUTO,
                                                                    distortion,
                                                                    0);
        const auto [depths, depthAlphas] = gs.renderDepthsFromWorld(worldToCam,
                                                                    projection,
                                                                    kImageWidth,
                                                                    kImageHeight,
                                                                    kNearPlane,
                                                                    kFarPlane,
                                                                    cameraModel,
                                                                    ProjectionMethod::AUTO,
                                                                    distortion);
        const auto [rgbd, rgbdAlphas]    = gs.renderImagesAndDepthsFromWorld(worldToCam,
                                                                          projection,
                                                                          kImageWidth,
                                                                          kImageHeight,
                                                                          kNearPlane,
                                                                          kFarPlane,
                                                                          cameraModel,
                                                                          ProjectionMethod::AUTO,
                                                                          distortion,
                                                                          0);

        EXPECT_TRUE(torch::allclose(imageAlphas, depthAlphas, 1e-5, 1e-5));
        EXPECT_TRUE(torch::allclose(imageAlphas, rgbdAlphas, 1e-5, 1e-5));
        EXPECT_TRUE(torch::allclose(rgbd.index({torch::indexing::Slice(),
                                                torch::indexing::Slice(),
                                                torch::indexing::Slice(),
                                                torch::indexing::Slice(0, 3)}),
                                    images,
                                    1e-5,
                                    1e-5));
        EXPECT_TRUE(torch::allclose(rgbd.index({torch::indexing::Slice(),
                                                torch::indexing::Slice(),
                                                torch::indexing::Slice(),
                                                torch::indexing::Slice(3, 4)}),
                                    depths,
                                    1e-5,
                                    1e-5));
    }
}

} // namespace
