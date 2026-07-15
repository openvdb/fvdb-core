// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <fvdb/detail/ops/gsplat/EvaluateSphericalHarmonicsForward.h>
#include <fvdb/detail/ops/gsplat/ProjectGaussiansAnalyticForward.h>
#include <fvdb/detail/ops/gsplat/RasterizeScreenSpaceGaussiansForward.h>
#include <fvdb/detail/ops/gsplat/RasterizeWorldSpaceGaussiansBackward.h>
#include <fvdb/detail/ops/gsplat/RasterizeWorldSpaceGaussiansForward.h>

#include <torch/types.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <tuple>

namespace {

using fvdb::detail::ops::DistortionModel;
using fvdb::detail::ops::RollingShutterType;

constexpr uint32_t kImageWidth  = 8;
constexpr uint32_t kImageHeight = 8;
constexpr uint32_t kTileSize    = 8;

struct WorldSpaceInputs {
    torch::Tensor means;
    torch::Tensor quats;
    torch::Tensor logScales;
    torch::Tensor features;
    torch::Tensor opacities;
    torch::Tensor worldToCam;
    torch::Tensor intrinsics;
    torch::Tensor distortion;
    torch::Tensor tileOffsets;
    torch::Tensor tileGaussianIds;
};

WorldSpaceInputs
makeInputs() {
    const auto floats = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    const auto ints   = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto intrinsics =
        torch::tensor({{{12.0f, 0.0f, 3.5f}, {0.0f, 12.0f, 3.5f}, {0.0f, 0.0f, 1.0f}}}, floats);
    return {
        torch::tensor({{0.05f, -0.03f, 2.5f}}, floats),
        torch::tensor({{0.97321693f, 0.05016582f, 0.20066328f, -0.10033164f}}, floats),
        torch::tensor({{-0.6f, -0.9f, -0.4f}}, floats),
        torch::tensor({{{0.2f, 0.6f, 0.9f}}}, floats),
        torch::tensor({{0.8f}}, floats),
        torch::eye(4, floats).unsqueeze(0),
        intrinsics,
        torch::zeros({1, 12}, floats),
        torch::zeros({1, 1, 1}, ints),
        torch::tensor({0}, ints),
    };
}

using RasterResult = std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>;
using RasterBackwardResult =
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>;

RasterResult
rasterize(const WorldSpaceInputs &inputs,
          const torch::Tensor &features,
          const DistortionModel cameraModel              = DistortionModel::PINHOLE,
          const at::optional<torch::Tensor> &backgrounds = at::nullopt,
          const at::optional<torch::Tensor> &masks       = at::nullopt) {
    return fvdb::detail::ops::rasterizeWorldSpaceGaussiansFwd(inputs.means,
                                                              inputs.quats,
                                                              inputs.logScales,
                                                              features,
                                                              inputs.opacities,
                                                              inputs.worldToCam,
                                                              inputs.worldToCam,
                                                              inputs.intrinsics,
                                                              inputs.distortion,
                                                              RollingShutterType::NONE,
                                                              cameraModel,
                                                              kImageWidth,
                                                              kImageHeight,
                                                              0,
                                                              0,
                                                              kTileSize,
                                                              inputs.tileOffsets,
                                                              inputs.tileGaussianIds,
                                                              backgrounds,
                                                              masks);
}

RasterBackwardResult
rasterizeBackward(const WorldSpaceInputs &inputs,
                  const torch::Tensor &features,
                  const RasterResult &forward,
                  const DistortionModel cameraModel              = DistortionModel::PINHOLE,
                  const at::optional<torch::Tensor> &backgrounds = at::nullopt,
                  const at::optional<torch::Tensor> &masks       = at::nullopt) {
    return fvdb::detail::ops::rasterizeWorldSpaceGaussiansBwd(
        inputs.means,
        inputs.quats,
        inputs.logScales,
        features,
        inputs.opacities,
        inputs.worldToCam,
        inputs.worldToCam,
        inputs.intrinsics,
        inputs.distortion,
        RollingShutterType::NONE,
        cameraModel,
        kImageWidth,
        kImageHeight,
        0,
        0,
        kTileSize,
        inputs.tileOffsets,
        inputs.tileGaussianIds,
        std::get<1>(forward),
        std::get<2>(forward),
        torch::ones_like(std::get<0>(forward)),
        torch::ones_like(std::get<1>(forward)),
        backgrounds,
        masks);
}

torch::Tensor
loss(const WorldSpaceInputs &inputs) {
    const auto result = rasterize(inputs, inputs.features);
    return std::get<0>(result).sum() + std::get<1>(result).sum();
}

void
expectFiniteRaster(const RasterResult &result, const int64_t channels) {
    EXPECT_TRUE(std::get<0>(result).sizes() ==
                torch::IntArrayRef({1, kImageHeight, kImageWidth, channels}));
    EXPECT_TRUE(std::get<1>(result).sizes() ==
                torch::IntArrayRef({1, kImageHeight, kImageWidth, 1}));
    EXPECT_TRUE(torch::isfinite(std::get<0>(result)).all().item<bool>());
    EXPECT_TRUE(torch::isfinite(std::get<1>(result)).all().item<bool>());
    EXPECT_GT(std::get<1>(result).sum().item<float>(), 0.0f);
}

} // namespace

TEST(GaussianRasterizeWorldSpaceTest, ForwardSupportsSh0AndHigherOrderFeatures) {
    const WorldSpaceInputs inputs = makeInputs();
    const auto intOptions = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    const auto radii      = torch::ones({1, 1, 2}, intOptions);
    const auto sh0        = torch::tensor({{{0.2f, -0.1f, 0.4f}}}, inputs.means.options());
    const auto noShN      = torch::empty({1, 0, 3}, inputs.means.options());
    const auto shN        = torch::full({1, 3, 3}, 0.1f, inputs.means.options());
    const auto noIds      = torch::empty({0}, intOptions);

    const auto sh0Features = fvdb::detail::ops::evaluateSphericalHarmonicsFwd(
        0, 1, inputs.means, inputs.worldToCam, noIds, noIds, sh0, noShN, radii);
    const auto shNFeatures = fvdb::detail::ops::evaluateSphericalHarmonicsFwd(
        1, 1, inputs.means, inputs.worldToCam, noIds, noIds, sh0, shN, radii);

    expectFiniteRaster(rasterize(inputs, sh0Features), 3);
    expectFiniteRaster(rasterize(inputs, shNFeatures), 3);
    EXPECT_FALSE(torch::allclose(sh0Features, shNFeatures));
}

TEST(GaussianRasterizeWorldSpaceTest, BackwardMatchesFiniteDifference) {
    const WorldSpaceInputs inputs = makeInputs();
    const auto forward            = rasterize(inputs, inputs.features);
    const auto backward           = rasterizeBackward(inputs, inputs.features, forward);

    constexpr float step   = 1e-3f;
    WorldSpaceInputs plus  = inputs;
    WorldSpaceInputs minus = inputs;
    plus.means             = inputs.means.clone();
    minus.means            = inputs.means.clone();
    plus.means.index_put_({0, 0}, inputs.means.index({0, 0}).item<float>() + step);
    minus.means.index_put_({0, 0}, inputs.means.index({0, 0}).item<float>() - step);
    const float finiteDifference =
        (loss(plus).item<float>() - loss(minus).item<float>()) / (2 * step);
    const float analytic = std::get<0>(backward).index({0, 0}).item<float>();

    EXPECT_NEAR(analytic, finiteDifference, 2e-2f + 2e-2f * std::abs(finiteDifference));
    EXPECT_TRUE(torch::isfinite(std::get<1>(backward)).all().item<bool>());
    EXPECT_TRUE(torch::isfinite(std::get<2>(backward)).all().item<bool>());
    EXPECT_TRUE(torch::isfinite(std::get<3>(backward)).all().item<bool>());
    EXPECT_TRUE(torch::isfinite(std::get<4>(backward)).all().item<bool>());
}

TEST(GaussianRasterizeWorldSpaceTest, MasksBackgroundsAndNoIntersections) {
    const WorldSpaceInputs inputs = makeInputs();
    const auto background         = torch::tensor({{0.1f, -0.2f, 0.3f}}, inputs.means.options());
    const auto mask =
        torch::zeros({1, 1, 1}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));

    const auto masked =
        rasterize(inputs, inputs.features, DistortionModel::PINHOLE, background, mask);
    const auto expectedBackground = background.view({1, 1, 1, 3}).expand_as(std::get<0>(masked));
    EXPECT_TRUE(torch::equal(std::get<0>(masked), expectedBackground));
    EXPECT_EQ(std::get<1>(masked).count_nonzero().item<int64_t>(), 0);
    const auto maskedBackward = rasterizeBackward(
        inputs, inputs.features, masked, DistortionModel::PINHOLE, background, mask);
    EXPECT_EQ(std::get<0>(maskedBackward).count_nonzero().item<int64_t>(), 0);
    EXPECT_EQ(std::get<1>(maskedBackward).count_nonzero().item<int64_t>(), 0);
    EXPECT_EQ(std::get<2>(maskedBackward).count_nonzero().item<int64_t>(), 0);
    EXPECT_EQ(std::get<3>(maskedBackward).count_nonzero().item<int64_t>(), 0);
    EXPECT_EQ(std::get<4>(maskedBackward).count_nonzero().item<int64_t>(), 0);

    WorldSpaceInputs empty = inputs;
    empty.tileGaussianIds =
        torch::empty({0}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    const auto noIntersections =
        rasterize(empty, empty.features, DistortionModel::PINHOLE, background);
    EXPECT_TRUE(torch::equal(std::get<0>(noIntersections), expectedBackground));
    EXPECT_EQ(std::get<1>(noIntersections).count_nonzero().item<int64_t>(), 0);
}

TEST(GaussianRasterizeWorldSpaceTest, SupportsPinholeOrthographicAndOpenCvCameras) {
    const WorldSpaceInputs inputs = makeInputs();
    for (const DistortionModel cameraModel: {DistortionModel::PINHOLE,
                                             DistortionModel::ORTHOGRAPHIC,
                                             DistortionModel::OPENCV_RADTAN_5}) {
        const auto forward = rasterize(inputs, inputs.features, cameraModel);
        expectFiniteRaster(forward, 3);
        const auto backward = rasterizeBackward(inputs, inputs.features, forward, cameraModel);
        EXPECT_TRUE(torch::isfinite(std::get<0>(backward)).all().item<bool>());
        EXPECT_TRUE(torch::isfinite(std::get<1>(backward)).all().item<bool>());
        EXPECT_TRUE(torch::isfinite(std::get<2>(backward)).all().item<bool>());
        EXPECT_GT(std::get<0>(backward).abs().sum().item<float>(), 0.0f);
    }
}

TEST(GaussianRasterizeWorldSpaceTest, PinholeAndOrthographicIgnoreDistortionCoefficients) {
    const WorldSpaceInputs undistorted = makeInputs();
    WorldSpaceInputs nonzeroDistortion = undistorted;
    nonzeroDistortion.distortion       = torch::full_like(undistorted.distortion, 0.05f);

    for (const DistortionModel cameraModel:
         {DistortionModel::PINHOLE, DistortionModel::ORTHOGRAPHIC}) {
        const auto expected = rasterize(undistorted, undistorted.features, cameraModel);
        const auto actual   = rasterize(nonzeroDistortion, nonzeroDistortion.features, cameraModel);

        EXPECT_TRUE(torch::equal(std::get<0>(actual), std::get<0>(expected)));
        EXPECT_TRUE(torch::equal(std::get<1>(actual), std::get<1>(expected)));
    }
}

TEST(GaussianRasterizeWorldSpaceTest, DepthAndRgbdChannelsAreConsistent) {
    const WorldSpaceInputs inputs = makeInputs();
    const auto depths             = inputs.means.select(1, 2).view({1, 1, 1});
    const auto rgbdFeatures       = torch::cat({inputs.features, depths}, 2);

    for (const DistortionModel cameraModel: {DistortionModel::PINHOLE,
                                             DistortionModel::ORTHOGRAPHIC,
                                             DistortionModel::OPENCV_RADTAN_5}) {
        const auto depth = rasterize(inputs, depths, cameraModel);
        const auto rgbd  = rasterize(inputs, rgbdFeatures, cameraModel);

        EXPECT_TRUE(
            torch::allclose(std::get<0>(depth),
                            std::get<0>(rgbd).index({torch::indexing::Ellipsis, 3}).unsqueeze(-1)));
        EXPECT_TRUE(torch::equal(std::get<1>(depth), std::get<1>(rgbd)));
    }
}

TEST(GaussianRasterizeWorldSpaceTest, HasStructuralParityWithProjectedRasterPath) {
    const WorldSpaceInputs inputs = makeInputs();
    const auto depth              = inputs.means.select(1, 2).view({1, 1, 1});
    const auto structuralFeatures = torch::cat({inputs.features, depth}, 2);
    const auto projection         = fvdb::detail::ops::projectGaussiansAnalyticFwd(inputs.means,
                                                                           inputs.quats,
                                                                           inputs.logScales,
                                                                           inputs.worldToCam,
                                                                           inputs.intrinsics,
                                                                           kImageWidth,
                                                                           kImageHeight,
                                                                           0.3f,
                                                                           0.01f,
                                                                           100.0f,
                                                                           0.0f,
                                                                           false,
                                                                           false);
    const auto &radii             = std::get<0>(projection);
    const auto &means2d           = std::get<1>(projection);
    const auto &conics            = std::get<3>(projection);
    ASSERT_GT(radii.sum().item<int64_t>(), 0);
    const auto projected =
        fvdb::detail::ops::rasterizeScreenSpaceGaussiansFwd(means2d,
                                                            conics,
                                                            structuralFeatures,
                                                            inputs.opacities,
                                                            kImageWidth,
                                                            kImageHeight,
                                                            0,
                                                            0,
                                                            kTileSize,
                                                            inputs.tileOffsets,
                                                            inputs.tileGaussianIds);
    const auto world            = rasterize(inputs, structuralFeatures);
    const auto worldAlpha       = std::get<1>(world).squeeze(-1);
    const auto projectedAlpha   = std::get<1>(projected).squeeze(-1);
    const auto worldSupport     = worldAlpha > 1e-4f;
    const auto projectedSupport = projectedAlpha > 1e-4f;
    const float intersection    = worldSupport.logical_and(projectedSupport).sum().item<float>();
    const float supportUnion    = worldSupport.logical_or(projectedSupport).sum().item<float>();
    ASSERT_GT(supportUnion, 0.0f);
    EXPECT_GT(intersection / supportUnion, 0.5f);

    const auto x             = torch::arange(kImageWidth, inputs.means.options()).view({1, 1, -1});
    const auto y             = torch::arange(kImageHeight, inputs.means.options()).view({1, -1, 1});
    const auto worldMass     = worldAlpha.sum();
    const auto projectedMass = projectedAlpha.sum();
    ASSERT_GT(worldMass.item<float>(), 0.0f);
    ASSERT_GT(projectedMass.item<float>(), 0.0f);
    EXPECT_LT(
        torch::abs((worldAlpha * x).sum() / worldMass - (projectedAlpha * x).sum() / projectedMass)
            .item<float>(),
        1.0f);
    EXPECT_LT(
        torch::abs((worldAlpha * y).sum() / worldMass - (projectedAlpha * y).sum() / projectedMass)
            .item<float>(),
        1.0f);

    const auto worldMean     = std::get<0>(world).sum({0, 1, 2}) / worldMass;
    const auto projectedMean = std::get<0>(projected).sum({0, 1, 2}) / projectedMass;
    EXPECT_TRUE(torch::allclose(worldMean.index({torch::indexing::Slice(0, 3)}),
                                projectedMean.index({torch::indexing::Slice(0, 3)}),
                                5e-2,
                                5e-2));
    EXPECT_NEAR(worldMean.index({3}).item<float>(), projectedMean.index({3}).item<float>(), 5e-2f);
}
