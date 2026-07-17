// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <fvdb/detail/ops/gsplat/ProjectGaussiansAnalyticForward.h>
#include <fvdb/detail/ops/gsplat/ProjectGaussiansAnalyticJaggedBackward.h>
#include <fvdb/detail/ops/gsplat/ProjectGaussiansAnalyticJaggedForward.h>

#include <torch/types.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <tuple>
#include <vector>

namespace {

constexpr uint32_t kImageWidth  = 64;
constexpr uint32_t kImageHeight = 48;
constexpr float kEps2d          = 0.3f;
constexpr float kNearPlane      = 0.01f;
constexpr float kFarPlane       = 100.0f;

struct ProjectionInputs {
    torch::Tensor gSizes;
    torch::Tensor means;
    torch::Tensor quats;
    torch::Tensor scales;
    torch::Tensor cSizes;
    torch::Tensor worldToCamMatrices;
    torch::Tensor projectionMatrices;
};

ProjectionInputs
makeInputs() {
    const auto floatOptions = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    const auto int64Options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);

    auto worldToCamMatrices = torch::eye(4, floatOptions).unsqueeze(0).repeat({3, 1, 1});
    worldToCamMatrices.index_put_({1, 0, 3}, 0.1f);
    worldToCamMatrices.index_put_({2, 1, 3}, -0.1f);

    auto projectionMatrices = torch::zeros({3, 3, 3}, floatOptions);
    projectionMatrices.index_put_({torch::indexing::Slice(), 0, 0}, 30.0f);
    projectionMatrices.index_put_({torch::indexing::Slice(), 1, 1}, 30.0f);
    projectionMatrices.index_put_({torch::indexing::Slice(), 0, 2}, 31.5f);
    projectionMatrices.index_put_({torch::indexing::Slice(), 1, 2}, 23.5f);
    projectionMatrices.index_put_({torch::indexing::Slice(), 2, 2}, 1.0f);

    return {
        torch::tensor({2, 1}, int64Options),
        torch::tensor({{-0.2f, 0.1f, 2.5f}, {0.3f, -0.1f, 3.0f}, {0.0f, 0.2f, 2.0f}}, floatOptions),
        torch::tensor(
            {{1.0f, 0.0f, 0.0f, 0.0f}, {0.98f, 0.1f, 0.0f, 0.0f}, {0.99f, 0.0f, 0.1f, 0.0f}},
            floatOptions),
        torch::tensor({{0.22f, 0.18f, 0.20f}, {0.25f, 0.20f, 0.17f}, {0.18f, 0.22f, 0.20f}},
                      floatOptions),
        torch::tensor({2, 1}, int64Options),
        worldToCamMatrices,
        projectionMatrices,
    };
}

using ForwardResult =
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>;

ForwardResult
projectDenseScene(const ProjectionInputs &inputs,
                  const int64_t gaussianBegin,
                  const int64_t gaussianEnd,
                  const int64_t cameraBegin,
                  const int64_t cameraEnd) {
    using torch::indexing::Slice;
    return fvdb::detail::ops::projectGaussiansAnalyticFwd(
        inputs.means.index({Slice(gaussianBegin, gaussianEnd)}).contiguous(),
        inputs.quats.index({Slice(gaussianBegin, gaussianEnd)}).contiguous(),
        torch::log(inputs.scales.index({Slice(gaussianBegin, gaussianEnd)})).contiguous(),
        inputs.worldToCamMatrices.index({Slice(cameraBegin, cameraEnd)}).contiguous(),
        inputs.projectionMatrices.index({Slice(cameraBegin, cameraEnd)}).contiguous(),
        kImageWidth,
        kImageHeight,
        kEps2d,
        kNearPlane,
        kFarPlane,
        0.0f,
        false,
        false);
}

torch::Tensor
flattenAndConcatenate(const torch::Tensor &first, const torch::Tensor &second) {
    std::vector<int64_t> flattenedShape{-1};
    for (int64_t dim = 2; dim < first.dim(); ++dim) {
        flattenedShape.push_back(first.size(dim));
    }
    return torch::cat({first.reshape(flattenedShape), second.reshape(flattenedShape)}, 0);
}

ForwardResult
projectJaggedSingleCamera(const ProjectionInputs &inputs,
                          const int64_t gaussianBegin,
                          const int64_t gaussianEnd,
                          const int64_t cameraIndex) {
    using torch::indexing::Slice;
    const auto sizes =
        torch::tensor({gaussianEnd - gaussianBegin},
                      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    const auto cameraSizes = torch::ones_like(sizes);
    return fvdb::detail::ops::projectGaussiansAnalyticJaggedFwd(
        sizes,
        inputs.means.index({Slice(gaussianBegin, gaussianEnd)}).contiguous(),
        inputs.quats.index({Slice(gaussianBegin, gaussianEnd)}).contiguous(),
        inputs.scales.index({Slice(gaussianBegin, gaussianEnd)}).contiguous(),
        cameraSizes,
        inputs.worldToCamMatrices.index({Slice(cameraIndex, cameraIndex + 1)}).contiguous(),
        inputs.projectionMatrices.index({Slice(cameraIndex, cameraIndex + 1)}).contiguous(),
        kImageWidth,
        kImageHeight,
        kEps2d,
        kNearPlane,
        kFarPlane,
        0.0f,
        false);
}

ForwardResult
backwardJaggedSingleCamera(const ProjectionInputs &inputs,
                           const int64_t gaussianBegin,
                           const int64_t gaussianEnd,
                           const int64_t cameraIndex) {
    using torch::indexing::Slice;
    const auto sizes =
        torch::tensor({gaussianEnd - gaussianBegin},
                      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    const auto cameraSizes = torch::ones_like(sizes);
    const auto forward = projectJaggedSingleCamera(inputs, gaussianBegin, gaussianEnd, cameraIndex);
    return fvdb::detail::ops::projectGaussiansAnalyticJaggedBwd(
        sizes,
        inputs.means.index({Slice(gaussianBegin, gaussianEnd)}).contiguous(),
        inputs.quats.index({Slice(gaussianBegin, gaussianEnd)}).contiguous(),
        inputs.scales.index({Slice(gaussianBegin, gaussianEnd)}).contiguous(),
        cameraSizes,
        inputs.worldToCamMatrices.index({Slice(cameraIndex, cameraIndex + 1)}).contiguous(),
        inputs.projectionMatrices.index({Slice(cameraIndex, cameraIndex + 1)}).contiguous(),
        kImageWidth,
        kImageHeight,
        kEps2d,
        std::get<0>(forward),
        std::get<3>(forward),
        torch::ones_like(std::get<1>(forward)),
        torch::ones_like(std::get<2>(forward)),
        torch::ones_like(std::get<3>(forward)),
        true,
        false);
}

} // namespace

TEST(GaussianProjectionJaggedTest, ForwardMatchesDensePerSceneProjection) {
    const ProjectionInputs inputs = makeInputs();
    const auto jagged =
        fvdb::detail::ops::projectGaussiansAnalyticJaggedFwd(inputs.gSizes,
                                                             inputs.means,
                                                             inputs.quats,
                                                             inputs.scales,
                                                             inputs.cSizes,
                                                             inputs.worldToCamMatrices,
                                                             inputs.projectionMatrices,
                                                             kImageWidth,
                                                             kImageHeight,
                                                             kEps2d,
                                                             kNearPlane,
                                                             kFarPlane,
                                                             0.0f,
                                                             false);
    const auto firstScene  = projectDenseScene(inputs, 0, 2, 0, 2);
    const auto secondScene = projectDenseScene(inputs, 2, 3, 2, 3);

    const auto expectedRadii =
        flattenAndConcatenate(std::get<0>(firstScene), std::get<0>(secondScene));
    EXPECT_TRUE(torch::equal(std::get<0>(jagged), expectedRadii));

    const auto visible = std::get<0>(jagged).select(1, 0) > 0;
    EXPECT_TRUE(torch::all(visible).item<bool>());
    EXPECT_TRUE(
        torch::allclose(std::get<1>(jagged),
                        flattenAndConcatenate(std::get<1>(firstScene), std::get<1>(secondScene)),
                        1e-5,
                        1e-5));
    EXPECT_TRUE(
        torch::allclose(std::get<2>(jagged),
                        flattenAndConcatenate(std::get<2>(firstScene), std::get<2>(secondScene)),
                        1e-5,
                        1e-5));
    EXPECT_TRUE(
        torch::allclose(std::get<3>(jagged),
                        flattenAndConcatenate(std::get<3>(firstScene), std::get<3>(secondScene)),
                        1e-5,
                        1e-5));
}

TEST(GaussianProjectionJaggedTest, BackwardAddsGradientsAcrossCameras) {
    const ProjectionInputs inputs = makeInputs();
    const auto jaggedForward =
        fvdb::detail::ops::projectGaussiansAnalyticJaggedFwd(inputs.gSizes,
                                                             inputs.means,
                                                             inputs.quats,
                                                             inputs.scales,
                                                             inputs.cSizes,
                                                             inputs.worldToCamMatrices,
                                                             inputs.projectionMatrices,
                                                             kImageWidth,
                                                             kImageHeight,
                                                             kEps2d,
                                                             kNearPlane,
                                                             kFarPlane,
                                                             0.0f,
                                                             false);
    const auto jaggedBackward = fvdb::detail::ops::projectGaussiansAnalyticJaggedBwd(
        inputs.gSizes,
        inputs.means,
        inputs.quats,
        inputs.scales,
        inputs.cSizes,
        inputs.worldToCamMatrices,
        inputs.projectionMatrices,
        kImageWidth,
        kImageHeight,
        kEps2d,
        std::get<0>(jaggedForward),
        std::get<3>(jaggedForward),
        torch::ones_like(std::get<1>(jaggedForward)),
        torch::ones_like(std::get<2>(jaggedForward)),
        torch::ones_like(std::get<3>(jaggedForward)),
        true,
        false);
    const auto firstCamera   = backwardJaggedSingleCamera(inputs, 0, 2, 0);
    const auto secondCamera  = backwardJaggedSingleCamera(inputs, 0, 2, 1);
    const auto thirdCamera   = backwardJaggedSingleCamera(inputs, 2, 3, 2);
    const auto expectedMeans = torch::cat(
        {std::get<0>(firstCamera) + std::get<0>(secondCamera), std::get<0>(thirdCamera)});
    const auto expectedQuats = torch::cat(
        {std::get<2>(firstCamera) + std::get<2>(secondCamera), std::get<2>(thirdCamera)});
    const auto expectedScales = torch::cat(
        {std::get<3>(firstCamera) + std::get<3>(secondCamera), std::get<3>(thirdCamera)});
    const auto expectedWorldToCam =
        torch::cat({std::get<4>(firstCamera), std::get<4>(secondCamera), std::get<4>(thirdCamera)});

    EXPECT_TRUE(torch::allclose(std::get<0>(jaggedBackward), expectedMeans, 1e-4, 1e-4));
    EXPECT_TRUE(torch::allclose(std::get<2>(jaggedBackward), expectedQuats, 1e-4, 1e-4));
    EXPECT_TRUE(torch::allclose(std::get<3>(jaggedBackward), expectedScales, 1e-4, 1e-4));
    EXPECT_TRUE(torch::allclose(std::get<4>(jaggedBackward), expectedWorldToCam, 1e-4, 1e-4));
}
