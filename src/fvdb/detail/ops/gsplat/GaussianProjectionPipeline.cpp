// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/gsplat/GaussianCameraValidation.h>
#include <fvdb/detail/ops/gsplat/GaussianDeduplicatePixels.h>
#include <fvdb/detail/ops/gsplat/GaussianProjectionForward.h>
#include <fvdb/detail/ops/gsplat/GaussianProjectionPipeline.h>
#include <fvdb/detail/ops/gsplat/GaussianProjectionUT.h>
#include <fvdb/detail/ops/gsplat/GaussianRasterizeForward.h>
#include <fvdb/detail/ops/gsplat/GaussianSphericalHarmonicsForward.h>
#include <fvdb/detail/ops/gsplat/GaussianSplatSparse.h>
#include <fvdb/detail/ops/gsplat/GaussianTileIntersection.h>
#include <fvdb/detail/utils/Nvtx.h>
#include <fvdb/detail/utils/Utils.h>

#include <torch/csrc/autograd/generated/variable_factories.h>

namespace fvdb::detail::ops {

namespace {

/// @brief Compute render quantities (SH colors, depths, or both) for projected Gaussians.
torch::Tensor
computeRenderQuantity(const torch::Tensor &means,
                      const torch::Tensor &sh0,
                      const torch::Tensor &shN,
                      const RenderSettings &settings,
                      const torch::Tensor &worldToCameraMatrices,
                      const torch::Tensor &perGaussianRadius,
                      const torch::Tensor &perGaussianDepth) {
    using RenderMode = RenderSettings::RenderMode;
    if (settings.renderMode == RenderMode::DEPTH) {
        return perGaussianDepth.unsqueeze(-1); // [C, N, 1]
    } else if (settings.renderMode == RenderMode::FEATURES ||
               settings.renderMode == RenderMode::FEATURES_AND_DEPTH) {
        auto renderQuantity = evalSphericalHarmonics(
            means, sh0, shN, settings.shDegreeToUse, worldToCameraMatrices, perGaussianRadius);
        if (settings.renderMode == RenderMode::FEATURES_AND_DEPTH) {
            renderQuantity =
                torch::cat({renderQuantity, perGaussianDepth.unsqueeze(-1)}, -1); // [C, N, D + 1]
        }
        return renderQuantity;
    } else {
        TORCH_CHECK_VALUE(false, "Invalid render mode");
    }
}

/// @brief Prepare accumulator tensors for projection, lazily initializing them if needed.
void
prepareAccumulators(const int64_t N,
                    const torch::Tensor &means,
                    bool accumulateMean2dGradients,
                    bool accumulateMax2dRadii,
                    std::optional<torch::Tensor> &accumGradNorms,
                    std::optional<torch::Tensor> &accumStepCounts,
                    std::optional<torch::Tensor> &accumMax2dRadii) {
    if (accumulateMean2dGradients) {
        if (!accumGradNorms.has_value() || accumGradNorms->numel() != N) {
            accumGradNorms = torch::zeros({N}, means.options());
        }
        if (!accumStepCounts.has_value() || accumStepCounts->numel() != N) {
            accumStepCounts = torch::zeros(
                {N}, torch::TensorOptions().dtype(torch::kInt32).device(means.device()));
        }
    }
    if (accumulateMax2dRadii) {
        if (!accumMax2dRadii.has_value() || accumMax2dRadii->numel() != N) {
            accumMax2dRadii = torch::zeros(
                {N}, torch::TensorOptions().dtype(torch::kInt32).device(means.device()));
        }
    }
}

} // anonymous namespace

fvdb::ProjectedGaussianSplats
projectGaussiansAnalytic(const torch::Tensor &means,
                         const torch::Tensor &quats,
                         const torch::Tensor &logScales,
                         const torch::Tensor &logitOpacities,
                         const torch::Tensor &sh0,
                         const torch::Tensor &shN,
                         const torch::Tensor &worldToCameraMatrices,
                         const torch::Tensor &projectionMatrices,
                         const RenderSettings &settings,
                         const DistortionModel cameraModel,
                         const bool accumulateMean2dGradients,
                         const bool accumulateMax2dRadii,
                         std::optional<torch::Tensor> &accumGradNorms,
                         std::optional<torch::Tensor> &accumStepCounts,
                         std::optional<torch::Tensor> &accumMax2dRadii) {
    FVDB_FUNC_RANGE();
    const bool ortho = cameraModel == DistortionModel::ORTHOGRAPHIC;
    const int C      = worldToCameraMatrices.size(0);
    const int N      = means.size(0);

    fvdb::ProjectedGaussianSplats ret;
    ret.mRenderSettings   = settings;
    ret.mCameraModel      = cameraModel;
    ret.mProjectionMethod = ProjectionMethod::ANALYTIC;

    // Prepare accumulators
    prepareAccumulators(N,
                        means,
                        accumulateMean2dGradients,
                        accumulateMax2dRadii,
                        accumGradNorms,
                        accumStepCounts,
                        accumMax2dRadii);

    std::optional<torch::Tensor> maybeGradNorms =
        accumulateMean2dGradients ? accumGradNorms : std::nullopt;
    std::optional<torch::Tensor> maybeStepCounts =
        accumulateMean2dGradients ? accumStepCounts : std::nullopt;
    std::optional<torch::Tensor> maybeMax2dRadii =
        accumulateMax2dRadii ? accumMax2dRadii : std::nullopt;

    // Project to image plane
    auto variables        = gaussianProjectionForward(means,
                                               quats,
                                               logScales,
                                               worldToCameraMatrices,
                                               projectionMatrices,
                                               settings.imageWidth,
                                               settings.imageHeight,
                                               settings.eps2d,
                                               settings.nearPlane,
                                               settings.farPlane,
                                               settings.radiusClip,
                                               settings.antialias,
                                               ortho);
    ret.perGaussianRadius = std::get<0>(variables);
    ret.perGaussian2dMean = std::get<1>(variables);
    ret.perGaussianDepth  = std::get<2>(variables);
    ret.perGaussianConic  = std::get<3>(variables);

    ret.perGaussianOpacity = torch::sigmoid(logitOpacities).repeat({C, 1});
    if (settings.antialias) {
        ret.perGaussianOpacity *= std::get<4>(variables);
        ret.perGaussianOpacity = ret.perGaussianOpacity.contiguous();
    }

    ret.perGaussianRenderQuantity = computeRenderQuantity(means,
                                                          sh0,
                                                          shN,
                                                          settings,
                                                          worldToCameraMatrices,
                                                          ret.perGaussianRadius,
                                                          ret.perGaussianDepth);

    // Intersect projected Gaussians with image tiles [non-differentiable]
    const int numTilesW = std::ceil(settings.imageWidth / static_cast<float>(settings.tileSize));
    const int numTilesH = std::ceil(settings.imageHeight / static_cast<float>(settings.tileSize));
    const auto [tileOffsets, tileGaussianIds] = gaussianTileIntersection(ret.perGaussian2dMean,
                                                                         ret.perGaussianRadius,
                                                                         ret.perGaussianDepth,
                                                                         at::nullopt,
                                                                         C,
                                                                         settings.tileSize,
                                                                         numTilesH,
                                                                         numTilesW);
    ret.tileOffsets                           = tileOffsets;
    ret.tileGaussianIds                       = tileGaussianIds;

    return ret;
}

fvdb::ProjectedGaussianSplats
projectGaussiansUT(const torch::Tensor &means,
                   const torch::Tensor &quats,
                   const torch::Tensor &logScales,
                   const torch::Tensor &logitOpacities,
                   const torch::Tensor &sh0,
                   const torch::Tensor &shN,
                   const torch::Tensor &worldToCameraMatrices,
                   const torch::Tensor &projectionMatrices,
                   const torch::Tensor &distortionCoeffs,
                   const RenderSettings &settings,
                   const DistortionModel cameraModel) {
    FVDB_FUNC_RANGE();
    const int C = worldToCameraMatrices.size(0);

    fvdb::ProjectedGaussianSplats ret;
    ret.mRenderSettings   = settings;
    ret.mCameraModel      = cameraModel;
    ret.mProjectionMethod = resolveProjectionMethod(cameraModel, ProjectionMethod::UNSCENTED);

    UTParams utParams = UTParams{};
    const auto projectionResults =
        gaussianProjectionForwardUT(means,
                                    quats,
                                    logScales,
                                    worldToCameraMatrices,
                                    worldToCameraMatrices,
                                    projectionMatrices,
                                    RollingShutterType::NONE,
                                    utParams,
                                    cameraModel,
                                    distortionCoeffs,
                                    static_cast<int64_t>(settings.imageWidth),
                                    static_cast<int64_t>(settings.imageHeight),
                                    settings.eps2d,
                                    settings.nearPlane,
                                    settings.farPlane,
                                    settings.radiusClip,
                                    settings.antialias);

    ret.perGaussianRadius = std::get<0>(projectionResults);
    ret.perGaussian2dMean = std::get<1>(projectionResults);
    ret.perGaussianDepth  = std::get<2>(projectionResults);
    ret.perGaussianConic  = std::get<3>(projectionResults);

    ret.perGaussianOpacity = torch::sigmoid(logitOpacities).repeat({C, 1});
    if (settings.antialias) {
        const torch::Tensor compensations = std::get<4>(projectionResults);
        TORCH_CHECK(compensations.defined(),
                    "UT projection returned an undefined compensation tensor in antialias mode");
        ret.perGaussianOpacity *= compensations;
        ret.perGaussianOpacity = ret.perGaussianOpacity.contiguous();
    }

    ret.perGaussianRenderQuantity = computeRenderQuantity(means,
                                                          sh0,
                                                          shN,
                                                          settings,
                                                          worldToCameraMatrices,
                                                          ret.perGaussianRadius,
                                                          ret.perGaussianDepth);

    const int numTilesW = std::ceil(settings.imageWidth / static_cast<float>(settings.tileSize));
    const int numTilesH = std::ceil(settings.imageHeight / static_cast<float>(settings.tileSize));
    std::tie(ret.tileOffsets, ret.tileGaussianIds) = gaussianTileIntersection(ret.perGaussian2dMean,
                                                                              ret.perGaussianRadius,
                                                                              ret.perGaussianDepth,
                                                                              at::nullopt,
                                                                              C,
                                                                              settings.tileSize,
                                                                              numTilesH,
                                                                              numTilesW);

    return ret;
}

fvdb::ProjectedGaussianSplats
projectGaussiansForCamera(const torch::Tensor &means,
                          const torch::Tensor &quats,
                          const torch::Tensor &logScales,
                          const torch::Tensor &logitOpacities,
                          const torch::Tensor &sh0,
                          const torch::Tensor &shN,
                          const torch::Tensor &worldToCameraMatrices,
                          const torch::Tensor &projectionMatrices,
                          const RenderSettings &settings,
                          const DistortionModel cameraModel,
                          const ProjectionMethod projectionMethod,
                          const std::optional<torch::Tensor> &distortionCoeffs,
                          const bool accumulateMean2dGradients,
                          const bool accumulateMax2dRadii,
                          std::optional<torch::Tensor> &accumGradNorms,
                          std::optional<torch::Tensor> &accumStepCounts,
                          std::optional<torch::Tensor> &accumMax2dRadii) {
    FVDB_FUNC_RANGE();
    validateCameraProjectionArgs(
        worldToCameraMatrices, projectionMatrices, cameraModel, projectionMethod, distortionCoeffs);

    const ProjectionMethod resolvedMethod = resolveProjectionMethod(cameraModel, projectionMethod);

    if (resolvedMethod == ProjectionMethod::ANALYTIC) {
        return projectGaussiansAnalytic(means,
                                        quats,
                                        logScales,
                                        logitOpacities,
                                        sh0,
                                        shN,
                                        worldToCameraMatrices,
                                        projectionMatrices,
                                        settings,
                                        cameraModel,
                                        accumulateMean2dGradients,
                                        accumulateMax2dRadii,
                                        accumGradNorms,
                                        accumStepCounts,
                                        accumMax2dRadii);
    }

    const int C                                = worldToCameraMatrices.size(0);
    const torch::Tensor distortionCoeffsTensor = distortionCoeffs.has_value()
                                                     ? distortionCoeffs.value()
                                                     : torch::empty({C, 0}, means.options());

    return projectGaussiansUT(means,
                              quats,
                              logScales,
                              logitOpacities,
                              sh0,
                              shN,
                              worldToCameraMatrices,
                              projectionMatrices,
                              distortionCoeffsTensor,
                              settings,
                              cameraModel);
}

// ---------------------------------------------------------------------------
// Sparse projection: analytic path
// ---------------------------------------------------------------------------

fvdb::SparseProjectedGaussianSplats
sparseProjectGaussiansAnalytic(const fvdb::JaggedTensor &pixelsToRender,
                               const torch::Tensor &means,
                               const torch::Tensor &quats,
                               const torch::Tensor &logScales,
                               const torch::Tensor &logitOpacities,
                               const torch::Tensor &sh0,
                               const torch::Tensor &shN,
                               const torch::Tensor &worldToCameraMatrices,
                               const torch::Tensor &projectionMatrices,
                               const RenderSettings &settings,
                               const DistortionModel cameraModel,
                               const bool accumulateMean2dGradients,
                               const bool accumulateMax2dRadii,
                               std::optional<torch::Tensor> &accumGradNorms,
                               std::optional<torch::Tensor> &accumStepCounts,
                               std::optional<torch::Tensor> &accumMax2dRadii) {
    FVDB_FUNC_RANGE();
    const bool ortho = cameraModel == DistortionModel::ORTHOGRAPHIC;
    const int C      = worldToCameraMatrices.size(0);
    const int N      = means.size(0);
    TORCH_CHECK(static_cast<int64_t>(pixelsToRender.num_outer_lists()) == C,
                "pixelsToRender must have the same number of outer lists as the number of cameras. "
                "Got ",
                pixelsToRender.num_outer_lists(),
                " outer lists but ",
                C,
                " cameras. ");

    fvdb::SparseProjectedGaussianSplats ret;
    ret.mRenderSettings   = settings;
    ret.mCameraModel      = cameraModel;
    ret.mProjectionMethod = ProjectionMethod::ANALYTIC;

    // Deduplicate pixel coordinates
    auto [uniquePixels, inverseIndices, hasDuplicates] =
        deduplicatePixels(pixelsToRender, settings.imageWidth, settings.imageHeight);
    ret.inverseIndices       = inverseIndices;
    ret.uniquePixelsToRender = uniquePixels;
    ret.hasDuplicates        = hasDuplicates;

    // Compute sparse tile info using deduplicated pixels
    const int numTilesW = std::ceil(settings.imageWidth / static_cast<float>(settings.tileSize));
    const int numTilesH = std::ceil(settings.imageHeight / static_cast<float>(settings.tileSize));
    const auto [activeTiles, activeTileMask, tilePixelMask, tilePixelCumsum, pixelMap] =
        computeSparseInfo(settings.tileSize, numTilesW, numTilesH, uniquePixels);
    ret.activeTiles     = activeTiles;
    ret.activeTileMask  = activeTileMask;
    ret.tilePixelMask   = tilePixelMask;
    ret.tilePixelCumsum = tilePixelCumsum;
    ret.pixelMap        = pixelMap;

    // Prepare accumulators
    prepareAccumulators(N,
                        means,
                        accumulateMean2dGradients,
                        accumulateMax2dRadii,
                        accumGradNorms,
                        accumStepCounts,
                        accumMax2dRadii);

    std::optional<torch::Tensor> maybeGradNorms =
        accumulateMean2dGradients ? accumGradNorms : std::nullopt;
    std::optional<torch::Tensor> maybeStepCounts =
        accumulateMean2dGradients ? accumStepCounts : std::nullopt;
    std::optional<torch::Tensor> maybeMax2dRadii =
        accumulateMax2dRadii ? accumMax2dRadii : std::nullopt;

    // Project to image plane
    auto variables        = gaussianProjectionForward(means,
                                               quats,
                                               logScales,
                                               worldToCameraMatrices,
                                               projectionMatrices,
                                               settings.imageWidth,
                                               settings.imageHeight,
                                               settings.eps2d,
                                               settings.nearPlane,
                                               settings.farPlane,
                                               settings.radiusClip,
                                               settings.antialias,
                                               ortho);
    ret.perGaussianRadius = std::get<0>(variables);
    ret.perGaussian2dMean = std::get<1>(variables);
    ret.perGaussianDepth  = std::get<2>(variables);
    ret.perGaussianConic  = std::get<3>(variables);

    ret.perGaussianOpacity = torch::sigmoid(logitOpacities).repeat({C, 1});
    if (settings.antialias) {
        ret.perGaussianOpacity *= std::get<4>(variables);
        ret.perGaussianOpacity = ret.perGaussianOpacity.contiguous();
    }

    ret.perGaussianRenderQuantity = computeRenderQuantity(means,
                                                          sh0,
                                                          shN,
                                                          settings,
                                                          worldToCameraMatrices,
                                                          ret.perGaussianRadius,
                                                          ret.perGaussianDepth);

    // Sparse tile intersection
    const auto [sparseTileOffsets, tileGaussianIds] =
        gaussianSparseTileIntersection(ret.perGaussian2dMean,
                                       ret.perGaussianRadius,
                                       ret.perGaussianDepth,
                                       ret.activeTileMask,
                                       ret.activeTiles,
                                       at::nullopt,
                                       C,
                                       settings.tileSize,
                                       numTilesH,
                                       numTilesW);
    ret.tileOffsets     = sparseTileOffsets;
    ret.tileGaussianIds = tileGaussianIds;

    return ret;
}

// ---------------------------------------------------------------------------
// Sparse projection: unscented transform path
// ---------------------------------------------------------------------------

fvdb::SparseProjectedGaussianSplats
sparseProjectGaussiansUT(const fvdb::JaggedTensor &pixelsToRender,
                         const torch::Tensor &means,
                         const torch::Tensor &quats,
                         const torch::Tensor &logScales,
                         const torch::Tensor &logitOpacities,
                         const torch::Tensor &sh0,
                         const torch::Tensor &shN,
                         const torch::Tensor &worldToCameraMatrices,
                         const torch::Tensor &projectionMatrices,
                         const torch::Tensor &distortionCoeffs,
                         const RenderSettings &settings,
                         const DistortionModel cameraModel) {
    FVDB_FUNC_RANGE();
    const int C = worldToCameraMatrices.size(0);
    TORCH_CHECK(static_cast<int64_t>(pixelsToRender.num_outer_lists()) == C,
                "pixelsToRender must have the same number of outer lists as the number of cameras. "
                "Got ",
                pixelsToRender.num_outer_lists(),
                " outer lists but ",
                C,
                " cameras. ");

    fvdb::SparseProjectedGaussianSplats ret;
    ret.mRenderSettings   = settings;
    ret.mCameraModel      = cameraModel;
    ret.mProjectionMethod = resolveProjectionMethod(cameraModel, ProjectionMethod::UNSCENTED);

    // Deduplicate pixel coordinates
    auto [uniquePixels, inverseIndices, hasDuplicates] =
        deduplicatePixels(pixelsToRender, settings.imageWidth, settings.imageHeight);
    ret.inverseIndices       = inverseIndices;
    ret.uniquePixelsToRender = uniquePixels;
    ret.hasDuplicates        = hasDuplicates;

    // Compute sparse tile info
    const int numTilesW = std::ceil(settings.imageWidth / static_cast<float>(settings.tileSize));
    const int numTilesH = std::ceil(settings.imageHeight / static_cast<float>(settings.tileSize));
    const auto [activeTiles, activeTileMask, tilePixelMask, tilePixelCumsum, pixelMap] =
        computeSparseInfo(settings.tileSize, numTilesW, numTilesH, uniquePixels);
    ret.activeTiles     = activeTiles;
    ret.activeTileMask  = activeTileMask;
    ret.tilePixelMask   = tilePixelMask;
    ret.tilePixelCumsum = tilePixelCumsum;
    ret.pixelMap        = pixelMap;

    // UT projection
    UTParams utParams = UTParams{};
    const auto projectionResults =
        gaussianProjectionForwardUT(means,
                                    quats,
                                    logScales,
                                    worldToCameraMatrices,
                                    worldToCameraMatrices,
                                    projectionMatrices,
                                    RollingShutterType::NONE,
                                    utParams,
                                    cameraModel,
                                    distortionCoeffs,
                                    static_cast<int64_t>(settings.imageWidth),
                                    static_cast<int64_t>(settings.imageHeight),
                                    settings.eps2d,
                                    settings.nearPlane,
                                    settings.farPlane,
                                    settings.radiusClip,
                                    settings.antialias);

    ret.perGaussianRadius = std::get<0>(projectionResults);
    ret.perGaussian2dMean = std::get<1>(projectionResults);
    ret.perGaussianDepth  = std::get<2>(projectionResults);
    ret.perGaussianConic  = std::get<3>(projectionResults);

    ret.perGaussianOpacity = torch::sigmoid(logitOpacities).repeat({C, 1});
    if (settings.antialias) {
        const torch::Tensor compensations = std::get<4>(projectionResults);
        TORCH_CHECK(compensations.defined(),
                    "UT projection returned an undefined compensation tensor in antialias mode");
        ret.perGaussianOpacity *= compensations;
        ret.perGaussianOpacity = ret.perGaussianOpacity.contiguous();
    }

    ret.perGaussianRenderQuantity = computeRenderQuantity(means,
                                                          sh0,
                                                          shN,
                                                          settings,
                                                          worldToCameraMatrices,
                                                          ret.perGaussianRadius,
                                                          ret.perGaussianDepth);

    // Sparse tile intersection
    const auto [sparseTileOffsets, tileGaussianIds] =
        gaussianSparseTileIntersection(ret.perGaussian2dMean,
                                       ret.perGaussianRadius,
                                       ret.perGaussianDepth,
                                       ret.activeTileMask,
                                       ret.activeTiles,
                                       at::nullopt,
                                       C,
                                       settings.tileSize,
                                       numTilesH,
                                       numTilesW);
    ret.tileOffsets     = sparseTileOffsets;
    ret.tileGaussianIds = tileGaussianIds;

    return ret;
}

// ---------------------------------------------------------------------------
// Sparse projection: camera dispatch (analytic vs UT)
// ---------------------------------------------------------------------------

fvdb::SparseProjectedGaussianSplats
sparseProjectGaussiansForCamera(const fvdb::JaggedTensor &pixelsToRender,
                                const torch::Tensor &means,
                                const torch::Tensor &quats,
                                const torch::Tensor &logScales,
                                const torch::Tensor &logitOpacities,
                                const torch::Tensor &sh0,
                                const torch::Tensor &shN,
                                const torch::Tensor &worldToCameraMatrices,
                                const torch::Tensor &projectionMatrices,
                                const RenderSettings &settings,
                                const DistortionModel cameraModel,
                                const ProjectionMethod projectionMethod,
                                const std::optional<torch::Tensor> &distortionCoeffs,
                                const bool accumulateMean2dGradients,
                                const bool accumulateMax2dRadii,
                                std::optional<torch::Tensor> &accumGradNorms,
                                std::optional<torch::Tensor> &accumStepCounts,
                                std::optional<torch::Tensor> &accumMax2dRadii) {
    FVDB_FUNC_RANGE();
    validateCameraProjectionArgs(
        worldToCameraMatrices, projectionMatrices, cameraModel, projectionMethod, distortionCoeffs);

    const ProjectionMethod resolvedMethod = resolveProjectionMethod(cameraModel, projectionMethod);

    if (resolvedMethod == ProjectionMethod::ANALYTIC) {
        return sparseProjectGaussiansAnalytic(pixelsToRender,
                                              means,
                                              quats,
                                              logScales,
                                              logitOpacities,
                                              sh0,
                                              shN,
                                              worldToCameraMatrices,
                                              projectionMatrices,
                                              settings,
                                              cameraModel,
                                              accumulateMean2dGradients,
                                              accumulateMax2dRadii,
                                              accumGradNorms,
                                              accumStepCounts,
                                              accumMax2dRadii);
    }

    const int C                                = worldToCameraMatrices.size(0);
    const torch::Tensor distortionCoeffsTensor = distortionCoeffs.has_value()
                                                     ? distortionCoeffs.value()
                                                     : torch::empty({C, 0}, means.options());

    return sparseProjectGaussiansUT(pixelsToRender,
                                    means,
                                    quats,
                                    logScales,
                                    logitOpacities,
                                    sh0,
                                    shN,
                                    worldToCameraMatrices,
                                    projectionMatrices,
                                    distortionCoeffsTensor,
                                    settings,
                                    cameraModel);
}

// ---------------------------------------------------------------------------
// Sparse render: project + rasterize + duplicate scatter-back
// ---------------------------------------------------------------------------

std::tuple<fvdb::JaggedTensor, fvdb::JaggedTensor>
sparseRender(const fvdb::JaggedTensor &pixelsToRender,
             const torch::Tensor &means,
             const torch::Tensor &quats,
             const torch::Tensor &logScales,
             const torch::Tensor &logitOpacities,
             const torch::Tensor &sh0,
             const torch::Tensor &shN,
             const torch::Tensor &worldToCameraMatrices,
             const torch::Tensor &projectionMatrices,
             const RenderSettings &settings,
             const DistortionModel cameraModel,
             const ProjectionMethod projectionMethod,
             const std::optional<torch::Tensor> &distortionCoeffs,
             const std::optional<torch::Tensor> &backgrounds,
             const std::optional<torch::Tensor> &masks,
             const bool accumulateMean2dGradients,
             const bool accumulateMax2dRadii,
             std::optional<torch::Tensor> &accumGradNorms,
             std::optional<torch::Tensor> &accumStepCounts,
             std::optional<torch::Tensor> &accumMax2dRadii) {
    FVDB_FUNC_RANGE();

    const auto state = sparseProjectGaussiansForCamera(pixelsToRender,
                                                       means,
                                                       quats,
                                                       logScales,
                                                       logitOpacities,
                                                       sh0,
                                                       shN,
                                                       worldToCameraMatrices,
                                                       projectionMatrices,
                                                       settings,
                                                       cameraModel,
                                                       projectionMethod,
                                                       distortionCoeffs,
                                                       accumulateMean2dGradients,
                                                       accumulateMax2dRadii,
                                                       accumGradNorms,
                                                       accumStepCounts,
                                                       accumMax2dRadii);

    // Render using unique (deduplicated) pixels
    const auto &renderPixels = state.hasDuplicates ? state.uniquePixelsToRender : pixelsToRender;

    auto rasterizeResult =
        gaussianSparseRasterizeForward(renderPixels,
                                       state.perGaussian2dMean,
                                       state.perGaussianConic,
                                       state.perGaussianRenderQuantity,
                                       state.perGaussianOpacity,
                                       static_cast<uint32_t>(settings.imageWidth),
                                       static_cast<uint32_t>(settings.imageHeight),
                                       0,
                                       0,
                                       settings.tileSize,
                                       state.tileOffsets,
                                       state.tileGaussianIds,
                                       state.activeTiles,
                                       state.tilePixelMask,
                                       state.tilePixelCumsum,
                                       state.pixelMap,
                                       backgrounds,
                                       masks);
    auto renderedColorsJT = std::get<0>(rasterizeResult);
    auto renderedAlphasJT = std::get<1>(rasterizeResult);

    // Scatter unique results back to all original positions (including duplicates).
    if (state.hasDuplicates) {
        auto renderedPixelsJData = renderedColorsJT.jdata().index_select(0, state.inverseIndices);
        auto renderedAlphasJData = renderedAlphasJT.jdata().index_select(0, state.inverseIndices);
        return {pixelsToRender.jagged_like(renderedPixelsJData),
                pixelsToRender.jagged_like(renderedAlphasJData)};
    }

    return {pixelsToRender.jagged_like(renderedColorsJT.jdata()),
            pixelsToRender.jagged_like(renderedAlphasJT.jdata())};
}

} // namespace fvdb::detail::ops
