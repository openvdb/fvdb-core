// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/gsplat/GaussianSplatOps.h>

#include <fvdb/detail/autograd/EvaluateSphericalHarmonics.h>
#include <fvdb/detail/autograd/GaussianProjection.h>
#include <fvdb/detail/ops/gsplat/GaussianProjectionUT.h>
#include <fvdb/detail/ops/gsplat/GaussianTileIntersection.h>
#include <fvdb/detail/utils/Nvtx.h>
#include <fvdb/detail/utils/Utils.h>

#include <torch/csrc/autograd/generated/variable_factories.h>

namespace fvdb::detail::ops {

void
validateCameraProjectionArgs(const torch::Tensor &worldToCameraMatrices,
                             const torch::Tensor &projectionMatrices,
                             const DistortionModel cameraModel,
                             const ProjectionMethod requestedProjectionMethod,
                             const std::optional<torch::Tensor> &distortionCoeffs) {
    const int64_t C = worldToCameraMatrices.size(0);
    TORCH_CHECK(C > 0, "At least one camera must be provided (got 0)");
    TORCH_CHECK(worldToCameraMatrices.sizes() == torch::IntArrayRef({C, 4, 4}),
                "worldToCameraMatrices must have shape (C, 4, 4)");
    TORCH_CHECK(projectionMatrices.sizes() == torch::IntArrayRef({C, 3, 3}),
                "projectionMatrices must have shape (C, 3, 3)");
    TORCH_CHECK(worldToCameraMatrices.is_contiguous(), "worldToCameraMatrices must be contiguous");
    TORCH_CHECK(projectionMatrices.is_contiguous(), "projectionMatrices must be contiguous");

    const ProjectionMethod resolvedProjectionMethod =
        resolveProjectionMethod(cameraModel, requestedProjectionMethod);

    if (distortionCoeffs.has_value()) {
        TORCH_CHECK(distortionCoeffs->sizes() == torch::IntArrayRef({C, 12}),
                    "distortionCoeffs must have shape (C, 12)");
        TORCH_CHECK(distortionCoeffs->is_contiguous(), "distortionCoeffs must be contiguous");
    }

    if (usesOpenCVDistortion(cameraModel)) {
        TORCH_CHECK(distortionCoeffs.has_value(),
                    "distortionCoeffs must be provided for OpenCV camera models");
        TORCH_CHECK(resolvedProjectionMethod == ProjectionMethod::UNSCENTED,
                    "OpenCV camera models require ProjectionMethod::UNSCENTED or AUTO");
    }
}

void
checkGaussianState(const torch::Tensor &means,
                   const torch::Tensor &quats,
                   const torch::Tensor &logScales,
                   const torch::Tensor &logitOpacities,
                   const torch::Tensor &sh0,
                   const torch::Tensor &shN) {
    const int64_t N = means.size(0); // number of gaussians

    TORCH_CHECK_VALUE(means.sizes() == torch::IntArrayRef({N, 3}), "means must have shape (N, 3)");
    TORCH_CHECK_VALUE(quats.sizes() == torch::IntArrayRef({N, 4}), "quats must have shape (N, 4)");
    TORCH_CHECK_VALUE(logScales.sizes() == torch::IntArrayRef({N, 3}),
                      "scales must have shape (N, 3)");
    TORCH_CHECK_VALUE(logitOpacities.sizes() == torch::IntArrayRef({N}),
                      "opacities must have shape (N)");
    TORCH_CHECK_VALUE(sh0.size(0) == N, "sh0 must have shape (N, 1, D)");
    TORCH_CHECK_VALUE(sh0.size(1) == 1, "sh0 must have shape (N, 1, D)");
    TORCH_CHECK_VALUE(sh0.dim() == 3, "sh0 must have shape (N, 1, D)");
    TORCH_CHECK_VALUE(shN.size(0) == N, "shN must have shape (N, K-1, D)");
    TORCH_CHECK_VALUE(shN.dim() == 3, "shN must have shape (N, K-1, D)");

    TORCH_CHECK_VALUE(means.device() == quats.device(), "All tensors must be on the same device");
    TORCH_CHECK_VALUE(means.device() == logScales.device(),
                      "All tensors must be on the same device");
    TORCH_CHECK_VALUE(means.device() == logitOpacities.device(),
                      "All tensors must be on the same device");
    TORCH_CHECK_VALUE(means.device() == sh0.device(), "All tensors must be on the same device");
    TORCH_CHECK_VALUE(means.device() == shN.device(), "All tensors must be on the same device");

    TORCH_CHECK_VALUE(torch::isFloatingType(means.scalar_type()),
                      "All tensors must be of floating point type");
    TORCH_CHECK_VALUE(means.scalar_type() == quats.scalar_type(),
                      "All tensors must be of the same type");
    TORCH_CHECK_VALUE(means.scalar_type() == logScales.scalar_type(),
                      "All tensors must be of the same type");
    TORCH_CHECK_VALUE(means.scalar_type() == logitOpacities.scalar_type(),
                      "All tensors must be of the same type");
    TORCH_CHECK_VALUE(means.scalar_type() == sh0.scalar_type(),
                      "All tensors must be of the same type");
    TORCH_CHECK_VALUE(means.scalar_type() == shN.scalar_type(),
                      "All tensors must be of the same type");
}

std::tuple<fvdb::JaggedTensor, torch::Tensor, bool>
deduplicatePixels(const fvdb::JaggedTensor &pixelsToRender,
                  int64_t imageWidth,
                  int64_t imageHeight) {
    const auto totalPixels = pixelsToRender.rsize(0);
    if (totalPixels == 0) {
        auto emptyInverse = torch::empty({0}, pixelsToRender.jdata().options().dtype(torch::kLong));
        return {pixelsToRender, emptyInverse, false};
    }

    const auto device               = pixelsToRender.device();
    const auto jdata                = pixelsToRender.jdata();
    const auto jidx                 = pixelsToRender.jidx();
    const int64_t numPixelsPerImage = imageHeight * imageWidth;
    const auto longOpts             = torch::TensorOptions().device(device).dtype(torch::kLong);
    const auto boolOpts             = torch::TensorOptions().device(device).dtype(torch::kBool);

    // Encode (batchIdx, row, col) into a single int64 key:
    //   key = batchIdx * (H * W) + row * W + col
    // For single-list JaggedTensors, jidx is empty so we skip the batch term entirely.
    const bool singleList = (jidx.size(0) == 0);
    torch::Tensor rows, cols;
    if (jdata.scalar_type() == torch::kInt32) {
        rows = jdata.select(1, 0).to(torch::kLong);
        cols = jdata.select(1, 1).to(torch::kLong);
    } else {
        rows = jdata.select(1, 0);
        cols = jdata.select(1, 1);
    }
    torch::Tensor keys;
    if (singleList) {
        keys = rows * imageWidth + cols;
    } else {
        auto jidxLong = jidx.to(torch::kLong);
        keys          = jidxLong * numPixelsPerImage + rows * imageWidth + cols;
    }

    // Sort keys and find group boundaries
    auto [sortedKeys, sortPerm] = keys.sort();

    auto isGroupStart = torch::ones({totalPixels}, boolOpts);
    if (totalPixels > 1) {
        isGroupStart.slice(0, 1).copy_(sortedKeys.slice(0, 1) != sortedKeys.slice(0, 0, -1));
    }

    // Extract first-of-group positions before mutating isGroupStart
    auto firstInSorted = isGroupStart.nonzero().squeeze(1);

    // Assign a group ID (0-based) to each sorted position via in-place cumsum
    auto groupIds = isGroupStart.to(torch::kLong);
    groupIds.cumsum_(0).sub_(1);
    const auto numUnique = groupIds[-1].item<int64_t>() + 1;

    if (numUnique == totalPixels) {
        return {pixelsToRender, torch::arange(totalPixels, longOpts), false};
    }

    // inverseIndices: map each original position to its group ID (= index in unique output)
    auto inverseIndices = torch::empty({totalPixels}, longOpts);
    inverseIndices.index_put_({sortPerm}, groupIds);

    // Pick the first occurrence of each group (in sorted order) and map to original indices
    auto uniqueOrigIndices = sortPerm.index_select(0, firstInSorted);
    auto uniqueJData       = jdata.index_select(0, uniqueOrigIndices);

    // Build new JaggedTensor offsets for the unique pixels
    auto uniqueBatchIdx = singleList ? torch::zeros({numUnique}, longOpts)
                                     : jidx.to(torch::kLong).index_select(0, uniqueOrigIndices);
    auto numLists       = pixelsToRender.num_outer_lists();
    auto countsPerList  = torch::bincount(uniqueBatchIdx, {}, numLists);
    auto newOffsets     = torch::zeros({numLists + 1}, longOpts);
    newOffsets.slice(0, 1).copy_(countsPerList.cumsum(0));

    auto newJidx = uniqueBatchIdx.to(fvdb::JIdxScalarType);

    auto uniquePixels = fvdb::JaggedTensor::from_jdata_joffsets_jidx_and_lidx_unsafe(
        uniqueJData, newOffsets, newJidx, pixelsToRender.jlidx(), numLists);

    return {uniquePixels, inverseIndices, true};
}

torch::Tensor
evalSphericalHarmonics(const torch::Tensor &means,
                       const torch::Tensor &sh0,
                       const torch::Tensor &shN,
                       const int64_t shDegreeToUse,
                       const torch::Tensor &worldToCameraMatrices,
                       const torch::Tensor &perGaussianProjectedRadii) {
    FVDB_FUNC_RANGE();
    const auto K              = shN.size(1) + 1;              // number of SH bases
    const auto C              = worldToCameraMatrices.size(0); // number of cameras
    const auto actualShDegree = shDegreeToUse < 0 ? (std::sqrt(K) - 1) : shDegreeToUse;
    if (actualShDegree == 0) {
        return fvdb::detail::autograd::EvaluateSphericalHarmonics::apply(
            actualShDegree, C, torch::nullopt, sh0, torch::nullopt, perGaussianProjectedRadii)[0];
    } else {
        auto [camToWorldMatrices, info] = torch::linalg_inv_ex(worldToCameraMatrices);
        const torch::Tensor viewDirs =
            means.index(
                {torch::indexing::None, torch::indexing::Slice(), torch::indexing::Slice()}) -
            camToWorldMatrices.index({torch::indexing::Slice(),
                                      torch::indexing::None,
                                      torch::indexing::Slice(0, 3),
                                      3}); // [1, N, 3] - [C, 1, 3]
        return fvdb::detail::autograd::EvaluateSphericalHarmonics::apply(
            actualShDegree, C, viewDirs, sh0, shN, perGaussianProjectedRadii)[0];
    }
}

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
    } else if (settings.renderMode == RenderMode::RGB || settings.renderMode == RenderMode::RGBD) {
        auto renderQuantity = evalSphericalHarmonics(
            means, sh0, shN, settings.shDegreeToUse, worldToCameraMatrices, perGaussianRadius);
        if (settings.renderMode == RenderMode::RGBD) {
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

fvdb::GaussianSplat3d::ProjectedGaussianSplats
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

    fvdb::GaussianSplat3d::ProjectedGaussianSplats ret;
    ret.mRenderSettings   = settings;
    ret.mCameraModel      = cameraModel;
    ret.mProjectionMethod = ProjectionMethod::ANALYTIC;

    // Prepare accumulators
    prepareAccumulators(
        N, means, accumulateMean2dGradients, accumulateMax2dRadii, accumGradNorms, accumStepCounts,
        accumMax2dRadii);

    std::optional<torch::Tensor> maybeGradNorms =
        accumulateMean2dGradients ? accumGradNorms : std::nullopt;
    std::optional<torch::Tensor> maybeStepCounts =
        accumulateMean2dGradients ? accumStepCounts : std::nullopt;
    std::optional<torch::Tensor> maybeMax2dRadii =
        accumulateMax2dRadii ? accumMax2dRadii : std::nullopt;

    // Project to image plane
    const auto projectionResults =
        fvdb::detail::autograd::ProjectGaussians::apply(means,
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
                                                        ortho,
                                                        maybeGradNorms,
                                                        maybeMax2dRadii,
                                                        maybeStepCounts);
    ret.perGaussianRadius = projectionResults[0];
    ret.perGaussian2dMean = projectionResults[1];
    ret.perGaussianDepth  = projectionResults[2];
    ret.perGaussianConic  = projectionResults[3];

    ret.perGaussianOpacity = torch::sigmoid(logitOpacities).repeat({C, 1});
    if (settings.antialias) {
        ret.perGaussianOpacity *= projectionResults[4];
        ret.perGaussianOpacity = ret.perGaussianOpacity.contiguous();
    }

    ret.perGaussianRenderQuantity = computeRenderQuantity(
        means, sh0, shN, settings, worldToCameraMatrices, ret.perGaussianRadius,
        ret.perGaussianDepth);

    // Intersect projected Gaussians with image tiles [non-differentiable]
    const int numTilesW = std::ceil(settings.imageWidth / static_cast<float>(settings.tileSize));
    const int numTilesH = std::ceil(settings.imageHeight / static_cast<float>(settings.tileSize));
    const auto [tileOffsets, tileGaussianIds] = FVDB_DISPATCH_KERNEL(means.device(), [&]() {
        return dispatchGaussianTileIntersection<DeviceTag>(ret.perGaussian2dMean,
                                                           ret.perGaussianRadius,
                                                           ret.perGaussianDepth,
                                                           at::nullopt,
                                                           C,
                                                           settings.tileSize,
                                                           numTilesH,
                                                           numTilesW);
    });
    ret.tileOffsets     = tileOffsets;
    ret.tileGaussianIds = tileGaussianIds;

    return ret;
}

fvdb::GaussianSplat3d::ProjectedGaussianSplats
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

    fvdb::GaussianSplat3d::ProjectedGaussianSplats ret;
    ret.mRenderSettings   = settings;
    ret.mCameraModel      = cameraModel;
    ret.mProjectionMethod = resolveProjectionMethod(cameraModel, ProjectionMethod::UNSCENTED);

    UTParams utParams            = UTParams{};
    const auto projectionResults = FVDB_DISPATCH_KERNEL(means.device(), [&]() {
        return dispatchGaussianProjectionForwardUT<DeviceTag>(means,
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
    });

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

    ret.perGaussianRenderQuantity = computeRenderQuantity(
        means, sh0, shN, settings, worldToCameraMatrices, ret.perGaussianRadius,
        ret.perGaussianDepth);

    const int numTilesW = std::ceil(settings.imageWidth / static_cast<float>(settings.tileSize));
    const int numTilesH = std::ceil(settings.imageHeight / static_cast<float>(settings.tileSize));
    std::tie(ret.tileOffsets, ret.tileGaussianIds) = FVDB_DISPATCH_KERNEL(means.device(), [&]() {
        return dispatchGaussianTileIntersection<DeviceTag>(ret.perGaussian2dMean,
                                                           ret.perGaussianRadius,
                                                           ret.perGaussianDepth,
                                                           at::nullopt,
                                                           C,
                                                           settings.tileSize,
                                                           numTilesH,
                                                           numTilesW);
    });

    return ret;
}

fvdb::GaussianSplat3d::ProjectedGaussianSplats
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
        return projectGaussiansAnalytic(means, quats, logScales, logitOpacities, sh0, shN,
                                        worldToCameraMatrices, projectionMatrices, settings,
                                        cameraModel, accumulateMean2dGradients, accumulateMax2dRadii,
                                        accumGradNorms, accumStepCounts, accumMax2dRadii);
    }

    const int C                              = worldToCameraMatrices.size(0);
    const torch::Tensor distortionCoeffsTensor = distortionCoeffs.has_value()
                                                     ? distortionCoeffs.value()
                                                     : torch::empty({C, 0}, means.options());

    return projectGaussiansUT(means, quats, logScales, logitOpacities, sh0, shN,
                              worldToCameraMatrices, projectionMatrices, distortionCoeffsTensor,
                              settings, cameraModel);
}

} // namespace fvdb::detail::ops
