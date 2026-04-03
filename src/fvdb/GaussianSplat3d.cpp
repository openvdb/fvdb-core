// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/GaussianSplat3d.h>
#include <fvdb/detail/io/GaussianPlyIO.h>
#include <fvdb/detail/ops/gsplat/GaussianMCMCAddNoise.h>
#include <fvdb/detail/ops/gsplat/GaussianMCMCRelocation.h>
#include <fvdb/detail/ops/gsplat/GaussianSplatOps.h>
#include <fvdb/detail/utils/Nvtx.h>
#include <fvdb/detail/utils/Utils.h>

// Ops headers
#include <fvdb/detail/ops/gsplat/GaussianProjectionForward.h>
#include <fvdb/detail/ops/gsplat/GaussianProjectionJaggedForward.h>
#include <fvdb/detail/ops/gsplat/GaussianRasterizeForward.h>
#include <fvdb/detail/ops/gsplat/GaussianRasterizeFromWorldForward.h>
#include <fvdb/detail/ops/gsplat/GaussianSphericalHarmonicsForward.h>
#include <fvdb/detail/ops/gsplat/GaussianCameras.cuh>
#include <fvdb/detail/ops/gsplat/GaussianProjectionUT.h>
#include <fvdb/detail/ops/gsplat/GaussianRasterizeContributingGaussianIds.h>
#include <fvdb/detail/ops/gsplat/GaussianRasterizeNumContributingGaussians.h>
#include <fvdb/detail/ops/gsplat/GaussianRasterizeTopContributingGaussianIds.h>
#include <fvdb/detail/ops/gsplat/GaussianRenderSettings.h>
#include <fvdb/detail/ops/gsplat/GaussianSplatSparse.h>
#include <fvdb/detail/ops/gsplat/GaussianTileIntersection.h>

#include <ATen/core/TensorBody.h>
#include <ATen/core/grad_mode.h>
#include <c10/core/ScalarType.h>
#include <torch/csrc/autograd/generated/variable_factories.h>

namespace fvdb {

using RenderMode     = fvdb::detail::ops::RenderSettings::RenderMode;
using RenderSettings = fvdb::detail::ops::RenderSettings;

using CameraModel      = fvdb::GaussianSplat3d::CameraModel;
using ProjectionMethod = fvdb::GaussianSplat3d::ProjectionMethod;

namespace gsplat = fvdb::detail::ops;

torch::Tensor
GaussianSplat3d::evalSphericalHarmonicsImpl(const int64_t shDegreeToUse,
                                            const torch::Tensor &worldToCameraMatrices,
                                            const torch::Tensor &perGaussianProjectedRadii) const {
    return gsplat::evalSphericalHarmonics(
        mMeans, mSh0, mShN, shDegreeToUse, worldToCameraMatrices, perGaussianProjectedRadii);
}

void
GaussianSplat3d::checkState(const torch::Tensor &means,
                            const torch::Tensor &quats,
                            const torch::Tensor &logScales,
                            const torch::Tensor &logitOpacities,
                            const torch::Tensor &sh0,
                            const torch::Tensor &shN) {
    gsplat::checkGaussianState(means, quats, logScales, logitOpacities, sh0, shN);
}

GaussianSplat3d::GaussianSplat3d(const torch::Tensor &means,
                                 const torch::Tensor &quats,
                                 const torch::Tensor &logScales,
                                 const torch::Tensor &logitOpacities,
                                 const torch::Tensor &sh0,
                                 const torch::Tensor &shN,
                                 const bool accumulateMeans2dGradients,
                                 const bool accumulateMax2dRadii,
                                 const bool copyAndDetach)
    : mMeans(means), mQuats(quats), mLogScales(logScales), mLogitOpacities(logitOpacities),
      mSh0(sh0), mShN(shN), mAccumulateMean2dGradients(accumulateMeans2dGradients),
      mAccumulateMax2dRadii(accumulateMax2dRadii) {
    const int64_t N = means.size(0); // number of gaussians
    if (mSh0.dim() == 2) {
        TORCH_CHECK(mSh0.size(0) == N, "sh0 must have shape (N, 1, D) or (N, D)");
        mSh0 = mSh0.unsqueeze(1);
    }
    if (copyAndDetach) {
        mMeans          = means.detach();
        mQuats          = quats.detach();
        mLogScales      = logScales.detach();
        mLogitOpacities = logitOpacities.detach();
        mSh0            = sh0.detach();
        mShN            = shN.detach();
    }
    checkState(mMeans, mQuats, mLogScales, mLogitOpacities, mSh0, mShN);
}

void
GaussianSplat3d::setState(const torch::Tensor &means,
                          const torch::Tensor &quats,
                          const torch::Tensor &logScales,
                          const torch::Tensor &logitOpacities,
                          const torch::Tensor &sh0,
                          const torch::Tensor &shN) {
    checkState(means, quats, logScales, logitOpacities, sh0, shN);
    resetAccumulatedGradientState();

    mMeans          = means;
    mQuats          = quats;
    mLogScales      = logScales;
    mLogitOpacities = logitOpacities;
    mSh0            = sh0;
    mShN            = shN;
}

std::unordered_map<std::string, torch::Tensor>
GaussianSplat3d::stateDict() const {
    auto ret = std::unordered_map<std::string, torch::Tensor>{{"means", mMeans},
                                                              {"quats", mQuats},
                                                              {"log_scales", mLogScales},
                                                              {"logit_opacities", mLogitOpacities},
                                                              {"sh0", mSh0},
                                                              {"shN", mShN}};

    const auto boolOpts = torch::TensorOptions().dtype(torch::kBool);
    ret["accumulate_means_2d_gradients"] =
        mAccumulateMean2dGradients ? torch::ones({}, boolOpts) : torch::zeros({}, boolOpts);
    ret["accumulate_max_2d_radii"] =
        mAccumulateMax2dRadii ? torch::ones({}, boolOpts) : torch::zeros({}, boolOpts);

    if (mAccumulatedNormalized2dMeansGradientNormsForGrad.numel() != 0) {
        ret["accumulated_mean_2d_gradient_norms_for_grad"] =
            mAccumulatedNormalized2dMeansGradientNormsForGrad;
    }
    if (mAccumulated2dRadiiForGrad.numel() != 0) {
        ret["accumulated_max_2d_radii_for_grad"] = mAccumulated2dRadiiForGrad;
    }
    if (mGradientStepCountForGrad.numel() != 0) {
        ret["accumulated_gradient_step_counts_for_grad"] = mGradientStepCountForGrad;
    }
    return ret;
}

void
GaussianSplat3d::loadStateDict(const std::unordered_map<std::string, torch::Tensor> &stateDict) {
    TORCH_CHECK_VALUE(stateDict.count("means") == 1, "Missing key 'means' in state dict");
    TORCH_CHECK_VALUE(stateDict.count("quats") == 1, "Missing key 'quats' in state dict");
    TORCH_CHECK_VALUE(stateDict.count("log_scales") == 1, "Missing key 'log_scales' in state dict");
    TORCH_CHECK_VALUE(stateDict.count("logit_opacities") == 1,
                      "Missing key 'logit_opacities' in state dict");
    TORCH_CHECK_VALUE(stateDict.count("sh0") == 1, "Missing key 'sh0' in state dict");
    TORCH_CHECK_VALUE(stateDict.count("shN") == 1, "Missing key 'shN' in state dict");

    TORCH_CHECK_VALUE(stateDict.count("accumulate_means_2d_gradients") == 1,
                      "Missing key 'accumulate_means_2d_gradients' in state dict");

    TORCH_CHECK_VALUE(stateDict.count("accumulate_max_2d_radii") == 1,
                      "Missing key 'accumulate_max_2d_radii' in state dict");

    const torch::Tensor means          = stateDict.at("means");
    const torch::Tensor quats          = stateDict.at("quats");
    const torch::Tensor logScales      = stateDict.at("log_scales");
    const torch::Tensor logitOpacities = stateDict.at("logit_opacities");
    const torch::Tensor sh0            = stateDict.at("sh0");
    const torch::Tensor shN            = stateDict.at("shN");

    const int64_t N = means.size(0); // number of gaussians

    checkState(means, quats, logScales, logitOpacities, sh0, shN);

    const bool accumulateMeans2dGrad =
        stateDict.at("accumulate_means_2d_gradients").item().toBool();
    const bool accumulateMax2dRadii = stateDict.at("accumulate_max_2d_radii").item().toBool();
    torch::Tensor accumulatedNormalized2dMeansGradientNormsForGrad;
    torch::Tensor accumulated2dRadiiForGrad;
    torch::Tensor gradientStepCountForGrad;

    if (stateDict.count("accumulated_mean_2d_gradient_norms_for_grad") > 0) {
        accumulatedNormalized2dMeansGradientNormsForGrad =
            stateDict.at("accumulated_mean_2d_gradient_norms_for_grad");
        TORCH_CHECK_VALUE(accumulatedNormalized2dMeansGradientNormsForGrad.numel() == N,
                          "accumulated_mean_2d_gradient_norms_for_grad must have shape (N)");
        TORCH_CHECK_VALUE(
            accumulatedNormalized2dMeansGradientNormsForGrad.device() == means.device(),
            "accumulated_mean_2d_gradient_norms_for_grad must be on the same device as "
            "means");
        TORCH_CHECK_VALUE(accumulatedNormalized2dMeansGradientNormsForGrad.dim() == 1,
                          "accumulated_mean_2d_gradient_norms_for_grad must have one dimension");
        TORCH_CHECK_VALUE(accumulatedNormalized2dMeansGradientNormsForGrad.scalar_type() ==
                              means.scalar_type(),
                          "accumulated_mean_2d_gradient_norms_for_grad must have the same type as "
                          "means");
        TORCH_CHECK_VALUE(stateDict.count("accumulated_gradient_step_counts_for_grad") != 0,
                          "gradient_step_counts_for_grad "
                          "must be non-empty if "
                          "accumulated_mean_2d_gradient_norms_for_grad "
                          "is non-empty");
        gradientStepCountForGrad = stateDict.at("accumulated_gradient_step_counts_for_grad");
        TORCH_CHECK_VALUE(gradientStepCountForGrad.numel() != 0,
                          "gradient_step_counts_for_grad "
                          "must be non-empty if "
                          "accumulated_mean_2d_gradient_norms_for_grad "
                          "is non-empty");
        TORCH_CHECK_VALUE(gradientStepCountForGrad.numel() == N,
                          "accumulated_gradient_step_counts_for_grad must have shape (N)");
        TORCH_CHECK_VALUE(gradientStepCountForGrad.device() == means.device(),
                          "accumulated_gradient_step_counts_for_grad must be on the same device as "
                          "means");
        TORCH_CHECK_VALUE(gradientStepCountForGrad.dim() == 1,
                          "accumulated_gradient_step_counts_for_grad must have one dimension");
        TORCH_CHECK_VALUE(gradientStepCountForGrad.scalar_type() == torch::kInt32,
                          "accumulated_gradient_step_counts_for_grad must be of type int32");
    }

    if (stateDict.count("accumulated_max_2d_radii_for_grad") > 0) {
        accumulated2dRadiiForGrad = stateDict.at("accumulated_max_2d_radii_for_grad");
        TORCH_CHECK_VALUE(accumulated2dRadiiForGrad.numel() == N,
                          "accumulated_max_2d_radii_for_grad must have shape (N)");
        TORCH_CHECK_VALUE(accumulated2dRadiiForGrad.device() == means.device(),
                          "accumulated_max_2d_radii_for_grad must be on the same device as "
                          "means");
        TORCH_CHECK_VALUE(accumulated2dRadiiForGrad.dim() == 1,
                          "accumulated_max_2d_radii_for_grad must have one dimension");
        TORCH_CHECK_VALUE(accumulated2dRadiiForGrad.scalar_type() == torch::kInt32,
                          "accumulated_max_2d_radii_for_grad must be of type int32");
    }

    mMeans          = means;
    mQuats          = quats;
    mLogScales      = logScales;
    mLogitOpacities = logitOpacities;
    mSh0            = sh0;
    mShN            = shN;

    mAccumulateMean2dGradients = accumulateMeans2dGrad;
    mAccumulateMax2dRadii      = accumulateMax2dRadii;
    mAccumulatedNormalized2dMeansGradientNormsForGrad =
        accumulatedNormalized2dMeansGradientNormsForGrad;
    mAccumulated2dRadiiForGrad = accumulated2dRadiiForGrad;
    mGradientStepCountForGrad  = gradientStepCountForGrad;
}

GaussianSplat3d::ProjectedGaussianSplats
GaussianSplat3d::projectGaussiansImpl(const torch::Tensor &worldToCameraMatrices,
                                      const torch::Tensor &projectionMatrices,
                                      const RenderSettings &settings,
                                      const CameraModel cameraModel) {
    std::optional<torch::Tensor> accumGradNorms =
        mAccumulatedNormalized2dMeansGradientNormsForGrad.defined()
            ? std::optional<torch::Tensor>(mAccumulatedNormalized2dMeansGradientNormsForGrad)
            : std::nullopt;
    std::optional<torch::Tensor> accumStepCounts =
        mGradientStepCountForGrad.defined()
            ? std::optional<torch::Tensor>(mGradientStepCountForGrad)
            : std::nullopt;
    std::optional<torch::Tensor> accumMax2dRadii =
        mAccumulated2dRadiiForGrad.defined()
            ? std::optional<torch::Tensor>(mAccumulated2dRadiiForGrad)
            : std::nullopt;

    auto ret = gsplat::projectGaussiansAnalytic(
        mMeans, mQuats, mLogScales, mLogitOpacities, mSh0, mShN, worldToCameraMatrices,
        projectionMatrices, settings, cameraModel, mAccumulateMean2dGradients,
        mAccumulateMax2dRadii, accumGradNorms, accumStepCounts, accumMax2dRadii);

    // Write back accumulators
    if (accumGradNorms.has_value())
        mAccumulatedNormalized2dMeansGradientNormsForGrad = accumGradNorms.value();
    if (accumStepCounts.has_value())
        mGradientStepCountForGrad = accumStepCounts.value();
    if (accumMax2dRadii.has_value())
        mAccumulated2dRadiiForGrad = accumMax2dRadii.value();

    return ret;
}

GaussianSplat3d::ProjectedGaussianSplats
GaussianSplat3d::projectGaussiansForCameraImpl(
    const torch::Tensor &worldToCameraMatrices,
    const torch::Tensor &projectionMatrices,
    const RenderSettings &settings,
    const CameraModel cameraModel,
    const ProjectionMethod projectionMethod,
    const std::optional<torch::Tensor> &distortionCoeffs) {
    std::optional<torch::Tensor> accumGradNorms =
        mAccumulatedNormalized2dMeansGradientNormsForGrad.defined()
            ? std::optional<torch::Tensor>(mAccumulatedNormalized2dMeansGradientNormsForGrad)
            : std::nullopt;
    std::optional<torch::Tensor> accumStepCounts =
        mGradientStepCountForGrad.defined()
            ? std::optional<torch::Tensor>(mGradientStepCountForGrad)
            : std::nullopt;
    std::optional<torch::Tensor> accumMax2dRadii =
        mAccumulated2dRadiiForGrad.defined()
            ? std::optional<torch::Tensor>(mAccumulated2dRadiiForGrad)
            : std::nullopt;

    auto ret = gsplat::projectGaussiansForCamera(
        mMeans, mQuats, mLogScales, mLogitOpacities, mSh0, mShN, worldToCameraMatrices,
        projectionMatrices, settings, cameraModel, projectionMethod, distortionCoeffs,
        mAccumulateMean2dGradients, mAccumulateMax2dRadii, accumGradNorms, accumStepCounts,
        accumMax2dRadii);

    // Write back accumulators
    if (accumGradNorms.has_value())
        mAccumulatedNormalized2dMeansGradientNormsForGrad = accumGradNorms.value();
    if (accumStepCounts.has_value())
        mGradientStepCountForGrad = accumStepCounts.value();
    if (accumMax2dRadii.has_value())
        mAccumulated2dRadiiForGrad = accumMax2dRadii.value();

    return ret;
}

GaussianSplat3d::SparseProjectedGaussianSplats
GaussianSplat3d::sparseProjectGaussiansImpl(const JaggedTensor &pixelsToRender,
                                            const torch::Tensor &worldToCameraMatrices,
                                            const torch::Tensor &projectionMatrices,
                                            const RenderSettings &settings,
                                            const CameraModel cameraModel) {
    std::optional<torch::Tensor> accumGradNorms =
        mAccumulatedNormalized2dMeansGradientNormsForGrad.defined()
            ? std::optional<torch::Tensor>(mAccumulatedNormalized2dMeansGradientNormsForGrad)
            : std::nullopt;
    std::optional<torch::Tensor> accumStepCounts =
        mGradientStepCountForGrad.defined()
            ? std::optional<torch::Tensor>(mGradientStepCountForGrad)
            : std::nullopt;
    std::optional<torch::Tensor> accumMax2dRadii =
        mAccumulated2dRadiiForGrad.defined()
            ? std::optional<torch::Tensor>(mAccumulated2dRadiiForGrad)
            : std::nullopt;

    auto ret = gsplat::sparseProjectGaussiansAnalytic(
        pixelsToRender, mMeans, mQuats, mLogScales, mLogitOpacities, mSh0, mShN,
        worldToCameraMatrices, projectionMatrices, settings, cameraModel,
        mAccumulateMean2dGradients, mAccumulateMax2dRadii, accumGradNorms, accumStepCounts,
        accumMax2dRadii);

    if (accumGradNorms.has_value())
        mAccumulatedNormalized2dMeansGradientNormsForGrad = accumGradNorms.value();
    if (accumStepCounts.has_value())
        mGradientStepCountForGrad = accumStepCounts.value();
    if (accumMax2dRadii.has_value())
        mAccumulated2dRadiiForGrad = accumMax2dRadii.value();

    return ret;
}

GaussianSplat3d::SparseProjectedGaussianSplats
GaussianSplat3d::sparseProjectGaussiansForCameraImpl(
    const JaggedTensor &pixelsToRender,
    const torch::Tensor &worldToCameraMatrices,
    const torch::Tensor &projectionMatrices,
    const RenderSettings &settings,
    const CameraModel cameraModel,
    const ProjectionMethod projectionMethod,
    const std::optional<torch::Tensor> &distortionCoeffs) {
    std::optional<torch::Tensor> accumGradNorms =
        mAccumulatedNormalized2dMeansGradientNormsForGrad.defined()
            ? std::optional<torch::Tensor>(mAccumulatedNormalized2dMeansGradientNormsForGrad)
            : std::nullopt;
    std::optional<torch::Tensor> accumStepCounts =
        mGradientStepCountForGrad.defined()
            ? std::optional<torch::Tensor>(mGradientStepCountForGrad)
            : std::nullopt;
    std::optional<torch::Tensor> accumMax2dRadii =
        mAccumulated2dRadiiForGrad.defined()
            ? std::optional<torch::Tensor>(mAccumulated2dRadiiForGrad)
            : std::nullopt;

    auto ret = gsplat::sparseProjectGaussiansForCamera(
        pixelsToRender, mMeans, mQuats, mLogScales, mLogitOpacities, mSh0, mShN,
        worldToCameraMatrices, projectionMatrices, settings, cameraModel, projectionMethod,
        distortionCoeffs, mAccumulateMean2dGradients, mAccumulateMax2dRadii, accumGradNorms,
        accumStepCounts, accumMax2dRadii);

    if (accumGradNorms.has_value())
        mAccumulatedNormalized2dMeansGradientNormsForGrad = accumGradNorms.value();
    if (accumStepCounts.has_value())
        mGradientStepCountForGrad = accumStepCounts.value();
    if (accumMax2dRadii.has_value())
        mAccumulated2dRadiiForGrad = accumMax2dRadii.value();

    return ret;
}

std::tuple<torch::Tensor, torch::Tensor>
GaussianSplat3d::renderCropFromProjectedGaussiansImpl(
    const ProjectedGaussianSplats &projectedGaussians,
    const size_t tileSize,
    const ssize_t cropWidth,
    const ssize_t cropHeight,
    const ssize_t cropOriginW,
    const ssize_t cropOriginH,
    const std::optional<torch::Tensor> &backgrounds,
    const std::optional<torch::Tensor> &masks) {
    return gsplat::renderCropFromProjected(
        projectedGaussians, tileSize, cropWidth, cropHeight, cropOriginW, cropOriginH, backgrounds,
        masks);
}

std::tuple<JaggedTensor, JaggedTensor>
GaussianSplat3d::sparseRenderImpl(const JaggedTensor &pixelsToRender,
                                  const torch::Tensor &worldToCameraMatrices,
                                  const torch::Tensor &projectionMatrices,
                                  const fvdb::detail::ops::RenderSettings &settings,
                                  const CameraModel cameraModel,
                                  const ProjectionMethod projectionMethod,
                                  const std::optional<torch::Tensor> &distortionCoeffs,
                                  const std::optional<torch::Tensor> &backgrounds,
                                  const std::optional<torch::Tensor> &masks) {
    std::optional<torch::Tensor> accumGradNorms =
        mAccumulatedNormalized2dMeansGradientNormsForGrad.defined()
            ? std::optional<torch::Tensor>(mAccumulatedNormalized2dMeansGradientNormsForGrad)
            : std::nullopt;
    std::optional<torch::Tensor> accumStepCounts =
        mGradientStepCountForGrad.defined()
            ? std::optional<torch::Tensor>(mGradientStepCountForGrad)
            : std::nullopt;
    std::optional<torch::Tensor> accumMax2dRadii =
        mAccumulated2dRadiiForGrad.defined()
            ? std::optional<torch::Tensor>(mAccumulated2dRadiiForGrad)
            : std::nullopt;

    auto ret = gsplat::sparseRender(
        pixelsToRender, mMeans, mQuats, mLogScales, mLogitOpacities, mSh0, mShN,
        worldToCameraMatrices, projectionMatrices, settings, cameraModel, projectionMethod,
        distortionCoeffs, backgrounds, masks, mAccumulateMean2dGradients, mAccumulateMax2dRadii,
        accumGradNorms, accumStepCounts, accumMax2dRadii);

    if (accumGradNorms.has_value())
        mAccumulatedNormalized2dMeansGradientNormsForGrad = accumGradNorms.value();
    if (accumStepCounts.has_value())
        mGradientStepCountForGrad = accumStepCounts.value();
    if (accumMax2dRadii.has_value())
        mAccumulated2dRadiiForGrad = accumMax2dRadii.value();

    return ret;
}

std::tuple<torch::Tensor, torch::Tensor>
GaussianSplat3d::renderNumContributingGaussiansImpl(
    const torch::Tensor &worldToCameraMatrices,
    const torch::Tensor &projectionMatrices,
    const fvdb::detail::ops::RenderSettings &settings,
    const CameraModel cameraModel,
    const ProjectionMethod projectionMethod,
    const std::optional<torch::Tensor> &distortionCoeffs) {
    const ProjectedGaussianSplats &state = projectGaussiansForCameraImpl(worldToCameraMatrices,
                                                                         projectionMatrices,
                                                                         settings,
                                                                         cameraModel,
                                                                         projectionMethod,
                                                                         distortionCoeffs);
    return gsplat::renderNumContributing(state, settings);
}

std::tuple<fvdb::JaggedTensor, fvdb::JaggedTensor>
GaussianSplat3d::sparseRenderNumContributingGaussiansImpl(
    const fvdb::JaggedTensor &pixelsToRender,
    const torch::Tensor &worldToCameraMatrices,
    const torch::Tensor &projectionMatrices,
    const fvdb::detail::ops::RenderSettings &settings,
    const CameraModel cameraModel,
    const ProjectionMethod projectionMethod,
    const std::optional<torch::Tensor> &distortionCoeffs) {
    const SparseProjectedGaussianSplats &state =
        sparseProjectGaussiansForCameraImpl(pixelsToRender,
                                            worldToCameraMatrices,
                                            projectionMatrices,
                                            settings,
                                            cameraModel,
                                            projectionMethod,
                                            distortionCoeffs);
    return gsplat::sparseRenderNumContributing(state, pixelsToRender, settings);
}

std::tuple<fvdb::JaggedTensor, fvdb::JaggedTensor>
GaussianSplat3d::renderContributingGaussianIdsImpl(
    const torch::Tensor &worldToCameraMatrices,
    const torch::Tensor &projectionMatrices,
    const fvdb::detail::ops::RenderSettings &settings,
    const CameraModel cameraModel,
    const ProjectionMethod projectionMethod,
    const std::optional<torch::Tensor> &distortionCoeffs,
    const std::optional<torch::Tensor> &maybeNumContributingGaussians) {
    const ProjectedGaussianSplats &state = projectGaussiansForCameraImpl(worldToCameraMatrices,
                                                                         projectionMatrices,
                                                                         settings,
                                                                         cameraModel,
                                                                         projectionMethod,
                                                                         distortionCoeffs);
    return gsplat::renderContributingIds(state, settings, maybeNumContributingGaussians);
}

std::tuple<fvdb::JaggedTensor, fvdb::JaggedTensor>
GaussianSplat3d::sparseRenderContributingGaussianIdsImpl(
    const fvdb::JaggedTensor &pixelsToRender,
    const torch::Tensor &worldToCameraMatrices,
    const torch::Tensor &projectionMatrices,
    const fvdb::detail::ops::RenderSettings &settings,
    const CameraModel cameraModel,
    const ProjectionMethod projectionMethod,
    const std::optional<torch::Tensor> &distortionCoeffs,
    const std::optional<fvdb::JaggedTensor> &maybeNumContributingGaussians) {
    const SparseProjectedGaussianSplats &state =
        sparseProjectGaussiansForCameraImpl(pixelsToRender,
                                            worldToCameraMatrices,
                                            projectionMatrices,
                                            settings,
                                            cameraModel,
                                            projectionMethod,
                                            distortionCoeffs);
    return gsplat::sparseRenderContributingIds(
        state, pixelsToRender, settings, maybeNumContributingGaussians);
}

GaussianSplat3d::ProjectedGaussianSplats
GaussianSplat3d::projectGaussiansForImages(const torch::Tensor &worldToCameraMatrices,
                                           const torch::Tensor &projectionMatrices,
                                           size_t imageWidth,
                                           size_t imageHeight,
                                           const float near,
                                           const float far,
                                           const CameraModel cameraModel,
                                           const ProjectionMethod projectionMethod,
                                           const std::optional<torch::Tensor> &distortionCoeffs,
                                           const int64_t shDegreeToUse,
                                           const float minRadius2d,
                                           const float eps2d,
                                           const bool antialias) {
    RenderSettings settings;
    settings.imageWidth    = imageWidth;
    settings.imageHeight   = imageHeight;
    settings.nearPlane     = near;
    settings.farPlane      = far;
    settings.shDegreeToUse = shDegreeToUse;
    settings.radiusClip    = minRadius2d;
    settings.eps2d         = eps2d;
    settings.antialias     = antialias;

    settings.renderMode = RenderMode::RGB;

    return projectGaussiansForCameraImpl(worldToCameraMatrices,
                                         projectionMatrices,
                                         settings,
                                         cameraModel,
                                         projectionMethod,
                                         distortionCoeffs);
}

GaussianSplat3d::ProjectedGaussianSplats
GaussianSplat3d::projectGaussiansForDepths(const torch::Tensor &worldToCameraMatrices,
                                           const torch::Tensor &projectionMatrices,
                                           size_t imageWidth,
                                           size_t imageHeight,
                                           const float near,
                                           const float far,
                                           const CameraModel cameraModel,
                                           const ProjectionMethod projectionMethod,
                                           const std::optional<torch::Tensor> &distortionCoeffs,
                                           const float minRadius2d,
                                           const float eps2d,
                                           const bool antialias) {
    RenderSettings settings;
    settings.imageWidth    = imageWidth;
    settings.imageHeight   = imageHeight;
    settings.nearPlane     = near;
    settings.farPlane      = far;
    settings.shDegreeToUse = -1;
    settings.radiusClip    = minRadius2d;
    settings.eps2d         = eps2d;
    settings.antialias     = antialias;
    settings.renderMode    = RenderMode::DEPTH;

    return projectGaussiansForCameraImpl(worldToCameraMatrices,
                                         projectionMatrices,
                                         settings,
                                         cameraModel,
                                         projectionMethod,
                                         distortionCoeffs);
}

GaussianSplat3d::ProjectedGaussianSplats
GaussianSplat3d::projectGaussiansForImagesAndDepths(
    const torch::Tensor &worldToCameraMatrices,
    const torch::Tensor &projectionMatrices,
    size_t imageWidth,
    size_t imageHeight,
    const float near,
    const float far,
    const CameraModel cameraModel,
    const ProjectionMethod projectionMethod,
    const std::optional<torch::Tensor> &distortionCoeffs,
    const int64_t shDegreeToUse,
    const float minRadius2d,
    const float eps2d,
    const bool antialias) {
    RenderSettings settings;
    settings.imageWidth    = imageWidth;
    settings.imageHeight   = imageHeight;
    settings.nearPlane     = near;
    settings.farPlane      = far;
    settings.shDegreeToUse = shDegreeToUse;
    settings.radiusClip    = minRadius2d;
    settings.eps2d         = eps2d;
    settings.antialias     = antialias;

    settings.renderMode = RenderMode::RGBD;

    return projectGaussiansForCameraImpl(worldToCameraMatrices,
                                         projectionMatrices,
                                         settings,
                                         cameraModel,
                                         projectionMethod,
                                         distortionCoeffs);
}

namespace {

/// @brief Get a uint8_t pointer to the data of a tensor
/// @param tensor The tensor to get the pointer to
/// @return A uint8_t pointer to the data of the tensor
inline uint8_t *
tensorBytePointer(const torch::Tensor &tensor) {
    return static_cast<uint8_t *>(tensor.data_ptr());
}

} // namespace

void
GaussianSplat3d::savePly(
    const std::string &filename,
    std::optional<std::unordered_map<std::string, PlyMetadataTypes>> metadata) const {
    detail::io::saveGaussianPly(filename, mMeans, mQuats, mLogScales, mLogitOpacities,
                                mSh0, mShN, metadata);
}

std::tuple<GaussianSplat3d, std::unordered_map<std::string, GaussianSplat3d::PlyMetadataTypes>>
GaussianSplat3d::fromPly(const std::string &filename, torch::Device device) {
    auto [means, quats, logScales, logitOpacities, sh0, shN, metadata] =
        detail::io::loadGaussianPly(filename, device);
    return std::make_tuple(
        GaussianSplat3d(means, quats, logScales, logitOpacities, sh0, shN, false, false, false),
        metadata);
}

std::tuple<torch::Tensor, torch::Tensor>
GaussianSplat3d::renderFromProjectedGaussians(
    const GaussianSplat3d::ProjectedGaussianSplats &projectedGaussians,
    const ssize_t cropWidth,
    const ssize_t cropHeight,
    const ssize_t cropOriginW,
    const ssize_t cropOriginH,
    const size_t tileSize,
    const std::optional<torch::Tensor> &backgrounds,
    const std::optional<torch::Tensor> &masks) {
    return renderCropFromProjectedGaussiansImpl(projectedGaussians,
                                                tileSize,
                                                cropWidth,
                                                cropHeight,
                                                cropOriginW,
                                                cropOriginH,
                                                backgrounds,
                                                masks);
}

std::tuple<torch::Tensor, torch::Tensor>
GaussianSplat3d::renderImages(const torch::Tensor &worldToCameraMatrices,
                              const torch::Tensor &projectionMatrices,
                              const size_t imageWidth,
                              const size_t imageHeight,
                              const float near,
                              const float far,
                              const CameraModel cameraModel,
                              const ProjectionMethod projectionMethod,
                              const std::optional<torch::Tensor> &distortionCoeffs,
                              const int64_t shDegreeToUse,
                              const size_t tileSize,
                              const float minRadius2d,
                              const float eps2d,
                              const bool antialias,
                              const std::optional<torch::Tensor> &backgrounds,
                              const std::optional<torch::Tensor> &masks) {
    RenderSettings settings;
    settings.imageWidth    = imageWidth;
    settings.imageHeight   = imageHeight;
    settings.nearPlane     = near;
    settings.farPlane      = far;
    settings.shDegreeToUse = shDegreeToUse;
    settings.radiusClip    = minRadius2d;
    settings.eps2d         = eps2d;
    settings.antialias     = antialias;
    settings.tileSize      = tileSize;
    settings.renderMode    = RenderSettings::RenderMode::RGB;

    const ProjectedGaussianSplats state = projectGaussiansForCameraImpl(worldToCameraMatrices,
                                                                        projectionMatrices,
                                                                        settings,
                                                                        cameraModel,
                                                                        projectionMethod,
                                                                        distortionCoeffs);
    return renderCropFromProjectedGaussiansImpl(state,
                                                settings.tileSize,
                                                settings.imageWidth,
                                                settings.imageHeight,
                                                0,
                                                0,
                                                backgrounds,
                                                masks);
}

std::tuple<torch::Tensor, torch::Tensor>
GaussianSplat3d::renderImagesFromWorld(const torch::Tensor &worldToCameraMatrices,
                                       const torch::Tensor &projectionMatrices,
                                       const size_t imageWidth,
                                       const size_t imageHeight,
                                       const float near,
                                       const float far,
                                       const CameraModel cameraModel,
                                       const ProjectionMethod projectionMethod,
                                       const std::optional<torch::Tensor> &distortionCoeffs,
                                       const int64_t shDegreeToUse,
                                       const size_t tileSize,
                                       const float minRadius2d,
                                       const float eps2d,
                                       const bool antialias,
                                       const std::optional<torch::Tensor> &backgrounds,
                                       const std::optional<torch::Tensor> &masks) {
    FVDB_FUNC_RANGE();
    const int C = worldToCameraMatrices.size(0); // number of cameras
    TORCH_CHECK(C > 0, "At least one camera must be provided (got 0)");
    RenderSettings settings;
    settings.imageWidth    = imageWidth;
    settings.imageHeight   = imageHeight;
    settings.nearPlane     = near;
    settings.farPlane      = far;
    settings.shDegreeToUse = shDegreeToUse;
    settings.radiusClip    = minRadius2d;
    settings.eps2d         = eps2d;
    settings.antialias     = antialias;
    settings.tileSize      = tileSize;
    settings.renderMode    = RenderSettings::RenderMode::RGB;

    const ProjectedGaussianSplats state = projectGaussiansForCameraImpl(worldToCameraMatrices,
                                                                        projectionMatrices,
                                                                        settings,
                                                                        cameraModel,
                                                                        projectionMethod,
                                                                        distortionCoeffs);

    const torch::Tensor distortionCoeffsForRaster = distortionCoeffs.has_value()
                                                        ? distortionCoeffs.value()
                                                        : torch::empty({C, 0}, mMeans.options());

    return gsplat::rasterizeFromWorld(mMeans, mQuats, mLogScales, state, worldToCameraMatrices,
                                      projectionMatrices, distortionCoeffsForRaster, cameraModel,
                                      static_cast<uint32_t>(imageWidth),
                                      static_cast<uint32_t>(imageHeight),
                                      static_cast<uint32_t>(tileSize), backgrounds, masks);
}

std::tuple<torch::Tensor, torch::Tensor>
GaussianSplat3d::renderDepths(const torch::Tensor &worldToCameraMatrices,
                              const torch::Tensor &projectionMatrices,
                              const size_t imageWidth,
                              const size_t imageHeight,
                              const float near,
                              const float far,
                              const CameraModel cameraModel,
                              const ProjectionMethod projectionMethod,
                              const std::optional<torch::Tensor> &distortionCoeffs,
                              const size_t tileSize,
                              const float minRadius2d,
                              const float eps2d,
                              const bool antialias,
                              const std::optional<torch::Tensor> &backgrounds,
                              const std::optional<torch::Tensor> &masks) {
    RenderSettings settings;
    settings.imageWidth    = imageWidth;
    settings.imageHeight   = imageHeight;
    settings.nearPlane     = near;
    settings.farPlane      = far;
    settings.shDegreeToUse = -1;
    settings.radiusClip    = minRadius2d;
    settings.eps2d         = eps2d;
    settings.antialias     = antialias;
    settings.tileSize      = tileSize;
    settings.renderMode    = RenderSettings::RenderMode::DEPTH;

    const ProjectedGaussianSplats state = projectGaussiansForCameraImpl(worldToCameraMatrices,
                                                                        projectionMatrices,
                                                                        settings,
                                                                        cameraModel,
                                                                        projectionMethod,
                                                                        distortionCoeffs);
    return renderCropFromProjectedGaussiansImpl(state,
                                                settings.tileSize,
                                                settings.imageWidth,
                                                settings.imageHeight,
                                                0,
                                                0,
                                                backgrounds,
                                                masks);
}

std::tuple<torch::Tensor, torch::Tensor>
GaussianSplat3d::renderDepthsFromWorld(const torch::Tensor &worldToCameraMatrices,
                                       const torch::Tensor &projectionMatrices,
                                       const size_t imageWidth,
                                       const size_t imageHeight,
                                       const float near,
                                       const float far,
                                       const CameraModel cameraModel,
                                       const ProjectionMethod projectionMethod,
                                       const std::optional<torch::Tensor> &distortionCoeffs,
                                       const size_t tileSize,
                                       const float minRadius2d,
                                       const float eps2d,
                                       const bool antialias,
                                       const std::optional<torch::Tensor> &backgrounds,
                                       const std::optional<torch::Tensor> &masks) {
    FVDB_FUNC_RANGE();
    const int C = worldToCameraMatrices.size(0);
    TORCH_CHECK(C > 0, "At least one camera must be provided (got 0)");

    RenderSettings settings;
    settings.imageWidth    = imageWidth;
    settings.imageHeight   = imageHeight;
    settings.nearPlane     = near;
    settings.farPlane      = far;
    settings.shDegreeToUse = -1;
    settings.radiusClip    = minRadius2d;
    settings.eps2d         = eps2d;
    settings.antialias     = antialias;
    settings.tileSize      = tileSize;
    settings.renderMode    = RenderSettings::RenderMode::DEPTH;

    const ProjectedGaussianSplats state = projectGaussiansForCameraImpl(worldToCameraMatrices,
                                                                        projectionMatrices,
                                                                        settings,
                                                                        cameraModel,
                                                                        projectionMethod,
                                                                        distortionCoeffs);
    const torch::Tensor distortionCoeffsForRaster = distortionCoeffs.has_value()
                                                        ? distortionCoeffs.value()
                                                        : torch::empty({C, 0}, mMeans.options());

    return gsplat::rasterizeFromWorld(mMeans, mQuats, mLogScales, state, worldToCameraMatrices,
                                      projectionMatrices, distortionCoeffsForRaster, cameraModel,
                                      static_cast<uint32_t>(imageWidth),
                                      static_cast<uint32_t>(imageHeight),
                                      static_cast<uint32_t>(tileSize), backgrounds, masks);
}

std::tuple<torch::Tensor, torch::Tensor>
GaussianSplat3d::renderNumContributingGaussians(
    const torch::Tensor &worldToCameraMatrices,
    const torch::Tensor &projectionMatrices,
    const size_t imageWidth,
    const size_t imageHeight,
    const float near,
    const float far,
    const CameraModel cameraModel,
    const ProjectionMethod projectionMethod,
    const std::optional<torch::Tensor> &distortionCoeffs,
    const size_t tileSize,
    const float minRadius2d,
    const float eps2d,
    const bool antialias) {
    RenderSettings settings;
    settings.imageWidth    = imageWidth;
    settings.imageHeight   = imageHeight;
    settings.nearPlane     = near;
    settings.farPlane      = far;
    settings.shDegreeToUse = 0;
    settings.tileSize      = tileSize;
    settings.radiusClip    = minRadius2d;
    settings.eps2d         = eps2d;
    settings.antialias     = antialias;
    settings.renderMode    = RenderSettings::RenderMode::DEPTH;

    return renderNumContributingGaussiansImpl(worldToCameraMatrices,
                                              projectionMatrices,
                                              settings,
                                              cameraModel,
                                              projectionMethod,
                                              distortionCoeffs);
}

std::tuple<JaggedTensor, JaggedTensor>
GaussianSplat3d::sparseRenderDepths(const fvdb::JaggedTensor &pixelsToRender,
                                    const torch::Tensor &worldToCameraMatrices,
                                    const torch::Tensor &projectionMatrices,
                                    const size_t imageWidth,
                                    const size_t imageHeight,
                                    const float near,
                                    const float far,
                                    const CameraModel cameraModel,
                                    const ProjectionMethod projectionMethod,
                                    const std::optional<torch::Tensor> &distortionCoeffs,
                                    const size_t tileSize,
                                    const float minRadius2d,
                                    const float eps2d,
                                    const bool antialias,
                                    const std::optional<torch::Tensor> &backgrounds,
                                    const std::optional<torch::Tensor> &masks) {
    RenderSettings settings;
    settings.imageWidth    = imageWidth;
    settings.imageHeight   = imageHeight;
    settings.nearPlane     = near;
    settings.farPlane      = far;
    settings.shDegreeToUse = 0;
    settings.tileSize      = tileSize;
    settings.radiusClip    = minRadius2d;
    settings.eps2d         = eps2d;
    settings.antialias     = antialias;
    settings.renderMode    = RenderSettings::RenderMode::DEPTH;

    return sparseRenderImpl(pixelsToRender,
                            worldToCameraMatrices,
                            projectionMatrices,
                            settings,
                            cameraModel,
                            projectionMethod,
                            distortionCoeffs,
                            backgrounds,
                            masks);
}

std::tuple<JaggedTensor, JaggedTensor>
GaussianSplat3d::sparseRenderImages(const fvdb::JaggedTensor &pixelsToRender,
                                    const torch::Tensor &worldToCameraMatrices,
                                    const torch::Tensor &projectionMatrices,
                                    const size_t imageWidth,
                                    const size_t imageHeight,
                                    const float near,
                                    const float far,
                                    const CameraModel cameraModel,
                                    const ProjectionMethod projectionMethod,
                                    const std::optional<torch::Tensor> &distortionCoeffs,
                                    const int64_t shDegreeToUse,
                                    const size_t tileSize,
                                    const float minRadius2d,
                                    const float eps2d,
                                    const bool antialias,
                                    const std::optional<torch::Tensor> &backgrounds,
                                    const std::optional<torch::Tensor> &masks) {
    RenderSettings settings;
    settings.imageWidth    = imageWidth;
    settings.imageHeight   = imageHeight;
    settings.nearPlane     = near;
    settings.farPlane      = far;
    settings.shDegreeToUse = shDegreeToUse;
    settings.tileSize      = tileSize;
    settings.radiusClip    = minRadius2d;
    settings.eps2d         = eps2d;
    settings.antialias     = antialias;
    settings.renderMode    = RenderSettings::RenderMode::RGB;

    return sparseRenderImpl(pixelsToRender,
                            worldToCameraMatrices,
                            projectionMatrices,
                            settings,
                            cameraModel,
                            projectionMethod,
                            distortionCoeffs,
                            backgrounds,
                            masks);
}

std::tuple<JaggedTensor, JaggedTensor>
GaussianSplat3d::sparseRenderImagesAndDepths(const fvdb::JaggedTensor &pixelsToRender,
                                             const torch::Tensor &worldToCameraMatrices,
                                             const torch::Tensor &projectionMatrices,
                                             const size_t imageWidth,
                                             const size_t imageHeight,
                                             const float near,
                                             const float far,
                                             const CameraModel cameraModel,
                                             const ProjectionMethod projectionMethod,
                                             const std::optional<torch::Tensor> &distortionCoeffs,
                                             const int64_t shDegreeToUse,
                                             const size_t tileSize,
                                             const float minRadius2d,
                                             const float eps2d,
                                             const bool antialias,
                                             const std::optional<torch::Tensor> &backgrounds,
                                             const std::optional<torch::Tensor> &masks) {
    RenderSettings settings;
    settings.imageWidth    = imageWidth;
    settings.imageHeight   = imageHeight;
    settings.nearPlane     = near;
    settings.farPlane      = far;
    settings.shDegreeToUse = shDegreeToUse;
    settings.tileSize      = tileSize;
    settings.radiusClip    = minRadius2d;
    settings.eps2d         = eps2d;
    settings.antialias     = antialias;
    settings.renderMode    = RenderSettings::RenderMode::RGBD;

    return sparseRenderImpl(pixelsToRender,
                            worldToCameraMatrices,
                            projectionMatrices,
                            settings,
                            cameraModel,
                            projectionMethod,
                            distortionCoeffs,
                            backgrounds,
                            masks);
}

std::tuple<fvdb::JaggedTensor, fvdb::JaggedTensor>
GaussianSplat3d::sparseRenderNumContributingGaussians(
    const fvdb::JaggedTensor &pixelsToRender,
    const torch::Tensor &worldToCameraMatrices,
    const torch::Tensor &projectionMatrices,
    const size_t imageWidth,
    const size_t imageHeight,
    const float near,
    const float far,
    const CameraModel cameraModel,
    const ProjectionMethod projectionMethod,
    const std::optional<torch::Tensor> &distortionCoeffs,
    const size_t tileSize,
    const float minRadius2d,
    const float eps2d,
    const bool antialias) {
    RenderSettings settings;
    settings.imageWidth    = imageWidth;
    settings.imageHeight   = imageHeight;
    settings.nearPlane     = near;
    settings.farPlane      = far;
    settings.shDegreeToUse = 0;
    settings.tileSize      = tileSize;
    settings.radiusClip    = minRadius2d;
    settings.eps2d         = eps2d;
    settings.antialias     = antialias;
    settings.renderMode    = RenderSettings::RenderMode::DEPTH;

    return sparseRenderNumContributingGaussiansImpl(pixelsToRender,
                                                    worldToCameraMatrices,
                                                    projectionMatrices,
                                                    settings,
                                                    cameraModel,
                                                    projectionMethod,
                                                    distortionCoeffs);
}

std::tuple<fvdb::JaggedTensor, fvdb::JaggedTensor>
GaussianSplat3d::renderContributingGaussianIds(const torch::Tensor &worldToCameraMatrices,
                                               const torch::Tensor &projectionMatrices,
                                               const size_t imageWidth,
                                               const size_t imageHeight,
                                               const float near,
                                               const float far,
                                               const CameraModel cameraModel,
                                               const ProjectionMethod projectionMethod,
                                               const std::optional<torch::Tensor> &distortionCoeffs,
                                               const size_t tileSize,
                                               const float minRadius2d,
                                               const float eps2d,
                                               const bool antialias,
                                               const int topKContributors) {
    RenderSettings settings;
    settings.imageWidth      = imageWidth;
    settings.imageHeight     = imageHeight;
    settings.nearPlane       = near;
    settings.farPlane        = far;
    settings.shDegreeToUse   = 0;
    settings.tileSize        = tileSize;
    settings.radiusClip      = minRadius2d;
    settings.eps2d           = eps2d;
    settings.renderMode      = RenderSettings::RenderMode::DEPTH;
    settings.numDepthSamples = topKContributors;

    if (topKContributors > 0) {
        return renderContributingGaussianIdsImpl(worldToCameraMatrices,
                                                 projectionMatrices,
                                                 settings,
                                                 cameraModel,
                                                 projectionMethod,
                                                 distortionCoeffs);
    } else {
        // Use the standard path - compute actual number of contributing gaussians
        torch::Tensor numContributingGaussians, weights;
        std::tie(numContributingGaussians, weights) =
            renderNumContributingGaussiansImpl(worldToCameraMatrices,
                                               projectionMatrices,
                                               settings,
                                               cameraModel,
                                               projectionMethod,
                                               distortionCoeffs);
        return renderContributingGaussianIdsImpl(worldToCameraMatrices,
                                                 projectionMatrices,
                                                 settings,
                                                 cameraModel,
                                                 projectionMethod,
                                                 distortionCoeffs,
                                                 numContributingGaussians);
    }
}

std::tuple<fvdb::JaggedTensor, fvdb::JaggedTensor>
GaussianSplat3d::sparseRenderContributingGaussianIds(
    const fvdb::JaggedTensor &pixelsToRender,
    const torch::Tensor &worldToCameraMatrices,
    const torch::Tensor &projectionMatrices,
    const size_t imageWidth,
    const size_t imageHeight,
    const float near,
    const float far,
    const CameraModel cameraModel,
    const ProjectionMethod projectionMethod,
    const std::optional<torch::Tensor> &distortionCoeffs,
    const size_t tileSize,
    const float minRadius2d,
    const float eps2d,
    const bool antialias,
    const int topKContributors) {
    RenderSettings settings;
    settings.imageWidth      = imageWidth;
    settings.imageHeight     = imageHeight;
    settings.nearPlane       = near;
    settings.farPlane        = far;
    settings.shDegreeToUse   = 0;
    settings.tileSize        = tileSize;
    settings.radiusClip      = minRadius2d;
    settings.eps2d           = eps2d;
    settings.renderMode      = RenderSettings::RenderMode::DEPTH;
    settings.numDepthSamples = topKContributors;

    if (topKContributors > 0) {
        return sparseRenderContributingGaussianIdsImpl(pixelsToRender,
                                                       worldToCameraMatrices,
                                                       projectionMatrices,
                                                       settings,
                                                       cameraModel,
                                                       projectionMethod,
                                                       distortionCoeffs);
    } else {
        fvdb::JaggedTensor numContributingGaussians, weights;
        std::tie(numContributingGaussians, weights) =
            sparseRenderNumContributingGaussiansImpl(pixelsToRender,
                                                     worldToCameraMatrices,
                                                     projectionMatrices,
                                                     settings,
                                                     cameraModel,
                                                     projectionMethod,
                                                     distortionCoeffs);
        return sparseRenderContributingGaussianIdsImpl(pixelsToRender,
                                                       worldToCameraMatrices,
                                                       projectionMatrices,
                                                       settings,
                                                       cameraModel,
                                                       projectionMethod,
                                                       distortionCoeffs,
                                                       numContributingGaussians);
    }
}

std::tuple<torch::Tensor, torch::Tensor>
GaussianSplat3d::renderImagesAndDepths(const torch::Tensor &worldToCameraMatrices,
                                       const torch::Tensor &projectionMatrices,
                                       const size_t imageWidth,
                                       const size_t imageHeight,
                                       const float near,
                                       const float far,
                                       const CameraModel cameraModel,
                                       const ProjectionMethod projectionMethod,
                                       const std::optional<torch::Tensor> &distortionCoeffs,
                                       const int64_t shDegreeToUse,
                                       const size_t tileSize,
                                       const float minRadius2d,
                                       const float eps2d,
                                       const bool antialias,
                                       const std::optional<torch::Tensor> &backgrounds,
                                       const std::optional<torch::Tensor> &masks) {
    RenderSettings settings;
    settings.imageWidth    = imageWidth;
    settings.imageHeight   = imageHeight;
    settings.nearPlane     = near;
    settings.farPlane      = far;
    settings.shDegreeToUse = shDegreeToUse;
    settings.radiusClip    = minRadius2d;
    settings.eps2d         = eps2d;
    settings.antialias     = antialias;
    settings.tileSize      = tileSize;
    settings.renderMode    = RenderSettings::RenderMode::RGBD;

    const ProjectedGaussianSplats state = projectGaussiansForCameraImpl(worldToCameraMatrices,
                                                                        projectionMatrices,
                                                                        settings,
                                                                        cameraModel,
                                                                        projectionMethod,
                                                                        distortionCoeffs);
    return renderCropFromProjectedGaussiansImpl(state,
                                                settings.tileSize,
                                                settings.imageWidth,
                                                settings.imageHeight,
                                                0,
                                                0,
                                                backgrounds,
                                                masks);
}

std::tuple<torch::Tensor, torch::Tensor>
GaussianSplat3d::renderImagesAndDepthsFromWorld(
    const torch::Tensor &worldToCameraMatrices,
    const torch::Tensor &projectionMatrices,
    const size_t imageWidth,
    const size_t imageHeight,
    const float near,
    const float far,
    const CameraModel cameraModel,
    const ProjectionMethod projectionMethod,
    const std::optional<torch::Tensor> &distortionCoeffs,
    const int64_t shDegreeToUse,
    const size_t tileSize,
    const float minRadius2d,
    const float eps2d,
    const bool antialias,
    const std::optional<torch::Tensor> &backgrounds,
    const std::optional<torch::Tensor> &masks) {
    FVDB_FUNC_RANGE();
    const int C = worldToCameraMatrices.size(0);
    TORCH_CHECK(C > 0, "At least one camera must be provided (got 0)");

    RenderSettings settings;
    settings.imageWidth    = imageWidth;
    settings.imageHeight   = imageHeight;
    settings.nearPlane     = near;
    settings.farPlane      = far;
    settings.shDegreeToUse = shDegreeToUse;
    settings.radiusClip    = minRadius2d;
    settings.eps2d         = eps2d;
    settings.antialias     = antialias;
    settings.tileSize      = tileSize;
    settings.renderMode    = RenderSettings::RenderMode::RGBD;

    const ProjectedGaussianSplats state = projectGaussiansForCameraImpl(worldToCameraMatrices,
                                                                        projectionMatrices,
                                                                        settings,
                                                                        cameraModel,
                                                                        projectionMethod,
                                                                        distortionCoeffs);
    const torch::Tensor distortionCoeffsForRaster = distortionCoeffs.has_value()
                                                        ? distortionCoeffs.value()
                                                        : torch::empty({C, 0}, mMeans.options());

    return gsplat::rasterizeFromWorld(mMeans, mQuats, mLogScales, state, worldToCameraMatrices,
                                      projectionMatrices, distortionCoeffsForRaster, cameraModel,
                                      static_cast<uint32_t>(imageWidth),
                                      static_cast<uint32_t>(imageHeight),
                                      static_cast<uint32_t>(tileSize), backgrounds, masks);
}

std::tuple<torch::Tensor, torch::Tensor>
GaussianSplat3d::relocateGaussians(const torch::Tensor &logScales,
                                   const torch::Tensor &logitOpacities,
                                   const torch::Tensor &ratios,
                                   const torch::Tensor &binomialCoeffs,
                                   const int nMax,
                                   const float minOpacity) {
    return FVDB_DISPATCH_KERNEL(logScales.device(), [&]() {
        return detail::ops::dispatchGaussianRelocation<DeviceTag>(
            logScales, logitOpacities, ratios, binomialCoeffs, nMax, minOpacity);
    });
}

void
GaussianSplat3d::addNoiseToMeans(const float noiseScale, const float t, const float k) {
    FVDB_DISPATCH_KERNEL(mMeans.device(), [&]() {
        return detail::ops::dispatchGaussianMCMCAddNoise<DeviceTag>(
            mMeans, mLogScales, mLogitOpacities, mQuats, noiseScale, t, k);
    });
}

GaussianSplat3d
GaussianSplat3d::tensorIndexGetImpl(const torch::Tensor &indices) const {
    auto ret = GaussianSplat3d(mMeans.index({indices}),
                               mQuats.index({indices}),
                               mLogScales.index({indices}),
                               mLogitOpacities.index({indices}),
                               mSh0.index({indices}),
                               mShN.index({indices}),
                               mAccumulateMean2dGradients,
                               mAccumulateMax2dRadii,
                               false);

    if (mAccumulated2dRadiiForGrad.numel() > 0) {
        ret.mAccumulated2dRadiiForGrad = mAccumulated2dRadiiForGrad.index({indices});
    }

    if (mGradientStepCountForGrad.numel() > 0) {
        ret.mGradientStepCountForGrad = mGradientStepCountForGrad.index({indices});
    }

    if (mAccumulatedNormalized2dMeansGradientNormsForGrad.numel() > 0) {
        ret.mAccumulatedNormalized2dMeansGradientNormsForGrad =
            mAccumulatedNormalized2dMeansGradientNormsForGrad.index({indices});
    }

    return ret;
}
GaussianSplat3d
GaussianSplat3d::sliceSelect(const int64_t begin, const int64_t stop, const int64_t step) const {
    auto slice = torch::indexing::Slice(begin, stop, step);

    auto ret = GaussianSplat3d(mMeans.index({slice}),
                               mQuats.index({slice}),
                               mLogScales.index({slice}),
                               mLogitOpacities.index({slice}),
                               mSh0.index({slice}),
                               mShN.index({slice}),
                               mAccumulateMean2dGradients,
                               mAccumulateMax2dRadii,
                               false);

    if (mAccumulated2dRadiiForGrad.numel() > 0) {
        ret.mAccumulated2dRadiiForGrad = mAccumulated2dRadiiForGrad.index({slice});
    }

    if (mGradientStepCountForGrad.numel() > 0) {
        ret.mGradientStepCountForGrad = mGradientStepCountForGrad.index({slice});
    }

    if (mAccumulatedNormalized2dMeansGradientNormsForGrad.numel() > 0) {
        ret.mAccumulatedNormalized2dMeansGradientNormsForGrad =
            mAccumulatedNormalized2dMeansGradientNormsForGrad.index({slice});
    }

    return ret;
}

GaussianSplat3d
GaussianSplat3d::indexSelect(const torch::Tensor &indices) const {
    TORCH_CHECK_VALUE(indices.dim() == 1, "indices must be a 1D tensor");
    TORCH_CHECK_VALUE(indices.dtype() == torch::kInt64 || indices.dtype() == torch::kInt32,
                      "indices must be of type int64 or int32");
    TORCH_CHECK_VALUE(indices.device() == indices.device(),
                      "indices must be on the same device as the GaussianSplat3d object");

    return tensorIndexGetImpl(indices);
}

GaussianSplat3d
GaussianSplat3d::maskSelect(const torch::Tensor &mask) const {
    TORCH_CHECK_VALUE(mask.dim() == 1, "mask must be a 1D tensor");
    TORCH_CHECK_VALUE(mask.dtype() == torch::kBool, "mask must be of type bool");
    TORCH_CHECK_VALUE(mask.device() == mMeans.device(),
                      "mask must be on the same device as the GaussianSplat3d object");
    TORCH_CHECK_VALUE(mask.size(0) == mMeans.size(0),
                      "mask must have the same size as the number of gaussians");

    return tensorIndexGetImpl(mask);
}

void
GaussianSplat3d::tensorIndexSetImpl(const torch::Tensor &indices, const GaussianSplat3d &other) {
    mMeans          = mMeans.index_put({indices}, other.mMeans);
    mQuats          = mQuats.index_put({indices}, other.mQuats);
    mLogScales      = mLogScales.index_put({indices}, other.mLogScales);
    mLogitOpacities = mLogitOpacities.index_put({indices}, other.mLogitOpacities);
    mSh0            = mSh0.index_put({indices}, other.mSh0);
    mShN            = mShN.index_put({indices}, other.mShN);

    if (mAccumulated2dRadiiForGrad.numel() > 0) {
        if (other.mAccumulated2dRadiiForGrad.numel() > 0) {
            // If other is also tracking max 2d radii, make sure we copy them over
            mAccumulated2dRadiiForGrad.index_put_({indices}, other.mAccumulated2dRadiiForGrad);
        } else {
            // If the other does not have accumulated radii, we set it to zero
            mAccumulated2dRadiiForGrad.index_put_(
                {indices},
                torch::zeros(other.numGaussians(), mAccumulated2dRadiiForGrad.options()));
        }
    }

    if (mAccumulatedNormalized2dMeansGradientNormsForGrad.numel() > 0) {
        if (other.mAccumulatedNormalized2dMeansGradientNormsForGrad.numel() > 0) {
            // If other is also tracking accumulated normalized means gradient norms,
            // make sure we copy them over
            mAccumulatedNormalized2dMeansGradientNormsForGrad.index_put_(
                {indices}, other.mAccumulatedNormalized2dMeansGradientNormsForGrad);
        } else {
            // If the other does not have accumulated normalized means gradient norms, we set it to
            // zero
            mAccumulatedNormalized2dMeansGradientNormsForGrad.index_put_(
                {indices},
                torch::zeros(other.numGaussians(),
                             mAccumulatedNormalized2dMeansGradientNormsForGrad.options()));
        }
    }

    if (mGradientStepCountForGrad.numel() > 0) {
        if (other.mGradientStepCountForGrad.numel() > 0) {
            // If other is also tracking gradient step counts, make sure we copy them over
            mGradientStepCountForGrad.index_put_({indices}, other.mGradientStepCountForGrad);
        } else {
            // If the other does not have gradient step counts, we set it to zero
            mGradientStepCountForGrad.index_put_(
                {indices}, torch::zeros(other.numGaussians(), mGradientStepCountForGrad.options()));
        }
    }
}

void
GaussianSplat3d::sliceSet(const int64_t begin,
                          const int64_t end,
                          const int64_t step,
                          const GaussianSplat3d &other) {
    const auto slice = torch::indexing::Slice(begin, end, step);

    mMeans.index({slice})          = other.mMeans;
    mQuats.index({slice})          = other.mQuats;
    mLogScales.index({slice})      = other.mLogScales;
    mLogitOpacities.index({slice}) = other.mLogitOpacities;
    mSh0.index({slice})            = other.mSh0;
    mShN.index({slice})            = other.mShN;

    if (mAccumulated2dRadiiForGrad.numel() > 0) {
        if (other.mAccumulated2dRadiiForGrad.numel() > 0) {
            // If other is also tracking max 2d radii, make sure we copy them over
            mAccumulated2dRadiiForGrad.index({slice}) = other.mAccumulated2dRadiiForGrad;
        } else {
            // If the other does not have accumulated radii, we set it to zero
            mAccumulated2dRadiiForGrad.index({slice}) =
                torch::zeros(other.numGaussians(), mAccumulated2dRadiiForGrad.options());
        }
    }

    if (mAccumulatedNormalized2dMeansGradientNormsForGrad.numel() > 0) {
        if (other.mAccumulatedNormalized2dMeansGradientNormsForGrad.numel() > 0) {
            // If other is also tracking accumulated normalized means gradient norms,
            // make sure we copy them over
            mAccumulatedNormalized2dMeansGradientNormsForGrad.index({slice}) =
                other.mAccumulatedNormalized2dMeansGradientNormsForGrad;
        } else {
            // If the other does not have accumulated normalized means gradient norms, we set it to
            // zero
            mAccumulatedNormalized2dMeansGradientNormsForGrad.index({slice}) = torch::zeros(
                other.numGaussians(), mAccumulatedNormalized2dMeansGradientNormsForGrad.options());
        }
    }

    if (mGradientStepCountForGrad.numel() > 0) {
        if (other.mGradientStepCountForGrad.numel() > 0) {
            // If other is also tracking gradient step counts, make sure we copy them over
            mGradientStepCountForGrad.index({slice}) = other.mGradientStepCountForGrad;
        } else {
            // If the other does not have gradient step counts, we set it to zero
            mGradientStepCountForGrad.index({slice}) =
                torch::zeros(other.numGaussians(), mGradientStepCountForGrad.options());
        }
    }
}

void
GaussianSplat3d::indexSet(const torch::Tensor &indices, const GaussianSplat3d &other) {
    TORCH_CHECK_VALUE(indices.dim() == 1, "indices must be a 1D tensor");
    TORCH_CHECK_VALUE(indices.dtype() == torch::kInt64 || indices.dtype() == torch::kInt32,
                      "indices must be of type int64 or int32");
    TORCH_CHECK_VALUE(indices.device() == indices.device(),
                      "indices must be on the same device as the GaussianSplat3d object");

    tensorIndexSetImpl(indices, other);
}

void
GaussianSplat3d::maskSet(const torch::Tensor &mask, const GaussianSplat3d &other) {
    TORCH_CHECK_VALUE(mask.dim() == 1, "mask must be a 1D tensor");
    TORCH_CHECK_VALUE(mask.dtype() == torch::kBool, "mask must be of type bool");
    TORCH_CHECK_VALUE(mask.device() == mMeans.device(),
                      "mask must be on the same device as the GaussianSplat3d object");
    TORCH_CHECK_VALUE(mask.size(0) == mMeans.size(0),
                      "mask must have the same size as the number of gaussians");

    tensorIndexSetImpl(mask, other);
}

// TODO: Make a batched class
std::tuple<torch::Tensor, torch::Tensor, std::unordered_map<std::string, torch::Tensor>>
gaussianRenderJagged(const JaggedTensor &means,     // [N1 + N2 + ..., 3]
                     const JaggedTensor &quats,     // [N1 + N2 + ..., 4]
                     const JaggedTensor &scales,    // [N1 + N2 + ..., 3]
                     const JaggedTensor &opacities, // [N1 + N2 + ...]
                     const JaggedTensor &sh_coeffs, // [N1 + N2 + ..., K, 3]
                     const JaggedTensor &viewmats,  // [C1 + C2 + ..., 4, 4]
                     const JaggedTensor &Ks,        // [C1 + C2 + ..., 3, 3]
                     const uint32_t image_width,
                     const uint32_t image_height,
                     const float near_plane,
                     const float far_plane,
                     const int sh_degree_to_use,
                     const int tile_size,
                     const float radius_clip,
                     const float eps2d,
                     const bool antialias,
                     const bool render_depth_channel,
                     const bool return_debug_info,
                     const bool render_depth_only,
                     const bool ortho,
                     const std::optional<torch::Tensor> &backgrounds,
                     const std::optional<torch::Tensor> &masks) {
    const int ccz = viewmats.rsize(0);                           // number of cameras
    const int ggz = means.rsize(0);                              // number of gaussians
    const int D   = render_depth_only ? 1 : sh_coeffs.rsize(-1); // Dimension of output

    using namespace torch::indexing;                             // For the Slice operation

    TORCH_CHECK(means.rsizes() == torch::IntArrayRef({ggz, 3}), "means must have shape (ggz, 3)");
    TORCH_CHECK(quats.rsizes() == torch::IntArrayRef({ggz, 4}), "quats must have shape (ggz, 4)");
    TORCH_CHECK(scales.rsizes() == torch::IntArrayRef({ggz, 3}), "scales must have shape (ggz, 3)");
    TORCH_CHECK(opacities.rsizes() == torch::IntArrayRef({ggz}), "opacities must have shape (ggz)");
    TORCH_CHECK(viewmats.rsizes() == torch::IntArrayRef({ccz, 4, 4}),
                "viewmats must have shape (C, 4, 4)");
    TORCH_CHECK(Ks.rsizes() == torch::IntArrayRef({ccz, 3, 3}), "Ks must have shape (ccz, 3, 3)");

    TORCH_CHECK(means.is_contiguous(), "means must be contiguous");
    TORCH_CHECK(quats.is_contiguous(), "quats must be contiguous");
    TORCH_CHECK(scales.is_contiguous(), "scales must be contiguous");
    TORCH_CHECK(opacities.is_contiguous(), "opacities must be contiguous");
    TORCH_CHECK(viewmats.is_contiguous(), "viewmats must be contiguous");
    TORCH_CHECK(Ks.is_contiguous(), "Ks must be contiguous");

    // Check after we dispatch the unbatched version since the unbatched version accepts a
    // [K, N, D] tensor for sh_coeffs while the batched version accepts a [ggz, K, D] tensor,
    // which gets permuted later on.
    const int K = render_depth_only ? 1 : sh_coeffs.rsize(-2); // number of SH bases
    TORCH_CHECK(render_depth_only || sh_coeffs.rsizes() == torch::IntArrayRef({ggz, K, D}),
                "sh_coeffs must have shape (ggz, K, D)");

    // TODO: this part is very convoluted. But I don't have a better way of coding it without
    // customized CUDA kernels. The idea is that given Gaussians with shape [\sum(N_i), ...] and
    // cameras with shape [\sum(C_i), ...], we would calculate the intersection of each Gaussian
    // with each camera, which result in a JaggedTensor with shape
    // [\sum(C_i * N_i), ...]. And I need to keep track of the camera and Gaussian IDs (the index in
    // the jagged tensor) for each intersection:
    // - camera_ids: Shape of [\sum(C_i * N_i), ...], with each value \in [0, \sum(C_i))
    // - gaussian_ids: Shape of [\sum(C_i * N_i), ...], with each value \in [0, \sum(N_i))

    // g_sizes is [N1, N2, ...]
    torch::Tensor g_sizes =
        means.joffsets().index({Slice(1, None)}) - means.joffsets().index({Slice(0, -1)});
    // c_sizes is [C1, C2, ...]
    torch::Tensor c_sizes =
        Ks.joffsets().index({Slice(1, None)}) - Ks.joffsets().index({Slice(0, -1)});
    // camera_ids is [0, 0, ..., 1, 1, ...]
    torch::Tensor tt = g_sizes.repeat_interleave(c_sizes);
    torch::Tensor camera_ids =
        torch::arange(viewmats.rsize(0), means.options().dtype(torch::kInt32))
            .repeat_interleave(tt, 0);
    // gaussian_ids is [0, 1, ..., 0, 1, ...]
    torch::Tensor dd0    = means.joffsets().index({Slice(0, -1)}).repeat_interleave(c_sizes, 0);
    torch::Tensor dd1    = means.joffsets().index({Slice(1, None)}).repeat_interleave(c_sizes, 0);
    torch::Tensor shifts = dd0.index({Slice(1, None)}) - dd1.index({Slice(0, -1)});
    shifts               = torch::cat({torch::tensor({0}, means.device()), shifts});
    torch::Tensor shifts_cumsum = shifts.cumsum(0);
    torch::Tensor gaussian_ids =
        torch::arange(camera_ids.size(0), means.options().dtype(torch::kInt32));
    gaussian_ids += shifts_cumsum.repeat_interleave(tt, 0);

    // Project to image plane [non-differentiable at C++ level]
    auto projection_results = FVDB_DISPATCH_KERNEL_DEVICE(means.device(), [&]() {
        return detail::ops::dispatchGaussianProjectionJaggedForward<DeviceTag>(g_sizes,
                                                                               means.jdata(),
                                                                               quats.jdata(),
                                                                               scales.jdata(),
                                                                               c_sizes,
                                                                               viewmats.jdata(),
                                                                               Ks.jdata(),
                                                                               image_width,
                                                                               image_height,
                                                                               eps2d,
                                                                               near_plane,
                                                                               far_plane,
                                                                               radius_clip,
                                                                               ortho);
    });
    torch::Tensor radii     = std::get<0>(projection_results);
    torch::Tensor means2d   = std::get<1>(projection_results);
    torch::Tensor depths    = std::get<2>(projection_results);
    torch::Tensor conics    = std::get<3>(projection_results);

    // Turn [N1 + N2 + N3 + ..., ...] into [C1*N1 + C2*N2 + ..., ...]
    torch::Tensor opacities_batched = opacities.jdata().index({gaussian_ids}); // [M]
    if (antialias) {
        // Note: jagged projection does not compute compensations; use radii as a no-op
        // multiplier (compensations were never supported for the jagged path).
        (void)0;
    }

    std::unordered_map<std::string, torch::Tensor> debug_info;
    if (return_debug_info) {
        debug_info["camera_ids"]   = camera_ids;
        debug_info["gaussian_ids"] = gaussian_ids;
        debug_info["radii"]        = radii;
        debug_info["means2d"]      = means2d;
        debug_info["depths"]       = depths;
        debug_info["conics"]       = conics;
        debug_info["opacities"]    = opacities_batched;
    }

    torch::Tensor renderQuantities;
    if (render_depth_only) {
        renderQuantities = depths.index({gaussian_ids}).unsqueeze(-1); // [nnz, 1]
    } else {
        // Render quantities from SH coefficients [differentiable]
        const torch::Tensor sh_coeffs_batched = sh_coeffs.jdata().permute({1, 0, 2}).index(
            {Slice(), gaussian_ids, Slice()});                // [K, nnz, 3]

        const int K              = sh_coeffs_batched.size(0); // number of SH bases
        const int actualShDegree = sh_degree_to_use < 0 ? (std::sqrt(K) - 1) : sh_degree_to_use;
        TORCH_CHECK(K >= (actualShDegree + 1) * (actualShDegree + 1),
                    "K must be at least (shDegreeToUse + 1)^2");

        if (actualShDegree == 0) {
            const auto sh0 =
                sh_coeffs_batched.index({0, Slice(), Slice()}).unsqueeze(0); // [1, nnz, 3]
            renderQuantities = FVDB_DISPATCH_KERNEL(sh0.device(), [&]() {
                return detail::ops::dispatchSphericalHarmonicsForward<DeviceTag>(
                    actualShDegree, 1, torch::Tensor(), sh0.permute({1, 0, 2}),
                    torch::Tensor(), radii.unsqueeze(0));
            });
        } else {
            const auto sh0 =
                sh_coeffs_batched.index({0, Slice(), Slice()}).unsqueeze(0);   // [1, nnz, 3]
            const auto shN =
                sh_coeffs_batched.index({Slice(1, None), Slice(), Slice()});   // [K-1, nnz, 3]
            auto [camtoworlds, info] = torch::linalg_inv_ex(viewmats.jdata()); // [ccz, 4, 4]
            const torch::Tensor dirs = means.jdata().index({gaussian_ids, Slice()}) -
                                       camtoworlds.index({camera_ids, Slice(None, 3), 3});
            renderQuantities = FVDB_DISPATCH_KERNEL(sh0.device(), [&]() {
                return detail::ops::dispatchSphericalHarmonicsForward<DeviceTag>(
                    actualShDegree, 1, dirs.unsqueeze(0), sh0.permute({1, 0, 2}),
                    shN.permute({1, 0, 2}), radii.unsqueeze(0));
            }).squeeze(0);
        }

        if (render_depth_channel) {
            renderQuantities =
                torch::cat({renderQuantities, depths.index({gaussian_ids}).unsqueeze(-1)}, -1);
        }
    }

    // Intersect projected Gaussians with image tiles [non-differentiable]
    const int num_tiles_w = std::ceil(image_width / static_cast<float>(tile_size));
    const int num_tiles_h = std::ceil(image_height / static_cast<float>(tile_size));
    std::tuple<torch::Tensor, torch::Tensor> tile_intersections =
        FVDB_DISPATCH_KERNEL_DEVICE(means.device(), [&]() {
            return detail::ops::dispatchGaussianTileIntersection<DeviceTag>(
                means2d, radii, depths, camera_ids, ccz, tile_size, num_tiles_h, num_tiles_w);
        });
    torch::Tensor tile_offsets      = std::get<0>(tile_intersections);
    torch::Tensor tile_gaussian_ids = std::get<1>(tile_intersections);
    if (return_debug_info) {
        debug_info["tile_offsets"]      = tile_offsets;
        debug_info["tile_gaussian_ids"] = tile_gaussian_ids;
    }

    // Rasterize projected Gaussians to pixels [non-differentiable at C++ level]
    const detail::ops::RenderWindow2D renderWindow{image_width, image_height, 0, 0};
    auto outputs = FVDB_DISPATCH_KERNEL(means2d.device(), [&]() {
        return detail::ops::dispatchGaussianRasterizeForward<DeviceTag>(
            means2d,
            conics,
            renderQuantities,
            opacities_batched.contiguous(),
            renderWindow,
            tile_size,
            tile_offsets,
            tile_gaussian_ids,
            backgrounds,
            masks);
    });
    torch::Tensor renderedImages      = std::get<0>(outputs);
    torch::Tensor renderedAlphaImages = std::get<1>(outputs);

    return {renderedImages, renderedAlphaImages, debug_info};
}

} // namespace fvdb
