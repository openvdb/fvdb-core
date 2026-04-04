// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_GAUSSIANSPLATOPS_H
#define FVDB_DETAIL_OPS_GSPLAT_GAUSSIANSPLATOPS_H

#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/ops/gsplat/GaussianCameras.cuh>
#include <fvdb/detail/ops/gsplat/GaussianProjectionTypes.h>
#include <fvdb/detail/ops/gsplat/GaussianRenderSettings.h>

#include <ATen/core/TensorBody.h>

#include <optional>
#include <tuple>

namespace fvdb::detail::ops {

/// @brief Check whether a camera model uses OpenCV-style distortion.
inline bool
usesOpenCVDistortion(const DistortionModel cameraModel) {
    return cameraModel == DistortionModel::OPENCV_RADTAN_5 ||
           cameraModel == DistortionModel::OPENCV_RATIONAL_8 ||
           cameraModel == DistortionModel::OPENCV_RADTAN_THIN_PRISM_9 ||
           cameraModel == DistortionModel::OPENCV_THIN_PRISM_12;
}

/// @brief Resolve the projection method for a given camera model.
///        AUTO is resolved to UNSCENTED for OpenCV models, and ANALYTIC otherwise.
inline ProjectionMethod
resolveProjectionMethod(const DistortionModel cameraModel,
                        const ProjectionMethod projectionMethod) {
    if (projectionMethod == ProjectionMethod::AUTO) {
        return usesOpenCVDistortion(cameraModel) ? ProjectionMethod::UNSCENTED
                                                 : ProjectionMethod::ANALYTIC;
    }
    return projectionMethod;
}

/// @brief Validate the camera projection arguments (matrix shapes, distortion coefficients, etc.).
void validateCameraProjectionArgs(const torch::Tensor &worldToCameraMatrices,
                                  const torch::Tensor &projectionMatrices,
                                  DistortionModel cameraModel,
                                  ProjectionMethod requestedProjectionMethod,
                                  const std::optional<torch::Tensor> &distortionCoeffs);

/// @brief Validate tensor shapes, devices, and types for Gaussian splat state.
void checkGaussianState(const torch::Tensor &means,
                        const torch::Tensor &quats,
                        const torch::Tensor &logScales,
                        const torch::Tensor &logitOpacities,
                        const torch::Tensor &sh0,
                        const torch::Tensor &shN);

/// @brief Deduplicate pixel coordinates in a JaggedTensor.
///
/// Encodes each pixel as a single int64 key incorporating its batch index and 2D coordinate,
/// sorts keys to find unique groups and builds an inverse mapping. Returns the deduplicated
/// pixels as a new JaggedTensor, the inverse index tensor, and a flag indicating whether any
/// duplicates were found.
///
/// @param pixelsToRender The input JaggedTensor of pixel coordinates [total_pixels, 2]
/// @param imageWidth Width of each image in pixels
/// @param imageHeight Height of each image in pixels
/// @return Tuple of (uniquePixels JaggedTensor, inverseIndices tensor, hasDuplicates bool)
std::tuple<fvdb::JaggedTensor, torch::Tensor, bool> deduplicatePixels(
    const fvdb::JaggedTensor &pixelsToRender, int64_t imageWidth, int64_t imageHeight);

/// @brief Evaluate spherical harmonics for Gaussian splats.
/// @param means [N, 3] Gaussian means
/// @param sh0 [N, 1, D] Degree-0 SH coefficients
/// @param shN [N, K-1, D] Higher-degree SH coefficients
/// @param shDegreeToUse SH degree to use (-1 for all)
/// @param worldToCameraMatrices [C, 4, 4] Camera matrices
/// @param perGaussianProjectedRadii [C, N] Projected radii
/// @return [C, N, D] Evaluated SH features
torch::Tensor evalSphericalHarmonics(const torch::Tensor &means,
                                     const torch::Tensor &sh0,
                                     const torch::Tensor &shN,
                                     int64_t shDegreeToUse,
                                     const torch::Tensor &worldToCameraMatrices,
                                     const torch::Tensor &perGaussianProjectedRadii);

/// @brief Project Gaussians using analytic projection, evaluate SH, and compute tile intersection.
/// @param means [N, 3]
/// @param quats [N, 4]
/// @param logScales [N, 3]
/// @param logitOpacities [N]
/// @param sh0 [N, 1, D]
/// @param shN [N, K-1, D]
/// @param worldToCameraMatrices [C, 4, 4]
/// @param projectionMatrices [C, 3, 3]
/// @param settings Render settings
/// @param cameraModel Camera/distortion model
/// @param accumulateMean2dGradients Whether to accumulate mean 2D gradient norms
/// @param accumulateMax2dRadii Whether to accumulate max 2D radii
/// @param accumGradNorms [in/out] Optional accumulator for gradient norms (lazily initialized)
/// @param accumStepCounts [in/out] Optional accumulator for step counts (lazily initialized)
/// @param accumMax2dRadii [in/out] Optional accumulator for max 2D radii (lazily initialized)
/// @return ProjectedGaussianSplats with all projection results
///
/// @note **Accumulator mutability:** The accumulator tensors (accumGradNorms, accumStepCounts,
///       accumMax2dRadii) are passed by mutable reference because the CUDA projection backward
///       kernel writes into them in-place via atomicAdd during backpropagation. This in-place
///       mutation is used by Gaussian densification strategies (split/clone/prune) to track
///       per-Gaussian gradient statistics across training steps. The forward pass itself is pure.
///       The pybind11 layer (GaussianSplatOps.cpp) hides this mutability behind lambda wrappers
///       that present a strictly functional interface to Python. When autograd moves to Python
///       (Milestones 5-7), the Python GaussianSplat3d class will own these accumulators and
///       pass them through to the C++ backward dispatch.
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
                         DistortionModel cameraModel,
                         bool accumulateMean2dGradients,
                         bool accumulateMax2dRadii,
                         std::optional<torch::Tensor> &accumGradNorms,
                         std::optional<torch::Tensor> &accumStepCounts,
                         std::optional<torch::Tensor> &accumMax2dRadii);

/// @brief Project Gaussians using unscented transform, evaluate SH, and compute tile intersection.
///        Non-differentiable (no gradient accumulation support).
fvdb::ProjectedGaussianSplats projectGaussiansUT(const torch::Tensor &means,
                                                 const torch::Tensor &quats,
                                                 const torch::Tensor &logScales,
                                                 const torch::Tensor &logitOpacities,
                                                 const torch::Tensor &sh0,
                                                 const torch::Tensor &shN,
                                                 const torch::Tensor &worldToCameraMatrices,
                                                 const torch::Tensor &projectionMatrices,
                                                 const torch::Tensor &distortionCoeffs,
                                                 const RenderSettings &settings,
                                                 DistortionModel cameraModel);

/// @brief Project Gaussians for a given camera model, dispatching between analytic and UT paths.
///        Validates camera args and resolves projection method.
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
                          DistortionModel cameraModel,
                          ProjectionMethod projectionMethod,
                          const std::optional<torch::Tensor> &distortionCoeffs,
                          bool accumulateMean2dGradients,
                          bool accumulateMax2dRadii,
                          std::optional<torch::Tensor> &accumGradNorms,
                          std::optional<torch::Tensor> &accumStepCounts,
                          std::optional<torch::Tensor> &accumMax2dRadii);

// ---------------------------------------------------------------------------
// Sparse projection functions
// ---------------------------------------------------------------------------

/// @brief Project Gaussians using analytic projection for sparse rendering.
///        Deduplicates pixels, computes sparse tile info, runs analytic projection,
///        evaluates SH, and performs sparse tile intersection.
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
                               DistortionModel cameraModel,
                               bool accumulateMean2dGradients,
                               bool accumulateMax2dRadii,
                               std::optional<torch::Tensor> &accumGradNorms,
                               std::optional<torch::Tensor> &accumStepCounts,
                               std::optional<torch::Tensor> &accumMax2dRadii);

/// @brief Project Gaussians using unscented transform for sparse rendering.
///        Non-differentiable (no gradient accumulation support).
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
                         DistortionModel cameraModel);

/// @brief Project Gaussians for a given camera model for sparse rendering,
///        dispatching between analytic and UT paths.
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
                                DistortionModel cameraModel,
                                ProjectionMethod projectionMethod,
                                const std::optional<torch::Tensor> &distortionCoeffs,
                                bool accumulateMean2dGradients,
                                bool accumulateMax2dRadii,
                                std::optional<torch::Tensor> &accumGradNorms,
                                std::optional<torch::Tensor> &accumStepCounts,
                                std::optional<torch::Tensor> &accumMax2dRadii);

// ---------------------------------------------------------------------------
// Rasterization functions
// ---------------------------------------------------------------------------

/// @brief Rasterize a cropped region from pre-projected Gaussians.
///        Validates crop dimensions and calls dispatchGaussianRasterizeForward.
std::tuple<torch::Tensor, torch::Tensor>
renderCropFromProjected(const fvdb::ProjectedGaussianSplats &projectedGaussians,
                        size_t tileSize,
                        ssize_t cropWidth,
                        ssize_t cropHeight,
                        ssize_t cropOriginW,
                        ssize_t cropOriginH,
                        const std::optional<torch::Tensor> &backgrounds,
                        const std::optional<torch::Tensor> &masks);

/// @brief Sparse render: projects Gaussians, rasterizes at specified pixels,
///        and scatters results back if there were duplicates.
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
             DistortionModel cameraModel,
             ProjectionMethod projectionMethod,
             const std::optional<torch::Tensor> &distortionCoeffs,
             const std::optional<torch::Tensor> &backgrounds,
             const std::optional<torch::Tensor> &masks,
             bool accumulateMean2dGradients,
             bool accumulateMax2dRadii,
             std::optional<torch::Tensor> &accumGradNorms,
             std::optional<torch::Tensor> &accumStepCounts,
             std::optional<torch::Tensor> &accumMax2dRadii);

/// @brief Rasterize from world-space 3D Gaussians using a pre-projected state.
///        Calls dispatchGaussianRasterizeFromWorld3DGSForward.
std::tuple<torch::Tensor, torch::Tensor>
rasterizeFromWorld(const torch::Tensor &means,
                   const torch::Tensor &quats,
                   const torch::Tensor &logScales,
                   const fvdb::ProjectedGaussianSplats &projectedState,
                   const torch::Tensor &worldToCameraMatrices,
                   const torch::Tensor &projectionMatrices,
                   const torch::Tensor &distortionCoeffs,
                   DistortionModel cameraModel,
                   uint32_t imageWidth,
                   uint32_t imageHeight,
                   uint32_t tileSize,
                   const std::optional<torch::Tensor> &backgrounds,
                   const std::optional<torch::Tensor> &masks);

// ---------------------------------------------------------------------------
// Query operations on projected states
// ---------------------------------------------------------------------------

/// @brief Render the number of contributing Gaussians per pixel (dense).
std::tuple<torch::Tensor, torch::Tensor>
renderNumContributing(const fvdb::ProjectedGaussianSplats &state, const RenderSettings &settings);

/// @brief Render the number of contributing Gaussians per pixel (sparse).
std::tuple<fvdb::JaggedTensor, fvdb::JaggedTensor>
sparseRenderNumContributing(const fvdb::SparseProjectedGaussianSplats &state,
                            const fvdb::JaggedTensor &pixelsToRender,
                            const RenderSettings &settings);

/// @brief Render the IDs of contributing Gaussians per pixel (dense).
std::tuple<fvdb::JaggedTensor, fvdb::JaggedTensor>
renderContributingIds(const fvdb::ProjectedGaussianSplats &state,
                      const RenderSettings &settings,
                      const std::optional<torch::Tensor> &maybeNumContributingGaussians);

/// @brief Render the IDs of contributing Gaussians per pixel (sparse).
std::tuple<fvdb::JaggedTensor, fvdb::JaggedTensor>
sparseRenderContributingIds(const fvdb::SparseProjectedGaussianSplats &state,
                            const fvdb::JaggedTensor &pixelsToRender,
                            const RenderSettings &settings,
                            const std::optional<fvdb::JaggedTensor> &maybeNumContributingGaussians);

} // namespace fvdb::detail::ops

#endif // FVDB_DETAIL_OPS_GSPLAT_GAUSSIANSPLATOPS_H
