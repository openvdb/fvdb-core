// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_GAUSSIANPROJECTIONPIPELINE_H
#define FVDB_DETAIL_OPS_GSPLAT_GAUSSIANPROJECTIONPIPELINE_H

#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/ops/gsplat/GaussianCameras.cuh>
#include <fvdb/detail/ops/gsplat/GaussianProjectionTypes.h>
#include <fvdb/detail/ops/gsplat/GaussianRenderSettings.h>

#include <torch/types.h>

#include <optional>
#include <tuple>

namespace fvdb::detail::ops {

/// @brief Project 3D Gaussians to 2D using analytic projection, evaluate SH, and intersect tiles.
///
/// Transforms 3D Gaussians to 2D screen space via analytic camera projection, evaluates
/// spherical harmonics for view-dependent appearance, computes per-Gaussian opacities, and
/// performs tile intersection for efficient rasterization. Returns all results in a
/// @c ProjectedGaussianSplats struct.
///
/// The origin of 2D pixel coordinates is the top-left corner of the image, with +x right
/// and +y downwards. Gaussians that are culled (behind near/far planes, projected too small)
/// have their radii set to zero.
///
/// @param[in] means                     3D Gaussian positions [N, 3]
/// @param[in] quats                     Quaternion rotations [N, 4] (x, y, z, w)
/// @param[in] logScales                 Log-scale factors [N, 3]
/// @param[in] logitOpacities            Logit-space opacities [N]
/// @param[in] sh0                       Degree-0 SH coefficients [N, 1, D]
/// @param[in] shN                       Higher-degree SH coefficients [N, K-1, D]
/// @param[in] worldToCameraMatrices     Camera extrinsics [C, 4, 4]
/// @param[in] projectionMatrices        Camera intrinsics [C, 3, 3]
/// @param[in] settings                  Render settings (image size, tile size, near/far, etc.)
/// @param[in] cameraModel               Camera/distortion model
/// @param[in] accumulateMean2dGradients Whether to accumulate mean-2D gradient norms
/// @param[in] accumulateMax2dRadii      Whether to accumulate max 2D radii
/// @param[in,out] accumGradNorms        Gradient norm accumulator [N] (lazily initialized)
/// @param[in,out] accumStepCounts       Step count accumulator [N] (lazily initialized)
/// @param[in,out] accumMax2dRadii       Max 2D radii accumulator [N] (lazily initialized)
///
/// @note **Accumulator mutability:** The accumulator tensors are passed by mutable reference
///       because the CUDA projection backward kernel writes into them in-place via atomicAdd.
///       The forward pass itself is pure.
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

/// @brief Project 3D Gaussians to 2D using unscented transform, evaluate SH, and intersect tiles.
///
/// Same pipeline as @c projectGaussiansAnalytic but uses the Unscented Transform (UT) for
/// projection, which supports non-linear distortion models (e.g., OpenCV radial-tangential).
/// Non-differentiable — no gradient accumulation support.
///
/// @param[in] means                 3D Gaussian positions [N, 3]
/// @param[in] quats                 Quaternion rotations [N, 4]
/// @param[in] logScales             Log-scale factors [N, 3]
/// @param[in] logitOpacities        Logit-space opacities [N]
/// @param[in] sh0                   Degree-0 SH coefficients [N, 1, D]
/// @param[in] shN                   Higher-degree SH coefficients [N, K-1, D]
/// @param[in] worldToCameraMatrices Camera extrinsics [C, 4, 4]
/// @param[in] projectionMatrices    Camera intrinsics [C, 3, 3]
/// @param[in] distortionCoeffs      Distortion coefficients [C, 12] for OpenCV, or [C, 0]
/// @param[in] settings              Render settings
/// @param[in] cameraModel           Camera/distortion model
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
///
/// Validates camera arguments, resolves @c ProjectionMethod::AUTO to the appropriate concrete
/// method, and delegates to @c projectGaussiansAnalytic or @c projectGaussiansUT.
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
///
/// Sparse variant of @c projectGaussiansAnalytic. Deduplicates pixel coordinates,
/// computes sparse tile info, runs analytic projection, evaluates SH, and performs
/// sparse tile intersection. Only pixels in @p pixelsToRender are considered.
///
/// @param[in] pixelsToRender Pixel coordinates to render [total_pixels, 2] (JaggedTensor,
///            one list per camera; must have C outer lists matching number of cameras)
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
///
/// Sparse variant of @c projectGaussiansUT. Non-differentiable.
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

/// @brief Project Gaussians for a given camera model for sparse rendering.
///
/// Validates camera args, resolves projection method, and delegates to
/// @c sparseProjectGaussiansAnalytic or @c sparseProjectGaussiansUT.
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
// Sparse render: project + rasterize + duplicate scatter-back
// ---------------------------------------------------------------------------

/// @brief End-to-end sparse render: project, rasterize, and scatter-back.
///
/// Projects Gaussians (via @c sparseProjectGaussiansForCamera), rasterizes only the
/// specified pixels using the sparse rasterization kernel, and scatters results back to
/// duplicate pixel positions if any were removed during deduplication. This is the
/// sparse-rendering equivalent of project + @c renderCropFromProjected.
///
/// @param[in] pixelsToRender  Pixel coordinates to render (JaggedTensor, one list per camera)
/// @param[in] backgrounds     Optional per-camera background color [C, D]
/// @param[in] masks           Optional per-tile boolean mask [C, tile_height, tile_width]
///
/// @return std::tuple containing:
///         - Rendered colors per pixel (JaggedTensor, same structure as @p pixelsToRender)
///         - Alpha values per pixel (JaggedTensor, same structure)
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

} // namespace fvdb::detail::ops

#endif // FVDB_DETAIL_OPS_GSPLAT_GAUSSIANPROJECTIONPIPELINE_H
