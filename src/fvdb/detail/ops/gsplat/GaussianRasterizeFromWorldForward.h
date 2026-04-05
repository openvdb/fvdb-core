// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_GAUSSIANRASTERIZEFROMWORLDFORWARD_H
#define FVDB_DETAIL_OPS_GSPLAT_GAUSSIANRASTERIZEFROMWORLDFORWARD_H

#include <fvdb/detail/ops/gsplat/GaussianCameras.cuh>
#include <fvdb/detail/ops/gsplat/GaussianProjectionTypes.h>
#include <fvdb/detail/ops/gsplat/GaussianRenderSettings.h>

#include <torch/types.h>

#include <optional>
#include <tuple>

namespace fvdb::detail::ops {

/// @brief Rasterize images directly from 3D Gaussians using per-pixel rays.
///
/// This kernel follows the gsplat "RasterizeToPixelsFromWorld3DGS" algorithm, but is wired to
/// FVDB's existing tile intersection representation (`tileOffsets`, `tileGaussianIds`).
///
/// Inputs are world-space Gaussians (means/quats/logScales) and per-camera per-gaussian features
/// and opacities. The camera is defined via world->camera matrices (start/end), intrinsics,
/// `DistortionModel`, rolling shutter policy, and packed OpenCV distortion coefficients.
///
/// This is a dense-only rasterizer: outputs are dense tensors of shape
/// - renderedFeatures: [C, H, W, D]
/// - renderedAlphas:   [C, H, W, 1]
/// - lastIds:          [C, H, W]
///
/// @param[in] means Gaussian mean positions [N, 3]
/// @param[in] quats Gaussian quaternion rotations [N, 4] (w,x,y,z)
/// @param[in] logScales Gaussian log-scale factors [N, 3]
/// @param[in] features Feature/color values [C, N, D]
/// @param[in] opacities Opacity values [C, N]
/// @param[in] worldToCamMatricesStart World-to-camera matrices (start) [C, 4, 4]
/// @param[in] worldToCamMatricesEnd World-to-camera matrices (end) [C, 4, 4]
/// @param[in] projectionMatrices Camera intrinsics [C, 3, 3]
/// @param[in] distortionCoeffs Distortion coefficients [C, K] (K=0 or 12)
/// @param[in] rollingShutterType Rolling shutter policy
/// @param[in] cameraModel Camera/distortion model
/// @param[in] settings Render settings (image dimensions, tile size, etc.)
/// @param[in] tileOffsets Tile offsets [C, tileH, tileW]
/// @param[in] tileGaussianIds Tile Gaussian IDs [n_isects]
/// @param[in] backgrounds Optional per-camera background [C, D]
/// @param[in] masks Optional per-tile boolean mask [C, tileH, tileW]
///
/// @return std::tuple containing:
///         - Rendered features [C, H, W, D]
///         - Alpha values [C, H, W, 1]
///         - Last Gaussian ID per pixel [C, H, W]
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
gaussianRasterizeFromWorldForward(const torch::Tensor &means,
                                  const torch::Tensor &quats,
                                  const torch::Tensor &logScales,
                                  const torch::Tensor &features,
                                  const torch::Tensor &opacities,
                                  const torch::Tensor &worldToCamMatricesStart,
                                  const torch::Tensor &worldToCamMatricesEnd,
                                  const torch::Tensor &projectionMatrices,
                                  const torch::Tensor &distortionCoeffs,
                                  RollingShutterType rollingShutterType,
                                  DistortionModel cameraModel,
                                  const RenderSettings &settings,
                                  const torch::Tensor &tileOffsets,
                                  const torch::Tensor &tileGaussianIds,
                                  const at::optional<torch::Tensor> &backgrounds = at::nullopt,
                                  const at::optional<torch::Tensor> &masks       = at::nullopt);

/// @brief Rasterize images directly from 3D world-space Gaussians using per-pixel rays.
///
/// Uses the gsplat "RasterizeToPixelsFromWorld3DGS" algorithm with FVDB's tile intersection
/// representation. World-space Gaussians (means/quats/logScales) are projected per-pixel
/// during rasterization, using the camera matrices and distortion model. Features and
/// opacities come from the pre-projected state.
///
/// This is a dense-only rasterizer (CUDA only, no CPU implementation).
///
/// @param[in] means                  Gaussian mean positions [N, 3]
/// @param[in] quats                  Gaussian quaternion rotations [N, 4] (w,x,y,z)
/// @param[in] logScales              Gaussian log-scale factors [N, 3]
/// @param[in] projectedState         Pre-projected state containing per-Gaussian features
///                                   [C, N, D], opacities [C, N], tile offsets
///                                   [C, tileH, tileW], and tile Gaussian IDs [n_isects]
/// @param[in] worldToCameraMatrices  Camera extrinsics [C, 4, 4]
/// @param[in] projectionMatrices     Camera intrinsics [C, 3, 3]
/// @param[in] distortionCoeffs       Distortion coefficients [C, K] (K=0 or 12)
/// @param[in] cameraModel            Camera/distortion model
/// @param[in] imageWidth             Output image width in pixels
/// @param[in] imageHeight            Output image height in pixels
/// @param[in] tileSize               Tile size for rasterization
/// @param[in] backgrounds            Optional per-camera background [C, D] (black if absent)
/// @param[in] masks                  Optional per-tile boolean mask [C, tileH, tileW]
///
/// @return std::tuple containing:
///         - Rendered features [C, H, W, D]
///         - Alpha values [C, H, W, 1]
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

} // namespace fvdb::detail::ops

#endif // FVDB_DETAIL_OPS_GSPLAT_GAUSSIANRASTERIZEFROMWORLDFORWARD_H
