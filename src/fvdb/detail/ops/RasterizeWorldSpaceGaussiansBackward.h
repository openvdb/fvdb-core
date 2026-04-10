// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_RASTERIZEWORLDSPACEGAUSSIANSBACKWARD_H
#define FVDB_DETAIL_OPS_RASTERIZEWORLDSPACEGAUSSIANSBACKWARD_H

#include <fvdb/detail/utils/gaussian/GaussianCameras.cuh>

#include <torch/types.h>

#include <cstdint>
#include <optional>
#include <tuple>

namespace fvdb::detail::ops {

/// @brief Backward pass for dense rasterization from 3D Gaussians using per-pixel rays.
///
/// Gradients are produced for:
/// - means:     [N, 3]
/// - quats:     [N, 4]
/// - logScales: [N, 3]
/// - features:  [C, N, D]
/// - opacities: [C, N]
///
/// @param[in] means Gaussian mean positions [N, 3]
/// @param[in] quats Gaussian quaternion rotations [N, 4]
/// @param[in] logScales Gaussian log-scale factors [N, 3]
/// @param[in] features Feature/color values [C, N, D]
/// @param[in] opacities Opacity values [C, N]
/// @param[in] worldToCamMatricesStart World-to-camera matrices (start) [C, 4, 4]
/// @param[in] worldToCamMatricesEnd World-to-camera matrices (end) [C, 4, 4]
/// @param[in] projectionMatrices Camera intrinsics [C, 3, 3]
/// @param[in] distortionCoeffs Distortion coefficients [C, K]
/// @param[in] rollingShutterType Rolling shutter policy
/// @param[in] cameraModel Camera/distortion model
/// @param[in] settings Render settings (image dimensions, tile size, etc.)
/// @param[in] tileOffsets Tile offsets [C, tileH, tileW]
/// @param[in] tileGaussianIds Tile Gaussian IDs [n_isects]
/// @param[in] renderedAlphas Alpha values from forward pass [C, H, W, 1]
/// @param[in] lastIds Last Gaussian ID per pixel [C, H, W]
/// @param[in] dLossDRenderedFeatures Gradients w.r.t. rendered features [C, H, W, D]
/// @param[in] dLossDRenderedAlphas Gradients w.r.t. rendered alphas [C, H, W, 1]
/// @param[in] backgrounds Optional per-camera background [C, D]
/// @param[in] masks Optional per-tile boolean mask [C, tileH, tileW]
///
/// @return std::tuple containing gradients:
///         - dL/dmeans [N, 3]
///         - dL/dquats [N, 4]
///         - dL/dlogScales [N, 3]
///         - dL/dfeatures [C, N, D]
///         - dL/dopacities [C, N]
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
rasterize_world_space_gaussians_bwd(const torch::Tensor &means,
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
                                    uint32_t imageWidth,
                                    uint32_t imageHeight,
                                    uint32_t imageOriginW,
                                    uint32_t imageOriginH,
                                    uint32_t tileSize,
                                    const torch::Tensor &tileOffsets,
                                    const torch::Tensor &tileGaussianIds,
                                    const torch::Tensor &renderedAlphas,
                                    const torch::Tensor &lastIds,
                                    const torch::Tensor &dLossDRenderedFeatures,
                                    const torch::Tensor &dLossDRenderedAlphas,
                                    const at::optional<torch::Tensor> &backgrounds = at::nullopt,
                                    const at::optional<torch::Tensor> &masks       = at::nullopt);

} // namespace fvdb::detail::ops

#endif // FVDB_DETAIL_OPS_RASTERIZEWORLDSPACEGAUSSIANSBACKWARD_H
