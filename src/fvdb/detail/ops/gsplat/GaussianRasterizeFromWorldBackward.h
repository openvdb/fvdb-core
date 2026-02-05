// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_GAUSSIANRASTERIZEFROMWORLDBACKWARD_H
#define FVDB_DETAIL_OPS_GSPLAT_GAUSSIANRASTERIZEFROMWORLDBACKWARD_H

#include <fvdb/detail/ops/gsplat/GaussianCameraModels.h>

#include <torch/types.h>

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
/// @tparam DeviceType torch::kCUDA (CPU not implemented).
template <torch::DeviceType>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianRasterizeFromWorld3DGSBackward(
    // Gaussian parameters (world space)
    const torch::Tensor &means,     // [N, 3]
    const torch::Tensor &quats,     // [N, 4]
    const torch::Tensor &logScales, // [N, 3]
    // Per-camera quantities
    const torch::Tensor &features,  // [C, N, D]
    const torch::Tensor &opacities, // [C, N]
    const torch::Tensor &worldToCamMatricesStart, // [C, 4, 4]
    const torch::Tensor &worldToCamMatricesEnd,   // [C, 4, 4]
    const torch::Tensor &projectionMatrices,      // [C, 3, 3]
    const torch::Tensor &distortionCoeffs,        // [C, K] (K=0 or 12)
    const RollingShutterType rollingShutterType,
    const CameraModel cameraModel,
    // Render settings
    const uint32_t imageWidth,
    const uint32_t imageHeight,
    const uint32_t imageOriginW,
    const uint32_t imageOriginH,
    const uint32_t tileSize,
    // Intersections
    const torch::Tensor &tileOffsets,     // [C, tileH, tileW]
    const torch::Tensor &tileGaussianIds, // [n_isects] values in [0, C*N)
    // Forward outputs needed for backward
    const torch::Tensor &renderedAlphas, // [C, H, W, 1]
    const torch::Tensor &lastIds,        // [C, H, W]
    // Gradients of outputs
    const torch::Tensor &dLossDRenderedFeatures, // [C, H, W, D]
    const torch::Tensor &dLossDRenderedAlphas,   // [C, H, W, 1]
    // Optional background (only affects alpha gradient term)
    const at::optional<torch::Tensor> &backgrounds = at::nullopt, // [C, D]
    // Optional tile masks (parity with classic rasterizer)
    const at::optional<torch::Tensor> &masks = at::nullopt // [C, tileH, tileW] bool
);

} // namespace fvdb::detail::ops

#endif // FVDB_DETAIL_OPS_GSPLAT_GAUSSIANRASTERIZEFROMWORLDBACKWARD_H
