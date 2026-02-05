// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_GAUSSIANRASTERIZEFROMWORLD3DGSFORWARD_H
#define FVDB_DETAIL_OPS_GSPLAT_GAUSSIANRASTERIZEFROMWORLD3DGSFORWARD_H

#include <fvdb/detail/ops/gsplat/GaussianCameraModels.h>

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
/// `CameraModel`, rolling shutter policy, and packed OpenCV distortion coefficients.
///
/// This is a dense-only rasterizer: outputs are dense tensors of shape
/// - renderedFeatures: [C, H, W, D]
/// - renderedAlphas:   [C, H, W, 1]
/// - lastIds:          [C, H, W]
///
/// @tparam DeviceType torch::kCUDA (CPU not implemented).
template <torch::DeviceType>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> dispatchGaussianRasterizeFromWorld3DGSForward(
    // Gaussian parameters (world space)
    const torch::Tensor &means,     // [N, 3]
    const torch::Tensor &quats,     // [N, 4] (w,x,y,z)
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
    // Optional background
    const at::optional<torch::Tensor> &backgrounds = at::nullopt // [C, D]
);

} // namespace fvdb::detail::ops

#endif // FVDB_DETAIL_OPS_GSPLAT_GAUSSIANRASTERIZEFROMWORLD3DGSFORWARD_H
