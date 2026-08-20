// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_INTERSECTGAUSSIANTILES_H
#define FVDB_DETAIL_OPS_GSPLAT_INTERSECTGAUSSIANTILES_H

#include <torch/types.h>

#include <tuple>

namespace fvdb {
namespace detail {
namespace ops {

/// @brief Compute the intersection of 2D Gaussians with image tiles for efficient rasterization
/// @return tile offsets and Gaussian IDs
std::tuple<torch::Tensor, torch::Tensor> intersectGaussianTiles(
    const torch::Tensor &means2d,                 // [C, N, 2] or [M, 2]
    const torch::Tensor &radii,                   // [C, N, 2] or [M, 2]
    const torch::Tensor &depths,                  // [C, N] or [M]
    const at::optional<torch::Tensor> &cameraIds, // NULL or [M]
    const uint32_t numCameras,
    const uint32_t tileSize,
    const uint32_t numTilesH,
    const uint32_t numTilesW,
    const at::optional<torch::Tensor> &conics    = at::nullopt,  // [C, N, 3] or [M, 3]
    const at::optional<torch::Tensor> &opacities = at::nullopt); // [C, N] or [M]

/// @brief Compute the intersection of 2D Gaussians with image tiles for sparse rendering
/// @return tile offsets and Gaussian IDs
std::tuple<torch::Tensor, torch::Tensor> intersectGaussianTilesSparse(
    const torch::Tensor &means2d,                 // [C, N, 2] or [M, 2]
    const torch::Tensor &radii,                   // [C, N, 2] or [M, 2]
    const torch::Tensor &depths,                  // [C, N] or [M]
    const torch::Tensor &tileMask,                // [C, H, W]
    const torch::Tensor &activeTiles,             // [num_active_tiles]
    const at::optional<torch::Tensor> &cameraIds, // NULL or [M]
    const uint32_t numCameras,
    const uint32_t tileSize,
    const uint32_t numTilesH,
    const uint32_t numTilesW,
    const at::optional<torch::Tensor> &conics    = at::nullopt,  // [C, N, 3] or [M, 3]
    const at::optional<torch::Tensor> &opacities = at::nullopt); // [C, N] or [M]

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_GSPLAT_INTERSECTGAUSSIANTILES_H
