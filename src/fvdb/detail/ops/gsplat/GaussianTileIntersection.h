// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_GAUSSIANTILEINTERSECTION_H
#define FVDB_DETAIL_OPS_GSPLAT_GAUSSIANTILEINTERSECTION_H

#include <torch/types.h>

#include <tuple>

namespace fvdb {
namespace detail {
namespace ops {

/// @brief Compute the intersection of 2D Gaussians with image tiles for efficient rasterization
template <torch::DeviceType>
std::tuple<torch::Tensor, torch::Tensor>
dispatchGaussianTileIntersection(const torch::Tensor &means2d, // [C, N, 2] or [M, 2]
                                 const torch::Tensor &radii,   // [C, N] or [M]
                                 const torch::Tensor &depths,  // [C, N] or [M]
                                 const at::optional<torch::Tensor> &cameraIds, // NULL or [M]
                                 const uint32_t numCameras,
                                 const uint32_t tileSize,
                                 const uint32_t numTilesH,
                                 const uint32_t numTilesW);

/// @brief Compute the intersection of 2D Gaussians with image tiles for sparse rendering
template <torch::DeviceType>
std::tuple<torch::Tensor, torch::Tensor>
dispatchGaussianTileIntersectionSparse(const torch::Tensor &means2d,     // [C, N, 2] or [M, 2]
                                       const torch::Tensor &radii,       // [C, N] or [M]
                                       const torch::Tensor &depths,      // [C, N] or [M]
                                       const torch::Tensor &tileMask,    // [C, H, W]
                                       const torch::Tensor &activeTiles, // [num_active_tiles]
                                       const at::optional<torch::Tensor> &cameraIds, // NULL or [M]
                                       const uint32_t numCameras,
                                       const uint32_t tileSize,
                                       const uint32_t numTilesH,
                                       const uint32_t numTilesW);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_GSPLAT_GAUSSIANTILEINTERSECTION_H
