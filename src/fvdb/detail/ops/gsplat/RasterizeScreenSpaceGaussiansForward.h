// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_RASTERIZESCREENSPACEGAUSSIANSFORWARD_H
#define FVDB_DETAIL_OPS_GSPLAT_RASTERIZESCREENSPACEGAUSSIANSFORWARD_H

#include <fvdb/JaggedTensor.h>

#include <nanovdb/math/Math.h>

#include <torch/types.h>

#include <optional>
#include <tuple>

namespace fvdb {
namespace detail {
namespace ops {

/// @brief Perform Gaussian rasterization to render an image (forward pass)
///
/// This function rasterizes 2D Gaussians into an image using a tile-based approach for efficiency.
/// Each Gaussian is represented by its 2D projected center, covariance matrix in conic form,
/// feature/color, and opacity. The function performs alpha-blending of the Gaussians to generate
/// the final rendered image.
///
/// @param[in] means2d 2D projected Gaussian centers [C, N, 2]
/// @param[in] conics Gaussian covariance matrices in conic form [C, N, 3] representing (a, b, c) in
/// ax² + 2bxy + cy²
/// @param[in] features Feature / color values of Gaussians [C, N, D]
/// @param[in] opacities Opacity values for each Gaussian [N]
/// @param[in] imageWidth Width of the render window in pixels
/// @param[in] imageHeight Height of the render window in pixels
/// @param[in] imageOriginW Horizontal origin of the render window
/// @param[in] imageOriginH Vertical origin of the render window
/// @param[in] tileSize Size of tiles used for rasterization optimization
/// @param[in] tileOffsets Offsets for tiles [C, tile_height, tile_width] indicating for each tile
/// where its Gaussians start
/// @param[in] tileGaussianIds Flattened Gaussian IDs for tile intersection [n_isects] indicating
/// which Gaussians affect each tile
/// @param[in] backgrounds Optional background color per camera [C, D]. If provided, background
/// colors will be blended with transparent pixels. If not provided, background is assumed to be
/// black.
/// @param[in] masks Optional per-tile boolean mask [C, tile_height, tile_width]
///
/// @return std::tuple containing:
///         - Rendered image features/colors [C, render_height, render_width, D]
///         - Alpha values [C, render_height, render_width, 1]
///         - Last Gaussian ID rendered at each pixel [C, render_height, render_width]
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
rasterizeScreenSpaceGaussiansFwd(const torch::Tensor &means2d,
                                 const torch::Tensor &conics,
                                 const torch::Tensor &features,
                                 const torch::Tensor &opacities,
                                 uint32_t imageWidth,
                                 uint32_t imageHeight,
                                 uint32_t imageOriginW,
                                 uint32_t imageOriginH,
                                 uint32_t tileSize,
                                 const torch::Tensor &tileOffsets,
                                 const torch::Tensor &tileGaussianIds,
                                 const at::optional<torch::Tensor> &backgrounds = at::nullopt,
                                 const at::optional<torch::Tensor> &masks       = at::nullopt);

/// @brief Sparse Gaussian rasterization forward pass. Renders only specified pixels.
///
/// @param pixelsToRender Tensor containing the indices of pixels to render [C, NumPixels, 2].
/// @param means2d Tensor of 2D means.
/// @param conics Tensor of conic parameters.
/// @param features Tensor of features (colors, etc).
/// @param opacities Tensor of opacities.
/// @param imageWidth Width of the render window in pixels
/// @param imageHeight Height of the render window in pixels
/// @param imageOriginW Horizontal origin of the render window
/// @param imageOriginH Vertical origin of the render window
/// @param tileSize Size of the tiles used for processing.
/// @param tileOffsets Tensor containing offsets for each tile.
/// @param tileGaussianIds Tensor mapping tiles to Gaussian IDs.
/// @param activeTiles Tensor containing the indices of active tiles.
/// @param tilePixelMask Tensor containing the mask for each tile pixel.
/// @param tilePixelCumsum Tensor containing the cumulative sum of tile pixels.
/// @param pixelMap Tensor containing the mapping of pixels to output indices.
/// @param backgrounds Optional background color per camera [C, D]. If provided, background
/// colors will be blended with transparent pixels. If not provided, background is assumed to be
/// black.
/// @param masks Optional per-tile boolean mask [C, tile_height, tile_width]
/// @return A tuple containing:
///         - Output colors JaggedTensor for the specified pixels.
///         - Output alphas JaggedTensor for the specified pixels.
///         - Output last Gaussian IDs JaggedTensor for the specified pixels.
std::tuple<fvdb::JaggedTensor, fvdb::JaggedTensor, fvdb::JaggedTensor>
rasterizeScreenSpaceGaussiansSparseFwd(const fvdb::JaggedTensor &pixelsToRender,
                                       const torch::Tensor &means2d,
                                       const torch::Tensor &conics,
                                       const torch::Tensor &features,
                                       const torch::Tensor &opacities,
                                       uint32_t imageWidth,
                                       uint32_t imageHeight,
                                       uint32_t imageOriginW,
                                       uint32_t imageOriginH,
                                       uint32_t tileSize,
                                       const torch::Tensor &tileOffsets,
                                       const torch::Tensor &tileGaussianIds,
                                       const torch::Tensor &activeTiles,
                                       const torch::Tensor &tilePixelMask,
                                       const torch::Tensor &tilePixelCumsum,
                                       const torch::Tensor &pixelMap,
                                       const at::optional<torch::Tensor> &backgrounds = at::nullopt,
                                       const at::optional<torch::Tensor> &masks = at::nullopt);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_GSPLAT_RASTERIZESCREENSPACEGAUSSIANSFORWARD_H
