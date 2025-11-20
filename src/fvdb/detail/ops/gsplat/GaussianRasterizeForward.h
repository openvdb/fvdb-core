// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_GAUSSIANRASTERIZEFORWARD_H
#define FVDB_DETAIL_OPS_GSPLAT_GAUSSIANRASTERIZEFORWARD_H

#include <fvdb/JaggedTensor.h>

#include <torch/types.h>

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
/// @tparam DeviceType Device type template parameter (torch::kCUDA or torch::kCPU)
///
/// @param[in] means2d 2D projected Gaussian centers [C, N, 2]
/// @param[in] conics Gaussian covariance matrices in conic form [C, N, 3] representing (a, b, c) in
/// ax² + 2bxy + cy²
/// @param[in] features Feature / color values of Gaussians [C, N, D]
/// @param[in] opacities Opacity values for each Gaussian [N]
/// @param[in] imageWidth Width of the output image in pixels
/// @param[in] imageHeight Height of the output image in pixels
/// @param[in] imageOriginW X-coordinate of the image origin (left)
/// @param[in] imageOriginH Y-coordinate of the image origin (top)
/// @param[in] tileSize Size of tiles used for rasterization optimization
/// @param[in] tileOffsets Offsets for tiles [C, tile_height, tile_width] indicating for each tile
/// where its Gaussians start
/// @param[in] tileGaussianIds Flattened Gaussian IDs for tile intersection [n_isects] indicating
/// which Gaussians affect each tile
/// @param[in] backgrounds Optional background color per camera [C, D]. If provided, background
/// colors will be blended with transparent pixels. If not provided, background is assumed to be
/// black.
///
/// @return std::tuple containing:
///         - Rendered image features/colors [C, image_height, image_width, D]
///         - Alpha values [C, image_height, image_width, 1]
///         - Last Gaussian ID rendered at each pixel [C, image_height, image_width]
template <torch::DeviceType>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> dispatchGaussianRasterizeForward(
    const torch::Tensor &means2d,   // [C, N, 2]
    const torch::Tensor &conics,    // [C, N, 3]
    const torch::Tensor &features,  // [C, N, D]
    const torch::Tensor &opacities, // [N]
    const uint32_t imageWidth,
    const uint32_t imageHeight,
    const uint32_t imageOriginW,
    const uint32_t imageOriginH,
    const uint32_t tileSize,
    const torch::Tensor &tileOffsets,                            // [C, tile_height, tile_width]
    const torch::Tensor &tileGaussianIds,                        // [n_isects]
    const at::optional<torch::Tensor> &backgrounds = at::nullopt // [C, D]
);

/// @brief Dispatches the sparse Gaussian rasterization forward pass to the specified device.
/// Renders only specified pixels.
/// @tparam Device The device type (e.g., torch::kCPU or torch::kCUDA).
/// @param pixelsToRender Tensor containing the indices of pixels to render [C, NumPixels, 2].
/// @param means2d Tensor of 2D means.
/// @param conics Tensor of conic parameters.
/// @param features Tensor of features (colors, etc).
/// @param opacities Tensor of opacities.
/// @param imageWidth Width of the full image (for coordinate calculations).
/// @param imageHeight Height of the full image (for coordinate calculations).
/// @param imageOriginW Horizontal origin of the image grid.
/// @param imageOriginH Vertical origin of the image grid.
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
/// @return A tuple containing:
///         - Output colors JaggedTensor for the specified pixels.
///         - Output alphas JaggedTensor for the specified pixels.
///         - Output last Gaussian IDs JaggedTensor for the specified pixels.
template <torch::DeviceType Device>
std::tuple<fvdb::JaggedTensor, fvdb::JaggedTensor, fvdb::JaggedTensor>
dispatchGaussianSparseRasterizeForward(
    const fvdb::JaggedTensor &pixelsToRender,
    const torch::Tensor &means2d,
    const torch::Tensor &conics,
    const torch::Tensor &features,
    const torch::Tensor &opacities,
    const uint32_t imageWidth,
    const uint32_t imageHeight,
    const uint32_t imageOriginW,
    const uint32_t imageOriginH,
    const uint32_t tileSize,
    const torch::Tensor &tileOffsets,
    const torch::Tensor &tileGaussianIds,
    const torch::Tensor &activeTiles,
    const torch::Tensor &tilePixelMask,
    const torch::Tensor &tilePixelCumsum,
    const torch::Tensor &pixelMap,
    const at::optional<torch::Tensor> &backgrounds = at::nullopt);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_GSPLAT_GAUSSIANRASTERIZEFORWARD_H
