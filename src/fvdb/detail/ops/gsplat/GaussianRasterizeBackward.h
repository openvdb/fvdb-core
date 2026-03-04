// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_GAUSSIANRASTERIZEBACKWARD_H
#define FVDB_DETAIL_OPS_GSPLAT_GAUSSIANRASTERIZEBACKWARD_H

#include <fvdb/JaggedTensor.h>

#include <torch/types.h>

#include <tuple>

namespace fvdb {
namespace detail {
namespace ops {

/// @brief Calculate gradients for the Gaussian rasterization process (backward pass)
///
/// This function computes the gradients of the Gaussian splatting rendering with respect to
/// its input parameters: 2D projected Gaussian means, conics, features/colors, and opacities.
/// It is used during backpropagation to update the Gaussian parameters during training.
///
/// @tparam DeviceType Device type template parameter (torch::kCUDA or torch::kCPU)
///
/// @param[in] means2d 2D projected Gaussian centers [C, N, 2]
/// @param[in] conics Gaussian covariance matrices in conic form [C, N, 3] representing (a, b, c) in
/// ax² + 2bxy + cy²
/// @param[in] features Feature / color values of Gaussians [C, N, D]
/// @param[in] opacities Opacity values for each Gaussian [N]
/// @param[in] imageWidth Width of the rendered image
/// @param[in] imageHeight Height of the rendered image
/// @param[in] imageOriginW X-coordinate of the image origin (left)
/// @param[in] imageOriginH Y-coordinate of the image origin (top)
/// @param[in] tileSize Size of tiles used for rasterization optimization
/// @param[in] tileOffsets Offsets for tiles [C, tile_height, tile_width]
/// @param[in] tileGaussianIds Flattened Gaussian IDs for tile intersection [n_isects]
/// @param[in] renderedAlphas Alpha values from forward pass [C, image_height, image_width, 1]
/// @param[in] lastIds Last Gaussian IDs per pixel from forward pass [C, image_height, image_width]
/// @param[out] dLossDRenderedFeatures Gradients of loss with respect to rendered features [C,
/// image_height, image_width, D]
/// @param[out] dLossDRenderedAlphas Gradients of loss with respect to rendered alphas [C,
/// image_height, image_width, 1]
/// @param[in] absGrad Whether to use absolute gradients
/// @param[in] numSharedChannelsOverride Override for number of shared memory channels (-1 means
/// auto-select)
/// @param[in] backgrounds Optional background color per camera [C, D]. If provided, background
/// colors affect gradient computation for transparent pixels. If not provided, background is
/// assumed to be black.
///
/// @return std::tuple containing gradients of the loss function with respect to the input
/// parameters:
///         - Absolute value of 2D means [C, N, 2] - gradients ∂L/∂|means2d| (optional: if
///         absGrad is true, this tensor is returned, otherwise it is an empty tensor)
///         - 2D means [C, N, 2] - gradients ∂L/∂means2d
///         - conics [C, N, 3] - gradients ∂L/∂conics
///         - features [C, N, D] - gradients ∂L/∂features
///         - opacities [N] - gradients ∂L/∂opacities
template <torch::DeviceType>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianRasterizeBackward(
    // Gaussian parameters
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
    const torch::Tensor &renderedAlphas,                         // [C, imageHeight, imageWidth, 1]
    const torch::Tensor &lastIds,                                // [C, imageHeight, imageWidth]
    const torch::Tensor &dLossDRenderedFeatures,                 // [C, imageHeight, imageWidth, D]
    const torch::Tensor &dLossDRenderedAlphas,                   // [C, imageHeight, imageWidth, 1]
    const bool absGrad,
    const int64_t numSharedChannelsOverride        = -1,
    const at::optional<torch::Tensor> &backgrounds = at::nullopt // [C, D]
);

/// @brief Calculate gradients for the sparse Gaussian rasterization process (backward pass)
///
/// This function computes the gradients of the sparse Gaussian splatting rendering with respect to
/// its input parameters for only the specified pixels. It combines the efficiency of sparse
/// rasterization with gradient computation, processing only the pixels specified in pixelsToRender.
///
/// @tparam DeviceType Device type template parameter (torch::kCUDA or torch::kCPU)
///
/// @param[in] pixelsToRender JaggedTensor containing pixel coordinates to render [C, NumPixels, 2]
/// @param[in] means2d 2D projected Gaussian centers [C, N, 2]
/// @param[in] conics Gaussian covariance matrices in conic form [C, N, 3] representing (a, b, c) in
/// ax² + 2bxy + cy²
/// @param[in] features Feature / color values of Gaussians [C, N, D]
/// @param[in] opacities Opacity values for each Gaussian [N]
/// @param[in] imageWidth Width of the full image (for coordinate calculations)
/// @param[in] imageHeight Height of the full image (for coordinate calculations)
/// @param[in] imageOriginW X-coordinate of the image origin (left)
/// @param[in] imageOriginH Y-coordinate of the image origin (top)
/// @param[in] tileSize Size of tiles used for rasterization optimization
/// @param[in] tileOffsets Offsets for tiles [C, tile_height, tile_width]
/// @param[in] tileGaussianIds Flattened Gaussian IDs for tile intersection [n_isects]
/// @param[in] activeTiles Tensor containing indices of active tiles
/// @param[in] tilePixelMask Tensor containing the mask for each tile pixel
/// @param[in] tilePixelCumsum Tensor containing cumulative sum of tile pixels
/// @param[in] pixelMap Tensor containing mapping of pixels to output indices
/// @param[in] renderedAlphas Alpha values from sparse forward pass [JaggedTensor: C lists of
/// varying sizes, each element [1]]
/// @param[in] lastIds Last Gaussian IDs per pixel from sparse forward pass [JaggedTensor: C lists
/// of varying sizes]
/// @param[in] dLossDRenderedFeatures Gradients of loss w.r.t sparse rendered features
/// [JaggedTensor: C lists of varying sizes, each element [D]]
/// @param[in] dLossDRenderedAlphas Gradients of loss w.r.t sparse rendered alphas [JaggedTensor: C
/// lists of varying sizes, each element [1]]
/// @param[in] absGrad Whether to use absolute gradients
/// @param[in] numSharedChannelsOverride Override for number of shared memory channels (-1 means
/// auto-select)
/// @param[in] backgrounds Optional background color per camera [C, D]. If provided, background
/// colors affect gradient computation for transparent pixels. If not provided, background is
/// assumed to be black.
///
/// @return std::tuple containing gradients of the loss function with respect to the input
/// parameters:
///         - Absolute value of 2D means [C, N, 2] - gradients ∂L/∂|means2d| (optional: if
///         absGrad is true, this tensor is returned, otherwise it is an empty tensor)
///         - 2D means [C, N, 2] - gradients ∂L/∂means2d
///         - conics [C, N, 3] - gradients ∂L/∂conics
///         - features [C, N, D] - gradients ∂L/∂features
///         - opacities [N] - gradients ∂L/∂opacities
template <torch::DeviceType>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianSparseRasterizeBackward(
    // Sparse pixel coordinates and setup
    const fvdb::JaggedTensor &pixelsToRender, // [C, NumPixels, 2]
    // Gaussian parameters
    const torch::Tensor &means2d,   // [C, N, 2]
    const torch::Tensor &conics,    // [C, N, 3]
    const torch::Tensor &features,  // [C, N, D]
    const torch::Tensor &opacities, // [N]
    // Image and tile setup
    const uint32_t imageWidth,
    const uint32_t imageHeight,
    const uint32_t imageOriginW,
    const uint32_t imageOriginH,
    const uint32_t tileSize,
    const torch::Tensor &tileOffsets, // [C, tile_height, tile_width] (dense) or [AT + 1] (sparse)
    const torch::Tensor &tileGaussianIds, // [n_isects]
    // Forward pass outputs (sparse)
    const fvdb::JaggedTensor &renderedAlphas, // [C lists: varying sizes, each element [1]]
    const fvdb::JaggedTensor &lastIds,        // [C lists: varying sizes]
    // Gradients (sparse)
    const fvdb::JaggedTensor &dLossDRenderedFeatures, // [C lists: varying sizes, each element [D]]
    const fvdb::JaggedTensor &dLossDRenderedAlphas,   // [C lists: varying sizes, each element [1]]
    // Sparse processing setup
    const torch::Tensor &activeTiles,     // [AT]
    const torch::Tensor &tilePixelMask,   // [AT, wordsPerTile]
    const torch::Tensor &tilePixelCumsum, // [AT]
    const torch::Tensor &pixelMap,        // [AP]
    // Options
    const bool absGrad,
    const int64_t numSharedChannelsOverride        = -1,
    const at::optional<torch::Tensor> &backgrounds = at::nullopt // [C, D]
);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_GSPLAT_GAUSSIANRASTERIZEBACKWARD_H
