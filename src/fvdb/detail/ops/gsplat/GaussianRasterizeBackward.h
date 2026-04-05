// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_GAUSSIANRASTERIZEBACKWARD_H
#define FVDB_DETAIL_OPS_GSPLAT_GAUSSIANRASTERIZEBACKWARD_H

#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/ops/gsplat/GaussianRenderSettings.h>

#include <nanovdb/math/Math.h>

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
/// @param[in] tileOffsets Offsets for tiles [C, tile_height, tile_width]
/// @param[in] tileGaussianIds Flattened Gaussian IDs for tile intersection [n_isects]
/// @param[in] renderedAlphas Alpha values from forward pass [C, render_height, render_width, 1]
/// @param[in] lastIds Last Gaussian IDs per pixel from forward pass [C, render_height,
/// render_width]
/// @param[out] dLossDRenderedFeatures Gradients of loss with respect to rendered features [C,
/// render_height, render_width, D]
/// @param[out] dLossDRenderedAlphas Gradients of loss with respect to rendered alphas [C,
/// render_height, render_width, 1]
/// @param[in] absGrad Whether to use absolute gradients
/// @param[in] numSharedChannelsOverride Override for number of shared memory channels (-1 means
/// auto-select)
/// @param[in] backgrounds Optional background color per camera [C, D]. If provided, background
/// colors affect gradient computation for transparent pixels. If not provided, background is
/// assumed to be black.
/// @param[in] masks Optional per-tile boolean mask [C, tile_height, tile_width]
///
/// @return std::tuple containing gradients of the loss function with respect to the input
/// parameters:
///         - Absolute value of 2D means [C, N, 2] - gradients dL/d|means2d| (optional: if
///         absGrad is true, this tensor is returned, otherwise it is an empty tensor)
///         - 2D means [C, N, 2] - gradients dL/dmeans2d
///         - conics [C, N, 3] - gradients dL/dconics
///         - features [C, N, D] - gradients dL/dfeatures
///         - opacities [N] - gradients dL/dopacities
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
gaussianRasterizeBackward(const torch::Tensor &means2d,
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
                          const torch::Tensor &renderedAlphas,
                          const torch::Tensor &lastIds,
                          const torch::Tensor &dLossDRenderedFeatures,
                          const torch::Tensor &dLossDRenderedAlphas,
                          bool absGrad,
                          int64_t numSharedChannelsOverride              = -1,
                          const at::optional<torch::Tensor> &backgrounds = at::nullopt,
                          const at::optional<torch::Tensor> &masks       = at::nullopt);

/// @brief Calculate gradients for the sparse Gaussian rasterization process (backward pass)
///
/// This function computes the gradients of the sparse Gaussian splatting rendering with respect to
/// its input parameters for only the specified pixels. It combines the efficiency of sparse
/// rasterization with gradient computation, processing only the pixels specified in pixelsToRender.
///
/// @param[in] pixelsToRender JaggedTensor containing pixel coordinates to render [C, NumPixels, 2]
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
/// @param[in] tileOffsets Offsets for tiles [C, tile_height, tile_width]
/// @param[in] tileGaussianIds Flattened Gaussian IDs for tile intersection [n_isects]
/// @param[in] renderedAlphas Alpha values from sparse forward pass [JaggedTensor]
/// @param[in] lastIds Last Gaussian IDs per pixel from sparse forward pass [JaggedTensor]
/// @param[in] dLossDRenderedFeatures Gradients of loss w.r.t sparse rendered features
/// [JaggedTensor]
/// @param[in] dLossDRenderedAlphas Gradients of loss w.r.t sparse rendered alphas [JaggedTensor]
/// @param[in] activeTiles Tensor containing indices of active tiles
/// @param[in] tilePixelMask Tensor containing the mask for each tile pixel
/// @param[in] tilePixelCumsum Tensor containing cumulative sum of tile pixels
/// @param[in] pixelMap Tensor containing mapping of pixels to output indices
/// @param[in] absGrad Whether to use absolute gradients
/// @param[in] numSharedChannelsOverride Override for number of shared memory channels (-1 means
/// auto-select)
/// @param[in] backgrounds Optional background color per camera [C, D]
/// @param[in] masks Optional per-tile boolean mask [C, tile_height, tile_width]
///
/// @return std::tuple containing gradients of the loss function with respect to the input
/// parameters:
///         - Absolute value of 2D means [C, N, 2] - gradients dL/d|means2d|
///         - 2D means [C, N, 2] - gradients dL/dmeans2d
///         - conics [C, N, 3] - gradients dL/dconics
///         - features [C, N, D] - gradients dL/dfeatures
///         - opacities [N] - gradients dL/dopacities
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
gaussianSparseRasterizeBackward(const fvdb::JaggedTensor &pixelsToRender,
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
                                const fvdb::JaggedTensor &renderedAlphas,
                                const fvdb::JaggedTensor &lastIds,
                                const fvdb::JaggedTensor &dLossDRenderedFeatures,
                                const fvdb::JaggedTensor &dLossDRenderedAlphas,
                                const torch::Tensor &activeTiles,
                                const torch::Tensor &tilePixelMask,
                                const torch::Tensor &tilePixelCumsum,
                                const torch::Tensor &pixelMap,
                                bool absGrad,
                                int64_t numSharedChannelsOverride              = -1,
                                const at::optional<torch::Tensor> &backgrounds = at::nullopt,
                                const at::optional<torch::Tensor> &masks       = at::nullopt);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_GSPLAT_GAUSSIANRASTERIZEBACKWARD_H
