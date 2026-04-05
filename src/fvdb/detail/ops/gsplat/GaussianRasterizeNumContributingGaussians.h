// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_GAUSSIANRASTERIZENUMCONTRIBUTINGGAUSSIANS_H
#define FVDB_DETAIL_OPS_GSPLAT_GAUSSIANRASTERIZENUMCONTRIBUTINGGAUSSIANS_H

#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/ops/gsplat/GaussianProjectionTypes.h>
#include <fvdb/detail/ops/gsplat/GaussianRenderSettings.h>

#include <torch/types.h>

#include <tuple>

namespace fvdb {
namespace detail {
namespace ops {

/// @brief Render the number of contributing Gaussians per pixel (dense, dispatch wrapper).
std::tuple<torch::Tensor, torch::Tensor>
gaussianRasterizeNumContributingGaussians(const torch::Tensor &means2d,
                                          const torch::Tensor &conics,
                                          const torch::Tensor &opacities,
                                          const torch::Tensor &tileOffsets,
                                          const torch::Tensor &tileGaussianIds,
                                          const RenderSettings &settings);

/// @brief Render the number of contributing Gaussians at specified pixels (sparse, dispatch
/// wrapper).
std::tuple<fvdb::JaggedTensor, fvdb::JaggedTensor>
gaussianSparseRasterizeNumContributingGaussians(const torch::Tensor &means2d,
                                                const torch::Tensor &conics,
                                                const torch::Tensor &opacities,
                                                const torch::Tensor &tileOffsets,
                                                const torch::Tensor &tileGaussianIds,
                                                const fvdb::JaggedTensor &pixelsToRender,
                                                const torch::Tensor &activeTiles,
                                                const torch::Tensor &tilePixelMask,
                                                const torch::Tensor &tilePixelCumsum,
                                                const torch::Tensor &pixelMap,
                                                const RenderSettings &settings);

/// @brief Render the number of contributing Gaussians per pixel (dense).
///
/// For each pixel, counts how many Gaussians contribute non-negligible opacity
/// using the same tile-based traversal as the rasterizer.
///
/// @param[in] state    Pre-projected Gaussian state containing 2D means [C, N, 2],
///                     conics [C, N, 3], opacities [N], tile offsets, and tile Gaussian IDs
/// @param[in] settings Render settings (image dimensions, tile size)
///
/// @return std::tuple containing:
///         - Number of contributing Gaussians per pixel [C, H, W]
///         - Alpha values per pixel [C, H, W, 1]
std::tuple<torch::Tensor, torch::Tensor>
renderNumContributing(const fvdb::ProjectedGaussianSplats &state, const RenderSettings &settings);

/// @brief Render the number of contributing Gaussians at specified pixels (sparse).
///
/// Sparse variant that renders only at the requested pixel locations. If the input
/// pixels contained duplicates (detected during projection), results are scattered
/// back to all original positions via the inverse index mapping.
///
/// @param[in] state           Pre-projected sparse Gaussian state
/// @param[in] pixelsToRender  Pixel coordinates to query, as JaggedTensor [total_pixels, 2]
/// @param[in] settings        Render settings (image dimensions, tile size)
///
/// @return std::tuple containing:
///         - Number of contributing Gaussians per pixel (JaggedTensor)
///         - Alpha values per pixel (JaggedTensor)
std::tuple<fvdb::JaggedTensor, fvdb::JaggedTensor>
sparseRenderNumContributing(const fvdb::SparseProjectedGaussianSplats &state,
                            const fvdb::JaggedTensor &pixelsToRender,
                            const RenderSettings &settings);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_GSPLAT_GAUSSIANRASTERIZENUMCONTRIBUTINGGAUSSIANS_H
