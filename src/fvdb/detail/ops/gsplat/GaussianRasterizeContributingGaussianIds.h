// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_GAUSSIANRASTERIZECONTRIBUTINGGAUSSIANIDS_H
#define FVDB_DETAIL_OPS_GSPLAT_GAUSSIANRASTERIZECONTRIBUTINGGAUSSIANIDS_H

#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/ops/gsplat/GaussianProjectionTypes.h>
#include <fvdb/detail/ops/gsplat/GaussianRenderSettings.h>

#include <torch/types.h>

#include <tuple>

namespace fvdb {
namespace detail {
namespace ops {

/// @brief Performs deep image rasterization to render the IDs and weighted alpha values of the
/// contributing Gaussians for each pixel (dispatch wrapper).
std::tuple<fvdb::JaggedTensor, fvdb::JaggedTensor> gaussianRasterizeContributingGaussianIds(
    const torch::Tensor &means2d,
    const torch::Tensor &conics,
    const torch::Tensor &opacities,
    const torch::Tensor &tile_offsets,
    const torch::Tensor &tile_gaussian_ids,
    const RenderSettings &settings,
    const std::optional<torch::Tensor> &maybeNumContributingGaussians = std::nullopt);

/// @brief Performs sparse deep image rasterization to render the IDs and weighted alpha values of
/// the contributing Gaussians for each pixel. Renders only specified pixels (dispatch wrapper).
std::tuple<fvdb::JaggedTensor, fvdb::JaggedTensor> gaussianSparseRasterizeContributingGaussianIds(
    const torch::Tensor &means2d,
    const torch::Tensor &conics,
    const torch::Tensor &opacities,
    const torch::Tensor &tile_offsets,
    const torch::Tensor &tile_gaussian_ids,
    const fvdb::JaggedTensor &pixelsToRender,
    const torch::Tensor &activeTiles,
    const torch::Tensor &tilePixelMask,
    const torch::Tensor &tilePixelCumsum,
    const torch::Tensor &pixelMap,
    const RenderSettings &settings,
    const std::optional<fvdb::JaggedTensor> &maybeNumContributingGaussians = std::nullopt);

/// @brief Render the IDs and weighted alpha values of contributing Gaussians per pixel (dense).
///
/// Performs deep image rasterization: for each pixel, returns the IDs of all Gaussians that
/// contributed non-negligible opacity, along with their per-Gaussian alpha weights. Uses the
/// same tile-based traversal as the forward rasterizer.
///
/// @param[in] state   Pre-projected Gaussian state containing 2D means [C, N, 2],
///                    conics [C, N, 3], opacities [N], tile offsets, and tile Gaussian IDs
/// @param[in] settings Render settings (image dimensions, tile size)
/// @param[in] maybeNumContributingGaussians Optional pre-computed contributing counts [C, H, W];
///            if provided, avoids recomputing them internally
///
/// @return std::tuple containing:
///         - Gaussian IDs per pixel (JaggedTensor, variable-length per pixel)
///         - Weighted alpha values per pixel (JaggedTensor, same structure)
std::tuple<fvdb::JaggedTensor, fvdb::JaggedTensor>
renderContributingIds(const fvdb::ProjectedGaussianSplats &state,
                      const RenderSettings &settings,
                      const std::optional<torch::Tensor> &maybeNumContributingGaussians);

/// @brief Render the IDs and weighted alpha values of contributing Gaussians at specified
///        pixels (sparse).
///
/// Sparse variant of deep image rasterization that renders only at the requested pixel
/// locations. If the input pixels contained duplicates, results are scattered back to all
/// original positions via the inverse index mapping.
///
/// @param[in] state           Pre-projected sparse Gaussian state
/// @param[in] pixelsToRender  Pixel coordinates to query, as JaggedTensor [total_pixels, 2]
/// @param[in] settings        Render settings (image dimensions, tile size)
/// @param[in] maybeNumContributingGaussians Optional pre-computed contributing counts
///
/// @return std::tuple containing:
///         - Gaussian IDs per pixel (JaggedTensor)
///         - Weighted alpha values per pixel (JaggedTensor)
std::tuple<fvdb::JaggedTensor, fvdb::JaggedTensor>
sparseRenderContributingIds(const fvdb::SparseProjectedGaussianSplats &state,
                            const fvdb::JaggedTensor &pixelsToRender,
                            const RenderSettings &settings,
                            const std::optional<fvdb::JaggedTensor> &maybeNumContributingGaussians);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_GSPLAT_GAUSSIANRASTERIZECONTRIBUTINGGAUSSIANIDS_H
