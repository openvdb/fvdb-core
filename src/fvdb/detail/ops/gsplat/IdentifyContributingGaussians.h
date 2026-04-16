// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_IDENTIFYCONTRIBUTINGGAUSSIANS_H
#define FVDB_DETAIL_OPS_GSPLAT_IDENTIFYCONTRIBUTINGGAUSSIANS_H

#include <fvdb/JaggedTensor.h>

#include <torch/types.h>

#include <cstdint>
#include <tuple>

namespace fvdb {
namespace detail {
namespace ops {

/// @brief Identify contributing Gaussians per pixel (dense).
///
/// Deep image rasterization: for each pixel, returns the IDs of all Gaussians
/// that contributed non-negligible opacity, along with per-Gaussian alpha weights.
///
/// @return (gaussian_ids, weights) as JaggedTensors
std::tuple<fvdb::JaggedTensor, fvdb::JaggedTensor> identifyContributingGaussians(
    const torch::Tensor &means2d,
    const torch::Tensor &conics,
    const torch::Tensor &opacities,
    const torch::Tensor &tile_offsets,
    const torch::Tensor &tile_gaussian_ids,
    uint32_t imageWidth,
    uint32_t imageHeight,
    uint32_t imageOriginW,
    uint32_t imageOriginH,
    uint32_t tileSize,
    int numDepthSamples,
    const std::optional<torch::Tensor> &maybeNumContributingGaussians = std::nullopt);

/// @brief Identify contributing Gaussians at specified pixels (sparse).
///
/// Sparse variant that identifies only at the requested pixel locations.
///
/// @return (gaussian_ids, weights) as JaggedTensors
std::tuple<fvdb::JaggedTensor, fvdb::JaggedTensor> identifyContributingGaussiansSparse(
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
    uint32_t imageWidth,
    uint32_t imageHeight,
    uint32_t imageOriginW,
    uint32_t imageOriginH,
    uint32_t tileSize,
    int numDepthSamples,
    const std::optional<fvdb::JaggedTensor> &maybeNumContributingGaussians = std::nullopt);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_GSPLAT_IDENTIFYCONTRIBUTINGGAUSSIANS_H
