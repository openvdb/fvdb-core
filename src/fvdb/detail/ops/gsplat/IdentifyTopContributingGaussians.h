// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_IDENTIFYTOPCONTRIBUTINGGAUSSIANS_H
#define FVDB_DETAIL_OPS_GSPLAT_IDENTIFYTOPCONTRIBUTINGGAUSSIANS_H

#include <fvdb/JaggedTensor.h>

#include <torch/types.h>

#include <cstdint>
#include <tuple>

namespace fvdb {
namespace detail {
namespace ops {

/// @brief Performs deep image rasterization to render the IDs and weighted alpha values of the
/// top-K most visible Gaussians for each pixel (dispatch wrapper).
std::tuple<torch::Tensor, torch::Tensor>
identifyTopContributingGaussians(const torch::Tensor &means2d,
                                 const torch::Tensor &conics,
                                 const torch::Tensor &opacities,
                                 const torch::Tensor &tile_offsets,
                                 const torch::Tensor &tile_gaussian_ids,
                                 uint32_t imageWidth,
                                 uint32_t imageHeight,
                                 uint32_t imageOriginW,
                                 uint32_t imageOriginH,
                                 uint32_t tileSize,
                                 int numDepthSamples);

/// @brief Performs sparse deep image rasterization to render the IDs and weighted alpha values of
/// the top-K most visible Gaussians for each pixel. Renders only specified pixels (dispatch
/// wrapper).
std::tuple<fvdb::JaggedTensor, fvdb::JaggedTensor>
identifyTopContributingGaussiansSparse(const torch::Tensor &means2d,
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
                                       int numDepthSamples);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_GSPLAT_IDENTIFYTOPCONTRIBUTINGGAUSSIANS_H
