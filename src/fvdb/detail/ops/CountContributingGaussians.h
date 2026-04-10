// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_COUNTCONTRIBUTINGGAUSSIANS_H
#define FVDB_DETAIL_OPS_COUNTCONTRIBUTINGGAUSSIANS_H

#include <fvdb/JaggedTensor.h>

#include <torch/types.h>

#include <cstdint>
#include <tuple>

namespace fvdb {
namespace detail {
namespace ops {

/// @brief Count contributing Gaussians per pixel (dense).
///
/// For each pixel, counts how many Gaussians contribute non-negligible opacity
/// using the same tile-based traversal as the rasterizer.
///
/// @return (num_contributing [C, H, W], alpha [C, H, W, 1])
std::tuple<torch::Tensor, torch::Tensor>
count_contributing_gaussians(const torch::Tensor &means2d,
                             const torch::Tensor &conics,
                             const torch::Tensor &opacities,
                             const torch::Tensor &tileOffsets,
                             const torch::Tensor &tileGaussianIds,
                             uint32_t imageWidth,
                             uint32_t imageHeight,
                             uint32_t imageOriginW,
                             uint32_t imageOriginH,
                             uint32_t tileSize);

/// @brief Count contributing Gaussians at specified pixels (sparse).
///
/// Sparse variant that counts only at the requested pixel locations.
///
/// @return (num_contributing, alpha) as JaggedTensors
std::tuple<fvdb::JaggedTensor, fvdb::JaggedTensor>
count_contributing_gaussians_sparse(const torch::Tensor &means2d,
                                    const torch::Tensor &conics,
                                    const torch::Tensor &opacities,
                                    const torch::Tensor &tileOffsets,
                                    const torch::Tensor &tileGaussianIds,
                                    const fvdb::JaggedTensor &pixelsToRender,
                                    const torch::Tensor &activeTiles,
                                    const torch::Tensor &tilePixelMask,
                                    const torch::Tensor &tilePixelCumsum,
                                    const torch::Tensor &pixelMap,
                                    uint32_t imageWidth,
                                    uint32_t imageHeight,
                                    uint32_t imageOriginW,
                                    uint32_t imageOriginH,
                                    uint32_t tileSize);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_COUNTCONTRIBUTINGGAUSSIANS_H
