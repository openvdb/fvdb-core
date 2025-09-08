// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_GAUSSIANRASTERIZENUMCONTRIBUTINGGAUSSIANS_H
#define FVDB_DETAIL_OPS_GSPLAT_GAUSSIANRASTERIZENUMCONTRIBUTINGGAUSSIANS_H

#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/ops/gsplat/GaussianRenderSettings.h>

#include <torch/types.h>

#include <tuple>

namespace fvdb {
namespace detail {
namespace ops {

template <torch::DeviceType>
std::tuple<torch::Tensor, torch::Tensor> dispatchGaussianRasterizeNumContributingGaussians(
    // Gaussian parameters
    const torch::Tensor &means2d,         // [C, N, 2]
    const torch::Tensor &conics,          // [C, N, 3]
    const torch::Tensor &opacities,       // [N]
    const torch::Tensor &tileOffsets,     // [C, tile_height, tile_width]
    const torch::Tensor &tileGaussianIds, // [n_isects]
    const RenderSettings &settings        // render settings
);

template <torch::DeviceType>
std::tuple<fvdb::JaggedTensor, fvdb::JaggedTensor>
dispatchGaussianSparseRasterizeNumContributingGaussians(
    const torch::Tensor &means2d,         // [C, N, 2]
    const torch::Tensor &conics,          // [C, N, 3]
    const torch::Tensor &opacities,       // [N]
    const torch::Tensor &tileOffsets,     // [C, tile_height, tile_width]
    const torch::Tensor &tileGaussianIds, // [n_isects]
    const fvdb::JaggedTensor &pixelsToRender,
    const torch::Tensor &activeTiles,
    const torch::Tensor &tilePixelMask,
    const torch::Tensor &tilePixelCumsum,
    const torch::Tensor &pixelMap,
    const RenderSettings &settings // render settings
);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_GSPLAT_GAUSSIANRASTERIZENUMCONTRIBUTINGGAUSSIANS_H
