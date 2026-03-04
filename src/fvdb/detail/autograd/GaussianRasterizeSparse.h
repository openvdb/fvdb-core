// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_AUTOGRAD_GAUSSIANRASTERIZESPARSE_H
#define FVDB_DETAIL_AUTOGRAD_GAUSSIANRASTERIZESPARSE_H

#include <fvdb/JaggedTensor.h>

#include <torch/autograd.h>

namespace fvdb::detail::autograd {

struct RasterizeGaussiansToPixelsSparse
    : public torch::autograd::Function<RasterizeGaussiansToPixelsSparse> {
    using VariableList    = torch::autograd::variable_list;
    using AutogradContext = torch::autograd::AutogradContext;
    using Variable        = torch::autograd::Variable;

    static VariableList forward(
        AutogradContext *ctx,
        const JaggedTensor &pixelsToRender, // [C, num_pixels, 2]
        const Variable &means2d,            // [C, N, 2]
        const Variable &conics,             // [C, N, 3]
        const Variable &features,           // [C, N, D]
        const Variable &opacities,          // [N]
        const uint32_t imageWidth,
        const uint32_t imageHeight,
        const uint32_t imageOriginW,
        const uint32_t imageOriginH,
        const uint32_t tileSize,
        const Variable
            &tileOffsets, // [C, tile_height, tile_width] (dense) or [num_active_tiles + 1] (sparse)
        const Variable &tileGaussianIds, // [n_isects]
        const Variable &activeTiles,     // [num_active_tiles]
        const Variable &tilePixelMask,   // [num_active_tiles, tileSize, tileSize]
        const Variable &tilePixelCumsum, // [num_active_tiles + 1]
        const Variable &pixelMap,        // [num_pixels]
        const bool absgrad);

    static VariableList backward(AutogradContext *ctx, VariableList gradOutput);
};

} // namespace fvdb::detail::autograd

#endif // FVDB_DETAIL_AUTOGRAD_GAUSSIANRASTERIZESPARSE_H
