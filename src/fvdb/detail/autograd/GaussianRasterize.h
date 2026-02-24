// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_AUTOGRAD_GAUSSIANRASTERIZE_H
#define FVDB_DETAIL_AUTOGRAD_GAUSSIANRASTERIZE_H

#include <torch/autograd.h>

namespace fvdb::detail::autograd {

struct RasterizeGaussiansToPixels : public torch::autograd::Function<RasterizeGaussiansToPixels> {
    using VariableList    = torch::autograd::variable_list;
    using AutogradContext = torch::autograd::AutogradContext;
    using Variable        = torch::autograd::Variable;

    static VariableList forward(AutogradContext *ctx,
                                const Variable &means2d,   // [C, N, 2]
                                const Variable &conics,    // [C, N, 3]
                                const Variable &colors,    // [C, N, 3]
                                const Variable &opacities, // [N]
                                const uint32_t imageWidth,
                                const uint32_t imageHeight,
                                const uint32_t imageOriginW,
                                const uint32_t imageOriginH,
                                const uint32_t tileSize,
                                const Variable &tileOffsets,     // [C, tile_height, tile_width]
                                const Variable &tileGaussianIds, // [n_isects]
                                const bool absgrad,
                                std::optional<Variable> backgrounds = std::nullopt, // [C, D]
                                std::optional<Variable> masks = std::nullopt); // [C, tileH, tileW]

    static VariableList backward(AutogradContext *ctx, VariableList gradOutput);
};

} // namespace fvdb::detail::autograd

#endif // FVDB_DETAIL_AUTOGRAD_GAUSSIANRASTERIZE_H
