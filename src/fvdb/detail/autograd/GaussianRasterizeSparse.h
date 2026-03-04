// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_AUTOGRAD_GAUSSIANRASTERIZESPARSE_H
#define FVDB_DETAIL_AUTOGRAD_GAUSSIANRASTERIZESPARSE_H

#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/ops/gsplat/GaussianTileIntersection.h>

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
        const ops::RenderWindow2D &renderWindow,
        const ops::SparseTileIntersections &tileIntersections,
        const bool absgrad,
        std::optional<Variable> backgrounds = std::nullopt, // [C, D]
        std::optional<Variable> masks       = std::nullopt);      // [C, tileH, tileW]

    static VariableList backward(AutogradContext *ctx, VariableList gradOutput);
};

} // namespace fvdb::detail::autograd

#endif // FVDB_DETAIL_AUTOGRAD_GAUSSIANRASTERIZESPARSE_H
