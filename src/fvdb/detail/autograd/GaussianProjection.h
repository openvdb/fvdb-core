// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_AUTOGRAD_GAUSSIANPROJECTION_H
#define FVDB_DETAIL_AUTOGRAD_GAUSSIANPROJECTION_H

#include <torch/autograd.h>

namespace fvdb::detail::autograd {

struct ProjectGaussians : public torch::autograd::Function<ProjectGaussians> {
    using VariableList    = torch::autograd::variable_list;
    using AutogradContext = torch::autograd::AutogradContext;
    using Variable        = torch::autograd::Variable;

    static VariableList
    forward(AutogradContext *ctx,
            const Variable &means,              // [N, 3]
            const Variable &quats,              // [N, 4]
            const Variable &scales,             // [N, 3]
            const Variable &camToWorldMatrices, // [C, 4, 4]
            const Variable &projectionMatrices, // [C, 3, 3]
            const uint32_t imageWidth,
            const uint32_t imageHeight,
            const float eps2d,
            const float nearPlane,
            const float farPlane,
            const float minRadius2D,
            const bool calcCompensions,
            const bool ortho,
            std::optional<Variable> outNormalizeddLossdMeans2dNormAccum = std::nullopt,
            std::optional<Variable> outNormalizedMaxRadiiAccum          = std::nullopt,
            std::optional<Variable> outGradientStepCount                = std::nullopt);

    static VariableList backward(AutogradContext *ctx, VariableList gradOutput);
};

struct ProjectGaussiansJagged : public torch::autograd::Function<ProjectGaussiansJagged> {
    using VariableList    = torch::autograd::variable_list;
    using AutogradContext = torch::autograd::AutogradContext;
    using Variable        = torch::autograd::Variable;

    static VariableList forward(AutogradContext *ctx,
                                const Variable &gSizes,             // [B] gaussian sizes
                                const Variable &means,              // [ggz, 3]
                                const Variable &quats,              // [ggz, 4] optional
                                const Variable &scales,             // [ggz, 3] optional
                                const Variable &cSizes,             // [B] camera sizes
                                const Variable &camToWorldMatrices, // [ccz, 4, 4]
                                const Variable &projectionMatrices, // [ccz, 3, 3]
                                const uint32_t imageWidth,
                                const uint32_t imageHeight,
                                const float eps2d,
                                const float nearPlane,
                                const float farPlane,
                                const float minRadius2D,
                                const bool ortho);

    static VariableList backward(AutogradContext *ctx, VariableList gradOutput);
};

} // namespace fvdb::detail::autograd

#endif // FVDB_DETAIL_AUTOGRAD_GAUSSIANPROJECTION_H
