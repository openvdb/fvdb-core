// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_AUTOGRAD_GAUSSIANRASTERIZEFROMWORLD_H
#define FVDB_DETAIL_AUTOGRAD_GAUSSIANRASTERIZEFROMWORLD_H

#include <fvdb/detail/ops/gsplat/GaussianProjectionTypes.h>

#include <torch/autograd.h>

#include <optional>

namespace fvdb::detail::autograd {

/// @brief Autograd wrapper for dense rasterization from world-space 3D Gaussians.
struct RasterizeGaussiansToPixelsFromWorld3DGS
    : public torch::autograd::Function<RasterizeGaussiansToPixelsFromWorld3DGS> {
    using VariableList    = torch::autograd::variable_list;
    using AutogradContext = torch::autograd::AutogradContext;
    using Variable        = torch::autograd::Variable;

    static VariableList
    forward(AutogradContext *ctx,
            const Variable &means,                   // [N,3]
            const Variable &quats,                   // [N,4]
            const Variable &logScales,               // [N,3]
            const Variable &features,                // [C,N,D]
            const Variable &opacities,               // [C,N]
            const Variable &worldToCamMatricesStart, // [C,4,4]
            const Variable &worldToCamMatricesEnd,   // [C,4,4]
            const Variable &projectionMatrices,      // [C,3,3]
            const Variable &distortionCoeffs,        // [C,K]
            const fvdb::detail::ops::RollingShutterType rollingShutterType,
            const fvdb::detail::ops::CameraModel cameraModel,
            const uint32_t imageWidth,
            const uint32_t imageHeight,
            const uint32_t imageOriginW,
            const uint32_t imageOriginH,
            const uint32_t tileSize,
            const Variable &tileOffsets,                        // [C, tileH, tileW]
            const Variable &tileGaussianIds,                    // [n_isects]
            std::optional<Variable> backgrounds = std::nullopt, // [C,D]
            std::optional<Variable> masks       = std::nullopt);      // [C,tileH,tileW] bool

    static VariableList backward(AutogradContext *ctx, VariableList gradOutput);
};

} // namespace fvdb::detail::autograd

#endif // FVDB_DETAIL_AUTOGRAD_GAUSSIANRASTERIZEFROMWORLD_H
