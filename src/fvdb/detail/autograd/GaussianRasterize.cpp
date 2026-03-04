// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/autograd/GaussianRasterize.h>
#include <fvdb/detail/ops/gsplat/GaussianRasterizeBackward.h>
#include <fvdb/detail/ops/gsplat/GaussianRasterizeForward.h>
#include <fvdb/detail/utils/Nvtx.h>
#include <fvdb/detail/utils/Utils.h>

namespace fvdb::detail::autograd {

RasterizeGaussiansToPixels::VariableList
RasterizeGaussiansToPixels::forward(
    RasterizeGaussiansToPixels::AutogradContext *ctx,
    const RasterizeGaussiansToPixels::Variable &means2d,   // [C, N, 2]
    const RasterizeGaussiansToPixels::Variable &conics,    // [C, N, 3]
    const RasterizeGaussiansToPixels::Variable &colors,    // [C, N, 3]
    const RasterizeGaussiansToPixels::Variable &opacities, // [N]
    const ops::RenderWindow2D &renderWindow,
    const ops::DenseTileIntersections &tileIntersections,
    const bool absgrad,
    std::optional<RasterizeGaussiansToPixels::Variable> backgrounds,
    std::optional<RasterizeGaussiansToPixels::Variable> masks) {
    FVDB_FUNC_RANGE_WITH_NAME("RasterizeGaussiansToPixels::forward");

    const auto &tileOffsets     = tileIntersections.tileOffsets();
    const auto &tileGaussianIds = tileIntersections.tileGaussianIds();
    const uint32_t tileSize     = tileIntersections.tileSize();

    auto variables          = FVDB_DISPATCH_KERNEL(means2d.device(), [&]() {
        return ops::dispatchGaussianRasterizeForward<DeviceTag>(means2d,
                                                                conics,
                                                                colors,
                                                                opacities,
                                                                renderWindow,
                                                                tileSize,
                                                                tileOffsets,
                                                                tileGaussianIds,
                                                                backgrounds,
                                                                masks);
    });
    Variable renderedColors = std::get<0>(variables);
    Variable renderedAlphas = std::get<1>(variables);
    Variable lastIds        = std::get<2>(variables);

    std::vector<Variable> toSave = {
        means2d, conics, colors, opacities, tileOffsets, tileGaussianIds, renderedAlphas, lastIds};
    if (backgrounds.has_value()) {
        toSave.push_back(backgrounds.value());
        ctx->saved_data["has_backgrounds"] = true;
    } else {
        ctx->saved_data["has_backgrounds"] = false;
    }
    if (masks.has_value()) {
        toSave.push_back(masks.value());
        ctx->saved_data["has_masks"] = true;
    } else {
        ctx->saved_data["has_masks"] = false;
    }
    ctx->save_for_backward(toSave);

    ctx->saved_data["imageWidth"]   = (int64_t)renderWindow.width;
    ctx->saved_data["imageHeight"]  = (int64_t)renderWindow.height;
    ctx->saved_data["tileSize"]     = (int64_t)tileSize;
    ctx->saved_data["imageOriginW"] = (int64_t)renderWindow.originW;
    ctx->saved_data["imageOriginH"] = (int64_t)renderWindow.originH;
    ctx->saved_data["absgrad"]      = absgrad;

    return {renderedColors, renderedAlphas};
}

RasterizeGaussiansToPixels::VariableList
RasterizeGaussiansToPixels::backward(RasterizeGaussiansToPixels::AutogradContext *ctx,
                                     RasterizeGaussiansToPixels::VariableList gradOutput) {
    FVDB_FUNC_RANGE_WITH_NAME("RasterizeGaussiansToPixels::backward");
    Variable dLossDRenderedColors = gradOutput.at(0);
    Variable dLossDRenderedAlphas = gradOutput.at(1);

    if (dLossDRenderedColors.defined()) {
        dLossDRenderedColors = dLossDRenderedColors.contiguous();
    }
    if (dLossDRenderedAlphas.defined()) {
        dLossDRenderedAlphas = dLossDRenderedAlphas.contiguous();
    }

    VariableList saved       = ctx->get_saved_variables();
    Variable means2d         = saved.at(0);
    Variable conics          = saved.at(1);
    Variable colors          = saved.at(2);
    Variable opacities       = saved.at(3);
    Variable tileOffsets     = saved.at(4);
    Variable tileGaussianIds = saved.at(5);
    Variable renderedAlphas  = saved.at(6);
    Variable lastIds         = saved.at(7);

    const bool hasBackgrounds                = ctx->saved_data["has_backgrounds"].toBool();
    const bool hasMasks                      = ctx->saved_data["has_masks"].toBool();
    std::optional<torch::Tensor> backgrounds = std::nullopt;
    std::optional<torch::Tensor> masks       = std::nullopt;
    int64_t optIdx                           = 8;
    if (hasBackgrounds) {
        backgrounds = saved.at(optIdx++);
    }
    if (hasMasks) {
        masks = saved.at(optIdx++);
    }

    const ops::RenderWindow2D renderWindow{
        static_cast<uint32_t>(ctx->saved_data["imageWidth"].toInt()),
        static_cast<uint32_t>(ctx->saved_data["imageHeight"].toInt()),
        static_cast<uint32_t>(ctx->saved_data["imageOriginW"].toInt()),
        static_cast<uint32_t>(ctx->saved_data["imageOriginH"].toInt())};
    const int tileSize = (int)ctx->saved_data["tileSize"].toInt();
    const bool absgrad = ctx->saved_data["absgrad"].toBool();

    auto variables = FVDB_DISPATCH_KERNEL(means2d.device(), [&]() {
        return ops::dispatchGaussianRasterizeBackward<DeviceTag>(means2d,
                                                                 conics,
                                                                 colors,
                                                                 opacities,
                                                                 renderWindow,
                                                                 tileSize,
                                                                 tileOffsets,
                                                                 tileGaussianIds,
                                                                 renderedAlphas,
                                                                 lastIds,
                                                                 dLossDRenderedColors,
                                                                 dLossDRenderedAlphas,
                                                                 absgrad,
                                                                 -1,
                                                                 backgrounds,
                                                                 masks);
    });
    Variable dLossDMean2dAbs;
    if (absgrad) {
        dLossDMean2dAbs = std::get<0>(variables);
    } else {
        dLossDMean2dAbs = Variable();
    }
    Variable dLossDMeans2d   = std::get<1>(variables);
    Variable dLossDConics    = std::get<2>(variables);
    Variable dLossDColors    = std::get<3>(variables);
    Variable dLossDOpacities = std::get<4>(variables);

    // 9 forward params (excluding ctx): means2d, conics, colors, opacities,
    // renderWindow, tileIntersections, absgrad, backgrounds, masks
    return {
        dLossDMeans2d,
        dLossDConics,
        dLossDColors,
        dLossDOpacities,
        Variable(), // renderWindow
        Variable(), // tileIntersections
        Variable(), // absgrad
        Variable(), // backgrounds
        Variable(), // masks
    };
}

} // namespace fvdb::detail::autograd
