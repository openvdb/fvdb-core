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
    const uint32_t imageWidth,
    const uint32_t imageHeight,
    const uint32_t imageOriginW,
    const uint32_t imageOriginH,
    const uint32_t tileSize,
    const RasterizeGaussiansToPixels::Variable &tileOffsets,     // [C, tile_height, tile_width]
    const RasterizeGaussiansToPixels::Variable &tileGaussianIds, // [n_isects]
    const bool absgrad) {
    FVDB_FUNC_RANGE_WITH_NAME("RasterizeGaussiansToPixels::forward");
    // const int C = means2d.size(0);
    // const int N = means2d.size(1);

    auto variables          = FVDB_DISPATCH_KERNEL(means2d.device(), [&]() {
        return ops::dispatchGaussianRasterizeForward<DeviceTag>(means2d,
                                                                conics,
                                                                colors,
                                                                opacities,
                                                                imageWidth,
                                                                imageHeight,
                                                                imageOriginW,
                                                                imageOriginH,
                                                                tileSize,
                                                                tileOffsets,
                                                                tileGaussianIds);
    });
    Variable renderedColors = std::get<0>(variables);
    Variable renderedAlphas = std::get<1>(variables);
    Variable lastIds        = std::get<2>(variables);

    ctx->save_for_backward({means2d,
                            conics,
                            colors,
                            opacities,
                            tileOffsets,
                            tileGaussianIds,
                            renderedAlphas,
                            lastIds});
    ctx->saved_data["imageWidth"]   = (int64_t)imageWidth;
    ctx->saved_data["imageHeight"]  = (int64_t)imageHeight;
    ctx->saved_data["tileSize"]     = (int64_t)tileSize;
    ctx->saved_data["imageOriginW"] = (int64_t)imageOriginW;
    ctx->saved_data["imageOriginH"] = (int64_t)imageOriginH;
    ctx->saved_data["absgrad"]      = absgrad;

    return {renderedColors, renderedAlphas};
}

RasterizeGaussiansToPixels::VariableList
RasterizeGaussiansToPixels::backward(RasterizeGaussiansToPixels::AutogradContext *ctx,
                                     RasterizeGaussiansToPixels::VariableList gradOutput) {
    FVDB_FUNC_RANGE_WITH_NAME("RasterizeGaussiansToPixels::backward");
    Variable dLossDRenderedColors = gradOutput.at(0);
    Variable dLossDRenderedAlphas = gradOutput.at(1);

    // ensure the gradients are contiguous if they are not None
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

    const int imageWidth   = (int)ctx->saved_data["imageWidth"].toInt();
    const int imageHeight  = (int)ctx->saved_data["imageHeight"].toInt();
    const int tileSize     = (int)ctx->saved_data["tileSize"].toInt();
    const int imageOriginW = (int)ctx->saved_data["imageOriginW"].toInt();
    const int imageOriginH = (int)ctx->saved_data["imageOriginH"].toInt();
    const bool absgrad     = ctx->saved_data["absgrad"].toBool();

    auto variables = FVDB_DISPATCH_KERNEL(means2d.device(), [&]() {
        return ops::dispatchGaussianRasterizeBackward<DeviceTag>(means2d,
                                                                 conics,
                                                                 colors,
                                                                 opacities,
                                                                 imageWidth,
                                                                 imageHeight,
                                                                 imageOriginW,
                                                                 imageOriginH,
                                                                 tileSize,
                                                                 tileOffsets,
                                                                 tileGaussianIds,
                                                                 renderedAlphas,
                                                                 lastIds,
                                                                 dLossDRenderedColors,
                                                                 dLossDRenderedAlphas,
                                                                 absgrad);
    });
    Variable dLossDMean2dAbs;
    if (absgrad) {
        dLossDMean2dAbs = std::get<0>(variables);
        // means2d.absgrad = dLossDMean2dAbs;
    } else {
        dLossDMean2dAbs = Variable();
    }
    Variable dLossDMeans2d   = std::get<1>(variables);
    Variable dLossDConics    = std::get<2>(variables);
    Variable dLossDColors    = std::get<3>(variables);
    Variable dLossDOpacities = std::get<4>(variables);

    return {
        dLossDMeans2d,
        dLossDConics,
        dLossDColors,
        dLossDOpacities,
        Variable(),
        Variable(),
        Variable(),
        Variable(),
        Variable(),
        Variable(),
        Variable(),
        Variable(),
    };
}

} // namespace fvdb::detail::autograd
