// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/autograd/GaussianRasterizeSparse.h>
#include <fvdb/detail/ops/gsplat/GaussianRasterizeBackward.h>
#include <fvdb/detail/ops/gsplat/GaussianRasterizeForward.h>
#include <fvdb/detail/utils/Nvtx.h>
#include <fvdb/detail/utils/Utils.h>

namespace fvdb::detail::autograd {

RasterizeGaussiansToPixelsSparse::VariableList
RasterizeGaussiansToPixelsSparse::forward(
    RasterizeGaussiansToPixelsSparse::AutogradContext *ctx,
    const JaggedTensor &pixelsToRender,                          // [C, num_pixels, 2]
    const RasterizeGaussiansToPixelsSparse::Variable &means2d,   // [C, N, 2]
    const RasterizeGaussiansToPixelsSparse::Variable &conics,    // [C, N, 3]
    const RasterizeGaussiansToPixelsSparse::Variable &colors,    // [C, N, 3]
    const RasterizeGaussiansToPixelsSparse::Variable &opacities, // [N]
    const uint32_t imageWidth,
    const uint32_t imageHeight,
    const uint32_t imageOriginW,
    const uint32_t imageOriginH,
    const uint32_t tileSize,
    const RasterizeGaussiansToPixelsSparse::Variable
        &tileOffsets, // [C, tile_height, tile_width] (dense) or [num_active_tiles + 1] (sparse)
    const RasterizeGaussiansToPixelsSparse::Variable &tileGaussianIds, // [n_isects]
    const RasterizeGaussiansToPixelsSparse::Variable &activeTiles,     // [num_active_tiles]
    const RasterizeGaussiansToPixelsSparse::Variable
        &tilePixelMask, // [num_active_tiles, tileSize, tileSize]
    const RasterizeGaussiansToPixelsSparse::Variable &tilePixelCumsum, // [num_active_tiles + 1]
    const RasterizeGaussiansToPixelsSparse::Variable &pixelMap,        // [num_pixels]
    const bool absgrad) {
    FVDB_FUNC_RANGE_WITH_NAME("RasterizeGaussiansToPixelsSparse::forward");

    auto variables              = FVDB_DISPATCH_KERNEL(means2d.device(), [&]() {
        return ops::dispatchGaussianSparseRasterizeForward<DeviceTag>(pixelsToRender,
                                                                      means2d,
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
                                                                      activeTiles,
                                                                      tilePixelMask,
                                                                      tilePixelCumsum,
                                                                      pixelMap,
                                                                      at::nullopt); // backgrounds
    });
    JaggedTensor renderedColors = std::get<0>(variables);
    JaggedTensor renderedAlphas = std::get<1>(variables);
    JaggedTensor lastIds        = std::get<2>(variables);

    const auto joffsets      = pixelsToRender.joffsets();
    const auto jidx          = pixelsToRender.jidx();
    const auto jlidx         = pixelsToRender.jlidx();
    const auto numOuterLists = pixelsToRender.num_outer_lists();

    ctx->save_for_backward({means2d,
                            conics,
                            colors,
                            opacities,
                            tileOffsets,
                            tileGaussianIds,
                            pixelsToRender.jdata(),
                            renderedColors.jdata(),
                            renderedAlphas.jdata(),
                            lastIds.jdata(),
                            joffsets,
                            jidx,
                            jlidx,
                            activeTiles,
                            tilePixelMask,
                            tilePixelCumsum,
                            pixelMap});
    ctx->saved_data["imageWidth"]    = (int64_t)imageWidth;
    ctx->saved_data["imageHeight"]   = (int64_t)imageHeight;
    ctx->saved_data["tileSize"]      = (int64_t)tileSize;
    ctx->saved_data["imageOriginW"]  = (int64_t)imageOriginW;
    ctx->saved_data["imageOriginH"]  = (int64_t)imageOriginH;
    ctx->saved_data["numOuterLists"] = (int64_t)numOuterLists;
    ctx->saved_data["absgrad"]       = absgrad;

    return {renderedColors.jdata(), renderedAlphas.jdata()};
}

RasterizeGaussiansToPixelsSparse::VariableList
RasterizeGaussiansToPixelsSparse::backward(
    RasterizeGaussiansToPixelsSparse::AutogradContext *ctx,
    RasterizeGaussiansToPixelsSparse::VariableList gradOutput) {
    FVDB_FUNC_RANGE_WITH_NAME("RasterizeGaussiansToPixelsSparse::backward");
    Variable dLossDRenderedFeaturesJData = gradOutput.at(0);
    Variable dLossDRenderedAlphasJData   = gradOutput.at(1);

    // ensure the gradients are contiguous if they are not None
    if (dLossDRenderedFeaturesJData.defined()) {
        dLossDRenderedFeaturesJData = dLossDRenderedFeaturesJData.contiguous();
    }
    if (dLossDRenderedAlphasJData.defined()) {
        dLossDRenderedAlphasJData = dLossDRenderedAlphasJData.contiguous();
    }

    VariableList saved           = ctx->get_saved_variables();
    Variable means2d             = saved.at(0);
    Variable conics              = saved.at(1);
    Variable features            = saved.at(2);
    Variable opacities           = saved.at(3);
    Variable tileOffsets         = saved.at(4);
    Variable tileGaussianIds     = saved.at(5);
    Variable pixelsToRenderJData = saved.at(6);
    Variable renderedColorsJData = saved.at(7);
    Variable renderedAlphasJData = saved.at(8);
    Variable lastIdsJData        = saved.at(9);
    Variable joffsets            = saved.at(10);
    Variable jidx                = saved.at(11);
    Variable jlidx               = saved.at(12);
    Variable activeTiles         = saved.at(13);
    Variable tilePixelMask       = saved.at(14);
    Variable tilePixelCumsum     = saved.at(15);
    Variable pixelMap            = saved.at(16);

    const int imageWidth        = (int)ctx->saved_data["imageWidth"].toInt();
    const int imageHeight       = (int)ctx->saved_data["imageHeight"].toInt();
    const int tileSize          = (int)ctx->saved_data["tileSize"].toInt();
    const int imageOriginW      = (int)ctx->saved_data["imageOriginW"].toInt();
    const int imageOriginH      = (int)ctx->saved_data["imageOriginH"].toInt();
    const int64_t numOuterLists = ctx->saved_data["numOuterLists"].toInt();
    const bool absgrad          = ctx->saved_data["absgrad"].toBool();

    auto pixelsToRender = JaggedTensor::from_jdata_joffsets_jidx_and_lidx_unsafe(
        pixelsToRenderJData, joffsets, jidx, jlidx, numOuterLists);
    auto renderedAlphas = pixelsToRender.jagged_like(renderedAlphasJData);
    auto lastIds        = pixelsToRender.jagged_like(lastIdsJData);

    auto dLossDRenderedFeatures = pixelsToRender.jagged_like(dLossDRenderedFeaturesJData);
    auto dLossDRenderedAlphas   = pixelsToRender.jagged_like(dLossDRenderedAlphasJData);

    auto variables = FVDB_DISPATCH_KERNEL(means2d.device(), [&]() {
        return ops::dispatchGaussianSparseRasterizeBackward<DeviceTag>(pixelsToRender,
                                                                       means2d,
                                                                       conics,
                                                                       features,
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
                                                                       dLossDRenderedFeatures,
                                                                       dLossDRenderedAlphas,
                                                                       activeTiles,
                                                                       tilePixelMask,
                                                                       tilePixelCumsum,
                                                                       pixelMap,
                                                                       absgrad);
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

    return {
        Variable(),      // pixelsToRender
        dLossDMeans2d,   // means2d
        dLossDConics,    // conics
        dLossDColors,    // features
        dLossDOpacities, // opacities
        Variable(),      // imageWidth
        Variable(),      // imageHeight
        Variable(),      // imageOriginW
        Variable(),      // imageOriginH
        Variable(),      // tileSize
        Variable(),      // tileOffsets
        Variable(),      // tileGaussianIds
        Variable(),      // activeTiles
        Variable(),      // tilePixelMask
        Variable(),      // tilePixelCumsum
        Variable(),      // pixelMap
        Variable(),      // absgrad
    };
}

} // namespace fvdb::detail::autograd
