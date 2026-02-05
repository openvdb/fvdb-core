// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/autograd/GaussianRasterizeFromWorld3DGS.h>
#include <fvdb/detail/ops/gsplat/GaussianRasterizeFromWorld3DGSBackward.h>
#include <fvdb/detail/ops/gsplat/GaussianRasterizeFromWorld3DGSForward.h>
#include <fvdb/detail/utils/Nvtx.h>
#include <fvdb/detail/utils/Utils.h>

#include <vector>

namespace fvdb::detail::autograd {

RasterizeGaussiansToPixelsFromWorld3DGS::VariableList
RasterizeGaussiansToPixelsFromWorld3DGS::forward(
    RasterizeGaussiansToPixelsFromWorld3DGS::AutogradContext *ctx,
    const RasterizeGaussiansToPixelsFromWorld3DGS::Variable &means,
    const RasterizeGaussiansToPixelsFromWorld3DGS::Variable &quats,
    const RasterizeGaussiansToPixelsFromWorld3DGS::Variable &logScales,
    const RasterizeGaussiansToPixelsFromWorld3DGS::Variable &features,
    const RasterizeGaussiansToPixelsFromWorld3DGS::Variable &opacities,
    const RasterizeGaussiansToPixelsFromWorld3DGS::Variable &worldToCamMatricesStart,
    const RasterizeGaussiansToPixelsFromWorld3DGS::Variable &worldToCamMatricesEnd,
    const RasterizeGaussiansToPixelsFromWorld3DGS::Variable &projectionMatrices,
    const RasterizeGaussiansToPixelsFromWorld3DGS::Variable &distortionCoeffs,
    const fvdb::detail::ops::RollingShutterType rollingShutterType,
    const fvdb::detail::ops::CameraModel cameraModel,
    const uint32_t imageWidth,
    const uint32_t imageHeight,
    const uint32_t imageOriginW,
    const uint32_t imageOriginH,
    const uint32_t tileSize,
    const RasterizeGaussiansToPixelsFromWorld3DGS::Variable &tileOffsets,
    const RasterizeGaussiansToPixelsFromWorld3DGS::Variable &tileGaussianIds,
    std::optional<RasterizeGaussiansToPixelsFromWorld3DGS::Variable> backgrounds,
    std::optional<RasterizeGaussiansToPixelsFromWorld3DGS::Variable> masks) {
    FVDB_FUNC_RANGE_WITH_NAME("RasterizeGaussiansToPixelsFromWorld3DGS::forward");

    auto outputs = FVDB_DISPATCH_KERNEL_DEVICE(means.device(), [&]() {
        return ops::dispatchGaussianRasterizeFromWorld3DGSForward<DeviceTag>(
            means,
            quats,
            logScales,
            features,
            opacities,
            worldToCamMatricesStart,
            worldToCamMatricesEnd,
            projectionMatrices,
            distortionCoeffs,
            rollingShutterType,
            cameraModel,
            imageWidth,
            imageHeight,
            imageOriginW,
            imageOriginH,
            tileSize,
            tileOffsets,
            tileGaussianIds,
            backgrounds,
            masks);
    });

    Variable renderedFeatures = std::get<0>(outputs);
    Variable renderedAlphas   = std::get<1>(outputs);
    Variable lastIds          = std::get<2>(outputs);

    std::vector<Variable> toSave = {means,
                                    quats,
                                    logScales,
                                    features,
                                    opacities,
                                    worldToCamMatricesStart,
                                    worldToCamMatricesEnd,
                                    projectionMatrices,
                                    distortionCoeffs,
                                    tileOffsets,
                                    tileGaussianIds,
                                    renderedAlphas,
                                    lastIds};
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

    ctx->saved_data["imageWidth"]   = (int64_t)imageWidth;
    ctx->saved_data["imageHeight"]  = (int64_t)imageHeight;
    ctx->saved_data["imageOriginW"] = (int64_t)imageOriginW;
    ctx->saved_data["imageOriginH"] = (int64_t)imageOriginH;
    ctx->saved_data["tileSize"]     = (int64_t)tileSize;
    ctx->saved_data["cameraModel"]  = (int64_t)cameraModel;
    ctx->saved_data["rollingShutterType"] = (int64_t)rollingShutterType;

    return {renderedFeatures, renderedAlphas};
}

RasterizeGaussiansToPixelsFromWorld3DGS::VariableList
RasterizeGaussiansToPixelsFromWorld3DGS::backward(
    RasterizeGaussiansToPixelsFromWorld3DGS::AutogradContext *ctx,
    RasterizeGaussiansToPixelsFromWorld3DGS::VariableList gradOutput) {
    FVDB_FUNC_RANGE_WITH_NAME("RasterizeGaussiansToPixelsFromWorld3DGS::backward");

    Variable dLossDRenderedFeatures = gradOutput.at(0);
    Variable dLossDRenderedAlphas   = gradOutput.at(1);
    if (dLossDRenderedFeatures.defined()) {
        dLossDRenderedFeatures = dLossDRenderedFeatures.contiguous();
    }
    if (dLossDRenderedAlphas.defined()) {
        dLossDRenderedAlphas = dLossDRenderedAlphas.contiguous();
    }

    VariableList saved = ctx->get_saved_variables();
    Variable means     = saved.at(0);
    Variable quats     = saved.at(1);
    Variable logScales = saved.at(2);
    Variable features  = saved.at(3);
    Variable opacities = saved.at(4);
    Variable worldToCamMatricesStart = saved.at(5);
    Variable worldToCamMatricesEnd   = saved.at(6);
    Variable projectionMatrices      = saved.at(7);
    Variable distortionCoeffs        = saved.at(8);
    Variable tileOffsets             = saved.at(9);
    Variable tileGaussianIds         = saved.at(10);
    Variable renderedAlphas          = saved.at(11);
    Variable lastIds                 = saved.at(12);

    const bool hasBackgrounds                = ctx->saved_data["has_backgrounds"].toBool();
    const bool hasMasks                      = ctx->saved_data["has_masks"].toBool();
    std::optional<torch::Tensor> backgrounds = std::nullopt;
    std::optional<torch::Tensor> masks       = std::nullopt;
    int64_t optIdx                           = 13;
    if (hasBackgrounds) {
        backgrounds = saved.at(optIdx++);
    }
    if (hasMasks) {
        masks = saved.at(optIdx++);
    }

    const uint32_t imageWidth   = (uint32_t)ctx->saved_data["imageWidth"].toInt();
    const uint32_t imageHeight  = (uint32_t)ctx->saved_data["imageHeight"].toInt();
    const uint32_t imageOriginW = (uint32_t)ctx->saved_data["imageOriginW"].toInt();
    const uint32_t imageOriginH = (uint32_t)ctx->saved_data["imageOriginH"].toInt();
    const uint32_t tileSize     = (uint32_t)ctx->saved_data["tileSize"].toInt();
    const auto cameraModel =
        static_cast<fvdb::detail::ops::CameraModel>(ctx->saved_data["cameraModel"].toInt());
    const auto rollingShutterType = static_cast<fvdb::detail::ops::RollingShutterType>(
        ctx->saved_data["rollingShutterType"].toInt());

    auto grads = FVDB_DISPATCH_KERNEL_DEVICE(means.device(), [&]() {
        return ops::dispatchGaussianRasterizeFromWorld3DGSBackward<DeviceTag>(
            means,
            quats,
            logScales,
            features,
            opacities,
            worldToCamMatricesStart,
            worldToCamMatricesEnd,
            projectionMatrices,
            distortionCoeffs,
            rollingShutterType,
            cameraModel,
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
            backgrounds,
            masks);
    });

    Variable dMeans     = std::get<0>(grads);
    Variable dQuats     = std::get<1>(grads);
    Variable dLogScales = std::get<2>(grads);
    Variable dFeatures  = std::get<3>(grads);
    Variable dOpacities = std::get<4>(grads);

    // Return gradients in the same order as forward inputs.
    return {dMeans,
            dQuats,
            dLogScales,
            dFeatures,
            dOpacities,
            Variable(),
            Variable(),
            Variable(),
            Variable(),
            Variable(),
            Variable(),
            Variable(),
            Variable(),
            Variable(),
            Variable(),
            Variable(),
            Variable(),
            Variable(),
            Variable(),
            Variable(),
            Variable()};
}

} // namespace fvdb::detail::autograd

