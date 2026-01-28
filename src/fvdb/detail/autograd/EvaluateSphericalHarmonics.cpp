// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/autograd/EvaluateSphericalHarmonics.h>
#include <fvdb/detail/ops/gsplat/GaussianSphericalHarmonicsBackward.h>
#include <fvdb/detail/ops/gsplat/GaussianSphericalHarmonicsForward.h>
#include <fvdb/detail/utils/Nvtx.h>
#include <fvdb/detail/utils/Utils.h>

namespace fvdb {
namespace detail {
namespace autograd {

EvaluateSphericalHarmonics::VariableList
EvaluateSphericalHarmonics::forward(
    EvaluateSphericalHarmonics::AutogradContext *ctx,
    const ssize_t shDegreeToUse,
    const size_t numCameras,
    const std::optional<EvaluateSphericalHarmonics::Variable>
        viewDirections,                                                   // [C, N, 3] (optional)
    const EvaluateSphericalHarmonics::Variable &sh0Coeffs,                // [N, 1, D]
    const std::optional<EvaluateSphericalHarmonics::Variable> &shNCoeffs, // [N, K-1, D]
    const std::optional<EvaluateSphericalHarmonics::Variable> &radii      // [C, N] (optional)
) {
    FVDB_FUNC_RANGE_WITH_NAME("EvaluateSphericalHarmonics::forward");
    const Variable viewDirectionsValue = viewDirections.value_or(torch::Tensor());
    const Variable shNCoeffsValue      = shNCoeffs.value_or(torch::Tensor());
    const Variable radiiValue          = radii.value_or(torch::Tensor());
    const Variable renderQuantities    = FVDB_DISPATCH_KERNEL(sh0Coeffs.device(), [&]() {
        return ops::dispatchSphericalHarmonicsForward<DeviceTag>(
            shDegreeToUse, numCameras, viewDirectionsValue, sh0Coeffs, shNCoeffsValue, radiiValue);
    });
    // only save radii if defined to avoid device access issues
    const bool hasRadii = radii.has_value() && radii.value().defined();
    if (hasRadii) {
        ctx->save_for_backward({viewDirectionsValue, shNCoeffsValue, radiiValue});
    } else {
        ctx->save_for_backward({viewDirectionsValue, shNCoeffsValue});
    }
    ctx->saved_data["shDegreeToUse"] = static_cast<int64_t>(shDegreeToUse);
    ctx->saved_data["numCameras"]    = static_cast<int64_t>(numCameras);
    ctx->saved_data["numGaussians"]  = static_cast<int64_t>(sh0Coeffs.size(0));
    ctx->saved_data["hasRadii"]      = hasRadii;
    return {renderQuantities};
}

EvaluateSphericalHarmonics::VariableList
EvaluateSphericalHarmonics::backward(EvaluateSphericalHarmonics::AutogradContext *ctx,
                                     EvaluateSphericalHarmonics::VariableList gradOutput) {
    FVDB_FUNC_RANGE_WITH_NAME("EvaluateSphericalHarmonics::backward");
    Variable dLossDColors = gradOutput.at(0);

    // ensure the gradients are contiguous if they are not None
    auto const dLossdColors =
        gradOutput.at(0).defined() ? gradOutput.at(0).contiguous() : gradOutput.at(0);

    VariableList saved = ctx->get_saved_variables();
    Variable viewDirs  = saved.at(0);
    Variable shNCoeffs = saved.at(1);

    const int shDegreeToUse          = static_cast<int>(ctx->saved_data["shDegreeToUse"].toInt());
    const int numCameras             = static_cast<int>(ctx->saved_data["numCameras"].toInt());
    const int numGaussians           = static_cast<int>(ctx->saved_data["numGaussians"].toInt());
    const bool hasRadii              = ctx->saved_data["hasRadii"].toBool();
    const bool computeDLossDViewDirs = ctx->needs_input_grad(1);

    // Radii is only saved if it was defined in forward
    Variable radii = hasRadii ? saved.at(2) : torch::Tensor();

    auto variables           = FVDB_DISPATCH_KERNEL(dLossdColors.device(), [&]() {
        return ops::dispatchSphericalHarmonicsBackward<DeviceTag>(shDegreeToUse,
                                                                  numCameras,
                                                                  numGaussians,
                                                                  viewDirs,
                                                                  shNCoeffs,
                                                                  dLossDColors,
                                                                  radii,
                                                                  computeDLossDViewDirs);
    });
    Variable dLossDSh0Coeffs = std::get<0>(variables);
    Variable dLossDShNCoeffs = std::get<1>(variables);
    Variable dLossDViewDirs  = std::get<2>(variables);

    return {Variable(), Variable(), dLossDViewDirs, dLossDSh0Coeffs, dLossDShNCoeffs, Variable()};
}

} // namespace autograd
} // namespace detail
} // namespace fvdb
