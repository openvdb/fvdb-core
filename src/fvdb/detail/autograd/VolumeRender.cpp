// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/autograd/VolumeRender.h>
#include <fvdb/detail/ops/VolumeRender.h>
#include <fvdb/detail/utils/Utils.h>

namespace fvdb {
namespace detail {
namespace autograd {

VolumeRender::variable_list
VolumeRender::forward(VolumeRender::AutogradContext *ctx,
                      const VolumeRender::Variable &sigmas,
                      const VolumeRender::Variable &rgbs,
                      const VolumeRender::Variable &deltaTs,
                      const VolumeRender::Variable &ts,
                      const VolumeRender::Variable &jOffsets,
                      double tsmtThreshold) {
    auto [outRgb, outDepth, outOpacity, outWs, outTotalSamples] =
        ops::volumeRender(sigmas, rgbs, deltaTs, ts, jOffsets, tsmtThreshold);

    ctx->saved_data["tsmtThreshold"] = tsmtThreshold;

    ctx->save_for_backward(
        {sigmas, rgbs, deltaTs, ts, jOffsets, outOpacity, outDepth, outRgb, outWs});

    return {outRgb, outDepth, outOpacity, outWs, outTotalSamples};
}

VolumeRender::variable_list
VolumeRender::backward(VolumeRender::AutogradContext *ctx,
                       VolumeRender::variable_list grad_output) {
    Variable dLdRgb     = grad_output.at(0);
    Variable dLdDepth   = grad_output.at(1);
    Variable dLdOpacity = grad_output.at(2);
    Variable dLdWs      = grad_output.at(3);

    variable_list saved = ctx->get_saved_variables();
    Variable sigmas     = saved.at(0);
    Variable rgbs       = saved.at(1);
    Variable deltaTs    = saved.at(2);
    Variable ts         = saved.at(3);
    Variable jOffsets   = saved.at(4);

    Variable outOpacity        = saved.at(5);
    Variable outDepth          = saved.at(6);
    Variable outRgb            = saved.at(7);
    Variable outWs             = saved.at(8);
    const double tsmtThreshold = ctx->saved_data["tsmtThreshold"].toDouble();

    auto [dLdSigmas, dLdRgbs] = ops::volumeRenderBackward(dLdOpacity,
                                                          dLdDepth,
                                                          dLdRgb,
                                                          dLdWs,
                                                          sigmas,
                                                          rgbs,
                                                          outWs,
                                                          deltaTs,
                                                          ts,
                                                          jOffsets,
                                                          outOpacity,
                                                          outDepth,
                                                          outRgb,
                                                          tsmtThreshold);

    return {dLdSigmas, dLdRgbs, torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()};
}

} // namespace autograd
} // namespace detail
} // namespace fvdb
