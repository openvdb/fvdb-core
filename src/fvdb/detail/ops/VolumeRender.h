// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_VOLUMERENDER_H
#define FVDB_DETAIL_OPS_VOLUMERENDER_H

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

template <torch::DeviceType>
void dispatchVolumeRender(const torch::Tensor sigmas,
                          const torch::Tensor rgbs,
                          const torch::Tensor deltas,
                          const torch::Tensor ts,
                          const torch::Tensor raysAcc,
                          const float opacityThreshold,
                          torch::Tensor &outOpacity,
                          torch::Tensor &outDepth,
                          torch::Tensor &outRgb,
                          torch::Tensor &outWs,
                          torch::Tensor &outTotalSamples);

template <torch::DeviceType>
void dispatchVolumeRenderBackward(const torch::Tensor dLdOpacity,
                                  const torch::Tensor dLdDepth,
                                  const torch::Tensor dLdRgb,
                                  const torch::Tensor dLdWs,
                                  const torch::Tensor sigmas,
                                  const torch::Tensor rgbs,
                                  const torch::Tensor ws,
                                  const torch::Tensor deltas,
                                  const torch::Tensor ts,
                                  const torch::Tensor raysAcc,
                                  const torch::Tensor opacity,
                                  const torch::Tensor depth,
                                  const torch::Tensor rgb,
                                  const float opacityThreshold,
                                  torch::Tensor &outDLdSigmas,
                                  torch::Tensor &outDLdRbgs);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_VOLUMERENDER_H
