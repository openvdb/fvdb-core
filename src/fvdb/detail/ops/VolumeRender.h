// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_VOLUMERENDER_H
#define FVDB_DETAIL_OPS_VOLUMERENDER_H

#include <torch/types.h>

#include <tuple>

namespace fvdb {
namespace detail {
namespace ops {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
volumeRender(const torch::Tensor &sigmas,
             const torch::Tensor &rgbs,
             const torch::Tensor &deltaTs,
             const torch::Tensor &ts,
             const torch::Tensor &jOffsets,
             double tsmtThreshold);

std::tuple<torch::Tensor, torch::Tensor> volumeRenderBackward(const torch::Tensor &dLdOpacity,
                                                              const torch::Tensor &dLdDepth,
                                                              const torch::Tensor &dLdRgb,
                                                              const torch::Tensor &dLdWs,
                                                              const torch::Tensor &sigmas,
                                                              const torch::Tensor &rgbs,
                                                              const torch::Tensor &ws,
                                                              const torch::Tensor &deltas,
                                                              const torch::Tensor &ts,
                                                              const torch::Tensor &jOffsets,
                                                              const torch::Tensor &opacity,
                                                              const torch::Tensor &depth,
                                                              const torch::Tensor &rgb,
                                                              float tsmtThreshold);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_VOLUMERENDER_H
