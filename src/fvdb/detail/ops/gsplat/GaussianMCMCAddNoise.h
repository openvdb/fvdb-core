// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef FVDB_DETAIL_OPS_GSPLAT_GAUSSIANMCMCADDNOISE_H
#define FVDB_DETAIL_OPS_GSPLAT_GAUSSIANMCMCADDNOISE_H

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

template <torch::DeviceType DeviceType>
void dispatchGaussianMCMCAddNoise(torch::Tensor &means,                // [N, 3] input/output
                                  const torch::Tensor &logScales,      // [N]
                                  const torch::Tensor &logitOpacities, // [N]
                                  const torch::Tensor &quats,          // [N, 4]
                                  const float noiseScale,
                                  const float t,
                                  const float k);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_GSPLAT_GAUSSIANMCMCADDNOISE_H
