// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_SAMPLEBEZIERWITHGRADBACKWARD_H
#define FVDB_DETAIL_OPS_SAMPLEBEZIERWITHGRADBACKWARD_H

#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/GridBatchImpl.h>

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

torch::Tensor sampleBezierWithGradBackward(const GridBatchImpl &batchHdl,
                                               const JaggedTensor &points,
                                               const torch::Tensor &gradOutFeatures,
                                               const torch::Tensor &gradOutGradFeatures,
                                               const torch::Tensor &data);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_SAMPLEBEZIERWITHGRADBACKWARD_H
