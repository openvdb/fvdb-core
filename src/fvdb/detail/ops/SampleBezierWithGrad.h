// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_SAMPLEBEZIERWITHGRAD_H
#define FVDB_DETAIL_OPS_SAMPLEBEZIERWITHGRAD_H

#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/GridBatchData.h>

#include <torch/types.h>

#include <vector>

namespace fvdb {
namespace detail {
namespace ops {

std::vector<torch::Tensor> sampleBezierWithGrad(const GridBatchData &batchHdl,
                                                    const JaggedTensor &points,
                                                    const torch::Tensor &gridData);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_SAMPLEBEZIERWITHGRAD_H
