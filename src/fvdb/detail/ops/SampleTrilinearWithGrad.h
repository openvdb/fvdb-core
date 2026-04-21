// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_SAMPLETRILINEARWITHGRAD_H
#define FVDB_DETAIL_OPS_SAMPLETRILINEARWITHGRAD_H

#include <fvdb/GridBatchData.h>
#include <fvdb/JaggedTensor.h>

#include <torch/types.h>

#include <vector>

namespace fvdb {
namespace detail {
namespace ops {

std::vector<torch::Tensor> sampleTrilinearWithGrad(const GridBatchData &batchHdl,
                                                   const JaggedTensor &points,
                                                   const torch::Tensor &gridData);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_SAMPLETRILINEARWITHGRAD_H
