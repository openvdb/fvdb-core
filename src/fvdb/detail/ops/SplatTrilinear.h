// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_SPLATTRILINEAR_H
#define FVDB_DETAIL_OPS_SPLATTRILINEAR_H

#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/GridBatchData.h>

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

torch::Tensor splatTrilinear(const GridBatchData &batchHdl,
                             const JaggedTensor &points,
                             const torch::Tensor &pointsData);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_SPLATTRILINEAR_H
