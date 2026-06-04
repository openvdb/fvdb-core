// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_REFINE_H
#define FVDB_DETAIL_OPS_REFINE_H

#include <fvdb/GridBatchData.h>
#include <fvdb/detail/utils/Utils.h>

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

torch::Tensor refine(const GridBatchData &coarseBatchHdl,
                     const GridBatchData &fineBatchHdl,
                     const torch::Tensor &coarseData,
                     nanovdb::Coord upsamplingFactor);

torch::Tensor refineBackward(const GridBatchData &fineBatchHdl,
                             const GridBatchData &coarseBatchHdl,
                             const torch::Tensor &gradOut,
                             const torch::Tensor &coarseData,
                             nanovdb::Coord upsamplingFactor);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_REFINE_H
