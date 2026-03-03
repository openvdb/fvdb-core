// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_UPSAMPLEGRIDNEAREST_H
#define FVDB_DETAIL_OPS_UPSAMPLEGRIDNEAREST_H

#include <fvdb/detail/GridBatchImpl.h>
#include <fvdb/detail/utils/Utils.h>

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

torch::Tensor upsampleGridNearest(const GridBatchImpl &coarseBatchHdl,
                                  const GridBatchImpl &fineBatchHdl,
                                  const torch::Tensor &coarseData,
                                  nanovdb::Coord upsamplingFactor);

torch::Tensor upsampleGridNearestBackward(const GridBatchImpl &fineBatchHdl,
                                          const GridBatchImpl &coarseBatchHdl,
                                          const torch::Tensor &gradOut,
                                          const torch::Tensor &coarseData,
                                          nanovdb::Coord upsamplingFactor);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_UPSAMPLEGRIDNEAREST_H
