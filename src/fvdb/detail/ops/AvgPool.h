// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_AVGPOOL_H
#define FVDB_DETAIL_OPS_AVGPOOL_H

#include <fvdb/detail/GridBatchData.h>

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

torch::Tensor avgPool(const GridBatchData &fineBatchHdl,
                                    const GridBatchData &coarseBatchHdl,
                                    const torch::Tensor &fineData,
                                    nanovdb::Coord poolingFactor,
                                    nanovdb::Coord stride);

torch::Tensor avgPoolBackward(const GridBatchData &coarseBatchHdl,
                                            const GridBatchData &fineBatchHdl,
                                            const torch::Tensor &fineData,
                                            const torch::Tensor &coarseGradOut,
                                            nanovdb::Coord poolingFactor,
                                            nanovdb::Coord stride);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_AVGPOOL_H
