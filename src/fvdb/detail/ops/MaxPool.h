// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_MAXPOOL_H
#define FVDB_DETAIL_OPS_MAXPOOL_H

#include <fvdb/detail/GridBatchImpl.h>

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

torch::Tensor maxPool(const GridBatchImpl &fineBatchHdl,
                                    const GridBatchImpl &coarseBatchHdl,
                                    const torch::Tensor &fineData,
                                    nanovdb::Coord poolingFactor,
                                    nanovdb::Coord stride);

torch::Tensor maxPoolBackward(const GridBatchImpl &coarseBatchHdl,
                                            const GridBatchImpl &fineBatchHdl,
                                            const torch::Tensor &fineData,
                                            const torch::Tensor &coarseGradOut,
                                            nanovdb::Coord poolingFactor,
                                            nanovdb::Coord stride);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_MAXPOOL_H
