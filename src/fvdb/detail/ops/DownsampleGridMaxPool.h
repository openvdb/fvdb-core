// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_DOWNSAMPLEGRIDMAXPOOL_H
#define FVDB_DETAIL_OPS_DOWNSAMPLEGRIDMAXPOOL_H

#include <fvdb/detail/GridBatchImpl.h>

#include <nanovdb/NanoVDB.h>

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

torch::Tensor downsampleGridMaxPool(const GridBatchImpl &fineBatchHdl,
                                    const GridBatchImpl &coarseBatchHdl,
                                    const torch::Tensor &fineData,
                                    nanovdb::Coord poolingFactor,
                                    nanovdb::Coord stride);

torch::Tensor downsampleGridMaxPoolBackward(const GridBatchImpl &coarseBatchHdl,
                                            const GridBatchImpl &fineBatchHdl,
                                            const torch::Tensor &fineData,
                                            const torch::Tensor &coarseGradOut,
                                            nanovdb::Coord poolingFactor,
                                            nanovdb::Coord stride);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_DOWNSAMPLEGRIDMAXPOOL_H
