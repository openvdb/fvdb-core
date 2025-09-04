// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_DOWNSAMPLEGRIDAVGPOOL_H
#define FVDB_DETAIL_OPS_DOWNSAMPLEGRIDAVGPOOL_H

#include <fvdb/detail/GridBatchImpl.h>

#include <nanovdb/NanoVDB.h>

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

template <torch::DeviceType>
torch::Tensor dispatchDownsampleGridAvgPool(const GridBatchImpl &fineBatchHdl,
                                            const GridBatchImpl &coarseBatchHdl,
                                            const torch::Tensor &fineData,
                                            nanovdb::Coord poolingFactor,
                                            nanovdb::Coord stride);

template <torch::DeviceType>
torch::Tensor dispatchDownsampleGridAvgPoolBackward(const GridBatchImpl &coarseBatchHdl,
                                                    const GridBatchImpl &fineBatchHdl,
                                                    const torch::Tensor &fineData,
                                                    const torch::Tensor &coarseGradOut,
                                                    nanovdb::Coord poolingFactor,
                                                    nanovdb::Coord stride);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_DOWNSAMPLEGRIDAVGPOOL_H
