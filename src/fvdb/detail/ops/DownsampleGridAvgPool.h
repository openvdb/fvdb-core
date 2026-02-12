// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_DOWNSAMPLEGRIDAVGPOOL_H
#define FVDB_DETAIL_OPS_DOWNSAMPLEGRIDAVGPOOL_H

#include <nanovdb/NanoVDB.h>

#include <torch/types.h>

namespace fvdb {
namespace detail {

class GridBatchImpl;

namespace ops {

/// @brief Average-pool voxel features from a fine grid into a coarse grid.
torch::Tensor downsampleGridAvgPool(GridBatchImpl const &fineGrid,
                                    GridBatchImpl const &coarseGrid,
                                    torch::Tensor fineData,
                                    nanovdb::Coord poolingFactor,
                                    nanovdb::Coord stride);

/// @brief Backward pass for average-pool downsampling.
torch::Tensor downsampleGridAvgPoolBackward(GridBatchImpl const &coarseGrid,
                                            GridBatchImpl const &fineGrid,
                                            torch::Tensor fineData,
                                            torch::Tensor coarseGradOut,
                                            nanovdb::Coord poolingFactor,
                                            nanovdb::Coord stride);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_DOWNSAMPLEGRIDAVGPOOL_H
