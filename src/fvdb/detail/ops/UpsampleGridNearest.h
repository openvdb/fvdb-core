// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_UPSAMPLEGRIDNEAREST_H
#define FVDB_DETAIL_OPS_UPSAMPLEGRIDNEAREST_H

#include <nanovdb/NanoVDB.h>

#include <torch/types.h>

namespace fvdb {
namespace detail {

class GridBatchImpl;

namespace ops {

/// @brief Nearest-neighbor upsample from coarse grid to fine grid.
torch::Tensor upsampleGridNearest(GridBatchImpl const &coarseGrid,
                                  GridBatchImpl const &fineGrid,
                                  torch::Tensor coarseData,
                                  nanovdb::Coord upsamplingFactor);

/// @brief Backward pass for nearest-neighbor upsample.
torch::Tensor upsampleGridNearestBackward(GridBatchImpl const &fineGrid,
                                          GridBatchImpl const &coarseGrid,
                                          torch::Tensor gradOut,
                                          torch::Tensor coarseData,
                                          nanovdb::Coord upsamplingFactor);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_UPSAMPLEGRIDNEAREST_H
