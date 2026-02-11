// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_ACTIVEVOXELSINBOUNDSMASK_H
#define FVDB_DETAIL_OPS_ACTIVEVOXELSINBOUNDSMASK_H

#include <fvdb/JaggedTensor.h>

#include <nanovdb/NanoVDB.h>

#include <vector>

namespace fvdb {
namespace detail {

class GridBatchImpl;

namespace ops {

/// @brief Return a boolean mask of active voxels that lie inside per-batch bounding boxes.
/// Device dispatch is handled internally -- no template parameter needed.
JaggedTensor activeVoxelsInBoundsMask(GridBatchImpl const &grid,
                                      std::vector<nanovdb::Coord> const &bboxMins,
                                      std::vector<nanovdb::Coord> const &bboxMaxs);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_ACTIVEVOXELSINBOUNDSMASK_H
