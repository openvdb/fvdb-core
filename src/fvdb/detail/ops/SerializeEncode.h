// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_SERIALIZEENCODE_H
#define FVDB_DETAIL_OPS_SERIALIZEENCODE_H

#include <fvdb/JaggedTensor.h>
#include <fvdb/Types.h>

#include <nanovdb/NanoVDB.h>

namespace fvdb {
namespace detail {

class GridBatchImpl;

namespace ops {

/// @brief Get the space-filling curve codes for active voxels in a batch of grids.
/// @param grid       The batch of grids.
/// @param order_type The type of space-filling curve to use for encoding.
/// @param offset     Offset to apply to voxel coordinates before encoding.
/// @return A JaggedTensor of int64 space-filling curve codes for active voxels.
JaggedTensor serializeEncode(GridBatchImpl const &grid,
                             SpaceFillingCurveType order_type,
                             nanovdb::Coord const &offset);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_SERIALIZEENCODE_H
