// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_POINTSINGRID_H
#define FVDB_DETAIL_OPS_POINTSINGRID_H

#include <fvdb/JaggedTensor.h>

namespace fvdb {
namespace detail {

class GridBatchImpl;

namespace ops {

/// @brief Check which points fall inside active grid voxels.
/// @param grid   The grid batch to test against.
/// @param points JaggedTensor of [N, 3] floating-point coordinates.
/// @return JaggedTensor of [N] bool mask (true if point is in an active voxel).
JaggedTensor pointsInGrid(GridBatchImpl const &grid, JaggedTensor const &points);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_POINTSINGRID_H
