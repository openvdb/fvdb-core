// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_COORDSINGRID_H
#define FVDB_DETAIL_OPS_COORDSINGRID_H

#include <fvdb/JaggedTensor.h>

namespace fvdb {
namespace detail {

class GridBatchImpl;

namespace ops {

/// @brief Check which ijk coordinates are active voxels in the grid.
/// @param grid  The grid batch to test against.
/// @param ijk   JaggedTensor of [N, 3] integer coordinates.
/// @return JaggedTensor of [N] bool mask (true if the voxel at ijk is active).
JaggedTensor coordsInGrid(GridBatchImpl const &grid, JaggedTensor ijk);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_COORDSINGRID_H
