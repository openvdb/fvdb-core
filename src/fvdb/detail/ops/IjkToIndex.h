// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_IJKTOINDEX_H
#define FVDB_DETAIL_OPS_IJKTOINDEX_H

#include <fvdb/JaggedTensor.h>

namespace fvdb {
namespace detail {

class GridBatchImpl;

namespace ops {

/// @brief Convert ijk coordinates to linear voxel indices.
/// @param grid       The grid batch to look up coordinates in.
/// @param ijk        JaggedTensor of [N, 3] integer coordinates.
/// @param cumulative If true, indices are cumulative across the batch.
/// @return JaggedTensor of [N] int64 indices (-1 for inactive voxels).
JaggedTensor ijkToIndex(GridBatchImpl const &grid, JaggedTensor ijk, bool cumulative);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_IJKTOINDEX_H
