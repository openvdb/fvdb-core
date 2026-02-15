// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_IJKTOINVINDEX_H
#define FVDB_DETAIL_OPS_IJKTOINVINDEX_H

#include <fvdb/JaggedTensor.h>

namespace fvdb {
namespace detail {

class GridBatchImpl;

namespace ops {

/// @brief For each ijk coordinate, write its element index into the
///        corresponding voxel position in the output array.
/// @param grid       The grid batch to look up coordinates in.
/// @param ijk        JaggedTensor of [N, 3] integer coordinates.
/// @param cumulative If true, element indices are global (cumulative).
///                   If false, element indices are per-batch (local).
/// @return JaggedTensor of [totalVoxels] int64 inverse indices (-1 for unmapped).
JaggedTensor ijkToInvIndex(GridBatchImpl const &grid, JaggedTensor ijk, bool cumulative);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_IJKTOINVINDEX_H
