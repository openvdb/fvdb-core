// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GRIDEDGENETWORK_H
#define FVDB_DETAIL_OPS_GRIDEDGENETWORK_H

#include <fvdb/JaggedTensor.h>

#include <vector>

namespace fvdb {
namespace detail {

class GridBatchImpl;

namespace ops {

/// @brief Build a wireframe cube mesh for every active voxel:
///        8 vertices and 12 edges per voxel.
/// @return {vertices_jagged, edges_jagged}
std::vector<JaggedTensor> gridEdgeNetwork(GridBatchImpl const &grid, bool returnVoxelCoordinates);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_GRIDEDGENETWORK_H
