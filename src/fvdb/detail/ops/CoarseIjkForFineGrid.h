// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_COARSEIJKFORFINEGRID_H
#define FVDB_DETAIL_OPS_COARSEIJKFORFINEGRID_H

#include <fvdb/JaggedTensor.h>

#include <nanovdb/NanoVDB.h>

namespace fvdb {
namespace detail {

class GridBatchImpl;

namespace ops {

/// @brief For each active voxel in the fine grid, compute the coarse ijk coordinate
///        by dividing and flooring by the coarsening factor.
/// Device dispatch is handled internally -- no template parameter needed.
JaggedTensor coarseIjkForFineGrid(GridBatchImpl const &fineGrid, nanovdb::Coord coarseningFactor);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_COARSEIJKFORFINEGRID_H
