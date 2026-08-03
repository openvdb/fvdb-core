// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_BUILDPADDEDGRID_H
#define FVDB_DETAIL_OPS_BUILDPADDEDGRID_H

#include <fvdb/GridBatchData.h>

namespace fvdb {
namespace detail {
namespace ops {

/// @brief Build a grid whose topology is the source padded or eroded by the box [bmin, bmax]^3.
///
/// With @p excludeBorder false the output coordinate set is the Minkowski sum of the source with
/// the structuring element, so a voxel is active if any coordinate in its [bmin, bmax]^3
/// neighborhood is active in the source. With @p excludeBorder true the operation is the
/// corresponding erosion, so a voxel is active only if *every* coordinate in that neighborhood is
/// active in the source, and the result is a subset of the source.
///
/// @p dualTransform selects which lattice the result is interpreted on. When true, the source's
/// primal and dual transforms are swapped, placing the result's voxel centers at the corners of
/// the source's voxels and shifting the origin by half a voxel; this is the interpretation used
/// by `dual_grid`. When false the result stays on the same lattice as the source and carries the
/// source's transforms over unchanged, as required for a plain padded grid.
///
/// @param baseBatchHdl The source grid batch to pad or erode
/// @param bmin Lower corner of the structuring element, must be <= 0
/// @param bmax Upper corner of the structuring element, must be >= 0
/// @param excludeBorder Erode by the structuring element instead of padding by it
/// @param dualTransform Reinterpret the result on the dual (corner) lattice by swapping the
///        source's primal and dual transforms
/// @return A new grid batch with the padded or eroded topology
c10::intrusive_ptr<GridBatchData> buildPaddedGrid(
    const GridBatchData &baseBatchHdl, int bmin, int bmax, bool excludeBorder, bool dualTransform);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_BUILDPADDEDGRID_H
