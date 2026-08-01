// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_BUILDPADDEDGRID_H
#define FVDB_DETAIL_OPS_BUILDPADDEDGRID_H

#include <fvdb/GridBatchData.h>

namespace fvdb {
namespace detail {
namespace ops {

// Build a grid whose topology is the source padded (Minkowski-summed) / eroded by the box
// [bmin, bmax]^3. When `dualTransform` is true the result is reinterpreted onto the dual
// (corner) lattice -- the source's primal/dual transforms are swapped -- which is what
// `dual_grid` wants. When false the result stays on the *same* lattice as the source and keeps
// the source's transforms verbatim, which is what a plain padded grid wants.
c10::intrusive_ptr<GridBatchData> buildPaddedGrid(
    const GridBatchData &baseBatchHdl, int bmin, int bmax, bool excludeBorder, bool dualTransform);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_BUILDPADDEDGRID_H
