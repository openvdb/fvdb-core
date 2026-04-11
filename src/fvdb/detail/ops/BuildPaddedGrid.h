// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_BUILDPADDEDGRID_H
#define FVDB_DETAIL_OPS_BUILDPADDEDGRID_H

#include <fvdb/detail/GridBatchData.h>

namespace fvdb {
namespace detail {
namespace ops {

c10::intrusive_ptr<GridBatchData>
buildPaddedGrid(const GridBatchData &baseBatchHdl, int bmin, int bmax, bool excludeBorder);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_BUILDPADDEDGRID_H
