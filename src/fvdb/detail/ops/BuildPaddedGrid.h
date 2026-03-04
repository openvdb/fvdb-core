// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_BUILDPADDEDGRID_H
#define FVDB_DETAIL_OPS_BUILDPADDEDGRID_H

#include <fvdb/detail/GridBatchImpl.h>

namespace fvdb {
namespace detail {
namespace ops {

c10::intrusive_ptr<GridBatchImpl>
buildPaddedGrid(const GridBatchImpl &baseBatchHdl, int bmin, int bmax, bool excludeBorder);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_BUILDPADDEDGRID_H
