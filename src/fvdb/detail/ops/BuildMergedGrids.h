// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_BUILDMERGEDGRIDS_H
#define FVDB_DETAIL_OPS_BUILDMERGEDGRIDS_H

#include <fvdb/detail/GridBatchData.h>

namespace fvdb {
namespace detail {
namespace ops {

c10::intrusive_ptr<GridBatchData> mergeGrids(const GridBatchData &gridBatch1,
                                             const GridBatchData &gridBatch2);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_BUILDMERGEDGRIDS_H
