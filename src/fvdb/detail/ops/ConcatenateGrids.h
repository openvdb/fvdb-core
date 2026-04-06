// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_CONCATENATEGRIDS_H
#define FVDB_DETAIL_OPS_CONCATENATEGRIDS_H

#include <fvdb/detail/GridBatchData.h>

#include <vector>

namespace fvdb {
namespace detail {
namespace ops {

c10::intrusive_ptr<GridBatchData>
concatenateGrids(const std::vector<c10::intrusive_ptr<GridBatchData>> &elements);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_CONCATENATEGRIDS_H
