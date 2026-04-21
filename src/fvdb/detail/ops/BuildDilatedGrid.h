// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_BUILDDILATEDGRID_H
#define FVDB_DETAIL_OPS_BUILDDILATEDGRID_H

#include <fvdb/GridBatchData.h>

#include <cstdint>
#include <vector>

namespace fvdb {
namespace detail {
namespace ops {

c10::intrusive_ptr<GridBatchData> dilateGrid(const GridBatchData &gridBatch,
                                             const std::vector<int64_t> &dilationAmount);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_BUILDDILATEDGRID_H
