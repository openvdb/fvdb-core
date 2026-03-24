// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_CLONEGRID_H
#define FVDB_DETAIL_OPS_CLONEGRID_H

#include <fvdb/detail/GridBatchData.h>

namespace fvdb {
namespace detail {
namespace ops {

c10::intrusive_ptr<GridBatchData> cloneGrid(const GridBatchData &grid,
                                            const torch::Device &device,
                                            bool blocking = false);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_CLONEGRID_H
