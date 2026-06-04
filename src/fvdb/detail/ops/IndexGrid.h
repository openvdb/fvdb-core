// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_INDEXGRID_H
#define FVDB_DETAIL_OPS_INDEXGRID_H

#include <fvdb/GridBatchData.h>

#include <cstdint>
#include <vector>

namespace fvdb {
namespace detail {
namespace ops {

c10::intrusive_ptr<GridBatchData> indexGrid(const GridBatchData &grid, int64_t bi);

c10::intrusive_ptr<GridBatchData>
indexGrid(const GridBatchData &grid, ssize_t start, ssize_t stop, ssize_t step);

c10::intrusive_ptr<GridBatchData> indexGrid(const GridBatchData &grid,
                                            const torch::Tensor &indices);

c10::intrusive_ptr<GridBatchData> indexGrid(const GridBatchData &grid,
                                            const std::vector<int64_t> &indices);

c10::intrusive_ptr<GridBatchData> indexGrid(const GridBatchData &grid,
                                            const std::vector<bool> &indices);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_INDEXGRID_H
