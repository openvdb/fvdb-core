// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_SERIALIZEGRID_H
#define FVDB_DETAIL_OPS_SERIALIZEGRID_H

#include <fvdb/GridBatchData.h>

namespace fvdb {
namespace detail {
namespace ops {

torch::Tensor serializeGrid(const GridBatchData &grid);

c10::intrusive_ptr<GridBatchData> deserializeGrid(const torch::Tensor &serialized);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_SERIALIZEGRID_H
