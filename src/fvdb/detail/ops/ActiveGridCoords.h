// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_ACTIVEGRIDCOORDS_H
#define FVDB_DETAIL_OPS_ACTIVEGRIDCOORDS_H

#include <fvdb/GridBatchData.h>
#include <fvdb/JaggedTensor.h>

namespace fvdb {
namespace detail {
namespace ops {

JaggedTensor activeGridCoords(GridBatchData const &gridBatch);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_ACTIVEGRIDCOORDS_H
