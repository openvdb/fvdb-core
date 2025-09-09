// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_ACTIVEGRIDGOORDS_H
#define FVDB_DETAIL_OPS_ACTIVEGRIDGOORDS_H

#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/GridBatchImpl.h>

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

template <torch::DeviceType>
JaggedTensor dispatchActiveGridCoords(const GridBatchImpl &gridAccessor);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_ACTIVEGRIDGOORDS_H
