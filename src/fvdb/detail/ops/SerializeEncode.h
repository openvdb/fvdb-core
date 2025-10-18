// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_SERIALIZEENCODE_H
#define FVDB_DETAIL_OPS_SERIALIZEENCODE_H

#include <fvdb/JaggedTensor.h>
#include <fvdb/Types.h>
#include <fvdb/detail/GridBatchImpl.h>

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

template <torch::DeviceType>
JaggedTensor
dispatchSerializeEncode(const GridBatchImpl &gridBatch,
                        SpaceFillingCurveType order_type = SpaceFillingCurveType::ZOrder);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_SERIALIZEENCODE_H
