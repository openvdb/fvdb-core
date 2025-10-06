// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_SERIALIZEENCODE_H
#define FVDB_DETAIL_OPS_SERIALIZEENCODE_H

#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/GridBatchImpl.h>

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

// Order type constants for serialize encode
constexpr int ORDER_TYPE_Z = 0;        // Regular z-order (xyz)
constexpr int ORDER_TYPE_Z_TRANS = 1;  // Transposed z-order (zyx)
constexpr int ORDER_TYPE_HILBERT = 2;  // Regular Hilbert curve (xyz)
constexpr int ORDER_TYPE_HILBERT_TRANS = 3;  // Transposed Hilbert curve (zyx)

template <torch::DeviceType>
JaggedTensor dispatchSerializeEncode(const GridBatchImpl &gridBatch, const std::string &order_type = "z");

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_SERIALIZEENCODE_H
