// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_JAGGEDTENSORINDEX_H
#define FVDB_DETAIL_OPS_JAGGEDTENSORINDEX_H

#include <fvdb/JaggedTensor.h>

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

template <torch::DeviceType>
JaggedTensor dispatchJaggedTensorIndexInt(const JaggedTensor &jt, int64_t idxVal);

template <torch::DeviceType>
JaggedTensor
dispatchJaggedTensorIndexSlice(const JaggedTensor &jt, int64_t start, int64_t end, int64_t step);

template <torch::DeviceType>
JaggedTensor dispatchJaggedTensorIndexJaggedTensor(const JaggedTensor &jt, const JaggedTensor &idx);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_JAGGEDTENSORINDEX_H
