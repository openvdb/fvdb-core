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

JaggedTensor jaggedTensorIndexInt(const JaggedTensor &jt, int64_t idxVal);

JaggedTensor
jaggedTensorIndexSlice(const JaggedTensor &jt, int64_t start, int64_t end, int64_t step);

JaggedTensor jaggedTensorIndexJaggedTensor(const JaggedTensor &jt, const JaggedTensor &idx);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_JAGGEDTENSORINDEX_H
