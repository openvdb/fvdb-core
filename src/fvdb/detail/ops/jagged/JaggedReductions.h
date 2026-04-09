// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_JAGGED_JAGGEDREDUCTIONS_H
#define FVDB_DETAIL_OPS_JAGGED_JAGGEDREDUCTIONS_H

#include <fvdb/JaggedTensor.h>

#include <vector>

namespace fvdb {
namespace detail {
namespace ops {

JaggedTensor jaggedSum(const JaggedTensor &jt, int64_t dim, bool keepdim);
std::vector<JaggedTensor> jaggedMin(const JaggedTensor &jt, int64_t dim, bool keepdim);
std::vector<JaggedTensor> jaggedMax(const JaggedTensor &jt, int64_t dim, bool keepdim);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_JAGGED_JAGGEDREDUCTIONS_H
