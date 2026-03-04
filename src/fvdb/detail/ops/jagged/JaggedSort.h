// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_JAGGED_JAGGEDSORT_H
#define FVDB_DETAIL_OPS_JAGGED_JAGGEDSORT_H

#include <fvdb/JaggedTensor.h>

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

torch::Tensor jaggedArgsort(const JaggedTensor &jt);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_JAGGED_JAGGEDSORT_H
