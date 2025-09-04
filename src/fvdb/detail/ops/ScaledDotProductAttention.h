// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_SCALEDDOTPRODUCTATTENTION_H
#define FVDB_DETAIL_OPS_SCALEDDOTPRODUCTATTENTION_H

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

template <torch::DeviceType>
torch::Tensor dispatchScaledDotProductAttention(const torch::Tensor &query,
                                                const torch::Tensor &key,
                                                const torch::Tensor &value,
                                                const torch::Tensor &qLengths,
                                                const torch::Tensor &kvLengths,
                                                bool training,
                                                float scale);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_SCALEDDOTPRODUCTATTENTION_H
