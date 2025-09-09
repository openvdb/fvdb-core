// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_JOFFSETSFROMJIDX_H
#define FVDB_DETAIL_OPS_JOFFSETSFROMJIDX_H

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

template <torch::DeviceType>
torch::Tensor dispatchJOffsetsFromJIdx(torch::Tensor jidx, torch::Tensor jdata, int64_t numTensors);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_JOFFSETSFROMJIDX_H
