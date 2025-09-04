// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_JAGGED_JAGGEDREDUCE_H
#define FVDB_DETAIL_OPS_JAGGED_JAGGEDREDUCE_H

#include <fvdb/JaggedTensor.h>

#include <torch/types.h>

#include <cstdint>
#include <vector>

namespace fvdb {
namespace detail {
namespace ops {

template <torch::DeviceType>
torch::Tensor dispatchJaggedSum(const torch::Tensor &jdata,
                                const torch::Tensor &jidx,
                                const torch::Tensor &joffsets,
                                int64_t dimSize);

template <torch::DeviceType>
std::vector<torch::Tensor> dispatchJaggedMin(const torch::Tensor &jdata,
                                             const torch::Tensor &jidx,
                                             const torch::Tensor &joffsets,
                                             int64_t dimSize);

template <torch::DeviceType>
std::vector<torch::Tensor> dispatchJaggedMax(const torch::Tensor &jdata,
                                             const torch::Tensor &jidx,
                                             const torch::Tensor &joffsets,
                                             int64_t dimSize);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_JAGGED_JAGGEDREDUCE_H
