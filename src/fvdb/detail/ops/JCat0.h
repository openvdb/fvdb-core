// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_JCAT0_H
#define FVDB_DETAIL_OPS_JCAT0_H

#include <fvdb/JaggedTensor.h>

#include <torch/types.h>

#include <vector>

namespace fvdb {
namespace detail {
namespace ops {

template <torch::DeviceType> JaggedTensor dispatchJCat0(const std::vector<JaggedTensor> &tensors);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_JCAT0_H
