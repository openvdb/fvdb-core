// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_VOXELSALONGRAYS_H
#define FVDB_DETAIL_OPS_VOXELSALONGRAYS_H

#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/GridBatchImpl.h>

#include <torch/types.h>

#include <vector>

namespace fvdb {
namespace detail {
namespace ops {

template <torch::DeviceType>
std::vector<JaggedTensor> dispatchVoxelsAlongRays(const GridBatchImpl &batchHdl,
                                                  const JaggedTensor &rayOrigins,
                                                  const JaggedTensor &rayDirections,
                                                  int64_t maxVox,
                                                  float eps,
                                                  bool returnIjk,
                                                  bool cumulative);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_VOXELSALONGRAYS_H
