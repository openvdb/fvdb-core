// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_VOXELSALONGRAYS_H
#define FVDB_DETAIL_OPS_VOXELSALONGRAYS_H

#include <fvdb/GridBatchData.h>
#include <fvdb/JaggedTensor.h>

#include <torch/types.h>

#include <vector>

namespace fvdb {
namespace detail {
namespace ops {

std::vector<JaggedTensor> voxelsAlongRays(const GridBatchData &batchHdl,
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
