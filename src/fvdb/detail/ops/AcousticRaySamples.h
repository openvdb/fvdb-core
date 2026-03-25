// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_ACOUSTICRAYSAMPLES_H
#define FVDB_DETAIL_OPS_ACOUSTICRAYSAMPLES_H

#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/GridBatchImpl.h>

#include <torch/types.h>

#include <vector>

namespace fvdb {
namespace detail {
namespace ops {

// Generate samples along each ray with a given step size where samples are taken
// by bending the ray according to Snell's law.
std::vector<JaggedTensor> acousticRaySamples(const GridBatchImpl &batchHdl,
                                             const JaggedTensor &rayOrigins,
                                             const JaggedTensor &rayDirections,
                                             const JaggedTensor &soundSpeeds,
                                             float stepSize);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_ACOUSTICRAYSAMPLES_H
