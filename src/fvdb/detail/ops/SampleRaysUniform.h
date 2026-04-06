// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_SAMPLERAYSUNIFORM_H
#define FVDB_DETAIL_OPS_SAMPLERAYSUNIFORM_H

#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/GridBatchData.h>

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

JaggedTensor uniformRaySamples(const GridBatchData &batchHdl,
                               const JaggedTensor &rayO,
                               const JaggedTensor &rayD,
                               const JaggedTensor &tMin,
                               const JaggedTensor &tMax,
                               const double minStepSize,
                               const double coneAngle,
                               const bool includeEndSegments,
                               const bool return_midpoint,
                               const double eps);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_SAMPLERAYSUNIFORM_H
