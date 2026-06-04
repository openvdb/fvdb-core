// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_SEGMENTSALONGRAYS_H
#define FVDB_DETAIL_OPS_SEGMENTSALONGRAYS_H

#include <fvdb/GridBatchData.h>
#include <fvdb/JaggedTensor.h>

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

JaggedTensor segmentsAlongRays(const GridBatchData &batchHdl,
                               const JaggedTensor &rayOrigins,
                               const JaggedTensor &rayDirections,
                               int64_t maxSegments,
                               const double eps);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_SEGMENTSALONGRAYS_H
