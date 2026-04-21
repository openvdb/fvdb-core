// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_RAYIMPLICITINTERSECTION_H
#define FVDB_DETAIL_OPS_RAYIMPLICITINTERSECTION_H

#include <fvdb/GridBatchData.h>
#include <fvdb/JaggedTensor.h>

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

JaggedTensor rayImplicitIntersection(const GridBatchData &batchHdl,
                                     const JaggedTensor &rayOrigins,
                                     const JaggedTensor &rayDirections,
                                     const JaggedTensor &gridScalars,
                                     float eps);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_RAYIMPLICITINTERSECTION_H
