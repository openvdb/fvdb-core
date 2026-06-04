// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_POINTSINGRID_H
#define FVDB_DETAIL_OPS_POINTSINGRID_H

#include <fvdb/GridBatchData.h>
#include <fvdb/JaggedTensor.h>

namespace fvdb {
namespace detail {
namespace ops {

JaggedTensor pointsInGrid(const GridBatchData &batchHdl, const JaggedTensor &points);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_POINTSINGRID_H
