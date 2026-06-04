// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_COORDSINGRID_H
#define FVDB_DETAIL_OPS_COORDSINGRID_H

#include <fvdb/GridBatchData.h>
#include <fvdb/JaggedTensor.h>

namespace fvdb {
namespace detail {
namespace ops {

JaggedTensor coordsInGrid(const GridBatchData &batchHdl, const JaggedTensor &coords);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_COORDSINGRID_H
