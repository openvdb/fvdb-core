// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_MARCHINGCUBES_H
#define FVDB_DETAIL_OPS_MARCHINGCUBES_H

#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/GridBatchData.h>

#include <torch/types.h>

#include <vector>

namespace fvdb {
namespace detail {
namespace ops {

std::vector<JaggedTensor>
marchingCubes(const GridBatchData &batchHdl, const JaggedTensor &field, double level);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_MARCHINGCUBES_H
