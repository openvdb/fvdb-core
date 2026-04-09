// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GRIDEDGENETWORK_H
#define FVDB_DETAIL_OPS_GRIDEDGENETWORK_H

#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/GridBatchData.h>

#include <vector>

namespace fvdb {
namespace detail {
namespace ops {

std::vector<JaggedTensor> gridEdgeNetwork(const GridBatchData &gridHdl,
                                          bool returnVoxelCoordinates);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_GRIDEDGENETWORK_H
