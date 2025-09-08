// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GRIDEDGENETWORK_H
#define FVDB_DETAIL_OPS_GRIDEDGENETWORK_H

#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/GridBatchImpl.h>

#include <torch/types.h>

#include <vector>

namespace fvdb {
namespace detail {
namespace ops {

template <torch::DeviceType>
std::vector<JaggedTensor> dispatchGridEdgeNetwork(const GridBatchImpl &gridHdl,
                                                  bool returnVoxelCoordinates);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_GRIDEDGENETWORK_H
