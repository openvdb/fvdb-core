// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_POINTSINGRID_H
#define FVDB_DETAIL_OPS_POINTSINGRID_H

#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/GridBatchImpl.h>

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

template <torch::DeviceType>
JaggedTensor dispatchPointsInGrid(const GridBatchImpl &batchHdl, const JaggedTensor &points);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_POINTSINGRID_H
