// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_COARSEIJKFORFINEGRID_H
#define FVDB_DETAIL_OPS_COARSEIJKFORFINEGRID_H

#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/GridBatchImpl.h>

#include <nanovdb/NanoVDB.h>

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

template <torch::DeviceType>
JaggedTensor dispatchCoarseIJKForFineGrid(const GridBatchImpl &batchHdl,
                                          nanovdb::Coord coarseningFactor);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_COARSEIJKFORFINEGRID_H
