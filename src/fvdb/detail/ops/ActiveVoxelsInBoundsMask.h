// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_ACTIVEVOXELSINBOUNDSMASK_H
#define FVDB_DETAIL_OPS_ACTIVEVOXELSINBOUNDSMASK_H

#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/GridBatchImpl.h>

#include <nanovdb/NanoVDB.h>

#include <torch/types.h>

#include <vector>

namespace fvdb {
namespace detail {
namespace ops {

template <torch::DeviceType>
JaggedTensor dispatchActiveVoxelsInBoundsMask(const GridBatchImpl &batchHdl,
                                              const std::vector<nanovdb::Coord> &bboxMins,
                                              const std::vector<nanovdb::Coord> &bboxMaxs);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_ACTIVEVOXELSINBOUNDSMASK_H
