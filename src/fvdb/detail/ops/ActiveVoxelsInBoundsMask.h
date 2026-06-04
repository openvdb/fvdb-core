// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_ACTIVEVOXELSINBOUNDSMASK_H
#define FVDB_DETAIL_OPS_ACTIVEVOXELSINBOUNDSMASK_H

#include <fvdb/GridBatchData.h>
#include <fvdb/JaggedTensor.h>

#include <vector>

namespace fvdb {
namespace detail {
namespace ops {

JaggedTensor activeVoxelsInBoundsMask(const GridBatchData &batchHdl,
                                      const std::vector<nanovdb::Coord> &bboxMins,
                                      const std::vector<nanovdb::Coord> &bboxMaxs);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_ACTIVEVOXELSINBOUNDSMASK_H
