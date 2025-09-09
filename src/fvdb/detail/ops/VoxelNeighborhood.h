// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_VOXELNEIGHBORHOOD_H
#define FVDB_DETAIL_OPS_VOXELNEIGHBORHOOD_H

#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/GridBatchImpl.h>
#include <fvdb/detail/utils/Utils.h>

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

template <torch::DeviceType>
JaggedTensor dispatchVoxelNeighborhood(const GridBatchImpl &batchHdl,
                                       const JaggedTensor &coords,
                                       nanovdb::Coord extentMin,
                                       nanovdb::Coord extentMax,
                                       int32_t shift);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_VOXELNEIGHBORHOOD_H
