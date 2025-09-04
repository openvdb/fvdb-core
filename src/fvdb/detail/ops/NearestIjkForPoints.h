// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_NEARESTIJKFORPOINTS_H
#define FVDB_DETAIL_OPS_NEARESTIJKFORPOINTS_H

#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/VoxelCoordTransform.h>

#include <torch/types.h>

#include <vector>

namespace fvdb {
namespace detail {
namespace ops {

template <torch::DeviceType>
JaggedTensor
dispatchNearestNeighborIJKForPoints(const JaggedTensor &points,
                                    const std::vector<VoxelCoordTransform> &transforms);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_NEARESTIJKFORPOINTS_H
