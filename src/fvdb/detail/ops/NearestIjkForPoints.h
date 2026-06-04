// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_NEARESTIJKFORPOINTS_H
#define FVDB_DETAIL_OPS_NEARESTIJKFORPOINTS_H

#include <fvdb/JaggedTensor.h>
#include <fvdb/VoxelCoordTransform.h>

#include <vector>

namespace fvdb {
namespace detail {
namespace ops {

JaggedTensor nearestNeighborIJKForPoints(const JaggedTensor &points,
                                         const std::vector<VoxelCoordTransform> &transforms);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_NEARESTIJKFORPOINTS_H
