// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_BUILDGRIDFROMNEARESTVOXELSTOPOINTS_H
#define FVDB_DETAIL_OPS_BUILDGRIDFROMNEARESTVOXELSTOPOINTS_H

#include <fvdb/GridBatchData.h>
#include <fvdb/JaggedTensor.h>

#include <vector>

namespace fvdb {
namespace detail {
namespace ops {

c10::intrusive_ptr<GridBatchData>
buildGridFromNearestVoxelsToPoints(const JaggedTensor &points,
                                   const std::vector<nanovdb::Vec3d> &voxelSizes,
                                   const std::vector<nanovdb::Vec3d> &origins);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_BUILDGRIDFROMNEARESTVOXELSTOPOINTS_H
