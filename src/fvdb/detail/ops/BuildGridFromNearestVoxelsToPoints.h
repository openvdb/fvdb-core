// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_BUILDGRIDFROMNEARESTVOXELSTOPOINTS_H
#define FVDB_DETAIL_OPS_BUILDGRIDFROMNEARESTVOXELSTOPOINTS_H

#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/GridBatchImpl.h>

#include <nanovdb/NanoVDB.h>

#include <vector>

namespace fvdb {
namespace detail {
namespace ops {

c10::intrusive_ptr<GridBatchImpl>
buildGridFromNearestVoxelsToPoints(const JaggedTensor &points,
                                   const std::vector<nanovdb::Vec3d> &voxelSizes,
                                   const std::vector<nanovdb::Vec3d> &origins);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_BUILDGRIDFROMNEARESTVOXELSTOPOINTS_H
