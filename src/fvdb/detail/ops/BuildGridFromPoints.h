// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_BUILDGRIDFROMPOINTS_H
#define FVDB_DETAIL_OPS_BUILDGRIDFROMPOINTS_H

#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/TorchDeviceBuffer.h>
#include <fvdb/detail/VoxelCoordTransform.h>

#include <nanovdb/GridHandle.h>

#include <vector>

namespace fvdb {
namespace detail {
namespace ops {

nanovdb::GridHandle<TorchDeviceBuffer>
buildGridFromPoints(const JaggedTensor &points, const std::vector<VoxelCoordTransform> &txs);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_BUILDGRIDFROMPOINTS_H
