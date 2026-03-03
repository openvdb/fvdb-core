// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_BUILDGRIDFROMMESH_H
#define FVDB_DETAIL_OPS_BUILDGRIDFROMMESH_H

#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/TorchDeviceBuffer.h>
#include <fvdb/detail/VoxelCoordTransform.h>

#include <nanovdb/GridHandle.h>

#include <vector>

namespace fvdb {
namespace detail {
namespace ops {

nanovdb::GridHandle<TorchDeviceBuffer>
buildGridFromMesh(const JaggedTensor &meshVertices,
                  const JaggedTensor &meshFaces,
                  const std::vector<VoxelCoordTransform> &tx);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_BUILDGRIDFROMMESH_H
