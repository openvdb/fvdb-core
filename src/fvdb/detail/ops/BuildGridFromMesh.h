// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_BUILDGRIDFROMMESH_H
#define FVDB_DETAIL_OPS_BUILDGRIDFROMMESH_H

#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/GridBatchImpl.h>
#include <fvdb/detail/TorchDeviceBuffer.h>
#include <fvdb/detail/VoxelCoordTransform.h>

#include <nanovdb/GridHandle.h>
#include <nanovdb/NanoVDB.h>

#include <torch/types.h>

#include <vector>

namespace fvdb {
namespace detail {
namespace ops {

template <torch::DeviceType>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchBuildGridFromMesh(const JaggedTensor &meshVertices,
                          const JaggedTensor &meshFaces,
                          const std::vector<VoxelCoordTransform> &tx);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_BUILDGRIDFROMMESH_H
