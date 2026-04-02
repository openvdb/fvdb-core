// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_BUILDGRIDFROMIJK_H
#define FVDB_DETAIL_OPS_BUILDGRIDFROMIJK_H

#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/GridBatchData.h>

#include <vector>

namespace fvdb {
namespace detail {
namespace ops {

// Internal helper used by other grid-building ops (BuildCoarseGridFromFine, BuildGridFromPoints,
// etc.)
nanovdb::GridHandle<TorchDeviceBuffer> _createNanoGridFromIJK(const JaggedTensor &ijk);

c10::intrusive_ptr<GridBatchData>
createNanoGridFromIJK(const JaggedTensor &ijk,
                      const std::vector<nanovdb::Vec3d> &voxelSizes,
                      const std::vector<nanovdb::Vec3d> &origins);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_BUILDGRIDFROMIJK_H
