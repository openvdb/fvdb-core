// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_POPULATEGRIDMETADATA_H
#define FVDB_DETAIL_OPS_POPULATEGRIDMETADATA_H

#include <fvdb/GridBatchData.h>

#include <torch/types.h>

#include <vector>

namespace fvdb {
namespace detail {
namespace ops {

void populateGridMetadata(const nanovdb::GridHandle<TorchDeviceBuffer> &batchHdl,
                          const std::vector<nanovdb::Vec3d> &voxelSizes,
                          const std::vector<nanovdb::Vec3d> &voxelOrigins,
                          torch::Tensor &outBatchOffsets,
                          GridBatchData::GridMetadata *outPerGridMetadataHost,
                          GridBatchData::GridMetadata *outPerGridMetadataDevice,
                          GridBatchData::GridBatchMetadata *outBatchMetadataHost);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_POPULATEGRIDMETADATA_H
