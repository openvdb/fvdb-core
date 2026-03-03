// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_POPULATEGRIDMETADATA_H
#define FVDB_DETAIL_OPS_POPULATEGRIDMETADATA_H

#include <fvdb/detail/GridBatchImpl.h>

#include <torch/types.h>

#include <vector>

namespace fvdb {
namespace detail {
namespace ops {

void populateGridMetadata(const nanovdb::GridHandle<TorchDeviceBuffer> &batchHdl,
                          const std::vector<nanovdb::Vec3d> &voxelSizes,
                          const std::vector<nanovdb::Vec3d> &voxelOrigins,
                          torch::Tensor &outBatchOffsets,
                          GridBatchImpl::GridMetadata *outPerGridMetadataHost,
                          GridBatchImpl::GridMetadata *outPerGridMetadataDevice,
                          GridBatchImpl::GridBatchMetadata *outBatchMetadataHost);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_POPULATEGRIDMETADATA_H
