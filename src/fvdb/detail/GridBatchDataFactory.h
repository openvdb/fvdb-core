// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_GRIDBATCHDATAFACTORY_H
#define FVDB_DETAIL_GRIDBATCHDATAFACTORY_H

#include <fvdb/detail/GridBatchData.h>

#include <torch/types.h>

#include <vector>

namespace fvdb {
namespace detail {

GridBatchData::GridMetadata *allocateHostGridMetadata(int64_t batchSize);
void freeHostGridMetadata(GridBatchData::GridMetadata *ptr);
GridBatchData::GridMetadata *allocateDeviceGridMetadata(torch::Device device, int64_t batchSize);
void freeDeviceGridMetadata(torch::Device device, GridBatchData::GridMetadata *ptr);
GridBatchData::GridMetadata *allocateUnifiedMemoryGridMetadata(int64_t batchSize);
void freeUnifiedMemoryGridMetadata(GridBatchData::GridMetadata *ptr);

void syncMetadataToDevice(GridBatchData::GridMetadata *hostMeta,
                          GridBatchData::GridMetadata *deviceMeta,
                          int64_t batchSize,
                          torch::Device device,
                          bool blocking);

torch::Tensor computeBatchOffsets(GridBatchData::GridMetadata *hostMeta,
                                  GridBatchData::GridMetadata *deviceMeta,
                                  int64_t batchSize,
                                  torch::Device device);

c10::intrusive_ptr<GridBatchData>
makeGridBatchData(nanovdb::GridHandle<TorchDeviceBuffer> &&gridHdl,
                  const std::vector<nanovdb::Vec3d> &voxelSizes,
                  const std::vector<nanovdb::Vec3d> &voxelOrigins);

c10::intrusive_ptr<GridBatchData> makeEmptyGridBatchData(const torch::Device &device);

c10::intrusive_ptr<GridBatchData> makeEmptyGridBatchData(const torch::Device &device,
                                                         const nanovdb::Vec3d &voxelSize,
                                                         const nanovdb::Vec3d &origin);

c10::intrusive_ptr<GridBatchData>
makeEmptyGridBatchData(const torch::Device &device,
                       const std::vector<nanovdb::Vec3d> &voxelSizes,
                       const std::vector<nanovdb::Vec3d> &origins);

} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_GRIDBATCHDATAFACTORY_H
