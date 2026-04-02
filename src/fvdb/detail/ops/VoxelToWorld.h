// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_VOXELTOWORLD_H
#define FVDB_DETAIL_OPS_VOXELTOWORLD_H

#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/GridBatchData.h>

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

torch::Tensor
voxelToWorld(const GridBatchData &batchHdl, const JaggedTensor &points, bool isPrimal);

torch::Tensor
worldToVoxel(const GridBatchData &batchHdl, const JaggedTensor &points, bool isPrimal);

torch::Tensor
voxelToWorldBackward(const GridBatchData &batchHdl, const JaggedTensor &gradOut, bool isPrimal);

torch::Tensor
worldToVoxelBackward(const GridBatchData &batchHdl, const JaggedTensor &gradOut, bool isPrimal);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_VOXELTOWORLD_H
