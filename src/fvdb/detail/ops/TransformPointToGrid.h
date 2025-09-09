// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_TRANSFORMPOINTTOGRID_H
#define FVDB_DETAIL_OPS_TRANSFORMPOINTTOGRID_H

#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/GridBatchImpl.h>

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

template <torch::DeviceType>
torch::Tensor dispatchTransformPointsToGrid(const GridBatchImpl &batchHdl,
                                            const JaggedTensor &points,
                                            bool isPrimal);

template <torch::DeviceType>
torch::Tensor dispatchInvTransformPointsToGrid(const GridBatchImpl &batchHdl,
                                               const JaggedTensor &points,
                                               bool isPrimal);

template <torch::DeviceType>
torch::Tensor dispatchTransformPointsToGridBackward(const GridBatchImpl &batchHdl,
                                                    const JaggedTensor &gradOut,
                                                    bool isPrimal);

template <torch::DeviceType>
torch::Tensor dispatchInvTransformPointsToGridBackward(const GridBatchImpl &batchHdl,
                                                       const JaggedTensor &gradOut,
                                                       bool isPrimal);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_TRANSFORMPOINTTOGRID_H
