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

torch::Tensor
transformPointsToGrid(const GridBatchImpl &batchHdl, const JaggedTensor &points, bool isPrimal);

torch::Tensor
invTransformPointsToGrid(const GridBatchImpl &batchHdl, const JaggedTensor &points, bool isPrimal);

torch::Tensor transformPointsToGridBackward(const GridBatchImpl &batchHdl,
                                            const JaggedTensor &gradOut,
                                            bool isPrimal);

torch::Tensor invTransformPointsToGridBackward(const GridBatchImpl &batchHdl,
                                               const JaggedTensor &gradOut,
                                               bool isPrimal);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_TRANSFORMPOINTTOGRID_H
