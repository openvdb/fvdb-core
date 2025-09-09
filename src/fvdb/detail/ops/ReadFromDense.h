// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_READFROMDENSE_H
#define FVDB_DETAIL_OPS_READFROMDENSE_H

#include <fvdb/detail/GridBatchImpl.h>

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

template <torch::DeviceType>
void dispatchReadFromDenseXyzc(const GridBatchImpl &batchHdl,
                               const torch::Tensor &inDenseTensor,
                               const torch::Tensor &denseOrigins,
                               torch::Tensor &outSparseTensor);

template <torch::DeviceType>
void dispatchReadFromDenseCzyx(const GridBatchImpl &batchHdl,
                               const torch::Tensor &inDenseTensor,
                               const torch::Tensor &denseOrigins,
                               torch::Tensor &outSparseTensor);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_READFROMDENSE_H
