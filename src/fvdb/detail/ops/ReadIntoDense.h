// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_READINTODENSE_H
#define FVDB_DETAIL_OPS_READINTODENSE_H

#include <fvdb/detail/GridBatchImpl.h>

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

template <torch::DeviceType>
void dispatchReadIntoDenseCminor(const GridBatchImpl &batchHdl,
                                 const torch::Tensor &inGridData,
                                 const torch::Tensor &denseOrigins,
                                 torch::Tensor &outDenseTensor);

template <torch::DeviceType>
void dispatchReadIntoDenseCmajor(const GridBatchImpl &batchHdl,
                                 const torch::Tensor &inGridData,
                                 const torch::Tensor &denseOrigins,
                                 torch::Tensor &outDenseTensor);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_READINTODENSE_H
