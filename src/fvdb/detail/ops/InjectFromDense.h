// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_INJECTFROMDENSE_H
#define FVDB_DETAIL_OPS_INJECTFROMDENSE_H

#include <fvdb/detail/GridBatchImpl.h>

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

torch::Tensor injectFromDenseCminor(const GridBatchImpl &batchHdl,
                                  const torch::Tensor &denseData,
                                  const torch::Tensor &denseOrigins);

torch::Tensor injectFromDenseCmajor(const GridBatchImpl &batchHdl,
                                  const torch::Tensor &denseData,
                                  const torch::Tensor &denseOrigins);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_INJECTFROMDENSE_H
