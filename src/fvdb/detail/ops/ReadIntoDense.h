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

torch::Tensor readIntoDenseCminor(const GridBatchImpl &batchHdl,
                                  const torch::Tensor &sparseData,
                                  const torch::Tensor &denseOrigins,
                                  const nanovdb::Coord &gridSize);

torch::Tensor readIntoDenseCmajor(const GridBatchImpl &batchHdl,
                                  const torch::Tensor &sparseData,
                                  const torch::Tensor &denseOrigins,
                                  const nanovdb::Coord &gridSize);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_READINTODENSE_H
