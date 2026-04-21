// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_INJECTTODENSE_H
#define FVDB_DETAIL_OPS_INJECTTODENSE_H

#include <fvdb/GridBatchData.h>

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

torch::Tensor injectToDenseCminor(const GridBatchData &batchHdl,
                                  const torch::Tensor &sparseData,
                                  const torch::Tensor &denseOrigins,
                                  const nanovdb::Coord &gridSize);

torch::Tensor injectToDenseCmajor(const GridBatchData &batchHdl,
                                  const torch::Tensor &sparseData,
                                  const torch::Tensor &denseOrigins,
                                  const nanovdb::Coord &gridSize);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_INJECTTODENSE_H
