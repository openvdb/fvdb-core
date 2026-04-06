// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_JIDXFORGRID_H
#define FVDB_DETAIL_OPS_JIDXFORGRID_H

#include <fvdb/detail/GridBatchData.h>

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

torch::Tensor jIdxForGrid(const GridBatchData &batchHdl);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_JIDXFORGRID_H
