// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_IJKTOINVINDEX_H
#define FVDB_DETAIL_OPS_IJKTOINVINDEX_H

#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/GridBatchData.h>

namespace fvdb {
namespace detail {
namespace ops {

JaggedTensor ijkToInvIndex(const GridBatchData &batchHdl, const JaggedTensor &ijk, bool cumulative);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_IJKTOINVINDEX_H
