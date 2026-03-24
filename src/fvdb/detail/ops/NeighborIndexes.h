// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_NEIGHBORINDEXES_H
#define FVDB_DETAIL_OPS_NEIGHBORINDEXES_H

#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/GridBatchData.h>

#include <cstdint>

namespace fvdb {
namespace detail {
namespace ops {

JaggedTensor neighborIndexes(const GridBatchData &batchHdl,
                               const JaggedTensor &coords,
                               int32_t extent,
                               int32_t shift);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_NEIGHBORINDEXES_H
