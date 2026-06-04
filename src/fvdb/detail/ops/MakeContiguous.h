// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_MAKECONTIGUOUS_H
#define FVDB_DETAIL_OPS_MAKECONTIGUOUS_H

#include <fvdb/GridBatchData.h>

namespace fvdb {
namespace detail {
namespace ops {

c10::intrusive_ptr<GridBatchData> makeContiguous(c10::intrusive_ptr<GridBatchData> input);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_MAKECONTIGUOUS_H
