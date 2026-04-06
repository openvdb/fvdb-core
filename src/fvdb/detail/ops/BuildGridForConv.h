// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_BUILDGRIDFORCONV_H
#define FVDB_DETAIL_OPS_BUILDGRIDFORCONV_H

#include <fvdb/detail/GridBatchData.h>

namespace fvdb {
namespace detail {
namespace ops {

c10::intrusive_ptr<GridBatchData> buildGridForConv(const GridBatchData &baseBatchHdl,
                                                   const nanovdb::Coord &kernelSize,
                                                   const nanovdb::Coord &stride);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_BUILDGRIDFORCONV_H
