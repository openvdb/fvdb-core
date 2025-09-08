// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_INJECT_H
#define FVDB_DETAIL_OPS_INJECT_H

#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/GridBatchImpl.h>

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

template <torch::DeviceType>
void dispatchInject(const GridBatchImpl &dstGridBatch,
                    const GridBatchImpl &srcGridBatch,
                    JaggedTensor &dst,
                    const JaggedTensor &src);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_INJECT_H
