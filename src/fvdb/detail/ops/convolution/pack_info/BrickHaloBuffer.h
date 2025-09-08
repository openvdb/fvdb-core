// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_CONVOLUTION_PACK_INFO_BRICKHALOBUFFER_H
#define FVDB_DETAIL_OPS_CONVOLUTION_PACK_INFO_BRICKHALOBUFFER_H

#include <fvdb/detail/GridBatchImpl.h>

#include <torch/types.h>

#include <vector>

namespace fvdb {
namespace detail {
namespace ops {

template <torch::DeviceType>
std::vector<torch::Tensor> dispatchBrickHaloBuffer(const GridBatchImpl &batchHdl, bool benchmark);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_CONVOLUTION_PACK_INFO_BRICKHALOBUFFER_H
