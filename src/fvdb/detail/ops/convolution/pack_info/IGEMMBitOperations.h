// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_CONVOLUTION_PACK_INFO_IGEMMBITOPERATIONS_H
#define FVDB_DETAIL_OPS_CONVOLUTION_PACK_INFO_IGEMMBITOPERATIONS_H

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

template <torch::DeviceType>
torch::Tensor
dispatchBitmaskFromOutInMap(const torch::Tensor &outInMap, const int splitMaskNum, int validN);

template <torch::DeviceType>
torch::Tensor dispatchReorderOutInMap(const torch::Tensor &outInMap,
                                      const torch::Tensor &reorderLoc);

template <torch::DeviceType>
torch::Tensor dispatchReduceMask(const torch::Tensor &bitmask, const int reduceTile);

template <torch::DeviceType>
void dispatchTransposeOutInMap(const torch::Tensor &outInMap, const torch::Tensor &outInMapT);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_CONVOLUTION_PACK_INFO_IGEMMBITOPERATIONS_H
