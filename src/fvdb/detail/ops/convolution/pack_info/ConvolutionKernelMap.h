// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_CONVOLUTION_PACK_INFO_CONVOLUTIONKERNELMAP_H
#define FVDB_DETAIL_OPS_CONVOLUTION_PACK_INFO_CONVOLUTIONKERNELMAP_H

#include <fvdb/Types.h>
#include <fvdb/detail/GridBatchImpl.h>

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

template <torch::DeviceType>
void dispatchConvolutionKernelMap(const GridBatchImpl &source,
                                  const GridBatchImpl &target,
                                  torch::Tensor &kernelMap,
                                  const Vec3iOrScalar &kernelSize,
                                  const Vec3iOrScalar &stride);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_CONVOLUTION_PACK_INFO_CONVOLUTIONKERNELMAP_H
