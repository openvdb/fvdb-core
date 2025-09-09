// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_CONVOLUTION_BACKEND_SPARSECONVOLUTIONHALOGRAD_H
#define FVDB_DETAIL_OPS_CONVOLUTION_BACKEND_SPARSECONVOLUTIONHALOGRAD_H

#include <fvdb/detail/GridBatchImpl.h>

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

template <torch::DeviceType>
torch::Tensor dispatchSparseConvolutionHaloGrad(const GridBatchImpl &batchHdl,
                                                const torch::Tensor &inFeatures,
                                                const torch::Tensor &gradOutFeatures);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_CONVOLUTION_BACKEND_SPARSECONVOLUTIONHALOGRAD_H
