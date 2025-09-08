// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_CONVOLUTION_BACKEND_MESPARSECONVOLUTION_H
#define FVDB_DETAIL_OPS_CONVOLUTION_BACKEND_MESPARSECONVOLUTION_H

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

void dispatchMESparseConvolutionKernelMapGrad(torch::Tensor in_feat,
                                              torch::Tensor grad_in_feat,
                                              torch::Tensor grad_out_feat,
                                              torch::Tensor kernel,
                                              torch::Tensor grad_kernel,
                                              torch::Tensor neighbor_map,
                                              torch::Tensor neighbor_offset,
                                              const bool transpose);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_CONVOLUTION_BACKEND_MESPARSECONVOLUTION_H
