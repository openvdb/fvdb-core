// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_AUTOGRAD_SPARSECONVOLUTIONKERNELMAP_H
#define FVDB_DETAIL_AUTOGRAD_SPARSECONVOLUTIONKERNELMAP_H

#include <torch/autograd.h>

namespace fvdb {
namespace detail {
namespace autograd {

struct SparseConvolutionKernelMap : public torch::autograd::Function<SparseConvolutionKernelMap> {
    using variable_list   = torch::autograd::variable_list;
    using AutogradContext = torch::autograd::AutogradContext;
    using Variable        = torch::autograd::Variable;

    static variable_list forward(AutogradContext *ctx,
                                 Variable inFeatures,
                                 Variable kernels,
                                 Variable neighborMap,
                                 Variable neighborSizes,
                                 int64_t srcVoxels,
                                 int64_t dstVoxels,
                                 bool middleAcceleration,
                                 bool transposed);

    static variable_list backward(AutogradContext *ctx, variable_list grad_output);
};

} // namespace autograd
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_AUTOGRAD_SPARSECONVOLUTIONKERNELMAP_H
