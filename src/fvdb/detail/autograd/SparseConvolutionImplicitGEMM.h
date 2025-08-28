// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_AUTOGRAD_SPARSECONVOLUTIONIMPLICITGEMM_H
#define FVDB_DETAIL_AUTOGRAD_SPARSECONVOLUTIONIMPLICITGEMM_H

#include <fvdb/SparseConvPackInfo.h>

#include <torch/autograd.h>

namespace fvdb {
namespace detail {
namespace autograd {

struct SparseConvolutionImplicitGEMM
    : public torch::autograd::Function<SparseConvolutionImplicitGEMM> {
    using variable_list   = torch::autograd::variable_list;
    using AutogradContext = torch::autograd::AutogradContext;
    using Variable        = torch::autograd::Variable;

    static variable_list forward(AutogradContext *ctx,
                                 Variable inFeatures,
                                 Variable kernels,
                                 SparseConvPackInfo packInfo,
                                 bool transposed);

    static variable_list backward(AutogradContext *ctx, variable_list grad_output);
};

} // namespace autograd
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_AUTOGRAD_SPARSECONVOLUTIONIMPLICITGEMM_H
