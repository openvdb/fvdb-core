// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_AUTOGRAD_READFROMDENSE_H
#define FVDB_DETAIL_AUTOGRAD_READFROMDENSE_H

#include <fvdb/detail/GridBatchData.h>

#include <torch/autograd.h>

namespace fvdb {
namespace detail {
namespace autograd {

struct ReadFromDenseCminor : public torch::autograd::Function<ReadFromDenseCminor> {
    using variable_list   = torch::autograd::variable_list;
    using AutogradContext = torch::autograd::AutogradContext;
    using Variable        = torch::autograd::Variable;

    static variable_list forward(AutogradContext *ctx,
                                 c10::intrusive_ptr<GridBatchData> grid,
                                 Variable denseData,
                                 const torch::Tensor &denseOrigins);

    static variable_list backward(AutogradContext *ctx, variable_list grad_output);
};

struct ReadFromDenseCmajor : public torch::autograd::Function<ReadFromDenseCmajor> {
    using variable_list   = torch::autograd::variable_list;
    using AutogradContext = torch::autograd::AutogradContext;
    using Variable        = torch::autograd::Variable;

    static variable_list forward(AutogradContext *ctx,
                                 c10::intrusive_ptr<GridBatchData> grid,
                                 Variable denseData,
                                 const torch::Tensor &denseOrigins);

    static variable_list backward(AutogradContext *ctx, variable_list grad_output);
};

} // namespace autograd
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_AUTOGRAD_READFROMDENSE_H
