// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_AUTOGRAD_INJECT_H
#define FVDB_DETAIL_AUTOGRAD_INJECT_H

#include <fvdb/detail/GridBatchImpl.h>

#include <torch/autograd.h>

namespace fvdb {
namespace detail {
namespace autograd {

struct Inject : public torch::autograd::Function<Inject> {
    using variable_list   = torch::autograd::variable_list;
    using AutogradContext = torch::autograd::AutogradContext;
    using Variable        = torch::autograd::Variable;

    static variable_list forward(AutogradContext *ctx,
                                 const c10::intrusive_ptr<GridBatchImpl> fromGrid,
                                 Variable const &fromFeaturesJData,
                                 Variable const &fromFeaturesJOffsets,
                                 Variable const &fromFeaturesJIdx,
                                 Variable const &fromFeaturesJLIdx,
                                 const c10::intrusive_ptr<GridBatchImpl> toGrid,
                                 Variable const &toFeaturesJData,
                                 Variable const &toFeaturesJOffsets,
                                 Variable const &toFeaturesJIdx,
                                 Variable const &toFeaturesJLIdx);

    static variable_list backward(AutogradContext *ctx, variable_list grad_output);
};

} // namespace autograd
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_AUTOGRAD_INJECT_H
