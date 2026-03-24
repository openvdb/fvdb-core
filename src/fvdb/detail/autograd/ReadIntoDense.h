// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_AUTOGRAD_READINTODENSE_H
#define FVDB_DETAIL_AUTOGRAD_READINTODENSE_H

#include <fvdb/detail/GridBatchData.h>

#include <nanovdb/NanoVDB.h>
#include <torch/autograd.h>

namespace fvdb {
namespace detail {
namespace autograd {

struct ReadIntoDenseCminor : public torch::autograd::Function<ReadIntoDenseCminor> {
    using variable_list   = torch::autograd::variable_list;
    using AutogradContext = torch::autograd::AutogradContext;
    using Variable        = torch::autograd::Variable;

    static variable_list forward(AutogradContext *ctx,
                                 c10::intrusive_ptr<GridBatchData> grid,
                                 Variable sparseData,
                                 const std::optional<torch::Tensor> &maybeMinCoord,
                                 const std::optional<nanovdb::Coord> &maybeGridSize);

    static variable_list backward(AutogradContext *ctx, variable_list grad_output);
};

struct ReadIntoDenseCmajor : public torch::autograd::Function<ReadIntoDenseCmajor> {
    using variable_list   = torch::autograd::variable_list;
    using AutogradContext = torch::autograd::AutogradContext;
    using Variable        = torch::autograd::Variable;

    static variable_list forward(AutogradContext *ctx,
                                 c10::intrusive_ptr<GridBatchData> grid,
                                 Variable sparseData,
                                 const std::optional<torch::Tensor> &maybeMinCoord,
                                 const std::optional<nanovdb::Coord> &maybeGridSize);

    static variable_list backward(AutogradContext *ctx, variable_list grad_output);
};

} // namespace autograd
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_AUTOGRAD_READINTODENSE_H
