// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/autograd/SampleGrid.h>
#include <fvdb/detail/ops/SampleGridBezier.h>
#include <fvdb/detail/ops/SampleGridBezierWithGrad.h>
#include <fvdb/detail/ops/SampleGridBezierWithGradBackward.h>
#include <fvdb/detail/ops/SampleGridTrilinear.h>
#include <fvdb/detail/ops/SampleGridTrilinearWithGrad.h>
#include <fvdb/detail/ops/SampleGridTrilinearWithGradBackward.h>
#include <fvdb/detail/ops/SplatIntoGridBezier.h>
#include <fvdb/detail/ops/SplatIntoGridTrilinear.h>

namespace fvdb {
namespace detail {
namespace autograd {

SampleGridTrilinear::variable_list
SampleGridTrilinear::forward(SampleGridTrilinear::AutogradContext *ctx,
                             c10::intrusive_ptr<GridBatchImpl> grid,
                             SampleGridTrilinear::JaggedVariable points,
                             SampleGridTrilinear::Variable data,
                             bool returnGrad) {
    std::vector<torch::Tensor> ret;
    if (returnGrad) {
        ret = ops::sampleGridTrilinearWithGrad(*grid, points, data);
    } else {
        ret = ops::sampleGridTrilinear(*grid, points, data);
    }

    // Save data for backward in context
    ctx->save_for_backward({data, points.jdata(), points.joffsets(), points.jlidx()});
    ctx->saved_data["grid"]        = grid;
    ctx->saved_data["return_grad"] = returnGrad;
    return ret;
}

SampleGridTrilinear::variable_list
SampleGridTrilinear::backward(SampleGridTrilinear::AutogradContext *ctx,
                              SampleGridTrilinear::variable_list grad_output) {
    // Use data saved in forward
    variable_list saved = ctx->get_saved_variables();
    Variable data       = saved.at(0);

    Variable pointCoords   = saved.at(1);
    Variable pointJOffsets = saved.at(2);
    Variable pointsJLidx   = saved.at(3);

    auto grid        = ctx->saved_data["grid"].toCustomClass<GridBatchImpl>();
    bool returnGrad  = ctx->saved_data["return_grad"].toBool();
    Variable gradOut = grad_output.at(0); // [B*M, *]

    torch::Tensor outGrad;
    if (returnGrad) {
        Variable gradPtsGrad = grad_output.at(1); // [B*M, -1, 3]
        outGrad              = ops::sampleGridTrilinearWithGradBackward(
            *grid,
            JaggedTensor::from_data_offsets_and_list_ids(pointCoords, pointJOffsets, pointsJLidx),
            data,
            gradOut,
            gradPtsGrad);
    } else {
        outGrad = ops::splatIntoGridTrilinear(
            *grid,
            JaggedTensor::from_data_offsets_and_list_ids(pointCoords, pointJOffsets, pointsJLidx),
            gradOut);
    }

    return {torch::Tensor(), torch::Tensor(), outGrad, torch::Tensor()};
}

SampleGridBezier::variable_list
SampleGridBezier::forward(SampleGridBezier::AutogradContext *ctx,
                          c10::intrusive_ptr<GridBatchImpl> grid,
                          SampleGridBezier::JaggedVariable points,
                          SampleGridBezier::Variable data,
                          bool returnGrad) {
    std::vector<torch::Tensor> ret;
    if (returnGrad) {
        ret = ops::sampleGridBezierWithGrad(*grid, points, data);
    } else {
        ret = ops::sampleGridBezier(*grid, points, data);
    }

    // Save data for backward in context
    ctx->save_for_backward({data, points.jdata(), points.joffsets(), points.jlidx()});
    ctx->saved_data["grid"]        = grid;
    ctx->saved_data["return_grad"] = returnGrad;

    return ret;
}

SampleGridBezier::variable_list
SampleGridBezier::backward(SampleGridBezier::AutogradContext *ctx,
                           SampleGridBezier::variable_list grad_output) {
    // Use data saved in forward
    variable_list saved = ctx->get_saved_variables();
    Variable data       = saved.at(0);

    Variable pointCoords   = saved.at(1);
    Variable pointJOffsets = saved.at(2);
    Variable pointsJLidx   = saved.at(3);

    auto grid        = ctx->saved_data["grid"].toCustomClass<GridBatchImpl>();
    bool returnGrad  = ctx->saved_data["return_grad"].toBool();
    Variable gradOut = grad_output.at(0); // [B*M, *]

    Variable outGrad;
    if (returnGrad) {
        Variable gradPtsGrad = grad_output.at(1); // [B*M, -1, 3]
        outGrad              = ops::sampleGridBezierWithGradBackward(
            *grid,
            JaggedTensor::from_data_offsets_and_list_ids(pointCoords, pointJOffsets, pointsJLidx),
            gradOut,
            gradPtsGrad,
            data);
    } else {
        outGrad = ops::splatIntoGridBezier(
            *grid,
            JaggedTensor::from_data_offsets_and_list_ids(pointCoords, pointJOffsets, pointsJLidx),
            gradOut);
    }

    return {torch::Tensor(), torch::Tensor(), outGrad, torch::Tensor()};
}

} // namespace autograd
} // namespace detail
} // namespace fvdb
