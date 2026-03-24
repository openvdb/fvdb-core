// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/autograd/SplatIntoGrid.h>
#include <fvdb/detail/ops/SampleBezier.h>
#include <fvdb/detail/ops/SampleTrilinear.h>
#include <fvdb/detail/ops/SplatBezier.h>
#include <fvdb/detail/ops/SplatTrilinear.h>

namespace fvdb {
namespace detail {
namespace autograd {

SplatIntoGridTrilinear::variable_list
SplatIntoGridTrilinear::forward(SplatIntoGridTrilinear::AutogradContext *ctx,
                                c10::intrusive_ptr<GridBatchData> grid,
                                SplatIntoGridTrilinear::JaggedVariable points,
                                SplatIntoGridTrilinear::Variable pointData) {
    torch::Tensor outGridData = ops::splatTrilinear(*grid, points, pointData);

    // Save data for backward in context
    ctx->save_for_backward({pointData, points.jdata(), points.joffsets(), points.jlidx()});
    ctx->saved_data["grid"] = grid;

    return variable_list({outGridData});
}

SplatIntoGridTrilinear::variable_list
SplatIntoGridTrilinear::backward(SplatIntoGridTrilinear::AutogradContext *ctx,
                                 SplatIntoGridTrilinear::variable_list grad_output) {
    // Use data saved in forward
    variable_list saved = ctx->get_saved_variables();
    Variable pointData  = saved.at(0);          // [B*M, *]

    Variable pointCoords   = saved.at(1);       // [B*M, 3]
    Variable pointJOffsets = saved.at(2);       // [B,]
    Variable pointsJLidx   = saved.at(3);       // [B,]
    auto grid              = ctx->saved_data["grid"].toCustomClass<GridBatchData>();
    Variable gradOut       = grad_output.at(0); // [N, *]

    auto ret = ops::sampleTrilinear(
        *grid,
        JaggedTensor::from_data_offsets_and_list_ids(pointCoords, pointJOffsets, pointsJLidx),
        gradOut);

    return {torch::Tensor(), torch::Tensor(), ret[0]};
}

SplatIntoGridBezier::variable_list
SplatIntoGridBezier::forward(SplatIntoGridBezier::AutogradContext *ctx,
                             c10::intrusive_ptr<GridBatchData> grid,
                             SplatIntoGridBezier::JaggedVariable points,
                             SplatIntoGridBezier::Variable pointData) {
    torch::Tensor outGridData = ops::splatBezier(*grid, points, pointData);

    // Save data for backward in context
    ctx->save_for_backward({pointData, points.jdata(), points.joffsets(), points.jlidx()});
    ctx->saved_data["grid"] = grid;

    return variable_list({outGridData});
}

SplatIntoGridBezier::variable_list
SplatIntoGridBezier::backward(SplatIntoGridBezier::AutogradContext *ctx,
                              SplatIntoGridBezier::variable_list grad_output) {
    // Use data saved in forward
    variable_list saved = ctx->get_saved_variables();
    Variable pointData  = saved.at(0);    // [B*M, *]

    Variable pointCoords   = saved.at(1); // [B*M, 3]
    Variable pointJOffsets = saved.at(2); // [B,]
    Variable pointsJLidx   = saved.at(3); // [B,]

    auto grid        = ctx->saved_data["grid"].toCustomClass<GridBatchData>();
    Variable gradOut = grad_output.at(0); // [N, *]

    torch::Tensor outGradIn = ops::sampleBezier(
        *grid,
        JaggedTensor::from_data_offsets_and_list_ids(pointCoords, pointJOffsets, pointsJLidx),
        gradOut)[0];

    return {torch::Tensor(), torch::Tensor(), outGradIn};
}

} // namespace autograd
} // namespace detail
} // namespace fvdb
