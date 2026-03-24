// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/autograd/TransformPoints.h>
#include <fvdb/detail/ops/VoxelToWorld.h>
#include <fvdb/detail/utils/Utils.h>

#include <nanovdb/NanoVDB.h>

#include <vector>

namespace fvdb {
namespace detail {
namespace autograd {

TransformPoints::variable_list
TransformPoints::forward(TransformPoints::AutogradContext *ctx,
                         c10::intrusive_ptr<GridBatchImpl> grid,
                         TransformPoints::JaggedVariable points,
                         Variable pointsData,
                         bool isInverse,
                         bool isDual) {
    // FIXME: (@fwilliams) This is a hack because we need to pass tensors to the autograd engine :/
    JaggedTensor pointsWrap = points.jagged_like(pointsData);

    torch::Tensor outTxPoints;
    if (isInverse) {
        outTxPoints = ops::worldToVoxel(*grid, pointsWrap, !isDual);
    } else {
        outTxPoints = ops::voxelToWorld(*grid, pointsWrap, !isDual);
    }

    ctx->save_for_backward({points.joffsets(), points.jlidx()});

    ctx->saved_data["grid"]      = grid;
    ctx->saved_data["isDual"]    = isDual;
    ctx->saved_data["isInverse"] = isInverse;

    return {outTxPoints}; // [B*N, 3]
}

TransformPoints::variable_list
TransformPoints::backward(TransformPoints::AutogradContext *ctx,
                          TransformPoints::variable_list grad_output) {
    variable_list saved = ctx->get_saved_variables();

    Variable pointsJOffsets = saved.at(0);
    Variable pointsJLidx    = saved.at(1);
    Variable gradOut        = grad_output.at(0); // [B*N, 3]

    // Use data saved in forward
    auto grid            = ctx->saved_data["grid"].toCustomClass<GridBatchImpl>();
    const bool isDual    = ctx->saved_data["isDual"].toBool();
    const bool isInverse = ctx->saved_data["isInverse"].toBool();

    Variable outGradIn;
    if (isInverse) {
        outGradIn = ops::worldToVoxelBackward(
            *grid,
            JaggedTensor::from_data_offsets_and_list_ids(gradOut, pointsJOffsets, pointsJLidx),
            !isDual);
    } else {
        outGradIn = ops::voxelToWorldBackward(
            *grid,
            JaggedTensor::from_data_offsets_and_list_ids(gradOut, pointsJOffsets, pointsJLidx),
            !isDual);
    }

    // Variable outGradIn = outGradInReshape.reshape(getShapeButReplaceFirstDim(fineData.size(0),
    // gradOut));
    return {torch::Tensor(), torch::Tensor(), outGradIn, torch::Tensor(), torch::Tensor()};
}

} // namespace autograd
} // namespace detail
} // namespace fvdb
