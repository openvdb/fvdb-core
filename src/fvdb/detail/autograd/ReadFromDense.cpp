// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/autograd/ReadFromDense.h>
#include <fvdb/detail/ops/ReadFromDense.h>
#include <fvdb/detail/ops/ReadIntoDense.h>
#include <fvdb/detail/utils/Utils.h>

#include <nanovdb/NanoVDB.h>

namespace fvdb {
namespace detail {
namespace autograd {

ReadFromDenseCminor::variable_list
ReadFromDenseCminor::forward(AutogradContext *ctx,
                             c10::intrusive_ptr<GridBatchImpl> grid,
                             Variable denseData,
                             const torch::Tensor &denseOrigins) {
    torch::Tensor denseOriginsI32 = denseOrigins.to(torch::kInt32).to(denseData.device());

    // [N, -1]
    torch::Tensor ret = ops::readFromDenseCminor(*grid, denseData, denseOriginsI32);

    // Reshape [N, -1] to [N, *] given [B, X, Y, Z, *]
    torch::Tensor retReshape = ret.view(spliceShape({grid->totalVoxels()}, denseData, 4));

    // Save shape information for backward
    ctx->saved_data["dense_origin"] = denseOriginsI32;
    ctx->saved_data["grid_size"] =
        coordToTensor(nanovdb::Coord(denseData.size(1), denseData.size(2), denseData.size(3)));
    ctx->saved_data["grid"] = grid;
    torch::Tensor retShape =
        torch::empty({(int64_t)denseData.dim()}, torch::TensorOptions().dtype(torch::kLong));
    auto acc = retShape.accessor<int64_t, 1>();
    for (int i = 0; i < denseData.dim(); i++) {
        acc[i] = denseData.size(i);
    }
    ctx->saved_data["final_shape"] = retShape;

    return variable_list({retReshape}); // [N, *]
}

ReadFromDenseCminor::variable_list
ReadFromDenseCminor::backward(AutogradContext *ctx, variable_list grad_output) {
    torch::Tensor denseOrigins = ctx->saved_data["dense_origin"].toTensor(); // [B, 3]
    nanovdb::Coord gridSize    = tensorToCoord(ctx->saved_data["grid_size"].toTensor());
    auto grid                  = ctx->saved_data["grid"].toCustomClass<GridBatchImpl>();
    std::vector<int64_t> finalShapeTensor =
        intTensor1DToStdVector(ctx->saved_data["final_shape"].toTensor());

    Variable gradOut = grad_output.at(0); // [N, *]

    // [B, X, Y, Z, -1]
    torch::Tensor ret = ops::readIntoDenseCminor(*grid, gradOut, denseOrigins, gridSize);

    torch::Tensor retReshape = ret.view(finalShapeTensor); // [B, W, H, D, *]

    return {torch::Tensor(), retReshape, torch::Tensor()};
}

ReadFromDenseCmajor::variable_list
ReadFromDenseCmajor::forward(AutogradContext *ctx,
                             c10::intrusive_ptr<GridBatchImpl> grid,
                             Variable denseData,
                             const torch::Tensor &denseOrigins) {
    torch::Tensor denseOriginsI32 = denseOrigins.to(torch::kInt32).to(denseData.device());

    // [N, -1]
    torch::Tensor ret = ops::readFromDenseCmajor(*grid, denseData, denseOriginsI32);

    // Reshape [N, -1] to [N, *] given [B, *, X, Y, Z]
    auto const feature_rank = denseData.dim() - 4;
    std::vector<int64_t> retShapeVec;
    retShapeVec.push_back(grid->totalVoxels());
    for (int i = 0; i < feature_rank; ++i) {
        retShapeVec.push_back(denseData.size(i + 1));
    }
    torch::Tensor retReshape = ret.view(retShapeVec);

    // Save shape information for backward
    ctx->saved_data["dense_origin"] = denseOriginsI32;
    auto const dense_x              = denseData.size(denseData.dim() - 3);
    auto const dense_y              = denseData.size(denseData.dim() - 2);
    auto const dense_z              = denseData.size(denseData.dim() - 1);
    ctx->saved_data["grid_size"]    = coordToTensor(nanovdb::Coord(dense_x, dense_y, dense_z));
    ctx->saved_data["grid"]         = grid;
    torch::Tensor retShape =
        torch::empty({(int64_t)denseData.dim()}, torch::TensorOptions().dtype(torch::kLong));
    auto acc = retShape.accessor<int64_t, 1>();
    for (int i = 0; i < denseData.dim(); i++) {
        acc[i] = denseData.size(i);
    }
    ctx->saved_data["final_shape"] = retShape;

    return variable_list({retReshape}); // [N, *]
}

ReadFromDenseCmajor::variable_list
ReadFromDenseCmajor::backward(AutogradContext *ctx, variable_list grad_output) {
    torch::Tensor denseOrigins = ctx->saved_data["dense_origin"].toTensor(); // [B, 3]
    nanovdb::Coord gridSize    = tensorToCoord(ctx->saved_data["grid_size"].toTensor());
    auto grid                  = ctx->saved_data["grid"].toCustomClass<GridBatchImpl>();
    std::vector<int64_t> finalShapeTensor =
        intTensor1DToStdVector(ctx->saved_data["final_shape"].toTensor());

    Variable gradOut = grad_output.at(0); // [N, *]

    // [B, -1, X, Y, Z]
    torch::Tensor ret = ops::readIntoDenseCmajor(*grid, gradOut, denseOrigins, gridSize);

    torch::Tensor retReshape = ret.view(finalShapeTensor); // [B, *, X, Y, Z]

    return {torch::Tensor(), retReshape, torch::Tensor()};
}

} // namespace autograd
} // namespace detail
} // namespace fvdb
