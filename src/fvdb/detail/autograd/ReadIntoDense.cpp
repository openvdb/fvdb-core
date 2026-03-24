// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/autograd/ReadIntoDense.h>
#include <fvdb/detail/ops/ReadFromDense.h>
#include <fvdb/detail/ops/ReadIntoDense.h>
#include <fvdb/detail/utils/Utils.h>

#include <nanovdb/NanoVDB.h>

namespace fvdb {
namespace detail {
namespace autograd {

ReadIntoDenseCminor::variable_list
ReadIntoDenseCminor::forward(AutogradContext *ctx,
                             c10::intrusive_ptr<GridBatchImpl> grid,
                             Variable sparseData,
                             const std::optional<torch::Tensor> &maybeMinCoord,
                             const std::optional<nanovdb::Coord> &maybeGridSize) {
    nanovdb::CoordBBox gridbb = grid->totalBBox();

    torch::Tensor denseOrigins;
    if (maybeMinCoord.has_value()) {
        denseOrigins = maybeMinCoord.value().to(torch::kInt32).to(sparseData.device());
    } else {
        denseOrigins = coordToTensor(gridbb.min())
                           .to(torch::kInt32)
                           .unsqueeze(0)
                           .repeat({grid->batchSize(), 1})
                           .to(sparseData.device());
    }

    nanovdb::Coord gridSize = gridbb.dim();
    if (maybeGridSize.has_value()) {
        gridSize = maybeGridSize.value();
    }

    // [B, X, Y, Z, -1]
    torch::Tensor ret = ops::readIntoDenseCminor(*grid, sparseData, denseOrigins, gridSize);

    torch::Tensor retReshape = ret.view(
        spliceShape({grid->batchSize(), gridSize[0], gridSize[1], gridSize[2]}, sparseData));

    // Save shape information for backward
    ctx->saved_data["dense_origins"] = denseOrigins;
    torch::Tensor retShape =
        torch::empty({(int64_t)sparseData.dim()}, torch::TensorOptions().dtype(torch::kLong));
    auto acc = retShape.accessor<int64_t, 1>();
    for (int i = 0; i < sparseData.dim(); i++) {
        acc[i] = sparseData.size(i);
    }
    ctx->saved_data["final_shape"] = retShape;
    ctx->saved_data["grid"]        = grid;

    return variable_list({retReshape});
}

ReadIntoDenseCminor::variable_list
ReadIntoDenseCminor::backward(AutogradContext *ctx, variable_list grad_output) {
    torch::Tensor denseOrigins = ctx->saved_data["dense_origins"].toTensor(); // [B, 3]
    std::vector<int64_t> finalShapeTensor =
        intTensor1DToStdVector(ctx->saved_data["final_shape"].toTensor());
    auto grid = ctx->saved_data["grid"].toCustomClass<GridBatchImpl>();

    Variable gradOut = grad_output.at(0); // [B, X, Y, Z, *]

    // [N, -1]
    torch::Tensor ret = ops::readFromDenseCminor(*grid, gradOut, denseOrigins);

    torch::Tensor retReshape = ret.view(finalShapeTensor); // [N, *]

    return {torch::Tensor(), retReshape, torch::Tensor(), torch::Tensor()};
}

ReadIntoDenseCmajor::variable_list
ReadIntoDenseCmajor::forward(AutogradContext *ctx,
                             c10::intrusive_ptr<GridBatchImpl> grid,
                             Variable sparseData,
                             const std::optional<torch::Tensor> &maybeMinCoord,
                             const std::optional<nanovdb::Coord> &maybeGridSize) {
    nanovdb::CoordBBox gridbb = grid->totalBBox();

    torch::Tensor denseOrigins;
    if (maybeMinCoord.has_value()) {
        denseOrigins = maybeMinCoord.value().to(torch::kInt32).to(sparseData.device());
    } else {
        denseOrigins = coordToTensor(gridbb.min())
                           .to(torch::kInt32)
                           .unsqueeze(0)
                           .repeat({grid->batchSize(), 1})
                           .to(sparseData.device());
    }

    nanovdb::Coord gridSize = gridbb.dim();
    if (maybeGridSize.has_value()) {
        gridSize = maybeGridSize.value();
    }

    // [B, -1, X, Y, Z]
    torch::Tensor ret = ops::readIntoDenseCmajor(*grid, sparseData, denseOrigins, gridSize);

    // Reshape [B, -1, X, Y, Z] to [B, *, X, Y, Z]
    std::vector<int64_t> retShapeVec;
    retShapeVec.push_back(static_cast<int64_t>(grid->batchSize()));
    for (int i = 1; i < sparseData.dim(); i++) {
        retShapeVec.push_back(static_cast<int64_t>(sparseData.size(i)));
    }
    retShapeVec.push_back(static_cast<int64_t>(gridSize[0]));
    retShapeVec.push_back(static_cast<int64_t>(gridSize[1]));
    retShapeVec.push_back(static_cast<int64_t>(gridSize[2]));

    torch::Tensor retReshape = ret.view(retShapeVec);

    // Save shape information for backward
    ctx->saved_data["dense_origins"] = denseOrigins;
    torch::Tensor retShape =
        torch::empty({(int64_t)sparseData.dim()}, torch::TensorOptions().dtype(torch::kLong));
    auto acc = retShape.accessor<int64_t, 1>();
    for (int i = 0; i < sparseData.dim(); i++) {
        acc[i] = sparseData.size(i);
    }
    ctx->saved_data["final_shape"] = retShape;
    ctx->saved_data["grid"]        = grid;

    return variable_list({retReshape});
}

ReadIntoDenseCmajor::variable_list
ReadIntoDenseCmajor::backward(AutogradContext *ctx, variable_list grad_output) {
    torch::Tensor denseOrigins = ctx->saved_data["dense_origins"].toTensor(); // [B, 3]
    std::vector<int64_t> finalShapeTensor =
        intTensor1DToStdVector(ctx->saved_data["final_shape"].toTensor());
    auto grid = ctx->saved_data["grid"].toCustomClass<GridBatchImpl>();

    Variable gradOut = grad_output.at(0); // [B, *, X, Y, Z]

    // [N, -1]
    torch::Tensor ret = ops::readFromDenseCmajor(*grid, gradOut, denseOrigins);

    torch::Tensor retReshape = ret.view(finalShapeTensor); // [N, *]

    return {torch::Tensor(), retReshape, torch::Tensor(), torch::Tensor()};
}

} // namespace autograd
} // namespace detail
} // namespace fvdb
