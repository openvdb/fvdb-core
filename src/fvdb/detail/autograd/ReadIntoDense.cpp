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
                             const std::optional<Vec3iBatch> &maybeMinCoord,
                             const std::optional<Vec3i> &maybeGridSize) {
    TORCH_CHECK_VALUE(sparseData.dim() > 1, "sparse_data must have shape [num_voxels, *]");
    TORCH_CHECK_VALUE(sparseData.size(0) == grid->totalVoxels(),
                      "sparseData must have shape (num_voxels, *) where num_voxels = " +
                          std::to_string(grid->totalVoxels()));
    TORCH_CHECK_VALUE(sparseData.is_contiguous(), "sparse_data must be contiguous");
    grid->checkDevice(sparseData);

    // Non empty
    grid->checkNonEmptyGrid();

    nanovdb::CoordBBox gridbb = grid->totalBBox(); // FIXME: Batched should use maximum bounding box
                                                   // which we need to compute

    // Min coord is an integer tensor of shape [3,] or [B, 3] representing the minimum coordinate of
    // the dense tensor
    torch::Tensor denseOrigins;
    if (maybeMinCoord.has_value()) {
        denseOrigins = maybeMinCoord.value()
                           .tensorValue(grid->batchSize(), false /*onlyPositive*/, "min_coord")
                           .to(sparseData.device());
    } else {
        denseOrigins = coordToTensor(gridbb.min())
                           .to(torch::kInt32)
                           .unsqueeze(0)
                           .repeat({grid->batchSize(), 1})
                           .to(sparseData.device());
    }
    TORCH_CHECK_VALUE(denseOrigins.dim() == 2, "min_coord must have shape [3,] or [B, 3]");
    TORCH_CHECK_VALUE(denseOrigins.size(0) == grid->batchSize(),
                      "min_coord must have shape [3,] or [B, 3]");
    TORCH_CHECK_VALUE(denseOrigins.size(1) == 3, "min_coord must have shape [3,] or [B, 3]");

    nanovdb::Coord gridSize = gridbb.dim();
    if (maybeGridSize.has_value()) {
        gridSize = maybeGridSize.value().value();
    }
    TORCH_CHECK_VALUE(gridSize[0] >= 0 && gridSize[1] >= 0 && gridSize[2] >= 0,
                      "grid_size must be non-negative");

    torch::Tensor sparseDataReshape = featureCoalescedView(sparseData); // [N, -1]
    TORCH_CHECK_VALUE(sparseDataReshape.is_contiguous(), "sparse_data must be contiguous");
    torch::Tensor ret = torch::zeros(
        {grid->batchSize(), gridSize[0], gridSize[1], gridSize[2], sparseDataReshape.size(1)},
        sparseData.options()); // [B, X, Y, Z, -1]
    FVDB_DISPATCH_KERNEL(grid->device(), [&]() {
        ops::dispatchReadIntoDenseCminor<DeviceTag>(*grid, sparseDataReshape, denseOrigins, ret);
    });
    torch::Tensor retReshape = ret.view(
        spliceShape({grid->batchSize(), gridSize[0], gridSize[1], gridSize[2]}, sparseData));
    TORCH_CHECK(retReshape.is_contiguous(), "retReshape must be contiguous");

    // Save shape information for backward
    ctx->saved_data["dense_origins"] = denseOrigins;
    ctx->saved_data["grid_size"]     = coordToTensor(gridSize);
    torch::Tensor retShape =
        torch::empty({(int64_t)sparseData.dim()}, torch::TensorOptions().dtype(torch::kLong));
    auto acc = retShape.accessor<int64_t, 1>();
    for (int i = 0; i < sparseData.dim(); i++) {
        acc[i] = sparseData.size(i);
    }
    ctx->saved_data["final_shape"]  = retShape;
    ctx->saved_data["first_dim"]    = sparseDataReshape.size(0);
    ctx->saved_data["last_dim"]     = sparseDataReshape.size(1);
    ctx->saved_data["dummy_tensor"] = torch::empty({0}, sparseData.options());
    ctx->saved_data["grid"]         = grid;

    return variable_list({retReshape});
}

ReadIntoDenseCminor::variable_list
ReadIntoDenseCminor::backward(AutogradContext *ctx, variable_list grad_output) {
    // Use data saved in forward
    torch::Tensor denseOrigins = ctx->saved_data["dense_origins"].toTensor(); // [B, 3]
    int64_t firstDim           = ctx->saved_data["first_dim"].toInt();
    int64_t lastDim            = ctx->saved_data["last_dim"].toInt();
    std::vector<int64_t> finalShapeTensor =
        intTensor1DToStdVector(ctx->saved_data["final_shape"].toTensor());
    torch::TensorOptions sparseDataOpts = ctx->saved_data["dummy_tensor"].toTensor().options();
    auto grid                           = ctx->saved_data["grid"].toCustomClass<GridBatchImpl>();
    Variable gradOut                    = grad_output.at(0);               // [B, X, Y, Z, *]

    torch::Tensor gradOutReshape = featureCoalescedView(gradOut, 4);       // [B, X, Y, Z, -1]

    torch::Tensor ret = torch::zeros({firstDim, lastDim}, sparseDataOpts); // [N, -1]

    FVDB_DISPATCH_KERNEL(grid->device(), [&]() {
        ops::dispatchReadFromDenseCminor<DeviceTag>(*grid, gradOutReshape, denseOrigins, ret);
    });

    torch::Tensor retReshape = ret.view(finalShapeTensor); // [N, *]

    return {torch::Tensor(), retReshape, torch::Tensor(), torch::Tensor()};
}

ReadIntoDenseCmajor::variable_list
ReadIntoDenseCmajor::forward(AutogradContext *ctx,
                             c10::intrusive_ptr<GridBatchImpl> grid,
                             Variable sparseData,
                             const std::optional<Vec3iBatch> &maybeMinCoord,
                             const std::optional<Vec3i> &maybeGridSize) {
    TORCH_CHECK_VALUE(sparseData.dim() > 1, "sparse_data must have shape [num_voxels, *]");
    TORCH_CHECK_VALUE(sparseData.size(0) == grid->totalVoxels(),
                      "sparseData must have shape (num_voxels, *) where num_voxels = " +
                          std::to_string(grid->totalVoxels()));
    TORCH_CHECK_VALUE(sparseData.is_contiguous(), "sparse_data must be contiguous");
    grid->checkDevice(sparseData);

    // Non empty
    grid->checkNonEmptyGrid();

    nanovdb::CoordBBox gridbb = grid->totalBBox(); // FIXME: Batched should use maximum bounding box
                                                   // which we need to compute

    // Min coord is an integer tensor of shape [3,] or [B, 3] representing the minimum coordinate of
    // the dense tensor
    torch::Tensor denseOrigins;
    if (maybeMinCoord.has_value()) {
        denseOrigins = maybeMinCoord.value()
                           .tensorValue(grid->batchSize(), false /*onlyPositive*/, "min_coord")
                           .to(sparseData.device());
    } else {
        denseOrigins = coordToTensor(gridbb.min())
                           .to(torch::kInt32)
                           .unsqueeze(0)
                           .repeat({grid->batchSize(), 1})
                           .to(sparseData.device());
    }
    TORCH_CHECK_VALUE(denseOrigins.dim() == 2, "min_coord must have shape [3,] or [B, 3]");
    TORCH_CHECK_VALUE(denseOrigins.size(0) == grid->batchSize(),
                      "min_coord must have shape [3,] or [B, 3]");
    TORCH_CHECK_VALUE(denseOrigins.size(1) == 3, "min_coord must have shape [3,] or [B, 3]");

    nanovdb::Coord gridSize = gridbb.dim();
    if (maybeGridSize.has_value()) {
        gridSize = maybeGridSize.value().value();
    }
    TORCH_CHECK_VALUE(gridSize[0] >= 0 && gridSize[1] >= 0 && gridSize[2] >= 0,
                      "grid_size must be non-negative");

    torch::Tensor sparseDataReshape = featureCoalescedView(sparseData); // [N, -1]
    TORCH_CHECK_VALUE(sparseDataReshape.is_contiguous(), "sparse_data must be contiguous");
    torch::Tensor ret = torch::zeros(
        {grid->batchSize(), sparseDataReshape.size(1), gridSize[0], gridSize[1], gridSize[2]},
        sparseData.options()); // [B, -1, X, Y, Z]
    FVDB_DISPATCH_KERNEL(grid->device(), [&]() {
        ops::dispatchReadIntoDenseCmajor<DeviceTag>(*grid, sparseDataReshape, denseOrigins, ret);
    });

    // Create the shape of the output view.
    std::vector<int64_t> retShapeVec;
    retShapeVec.push_back(static_cast<int64_t>(grid->batchSize()));
    for (int i = 1; i < sparseData.dim(); i++) {
        retShapeVec.push_back(static_cast<int64_t>(sparseData.size(i)));
    }
    retShapeVec.push_back(static_cast<int64_t>(gridSize[0]));
    retShapeVec.push_back(static_cast<int64_t>(gridSize[1]));
    retShapeVec.push_back(static_cast<int64_t>(gridSize[2]));

    // Re-order the dimensions of the tensor to [B, -1, X, Y, Z]
    torch::Tensor retReshape = ret.view(retShapeVec);
    TORCH_CHECK(retReshape.is_contiguous(), "retReshape must be contiguous");

    // Save shape information for backward
    ctx->saved_data["dense_origins"] = denseOrigins;
    ctx->saved_data["grid_size"]     = coordToTensor(gridSize);
    torch::Tensor retShape =
        torch::empty({(int64_t)sparseData.dim()}, torch::TensorOptions().dtype(torch::kLong));
    auto acc = retShape.accessor<int64_t, 1>();
    for (int i = 0; i < sparseData.dim(); i++) {
        acc[i] = sparseData.size(i);
    }
    ctx->saved_data["final_shape"]  = retShape;
    ctx->saved_data["first_dim"]    = sparseDataReshape.size(0);
    ctx->saved_data["last_dim"]     = sparseDataReshape.size(1);
    ctx->saved_data["dummy_tensor"] = torch::empty({0}, sparseData.options());
    ctx->saved_data["grid"]         = grid;

    return variable_list({retReshape});
}

ReadIntoDenseCmajor::variable_list
ReadIntoDenseCmajor::backward(AutogradContext *ctx, variable_list grad_output) {
    // Use data saved in forward
    torch::Tensor denseOrigins = ctx->saved_data["dense_origins"].toTensor(); // [B, 3]
    int64_t firstDim           = ctx->saved_data["first_dim"].toInt();
    int64_t lastDim            = ctx->saved_data["last_dim"].toInt();
    std::vector<int64_t> finalShapeTensor =
        intTensor1DToStdVector(ctx->saved_data["final_shape"].toTensor());
    torch::TensorOptions sparseDataOpts = ctx->saved_data["dummy_tensor"].toTensor().options();
    auto grid                           = ctx->saved_data["grid"].toCustomClass<GridBatchImpl>();
    Variable gradOut                    = grad_output.at(0);                    // [B, *, X, Y, Z]

    torch::Tensor gradOutReshape = featureCoalescedViewTrailing(gradOut, 1, 3); // [B, -1, X, Y, Z]

    torch::Tensor ret = torch::zeros({firstDim, lastDim}, sparseDataOpts);      // [N, -1]

    FVDB_DISPATCH_KERNEL(grid->device(), [&]() {
        ops::dispatchReadFromDenseCmajor<DeviceTag>(*grid, gradOutReshape, denseOrigins, ret);
    });

    torch::Tensor retReshape = ret.view(finalShapeTensor); // [N, *]

    return {torch::Tensor(), retReshape, torch::Tensor(), torch::Tensor()};
}

} // namespace autograd
} // namespace detail
} // namespace fvdb
