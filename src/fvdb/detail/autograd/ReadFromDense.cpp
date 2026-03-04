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
                             const Vec3iBatch &denseOrigins) {
    TORCH_CHECK_VALUE(denseData.dim() > 4, "dense data must have shape [B, X, Y, Z, C]");
    TORCH_CHECK_VALUE(denseData.size(0) == grid->batchSize(),
                      "dense data must have shape [B, X, Y, Z, *]");
    TORCH_CHECK_VALUE(denseData.is_contiguous(), "sparse_data must be contiguous");
    grid->checkDevice(denseData);

    // Non empty
    grid->checkNonEmptyGrid();

    // [B, X, Y, Z, -1]
    torch::Tensor denseDataReshape = featureCoalescedView(denseData, 4);

    // [N, -1]
    torch::Tensor ret =
        torch::zeros({grid->totalVoxels(), denseDataReshape.size(4)}, denseData.options());

    // nanovdb::Coord denseOriginNvdb = tensorToCoord(denseOrigins);
    // NanoVDB coordinates are int32
    torch::Tensor denseOriginsI32 =
        denseOrigins.tensorValue(grid->batchSize(), false /*onlyPositive*/, "dense_origins")
            .to(denseData.device());

    FVDB_DISPATCH_KERNEL(grid->device(), [&]() {
        ops::dispatchReadFromDenseCminor<DeviceTag>(*grid, denseDataReshape, denseOriginsI32, ret);
    });

    // Reshape [B, N, -1] to [B, N, *] given [B, X, Y, Z, *]
    torch::Tensor retReshape = ret.view(spliceShape({grid->totalVoxels()}, denseData, 4));

    // Save shape information for backward
    ctx->saved_data["dense_origin"] = denseOriginsI32;
    ctx->saved_data["grid_size"] =
        coordToTensor(nanovdb::Coord(denseData.size(1), denseData.size(2), denseData.size(3)));
    ctx->saved_data["grid"]         = grid;
    ctx->saved_data["dummy_tensor"] = torch::empty({0}, denseData.options());
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
    // Use data saved in forward
    torch::Tensor denseOrigins         = ctx->saved_data["dense_origin"].toTensor(); // [B, 3]
    nanovdb::Coord gridSize            = tensorToCoord(ctx->saved_data["grid_size"].toTensor());
    auto grid                          = ctx->saved_data["grid"].toCustomClass<GridBatchImpl>();
    torch::TensorOptions denseDataOpts = ctx->saved_data["dummy_tensor"].toTensor().options();
    std::vector<int64_t> finalShapeTensor =
        intTensor1DToStdVector(ctx->saved_data["final_shape"].toTensor());

    Variable gradOut             = grad_output.at(0);             // [N, *]
    torch::Tensor gradOutReshape = featureCoalescedView(gradOut); // [N, -1]
    torch::Tensor ret            = torch::zeros(
        {grid->batchSize(), gridSize[0], gridSize[1], gridSize[2], gradOutReshape.size(1)},
        denseDataOpts);                                // [B, X, Y, Z, -1]

    FVDB_DISPATCH_KERNEL(grid->device(), [&]() {
        ops::dispatchReadIntoDenseCminor<DeviceTag>(*grid, gradOutReshape, denseOrigins, ret);
    });

    torch::Tensor retReshape = ret.view(finalShapeTensor); // [B, W, H, D, *]

    return {torch::Tensor(), retReshape, torch::Tensor()};
}

ReadFromDenseCmajor::variable_list
ReadFromDenseCmajor::forward(AutogradContext *ctx,
                             c10::intrusive_ptr<GridBatchImpl> grid,
                             Variable denseData,
                             const Vec3iBatch &denseOrigins) {
    TORCH_CHECK_VALUE(denseData.dim() > 4, "dense data must have shape [B, *, X, Y, Z]");
    TORCH_CHECK_VALUE(denseData.size(0) == grid->batchSize(),
                      "dense data must have shape [B, *, X, Y, Z]");
    TORCH_CHECK_VALUE(denseData.is_contiguous(), "sparse_data must be contiguous");
    grid->checkDevice(denseData);

    // Non empty
    grid->checkNonEmptyGrid();
    auto const feature_rank = denseData.dim() - 4;
    TORCH_CHECK_VALUE(feature_rank > 0, "feature_rank must be greater than 0");

    // [B, -1, X, Y, Z]
    torch::Tensor denseDataReshape = featureCoalescedViewTrailing(denseData, 1, 3);

    auto const batch_count   = denseDataReshape.size(0); // B
    auto const feature_count = denseDataReshape.size(1); // F
    auto const dense_x       = denseDataReshape.size(2); // X
    auto const dense_y       = denseDataReshape.size(3); // Y
    auto const dense_z       = denseDataReshape.size(4); // Z
    auto const voxel_count   = grid->totalVoxels();      // N

    // [N, -1]
    torch::Tensor ret = torch::zeros({voxel_count, feature_count}, denseData.options());

    // nanovdb::Coord denseOriginNvdb = tensorToCoord(denseOrigins);
    // NanoVDB coordinates are int32
    torch::Tensor denseOriginsI32 =
        denseOrigins.tensorValue(batch_count, false /*onlyPositive*/, "dense_origins")
            .to(denseData.device());

    FVDB_DISPATCH_KERNEL(grid->device(), [&]() {
        ops::dispatchReadFromDenseCmajor<DeviceTag>(*grid, denseDataReshape, denseOriginsI32, ret);
    });

    // Reshape [N, -1] to [N, *] given [B, *, X, Y, Z]
    std::vector<int64_t> retShapeVec;
    retShapeVec.push_back(voxel_count);
    for (int i = 0; i < feature_rank; ++i) {
        // Offset by 1 because the first dimension is the batch dimension
        retShapeVec.push_back(denseData.size(i + 1));
    }
    torch::Tensor retReshape = ret.view(retShapeVec);

    // Save shape information for backward
    ctx->saved_data["dense_origin"] = denseOriginsI32;
    ctx->saved_data["grid_size"]    = coordToTensor(nanovdb::Coord(dense_x, dense_y, dense_z));
    ctx->saved_data["grid"]         = grid;
    ctx->saved_data["dummy_tensor"] = torch::empty({0}, denseData.options());
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
    // Use data saved in forward
    torch::Tensor denseOrigins         = ctx->saved_data["dense_origin"].toTensor(); // [B, 3]
    nanovdb::Coord gridSize            = tensorToCoord(ctx->saved_data["grid_size"].toTensor());
    auto grid                          = ctx->saved_data["grid"].toCustomClass<GridBatchImpl>();
    torch::TensorOptions denseDataOpts = ctx->saved_data["dummy_tensor"].toTensor().options();
    std::vector<int64_t> finalShapeTensor =
        intTensor1DToStdVector(ctx->saved_data["final_shape"].toTensor());

    Variable gradOut             = grad_output.at(0);             // [N, *]
    torch::Tensor gradOutReshape = featureCoalescedView(gradOut); // [N, -1]
    torch::Tensor ret            = torch::zeros(
        {grid->batchSize(), gradOutReshape.size(1), gridSize[0], gridSize[1], gridSize[2]},
        denseDataOpts);                                // [B, -1, X, Y, Z]

    FVDB_DISPATCH_KERNEL(grid->device(), [&]() {
        ops::dispatchReadIntoDenseCmajor<DeviceTag>(*grid, gradOutReshape, denseOrigins, ret);
    });

    torch::Tensor retReshape = ret.view(finalShapeTensor); // [B, *, X, Y, Z]

    return {torch::Tensor(), retReshape, torch::Tensor()};
}

} // namespace autograd
} // namespace detail
} // namespace fvdb
