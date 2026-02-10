// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#include <fvdb/detail/autograd/SparseConvolutionKernelMap.h>
#include <fvdb/detail/ops/convolution/backend/SparseConvolutionKernelMap.h>
#include <fvdb/detail/utils/Utils.h>

#include <torch/autograd.h>

#include <string>
#include <vector>

namespace fvdb {
namespace detail {
namespace autograd {

SparseConvolutionKernelMap::variable_list
SparseConvolutionKernelMap::forward(AutogradContext *ctx,
                                    Variable inFeatures,
                                    Variable kernels,
                                    Variable neighborMap,
                                    Variable neighborSizes,
                                    int64_t srcVoxels,
                                    int64_t dstVoxels,
                                    bool middleAcceleration,
                                    bool transposed) {
    torch::Tensor nbmaps  = neighborMap;
    torch::Tensor nbsizes = neighborSizes;

    // Check features
    TORCH_CHECK_VALUE(inFeatures.is_contiguous(), "features must be contiguous");
    TORCH_CHECK_TYPE(inFeatures.is_floating_point(), "features must have a floating point type");
    TORCH_CHECK_VALUE(
        inFeatures.dim() == 2,
        std::string("Expected features to have 2 dimensions (shape (n, nF)) but got ") +
            std::to_string(inFeatures.dim()) + " dimensions");

    // Check kernels
    TORCH_CHECK_TYPE(kernels.is_floating_point(), "kernels must have a floating point type");
    for (int i = 0; i < kernels.dim(); i += 1) {
        TORCH_CHECK_VALUE(kernels.size(i) != 0,
                          "kernels tensor has zero dimension (dim = " + std::to_string(i) + ")");
    }
    // Check neighbor map
    TORCH_CHECK(nbmaps.is_contiguous() && nbmaps.scalar_type() == torch::kInt32,
                "nbmaps must be contiguous int32");
    TORCH_CHECK(nbsizes.is_contiguous() && nbsizes.scalar_type() == torch::kInt32,
                "nbsizes must be contiguous int32");

    auto opt             = torch::TensorOptions().dtype(torch::kInt32).device(inFeatures.device());
    torch::Tensor kWidth = torch::empty(
        {
            3,
        },
        opt);
    if (!transposed) {
        TORCH_CHECK_VALUE(inFeatures.size(0) == srcVoxels,
                          "The number of input features must match the number of voxels");
        TORCH_CHECK_VALUE(
            kernels.dim() == 5,
            std::string(
                "Expected kernels to have 5 dimensions (shape (out_ch, in_ch, d, h, w)) but got ") +
                std::to_string(kernels.dim()) + " dimensions");
        TORCH_CHECK_VALUE(
            kernels.size(1) == inFeatures.size(1),
            "Expected input channels of kernels (" + std::to_string(kernels.size(1)) +
                ") to equal input channels of features: " + std::to_string(inFeatures.size(1)));
        const int outC = kernels.size(0), inC = kernels.size(1);
        kWidth[0] = kernels.size(2);
        kWidth[1] = kernels.size(3);
        kWidth[2] = kernels.size(4);
        kernels   = kernels.permute({2, 3, 4, 1, 0}).reshape({-1, inC, outC}).contiguous();
    } else {
        TORCH_CHECK_VALUE(inFeatures.size(0) == dstVoxels,
                          "The number of input features must match the number of voxels");
        TORCH_CHECK_VALUE(
            kernels.dim() == 5,
            std::string(
                "Expected kernels to have 5 dimensions (shape (in_ch, out_ch, d, h, w)) but got ") +
                std::to_string(kernels.dim()) + " dimensions");
        TORCH_CHECK_VALUE(
            kernels.size(0) == inFeatures.size(1),
            "Expected input channels of kernels (" + std::to_string(kernels.size(0)) +
                ") to equal input channels of features: " + std::to_string(inFeatures.size(1)));
        const int inC = kernels.size(0), outC = kernels.size(1);
        kWidth[0] = kernels.size(2);
        kWidth[1] = kernels.size(3);
        kWidth[2] = kernels.size(4);
        kernels   = kernels.permute({2, 3, 4, 0, 1}).reshape({-1, inC, outC}).contiguous();
    }

    // Save for backward
    ctx->save_for_backward({inFeatures, kernels, nbmaps, nbsizes});
    ctx->saved_data["transposed"]   = transposed;
    ctx->saved_data["kernel_width"] = kWidth;

    torch::Tensor output;
    if (dstVoxels > 0) {
        auto opt = torch::TensorOptions().dtype(inFeatures.dtype()).device(inFeatures.device());
        if (!transposed) {
            output = torch::zeros({dstVoxels, kernels.size(-1)}, opt);
        } else {
            output = torch::zeros({srcVoxels, kernels.size(-1)}, opt);
        }
        // NOTE: Francis: We need .cpu().contiguous() here because we copied the convolution
        //       implementation from torch_sparse which runs std::max_element on a pointer
        //       to this tensor D: which is fucking awful...
        // TODO: Francis: Fix torch_sparse conv to be robust
        FVDB_DISPATCH_KERNEL_DEVICE(inFeatures.device(), [&]() {
            ops::dispatchSparseConvolutionKernelMap<DeviceTag>(inFeatures,
                                                               output,
                                                               kernels,
                                                               nbmaps,
                                                               nbsizes.cpu().contiguous(),
                                                               transposed,
                                                               middleAcceleration);
        });
    } else {
        auto opt = torch::TensorOptions().dtype(inFeatures.dtype()).device(inFeatures.device());
        output   = torch::empty({0, kernels.size(-1)}, opt);
    }

    return {output};
}

SparseConvolutionKernelMap::variable_list
SparseConvolutionKernelMap::backward(AutogradContext *ctx, variable_list grad_output) {
    // Use data saved in forward
    variable_list saved  = ctx->get_saved_variables();
    Variable inFeatures  = saved.at(0);
    Variable kernels     = saved.at(1);
    Variable nbmaps      = saved.at(2);
    Variable nbsizes     = saved.at(3);
    bool transposed      = ctx->saved_data["transposed"].toBool();
    torch::Tensor kWidth = ctx->saved_data["kernel_width"].toTensor();

    torch::Tensor gradInput  = torch::zeros_like(inFeatures);
    torch::Tensor gradWeight = torch::zeros_like(kernels);

    Variable gradOut = grad_output.at(0);

    if (gradOut.size(0) != 0) {
        FVDB_DISPATCH_KERNEL_DEVICE(gradOut.device(), [&]() {
            ops::dispatchSparseConvolutionKernelMapGrad<DeviceTag>(inFeatures,
                                                                   gradInput,
                                                                   gradOut.contiguous(),
                                                                   kernels,
                                                                   gradWeight,
                                                                   nbmaps,
                                                                   nbsizes.cpu().contiguous(),
                                                                   transposed);
        });
    }

    const int outC = gradWeight.size(-1), inC = gradWeight.size(-2);
    if (!transposed) {
        gradWeight = gradWeight
                         .reshape({kWidth[2].item<int32_t>(),
                                   kWidth[1].item<int32_t>(),
                                   kWidth[0].item<int32_t>(),
                                   inC,
                                   outC})
                         .permute({4, 3, 2, 1, 0});
    } else {
        gradWeight = gradWeight
                         .reshape({kWidth[2].item<int32_t>(),
                                   kWidth[1].item<int32_t>(),
                                   kWidth[0].item<int32_t>(),
                                   inC,
                                   outC})
                         .permute({3, 4, 2, 1, 0});
    }
    // Gradients for: inFeatures, kernels, neighborMap, neighborSizes, srcVoxels, dstVoxels,
    //                middleAcceleration, transposed
    return {gradInput,
            gradWeight,
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor()};
}

} // namespace autograd
} // namespace detail
} // namespace fvdb
