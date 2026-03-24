// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/GridBatchData.h>
#include <fvdb/detail/autograd/Inject.h>
#include <fvdb/detail/ops/Inject.h>

#include <nanovdb/NanoVDB.h>

#include <torch/autograd.h>

namespace fvdb::detail::autograd {

Inject::variable_list
Inject::forward(AutogradContext *ctx,
                const c10::intrusive_ptr<GridBatchData> srcGrid,
                Inject::Variable const &srcFeaturesJData,
                Inject::Variable const &srcFeaturesJOffsets,
                Inject::Variable const &srcFeaturesJIdx,
                Inject::Variable const &srcFeaturesJLIdx,
                const c10::intrusive_ptr<GridBatchData> dstGrid,
                Inject::Variable const &dstFeaturesJData,
                Inject::Variable const &dstFeaturesJOffsets,
                Inject::Variable const &dstFeaturesJIdx,
                Inject::Variable const &dstFeaturesJLIdx) {
    const JaggedTensor srcFeatures =
        JaggedTensor::from_jdata_joffsets_jidx_and_lidx_unsafe(srcFeaturesJData,
                                                               srcFeaturesJOffsets,
                                                               srcFeaturesJIdx,
                                                               srcFeaturesJLIdx,
                                                               srcGrid->batchSize());
    JaggedTensor dstFeatures =
        JaggedTensor::from_jdata_joffsets_jidx_and_lidx_unsafe(dstFeaturesJData,
                                                               dstFeaturesJOffsets,
                                                               dstFeaturesJIdx,
                                                               dstFeaturesJLIdx,
                                                               dstGrid->batchSize());

    ops::inject(*dstGrid, *srcGrid, dstFeatures, srcFeatures);

    ctx->saved_data["src_grid"]     = srcGrid;
    ctx->saved_data["src_jidx"]     = srcFeaturesJIdx;
    ctx->saved_data["src_joffsets"] = srcFeaturesJOffsets;
    ctx->saved_data["src_jlidx"]    = srcFeaturesJLIdx;

    ctx->saved_data["dst_grid"]     = dstGrid;
    ctx->saved_data["dst_jidx"]     = dstFeaturesJIdx;
    ctx->saved_data["dst_joffsets"] = dstFeaturesJOffsets;
    ctx->saved_data["dst_jlidx"]    = dstFeaturesJLIdx;

    return {srcFeatures.jdata(), dstFeatures.jdata()};
}

Inject::variable_list
Inject::backward(AutogradContext *ctx, Inject::variable_list grad_output) {
    // We need to make copies here because we're going to modify these tensors and return them
    auto dLossDSrcFeaturesJData = grad_output[0].clone().contiguous();
    auto dLossDDstFeaturesJData = grad_output[1].clone().contiguous();

    const auto srcGrid     = ctx->saved_data["src_grid"].toCustomClass<GridBatchData>();
    const auto srcJOffsets = ctx->saved_data["src_joffsets"].toTensor();
    const auto srcJIdx     = ctx->saved_data["src_jidx"].toTensor();
    const auto srcJLIdx    = ctx->saved_data["src_jlidx"].toTensor();

    const auto dstGrid     = ctx->saved_data["dst_grid"].toCustomClass<GridBatchData>();
    const auto dstJOffsets = ctx->saved_data["dst_joffsets"].toTensor();
    const auto dstJIdx     = ctx->saved_data["dst_jidx"].toTensor();
    const auto dstJLIdx    = ctx->saved_data["dst_jlidx"].toTensor();

    // Rebuild JaggedTensors for the gradients using saved JOffsets, JIdx, and JLIdx
    JaggedTensor dLossDDstFeatures = JaggedTensor::from_jdata_joffsets_jidx_and_lidx_unsafe(
        dLossDDstFeaturesJData, dstJOffsets, dstJIdx, dstJLIdx, dstGrid->batchSize());
    JaggedTensor dLossDSrcFeatures = JaggedTensor::from_jdata_joffsets_jidx_and_lidx_unsafe(
        dLossDSrcFeaturesJData, srcJOffsets, srcJIdx, srcJLIdx, srcGrid->batchSize());

    ops::inject(*srcGrid, *dstGrid, dLossDSrcFeatures, dLossDDstFeatures);

    // We've already accounted for the dLoss/dSrcFeatures, so we inject zeros at those locations
    // into dLoss/dDstFeatures so the gradient is zero for those locations.
    // This is necessary because we may have injected into the same voxels multiple times, in which
    // case we only want to backpropagate the gradient for the last injected value.
    //
    // Note that we don't actually allocate a tensor of all zeros but instead rely on striding and
    // only allocate a tensor with one zero that we expand to the right shape.
    // This is a memory optimization to avoid allocating large tensors unnecessarily.
    const std::vector<int64_t> allOnesShape(dLossDSrcFeaturesJData.dim(), 1);
    const auto injectZerosJData = torch::zeros(allOnesShape, dLossDSrcFeaturesJData.options())
                                      .expand(dLossDSrcFeaturesJData.sizes());
    const JaggedTensor injectZeros = dLossDSrcFeatures.jagged_like(injectZerosJData);
    ops::inject(*dstGrid, *srcGrid, dLossDDstFeatures, injectZeros);

    return {
        torch::Tensor(),           // srcGrid
        dLossDSrcFeatures.jdata(), // srcFeaturesJData
        torch::Tensor(),           // srcFeaturesJOffsets
        torch::Tensor(),           // srcFeaturesJIdx
        torch::Tensor(),           // srcFeaturesJLIdx
        torch::Tensor(),           // dstGrid
        dLossDDstFeatures.jdata(), // dstFeaturesJData
        torch::Tensor(),           // dstFeaturesJOffsets
        torch::Tensor(),           // dstFeaturesJIdx
        torch::Tensor()            // dstFeaturesJLIdx
    };
}

} // namespace fvdb::detail::autograd
