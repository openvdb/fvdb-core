// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/gsplat/GaussianDeduplicatePixels.h>

namespace fvdb::detail::ops {

std::tuple<fvdb::JaggedTensor, torch::Tensor, bool>
deduplicatePixels(const fvdb::JaggedTensor &pixelsToRender,
                  int64_t imageWidth,
                  int64_t imageHeight) {
    const auto totalPixels = pixelsToRender.rsize(0);
    if (totalPixels == 0) {
        auto emptyInverse = torch::empty({0}, pixelsToRender.jdata().options().dtype(torch::kLong));
        return {pixelsToRender, emptyInverse, false};
    }

    const auto device               = pixelsToRender.device();
    const auto jdata                = pixelsToRender.jdata();
    const auto jidx                 = pixelsToRender.jidx();
    const int64_t numPixelsPerImage = imageHeight * imageWidth;
    const auto longOpts             = torch::TensorOptions().device(device).dtype(torch::kLong);
    const auto boolOpts             = torch::TensorOptions().device(device).dtype(torch::kBool);

    // Encode (batchIdx, row, col) into a single int64 key:
    //   key = batchIdx * (H * W) + row * W + col
    // For single-list JaggedTensors, jidx is empty so we skip the batch term entirely.
    const bool singleList = (jidx.size(0) == 0);
    torch::Tensor rows, cols;
    if (jdata.scalar_type() == torch::kInt32) {
        rows = jdata.select(1, 0).to(torch::kLong);
        cols = jdata.select(1, 1).to(torch::kLong);
    } else {
        rows = jdata.select(1, 0);
        cols = jdata.select(1, 1);
    }
    torch::Tensor keys;
    if (singleList) {
        keys = rows * imageWidth + cols;
    } else {
        auto jidxLong = jidx.to(torch::kLong);
        keys          = jidxLong * numPixelsPerImage + rows * imageWidth + cols;
    }

    // Sort keys and find group boundaries
    auto [sortedKeys, sortPerm] = keys.sort();

    auto isGroupStart = torch::ones({totalPixels}, boolOpts);
    if (totalPixels > 1) {
        isGroupStart.slice(0, 1).copy_(sortedKeys.slice(0, 1) != sortedKeys.slice(0, 0, -1));
    }

    // Extract first-of-group positions before mutating isGroupStart
    auto firstInSorted = isGroupStart.nonzero().squeeze(1);

    // Assign a group ID (0-based) to each sorted position via in-place cumsum
    auto groupIds = isGroupStart.to(torch::kLong);
    groupIds.cumsum_(0).sub_(1);
    const auto numUnique = groupIds[-1].item<int64_t>() + 1;

    if (numUnique == totalPixels) {
        return {pixelsToRender, torch::arange(totalPixels, longOpts), false};
    }

    // inverseIndices: map each original position to its group ID (= index in unique output)
    auto inverseIndices = torch::empty({totalPixels}, longOpts);
    inverseIndices.index_put_({sortPerm}, groupIds);

    // Pick the first occurrence of each group (in sorted order) and map to original indices
    auto uniqueOrigIndices = sortPerm.index_select(0, firstInSorted);
    auto uniqueJData       = jdata.index_select(0, uniqueOrigIndices);

    // Build new JaggedTensor offsets for the unique pixels
    auto uniqueBatchIdx = singleList ? torch::zeros({numUnique}, longOpts)
                                     : jidx.to(torch::kLong).index_select(0, uniqueOrigIndices);
    auto numLists       = pixelsToRender.num_outer_lists();
    auto countsPerList  = torch::bincount(uniqueBatchIdx, {}, numLists);
    auto newOffsets     = torch::zeros({numLists + 1}, longOpts);
    newOffsets.slice(0, 1).copy_(countsPerList.cumsum(0));

    auto newJidx = uniqueBatchIdx.to(fvdb::JIdxScalarType);

    auto uniquePixels = fvdb::JaggedTensor::from_jdata_joffsets_jidx_and_lidx_unsafe(
        uniqueJData, newOffsets, newJidx, pixelsToRender.jlidx(), numLists);

    return {uniquePixels, inverseIndices, true};
}

} // namespace fvdb::detail::ops
