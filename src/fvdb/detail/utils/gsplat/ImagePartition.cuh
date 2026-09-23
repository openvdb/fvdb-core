// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_GSPLAT_IMAGEPARTITION_CUH
#define FVDB_DETAIL_UTILS_GSPLAT_IMAGEPARTITION_CUH

#include <fvdb/detail/utils/cuda/Utils.cuh>

#include <torch/types.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace fvdb::detail {

// Keep image losses on the same full-width tile rows, including partial image tiles.
inline constexpr int kImageBlockWidth  = 16;
inline constexpr int kImageBlockHeight = 16;

struct ImageBlockChunk {
    size_t tileRowOffset;
    size_t tileRowCount;
    int blockOffset;
    int blockCount;
};

inline ImageBlockChunk
imageBlockChunk(int B, int H, int W, int deviceId) {
    // Blocks are flattened with x varying fastest. Split only at full-width tile-row
    // boundaries so each device's NCHW working set can be expressed as compact row ranges.
    const size_t blocksPerTileRow = (W + kImageBlockWidth - 1) / kImageBlockWidth;
    const size_t tileRowsPerImage = (H + kImageBlockHeight - 1) / kImageBlockHeight;
    const size_t globalTileRows   = B * tileRowsPerImage;

    size_t localTileRowOffset, localTileRowCount;
    std::tie(localTileRowOffset, localTileRowCount) = deviceChunk(globalTileRows, deviceId);

    return {localTileRowOffset,
            localTileRowCount,
            static_cast<int>(localTileRowOffset * blocksPerTileRow),
            static_cast<int>(localTileRowCount * blocksPerTileRow)};
}

inline void
appendImagePrefetchRanges(std::vector<void *> &prefetchPointers,
                          std::vector<size_t> &prefetchSizes,
                          const torch::TensorList &tensors,
                          size_t tileRowOffset,
                          size_t tileRowCount,
                          int B,
                          int CH,
                          int H,
                          int W) {
    if (!tileRowCount) {
        return;
    }

    const size_t tileRowsPerImage = (H + kImageBlockHeight - 1) / kImageBlockHeight;
    const size_t tileRowEnd       = tileRowOffset + tileRowCount;

    TORCH_CHECK(tileRowEnd <= B * tileRowsPerImage, "Invalid image tile-row range");

    for (const auto &tensor: tensors) {
        TORCH_CHECK(tensor.is_contiguous(), "Tensor to prefetch is not contiguous");
        TORCH_CHECK(tensor.dim() == 4 && tensor.size(0) == B && tensor.size(1) == CH &&
                        tensor.size(2) == H && tensor.size(3) == W,
                    "Tensor to prefetch does not match the input image shape");

        const size_t firstTensorRange = prefetchPointers.size();
        const size_t scalarSize       = c10::elementSize(tensor.scalar_type());
        auto *tensorData              = static_cast<uint8_t *>(tensor.data_ptr());

        const size_t firstBatch = tileRowOffset / tileRowsPerImage;
        const size_t lastBatch  = (tileRowEnd - 1) / tileRowsPerImage;
        for (size_t batch = firstBatch; batch <= lastBatch; ++batch) {
            const size_t batchTileRowOffset = batch * tileRowsPerImage;
            const size_t firstTileRow =
                std::max(tileRowOffset, batchTileRowOffset) - batchTileRowOffset;
            const size_t lastTileRow =
                std::min(tileRowEnd, batchTileRowOffset + tileRowsPerImage) - batchTileRowOffset;

            // Prefetch only the rows owned by this device. Halo reads remain demand-driven so
            // adjacent devices never issue prefetches for overlapping logical ranges.
            const size_t firstRow = firstTileRow * kImageBlockHeight;
            const size_t lastRow =
                std::min(lastTileRow * kImageBlockHeight, static_cast<size_t>(H));
            const size_t rowCount = lastRow - firstRow;

            for (int channel = 0; channel < CH; ++channel) {
                const size_t elementOffset = batch * tensor.stride(0) + channel * tensor.stride(1) +
                                             firstRow * tensor.stride(2);
                auto *pointer          = tensorData + elementOffset * scalarSize;
                const size_t byteCount = rowCount * static_cast<size_t>(W) * scalarSize;

                if (prefetchPointers.size() > firstTensorRange) {
                    auto *previousEnd =
                        static_cast<uint8_t *>(prefetchPointers.back()) + prefetchSizes.back();
                    if (previousEnd == pointer) {
                        prefetchSizes.back() += byteCount;
                        continue;
                    }
                }
                prefetchPointers.emplace_back(pointer);
                prefetchSizes.emplace_back(byteCount);
            }
        }
    }
}

} // namespace fvdb::detail

#endif // FVDB_DETAIL_UTILS_GSPLAT_IMAGEPARTITION_CUH
