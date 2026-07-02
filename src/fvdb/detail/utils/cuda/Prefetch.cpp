// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#include <fvdb/detail/utils/cuda/Prefetch.h>

#include <nanovdb/util/cuda/Util.h>

#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace fvdb {
namespace detail {

namespace {

void
appendPerTileTensorRanges(std::vector<void *> &prefetchPointers,
                          std::vector<size_t> &prefetchSizes,
                          const torch::Tensor &tensor,
                          const TilePrefetchRange &range) {
    const size_t scalarSize    = c10::elementSize(tensor.scalar_type());
    const size_t tileSizeBytes = tensor.stride(2) * scalarSize;
    const size_t prefetchSize  = range.count * tileSizeBytes;
    auto *const prefetchPointer =
        static_cast<uint8_t *>(tensor.data_ptr()) + range.offset * tileSizeBytes;
    if (prefetchSize > 0) {
        prefetchPointers.emplace_back(prefetchPointer);
        prefetchSizes.emplace_back(prefetchSize);
    }
}

void
appendPerTileImageRanges(std::vector<void *> &prefetchPointers,
                         std::vector<size_t> &prefetchSizes,
                         const torch::Tensor &tensor,
                         const TilePrefetchRange &range,
                         uint64_t tilesPerCamera) {
    const size_t scalarSize        = c10::elementSize(tensor.scalar_type());
    const size_t pixelStrideBytes  = tensor.stride(2) * scalarSize;
    const size_t rowStrideBytes    = tensor.stride(1) * scalarSize;
    const size_t cameraStrideBytes = tensor.stride(0) * scalarSize;
    const uint64_t tileRangeEnd    = static_cast<uint64_t>(range.offset) + range.count;
    uint64_t currentTile           = range.offset;

    while (currentTile < tileRangeEnd) {
        const uint64_t cameraId        = currentTile / tilesPerCamera;
        const uint64_t cameraTileBegin = cameraId * tilesPerCamera;
        const uint64_t nextCameraTile =
            std::min<uint64_t>(tileRangeEnd, cameraTileBegin + tilesPerCamera);
        const uint32_t firstTileInCamera = static_cast<uint32_t>(currentTile - cameraTileBegin);
        const uint32_t lastTileInCamera =
            static_cast<uint32_t>(nextCameraTile - cameraTileBegin - 1);

        const uint32_t firstTileRow = firstTileInCamera / range.numTilesW;
        const uint32_t firstTileCol = firstTileInCamera % range.numTilesW;
        const uint32_t lastTileRow  = lastTileInCamera / range.numTilesW;
        const uint32_t lastTileCol  = lastTileInCamera % range.numTilesW;

        const uint32_t rowStart = firstTileRow * range.tileSize;
        const uint32_t rowEnd =
            std::min<uint32_t>(range.imageHeight, (lastTileRow + 1) * range.tileSize);
        const uint32_t colStart = firstTileCol * range.tileSize;
        const uint32_t colEnd =
            std::min<uint32_t>(range.imageWidth, (lastTileCol + 1) * range.tileSize);

        if (rowStart < rowEnd) {
            auto *cameraPtr =
                static_cast<uint8_t *>(tensor.data_ptr()) + cameraId * cameraStrideBytes;
            auto *prefetchPointer =
                cameraPtr + rowStart * rowStrideBytes + colStart * pixelStrideBytes;
            auto *prefetchEnd =
                cameraPtr + (rowEnd - 1) * rowStrideBytes + colEnd * pixelStrideBytes;
            if (prefetchPointer < prefetchEnd) {
                const size_t prefetchSize = static_cast<size_t>(prefetchEnd - prefetchPointer);
                prefetchPointers.emplace_back(prefetchPointer);
                prefetchSizes.emplace_back(prefetchSize);
            }
        }

        currentTile = nextCameraTile;
    }
}

} // namespace

void
memPrefetchBatchAsync(std::vector<void *> &prefetchPointers,
                      std::vector<size_t> &prefetchSizes,
                      int deviceId,
                      cudaStream_t stream) {
    TORCH_CHECK(stream, "cudaMemPrefetchBatchAsync does not support the default stream");
    if (prefetchPointers.empty()) {
        return;
    }

#if (CUDART_VERSION < 13000)
    for (size_t i = 0; i < prefetchPointers.size(); ++i) {
        C10_CUDA_CHECK(nanovdb::util::cuda::memPrefetchAsync(
            prefetchPointers[i], prefetchSizes[i], deviceId, stream));
    }
#else
    const cudaMemLocation location                 = {cudaMemLocationTypeDevice, deviceId};
    std::vector<cudaMemLocation> prefetchLocations = {location};
    std::vector<size_t> prefetchLocationIndices    = {0};

    C10_CUDA_CHECK(cudaMemPrefetchBatchAsync(prefetchPointers.data(),
                                             prefetchSizes.data(),
                                             prefetchPointers.size(),
                                             prefetchLocations.data(),
                                             prefetchLocationIndices.data(),
                                             prefetchLocations.size(),
                                             0,
                                             stream));
#endif
}

void
perCameraPrefetchAsync(const torch::Tensor &tensor,
                       uint32_t cameraOffset,
                       uint32_t cameraCount,
                       int deviceId,
                       cudaStream_t stream) {
    TORCH_CHECK(tensor.is_contiguous(), "Tensor to prefetch is not contiguous");
    TORCH_CHECK(cameraOffset + cameraCount <= tensor.size(0),
                "Tensor does not have a batched first dimension");
    size_t scalarSize = c10::elementSize(tensor.scalar_type());
    nanovdb::util::cuda::memPrefetchAsync(static_cast<const uint8_t *>(tensor.const_data_ptr()) +
                                              cameraOffset * tensor.stride(0) * scalarSize,
                                          cameraCount * tensor.stride(0) * scalarSize,
                                          deviceId,
                                          stream);
}

void
perCameraPrefetchBatchAsync(const torch::TensorList &tensors,
                            uint32_t cameraOffset,
                            uint32_t cameraCount,
                            int deviceId,
                            cudaStream_t stream) {
    std::vector<void *> prefetchPointers;
    std::vector<size_t> prefetchSizes;
    appendPerCameraPrefetchRanges(
        prefetchPointers, prefetchSizes, tensors, cameraOffset, cameraCount);
    memPrefetchBatchAsync(prefetchPointers, prefetchSizes, deviceId, stream);
}

void
appendPerCameraPrefetchRanges(std::vector<void *> &prefetchPointers,
                              std::vector<size_t> &prefetchSizes,
                              const torch::TensorList &tensors,
                              uint32_t cameraOffset,
                              uint32_t cameraCount) {
    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto &tensor = tensors[i];
        TORCH_CHECK(tensor.is_contiguous(), "Tensor to prefetch is not contiguous");
        TORCH_CHECK(cameraOffset + cameraCount <= tensor.size(0),
                    "Tensor does not have a batched first dimension");
        const size_t scalarSize   = c10::elementSize(tensor.scalar_type());
        const size_t prefetchSize = cameraCount * tensor.stride(0) * scalarSize;
        if (prefetchSize > 0) {
            prefetchPointers.emplace_back(static_cast<uint8_t *>(tensor.data_ptr()) +
                                          cameraOffset * tensor.stride(0) * scalarSize);
            prefetchSizes.emplace_back(prefetchSize);
        }
    }
}

void
appendPerTilePrefetchRanges(std::vector<void *> &prefetchPointers,
                            std::vector<size_t> &prefetchSizes,
                            const torch::TensorList &tensors,
                            const TilePrefetchRange &range) {
    if (range.count == 0) {
        return;
    }
    TORCH_CHECK(range.numTilesH > 0 && range.numTilesW > 0,
                "Tile dimensions must be greater than 0");
    TORCH_CHECK(range.tileSize > 0, "Tile size must be greater than 0");
    const uint64_t tilesPerCamera =
        static_cast<uint64_t>(range.numTilesH) * static_cast<uint64_t>(range.numTilesW);

    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto &tensor = tensors[i];
        TORCH_CHECK(tensor.is_contiguous(), "Tensor to prefetch is not contiguous");
        TORCH_CHECK(tensor.dim() >= 3, "Tensor to prefetch must have at least 3 dimensions");
        TORCH_CHECK(static_cast<uint64_t>(range.offset) + static_cast<uint64_t>(range.count) <=
                        static_cast<uint64_t>(tensor.size(0)) * tilesPerCamera,
                    "Tile range exceeds tensor tile dimensions");

        if (tensor.size(1) == static_cast<int64_t>(range.numTilesH) &&
            tensor.size(2) == static_cast<int64_t>(range.numTilesW)) {
            appendPerTileTensorRanges(prefetchPointers, prefetchSizes, tensor, range);
        } else if (tensor.size(1) == static_cast<int64_t>(range.imageHeight) &&
                   tensor.size(2) == static_cast<int64_t>(range.imageWidth)) {
            appendPerTileImageRanges(
                prefetchPointers, prefetchSizes, tensor, range, tilesPerCamera);
        } else {
            TORCH_CHECK(false,
                        "Tensor to prefetch must have dimensions [C, numTilesH, numTilesW, ...] "
                        "or [C, imageHeight, imageWidth, ...]");
        }
    }
}

void
perCameraMemsetAsync(const torch::TensorList &tensors,
                     uint32_t cameraOffset,
                     uint32_t cameraCount,
                     int value,
                     cudaStream_t stream) {
    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto &tensor = tensors[i];
        TORCH_CHECK(tensor.is_contiguous(), "Tensor to prefetch is not contiguous");
        TORCH_CHECK(cameraOffset + cameraCount <= tensor.size(0),
                    "Tensor does not have a batched first dimension");
        size_t scalarSize = c10::elementSize(tensor.scalar_type());
        C10_CUDA_CHECK(cudaMemsetAsync(static_cast<uint8_t *>(tensor.data_ptr()) +
                                           cameraOffset * tensor.stride(0) * scalarSize,
                                       value,
                                       cameraCount * tensor.stride(0) * scalarSize,
                                       stream));
    }
}

} // namespace detail
} // namespace fvdb
