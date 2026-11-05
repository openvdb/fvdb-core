// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#include <fvdb/detail/utils/cuda/Prefetch.h>

#include <nanovdb/util/cuda/Util.h>

#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAException.h>

#include <cuda/cmath>

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
                         int64_t tilesPerCamera) {
    const size_t scalarSize        = c10::elementSize(tensor.scalar_type());
    const size_t rowStrideBytes    = tensor.stride(1) * scalarSize;
    const size_t cameraStrideBytes = tensor.stride(0) * scalarSize;
    const int64_t tileRangeEnd     = range.offset + range.count;
    int64_t currentTile            = range.offset;

    while (currentTile < tileRangeEnd) {
        const int64_t cameraId          = currentTile / tilesPerCamera;
        const int64_t cameraTileBegin   = cameraId * tilesPerCamera;
        const int64_t nextCameraTile    = std::min(tileRangeEnd, cameraTileBegin + tilesPerCamera);
        const int64_t firstTileInCamera = currentTile - cameraTileBegin;

        // Round inward to full-width tile rows owned entirely by this device. Partial tile rows
        // remain demand-paged, keeping one contiguous range per camera without logical overlap.
        const int64_t firstTileRow = cuda::ceil_div(firstTileInCamera, range.numTilesW);
        const int64_t tileRowEnd   = (nextCameraTile - cameraTileBegin) / range.numTilesW;

        const int64_t rowStart = firstTileRow * range.tileSize;
        const int64_t rowEnd   = std::min(range.imageHeight, tileRowEnd * range.tileSize);

        if (rowStart < rowEnd) {
            auto *cameraPtr =
                static_cast<uint8_t *>(tensor.data_ptr()) + cameraId * cameraStrideBytes;
            prefetchPointers.emplace_back(cameraPtr + rowStart * rowStrideBytes);
            prefetchSizes.emplace_back((rowEnd - rowStart) * rowStrideBytes);
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
memDiscardAndPrefetchBatchAsync(std::vector<void *> &prefetchPointers,
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

    C10_CUDA_CHECK(cudaMemDiscardAndPrefetchBatchAsync(prefetchPointers.data(),
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
    const int64_t tilesPerCamera = range.numTilesH * range.numTilesW;

    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto &tensor = tensors[i];
        TORCH_CHECK(tensor.is_contiguous(), "Tensor to prefetch is not contiguous");
        TORCH_CHECK(tensor.dim() >= 3, "Tensor to prefetch must have at least 3 dimensions");
        TORCH_CHECK(range.offset >= 0 && range.count >= 0 &&
                        range.offset + range.count <= tensor.size(0) * tilesPerCamera,
                    "Tile range exceeds tensor tile dimensions");

        if (tensor.size(1) == range.numTilesH && tensor.size(2) == range.numTilesW) {
            appendPerTileTensorRanges(prefetchPointers, prefetchSizes, tensor, range);
        } else if (tensor.size(1) == range.imageHeight && tensor.size(2) == range.imageWidth) {
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
