// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_CUDA_PREFETCH_H
#define FVDB_DETAIL_UTILS_CUDA_PREFETCH_H

#include <torch/types.h>

#include <cuda_runtime_api.h>

#include <vector>

namespace fvdb {
namespace detail {

struct TilePrefetchRange {
    uint32_t offset;
    uint32_t count;
    uint32_t numTilesH;
    uint32_t numTilesW;
    uint32_t imageHeight;
    uint32_t imageWidth;
    uint32_t tileSize;
};

/// Given a contiguous tensor with dimensions [C, ...] where C is the number of cameras, we prefetch
/// the slices [cameraOffset : cameraCount, ...] to the specified device ordered on the input
/// stream.
void perCameraPrefetchAsync(const torch::Tensor &tensor,
                            uint32_t cameraOffset,
                            uint32_t cameraCount,
                            int deviceId,
                            cudaStream_t stream);

/// Given a list of contiguous tensors each with dimensions [C, ...] where C is the number of
/// cameras, we prefetch the slices [cameraOffset : cameraCount, ...] to the specified device
/// ordered on the input stream in a single asynchronous batched prefetch call.
void perCameraPrefetchBatchAsync(const torch::TensorList &tensors,
                                 uint32_t cameraOffset,
                                 uint32_t cameraCount,
                                 int deviceId,
                                 cudaStream_t stream);

void appendPerCameraPrefetchRanges(std::vector<void *> &prefetchPointers,
                                   std::vector<size_t> &prefetchSizes,
                                   const torch::TensorList &tensors,
                                   uint32_t cameraOffset,
                                   uint32_t cameraCount);

/// Given a list of contiguous tensors laid out as either [C, numTilesH, numTilesW, ...] or
/// [C, imageHeight, imageWidth, ...], append the memory touched by the tile range
/// [tileOffset, tileOffset + tileCount) to the prefetch batch.
void appendPerTilePrefetchRanges(std::vector<void *> &prefetchPointers,
                                 std::vector<size_t> &prefetchSizes,
                                 const torch::TensorList &tensors,
                                 const TilePrefetchRange &range);

void memPrefetchBatchAsync(std::vector<void *> &prefetchPointers,
                           std::vector<size_t> &prefetchSizes,
                           int deviceId,
                           cudaStream_t stream);

/// Given a list of contiguous tensors each with dimensions [C, ...] where C is the number of
/// cameras, we memset the slices [cameraOffset : cameraCount, ...] to the specified value
/// on the input stream.
void perCameraMemsetAsync(const torch::TensorList &tensors,
                          uint32_t cameraOffset,
                          uint32_t cameraCount,
                          int value,
                          cudaStream_t stream);

} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_UTILS_CUDA_PREFETCH_H
