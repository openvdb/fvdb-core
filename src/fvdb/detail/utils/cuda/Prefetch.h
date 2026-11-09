// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_CUDA_PREFETCH_H
#define FVDB_DETAIL_UTILS_CUDA_PREFETCH_H

#include <torch/types.h>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <vector>

namespace fvdb {
namespace detail {

struct TilePrefetchRange {
    int64_t offset;
    int64_t count;
    int64_t numTilesH;
    int64_t numTilesW;
    int64_t imageHeight;
    int64_t imageWidth;
    int64_t tileSize;
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
/// [C, imageHeight, imageWidth, ...], append prefetch ranges for the tile range
/// [tileOffset, tileOffset + tileCount). Image tensors include only full-width tile rows entirely
/// owned by the range; partial tile rows are left to demand paging.
void appendPerTilePrefetchRanges(std::vector<void *> &prefetchPointers,
                                 std::vector<size_t> &prefetchSizes,
                                 const torch::TensorList &tensors,
                                 const TilePrefetchRange &range);

void memPrefetchBatchAsync(std::vector<void *> &prefetchPointers,
                           std::vector<size_t> &prefetchSizes,
                           int deviceId,
                           cudaStream_t stream);

/// Prefetch the memory ranges given by paired pointers and byte sizes to the specified device,
/// ordered on the non-default input stream. On CUDA 13+, discard existing contents to avoid
/// migrating data that will be overwritten; callers must overwrite discarded data before reading
/// it. On older CUDA versions, fall back to individual prefetch calls without discarding contents.
void memDiscardAndPrefetchBatchAsync(std::vector<void *> &prefetchPointers,
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
