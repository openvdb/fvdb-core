// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_CUDA_PREFETCH_H
#define FVDB_DETAIL_UTILS_CUDA_PREFETCH_H

#include <torch/types.h>

#include <cuda_runtime_api.h>

namespace fvdb {
namespace detail {

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

} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_UTILS_CUDA_PREFETCH_H
