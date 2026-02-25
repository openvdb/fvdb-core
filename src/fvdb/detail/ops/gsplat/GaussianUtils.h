// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_GAUSSIANUTILS_H
#define FVDB_DETAIL_OPS_GSPLAT_GAUSSIANUTILS_H

#include <torch/types.h>

#include <cuda_runtime_api.h>

namespace fvdb {
namespace detail {
namespace ops {

void perCameraPrefetchAsync(const torch::Tensor &tensor,
                            uint32_t cameraOffset,
                            uint32_t cameraCount,
                            int deviceId,
                            cudaStream_t stream);

void perCameraPrefetchBatchAsync(const torch::TensorList &tensors,
                                 uint32_t cameraOffset,
                                 uint32_t cameraCount,
                                 int deviceId,
                                 cudaStream_t stream);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_GSPLAT_GAUSSIANUTILS_H
