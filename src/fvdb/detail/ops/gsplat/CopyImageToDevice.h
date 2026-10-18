// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_COPYIMAGETODEVICE_H
#define FVDB_DETAIL_OPS_GSPLAT_COPYIMAGETODEVICE_H

#include <torch/types.h>

namespace fvdb::detail::ops {

// Copy a contiguous CPU NCHW image batch to the destination, preserving shape and dtype.
// Gradients are not supported. DGX sources must be pinned and remain alive and unchanged
// until the asynchronous copies finish on the current per-device CUDA streams.
torch::Tensor copyImageToDeviceCUDA(const torch::Tensor &image, torch::Device device);
torch::Tensor copyImageToDevicePrivateUse1(const torch::Tensor &image, torch::Device device);

} // namespace fvdb::detail::ops

#endif // FVDB_DETAIL_OPS_GSPLAT_COPYIMAGETODEVICE_H
