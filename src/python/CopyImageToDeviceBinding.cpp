// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/gsplat/CopyImageToDevice.h>

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

namespace {

torch::Tensor
copyImageToDevice(const torch::Tensor &image, torch::Device device) {
    TORCH_CHECK_VALUE(device.is_cuda() || device.type() == torch::kPrivateUse1,
                      "Image copy requires a CUDA or DGX destination");
    // Dispatch on the destination rather than the CPU source tensor.
    const auto key = device.is_cuda() ? c10::DispatchKey::CUDA : c10::DispatchKey::PrivateUse1;
    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("fvdb::_copy_image_to_device", "")
                         .typed<torch::Tensor(const torch::Tensor &, torch::Device)>();
    return op.redispatch(c10::DispatchKeySet(key), image, device);
}

} // namespace

TORCH_LIBRARY_IMPL(fvdb, BackendSelect, m) {
    m.impl("_copy_image_to_device", &copyImageToDevice);
}

TORCH_LIBRARY_IMPL(fvdb, CUDA, m) {
    m.impl("_copy_image_to_device", &fvdb::detail::ops::copyImageToDeviceCUDA);
}

TORCH_LIBRARY_IMPL(fvdb, PrivateUse1, m) {
    m.impl("_copy_image_to_device", &fvdb::detail::ops::copyImageToDevicePrivateUse1);
}
