// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/utils/cuda/LocalGradient.h>

#include <ATen/ops/from_blob.h>

namespace fvdb::detail {

torch::Tensor
makeLocalGradient(const torch::Tensor &tensor, c10::DeviceIndex deviceId, cudaStream_t stream) {
    const c10::Device device(c10::kCUDA, deviceId);
    const auto options = tensor.options().device(device);

    void *ptr = nullptr;
    if (tensor.numel() != 0) {
        cudaMallocAsync(&ptr, tensor.numel() * tensor.element_size(), stream);
        cudaMemsetAsync(ptr, 0, tensor.numel() * tensor.element_size(), stream);
    }
    return at::from_blob(
        ptr,
        tensor.sizes(),
        [deviceId, stream](void *data) noexcept {
            if (data == nullptr) {
                return;
            }

            // Select the correct device context before freeing the allocation.
            cudaSetDevice(deviceId);
            cudaFreeAsync(data, stream);
        },
        options,
        device);
}

} // namespace fvdb::detail
