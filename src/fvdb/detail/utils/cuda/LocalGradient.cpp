// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/utils/cuda/LocalGradient.h>

#include <ATen/ops/from_blob.h>
#include <c10/cuda/CUDAException.h>

namespace fvdb::detail {

torch::Tensor
makeLocalGradient(const torch::Tensor &tensor, c10::DeviceIndex deviceId, cudaStream_t stream) {
    const c10::Device device(c10::kCUDA, deviceId);
    const auto options = tensor.options().device(device);

    void *ptr = nullptr;
    if (tensor.numel() != 0) {
        C10_CUDA_CHECK(cudaMallocAsync(&ptr, tensor.numel() * tensor.element_size(), stream));
    }
    auto localGradient = at::from_blob(
        ptr,
        tensor.sizes(),
        [deviceId, stream](void *data) noexcept {
            if (data == nullptr) {
                return;
            }

            // Select the correct device context before freeing the allocation.
            C10_CUDA_CHECK(cudaSetDevice(deviceId));
            C10_CUDA_CHECK(cudaFreeAsync(data, stream));
        },
        options,
        device);
    if (tensor.numel() != 0) {
        C10_CUDA_CHECK(cudaMemsetAsync(ptr, 0, tensor.numel() * tensor.element_size(), stream));
    }
    return localGradient;
}

} // namespace fvdb::detail
