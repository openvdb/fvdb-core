// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/utils/cuda/LocalGradient.h>

#include <ATen/ops/from_blob.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>

#include <limits>

namespace fvdb::detail {

torch::Tensor
makeLocalGradient(const torch::Tensor &tensor, c10::DeviceIndex deviceId, cudaStream_t stream) {
    const c10::Device device(c10::kCUDA, deviceId);
    const auto options = tensor.options().device(device);

    const int64_t numElements = tensor.numel();
    const int64_t deviceCount = c10::cuda::device_count();
    TORCH_CHECK(deviceCount > 0, "Local gradients require at least one CUDA device");
    const int64_t padding = (deviceCount - numElements % deviceCount) % deviceCount;
    TORCH_CHECK(numElements <= std::numeric_limits<int64_t>::max() - padding,
                "Local gradient is too large to pad for reduction");
    const int64_t shardSize         = numElements / deviceCount + (numElements % deviceCount != 0);
    const int64_t paddedNumElements = shardSize * deviceCount;

    void *ptr = nullptr;
    if (tensor.numel() != 0) {
        C10_CUDA_CHECK(cudaMallocAsync(&ptr, paddedNumElements * tensor.element_size(), stream));
    }
    auto localGradient = at::from_blob(
        ptr,
        {paddedNumElements},
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
        C10_CUDA_CHECK(cudaMemsetAsync(ptr, 0, paddedNumElements * tensor.element_size(), stream));
    }
    // Keep the padded allocation alive through a view with the original logical shape. Reduction
    // may expose the zeroed tail with as_strided(), while kernels and output copies only see data.
    return localGradient.narrow(0, 0, numElements).view(tensor.sizes());
}

} // namespace fvdb::detail
