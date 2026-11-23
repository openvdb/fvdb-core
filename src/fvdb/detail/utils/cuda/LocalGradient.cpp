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

    const int64_t numElements       = tensor.numel();
    const int64_t deviceCount       = c10::cuda::device_count();
    const int64_t shardSize         = localGradientShardSize(numElements, deviceCount);
    const int64_t paddedNumElements = shardSize * deviceCount;
    TORCH_CHECK(paddedNumElements <= std::numeric_limits<int64_t>::max() / tensor.element_size(),
                "Local gradient allocation size overflows int64_t");
    const size_t numBytes = paddedNumElements * tensor.element_size();

    void *ptr = nullptr;
    if (paddedNumElements != 0) {
        C10_CUDA_CHECK(cudaMallocAsync(&ptr, numBytes, stream));
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
    if (paddedNumElements != 0) {
        C10_CUDA_CHECK(cudaMemsetAsync(ptr, 0, numBytes, stream));
    }
    // Keep the padded allocation alive through a view with the original logical shape. Reduction
    // may expose the zeroed tail with as_strided(), while kernels and output copies only see data.
    return localGradient.narrow(0, 0, numElements).view(tensor.sizes());
}

} // namespace fvdb::detail
