// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/utils/cuda/LocalGradient.h>

#include <ATen/ops/from_blob.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>

#include <algorithm>
#include <limits>

namespace fvdb::detail {
namespace {

int64_t
checkedNumElements(at::IntArrayRef sizes) {
    int64_t numElements = 1;
    for (const auto size: sizes) {
        TORCH_CHECK(size == 0 || numElements <= std::numeric_limits<int64_t>::max() / size,
                    "Local gradient allocation size overflows int64_t");
        numElements *= size;
    }
    return numElements;
}

torch::Tensor
makeGradientStorage(const torch::Tensor &tensor,
                    int64_t paddedNumElements,
                    c10::DeviceIndex deviceId,
                    cudaStream_t stream) {
    const c10::Device device(c10::kCUDA, deviceId);
    const auto options = tensor.options().device(device);

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
    return localGradient;
}

} // namespace

template <bool Flatten>
torch::Tensor
makeLocalGradient(const torch::Tensor &tensor,
                  c10::DeviceIndex deviceId,
                  cudaStream_t stream,
                  std::optional<int64_t> numLocalCameras) {
    const int64_t deviceCount = c10::cuda::device_count();
    if constexpr (Flatten) {
        TORCH_CHECK(!numLocalCameras, "Flattened gradients do not have a local camera dimension");
        const int64_t numElements       = tensor.numel();
        const int64_t shardSize         = localGradientShardSize(numElements, deviceCount);
        const int64_t paddedNumElements = shardSize * deviceCount;
        // Keep the padded allocation alive through a view with the original logical shape.
        return makeGradientStorage(tensor, paddedNumElements, deviceId, stream)
            .narrow(0, 0, numElements)
            .view(tensor.sizes());
    } else {
        TORCH_CHECK(tensor.dim() >= 2, "Camera gradients must have shape [C, N, ...]");
        TORCH_CHECK(deviceId >= 0 && deviceId < deviceCount, "Invalid local gradient device");
        const int64_t cameraCount = numLocalCameras.value_or(tensor.size(0));
        TORCH_CHECK(cameraCount >= 0 && cameraCount <= tensor.size(0),
                    "Local camera count must be within the full gradient's camera dimension");
        const int64_t shardSize         = localGradientShardSize(tensor.size(1), deviceCount);
        auto paddedSizes                = tensor.sizes().vec();
        paddedSizes[0]                  = cameraCount;
        paddedSizes[1]                  = shardSize * deviceCount;
        const int64_t paddedNumElements = checkedNumElements(paddedSizes);

        auto receiveSizes = tensor.sizes().vec();
        receiveSizes[0]   = tensor.size(0) - cameraCount;
        receiveSizes[1] =
            std::min(shardSize, std::max(int64_t{0}, tensor.size(1) - deviceId * shardSize));
        const int64_t receiveNumElements = checkedNumElements(receiveSizes);
        TORCH_CHECK(paddedNumElements <= std::numeric_limits<int64_t>::max() - receiveNumElements,
                    "Local gradient allocation size overflows int64_t");
        // One owning allocation keeps both accumulation and receive storage alive on the same
        // stream. When all cameras are local, the receive tail is empty.
        return makeGradientStorage(tensor, paddedNumElements + receiveNumElements, deviceId, stream)
            .narrow(0, 0, paddedNumElements)
            .view(paddedSizes)
            .narrow(1, 0, tensor.size(1));
    }
}

template torch::Tensor makeLocalGradient<true>(const torch::Tensor &,
                                               c10::DeviceIndex,
                                               cudaStream_t,
                                               std::optional<int64_t>);
template torch::Tensor makeLocalGradient<false>(const torch::Tensor &,
                                                c10::DeviceIndex,
                                                cudaStream_t,
                                                std::optional<int64_t>);

} // namespace fvdb::detail
