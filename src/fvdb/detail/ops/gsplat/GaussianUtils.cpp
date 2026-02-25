// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#include <fvdb/detail/ops/gsplat/GaussianUtils.h>

#include <nanovdb/util/cuda/Util.h>

#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAException.h>

namespace fvdb::detail::ops {

void
perCameraPrefetchAsync(const torch::Tensor &tensor,
                       uint32_t cameraOffset,
                       uint32_t cameraCount,
                       int deviceId,
                       cudaStream_t stream) {
    TORCH_CHECK(tensor.is_contiguous(), "Tensor to prefetch is not contiguous");
    TORCH_CHECK(cameraOffset + cameraCount <= tensor.size(0),
                "Tensor does not have a batched first dimension");
    size_t scalarSize = c10::elementSize(tensor.scalar_type());
    nanovdb::util::cuda::memPrefetchAsync(static_cast<const uint8_t *>(tensor.const_data_ptr()) +
                                              cameraOffset * tensor.stride(0) * scalarSize,
                                          cameraCount * tensor.stride(0) * scalarSize,
                                          deviceId,
                                          stream);
}

void
perCameraPrefetchBatchAsync(const torch::TensorList &tensors,
                            uint32_t cameraOffset,
                            uint32_t cameraCount,
                            int deviceId,
                            cudaStream_t stream) {
    TORCH_CHECK(stream, "cudaMemPrefetchBatchAsync does not support the default stream");

    std::vector<void *> prefetchPointers;
    std::vector<size_t> prefetchSizes;
    const cudaMemLocation location                 = {cudaMemLocationTypeDevice, deviceId};
    std::vector<cudaMemLocation> prefetchLocations = {location};
    std::vector<size_t> prefetchLocationIndices    = {0};

    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto &tensor = tensors[i];
        TORCH_CHECK(tensor.is_contiguous(), "Tensor to prefetch is not contiguous");
        TORCH_CHECK(cameraOffset + cameraCount <= tensor.size(0),
                    "Tensor does not have a batched first dimension");
        size_t scalarSize = c10::elementSize(tensor.scalar_type());
        prefetchPointers.emplace_back(static_cast<uint8_t *>(tensor.data_ptr()) +
                                      cameraOffset * tensor.stride(0) * scalarSize);
        prefetchSizes.emplace_back(cameraCount * tensor.stride(0) * scalarSize);
    }
    C10_CUDA_CHECK(cudaMemPrefetchBatchAsync(prefetchPointers.data(),
                                             prefetchSizes.data(),
                                             prefetchPointers.size(),
                                             prefetchLocations.data(),
                                             prefetchLocationIndices.data(),
                                             prefetchLocations.size(),
                                             0,
                                             stream));
}

} // namespace fvdb::detail::ops
