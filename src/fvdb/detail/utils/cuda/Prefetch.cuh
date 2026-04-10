// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_CUDA_PREFETCH_CUH
#define FVDB_DETAIL_UTILS_CUDA_PREFETCH_CUH

#include <nanovdb/util/cuda/Util.h>

#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAException.h>
#include <torch/types.h>

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace fvdb {
namespace detail {

/// Prefetch a contiguous batch of slices from tensor dimension 0 to a GPU.
///
/// Given a contiguous tensor with dimensions [C, ...] where C is the number of
/// cameras/batches, prefetch slices [offset, offset+count) to the specified device.
inline void
perCameraPrefetchBatchAsync(const torch::TensorList &tensors,
                            uint32_t cameraOffset,
                            uint32_t cameraCount,
                            int deviceId,
                            cudaStream_t stream) {
    TORCH_CHECK(stream, "cudaMemPrefetchBatchAsync does not support the default stream");
#if (CUDART_VERSION < 13000)
    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto &tensor = tensors[i];
        TORCH_CHECK(tensor.is_contiguous(), "Tensor to prefetch is not contiguous");
        TORCH_CHECK(cameraOffset + cameraCount <= tensor.size(0),
                    "Tensor does not have a batched first dimension");
        size_t scalarSize = c10::elementSize(tensor.scalar_type());
        C10_CUDA_CHECK(
            nanovdb::util::cuda::memPrefetchAsync(static_cast<uint8_t *>(tensor.data_ptr()) +
                                                      cameraOffset * tensor.stride(0) * scalarSize,
                                                  cameraCount * tensor.stride(0) * scalarSize,
                                                  deviceId,
                                                  stream));
    }
#else
    std::vector<void *> prefetchPointers;
    std::vector<size_t> prefetchSizes;
    const cudaMemLocation location                 = {cudaMemLocationTypeDevice, deviceId};
    std::vector<cudaMemLocation> prefetchLocations = {location};

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
    std::vector<size_t> prefetchLocationIndices(prefetchPointers.size(), 0);
    C10_CUDA_CHECK(cudaMemPrefetchBatchAsync(prefetchPointers.data(),
                                             prefetchSizes.data(),
                                             prefetchPointers.size(),
                                             prefetchLocations.data(),
                                             prefetchLocationIndices.data(),
                                             prefetchLocations.size(),
                                             0,
                                             stream));
#endif
}

/// Prefetch float tensors by raw element offset and count.
///
/// Used by FusedSSIM where the tensors are flat float buffers rather than
/// camera-batched tensors.
inline void
imagePrefetchBatchAsync(const torch::TensorList &tensors,
                        int localElementOffset,
                        int localElementCount,
                        int deviceId,
                        cudaStream_t stream) {
    TORCH_CHECK(stream, "cudaMemPrefetchBatchAsync does not support the default stream");
    for (size_t i = 0; i < tensors.size(); ++i) {
        TORCH_CHECK(tensors[i].scalar_type() == torch::kFloat32,
                    "imagePrefetchBatchAsync expects float32 tensors, got ",
                    tensors[i].scalar_type(),
                    " for tensor ",
                    i);
    }
#if (CUDART_VERSION < 13000)
    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto &tensor = tensors[i];
        TORCH_CHECK(tensor.is_contiguous(), "Tensor to prefetch is not contiguous");
        C10_CUDA_CHECK(
            nanovdb::util::cuda::memPrefetchAsync(tensor.data_ptr<float>() + localElementOffset,
                                                  localElementCount * sizeof(float),
                                                  deviceId,
                                                  stream));
    }
#else
    std::vector<void *> prefetchPointers;
    std::vector<size_t> prefetchSizes;
    cudaMemLocation location                       = {cudaMemLocationTypeDevice, deviceId};
    std::vector<cudaMemLocation> prefetchLocations = {location};

    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto &tensor = tensors[i];
        TORCH_CHECK(tensor.is_contiguous(), "Tensor to prefetch is not contiguous");
        prefetchPointers.emplace_back(tensor.data_ptr<float>() + localElementOffset);
        prefetchSizes.emplace_back(localElementCount * sizeof(float));
    }
    std::vector<size_t> prefetchLocationIndices(prefetchPointers.size(), 0);
    C10_CUDA_CHECK(cudaMemPrefetchBatchAsync(prefetchPointers.data(),
                                             prefetchSizes.data(),
                                             prefetchPointers.size(),
                                             prefetchLocations.data(),
                                             prefetchLocationIndices.data(),
                                             prefetchLocations.size(),
                                             0,
                                             stream));
#endif
}

} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_UTILS_CUDA_PREFETCH_CUH
