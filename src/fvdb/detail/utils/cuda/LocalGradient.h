// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_CUDA_LOCALGRADIENT_H
#define FVDB_DETAIL_UTILS_CUDA_LOCALGRADIENT_H

#include <c10/core/Device.h>
#include <torch/types.h>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <limits>

namespace fvdb::detail {

// Equal shard size, including padding. The full padded element count fits in int64_t.
inline int64_t
localGradientShardSize(int64_t numElements, int64_t deviceCount) {
    TORCH_CHECK(numElements >= 0, "Local gradient element count must be nonnegative");
    TORCH_CHECK(deviceCount > 0, "Local gradients require at least one CUDA device");
    const int64_t padding = (deviceCount - numElements % deviceCount) % deviceCount;
    TORCH_CHECK(numElements <= std::numeric_limits<int64_t>::max() - padding,
                "Local gradient is too large to pad for reduction");
    return (numElements + padding) / deviceCount;
}

/// @brief Create an owning, zeroed, contiguous CUDA tensor with the shape and dtype of tensor.
/// Storage is rounded up to an equal number of elements per CUDA device, with a zeroed tail for
/// reduceGradientShards(). The returned tensor retains the input's logical shape.
/// @note The caller must select deviceId and supply a stream on that device. Allocation, zeroing,
/// and freeing use the supplied stream; all uses of the tensor must be ordered before its free on
/// that stream. The last tensor owner must be released while the stream is still valid.
/// Freeing the allocation leaves deviceId current; callers must select their device for later work.
/// CUDA API calls are checked for errors.
torch::Tensor
makeLocalGradient(const torch::Tensor &tensor, c10::DeviceIndex deviceId, cudaStream_t stream);

} // namespace fvdb::detail

#endif // FVDB_DETAIL_UTILS_CUDA_LOCALGRADIENT_H
