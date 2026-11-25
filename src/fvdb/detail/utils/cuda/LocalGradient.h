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
#include <optional>

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

/// @brief Create an owning, zeroed CUDA gradient with tensor's dtype and optional camera
/// compaction. Flatten pads contiguous, flattened storage to equally sized element shards.
/// Otherwise, pad the Gaussian dimension of each camera in [C, N, ...] to equally sized Gaussian
/// shards; camera strides include this padding and must be honored by kernels. numLocalCameras
/// optionally limits the camera dimension; storage after the logical tensor holds this device's
/// Gaussian shards for the remaining cameras. Those shards use the logical (unpadded) device shard
/// size. The tail is ordered by global camera ID, omitting the contiguous local camera range. With
/// no local cameras the logical tensor is empty, but its storage can still hold receive shards.
/// Use the same Flatten value and camera offsets for reduction and output copies.
/// @note The caller must select deviceId and supply a stream on that device. Allocation, zeroing,
/// and freeing use the supplied stream; all uses of the tensor must be ordered before its free on
/// that stream. The last tensor owner must be released while the stream is still valid.
/// Freeing the allocation leaves deviceId current; callers must select their device for later work.
/// CUDA API calls are checked for errors.
template <bool Flatten = true>
torch::Tensor makeLocalGradient(const torch::Tensor &tensor,
                                c10::DeviceIndex deviceId,
                                cudaStream_t stream,
                                std::optional<int64_t> numLocalCameras = std::nullopt);

} // namespace fvdb::detail

#endif // FVDB_DETAIL_UTILS_CUDA_LOCALGRADIENT_H
