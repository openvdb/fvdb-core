// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_CUDA_GRADIENTREDUCTION_H
#define FVDB_DETAIL_UTILS_CUDA_GRADIENTREDUCTION_H

#include <fvdb/detail/utils/cuda/Utils.cuh>

#include <torch/csrc/cuda/nccl.h>
#include <torch/types.h>

#include <vector>

namespace fvdb::detail {

// Reduce in place into each device's owned slice of its local gradient buffer.
// Inputs must come from makeLocalGradient(), whose storage includes zeroed padding for equally
// sized NCCL shards. The logical tensor shapes and deviceChunk() output ownership stay unchanged.
inline void
reduceGradientShards(const std::vector<torch::Tensor> &localGradients) {
    const int64_t numElements = localGradients.front().numel();
    if (numElements == 0) {
        return;
    }

    const int64_t deviceCount       = c10::cuda::device_count();
    const int64_t shardSize         = numElements / deviceCount + (numElements % deviceCount != 0);
    const int64_t paddedNumElements = shardSize * deviceCount;
    std::vector<torch::Tensor> paddedGradients(deviceCount);
    std::vector<torch::Tensor> reducedShards(deviceCount);
    for (const auto deviceId: c10::irange(deviceCount)) {
        // Expose the allocation's zeroed tail without copying or changing the logical gradient.
        paddedGradients[deviceId] = localGradients[deviceId].as_strided({paddedNumElements}, {1});
        reducedShards[deviceId] =
            paddedGradients[deviceId].narrow(0, deviceId * shardSize, shardSize);
    }

    // NCCL supports in-place reduce-scatter when each receive buffer is its rank's input slice.
    // Ranks with no logical elements still participate using their zero-filled padded slice.
    torch::cuda::nccl::reduce_scatter(paddedGradients, reducedShards);
}

// Call after queuing the reductions and waiting for output prefetching on the current streams.
// Keep the local gradient buffers alive until all output copies have been queued.
template <typename ScalarType>
void
copyGradientShards(const std::vector<torch::Tensor> &localGradients,
                   torch::Tensor &outputGradient) {
    const int64_t numElements = localGradients.front().numel();
    for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
        const auto [shardOffset, shardSize] = deviceChunk(numElements, deviceId);
        if (shardSize == 0) {
            continue;
        }

        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        auto stream = c10::cuda::getCurrentCUDAStream(deviceId);
        C10_CUDA_CHECK(
            cudaMemcpyAsync(outputGradient.data_ptr<ScalarType>() + shardOffset,
                            localGradients[deviceId].data_ptr<ScalarType>() + shardOffset,
                            shardSize * sizeof(ScalarType),
                            cudaMemcpyDeviceToDevice,
                            stream));
    }
}

} // namespace fvdb::detail

#endif // FVDB_DETAIL_UTILS_CUDA_GRADIENTREDUCTION_H
