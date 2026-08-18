// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_CUDA_GRADIENTREDUCTION_H
#define FVDB_DETAIL_UTILS_CUDA_GRADIENTREDUCTION_H

#include <fvdb/detail/utils/cuda/Utils.cuh>

#include <nanovdb/util/cuda/Util.h>

#include <torch/csrc/cuda/nccl.h>
#include <torch/types.h>

#include <vector>

namespace fvdb::detail {

template <typename ScalarType>
void
reduceGradientShards(std::vector<torch::Tensor> &localGradients, torch::Tensor &outputGradient) {
    const int64_t numElements = localGradients.front().numel();
    std::vector<torch::Tensor> reducedShards(c10::cuda::device_count());
    for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
        const auto [shardOffset, shardSize] = deviceChunk(numElements, deviceId);
        if (shardSize == 0) {
            continue;
        }

        reducedShards[deviceId] =
            localGradients[deviceId].view({-1}).narrow(0, shardOffset, shardSize);
    }

    if (numElements % c10::cuda::device_count() == 0) {
        torch::cuda::nccl::reduce_scatter(localGradients, reducedShards);
    } else {
        // NCCL reduce-scatter requires equally sized shards. For an uneven tensor, reduce each
        // ceil-divided shard into its owning device's local receive slice.
        for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
            const auto [shardOffset, shardSize] = deviceChunk(numElements, deviceId);
            if (shardSize == 0) {
                continue;
            }

            std::vector<torch::Tensor> inputShards;
            inputShards.reserve(c10::cuda::device_count());
            for (const auto sourceDeviceId: c10::irange(c10::cuda::device_count())) {
                inputShards.emplace_back(
                    localGradients[sourceDeviceId].view({-1}).narrow(0, shardOffset, shardSize));
            }
            torch::cuda::nccl::reduce(
                inputShards, reducedShards[deviceId], static_cast<int32_t>(deviceId));
        }
    }

    for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
        const auto [shardOffset, shardSize] = deviceChunk(numElements, deviceId);
        if (shardSize == 0) {
            continue;
        }

        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        auto stream          = c10::cuda::getCurrentCUDAStream(deviceId);
        auto *outputShardPtr = outputGradient.data_ptr<ScalarType>() + shardOffset;
        C10_CUDA_CHECK(nanovdb::util::cuda::memPrefetchAsync(
            outputShardPtr, shardSize * sizeof(ScalarType), deviceId, stream));
        C10_CUDA_CHECK(cudaMemcpyAsync(outputShardPtr,
                                       reducedShards[deviceId].data_ptr<ScalarType>(),
                                       shardSize * sizeof(ScalarType),
                                       cudaMemcpyDeviceToDevice,
                                       stream));
    }
}

} // namespace fvdb::detail

#endif // FVDB_DETAIL_UTILS_CUDA_GRADIENTREDUCTION_H
