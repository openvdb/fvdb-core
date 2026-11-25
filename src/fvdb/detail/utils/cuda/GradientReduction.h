// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_CUDA_GRADIENTREDUCTION_H
#define FVDB_DETAIL_UTILS_CUDA_GRADIENTREDUCTION_H

#include <torch/types.h>

#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

namespace fvdb::detail {

// Lists of unique, sorted local CUDA device IDs, which also serve as NCCL ranks;
// the caller determines what each list represents.
using DeviceLists = std::vector<std::vector<c10::DeviceIndex>>;

// Return {firstCamera, cameraCount} for a device's assigned tile interval. Dense intervals
// must lie within the flattened camera tile range. Sparse membership is unknown on the host,
// so every camera is included when the device has active tiles. This helper allocates nothing.
inline std::tuple<int64_t, int64_t>
cameraRangeForTiles(int64_t numCameras,
                    int64_t tilesPerCamera,
                    int64_t tileOffset,
                    int64_t tileCount,
                    bool sparseTiles) {
    if (tileCount == 0) {
        return {0, 0};
    }
    if (sparseTiles) {
        return {0, numCameras};
    }
    const int64_t firstCamera = tileOffset / tilesPerCamera;
    const int64_t lastCamera  = (tileOffset + tileCount - 1) / tilesPerCamera;
    return {firstCamera, lastCamera - firstCamera + 1};
}

// Match rasterization's aligned, flattened tile partition. Sparse camera membership is not
// available on the host: conservatively include every device with assigned active tiles.
DeviceLists makeDeviceLists(int64_t numCameras,
                            int64_t tilesPerCamera,
                            int64_t tileAlignment,
                            int64_t deviceCount,
                            std::optional<int64_t> numActiveTiles = std::nullopt);

// Reduce in place into each device's owned slice of its local gradient buffer.
// Inputs must come from makeLocalGradient<Flatten>(). Flatten uses deviceChunk(numel, rank);
// otherwise each camera uses deviceChunk(N, rank) and requires its sorted contributor list.
// Per-camera reduction uses NCCL reduce-scatter for all-device contributors; all other cameras'
// send/recv transfers share one NCCL group. Noncontributors must have zero gradients for that
// camera. cameraOffsets optionally gives each device's first local camera. Its logical tensor
// contains a contiguous camera range; nonlocal cameras receive into the allocation's compact shard
// tail. Every contributor must contain the corresponding camera in its local range. Uses current
// CUDA streams; keep them unchanged until the local buffers have been released. Calls sharing
// PyTorch's NCCL communicator cache must be serialized on the host.
template <bool Flatten = true>
void reduceLocalGradients(const std::vector<torch::Tensor> &localGradients,
                          const DeviceLists &deviceLists            = {},
                          const std::vector<int64_t> &cameraOffsets = {});

// Publish each device's reduced slices using the same Flatten value as makeLocalGradient().
// Pass the same cameraOffsets as reduceLocalGradients() when using compact camera storage.
// Call after queuing reductions and waiting for output discard/prefetch on the current streams.
// Keep local gradient buffers alive until all output copies have been queued.
template <bool Flatten = true>
void copyLocalGradients(const std::vector<torch::Tensor> &localGradients,
                        torch::Tensor outputGradient,
                        const std::vector<int64_t> &cameraOffsets = {});

} // namespace fvdb::detail

#endif // FVDB_DETAIL_UTILS_CUDA_GRADIENTREDUCTION_H
