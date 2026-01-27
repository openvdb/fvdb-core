// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef DISPATCH_DISPATCH_FOR_EACH_TORCH_H
#define DISPATCH_DISPATCH_FOR_EACH_TORCH_H

#include "dispatch/dispatch_table.h"
#include "dispatch/tag_match.h"
#include "dispatch/torch.h"
#include "dispatch/torch_types.h"
#include "dispatch/types.h"

#include <ATen/Parallel.h>
#include <c10/cuda/CUDAStream.h>

#include <concepts>
#include <cstdint>
#include <functional>

namespace dispatch {

//------------------------------------------------------------------------------
// for_each configuration
//------------------------------------------------------------------------------
// Template parameters for performance tuning:
//   - GrainSize: Elements processed per thread in sequence (improves ILP)
//   - BlockDim: Threads per block (affects occupancy and register pressure)
//
// Defaults are reasonable for memory-bound element-wise operations.
// Compute-bound operations may benefit from different values.

struct for_each_config {
    static constexpr int64_t default_grain_size = 4;
    static constexpr int default_block_dim      = 256;
    static constexpr int default_max_grid_dim   = 65535;
};

#if defined(__CUDACC__)

//------------------------------------------------------------------------------
// CUDA for_each implementation
//------------------------------------------------------------------------------
// Uses grid-stride loop pattern with configurable grain size for ILP.
// Each thread processes GrainSize elements in sequence before striding.
//
// Memory access pattern (GrainSize=4, BlockDim=256):
//   Thread 0: elements 0,1,2,3, then 4096,4097,4098,4099, ...
//   Thread 1: elements 4,5,6,7, then 4100,4101,4102,4103, ...
//   ...
// This provides:
//   - Coalesced memory access (consecutive threads access consecutive memory)
//   - Improved ILP (compiler can overlap loads/stores within grain)
//   - Good load balancing via grid-stride

template <int64_t GrainSize, typename Tag, typename Func>
    requires tag_match<Tag, torch::kCUDA>
__global__ void
for_each_cuda_kernel(Tag t, int64_t count, Func func) {
    // Each thread handles GrainSize consecutive elements, then strides by grid size
    int64_t const thread_id    = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t const grid_stride  = static_cast<int64_t>(gridDim.x) * blockDim.x;
    int64_t const grain_stride = grid_stride * GrainSize;

    // Process elements in grain-sized chunks
    for (int64_t base = thread_id * GrainSize; base < count; base += grain_stride) {
// Unrolled inner loop for ILP
#pragma unroll
        for (int64_t g = 0; g < GrainSize; ++g) {
            int64_t const idx = base + g;
            if (idx < count) {
                func(t, idx);
            }
        }
    }
}

// Primary for_each interface with configurable performance parameters
template <int64_t GrainSize = for_each_config::default_grain_size,
          int BlockDim      = for_each_config::default_block_dim,
          typename Tag,
          typename Func>
    requires tag_match<Tag, torch::kCUDA>
void
for_each(Tag t, int64_t count, Func &&func) {
    if (count == 0)
        return;

    // Calculate grid size: enough threads to cover count, but capped
    int64_t const threads_needed = (count + GrainSize - 1) / GrainSize;
    int64_t const blocks_needed  = (threads_needed + BlockDim - 1) / BlockDim;
    int const grid_dim           = static_cast<int>(
        std::min(blocks_needed, static_cast<int64_t>(for_each_config::default_max_grid_dim)));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    for_each_cuda_kernel<GrainSize>
        <<<grid_dim, BlockDim, 0, stream>>>(t, count, std::forward<Func>(func));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

//------------------------------------------------------------------------------
// PrivateUse1 (Universal Memory) for_each implementation
//------------------------------------------------------------------------------
// Distributes work across all available CUDA devices.
// Uses same grain-stride pattern as single-GPU version.

template <int64_t GrainSize, typename Tag, typename Func>
    requires tag_match<Tag, torch::kPrivateUse1>
__global__ void
for_each_pvt1_kernel(Tag t, int64_t chunk_count, int64_t chunk_offset, Func func) {
    int64_t const thread_id    = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t const grid_stride  = static_cast<int64_t>(gridDim.x) * blockDim.x;
    int64_t const grain_stride = grid_stride * GrainSize;

    for (int64_t base = thread_id * GrainSize; base < chunk_count; base += grain_stride) {
#pragma unroll
        for (int64_t g = 0; g < GrainSize; ++g) {
            int64_t const local_idx = base + g;
            if (local_idx < chunk_count) {
                func(t, local_idx + chunk_offset);
            }
        }
    }
}

template <int64_t GrainSize = for_each_config::default_grain_size,
          int BlockDim      = for_each_config::default_block_dim,
          typename Tag,
          typename Func>
    requires tag_match<Tag, torch::kPrivateUse1>
void
for_each(Tag t, int64_t count, Func &&func) {
    if (count == 0)
        return;

    auto const device_count = c10::cuda::device_count();

    for (c10::DeviceIndex device_id = 0; device_id < device_count; ++device_id) {
        C10_CUDA_CHECK(cudaSetDevice(device_id));
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_id).stream();

        // Divide work evenly across devices
        auto const base_chunk  = count / device_count;
        auto const remainder   = count % device_count;
        auto const chunk_count = base_chunk + (device_id < remainder ? 1 : 0);

        // Calculate offset
        int64_t chunk_offset = 0;
        if (device_id < remainder) {
            chunk_offset = device_id * (base_chunk + 1);
        } else {
            chunk_offset = remainder * (base_chunk + 1) + (device_id - remainder) * base_chunk;
        }

        if (chunk_count > 0) {
            int64_t const threads_needed = (chunk_count + GrainSize - 1) / GrainSize;
            int64_t const blocks_needed  = (threads_needed + BlockDim - 1) / BlockDim;
            int const grid_dim           = static_cast<int>(std::min(
                blocks_needed, static_cast<int64_t>(for_each_config::default_max_grid_dim)));

            for_each_pvt1_kernel<GrainSize>
                <<<grid_dim, BlockDim, 0, stream>>>(t, chunk_count, chunk_offset, func);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }

    // Synchronize all streams
    for (c10::DeviceIndex device_id = 0; device_id < device_count; ++device_id) {
        c10::cuda::getCurrentCUDAStream(device_id).synchronize();
    }
}

#endif // __CUDACC__

//------------------------------------------------------------------------------
// CPU for_each implementation
//------------------------------------------------------------------------------
// Uses ATen's parallel_for which respects PyTorch's thread pool settings.
// GrainSize is used as the minimum work unit per thread.

template <int64_t GrainSize = for_each_config::default_grain_size,
          int BlockDim = for_each_config::default_block_dim, // unused on CPU, for API consistency
          typename Tag,
          typename Func>
    requires tag_match<Tag, torch::kCPU>
void
for_each(Tag t, int64_t count, Func &&func) {
    if (count == 0)
        return;

    // ATen's parallel_for handles work distribution
    // grain_size hints at minimum work per thread
    at::parallel_for(0, count, /*grain_size=*/GrainSize, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            std::invoke(func, t, i);
        }
    });
}

} // namespace dispatch

#endif // DISPATCH_DISPATCH_FOR_EACH_TORCH_H
