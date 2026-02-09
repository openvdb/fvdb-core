// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// for_each: Parallel index generation for CPU, CUDA, and PrivateUse1.
//
// for_each is a parallel index generator. It iterates [0, count) on the device
// specified by the tag. What those indices mean is the functor's business.
//
// Functor signature: void(Tag, int64_t idx)
//
// The tag is passed through to the functor, enabling concept-constrained
// overload sets that specialize per-device or per-scalar-type.
//
// Device selection:
//   CUDA:        grid-stride kernel with configurable ILP grain
//   CPU:         default_thread_pool (work-stealing)
//   PrivateUse1: multi-GPU distribution
//
#ifndef DISPATCH_DISPATCH_TORCH_FOR_EACH_H
#define DISPATCH_DISPATCH_TORCH_FOR_EACH_H

#include "dispatch/macros.h"
#include "dispatch/thread_pool.h"
#include "dispatch/with_value.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <type_traits>
#include <utility>

#if defined(__CUDACC__)
#include <c10/cuda/CUDAStream.h>
#endif

namespace dispatch {

// =============================================================================
// Configuration
// =============================================================================

struct for_each_cuda_config {
    static consteval int64_t
    default_grain_size() {
        return 4;
    }
    static consteval int
    default_block_dim() {
        return 256;
    }
    static consteval int
    default_max_grid_dim() {
        return 65535;
    }
};

struct for_each_cpu_config {
    static consteval int64_t
    default_grain_size() {
        return 0; // Let the thread pool decide
    }
};

// =============================================================================
// CPU for_each
// =============================================================================

template <typename Tag, typename Func>
    requires with_value<Tag, torch::kCPU>
void
for_each(Tag tag, int64_t count, Func &&func) {
    if (count == 0)
        return;

    default_thread_pool::instance().parallel_for(
        int64_t{0}, count, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i)
                std::invoke(func, tag, i);
        });
}

// =============================================================================
// CUDA for_each
// =============================================================================

#if defined(__CUDACC__)

template <int64_t GrainSize, typename Tag, typename Func>
    requires with_value<Tag, torch::kCUDA>
__global__ void
for_each_cuda_kernel(Tag tag, int64_t count, Func func) {
    int64_t const thread_id    = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t const grid_stride  = static_cast<int64_t>(gridDim.x) * blockDim.x;
    int64_t const grain_stride = grid_stride * GrainSize;

    for (int64_t base = thread_id * GrainSize; base < count; base += grain_stride) {
        DISPATCH_UNROLL
        for (int64_t g = 0; g < GrainSize; ++g) {
            int64_t const idx = base + g;
            if (idx < count) {
                func(tag, idx);
            }
        }
    }
}

template <int64_t GrainSize = for_each_cuda_config::default_grain_size(),
          int BlockDim      = for_each_cuda_config::default_block_dim(),
          typename Tag,
          typename Func>
    requires with_value<Tag, torch::kCUDA>
void
for_each(Tag tag, int64_t count, Func &&func) {
    if (count == 0)
        return;

    int64_t const threads_needed = (count + GrainSize - 1) / GrainSize;
    int64_t const blocks_needed  = (threads_needed + BlockDim - 1) / BlockDim;
    int const grid_dim           = static_cast<int>(std::min(
        blocks_needed, static_cast<int64_t>(for_each_cuda_config::default_max_grid_dim())));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    using F = std::decay_t<Func>;
    for_each_cuda_kernel<GrainSize>
        <<<grid_dim, BlockDim, 0, stream>>>(tag, count, F(std::forward<Func>(func)));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// =============================================================================
// PrivateUse1 for_each (multi-GPU)
// =============================================================================

template <int64_t GrainSize, typename Tag, typename Func>
    requires with_value<Tag, torch::kPrivateUse1>
__global__ void
for_each_pvt1_kernel(Tag tag, int64_t chunk_count, int64_t chunk_offset, Func func) {
    int64_t const thread_id    = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t const grid_stride  = static_cast<int64_t>(gridDim.x) * blockDim.x;
    int64_t const grain_stride = grid_stride * GrainSize;

    for (int64_t base = thread_id * GrainSize; base < chunk_count; base += grain_stride) {
        DISPATCH_UNROLL
        for (int64_t g = 0; g < GrainSize; ++g) {
            int64_t const local_idx = base + g;
            if (local_idx < chunk_count) {
                func(tag, local_idx + chunk_offset);
            }
        }
    }
}

template <int64_t GrainSize = for_each_cuda_config::default_grain_size(),
          int BlockDim      = for_each_cuda_config::default_block_dim(),
          typename Tag,
          typename Func>
    requires with_value<Tag, torch::kPrivateUse1>
void
for_each(Tag tag, int64_t count, Func &&func) {
    if (count == 0)
        return;

    using F = std::decay_t<Func>;
    F f(std::forward<Func>(func));

    auto const device_count = c10::cuda::device_count();

    for (c10::DeviceIndex device_id = 0; device_id < device_count; ++device_id) {
        C10_CUDA_CHECK(cudaSetDevice(device_id));
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_id).stream();

        auto const base_chunk  = count / device_count;
        auto const remainder   = count % device_count;
        auto const chunk_count = base_chunk + (device_id < remainder ? 1 : 0);

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
                blocks_needed, static_cast<int64_t>(for_each_cuda_config::default_max_grid_dim())));

            for_each_pvt1_kernel<GrainSize>
                <<<grid_dim, BlockDim, 0, stream>>>(tag, chunk_count, chunk_offset, f);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }

    for (c10::DeviceIndex device_id = 0; device_id < device_count; ++device_id) {
        c10::cuda::getCurrentCUDAStream(device_id).synchronize();
    }
}

#endif // __CUDACC__

} // namespace dispatch

#endif // DISPATCH_DISPATCH_TORCH_FOR_EACH_H
