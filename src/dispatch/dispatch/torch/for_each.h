// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// for_each: Parallel index generation for CPU, CUDA, and PrivateUse1.
//
// for_each is a parallel index generator. It iterates [0, count) on the device
// specified by the tag, calling the functor once per index.
//
// Functor signature: void(Tag, int64_t idx)
//
// The tag is passed through to the functor, enabling concept-constrained
// overload sets that specialize per-device or per-scalar-type.
//
// Device selection:
//   CPU:         default_thread_pool (work-stealing), per-element loop
//   CUDA:        one-element-per-thread grid-stride kernel
//   PrivateUse1: multi-GPU distribution, grid-stride per device
//
// GPU block size:
//   Defaults to 256. Op authors who need a different block size can inject
//   it into the tag via tag_add:
//
//     for_each(tag_add<Tag, block_dim::b128>{}, count, func);
//
//   The CUDA and PrivateUse1 overloads extract block_dim from the tag if
//   present, otherwise use 256.
//
// GPU grid size:
//   Determined by cudaOccupancyMaxActiveBlocksPerMultiprocessor — launches
//   exactly enough blocks to saturate all SMs on the current device, capped
//   at the number of blocks the work actually requires. The grid-stride loop
//   handles any count regardless of grid size.
//
// Vectorization:
//   for_each is a scalar index generator — one element per functor call,
//   one thread per element (interleaved), optimal coalescing for scalar
//   loads. It does NOT provide vectorization infrastructure. Vectorized
//   loads (float4, etc.) require contiguous-per-thread layout, masked
//   load/store for tail handling, vector type selection, and alignment
//   management — all of which belong in a future for_each_vectorized.
//   for_each does not prevent manual vectorization, but it does not
//   facilitate it.
//
#ifndef DISPATCH_DISPATCH_TORCH_FOR_EACH_H
#define DISPATCH_DISPATCH_TORCH_FOR_EACH_H

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

namespace detail {

// Extract block_dim from the tag if present, otherwise return 256.
template <typename Tag>
consteval int
for_each_block_dim() {
    if constexpr (with_type<Tag, block_dim>) {
        return static_cast<int>(tag_get<block_dim, Tag>());
    } else {
        return 256;
    }
}

// Compute the grid size for a grid-stride kernel: enough blocks to saturate
// all SMs on the current device, but no more than the work requires.
template <int BlockDim, typename Kernel>
int
for_each_grid_dim(Kernel kernel, int64_t count) {
    int blocks_per_sm = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, kernel, BlockDim, /*dynamicSMemSize=*/0));

    int device = 0;
    C10_CUDA_CHECK(cudaGetDevice(&device));

    int sm_count = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));

    int64_t const max_active_blocks = static_cast<int64_t>(blocks_per_sm) * sm_count;
    int64_t const blocks_needed     = (count + BlockDim - 1) / BlockDim;
    return static_cast<int>(std::min(blocks_needed, max_active_blocks));
}

} // namespace detail

template <int BlockDim, typename Tag, typename Func>
    requires with_value<Tag, torch::kCUDA>
__global__ __launch_bounds__(BlockDim) void
for_each_cuda_kernel(Tag tag, int64_t count, Func func) {
    int64_t const tid         = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t const grid_stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    for (int64_t idx = tid; idx < count; idx += grid_stride) {
        func(tag, idx);
    }
}

template <typename Tag, typename Func>
    requires with_value<Tag, torch::kCUDA>
void
for_each(Tag tag, int64_t count, Func &&func) {
    if (count == 0)
        return;

    constexpr int kBlockDim = detail::for_each_block_dim<Tag>();

    using F = std::decay_t<Func>;
    int const grid_dim =
        detail::for_each_grid_dim<kBlockDim>(for_each_cuda_kernel<kBlockDim, Tag, F>, count);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    for_each_cuda_kernel<kBlockDim>
        <<<grid_dim, kBlockDim, 0, stream>>>(tag, count, F(std::forward<Func>(func)));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// =============================================================================
// PrivateUse1 for_each (multi-GPU)
// =============================================================================

template <int BlockDim, typename Tag, typename Func>
    requires with_value<Tag, torch::kPrivateUse1>
__global__ __launch_bounds__(BlockDim) void
for_each_pvt1_kernel(Tag tag, int64_t chunk_count, int64_t chunk_offset, Func func) {
    int64_t const tid         = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t const grid_stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    for (int64_t local_idx = tid; local_idx < chunk_count; local_idx += grid_stride) {
        func(tag, local_idx + chunk_offset);
    }
}

template <typename Tag, typename Func>
    requires with_value<Tag, torch::kPrivateUse1>
void
for_each(Tag tag, int64_t count, Func &&func) {
    if (count == 0)
        return;

    constexpr int kBlockDim = detail::for_each_block_dim<Tag>();

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
            int const grid_dim = detail::for_each_grid_dim<kBlockDim>(
                for_each_pvt1_kernel<kBlockDim, Tag, F>, chunk_count);

            for_each_pvt1_kernel<kBlockDim>
                <<<grid_dim, kBlockDim, 0, stream>>>(tag, chunk_count, chunk_offset, f);
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
