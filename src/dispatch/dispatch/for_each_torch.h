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

#if defined(__CUDACC__)

// Func by VALUE, not by reference:
// CUDA kernel parameters are always copied to device memory regardless of how they're declared.
// Using a reference here would be misleading and potentially dangerous (dangling reference to
// host memory). By-value makes the copy semantics explicit.
//
// Direct call func(...), not std::invoke:
// std::invoke is not available in device code. Direct call syntax works for lambdas and
// function objects with operator(), which are the expected callable types for device code.
template <typename Tag, typename Func>
    requires tag_match<Tag, torch::kCUDA>
__global__ void
for_each_cuda_kernel(Tag t, int64_t count, Func func) {
    auto const idx = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
                     static_cast<int64_t>(threadIdx.x);
    if (idx < count) {
        func(t, idx);
    }
}

// Func&& is a FORWARDING REFERENCE (universal reference):
// In a template context, T&& where T is a deduced template parameter binds to both lvalues
// and rvalues. This allows callers to pass temporaries (moved) or named variables (copied).
//
// std::forward<Func>(func) when passing to the kernel:
// We use std::forward here because this is the FINAL DESTINATION for the callable - it will
// be copied into kernel parameters exactly once. Forwarding preserves move semantics if the
// caller passed an rvalue, avoiding an unnecessary copy before the kernel copies it again.
template <typename Tag, typename Func>
    requires tag_match<Tag, torch::kCUDA>
void
for_each(Tag t, int64_t count, Func &&func) {
    constexpr int block_dim = 256;
    auto const grid_dim     = (count + block_dim - 1) / block_dim;
    cudaStream_t stream     = c10::cuda::getCurrentCUDAStream().stream();
    for_each_cuda_kernel<<<grid_dim, block_dim, 0, stream>>>(t, count, std::forward<Func>(func));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

//------------------------------------------------------------------------------
// PrivateUse1 (Universal Memory) version
//------------------------------------------------------------------------------
// PrivateUse1 indicates data in CUDA Unified Memory, accessible from all GPUs.
// Work is distributed across all available CUDA devices, each processing a chunk.

// Func by VALUE (same rationale as CUDA kernel above).
//
// STRIDED ITERATION instead of early-return:
// Unlike the single-GPU CUDA kernel which launches exactly enough threads, the PrivateUse1
// kernel may have fewer threads than work items (when work is chunked across devices).
// Strided iteration (idx += blockDim.x * gridDim.x) allows each thread to process multiple
// elements, covering the full chunk regardless of grid size.
template <typename Tag, typename Func>
    requires tag_match<Tag, torch::kPrivateUse1>
__global__ void
for_each_pvt1_kernel(Tag t, int64_t chunk_count, int64_t chunk_offset, Func func) {
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
                       static_cast<int64_t>(threadIdx.x);
         idx < chunk_count;
         idx += static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x)) {
        func(t, idx + chunk_offset);
    }
}

// Func&& is a FORWARDING REFERENCE (same rationale as CUDA version above).
//
// NO std::forward when passing to multiple kernels:
// Unlike the single-GPU case, we launch kernels on MULTIPLE devices in a loop. If we forwarded
// on the first iteration, we could move from the callable, leaving it in a moved-from state
// for subsequent devices. We pass `func` as an lvalue to each kernel launch (each kernel will
// copy it to device memory anyway).
//
// SYNCHRONIZATION at the end:
// All device streams must complete before returning, since the caller expects the operation
// to be finished. Each device's stream is synchronized after all kernels are launched.
template <typename Tag, typename Func>
    requires tag_match<Tag, torch::kPrivateUse1>
void
for_each(Tag t, int64_t count, Func &&func) {
    constexpr int block_dim = 256;

    auto const device_count = c10::cuda::device_count();

    for (c10::DeviceIndex device_id = 0; device_id < device_count; ++device_id) {
        C10_CUDA_CHECK(cudaSetDevice(device_id));
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_id).stream();

        // Divide work evenly across devices, with remainder going to earlier devices
        auto const base_chunk  = count / device_count;
        auto const remainder   = count % device_count;
        auto const chunk_count = base_chunk + (device_id < remainder ? 1 : 0);

        // Calculate offset: sum of all previous chunks
        // Devices 0..remainder-1 get (base_chunk+1) each, devices remainder..N get base_chunk
        int64_t chunk_offset = 0;
        if (device_id < remainder) {
            chunk_offset = device_id * (base_chunk + 1);
        } else {
            chunk_offset = remainder * (base_chunk + 1) + (device_id - remainder) * base_chunk;
        }

        if (chunk_count > 0) {
            auto const grid_dim = (chunk_count + block_dim - 1) / block_dim;
            for_each_pvt1_kernel<<<grid_dim, block_dim, 0, stream>>>(
                t, chunk_count, chunk_offset, func);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }

    // Synchronize all streams - operation must be complete before returning
    for (c10::DeviceIndex device_id = 0; device_id < device_count; ++device_id) {
        c10::cuda::getCurrentCUDAStream(device_id).synchronize();
    }
}

#endif // __CUDACC__

// Func&& is a FORWARDING REFERENCE (universal reference):
// Same as CUDA version - accepts both lvalues and rvalues from callers.
//
// std::invoke(func, ...) instead of func(...):
// std::invoke is more general than direct call syntax. It handles member function pointers,
// member data pointers, and regular callables uniformly. This matches standard library
// algorithm conventions (e.g., std::for_each uses INVOKE semantics).
//
// NO std::forward when calling:
// We deliberately do NOT use std::forward<Func>(func)(t, i) here because the callable is
// invoked MULTIPLE TIMES in the loop. If we forwarded an rvalue, std::forward would cast it
// back to an rvalue reference on the first iteration, potentially moving from it and leaving
// it in a moved-from state for subsequent iterations. Treat as lvalue when reusing.
//
// at::parallel_for:
// Uses ATen's built-in parallel_for which respects PyTorch's thread pool settings
// (at::set_num_threads, OMP_NUM_THREADS, etc.) and provides efficient work distribution.
// The grain_size of 1 allows ATen to determine optimal chunking based on workload.
template <typename Tag, typename Func>
    requires tag_match<Tag, torch::kCPU>
void
for_each(Tag t, int64_t count, Func &&func) {
    at::parallel_for(0, count, /*grain_size=*/1, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            std::invoke(func, t, i);
        }
    });
}

} // namespace dispatch

#endif // DISPATCH_DISPATCH_FOR_EACH_TORCH_H
