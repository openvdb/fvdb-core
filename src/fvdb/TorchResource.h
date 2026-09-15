// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_TORCHRESOURCE_H
#define FVDB_TORCHRESOURCE_H

#include <nanovdb/cuda/DeviceResource.h>

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace fvdb {

/// @brief NanoVDB stream-ordered memory resource backed by PyTorch's currently
///        active CUDA allocator.
///
///        c10::cuda::CUDACachingAllocator is a namespace, not a concrete
///        allocator: its free functions raw_alloc_with_stream / raw_delete
///        dispatch through CUDACachingAllocator::get(), the runtime-swappable
///        c10::cuda::CUDAAllocator* Torch itself allocates tensors from. This
///        resource therefore follows whatever allocator the user has installed —
///        the native caching allocator (including PYTORCH_CUDA_ALLOC_CONF knobs),
///        the cudaMallocAsync backend (PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync),
///        or a user-provided allocator installed via
///        torch.cuda.memory.change_current_allocator(CUDAPluggableAllocator(...)).
///
///        Passed as the ResourceT template parameter of NanoVDB's CUDA builders
///        (PointsToGrid / DilateGrid / MergeGrids / PruneGrid / RefineGrid /
///        CoarsenGrid) — always via the fvdb::BuilderResource alias
///        (BuilderResource.h), never named directly at call sites — it routes
///        their internal device scratch — O(N-points) sort
///        keys, CUB temp storage, topology mask buffers — through the same pool
///        that fvdb / PyTorch tensors use. Without this, nanoVDB's default
///        DeviceResource allocates from a second cudaMallocAsync pool that
///        partitions VRAM against torch's pool, and large workloads (e.g.
///        multi-frame TSDF integration) OOM even when the GPU has free memory in
///        aggregate.
///
///        The resource is stateless, so builders can bind the shared instance
///        returned by nanovdb::cuda::default_resource<TorchResource>() — naming
///        the template parameter at a call site is sufficient, no instance needs
///        to be threaded through.
///
///        Set FVDB_NANOVDB_TRACE_ALLOCS=1 in the environment to trace allocations
///        of 256 KiB and larger to stderr (a value starting with '2' traces every
///        allocation). Useful for diagnosing topology-op memory blowup on large
///        scenes.
struct TorchResource : nanovdb::cuda::SyncFromAsync<TorchResource> {
    /// Alignment guaranteed by every allocation. Torch's native caching
    /// allocator returns blocks aligned to at least 512 bytes and the
    /// cudaMallocAsync backend to at least 256, so advertising nanoVDB's
    /// conventional 256 (matching cuda::DeviceResource) is satisfied and the
    /// alignment parameter below can be ignored. A pluggable allocator wrapping
    /// any cudaMalloc-family call satisfies 256 as well.
    static constexpr size_t DEFAULT_ALIGNMENT = 256;

    /// @brief Stream-ordered allocation from torch's active CUDA allocator, for scratch that
    ///        is allocated, used and freed on @p stream.
    /// @note The block is keyed to @p stream (raw_alloc_with_stream). In torch's native caching
    ///       allocator the allocation stream is the key a block is filed under: only later
    ///       allocations on that same stream can reuse it, and freeing does no synchronization.
    ///       That is what makes deallocate_async safe without ordering: the builders destroy
    ///       scratch right after enqueuing the kernels that read it (PointsToGrid frees its
    ///       per-tile counts behind an in-flight scan), and the block can only go to a later
    ///       allocation on the same stream, behind that work. Keyed to any other stream it could
    ///       be reused, or under the cudaMallocAsync backend freed, ahead of those reads.
    ///       Allocation happens on the current device, like cudaMallocAsync. The call dispatches
    ///       to CUDACachingAllocator::get(), so a swapped-in backend or pluggable allocator is
    ///       honored.
    ///
    ///       Consequently @p stream must outlive every block allocated on it (legacy stream 0
    ///       and torch's pool streams do). Storage that torch will own after the op, whose
    ///       writer stream may not, goes through TorchStorageResource instead.
    void *
    allocate_async(size_t bytes, size_t /*alignment*/, cudaStream_t stream) {
        if (const char *env = std::getenv("FVDB_NANOVDB_TRACE_ALLOCS")) {
            const size_t cutoff =
                (env[0] == '2') ? 0 : (1ull << 18); // '2' = trace all, else >= 256 KiB
            if (bytes >= cutoff) {
                std::fprintf(stderr,
                             "[fvdb/nanovdb] TorchResource alloc %12zu bytes (%.3f MB)\n",
                             bytes,
                             double(bytes) / 1e6);
            }
        }
        void *p = c10::cuda::CUDACachingAllocator::raw_alloc_with_stream(bytes, stream);
        if (!p) {
            throw std::runtime_error("fvdb: TorchResource::allocate_async failed");
        }
        return p;
    }

    /// @brief Free through torch's active CUDA allocator.
    /// @note The stream argument is deliberately ignored: raw_delete returns the block to the
    ///       stream it was allocated on (see allocate_async), where the next allocation is
    ///       ordered after this one's use by stream order. That is the same contract torch's own
    ///       tensor frees rely on. Use on any other stream is the caller's to order before the
    ///       free, as it is for tensors.
    void
    deallocate_async(void *p, size_t /*bytes*/, size_t /*alignment*/, cudaStream_t /*stream*/) {
        if (p == nullptr) {
            return;
        }
        c10::cuda::CUDACachingAllocator::raw_delete(p);
    }
};

static_assert(nanovdb::cuda::is_async_resource<TorchResource>::value,
              "TorchResource must model nanoVDB's stream-ordered AsyncResource concept");

/// @brief Torch-allocator resource for device storage that torch owns after the op that
///        produced it: grid buffers.
///
///        The block is always taken on the current device's current torch stream, never on
///        the caller's, because that is the only stream torch will reuse it from and it is a
///        stream that outlives the block. A nanovdb builder may write storage on a stream
///        of its own (DistributedPointsToGrid uses its DeviceMesh stream, which dies with the
///        mesh); a block keyed to such a stream would never be reused, would make torch
///        synchronize on a dead handle when releasing an expandable segment, and would be
///        cudaFreeAsync'd on it under that backend.
///
///        @p stream is the stream the caller will write the memory on. When it differs from
///        the allocation stream it is made to wait on an event recorded right after the
///        allocation, so the caller's writes are ordered behind the block's previous tenant,
///        whose work may still be queued on the torch stream. Nothing orders the free: as
///        for a tensor, the block returns to the torch stream on release, so a caller whose
///        writer stream differs must synchronize it before releasing the storage. The
///        builders fvdb hands TorchDeviceBuffer to (PointsToGrid, DistributedPointsToGrid,
///        the topology builders, PadGrid) do so before returning their handle.
///
///        This is the body #770's TorchDeviceResource adopts; TorchDeviceBuffer uses it now.
struct TorchStorageResource : nanovdb::cuda::SyncFromAsync<TorchStorageResource> {
    static constexpr size_t DEFAULT_ALIGNMENT = TorchResource::DEFAULT_ALIGNMENT;

    void *
    allocate_async(size_t bytes, size_t alignment, cudaStream_t stream) {
        const cudaStream_t allocStream = c10::cuda::getCurrentCUDAStream().stream();
        void *p = TorchResource{}.allocate_async(bytes, alignment, allocStream);
        if (stream == allocStream) {
            return p;
        }
        cudaEvent_t ready = nullptr;
        cudaError_t err   = cudaEventCreateWithFlags(&ready, cudaEventDisableTiming);
        if (err == cudaSuccess) {
            err = cudaEventRecord(ready, allocStream);
        }
        if (err == cudaSuccess) {
            err = cudaStreamWaitEvent(stream, ready, 0);
        }
        if (ready) {
            const cudaError_t destroyErr = cudaEventDestroy(ready);
            if (err == cudaSuccess) {
                err = destroyErr;
            }
        }
        if (err != cudaSuccess) {
            c10::cuda::CUDACachingAllocator::raw_delete(p);
            C10_CUDA_CHECK(err);
        }
        return p;
    }

    void
    deallocate_async(void *p, size_t bytes, size_t alignment, cudaStream_t stream) {
        TorchResource{}.deallocate_async(p, bytes, alignment, stream);
    }
};

static_assert(nanovdb::cuda::is_async_resource<TorchStorageResource>::value,
              "TorchStorageResource must model nanoVDB's stream-ordered AsyncResource concept");

} // namespace fvdb

#endif // FVDB_TORCHRESOURCE_H
