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

    /// @brief Allocation from torch's active CUDA allocator, ordered for use on @p stream.
    /// @note The block is always taken on the current device's current torch stream, never on
    ///       @p stream. In torch's native caching allocator the allocation stream is not an
    ///       ordering point but the key a block is filed under: only later allocations on that
    ///       same stream can reuse it, and freeing performs no synchronization. A block keyed to
    ///       a stream torch does not allocate from (a nanovdb DeviceMesh stream, destroyed with
    ///       its mesh) would never be reused, and with expandable segments would make torch
    ///       synchronize on a dead handle when it releases the segment. The cudaMallocAsync
    ///       backend would free on that handle. Keying to the torch stream sidesteps all three.
    ///
    ///       @p stream is the stream the caller will write the memory on. When it differs from
    ///       the allocation stream it is made to wait on an event recorded right after the
    ///       allocation, so the caller's writes are ordered after the block's previous tenant,
    ///       whose work may still be queued on the torch stream. When the two coincide, the
    ///       common case for fvdb's builders, this is exactly raw_alloc. The call dispatches to
    ///       CUDACachingAllocator::get(), so a swapped-in backend or pluggable allocator is
    ///       honored.
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
        const cudaStream_t allocStream = c10::cuda::getCurrentCUDAStream().stream();
        void *p = c10::cuda::CUDACachingAllocator::raw_alloc_with_stream(bytes, allocStream);
        if (!p) {
            throw std::runtime_error("fvdb: TorchResource::allocate_async failed");
        }
        if (stream != allocStream) {
            cudaEvent_t ready = nullptr;
            C10_CUDA_CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
            C10_CUDA_CHECK(cudaEventRecord(ready, allocStream));
            C10_CUDA_CHECK(cudaStreamWaitEvent(stream, ready, 0));
            C10_CUDA_CHECK(cudaEventDestroy(ready));
        }
        return p;
    }

    /// @brief Free through torch's active CUDA allocator.
    /// @note The stream argument is deliberately ignored: raw_delete returns the block to the
    ///       torch stream it was allocated on (see allocate_async), where the next allocation on
    ///       that stream is ordered after this one's use by stream order. That is the same
    ///       contract torch's own tensor frees rely on. Use on any other stream is the caller's
    ///       to order before the free, as it is for tensors.
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

} // namespace fvdb

#endif // FVDB_TORCHRESOURCE_H
