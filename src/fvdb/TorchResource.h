// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_TORCHRESOURCE_H
#define FVDB_TORCHRESOURCE_H

#include <nanovdb/cuda/DeviceResource.h>

#include <c10/cuda/CUDACachingAllocator.h>

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

    /// @brief Stream-ordered allocation from torch's active CUDA allocator.
    /// @note raw_alloc_with_stream records @p stream against the block so torch
    ///       defers reuse until work on it completes, matching the stream-ordered
    ///       semantics of the cudaMallocAsync call it replaces. Allocation
    ///       happens on the current device, like cudaMallocAsync. The call
    ///       dispatches to CUDACachingAllocator::get(), so a swapped-in backend
    ///       or pluggable allocator is honored.
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
    /// @note The stream argument is deliberately ignored: raw_delete relies on
    ///       the stream recorded at allocation time — the native backend's
    ///       per-stream event tracking, or the alloc-time stream Torch hands a
    ///       pluggable allocator's free function — so the free is safe without
    ///       ordering on the caller's stream. This is the same contract Torch's
    ///       own tensor frees rely on.
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
