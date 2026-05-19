// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// FVDB FORK: This header overrides the upstream nanoVDB copy of
// DeviceResource.h. The functional change is that device-side
// allocations go through PyTorch's CUDA caching allocator
// (c10::cuda::CUDACachingAllocator) instead of cudaMallocAsync /
// cudaFreeAsync.
//
// Why: `DeviceResource` is the allocator that nanoVDB's
// PointsToGrid / DilateGrid / MergeGrids / TopologyBuilder use for
// their *internal* scratch buffers (O(N_points) sort keys, CUB
// temporary storage, node-count arrays, etc.). Upstream routes these
// through cudaMallocAsync, which on a torch process creates a second,
// independent CUDA memory pool next to torch's caching allocator --
// so at scale (N ~ 10 M input points) the two pools partition VRAM
// and one of them OOMs even though the aggregate has plenty of free
// memory. Routing nanoVDB scratch through torch's allocator collapses
// the two pools into one, which is exactly what we want.
//
// This is the partner override to nanovdb_overrides/nanovdb/cuda/
// DeviceBuffer.h -- that file handles the grid-handle-sized
// allocations; this one handles the per-point scratch. Together they
// cover every cudaMallocAsync call site reachable from the nanoVDB
// topology ops we care about.
//
// If you need to resync with upstream, diff against
// build/.../_deps/nanovdb-src/nanovdb/nanovdb/cuda/DeviceResource.h
// and port the delta -- only allocateAsync / deallocateAsync should
// differ.

#ifndef NANOVDB_CUDA_DEVICERESOURCE_H_HAS_BEEN_INCLUDED
#define NANOVDB_CUDA_DEVICERESOURCE_H_HAS_BEEN_INCLUDED

#include <cuda_runtime_api.h>
#include <nanovdb/util/cuda/Util.h>

// FVDB FORK: route device allocations through the PyTorch caching allocator.
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdio>
#include <cstdlib>

namespace nanovdb {

namespace cuda {

class DeviceResource
{
public:
    // cudaMalloc aligns memory to 256 bytes by default
    static constexpr size_t DEFAULT_ALIGNMENT = 256;

    static void* allocateAsync(size_t bytes, size_t /*alignment*/, cudaStream_t stream) {
        // FVDB FORK: use PyTorch's caching allocator.
        //
        // `raw_alloc_with_stream` returns at least 512-byte aligned blocks,
        // which is stricter than nanoVDB's DEFAULT_ALIGNMENT of 256, so we
        // can safely ignore the alignment parameter here.
        //
        // We go through `raw_alloc_with_stream` (rather than the plain
        // `raw_alloc`) so that torch's allocator records this stream
        // against the block and defers reuse until work on the stream
        // completes. That matches the stream-ordered semantics the
        // original `cudaMallocAsync(..., stream)` call had.
        if (const char *env = std::getenv("FVDB_NANOVDB_TRACE_ALLOCS")) {
            const size_t cutoff = (env[0] == '2') ? 0 : (1ull << 18);
            if (bytes >= cutoff) {
                std::fprintf(stderr,
                             "[fvdb/nanovdb] DeviceResource  alloc %12zu bytes (%.3f MB)\n",
                             bytes, double(bytes) / 1e6);
            }
        }
        void *p = c10::cuda::CUDACachingAllocator::raw_alloc_with_stream(bytes, stream);
        if (!p) {
            throw std::runtime_error("fvdb: DeviceResource::allocateAsync failed");
        }
        return p;
    }

    static void deallocateAsync(void *p, size_t /*bytes*/, size_t /*alignment*/, cudaStream_t stream) {
        (void)stream;
        if (p == nullptr) return;
        c10::cuda::CUDACachingAllocator::raw_delete(p);
    }
};

}

} // namespace nanovdb::cuda

#endif // end of NANOVDB_CUDA_DEVICERESOURCE_H_HAS_BEEN_INCLUDED
