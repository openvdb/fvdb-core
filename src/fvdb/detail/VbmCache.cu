// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/GridBatchData.h>
#include <fvdb/detail/VbmCache.h>

#include <nanovdb/tools/cuda/VoxelBlockManager.cuh>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

namespace fvdb {
namespace detail {

VbmCache::~VbmCache() {
    for (auto &[key, entry]: mEntries) {
        if (entry.builtOn) {
            c10::cuda::CUDAGuard guard(entry.deviceIdx);
            C10_CUDA_CHECK_WARN(cudaEventDestroy(entry.builtOn));
        }
    }
}

VbmCache::GridVbm
VbmCache::get(const GridBatchData &batch, int64_t bi) {
    TORCH_CHECK(batch.device().is_cuda(), "VbmCache is only supported for CUDA grids");
    const uint64_t key = batch.cumBytesAt(bi); // view-stable identity into the shared buffer

    std::lock_guard<std::mutex> lock(mMutex);
    const auto stream = at::cuda::getCurrentCUDAStream(batch.device().index());

    auto it = mEntries.find(key);
    if (it != mEntries.end()) {
        Entry &entry = it->second;
        if (entry.builtOn && stream.stream() != entry.builtStream) {
            // Execution ordering: this consumer stream must see the finished build.
            C10_CUDA_CHECK(cudaStreamWaitEvent(stream.stream(), entry.builtOn, 0));
            // Lifetime ordering: register the consumer stream with the caching allocator so
            // that destroying the owning GridBatchData while this stream's kernels are still
            // in flight cannot recycle the buffers under them.
            entry.firstLeafID.record_stream(stream.unwrap());
            entry.jumpMap.record_stream(stream.unwrap());
        }
        return entry.view;
    }

    Entry entry;
    entry.deviceIdx         = batch.device().index();
    const int64_t numVoxels = batch.numVoxelsAt(bi);
    if (numVoxels > 0) {
        c10::cuda::CUDAGuard guard(batch.device().index());
        const int64_t nBlocks = (numVoxels + kBlockWidth - 1) >> kLog2BlockWidth;

        // Tensor-backed buffers: the caching allocator ties frees to the allocation stream and
        // to any record_stream()ed consumer streams, which no raw allocation would. The
        // NanoVDB handle only sees non-owning views, and only for the duration of the build.
        // Sizing the VBM from host metadata (firstOffset = 1, lastOffset = numVoxels) selects
        // the in-place build overload, avoiding the blocking device read of activeVoxelCount
        // that the allocating overload performs.
        auto opts         = torch::TensorOptions().device(batch.device());
        entry.firstLeafID = torch::empty({nBlocks}, opts.dtype(torch::kInt32));
        entry.jumpMap     = torch::empty({nBlocks * kJumpMapWordCount}, opts.dtype(torch::kInt64));

        nanovdb::tools::VoxelBlockManagerHandle<VbmBufferView> handle(
            VbmBufferView(entry.firstLeafID.data_ptr()),
            VbmBufferView(entry.jumpMap.data_ptr()),
            uint64_t(nBlocks),
            /*firstOffset=*/1,
            /*lastOffset=*/uint64_t(numVoxels));
        nanovdb::tools::cuda::buildVoxelBlockManager<kLog2BlockWidth>(
            batch.deviceGridPtrAt(bi), handle, stream.stream());
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        C10_CUDA_CHECK(cudaEventCreateWithFlags(&entry.builtOn, cudaEventDisableTiming));
        C10_CUDA_CHECK(cudaEventRecord(entry.builtOn, stream.stream()));
        entry.builtStream = stream.stream();

        entry.view = GridVbm{reinterpret_cast<const uint32_t *>(entry.firstLeafID.data_ptr()),
                             reinterpret_cast<const uint64_t *>(entry.jumpMap.data_ptr()),
                             uint32_t(nBlocks),
                             /*firstOffset=*/1,
                             /*lastOffset=*/uint64_t(numVoxels)};
    }
    return mEntries.emplace(key, std::move(entry)).first->second.view;
}

} // namespace detail
} // namespace fvdb
