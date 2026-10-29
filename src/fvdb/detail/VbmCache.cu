// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/GridBatchData.h>
#include <fvdb/detail/VbmCache.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <cstring>
#include <limits>

namespace fvdb {
namespace detail {

// No device guard: setting a device fails in a child forked after CUDA initialized, where
// Python's cycle collector can free inherited grids. Destroying the event is best-effort.
VbmCache::~VbmCache() {
    if (mBuiltOn) {
        C10_CUDA_IGNORE_ERROR(cudaEventDestroy(mBuiltOn));
    }
}

const BatchVbm &
VbmCache::acquire(const GridBatchData &owner) {
    TORCH_CHECK(owner.device().is_cuda(), "VbmCache is only supported for CUDA grids");
    std::lock_guard<std::mutex> lock(mMutex);
    const auto stream = at::cuda::getCurrentCUDAStream(owner.device().index());

    if (!mOwner) {
        build(owner, stream);
        return mVbm;
    }
    TORCH_CHECK(mOwner == &owner, "VbmCache::acquire called with a batch that does not own it");

    const bool synced =
        stream == mBuiltStream ||
        std::find(mSyncedStreams.begin(), mSyncedStreams.end(), stream) != mSyncedStreams.end();
    if (mBuiltOn && !synced) {
        // Execution ordering: this consumer stream must see the finished build.
        C10_CUDA_CHECK(cudaStreamWaitEvent(stream, mBuiltOn, 0));
        // Lifetime ordering: destroying the owner while this stream's kernels are in flight
        // must not recycle the buffers under them.
        mFirstLeafID.record_stream(stream.unwrap());
        mJumpMap.record_stream(stream.unwrap());
        mEntries.record_stream(stream.unwrap());
        mSyncedStreams.push_back(stream);
    }
    return mVbm;
}

void
VbmCache::build(const GridBatchData &owner, const c10::cuda::CUDAStream &stream) {
    c10::cuda::CUDAGuard guard(owner.device());
    mOwner = &owner;

    // Bases come from host metadata, so the build reads nothing back from the device.
    const int64_t numGrids = owner.batchSize();
    mHostEntries.resize(numGrids);
    uint64_t blockCount = 0;
    uint64_t leafCount  = 0;
    for (int64_t bi = 0; bi < numGrids; ++bi) {
        const uint64_t numVoxels = uint64_t(owner.numVoxelsAt(bi));
        mHostEntries[bi] =
            BatchVbm::Entry{owner.deviceGridPtrAt(bi), numVoxels, blockCount, uint32_t(leafCount)};
        blockCount += (numVoxels + kBlockWidth - 1) >> kLog2BlockWidth;
        leafCount += uint64_t(owner.numLeavesAt(bi));
    }
    TORCH_CHECK(leafCount <= std::numeric_limits<uint32_t>::max(),
                "VbmCache supports at most 2^32 - 1 leaves per batch");
    if (blockCount == 0) {
        mVbm = BatchVbm{nullptr, nullptr, nullptr, numGrids, 0};
        return;
    }

    // Tensor-backed buffers: the caching allocator ties frees to the allocation stream and to
    // any record_stream()ed consumer streams.
    const auto opts = torch::TensorOptions().device(owner.device());
    mFirstLeafID    = torch::empty({int64_t(blockCount)}, opts.dtype(torch::kUInt32));
    mJumpMap = torch::empty({int64_t(blockCount) * kJumpMapWordCount}, opts.dtype(torch::kUInt64));

    const size_t entryBytes = mHostEntries.size() * sizeof(BatchVbm::Entry);
    auto hostEntries        = torch::empty(
        {int64_t(entryBytes)}, torch::TensorOptions().dtype(torch::kUInt8).pinned_memory(true));
    std::memcpy(hostEntries.data_ptr(), mHostEntries.data(), entryBytes);
    mEntries            = hostEntries.to(owner.device(), /*non_blocking=*/true);
    const auto *entries = reinterpret_cast<const BatchVbm::Entry *>(mEntries.const_data_ptr());

    nanovdb::tools::cuda::buildVoxelBlockManagerBatch<kLog2BlockWidth>(
        entries,
        uint32_t(numGrids),
        uint32_t(leafCount),
        blockCount,
        mFirstLeafID.mutable_data_ptr<uint32_t>(),
        mJumpMap.mutable_data_ptr<uint64_t>(),
        stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    C10_CUDA_CHECK(cudaEventCreateWithFlags(&mBuiltOn, cudaEventDisableTiming));
    C10_CUDA_CHECK(cudaEventRecord(mBuiltOn, stream));
    mBuiltStream = stream;

    mVbm = BatchVbm{mFirstLeafID.const_data_ptr<uint32_t>(),
                    mJumpMap.const_data_ptr<uint64_t>(),
                    entries,
                    numGrids,
                    blockCount};
}

GridVbm
VbmCache::gridVbm(int64_t bi) const {
    TORCH_CHECK(mOwner, "VbmCache::gridVbm called before the VBMs were built");
    const BatchVbm::Entry &entry = mHostEntries.at(bi);
    if (entry.voxelCount == 0) {
        return GridVbm{};
    }
    return GridVbm{mVbm.firstLeafID + entry.blockBase,
                   mVbm.jumpMap + entry.blockBase * kJumpMapWordCount,
                   uint32_t((entry.voxelCount + kBlockWidth - 1) >> kLog2BlockWidth),
                   /*firstOffset=*/1,
                   /*lastOffset=*/entry.voxelCount};
}

} // namespace detail
} // namespace fvdb
