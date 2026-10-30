// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/GridBatchData.h>
#include <fvdb/detail/VbmCache.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <limits>

namespace fvdb {
namespace detail {

// No device guard: setting a device fails in a child forked after CUDA initialized, where
// Python's cycle collector can free inherited grids. Destroying the event is best-effort.
// The event exists only if the batch has voxels.
VbmCache::~VbmCache() {
    if (mBuiltOn) {
        C10_CUDA_IGNORE_ERROR(cudaEventDestroy(mBuiltOn));
    }
}

const BatchVbm &
VbmCache::acquire(const GridBatchData &owner) {
    TORCH_CHECK(owner.device().is_cuda(), "VbmCache is only supported for CUDA grids");
    c10::cuda::CUDAGuard guard(owner.device());
    std::lock_guard<std::mutex> lock(mMutex);
    const auto stream = at::cuda::getCurrentCUDAStream(owner.device().index());

    cudaStreamCaptureStatus captureStatus;
    C10_CUDA_CHECK(cudaStreamIsCapturing(stream.stream(), &captureStatus));
    const bool capturing = captureStatus != cudaStreamCaptureStatusNone;

    if (!mOwner) {
        // A build captured into a graph would run only on replay, while the cache would already
        // count as built for eager use
        TORCH_CHECK(!capturing,
                    "The grid's VoxelBlockManagers are built on first use, which cannot happen "
                    "during CUDA graph capture. Run the op once before capturing.");
        build(owner, stream);
        return mVbm;
    }
    TORCH_CHECK(mOwner == &owner, "VbmCache::acquire called with a batch that does not own it");

    const bool synced =
        stream == mBuiltStream ||
        std::find(mSyncedStreams.begin(), mSyncedStreams.end(), stream) != mSyncedStreams.end();
    if (mBuffer.defined() && !synced) {
        if (capturing) {
            // A capture can only wait on an event recorded outside it as an external event node.
            // The wait orders replays, not later eager work, so the stream is not marked synced.
            C10_CUDA_CHECK(cudaStreamWaitEvent(stream, mBuiltOn, cudaEventWaitExternal));
            return mVbm;
        }
        // Execution ordering: this consumer stream must see the finished build.
        C10_CUDA_CHECK(cudaStreamWaitEvent(stream, mBuiltOn, 0));
        // Lifetime ordering: destroying the owner while this stream's kernels are in flight
        // must not recycle the buffer under them.
        mBuffer.record_stream(stream.unwrap());
        mSyncedStreams.push_back(stream);
    }
    return mVbm;
}

void
VbmCache::build(const GridBatchData &owner, const c10::cuda::CUDAStream &stream) {
    // Everything that can throw runs before any member is set, so a failed build leaves the cache
    // unbuilt and the next acquire() retries it

    // The host mirror of BatchVbmLayout, from host metadata, so the build reads nothing back from
    // the device. Grid placements ascend, so the last grid's end is the batch's block count.
    const int64_t numGrids = owner.batchSize();
    std::vector<nanovdb::tools::cuda::VoxelBlockManagerBatchEntry> hostEntries(numGrids);
    uint64_t blockCount = 0;
    uint64_t leafCount  = 0;
    for (int64_t bi = 0; bi < numGrids; ++bi) {
        const uint64_t numVoxels = uint64_t(owner.numVoxelsAt(bi));
        const uint64_t base      = blockBase(owner.cumVoxelsAt(bi), bi);
        hostEntries[bi] = {owner.deviceGridPtrAt(bi), numVoxels, base, uint32_t(leafCount)};
        blockCount      = base + ((numVoxels + kBlockWidth - 1) >> kLog2BlockWidth);
        leafCount += uint64_t(owner.numLeavesAt(bi));
    }
    TORCH_CHECK(leafCount <= std::numeric_limits<uint32_t>::max(),
                "VbmCache supports at most 2^32 - 1 leaves per batch");

    if (owner.totalVoxels() == 0) {
        mHostEntries = std::move(hostEntries);
        mVbm         = BatchVbm{nullptr, nullptr, numGrids, 0};
        mOwner       = &owner;
        return;
    }

    // One tensor-backed buffer: the caching allocator ties its free to the allocation stream and
    // to any record_stream()ed consumer streams. Layout, in uint64 words: jumpMap, then
    // firstLeafID packed two uint32 per word. firstLeafID is addressed through a uint32 pointer
    // into the same allocation, which avoids two tensor view calls on every cold build.
    const int64_t jumpMapWords     = int64_t(blockCount) * kJumpMapWordCount;
    const int64_t firstLeafIDWords = int64_t(blockCount + 1) / 2;
    torch::Tensor buffer =
        torch::empty({jumpMapWords + firstLeafIDWords},
                     torch::TensorOptions().dtype(torch::kUInt64).device(owner.device()));
    auto *jumpMap     = buffer.mutable_data_ptr<uint64_t>();
    auto *firstLeafID = reinterpret_cast<uint32_t *>(jumpMap + jumpMapWords);

    nanovdb::tools::cuda::buildVoxelBlockManagerBatch<kLog2BlockWidth>(BatchVbmLayout::of(owner),
                                                                       uint32_t(numGrids),
                                                                       uint32_t(leafCount),
                                                                       blockCount,
                                                                       firstLeafID,
                                                                       jumpMap,
                                                                       stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Recorded right after the build, so other streams wait on the build alone and the build
    // stream's handle is never used again
    cudaEvent_t builtOn = nullptr;
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&builtOn, cudaEventDisableTiming));
    const cudaError_t recorded = cudaEventRecord(builtOn, stream);
    if (recorded != cudaSuccess) {
        C10_CUDA_IGNORE_ERROR(cudaEventDestroy(builtOn));
        C10_CUDA_CHECK(recorded);
    }

    mHostEntries = std::move(hostEntries);
    mBuffer      = std::move(buffer);
    mBuiltOn     = builtOn;
    mBuiltStream = stream;
    mVbm         = BatchVbm{firstLeafID, jumpMap, numGrids, blockCount};
    mOwner       = &owner;
}

GridVbm
VbmCache::gridVbm(int64_t bi) const {
    TORCH_CHECK(mOwner, "VbmCache::gridVbm called before the VBMs were built");
    const auto &entry = mHostEntries.at(bi);
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
