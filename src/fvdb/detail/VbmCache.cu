// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/GridBatchData.h>
#include <fvdb/detail/VbmCache.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cub/block/block_scan.cuh>

#include <algorithm>
#include <limits>

namespace fvdb {
namespace detail {

namespace {

constexpr int kFillEntriesThreads = 256;

// Fills the batch's VBM entries on the device from its grid metadata, so the build needs no
// host-to-device upload. One block scans the grids' VBM block counts, kFillEntriesThreads grids
// per pass. Matches the host entries computed in VbmCache::build.
__global__ void __launch_bounds__(kFillEntriesThreads)
fillEntriesKernel(const GridBatchData::Accessor acc, BatchVbm::Entry *entries) {
    using BlockScan = cub::BlockScan<uint64_t, kFillEntriesThreads>;
    __shared__ typename BlockScan::TempStorage scanStorage;

    const int64_t numGrids = acc.batchSize();
    uint64_t carry         = 0;
    for (int64_t first = 0; first < numGrids; first += kFillEntriesThreads) {
        const int64_t bi = first + threadIdx.x;
        uint64_t voxels  = 0;
        if (bi < numGrids) {
            const int64_t end = bi + 1 < numGrids ? acc.voxelOffset(bi + 1) : acc.totalVoxels();
            voxels            = uint64_t(end - acc.voxelOffset(bi));
        }
        const uint64_t blocks = (voxels + VbmCache::kBlockWidth - 1) >> VbmCache::kLog2BlockWidth;
        uint64_t blockBase;
        uint64_t passBlocks;
        BlockScan(scanStorage).ExclusiveSum(blocks, blockBase, passBlocks);
        if (bi < numGrids) {
            entries[bi] = BatchVbm::Entry{
                acc.grid(bi), voxels, carry + blockBase, uint32_t(acc.leafOffset(bi))};
        }
        carry += passBlocks;
        __syncthreads(); // scanStorage is reused by the next pass
    }
}

} // namespace

// No device guard: setting a device fails in a child forked after CUDA initialized, where
// Python's cycle collector can free inherited grids. Destroying the event is best-effort.
// The event exists only if a second stream acquired the cache.
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
    if (mBuffer.defined() && !synced) {
        // Recorded on first need rather than at build time, since most batches only ever see one
        // stream. The build stream is in order, so a later record still follows the build.
        if (!mBuiltOn) {
            C10_CUDA_CHECK(cudaEventCreateWithFlags(&mBuiltOn, cudaEventDisableTiming));
            C10_CUDA_CHECK(cudaEventRecord(mBuiltOn, mBuiltStream));
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

    // One tensor-backed buffer: the caching allocator ties its free to the allocation stream and
    // to any record_stream()ed consumer streams. Layout, each part 8-byte aligned:
    // jumpMap (uint64), entries, firstLeafID (uint32).
    static_assert(sizeof(BatchVbm::Entry) % alignof(uint64_t) == 0);
    const size_t jumpMapBytes     = blockCount * kJumpMapWordCount * sizeof(uint64_t);
    const size_t entryBytes       = size_t(numGrids) * sizeof(BatchVbm::Entry);
    const size_t firstLeafIDBytes = blockCount * sizeof(uint32_t);
    mBuffer           = torch::empty({int64_t(jumpMapBytes + entryBytes + firstLeafIDBytes)},
                           torch::TensorOptions().dtype(torch::kUInt8).device(owner.device()));
    auto *base        = mBuffer.mutable_data_ptr<uint8_t>();
    auto *jumpMap     = reinterpret_cast<uint64_t *>(base);
    auto *entries     = reinterpret_cast<BatchVbm::Entry *>(base + jumpMapBytes);
    auto *firstLeafID = reinterpret_cast<uint32_t *>(base + jumpMapBytes + entryBytes);

    fillEntriesKernel<<<1, kFillEntriesThreads, 0, stream>>>(owner.deviceAccessor(), entries);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    nanovdb::tools::cuda::buildVoxelBlockManagerBatch<kLog2BlockWidth>(
        entries, uint32_t(numGrids), uint32_t(leafCount), blockCount, firstLeafID, jumpMap, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    mBuiltStream = stream;

    mVbm = BatchVbm{firstLeafID, jumpMap, entries, numGrids, blockCount};
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
