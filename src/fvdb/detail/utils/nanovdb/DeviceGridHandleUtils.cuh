// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_NANOVDB_DEVICEGRIDHANDLEUTILS_CUH
#define FVDB_DETAIL_UTILS_NANOVDB_DEVICEGRIDHANDLEUTILS_CUH

#include <fvdb/GridBatchData.h>
#include <fvdb/GridStorage.h>
#include <fvdb/TorchDeviceResource.h>
#include <fvdb/detail/utils/cuda/StreamOrdering.h>
#include <fvdb/detail/utils/nanovdb/CreateEmptyGridStorage.h>
#include <fvdb/detail/utils/nanovdb/GridHeaderUtils.h>

#include <nanovdb/GridHandle.h>
#include <nanovdb/cuda/HandleStorage.h>

#include <c10/core/DeviceGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace fvdb::detail {

/// @brief Stamps a grid's position in its batch into its header and disables the checksum, which
///        the write invalidates. The one header policy for every site that lays grids into a
///        buffer by hand (merge, replication, concatenation, compaction).
__device__ inline void
stampGridHeader(nanovdb::GridData *data, uint32_t gridIndex, uint32_t gridCount) {
    data->mGridIndex = gridIndex;
    data->mGridCount = gridCount;
    data->mChecksum.disable();
}

/// @brief stampGridHeader on one grid. One thread.
__global__ inline void
updateGridCountAndZeroChecksum(nanovdb::GridData *d_data, uint32_t gridIndex, uint32_t gridCount) {
    stampGridHeader(d_data, gridIndex, gridCount);
}

/// @brief stampGridHeader on @p gridCount grids laid back to back at a stride of @p gridSize
///        bytes from @p base. One thread per grid.
__global__ inline void
stampRepeatedGridHeaders(std::byte *base, uint64_t gridSize, uint32_t gridCount) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < gridCount) {
        stampGridHeader(
            nanovdb::util::PtrAdd<nanovdb::GridData>(base, static_cast<int64_t>(i * gridSize)),
            i,
            gridCount);
    }
}

/// @brief Debug builds only: reads back every grid header in @p bytes, device-accessible storage
///        on @p device, on @p stream and throws if its (gridIndex, gridCount, gridSize) disagree
///        with @p meta. Adopting a layout skips the parse that used to catch a header the writer
///        got wrong; this keeps that net where it costs nothing in release. Defined in
///        GridAssembly.cu. Synchronizes @p stream.
void checkAdoptedLayoutDebug(const void *bytes,
                             const torch::Device &device,
                             const std::vector<nanovdb::GridHandleMetaData> &meta,
                             cudaStream_t stream);

/// @brief Wraps CUDA (DeviceGridBuffer) or PrivateUse1 (UnifiedGridBuffer) bytes whose layout is
///        already known in a GridHandle without parsing them: @p meta must list every grid in
///        order, packed end to end from offset 0 at NanoVDB's data alignment, which is the only
///        layout the rest of the stack (the chain parse, the typed accessors) can consume. That and
///        the buffer's extent are checked; the grid types are taken on trust, as they are when
///        copyTo adopts metadata. The headers themselves are checked against @p meta in debug
///        builds (checkAdoptedLayoutDebug), which waits for the writers on a DeviceGridBuffer's
///        retained stream. A UnifiedGridBuffer retains no stream, so its writers must be finished
///        before the call. Release builds adopt without a readback.
template <typename BufferT>
nanovdb::GridHandle<BufferT>
makeDeviceHandleFromLayout(BufferT &&buffer, std::vector<nanovdb::GridHandleMetaData> meta) {
    static_assert(
        std::is_same_v<BufferT, DeviceGridBuffer> || std::is_same_v<BufferT, UnifiedGridBuffer>,
        "makeDeviceHandleFromLayout takes a DeviceGridBuffer or UnifiedGridBuffer rvalue");
    const uint64_t extent = buffer.size_bytes();
    TORCH_CHECK(meta.empty() == (extent == 0),
                "makeDeviceHandleFromLayout: metadata and buffer disagree about emptiness");
    uint64_t end = 0;
    for (const auto &m: meta) {
        // Checked without forming offset + size, which can wrap.
        TORCH_CHECK(
            m.offset == end && m.offset % NANOVDB_DATA_ALIGNMENT == 0 &&
                m.size >= sizeof(nanovdb::GridData) && m.size <= extent - m.offset,
            "makeDeviceHandleFromLayout: grids must be packed, aligned and inside the buffer");
        end = m.offset + m.size;
    }
#ifndef NDEBUG
    if constexpr (std::is_same_v<BufferT, DeviceGridBuffer>) {
        checkAdoptedLayoutDebug(buffer.data(), buffer.resource().device(), meta, buffer.stream());
    } else {
        checkAdoptedLayoutDebug(buffer.data(), torch::Device(torch::kPrivateUse1), meta, {});
    }
#endif
    return nanovdb::GridHandle<BufferT>::adopt(std::move(buffer), std::move(meta));
}

/// @brief One grid's bytes, wherever they live, to be laid into new storage.
struct GridSpan {
    const void *bytes;
    uint64_t size;
    nanovdb::GridType type;
};

/// @brief The grids taken from one storage. They share that storage's device and retained
///        stream, which is where their writers are queued and where their block returns when the
///        storage is released; the assembler orders against it once per source.
struct GridSpanSource {
    torch::Device device;
    cudaStream_t stream;
    std::vector<GridSpan> spans;
};

/// @brief Logical grids [@p first, @p first + @p count) of @p input as spans of its storage, read
///        from the side the batch lives on, as one source on the batch's storage stream: the one
///        spelling of "a batch's grids as spans" for compaction, cloning and concatenation.
///        Correct for sliced and non-contiguous views, whose storage holds more grids than they
///        select.
inline GridSpanSource
logicalGridSpans(const GridBatchData &input, int64_t first, int64_t count) {
    const bool onHost = input.device().is_cpu();
    GridSpanSource source{input.device(), input.storageStream(), {}};
    source.spans.reserve(count);
    for (int64_t i = first; i < first + count; ++i) {
        const void *grid = onHost ? input.hostGridPtrAt(i) : input.deviceGridPtrAt(i);
        source.spans.push_back(GridSpan{grid, input.numBytesAt(i), nanovdb::GridType::OnIndex});
    }
    return source;
}

/// @brief Lays the spans of every source end to end into fresh storage on @p device, made on
///        @p stream, each header stamped with its new (index, count) and its checksum disabled,
///        and wraps the result: the one copy-and-fix loop behind compaction, concatenation and
///        the device merge, and the one place their stream and checksum policy lives.
///
///        Ordering, device destination: @p stream first waits on each distinct source stream
///        (the spans' writers), and each CUDA source stream waits on @p stream at the end, so a
///        source may be released as soon as this returns. The result adopts the layout it just
///        wrote (no parse kernel); a CUDA result retains @p stream, and a PrivateUse1 result is
///        unified storage. When a host or PrivateUse1 source, or a PrivateUse1 destination, is
///        involved, @p stream is synchronized before returning: those bytes are not stream-ordered,
///        or are read by host code straight away.
///        Host destination: each device source is copied on its own retained stream under its own
///        device and waited for; @p stream is not used. The handle is constructed from the bytes,
///        which validates the chain, and is complete on return. Copies between two CUDA devices
///        are peer copies.
GridStorage assembleGridStorage(const std::vector<GridSpanSource> &sources,
                                const torch::Device &device,
                                cudaStream_t stream);

/// @brief @p count back-to-back copies of the single grid in @p source, in fresh storage on
///        @p device (a CUDA or PrivateUse1 device, which @p source must live on), made on
///        @p stream. Copy i is stamped as grid i of @p count. Writes log2(count) copies and one
///        header launch, where assembleGridStorage writes one copy and one launch per grid.
///        Same ordering contract and handle kinds as assembleGridStorage for a device destination.
GridStorage replicateGridStorage(const GridStorage &source,
                                 uint32_t count,
                                 const torch::Device &device,
                                 cudaStream_t stream);

/// @brief Collects a builder's per-item results and lays them end to end in one storage on
///        @p device, made on @p stream. Items are added in batch order: a storage the builder
///        produced (add), the shared voxel-less grid for an item with nothing in it (addEmpty:
///        built once per batch, on first use, however many items are empty), or a repeat of an
///        earlier item (addRepeat: no copy until the merge). Every added storage must live on
///        @p device.
///
///        merge() returns the single item's storage as is when there is exactly one. When every
///        item is the same single-grid device storage, it replicates that grid
///        (replicateGridStorage), so a batch of identical grids is one build and log2(B) copies.
///        Any other mix is assembled one copy per item (assembleGridStorage). Either way the
///        collected parts may be dropped afterwards. Like all assembled storage the result's
///        checksums are disabled; the builders do not compute one either (NanoVDB's default), and
///        nothing in fvdb reads them.
class GridStorageParts {
  public:
    GridStorageParts(const torch::Device &device, cudaStream_t stream, size_t expectedCount)
        : mDevice(device), mStream(stream) {
        mOwned.reserve(expectedCount);
        mItems.reserve(expectedCount);
    }
    GridStorageParts(const GridStorageParts &)            = delete;
    GridStorageParts &operator=(const GridStorageParts &) = delete;

    /// @brief Adds @p storage as the next item and returns its index.
    size_t
    add(GridStorage &&storage) {
        TORCH_CHECK(storage.device() == mDevice,
                    "GridStorageParts: part on ",
                    storage.device(),
                    " cannot be merged onto ",
                    mDevice);
        TORCH_CHECK(mOwned.size() < mOwned.capacity(),
                    "GridStorageParts: more parts than the expected count");
        mOwned.push_back(std::move(storage));
        mItems.push_back(&mOwned.back());
        return mItems.size() - 1;
    }

    /// @brief Adds the shared voxel-less grid as the next item.
    void
    addEmpty() {
        if (!mEmpty) {
            mEmpty.emplace(createEmptyGridStorage(mDevice));
        }
        mItems.push_back(&*mEmpty);
    }

    /// @brief Adds item @p index's grids again as the next item.
    void
    addRepeat(size_t index) {
        TORCH_CHECK(index < mItems.size(), "GridStorageParts: no item ", index, " to repeat");
        mItems.push_back(mItems[index]);
    }

    size_t
    size() const {
        return mItems.size();
    }

    GridStorage
    merge() {
        if (mItems.size() == 1) {
            return std::move(*mItems[0]);
        }
        if (isUniformRepeat()) {
            return replicateGridStorage(
                *mItems[0], static_cast<uint32_t>(mItems.size()), mDevice, mStream);
        }
        std::vector<GridSpanSource> sources;
        sources.reserve(mItems.size());
        for (const GridStorage *part: mItems) {
            if (part->isEmpty()) {
                continue;
            }
            const auto *base = static_cast<const std::byte *>(part->isHost() ? part->hostBytes()
                                                                             : part->deviceBytes());
            GridSpanSource source{mDevice, part->isHost() ? cudaStream_t{} : part->stream(), {}};
            source.spans.reserve(part->gridCount());
            for (uint32_t i = 0; i < part->gridCount(); ++i) {
                source.spans.push_back(
                    GridSpan{base + part->gridOffset(i), part->gridSize(i), part->gridType(i)});
            }
            sources.push_back(std::move(source));
        }
        if (sources.empty()) {
            // Keep the device and stream a non-empty result would have retained.
            return GridStorage::empty(mDevice, mStream);
        }
        return assembleGridStorage(sources, mDevice, mStream);
    }

  private:
    // Two or more items, all the same single-grid storage, on a device destination.
    bool
    isUniformRepeat() const {
        if (mItems.size() < 2 || mItems.size() > std::numeric_limits<uint32_t>::max() ||
            mDevice.is_cpu()) {
            return false;
        }
        const GridStorage *first = mItems[0];
        if (first->isEmpty() || first->isHost() || first->gridCount() != 1) {
            return false;
        }
        return std::all_of(
            mItems.begin(), mItems.end(), [first](const GridStorage *p) { return p == first; });
    }

    torch::Device mDevice;
    cudaStream_t mStream;
    std::vector<GridStorage> mOwned;   // reserved up front: mItems points into it
    std::optional<GridStorage> mEmpty; // the shared voxel-less grid, built on first addEmpty
    std::vector<GridStorage *> mItems; // batch order; may repeat
};

/// @brief GridStorageParts over storages the caller already holds, in order: a single part is
///        returned as is; empty parts (no bytes) contribute nothing; a part holding a voxel-less
///        grid contributes that grid.
inline GridStorage
mergeGridStorages(std::vector<GridStorage> &&parts,
                  const torch::Device &device,
                  cudaStream_t stream) {
    GridStorageParts collected(device, stream, parts.size());
    for (GridStorage &part: parts) {
        collected.add(std::move(part));
    }
    return collected.merge();
}

} // namespace fvdb::detail

#endif // FVDB_DETAIL_UTILS_NANOVDB_DEVICEGRIDHANDLEUTILS_CUH
